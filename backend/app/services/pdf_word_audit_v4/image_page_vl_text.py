from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings

from .common import has_high_value_field_content, looks_like_document_title, looks_like_organization_name, looks_like_paragraphized_table_fragment, looks_like_table_title, normalize_text
from .models import ContentCoverageReview, ConversionPreflightResult, DocxEvidenceUnit, ImagePageVlTextReview
from .partial_artifacts import load_partial_review_payload, write_partial_review_payload
from .qwen_vl_gate import OllamaQwenVlClient


def image_page_vl_text_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["text_risk", "no_obvious_issue", "mapping_uncertain", "unreadable"],
            },
            "decision": {
                "type": "string",
                "enum": ["allow_exact_replacements", "defer", "block"],
            },
            "visible_text_excerpt": {"type": "string"},
            "suspicious_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unit_id": {"type": "string"},
                        "docx_text": {"type": "string"},
                        "visible_text": {"type": "string"},
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "text_substitution",
                                "digit_or_date_error",
                                "name_or_address_error",
                                "missing_or_extra_text",
                                "reading_order_error",
                                "unreadable",
                                "other",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["unit_id", "docx_text", "visible_text", "issue_type", "severity", "reason"],
                },
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
            "next_route": {"type": "string"},
        },
        "required": ["verdict", "decision", "visible_text_excerpt", "suspicious_values", "reason", "confidence", "next_route"],
    }


class ImagePageVlTextBuilder:
    """Use Qwen-VL for page-level text fidelity on high-risk image pages.

    This layer is intentionally conservative. It does not try to summarize the
    whole page into a finding; it only returns exact replacements for DOCX text
    units that can be located on the rendered PDF page.
    """

    REVIEW_STATUSES = {
        "uncovered_docx_content",
        "mapping_uncertain",
        "needs_pdf_ocr",
        "needs_full_page_ocr",
        "needs_region_segmentation",
        "low_confidence_page_review",
        "needs_text_alignment",
        "diff_candidate",
        "covered_by_page_ocr",
        "covered_by_nearby_page_ocr",
        "covered",
    }
    def __init__(
        self,
        *,
        work_dir: Path,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_samples: Optional[int] = None,
        client: Any = None,
        unload_after: Optional[bool] = None,
        skip_pages_with_backfill: Optional[bool] = None,
        progress_callback: Any = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        global_vl_enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_ENABLED", True))
        local_enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.enabled = bool(global_vl_enabled and local_enabled)
        self.model = str(model or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MODEL", "qwen3-vl:8b") or "qwen3-vl:8b").strip()
        self.timeout = max(
            1,
            int(
                timeout
                or getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_TIMEOUT", None)
                or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_TIMEOUT", 180)
                or 180
            ),
        )
        self.max_pages = max(0, int(max_pages if max_pages is not None else getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_PAGES", 9999) or 9999))
        self.max_samples = max(1, int(max_samples if max_samples is not None else getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_SAMPLES", 10) or 10))
        raw_max_total_seconds = getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_TOTAL_SECONDS", 0)
        self.max_total_seconds = max(0, int(0 if raw_max_total_seconds is None else raw_max_total_seconds))
        max_unproductive_value = getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_UNPRODUCTIVE", 0)
        self.max_unproductive = max(0, int(0 if max_unproductive_value is None else max_unproductive_value))
        self.skip_pages_with_backfill = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_SKIP_BACKFILLED_PAGES", True)
            if skip_pages_with_backfill is None
            else skip_pages_with_backfill
        )
        self.client = client or OllamaQwenVlClient(model=self.model)
        self.unload_after = bool(True if unload_after is None else unload_after)
        self.progress_callback = progress_callback

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        rendered_pages: Dict[int, Path],
    ) -> List[ImagePageVlTextReview]:
        if not self.enabled or self.max_pages <= 0:
            return []
        selected = self._selected_pages(preflight_result=preflight_result)
        if not selected:
            return []
        selected = selected[: self.max_pages]
        selected_page_nos = [int(page["page_no"]) for page in selected]
        rows = self._load_partial_reviews(selected_page_nos=selected_page_nos)
        completed_pages = {int(row.page_no or 0) for row in rows}
        selected = [page for page in selected if int(page["page_no"]) not in completed_pages]
        if not selected:
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        available, error = self._preflight()
        if not available:
            if self.unload_after:
                self._unload()
            base_index = len(rows)
            rows.extend(
                ImagePageVlTextReview(
                    review_id=f"image_text_vl_{base_index + index:04d}",
                    page_no=page["page_no"],
                    attempted=False,
                    available=False,
                    model=self.model,
                    verdict="unreadable",
                    decision="defer",
                    reason="Qwen3-VL 当前不可用，无法执行图片页全文视觉复核。",
                    docx_samples=list(page["samples"]),
                    next_route="needs_human_page_review",
                    error=error or "qwen_vl_unavailable",
                    flags=["image_text_vl_preflight_failed"],
                )
                for index, page in enumerate(selected, start=1)
            )
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        started_at = time.perf_counter()
        unproductive_count = 0
        for relative_index, page in enumerate(selected, start=1):
            row_index = len(rows) + 1
            self._emit_progress(
                page_no=int(page["page_no"]),
                index=relative_index,
                total=len(selected),
                event="image_text_vl_page_start",
                message=f"图片页 Qwen3-VL 全文复核：第 {relative_index}/{len(selected)} 个候选页，PDF 第 {int(page['page_no'])} 页。",
            )
            if self._budget_exhausted(started_at):
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        page=skip_page,
                        reason="图片页整页 Qwen3-VL 已达到总耗时预算，跳过剩余页，避免单一阶段拖住全流程。",
                        flag="image_text_vl_total_budget_exhausted",
                    )
                    for skip_offset, skip_page in enumerate(selected[relative_index - 1 :], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            request_timeout = self._remaining_budget_seconds(started_at)
            if request_timeout <= 0:
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        page=skip_page,
                        reason="图片页整页 Qwen3-VL 已达到总耗时预算，跳过剩余页，避免单一阶段拖住全流程。",
                        flag="image_text_vl_total_budget_exhausted",
                    )
                    for skip_offset, skip_page in enumerate(selected[relative_index - 1 :], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            if self.max_unproductive and unproductive_count >= self.max_unproductive:
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        page=skip_page,
                        reason="前序图片页整页 Qwen3-VL 未输出有效结构化结果，已触发连续无效熔断。",
                        flag="image_text_vl_unproductive_guard",
                    )
                    for skip_offset, skip_page in enumerate(selected[relative_index - 1 :], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            image_path = rendered_pages.get(page["page_no"])
            if image_path is None or not Path(image_path).exists():
                rows.append(
                    ImagePageVlTextReview(
                        review_id=f"image_text_vl_{row_index:04d}",
                        page_no=page["page_no"],
                        attempted=False,
                        available=False,
                        model=self.model,
                        verdict="unreadable",
                        decision="defer",
                        reason="未找到 PDF 页面渲染图，无法执行图片页全文视觉复核。",
                        docx_samples=list(page["samples"]),
                        next_route="needs_human_page_review",
                        error="missing_page_image",
                        flags=["image_text_vl_missing_page_image"],
                    )
                )
                self._write_partial_reviews(rows=rows)
                continue
            review = self._review(index=row_index, page=page, image_path=Path(image_path), timeout=request_timeout)
            rows.append(review)
            self._emit_progress(
                page_no=int(page["page_no"]),
                index=relative_index,
                total=len(selected),
                event="image_text_vl_page_done",
                message=(
                    f"图片页 Qwen3-VL 全文复核完成：第 {relative_index}/{len(selected)} 个候选页，"
                    f"PDF 第 {int(page['page_no'])} 页，结果 {review.verdict}/{review.decision}。"
                ),
            )
            if self._unproductive_review(review):
                unproductive_count += 1
            else:
                unproductive_count = 0
            self._write_partial_reviews(rows=rows)
        if self.unload_after:
            self._unload()
        self._write_partial_reviews(rows=rows, status="done")
        return rows

    def _budget_exhausted(self, started_at: float) -> bool:
        return bool(self.max_total_seconds and (time.perf_counter() - started_at) >= self.max_total_seconds)

    def _remaining_budget_seconds(self, started_at: float) -> int:
        if not self.max_total_seconds:
            return self.timeout
        elapsed = time.perf_counter() - started_at
        remaining = int(self.max_total_seconds - elapsed)
        return max(0, min(self.timeout, remaining))

    def _emit_progress(self, *, page_no: int, index: int, total: int, event: str, message: str) -> None:
        callback = self.progress_callback
        if not callable(callback):
            return
        try:
            callback(
                {
                    "event": event,
                    "builder": "image_text_vl",
                    "page_no": int(page_no),
                    "item_current": int(index),
                    "item_total": int(total),
                    "message": message,
                }
            )
        except Exception:
            return

    def _write_partial_reviews(self, *, rows: Sequence[ImagePageVlTextReview], status: str = "running") -> None:
        write_partial_review_payload(
            work_dir=self.work_dir,
            filename="image_text_vl_reviews.partial.json",
            version="image_text_vl_v1",
            reviews=rows,
            extra={"status": status, "model": self.model, "resume_key": self._resume_key()},
        )

    def _load_partial_reviews(self, *, selected_page_nos: Sequence[int]) -> List[ImagePageVlTextReview]:
        loaded, _payload = load_partial_review_payload(
            work_dir=self.work_dir,
            filename="image_text_vl_reviews.partial.json",
            review_type=ImagePageVlTextReview,
            expected_resume_key=self._resume_key(),
        )
        if not loaded:
            return []
        order = {int(page_no): index for index, page_no in enumerate(selected_page_nos)}
        by_page: Dict[int, ImagePageVlTextReview] = {}
        for row in loaded:
            page_no = int(row.page_no or 0)
            if page_no in order:
                by_page[page_no] = row
        return [by_page[page_no] for page_no in selected_page_nos if page_no in by_page]

    def _resume_key(self) -> str:
        return (
            "image_text_vl_v2:"
            f"model={self.model}:"
            f"max_samples={self.max_samples}:"
            f"skip_backfill={int(self.skip_pages_with_backfill)}:"
            "strict_exact_replacements_v1"
        )

    def _skipped_review(self, *, index: int, page: Dict[str, Any], reason: str, flag: str) -> ImagePageVlTextReview:
        return ImagePageVlTextReview(
            review_id=f"image_text_vl_{index:04d}",
            page_no=int(page["page_no"]),
            attempted=False,
            available=False,
            model=self.model,
            verdict="unreadable",
            decision="defer",
            confidence=0.0,
            reason=reason,
            docx_samples=list(page["samples"]),
            next_route="needs_focused_region_review",
            error=flag,
            flags=["image_text_vl_skipped", flag],
        )

    def _unproductive_review(self, review: ImagePageVlTextReview) -> bool:
        if review.suspicious_values:
            return False
        flags = set(review.flags or [])
        if "image_text_vl_schema_incomplete" in flags:
            return True
        return bool(review.verdict in {"unreadable", "mapping_uncertain"} and float(review.confidence or 0.0) <= 0.1)

    def _selected_pages(self, *, preflight_result: ConversionPreflightResult) -> List[Dict[str, Any]]:
        docx_by_id = {unit.unit_id: unit for unit in preflight_result.docx_units}
        backfilled_pages = self._strong_backfilled_pages(preflight_result=preflight_result) if self.skip_pages_with_backfill else set()
        text_profile_by_page = {
            int(profile.page_no or 0): profile
            for profile in preflight_result.page_text_coverage_profiles
            if int(profile.page_no or 0) > 0
        }
        coverage_by_page: Dict[int, List[ContentCoverageReview]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.side != "docx" or review.status not in self.REVIEW_STATUSES:
                continue
            unit = docx_by_id.get(review.unit_id)
            if unit is None or unit.container_type == "table_cell":
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            coverage_by_page.setdefault(page_no, []).append(review)
        anchor_ocr_by_page = self._anchor_ocr_by_page(preflight_result.anchor_ocr)
        pages: List[Dict[str, Any]] = []
        for image_review in preflight_result.image_page_reviews:
            page_no = int(image_review.page_no or 0)
            if page_no <= 0:
                continue
            profile = dict(preflight_result.page_profiles.get(str(page_no)) or {})
            text_profile = text_profile_by_page.get(page_no)
            text_profile_status = str(getattr(text_profile, "status", "") or "")
            text_profile_risk = str(getattr(text_profile, "risk_level", "") or "")
            text_profile_high_risk = bool(
                text_profile_risk == "high"
                or text_profile_status in {"page_text_coverage_gap", "missing_side_text"}
            )
            if image_review.risk_level not in {"high", "medium"} and not text_profile_high_risk:
                continue
            if (
                page_no in backfilled_pages
                and text_profile_risk not in {"high", "medium"}
                and text_profile_status not in {"page_text_coverage_gap", "missing_side_text"}
            ):
                continue
            labels = set(profile.get("labels") or [])
            if (
                image_review.table_like
                or image_review.page_kind == "table_image_pdf"
                or bool(profile.get("table_like"))
                or bool(profile.get("needs_table_parser"))
                or "table_heavy" in labels
                or str(profile.get("primary_route") or "") in {"image_table_cell_compare", "native_table_compare"}
            ):
                continue
            samples = self._samples(
                rows=coverage_by_page.get(page_no, []),
                docx_by_id=docx_by_id,
            )
            if not samples:
                continue
            score = (
                (300 if image_review.risk_level == "high" else 100)
                + int(image_review.unresolved_count or 0)
                + sum(15 for item in samples if self._high_signal_text(item.get("text", "")))
                + (40 if image_review.fragment_anomaly_count else 0)
            )
            pages.append(
                {
                    "page_no": page_no,
                    "score": score,
                    "risk_level": image_review.risk_level,
                    "page_kind": image_review.page_kind,
                    "unresolved_count": int(image_review.unresolved_count or 0),
                    "samples": samples,
                    "pdf_ocr_excerpt": str((anchor_ocr_by_page.get(page_no) or {}).get("text") or "")[:900],
                    "image_review_id": image_review.review_id,
                }
            )
        pages.sort(key=lambda item: (-int(item["score"]), int(item["page_no"])))
        return pages

    def _strong_backfilled_pages(self, *, preflight_result: ConversionPreflightResult) -> set[int]:
        pages: set[int] = set()
        for backfill in preflight_result.content_coverage_backfills:
            if not backfill.available:
                continue
            page_no = int(backfill.page_no or 0)
            if page_no <= 0:
                continue
            flags = set(str(item) for item in backfill.flags)
            is_page_visual = bool("unit_type=visual_region" in flags and "region_type=visual_content" in flags)
            if not is_page_visual and backfill.side != "pdf":
                continue
            normalized = normalize_text(backfill.extracted_text or backfill.normalized_text or "")
            if len(normalized) < 120:
                continue
            if float(backfill.confidence or 0.0) < 0.68 and str(backfill.quality or "").lower() != "high":
                continue
            pages.add(page_no)
        return pages

    def _samples(
        self,
        *,
        rows: Sequence[ContentCoverageReview],
        docx_by_id: Dict[str, DocxEvidenceUnit],
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            rows,
            key=lambda review: (
                0 if review.decision != "covered" else 1,
                0 if self._high_signal_text(review.text) else 1,
                int(review.page_no or 9999),
                review.review_id,
            ),
        )
        samples: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for review in ordered:
            unit = docx_by_id.get(review.unit_id)
            if unit is None:
                continue
            text = " ".join(str(review.text or unit.text or "").split())[:150]
            if len(normalize_text(text)) < 4:
                continue
            if self._looks_like_table_fragment_text(text):
                continue
            compact = normalize_text(text)
            if compact in seen:
                continue
            seen.add(compact)
            samples.append(
                {
                    "unit_id": review.unit_id,
                    "text": text,
                    "coverage_review_id": review.review_id,
                    "status": review.status,
                    "decision": review.decision,
                    "order_index": int(unit.order_index or 0),
                    "container_type": unit.container_type,
                }
            )
            if len(samples) >= self.max_samples:
                break
        return samples

    def _review(self, *, index: int, page: Dict[str, Any], image_path: Path, timeout: int) -> ImagePageVlTextReview:
        result = self.client.structured_chat_with_image(
            image_path=image_path,
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(page=page, image_path=image_path),
            schema=image_page_vl_text_schema(),
            timeout=max(1, min(self.timeout, int(timeout or self.timeout))),
            num_predict=640,
            temperature=0.0,
            allow_generate_fallback=True,
        )
        relative = self._relative_path(image_path)
        if not getattr(result, "ok", False):
            return ImagePageVlTextReview(
                review_id=f"image_text_vl_{index:04d}",
                page_no=int(page["page_no"]),
                attempted=True,
                available=False,
                model=self.model,
                verdict="unreadable",
                decision="defer",
                reason="Qwen3-VL 图片页全文视觉复核请求失败。",
                page_image_path=relative,
                docx_samples=list(page["samples"]),
                next_route="needs_human_page_review",
                error=str(getattr(result, "error", "") or "qwen_vl_failed"),
                model_parse_error=str(getattr(result, "error", "") or ""),
                model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), 600),
                flags=["image_text_vl_failed"],
            )
        parsed = self._normalize_payload(dict(getattr(result, "parsed", {}) or {}))
        verdict_choices = {"text_risk", "no_obvious_issue", "mapping_uncertain", "unreadable"}
        decision_choices = {"allow_exact_replacements", "defer", "block"}
        raw_verdict = str(parsed.get("verdict") or "").strip()
        raw_decision = str(parsed.get("decision") or "").strip()
        valid_verdict = raw_verdict in verdict_choices
        valid_decision = raw_decision in decision_choices
        verdict = self._choice(parsed.get("verdict"), verdict_choices, "unreadable")
        decision = self._choice(parsed.get("decision"), decision_choices, "defer")
        suspicious_values = self._suspicious_values(parsed.get("suspicious_values"), samples=page["samples"])
        if suspicious_values and verdict in {"no_obvious_issue", "mapping_uncertain", "unreadable"}:
            verdict = "text_risk"
        schema_repaired = bool(suspicious_values and not valid_verdict)
        no_issue_repaired = bool(
            not suspicious_values
            and not valid_verdict
            and self._top_level_visible_matches_sample(parsed=parsed, samples=page["samples"])
        )
        if no_issue_repaired:
            verdict = "no_obvious_issue"
            decision = "defer"
        if schema_repaired and decision == "defer" and not valid_decision:
            decision = "allow_exact_replacements"
        if not suspicious_values and decision == "allow_exact_replacements":
            decision = "defer"
        flags = ["image_text_vl"]
        schema_incomplete = not valid_verdict and not schema_repaired and not no_issue_repaired
        if suspicious_values:
            flags.append("image_text_vl_suspicious_values")
        if schema_repaired or no_issue_repaired:
            flags.append("image_text_vl_payload_normalized")
        if schema_incomplete:
            flags.append("image_text_vl_schema_incomplete")
        confidence_floor = 0.72 if suspicious_values else (0.6 if no_issue_repaired else 0.0)
        confidence = max(self._confidence(parsed.get("confidence")), confidence_floor)
        unusable_uncertain = bool(
            not suspicious_values
            and verdict in {"mapping_uncertain", "unreadable"}
            and confidence <= 0.1
        )
        if unusable_uncertain:
            flags.append("image_text_vl_unusable_response")
        available = not bool((schema_incomplete or unusable_uncertain) and not suspicious_values)
        return ImagePageVlTextReview(
            review_id=f"image_text_vl_{index:04d}",
            page_no=int(page["page_no"]),
            attempted=True,
            available=available,
            model=self.model,
            verdict=verdict,
            decision=decision,
            confidence=confidence,
            reason=self._clip(parsed.get("reason"), 180),
            page_image_path=relative,
            visible_text_excerpt=self._clip(parsed.get("visible_text_excerpt"), 520),
            suspicious_values=suspicious_values,
            docx_samples=list(page["samples"]),
            next_route=str(parsed.get("next_route") or ("needs_human_page_review" if verdict != "no_obvious_issue" else "")).strip(),
            error=(
                "image_text_vl_schema_incomplete"
                if schema_incomplete and not suspicious_values
                else ("image_text_vl_unusable_response" if unusable_uncertain else "")
            ),
            model_parse_error=(
                "image_text_vl_schema_incomplete"
                if schema_incomplete and not suspicious_values
                else ("image_text_vl_unusable_response" if unusable_uncertain else "")
            ),
            model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), 600),
            model_parsed_keys=sorted(str(key) for key in parsed.keys()),
            flags=flags,
        )

    def _normalize_payload(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if not parsed:
            return {}
        payload = dict(parsed)
        values = payload.get("suspicious_values")
        normalized_values: List[Dict[str, Any]] = []
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    normalized_values.append(dict(value))
        if payload.get("unit_id") and payload.get("visible_text"):
            normalized_values.append(
                {
                    "unit_id": payload.get("unit_id"),
                    "docx_text": payload.get("docx_text") or "",
                    "visible_text": payload.get("visible_text"),
                    "issue_type": payload.get("issue_type") or "text_substitution",
                    "severity": payload.get("severity") or "medium",
                    "reason": payload.get("reason") or "",
                }
            )
        if normalized_values:
            payload["suspicious_values"] = normalized_values
        return payload

    def _suspicious_values(self, values: Any, *, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sample_by_id = {str(item.get("unit_id") or ""): item for item in samples}
        rows: List[Dict[str, Any]] = []
        seen_units: set[str] = set()
        for value in values if isinstance(values, list) else []:
            if not isinstance(value, dict):
                continue
            unit_id = str(value.get("unit_id") or "").strip()
            sample = sample_by_id.get(unit_id)
            if sample is None or unit_id in seen_units:
                continue
            docx_text = self._clean_text(value.get("docx_text") or sample.get("text") or "")
            visible_text = self._clean_text(value.get("visible_text") or "")
            if not self._valid_exact_replacement(old_text=docx_text, new_text=visible_text):
                continue
            seen_units.add(unit_id)
            rows.append(
                {
                    "unit_id": unit_id,
                    "docx_text": docx_text,
                    "visible_text": visible_text,
                    "issue_type": self._choice(
                        value.get("issue_type"),
                        {
                            "text_substitution",
                            "digit_or_date_error",
                            "name_or_address_error",
                            "missing_or_extra_text",
                            "reading_order_error",
                            "unreadable",
                            "other",
                        },
                        "text_substitution",
                    ),
                    "severity": self._choice(value.get("severity"), {"high", "medium", "low"}, "medium"),
                    "reason": self._clip(value.get("reason"), 160),
                }
            )
            if len(rows) >= 12:
                break
        return rows

    def _top_level_visible_matches_sample(self, *, parsed: Dict[str, Any], samples: Sequence[Dict[str, Any]]) -> bool:
        unit_id = str(parsed.get("unit_id") or "").strip()
        visible_text = self._clean_text(parsed.get("visible_text") or "")
        if not unit_id or not visible_text:
            return False
        sample_by_id = {str(item.get("unit_id") or ""): item for item in samples}
        sample_text = self._clean_text((sample_by_id.get(unit_id) or {}).get("text") or "")
        if not sample_text:
            return False
        return normalize_text(sample_text) == normalize_text(visible_text)

    def _valid_exact_replacement(self, *, old_text: str, new_text: str) -> bool:
        old_value = self._clean_text(old_text)
        new_value = self._clean_text(new_text)
        if not old_value or not new_value:
            return False
        if old_value == new_value:
            return False
        if normalize_text(old_value) == normalize_text(new_value):
            return False
        if len(new_value) > 180:
            return False
        if any(marker in new_value for marker in ("?", "？", "_", "看不清", "无法", "不确定", "参考", "仍需", "核对", "疑似")):
            return False
        if self._overexpanded_sample_completion(old_text=old_value, new_text=new_value):
            return False
        if self._unrelated_short_replacement(old_text=old_value, new_text=new_value):
            return False
        if self._date_like(old_value) and self._date_like(new_value) and not self._date_replacement_contextual(old_value, new_value):
            return False
        compact = normalize_text(new_value)
        if len(compact) >= 45:
            digit_ratio = sum(ch.isdigit() for ch in compact) / max(1, len(compact))
            ascii_ratio = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in compact) / max(1, len(compact))
            if digit_ratio > 0.45 or ascii_ratio > 0.22:
                old_compact = normalize_text(old_value)
                similarity = SequenceMatcher(None, old_compact, compact).ratio()
                shared_long_number = bool(
                    set(re.findall(r"\d{6,}", old_compact))
                    & set(re.findall(r"\d{6,}", compact))
                )
                old_cjk = {ch for ch in old_compact if "\u4e00" <= ch <= "\u9fff"}
                new_cjk = {ch for ch in compact if "\u4e00" <= ch <= "\u9fff"}
                if not (similarity >= 0.78 and (shared_long_number or len(old_cjk & new_cjk) >= 8)):
                    return False
        return True

    def _overexpanded_sample_completion(self, *, old_text: str, new_text: str) -> bool:
        old_key = normalize_text(old_text)
        new_key = normalize_text(new_text)
        if not old_key or not new_key:
            return False
        if len(new_key) <= len(old_key) + 24:
            return False
        if old_key in new_key or new_key.startswith(old_key[: max(20, min(len(old_key), 60))]):
            return True
        similarity = SequenceMatcher(None, old_key[:80], new_key[:80]).ratio()
        return similarity >= 0.92 and len(new_key) > len(old_key) + 36

    def _unrelated_short_replacement(self, *, old_text: str, new_text: str) -> bool:
        old_key = normalize_text(old_text)
        new_key = normalize_text(new_text)
        if not old_key or not new_key:
            return True
        if max(len(old_key), len(new_key)) > 18:
            return False
        if self._date_like(old_key) and self._date_like(new_key):
            return False
        if self._number_like(old_key) and self._number_like(new_key):
            return False
        old_cjk = {ch for ch in old_key if "\u4e00" <= ch <= "\u9fff"}
        new_cjk = {ch for ch in new_key if "\u4e00" <= ch <= "\u9fff"}
        shared_cjk = old_cjk & new_cjk
        similarity = SequenceMatcher(None, old_key, new_key).ratio()
        if similarity < 0.35 and len(shared_cjk) < 2:
            return True
        if min(len(old_key), len(new_key)) <= 4 and similarity < 0.42 and not shared_cjk:
            return True
        return False

    def _date_like(self, value: str) -> bool:
        return bool(re.search(r"\d{4}年\d{1,2}月|[一二三四五六七八九十〇○零]{2,4}年|年.*月.*日", value))

    def _date_replacement_contextual(self, old_value: str, new_value: str) -> bool:
        old_key = normalize_text(old_value)
        new_key = normalize_text(new_value)
        if not old_key or not new_key:
            return False
        old_numbers = re.findall(r"\d+", old_key)
        new_numbers = re.findall(r"\d+", new_key)
        if not old_numbers or not new_numbers:
            return SequenceMatcher(None, old_key, new_key).ratio() >= 0.62
        if set(old_numbers) & set(new_numbers):
            return True
        if len(old_numbers) >= 2 and len(new_numbers) >= 2 and old_numbers[-2:] == new_numbers[-2:]:
            return True
        old_year = old_numbers[0]
        new_year = new_numbers[0]
        same_month_day = len(old_numbers) >= 3 and len(new_numbers) >= 3 and old_numbers[1:] == new_numbers[1:]
        if same_month_day and abs(len(old_year) - len(new_year)) <= 1:
            return True
        return SequenceMatcher(None, old_key, new_key).ratio() >= 0.72

    def _number_like(self, value: str) -> bool:
        digit_count = sum(ch.isdigit() for ch in value)
        return digit_count >= 2

    def _preflight(self) -> tuple[bool, str]:
        preflight = getattr(self.client, "preflight", None)
        if not callable(preflight):
            return True, ""
        result = preflight(timeout=min(30, self.timeout))
        if bool(result.get("available")):
            return True, ""
        return False, str(result.get("error") or result.get("reason") or "qwen_vl_preflight_failed")

    def _unload(self) -> None:
        unload = getattr(self.client, "unload", None)
        if callable(unload):
            unload()

    def _system_prompt(self) -> str:
        return (
            "你是 WPS PDF 转 DOCX 忠实度审查的图片页全文视觉复核模型。"
            "PDF 页面图片是唯一可见内容基准。你要逐项核对给定 DOCX 文本单元是否忠实于 PDF 图片。"
            "只输出能在图片中明确定位、且能给出精确替换文本的错误；不确定、看不清、页映射不稳时不要输出为错误。"
            "不要复述输入，不要做法律分析，不要补写图片外内容。必须只输出一个 JSON 对象。"
        )

    def _user_prompt(self, *, page: Dict[str, Any], image_path: Path) -> str:
        payload = {
            "page_no": page["page_no"],
            "page_kind": page.get("page_kind"),
            "risk_level": page.get("risk_level"),
            "unresolved_count": page.get("unresolved_count"),
            "pdf_ocr_excerpt_for_reference_only": page.get("pdf_ocr_excerpt") or "",
            "docx_samples": page["samples"],
            "rules": [
                "先看 PDF 图片，不要只依赖 OCR 摘要；OCR 摘要只能辅助定位。",
                "逐项核对 docx_samples，每条 suspicious_values 必须带原 unit_id。",
                "visible_text 填 PDF 图片中对应位置应有的完整文本或可替换片段。",
                "只有明确可见且必须修改的内容才输出；看不清、无法定位、只是排版差异都不要输出。",
                "日期、身份证号、地址、人名、金额、标题、签署日期、零散页脚都要关注。",
                "如果整页方向、页码或对应关系不可靠，verdict=mapping_uncertain 且 suspicious_values 为空。",
                "不要输出思考过程，不要 Markdown，不要解释 JSON 之外的文字。",
            ],
        }
        return "对照图片审查这些 DOCX 文本单元，只输出固定 JSON。输入：" + json.dumps(payload, ensure_ascii=False)[:3600]

    def _anchor_ocr_by_page(self, anchor_ocr: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for page in anchor_ocr.get("pages") or []:
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            if page_no > 0:
                rows[page_no] = dict(page)
        return rows

    def _high_signal_text(self, text: Any) -> bool:
        value = str(text or "")
        compact = normalize_text(value)
        return bool(
            has_high_value_field_content(value)
            or looks_like_document_title(compact)
            or looks_like_organization_name(compact)
            or looks_like_table_title(compact)
        )

    def _looks_like_table_fragment_text(self, text: Any) -> bool:
        return looks_like_paragraphized_table_fragment(text)

    def _choice(self, value: Any, allowed: set[str], default: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else default

    def _confidence(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))

    def _clean_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _clip(self, value: Any, limit: int) -> str:
        text = self._clean_text(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    def _relative_path(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.work_dir.resolve()))
        except Exception:
            return str(path)
