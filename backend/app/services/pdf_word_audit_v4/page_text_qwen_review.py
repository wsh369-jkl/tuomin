from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings

from .common import looks_like_paragraphized_table_fragment, normalize_text
from .models import ContentCoverageReview, ConversionPreflightResult, DocxEvidenceUnit, PageTextQwenReview
from .partial_artifacts import load_partial_review_payload, write_partial_review_payload
from .qwen_gate import OllamaQwenGateClient, _audit_review_model


def page_text_qwen_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["text_risk", "no_obvious_issue", "mapping_uncertain", "ocr_insufficient"],
            },
            "decision": {
                "type": "string",
                "enum": ["allow_exact_replacements", "defer", "block"],
            },
            "suspicious_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unit_id": {"type": "string"},
                        "docx_text": {"type": "string"},
                        "pdf_text": {"type": "string"},
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "text_substitution",
                                "digit_or_date_error",
                                "name_or_address_error",
                                "missing_or_extra_text",
                                "reading_order_error",
                                "other",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["unit_id", "docx_text", "pdf_text", "issue_type", "severity", "reason"],
                },
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
            "next_route": {"type": "string"},
        },
        "required": ["verdict", "decision", "suspicious_values", "reason", "confidence", "next_route"],
    }


class PageTextQwenReviewBuilder:
    """Compare page-level PDF OCR text with DOCX units using text Qwen.

    This is a recall layer for image/scan PDFs. It does not look at the rendered
    page. It only accepts exact replacement suggestions that can be found in the
    page OCR text, so the model cannot invent a correction from context alone.
    """

    REVIEW_STATUSES = {
        "covered",
        "covered_by_page_ocr",
        "covered_by_nearby_page_ocr",
        "covered_by_backfill_ocr",
        "covered_by_backfill_ocr_fuzzy",
        "uncovered_docx_content",
        "mapping_uncertain",
        "needs_pdf_ocr",
        "needs_full_page_ocr",
        "needs_region_segmentation",
        "low_confidence_page_review",
        "needs_text_alignment",
        "diff_candidate",
    }
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_samples: Optional[int] = None,
        max_total_seconds: Optional[int] = None,
        work_dir: Optional[Path] = None,
        client: Any = None,
        progress_callback: Any = None,
    ) -> None:
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.model = str(model or _audit_review_model() or "qwen3.5:9b").strip()
        self.timeout = max(
            1,
            int(timeout if timeout is not None else getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_TIMEOUT", 120) or 120),
        )
        self.max_pages = max(
            0,
            int(max_pages if max_pages is not None else getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_PAGES", 9999) or 9999),
        )
        self.max_samples = max(
            1,
            int(max_samples if max_samples is not None else getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_SAMPLES", 16) or 16),
        )
        raw_max_total_seconds = (
            max_total_seconds
            if max_total_seconds is not None
            else getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_TOTAL_SECONDS", 0)
        )
        self.max_total_seconds = max(0, int(0 if raw_max_total_seconds is None else raw_max_total_seconds))
        self.max_pdf_chars = max(
            1200,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_PDF_CHARS", 3200) or 3200),
        )
        self.min_pdf_chars = max(
            8,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MIN_PDF_CHARS", 16) or 16),
        )
        self.num_predict = max(
            360,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_NUM_PREDICT", 900) or 900),
        )
        self.client = client or OllamaQwenGateClient(model=self.model)
        self.progress_callback = progress_callback

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[PageTextQwenReview]:
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
        available, preflight_error = self._preflight()
        if not available:
            self._unload()
            base_index = len(rows)
            rows.extend(
                PageTextQwenReview(
                    review_id=f"page_text_qwen_{base_index + index:04d}",
                    page_no=int(page["page_no"]),
                    attempted=False,
                    available=False,
                    model=self.model,
                    status="qwen_unavailable",
                    verdict="ocr_insufficient",
                    decision="defer",
                    reason="Qwen 文本模型不可用，无法执行页级 OCR-DOCX 全内容复核。",
                    pdf_text_excerpt=str(page.get("pdf_text") or "")[:500],
                    docx_samples=list(page.get("samples") or []),
                    next_route="needs_full_page_review",
                    error=preflight_error or "qwen_text_unavailable",
                    flags=["page_text_qwen_preflight_failed"],
                )
                for index, page in enumerate(selected, start=1)
            )
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        started_at = time.perf_counter()
        for relative_index, page in enumerate(selected, start=1):
            row_index = len(rows) + 1
            self._emit_progress(
                page_no=int(page["page_no"]),
                index=relative_index,
                total=len(selected),
                event="page_text_qwen_page_start",
                message=f"页级 OCR 文本 Qwen 复核：第 {relative_index}/{len(selected)} 页，PDF 第 {int(page['page_no'])} 页。",
            )
            if self._budget_exhausted(started_at):
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        page=skip_page,
                        reason="页级文本 Qwen 复核达到总耗时预算，跳过剩余页。",
                        flag="page_text_qwen_total_budget_exhausted",
                    )
                    for skip_offset, skip_page in enumerate(selected[relative_index - 1 :], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            timeout = self._remaining_budget_seconds(started_at)
            if timeout <= 0:
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        page=skip_page,
                        reason="页级文本 Qwen 复核达到总耗时预算，跳过剩余页。",
                        flag="page_text_qwen_total_budget_exhausted",
                    )
                    for skip_offset, skip_page in enumerate(selected[relative_index - 1 :], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            review = self._review(index=row_index, page=page, timeout=timeout)
            rows.append(review)
            self._emit_progress(
                page_no=int(page["page_no"]),
                index=relative_index,
                total=len(selected),
                event="page_text_qwen_page_done",
                message=(
                    f"页级 OCR 文本 Qwen 复核完成：第 {relative_index}/{len(selected)} 页，"
                    f"PDF 第 {int(page['page_no'])} 页，结果 {review.verdict}/{review.decision}。"
                ),
            )
            self._write_partial_reviews(rows=rows)
        self._unload()
        self._write_partial_reviews(rows=rows, status="done")
        return rows

    def _write_partial_reviews(self, *, rows: Sequence[PageTextQwenReview], status: str = "running") -> None:
        if self.work_dir is None:
            return
        write_partial_review_payload(
            work_dir=self.work_dir,
            filename="page_text_qwen_reviews.partial.json",
            version="page_text_qwen_review_v1",
            reviews=rows,
            extra={"status": status, "model": self.model, "resume_key": self._resume_key()},
        )

    def _load_partial_reviews(self, *, selected_page_nos: Sequence[int]) -> List[PageTextQwenReview]:
        if self.work_dir is None:
            return []
        loaded, _payload = load_partial_review_payload(
            work_dir=self.work_dir,
            filename="page_text_qwen_reviews.partial.json",
            review_type=PageTextQwenReview,
            expected_resume_key=self._resume_key(),
        )
        if not loaded:
            return []
        order = {int(page_no): index for index, page_no in enumerate(selected_page_nos)}
        by_page: Dict[int, PageTextQwenReview] = {}
        for row in loaded:
            page_no = int(row.page_no or 0)
            if page_no in order:
                by_page[page_no] = row
        return [by_page[page_no] for page_no in selected_page_nos if page_no in by_page]

    def _resume_key(self) -> str:
        return (
            "page_text_qwen_review_v2:"
            f"model={self.model}:"
            f"max_samples={self.max_samples}:"
            f"max_pdf_chars={self.max_pdf_chars}:"
            "exact_replacement_guard_v1"
        )

    def _selected_pages(self, *, preflight_result: ConversionPreflightResult) -> List[Dict[str, Any]]:
        docx_by_page = self._docx_by_page(preflight_result.docx_units)
        coverage_by_unit = {
            review.unit_id: review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx" and review.status in self.REVIEW_STATUSES
        }
        pdf_texts = self._pdf_texts_by_page(preflight_result=preflight_result)
        image_risk_pages = {
            int(review.page_no or 0): review.risk_level
            for review in preflight_result.image_page_reviews
            if int(review.page_no or 0) > 0 and review.risk_level in {"high", "medium"}
        }
        profile_by_page = {
            int(profile.page_no or 0): profile
            for profile in preflight_result.page_text_coverage_profiles
            if int(profile.page_no or 0) > 0
        }
        table_evidence_by_page: Dict[int, int] = {}
        for review in preflight_result.table_cell_evidence_reviews:
            page = int(review.page_no or 0)
            if page > 0 and review.decision == "confirmed_error":
                table_evidence_by_page[page] = table_evidence_by_page.get(page, 0) + 1
        pages: List[Dict[str, Any]] = []
        for page_no, units in docx_by_page.items():
            profile = dict(preflight_result.page_profiles.get(str(page_no)) or {})
            is_table_page = self._is_table_page(profile=profile, units=units)
            pdf_text = pdf_texts.get(page_no, "")
            if len(self._semantic_compact(pdf_text)) < self.min_pdf_chars:
                continue
            if is_table_page:
                continue
            text_profile = profile_by_page.get(page_no)
            risk = image_risk_pages.get(page_no, "")
            profile_status = str(getattr(text_profile, "status", "") or "")
            profile_risk = str(getattr(text_profile, "risk_level", "") or "")
            if profile_status == "page_text_coverage_supported" and profile_risk == "low":
                continue
            if not (
                risk in {"high", "medium"}
                or profile_risk in {"high", "medium"}
                or profile_status in {"page_text_coverage_gap", "page_text_coverage_uncertain"}
                or profile.get("needs_ocr")
            ):
                continue
            samples = self._samples(
                units=units,
                coverage_by_unit=coverage_by_unit,
                include_table_cells=False,
            )
            if not samples:
                continue
            unresolved_sample_count = sum(1 for item in samples if str(item.get("status") or "") != "covered")
            risk_bonus = 0
            if risk == "high" or profile_risk == "high":
                risk_bonus = 800
            elif risk == "medium" or profile_risk == "medium":
                risk_bonus = 250
            score = (
                risk_bonus
                + unresolved_sample_count * 14
                + min(180, int(getattr(text_profile, "docx_token_count", 0) or 0))
            )
            pages.append(
                {
                    "page_no": page_no,
                    "score": score,
                    "risk": risk or profile_risk,
                    "profile_status": profile_status,
                    "profile_risk": profile_risk,
                    "page_kind": "text_page",
                    "table_evidence_count": int(table_evidence_by_page.get(page_no, 0) or 0),
                    "unresolved_sample_count": unresolved_sample_count,
                    "pdf_text": pdf_text[: self.max_pdf_chars],
                    "samples": samples,
                }
            )
        pages.sort(key=lambda item: (-int(item["score"]), int(item["page_no"])))
        return pages

    def _review(self, *, index: int, page: Dict[str, Any], timeout: int) -> PageTextQwenReview:
        result = self._structured_chat(page=page, timeout=timeout, num_predict=self.num_predict)
        review_page = page
        retry_attempted = False
        retry_error = ""
        if self._should_minimal_retry(result=result):
            retry_result, retry_page = self._minimal_retry(page=page, timeout=timeout)
            retry_attempted = True
            retry_error = str(getattr(retry_result, "error", "") or "")
            if getattr(retry_result, "ok", False):
                result = retry_result
                review_page = retry_page
        if not getattr(result, "ok", False):
            flags = ["page_text_qwen_failed"]
            if retry_attempted:
                flags.append("page_text_qwen_minimal_retry_failed")
            error = str(getattr(result, "error", "") or "qwen_text_failed")
            if retry_error and retry_error != error:
                error = f"{error}; minimal_retry={retry_error}"
            return PageTextQwenReview(
                review_id=f"page_text_qwen_{index:04d}",
                page_no=int(review_page["page_no"]),
                attempted=True,
                available=False,
                model=self.model,
                status="qwen_failed",
                verdict="ocr_insufficient",
                decision="defer",
                reason="Qwen 页级 OCR 文本复核请求失败。",
                pdf_text_excerpt=str(review_page.get("pdf_text") or "")[:500],
                docx_samples=list(review_page.get("samples") or []),
                next_route="needs_full_page_review",
                error=error,
                model_parse_error=error,
                model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), 700),
                flags=flags,
            )
        parsed = dict(getattr(result, "parsed", {}) or {})
        if self._schema_incomplete(parsed) and not retry_attempted:
            retry_result, retry_page = self._minimal_retry(page=page, timeout=timeout)
            retry_attempted = True
            retry_error = str(getattr(retry_result, "error", "") or "")
            if getattr(retry_result, "ok", False):
                result = retry_result
                review_page = retry_page
                parsed = dict(getattr(result, "parsed", {}) or {})
        verdict = self._choice(parsed.get("verdict"), {"text_risk", "no_obvious_issue", "mapping_uncertain", "ocr_insufficient"}, "ocr_insufficient")
        decision = self._choice(parsed.get("decision"), {"allow_exact_replacements", "defer", "block"}, "defer")
        suspicious_values = self._suspicious_values(
            values=parsed.get("suspicious_values"),
            samples=review_page.get("samples") or [],
            pdf_text=str(review_page.get("pdf_text") or ""),
        )
        if suspicious_values and decision == "block":
            decision = "defer"
        if suspicious_values and verdict in {"no_obvious_issue", "mapping_uncertain", "ocr_insufficient"}:
            verdict = "text_risk"
        if not suspicious_values and decision == "allow_exact_replacements":
            decision = "defer"
        status = "qwen_text_supported_replacement" if suspicious_values else "qwen_text_no_exact_replacement"
        flags = ["page_text_qwen_review"]
        if retry_attempted:
            flags.append("page_text_qwen_minimal_retry_used")
        if suspicious_values:
            flags.append("page_text_qwen_exact_replacements")
        schema_incomplete = not parsed.get("verdict")
        if schema_incomplete:
            flags.append("page_text_qwen_schema_incomplete")
        return PageTextQwenReview(
            review_id=f"page_text_qwen_{index:04d}",
            page_no=int(review_page["page_no"]),
            attempted=True,
            available=not schema_incomplete,
            model=self.model,
            status=status if not schema_incomplete else "qwen_schema_incomplete",
            verdict=verdict,
            decision=decision,
            confidence=self._confidence(parsed.get("confidence")),
            reason=self._clip(parsed.get("reason"), 220),
            pdf_text_excerpt=str(review_page.get("pdf_text") or "")[:500],
            docx_samples=list(review_page.get("samples") or []),
            suspicious_values=suspicious_values,
            next_route=str(parsed.get("next_route") or ("needs_full_page_review" if decision == "defer" else "")).strip(),
            error=retry_error if schema_incomplete and retry_error else ("page_text_qwen_schema_incomplete" if schema_incomplete else ""),
            model_parse_error=retry_error if schema_incomplete and retry_error else ("page_text_qwen_schema_incomplete" if schema_incomplete else ""),
            model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), 700),
            model_parsed_keys=sorted(str(key) for key in parsed.keys()),
            flags=flags,
        )

    def _structured_chat(self, *, page: Dict[str, Any], timeout: int, num_predict: int) -> Any:
        return self.client.structured_chat(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(page=page),
            schema=page_text_qwen_schema(),
            timeout=max(1, min(self.timeout, int(timeout or self.timeout))),
            num_predict=num_predict,
            temperature=0.0,
        )

    def _minimal_retry(self, *, page: Dict[str, Any], timeout: int) -> tuple[Any, Dict[str, Any]]:
        retry_page = self._minimal_retry_page(page=page)
        retry_timeout = max(1, min(self.timeout, int(timeout or self.timeout), 45))
        result = self._structured_chat(
            page=retry_page,
            timeout=retry_timeout,
            num_predict=max(260, min(self.num_predict, 360)),
        )
        return result, retry_page

    def _minimal_retry_page(self, *, page: Dict[str, Any]) -> Dict[str, Any]:
        retry_page = dict(page)
        retry_page["retry_mode"] = "minimal_schema_retry"
        retry_page["pdf_text"] = str(page.get("pdf_text") or "")[:1200]
        samples: List[Dict[str, Any]] = []
        for sample in list(page.get("samples") or [])[:4]:
            row = dict(sample)
            row["text"] = self._clip(row.get("text"), 90)
            if row.get("row_context"):
                row["row_context"] = [
                    {"col_index": item.get("col_index"), "text": self._clip(item.get("text"), 20)}
                    for item in row.get("row_context")[:2]
                    if isinstance(item, dict)
                ]
            samples.append(row)
        retry_page["samples"] = samples
        retry_page["unresolved_sample_count"] = len(samples)
        return retry_page

    def _should_minimal_retry(self, *, result: Any) -> bool:
        if getattr(result, "ok", False):
            return False
        error = str(getattr(result, "error", "") or "").lower()
        if not error:
            return False
        if any(marker in error for marker in ("timeout", "readtimeout", "connection", "connecttimeout")):
            return False
        return any(
            marker in error
            for marker in (
                "json",
                "response_not_object",
                "object_not_found",
                "object_not_final",
                "empty_response",
            )
        )

    def _schema_incomplete(self, parsed: Dict[str, Any]) -> bool:
        return not (
            parsed.get("verdict")
            and parsed.get("decision")
            and isinstance(parsed.get("suspicious_values"), list)
            and "reason" in parsed
            and "confidence" in parsed
            and "next_route" in parsed
        )

    def _suspicious_values(
        self,
        *,
        values: Any,
        samples: Sequence[Dict[str, Any]],
        pdf_text: str,
    ) -> List[Dict[str, Any]]:
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
            pdf_visible = self._clean_text(value.get("pdf_text") or "")
            if not self._valid_exact_replacement(old_text=docx_text, new_text=pdf_visible):
                continue
            if self._is_short_numeric_table_replacement(sample=sample, old_text=docx_text, new_text=pdf_visible):
                continue
            support_score = self._pdf_support_score(pdf_text=pdf_text, suggested_text=pdf_visible, current_text=docx_text)
            if support_score < 0.78:
                continue
            page_kind = str(sample.get("page_kind") or "")
            if page_kind == "table_page" and support_score < 0.9:
                continue
            if page_kind == "table_page" and self._table_support_is_ambiguous(
                pdf_text=pdf_text,
                old_text=docx_text,
                new_text=pdf_visible,
            ):
                continue
            seen_units.add(unit_id)
            rows.append(
                {
                    "unit_id": unit_id,
                    "docx_text": docx_text,
                    "pdf_text": pdf_visible,
                    "issue_type": self._choice(
                        value.get("issue_type"),
                        {
                            "text_substitution",
                            "digit_or_date_error",
                            "name_or_address_error",
                            "missing_or_extra_text",
                            "reading_order_error",
                            "other",
                        },
                        "text_substitution",
                    ),
                    "severity": self._choice(value.get("severity"), {"high", "medium", "low"}, "medium"),
                    "reason": self._clip(value.get("reason"), 180),
                    "ocr_support_score": round(support_score, 4),
                    "source": "page_pdf_ocr_text",
                    "page_kind": page_kind or "text_page",
                    "container_type": sample.get("container_type") or "",
                    "table_index": sample.get("table_index"),
                    "row_index": sample.get("row_index"),
                    "col_index": sample.get("col_index"),
                }
            )
            if len(rows) >= 10:
                break
        return rows

    def _samples(
        self,
        *,
        units: Sequence[DocxEvidenceUnit],
        coverage_by_unit: Dict[str, ContentCoverageReview],
        include_table_cells: bool = False,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        page_units = sorted(units, key=lambda item: int(item.order_index or 0))
        for unit in sorted(units, key=lambda item: int(item.order_index or 0)):
            if unit.container_type == "table_cell" and not include_table_cells:
                continue
            text = " ".join(str(unit.text or "").split())
            if len(normalize_text(text)) < 3:
                continue
            if not include_table_cells and self._looks_like_table_fragment_text(text):
                continue
            review = coverage_by_unit.get(unit.unit_id)
            row = {
                "unit_id": unit.unit_id,
                "text": text[:160],
                "coverage_review_id": review.review_id if review else "",
                "status": review.status if review else "not_in_coverage",
                "decision": review.decision if review else "",
                "order_index": int(unit.order_index or 0),
                "container_type": unit.container_type,
                "page_kind": "table_page" if include_table_cells else "text_page",
            }
            if include_table_cells:
                row.update(
                    {
                        "table_index": int(unit.table_index or 0) or None,
                        "row_index": int(unit.row_index or 0) or None,
                        "col_index": int(unit.col_index or 0) or None,
                        "row_context": self._row_context(unit=unit, units=page_units),
                        "signal_score": self._sample_signal_score(unit=unit, review=review),
                    }
                )
            else:
                row["signal_score"] = self._sample_signal_score(unit=unit, review=review)
            rows.append(
                row
            )
        rows.sort(
            key=lambda item: (
                0 if item["status"] != "covered" else 1,
                -int(item.get("signal_score") or 0),
                int(item["order_index"]),
            )
        )
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        sample_limit = self._sample_limit(include_table_cells=include_table_cells)
        for row in rows:
            key = self._semantic_compact(row["text"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= sample_limit:
                break
        return deduped

    def _docx_by_page(self, units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[DocxEvidenceUnit]]:
        rows: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0 or not str(unit.text or "").strip():
                continue
            rows.setdefault(page_no, []).append(unit)
        return rows

    def _pdf_texts_by_page(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, str]:
        rows: Dict[int, List[str]] = {}
        seen: set[tuple[int, str]] = set()

        def append(page_no: int, text: Any) -> None:
            value = " ".join(str(text or "").split())
            if page_no <= 0 or not value:
                return
            key = (page_no, self._semantic_compact(value)[:260])
            if key in seen:
                return
            seen.add(key)
            rows.setdefault(page_no, []).append(value)

        for page in preflight_result.anchor_ocr.get("pages") or []:
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            append(page_no, page.get("text"))
            for line in page.get("lines") or []:
                append(page_no, line.get("text") if isinstance(line, dict) else line)
        for unit in preflight_result.pdf_units:
            if str(unit.unit_type or "") not in {"anchor_ocr_page", "anchor_ocr_line", "native_text_block"}:
                continue
            append(int(unit.page_no or 0), unit.text)
        for backfill in preflight_result.content_coverage_backfills:
            if not backfill.available:
                continue
            append(int(backfill.page_no or 0), backfill.extracted_text or backfill.normalized_text)
        return {page_no: "\n".join(parts) for page_no, parts in rows.items()}

    def _is_table_page(self, *, profile: Dict[str, Any], units: Sequence[DocxEvidenceUnit]) -> bool:
        labels = set(profile.get("labels") or [])
        primary = str(profile.get("primary_route") or "")
        canonical = str(profile.get("audit_canonical_page_type") or "")
        recognition_strategy = str(profile.get("recognition_strategy") or "")
        if profile.get("needs_table_parser") or profile.get("table_like") or "table_heavy" in labels:
            return True
        if canonical in {"table_image_page", "native_table_page"}:
            return True
        if recognition_strategy == "table_structure_and_cell_ocr":
            return True
        if primary in {"image_table_cell_compare", "native_table_compare"}:
            return True
        table_ratio = sum(1 for unit in units if unit.container_type == "table_cell") / max(1, len(units))
        return table_ratio >= 0.45

    def _row_context(self, *, unit: DocxEvidenceUnit, units: Sequence[DocxEvidenceUnit]) -> List[Dict[str, Any]]:
        if unit.container_type != "table_cell":
            return []
        row_units = [
            item
            for item in units
            if item.unit_id != unit.unit_id
            and item.container_type == "table_cell"
            and int(item.table_index or 0) == int(unit.table_index or 0)
            and int(item.row_index or 0) == int(unit.row_index or 0)
        ]
        row_units.sort(key=lambda item: (int(item.col_index or 9999), int(item.order_index or 0)))
        return [
            {
                "col_index": int(item.col_index or 0) or None,
                "text": self._clip(item.text, 36),
            }
            for item in row_units
            if self._clean_text(item.text)
        ][:4]

    def _sample_limit(self, *, include_table_cells: bool) -> int:
        if include_table_cells:
            return max(4, min(self.max_samples, 12))
        return max(4, min(self.max_samples, 16))

    def _sample_signal_score(self, *, unit: DocxEvidenceUnit, review: Optional[ContentCoverageReview]) -> int:
        text = self._clean_text(unit.text)
        compact = self._semantic_compact(text)
        score = 0
        status = str(getattr(review, "status", "") or "")
        if status and status != "covered":
            score += 30
        if unit.container_type == "table_cell":
            score += 14
        if re.search(r"\d+\.\d+", text):
            score += 18
        if re.search(r"[¥￥]\s*\d|\d+\s*元", text):
            score += 16
        if re.search(r"\d{4}[-/.年]\d{1,2}|[12]\d{10,}", text):
            score += 14
        if re.search(r"[A-Za-z]", text) and re.search(r"\d", text):
            score += 14
        chinese_count = sum("\u4e00" <= ch <= "\u9fff" for ch in compact)
        if chinese_count >= 4:
            score += 10
        if " " in text and chinese_count >= 2:
            score += 8
        if len(compact) <= 2:
            score -= 20
        if self._short_plain_number(text):
            score -= 16
        return score

    def _pdf_support_score(self, *, pdf_text: str, suggested_text: str, current_text: str) -> float:
        suggestion = self._semantic_compact(suggested_text)
        current = self._semantic_compact(current_text)
        page = self._semantic_compact(pdf_text)
        if len(suggestion) < 3 or not page:
            return 0.0
        if suggestion in page:
            return 0.96
        if len(suggestion) >= 8:
            score = self._best_window_ratio(needle=suggestion, haystack=page)
            if score >= 0.88:
                return max(0.78, min(0.92, score))
        if current and current in page and len(current) >= len(suggestion) - 1:
            return 0.0
        date_relaxed = suggestion.replace("日", "")
        if "年" in suggestion and "月" in suggestion and len(date_relaxed) >= 6 and date_relaxed in page:
            return 0.82
        return 0.0

    def _table_support_is_ambiguous(self, *, pdf_text: str, old_text: str, new_text: str) -> bool:
        page = self._semantic_compact(pdf_text)
        old_key = self._semantic_compact(old_text)
        new_key = self._semantic_compact(new_text)
        if not page or not old_key or not new_key:
            return True
        if old_key in new_key or new_key in old_key:
            return False
        if old_key in page and len(old_key) >= 3:
            return True
        return False

    def _is_short_numeric_table_replacement(self, *, sample: Dict[str, Any], old_text: str, new_text: str) -> bool:
        if str(sample.get("page_kind") or "") != "table_page":
            return False
        if not (self._short_plain_number(old_text) and self._short_plain_number(new_text)):
            return False
        return True

    def _short_plain_number(self, value: Any) -> bool:
        text = self._clean_text(value).replace(",", "")
        if not re.fullmatch(r"\d{1,3}", text):
            return False
        return True

    def _looks_like_table_fragment_text(self, text: Any) -> bool:
        return looks_like_paragraphized_table_fragment(text)

    def _valid_exact_replacement(self, *, old_text: str, new_text: str) -> bool:
        old_value = self._clean_text(old_text)
        new_value = self._clean_text(new_text)
        if not old_value or not new_value or old_value == new_value:
            return False
        if self._semantic_compact(old_value) == self._semantic_compact(new_value):
            return False
        if len(new_value) > 180:
            return False
        if any(marker in new_value for marker in ("?", "？", "_", "看不清", "无法", "不确定", "参考", "仍需", "核对", "疑似")):
            return False
        compact = self._semantic_compact(new_value)
        if len(compact) >= 45:
            digit_ratio = sum(ch.isdigit() for ch in compact) / max(1, len(compact))
            ascii_ratio = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in compact) / max(1, len(compact))
            if digit_ratio > 0.45 or ascii_ratio > 0.22:
                return False
        return True

    def _system_prompt(self) -> str:
        return (
            "你是 WPS PDF 转 DOCX 忠实度审查的页级文本复核模型。"
            "你看不到图片，只能把 PDF 页 OCR 文本当作可见内容证据，把 DOCX 文本单元当作 WPS 转换结果。"
            "逐项核查 DOCX 文本是否被 PDF OCR 支持。只输出能在 PDF OCR 文本中找到精确依据的替换项。"
            "不得总结文档，不得做数据清洗建议，不得按常识补写，不得做法律分析。"
            "不确定或 OCR 证据不足时必须 defer。必须只输出符合 schema 的 JSON 对象。"
        )

    def _user_prompt(self, *, page: Dict[str, Any]) -> str:
        payload = {
            "page_no": page["page_no"],
            "page_kind": page.get("page_kind") or "text_page",
            "page_risk": page.get("risk"),
            "coverage_profile": {
                "status": page.get("profile_status"),
                "risk": page.get("profile_risk"),
                "table_evidence_count": page.get("table_evidence_count"),
                "unresolved_sample_count": page.get("unresolved_sample_count"),
            },
            "retry_mode": page.get("retry_mode") or "",
            "pdf_ocr_text": page.get("pdf_text") or "",
            "docx_samples": page.get("samples") or [],
            "output_schema_hint": {
                "verdict": "text_risk|no_obvious_issue|mapping_uncertain|ocr_insufficient",
                "decision": "allow_exact_replacements|defer|block",
                "suspicious_values": [
                    {
                        "unit_id": "必须来自 docx_samples",
                        "docx_text": "DOCX 原文",
                        "pdf_text": "PDF OCR 中实际出现的完整替换片段",
                        "issue_type": "text_substitution|digit_or_date_error|name_or_address_error|missing_or_extra_text|reading_order_error|other",
                        "severity": "high|medium|low",
                        "reason": "一句话证据",
                    }
                ],
                "reason": "一句话",
                "confidence": 0.0,
                "next_route": "",
            },
            "rules": [
                "禁止输出 summary/key_issues/suggested_actions/estimated_total_amount/priority/next_step。",
                "如果 retry_mode=minimal_schema_retry，必须修正为目标 schema，不要复用上一次回答格式。",
                "禁止复述输入样本；禁止只输出某一个 sample 对象。",
                "逐项比较 docx_samples 与 pdf_ocr_text；每条 suspicious_values 必须引用原 unit_id。",
                "pdf_text 必须是 PDF OCR 文本中出现的完整替换片段，不允许凭常识改写。",
                "如果 page_kind=table_page，必须结合 table_index/row_index/col_index/row_context 判断；短数字单元没有上下文时不要输出。",
                "只输出有明确精确替换值的错误；格式、空格、换行、全半角、标点差异不要输出。",
                "如果 DOCX 内容在 PDF OCR 中也能找到，通常不是错误。",
                "如果 OCR 文本混乱导致无法确认，verdict=ocr_insufficient，decision=defer，suspicious_values=[]。",
                "如果只是页映射不稳，verdict=mapping_uncertain，decision=defer，suspicious_values=[]。",
            ],
        }
        return (
            "请基于 PDF OCR 文本核查 DOCX 文本单元。"
            "只输出一个 JSON 对象，不要解释，不要 Markdown。输入："
            + self._bounded_json_payload(payload)
        )

    def _bounded_json_payload(self, payload: Dict[str, Any]) -> str:
        value = dict(payload)
        max_len = 7200
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= max_len:
            return text
        samples = list(value.get("docx_samples") or [])
        compact_samples: List[Dict[str, Any]] = []
        for sample in samples:
            row = dict(sample)
            row["text"] = self._clip(row.get("text"), 96)
            if row.get("row_context"):
                row["row_context"] = [
                    {"col_index": item.get("col_index"), "text": self._clip(item.get("text"), 24)}
                    for item in row.get("row_context")[:3]
                    if isinstance(item, dict)
                ]
            compact_samples.append(row)
        value["docx_samples"] = compact_samples
        pdf_text = str(value.get("pdf_ocr_text") or "")
        for limit in (2600, 2000, 1400, 900):
            value["pdf_ocr_text"] = pdf_text[:limit]
            text = json.dumps(value, ensure_ascii=False)
            if len(text) <= max_len:
                return text
        while len(value.get("docx_samples") or []) > 4:
            value["docx_samples"] = list(value.get("docx_samples") or [])[: max(4, len(value["docx_samples"]) // 2)]
            text = json.dumps(value, ensure_ascii=False)
            if len(text) <= max_len:
                return text
        return json.dumps(value, ensure_ascii=False)

    def _preflight(self) -> tuple[bool, str]:
        preflight = getattr(self.client, "preflight", None)
        if not callable(preflight):
            return True, ""
        result = preflight(timeout=min(30, self.timeout))
        if bool(result.get("available")):
            return True, ""
        return False, str(result.get("error") or result.get("reason") or "qwen_text_preflight_failed")

    def _unload(self) -> None:
        unload = getattr(self.client, "unload", None)
        if callable(unload):
            unload()

    def _budget_exhausted(self, started_at: float) -> bool:
        return bool(self.max_total_seconds and (time.perf_counter() - started_at) >= self.max_total_seconds)

    def _remaining_budget_seconds(self, started_at: float) -> int:
        if not self.max_total_seconds:
            return self.timeout
        elapsed = time.perf_counter() - started_at
        return max(0, min(self.timeout, int(self.max_total_seconds - elapsed)))

    def _skipped_review(self, *, index: int, page: Dict[str, Any], reason: str, flag: str) -> PageTextQwenReview:
        return PageTextQwenReview(
            review_id=f"page_text_qwen_{index:04d}",
            page_no=int(page["page_no"]),
            attempted=False,
            available=False,
            model=self.model,
            status="qwen_skipped",
            verdict="ocr_insufficient",
            decision="defer",
            confidence=0.0,
            reason=reason,
            pdf_text_excerpt=str(page.get("pdf_text") or "")[:500],
            docx_samples=list(page.get("samples") or []),
            next_route="needs_full_page_review",
            error=flag,
            flags=["page_text_qwen_skipped", flag],
        )

    def _emit_progress(self, *, page_no: int, index: int, total: int, event: str, message: str) -> None:
        callback = self.progress_callback
        if not callable(callback):
            return
        try:
            callback(
                {
                    "event": event,
                    "builder": "page_text_qwen_review",
                    "page_no": int(page_no),
                    "item_current": int(index),
                    "item_total": int(total),
                    "message": message,
                }
            )
        except Exception:
            return

    def _choice(self, value: Any, allowed: Sequence[str] | set[str], fallback: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else fallback

    def _confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _clip(self, value: Any, limit: int = 160) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _clean_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _semantic_compact(self, text: Any) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _best_window_ratio(self, *, needle: str, haystack: str) -> float:
        if not needle or not haystack:
            return 0.0
        size = len(needle)
        if len(haystack) <= size + 8:
            return SequenceMatcher(None, needle, haystack).ratio()
        step = max(1, size // 4)
        best = 0.0
        for start in range(0, max(1, len(haystack) - size + 1), step):
            window = haystack[start : start + size + 8]
            best = max(best, SequenceMatcher(None, needle, window).ratio())
            if best >= 0.94:
                break
        return best
