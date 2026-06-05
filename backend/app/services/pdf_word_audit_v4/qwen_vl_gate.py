from __future__ import annotations

import base64
import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

from app.core.config import settings

from .common import normalize_text, similarity
from .models import ConversionDiffCandidate, ConversionPreflightResult, FocusedCandidateReview, QwenVlGateReview
from .partial_artifacts import load_partial_review_payload, write_partial_review_payload
from .qwen_gate import OllamaQwenGateClient, QwenStructuredResult, parse_json_object


def _vl_post_json_worker(queue: Any, endpoint: str, payload: Dict[str, Any], timeout_seconds: int) -> None:
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=(min(10, max(1, int(timeout_seconds or 1))), max(1, int(timeout_seconds or 1))),
        )
        queue.put(
            {
                "status_code": int(response.status_code),
                "text": response.text,
            }
        )
    except Exception as exc:
        queue.put({"error": type(exc).__name__})


def qwen_vl_gate_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["supports_pdf", "supports_docx", "conflict", "unreadable", "not_relevant"],
            },
            "decision": {
                "type": "string",
                "enum": ["allow_report_candidate", "block_candidate", "defer"],
            },
            "visible_text": {"type": "string"},
            "preferred_text": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
            "next_route": {"type": "string"},
        },
        "required": ["verdict", "decision", "visible_text", "preferred_text", "reason", "confidence", "next_route"],
    }


class QwenVlGateBuilder:
    """Execute the visual route that v4 already planned.

    The gate is deliberately small and serial: it only inspects local crops for
    candidates that the focused layer marked as visual or recall risk.
    """

    def __init__(
        self,
        *,
        work_dir: Path,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_candidates: Optional[int] = None,
        min_confidence: Optional[float] = None,
        client: Any = None,
        unload_after: Optional[bool] = None,
        progress_callback: Any = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_ENABLED", True) if enabled is None else enabled)
        self.model = str(model or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MODEL", "qwen3-vl:8b") or "qwen3-vl:8b").strip()
        self.timeout = max(1, int(timeout or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_TIMEOUT", 180) or 180))
        self.max_candidates = max(0, int(max_candidates if max_candidates is not None else getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MAX_CANDIDATES", 9999) or 9999))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence if min_confidence is not None else getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MIN_CONFIDENCE", 0.72) or 0.72)))
        self.strict_selection = bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_STRICT_SELECTION_ENABLED", True))
        self.fast_ocr_gate = bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_FAST_OCR_GATE_ENABLED", True))
        raw_max_total_seconds = getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_GATE_MAX_TOTAL_SECONDS", 0)
        self.max_total_seconds = max(0, int(0 if raw_max_total_seconds is None else raw_max_total_seconds))
        raw_max_failed_requests = getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_GATE_MAX_FAILED_REQUESTS", 0)
        self.max_failed_requests = max(0, int(0 if raw_max_failed_requests is None else raw_max_failed_requests))
        self.client = client or OllamaQwenVlClient(model=self.model)
        self.unload_after = bool(True if unload_after is None else unload_after)
        self.progress_callback = progress_callback

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[QwenVlGateReview]:
        if not self.enabled or self.max_candidates <= 0:
            return []
        selected = self._selected_reviews(preflight_result=preflight_result)
        if not selected:
            return []
        diff_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        selected_diff_ids = [str(review.diff_id or "") for review in selected]
        rows = self._load_partial_reviews(selected_diff_ids=selected_diff_ids)
        completed_diff_ids = {str(row.diff_id or "") for row in rows}
        selected = [review for review in selected if str(review.diff_id or "") not in completed_diff_ids]
        if not selected:
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        pending: List[FocusedCandidateReview] = []
        for review in selected:
            fast_review = self._fast_ocr_gate_review(index=len(rows) + 1, review=review, diff=diff_by_id.get(review.diff_id))
            if fast_review is not None:
                rows.append(fast_review)
                self._write_partial_reviews(rows=rows)
            else:
                pending.append(review)
        if not pending:
            if self.unload_after:
                self._unload()
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        available, preflight_error = self._preflight()
        if not available:
            if self.unload_after:
                self._unload()
            base_index = len(rows)
            rows.extend(
                QwenVlGateReview(
                    vl_id=f"qwen_vl_{base_index + index:04d}",
                    diff_id=review.diff_id,
                    page_no=int(review.page_no or 0),
                    attempted=False,
                    available=False,
                    model=self.model,
                    verdict="unreadable",
                    decision="defer",
                    next_route=review.next_route or "needs_human_visual_review",
                    crop_path=review.crop_path,
                    error=preflight_error or "qwen_vl_unavailable",
                    flags=["qwen_vl_preflight_failed"],
                )
                for index, review in enumerate(pending, start=1)
            )
            self._write_partial_reviews(rows=rows, status="done")
            return rows

        started_at = time.perf_counter()
        failed_requests = 0
        for pending_index, review in enumerate(pending):
            index = len(rows) + 1
            self._emit_progress(
                diff_id=review.diff_id,
                page_no=int(review.page_no or 0),
                index=pending_index + 1,
                total=len(pending),
                event="qwen_vl_gate_candidate_start",
                message=f"Qwen3-VL 视觉门槛复核：第 {pending_index + 1}/{len(pending)} 个候选，PDF 第 {int(review.page_no or 0)} 页。",
            )
            if self._budget_exhausted(started_at):
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        review=skip_review,
                        error="qwen_vl_gate_budget_exhausted",
                        reason="Qwen3-VL 视觉门槛阶段达到总耗时预算，剩余候选转入后续局部 OCR/人工视觉复核。",
                        flag="qwen_vl_gate_budget_exhausted",
                    )
                    for skip_offset, skip_review in enumerate(pending[pending_index:], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            if self.max_failed_requests and failed_requests >= self.max_failed_requests:
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        review=skip_review,
                        error="qwen_vl_gate_failure_circuit_open",
                        reason="Qwen3-VL 视觉门槛连续请求失败或超时，剩余候选转入后续局部 OCR/人工视觉复核。",
                        flag="qwen_vl_gate_failure_circuit_open",
                    )
                    for skip_offset, skip_review in enumerate(pending[pending_index:], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            diff = diff_by_id.get(review.diff_id)
            image_path = self._resolve_crop(review.crop_path)
            if image_path is None or not image_path.exists():
                rows.append(
                    QwenVlGateReview(
                        vl_id=f"qwen_vl_{index:04d}",
                        diff_id=review.diff_id,
                        page_no=int(review.page_no or 0),
                        attempted=False,
                        available=False,
                        model=self.model,
                        verdict="unreadable",
                        decision="defer",
                        next_route="needs_region_ocr",
                        crop_path=review.crop_path,
                        error="qwen_vl_crop_missing",
                        flags=["qwen_vl_missing_crop"],
                    )
                )
                self._write_partial_reviews(rows=rows)
                continue
            request_timeout = self._remaining_budget_seconds(started_at)
            if request_timeout <= 0:
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset,
                        review=skip_review,
                        error="qwen_vl_gate_budget_exhausted",
                        reason="Qwen3-VL 视觉门槛阶段达到总耗时预算，剩余候选转入后续局部 OCR/人工视觉复核。",
                        flag="qwen_vl_gate_budget_exhausted",
                    )
                    for skip_offset, skip_review in enumerate(pending[pending_index:], start=1)
                )
                self._write_partial_reviews(rows=rows)
                break
            gate_review = self._review(
                index=index,
                review=review,
                diff=diff,
                image_path=image_path,
                timeout=request_timeout,
            )
            rows.append(gate_review)
            self._emit_progress(
                diff_id=review.diff_id,
                page_no=int(review.page_no or 0),
                index=pending_index + 1,
                total=len(pending),
                event="qwen_vl_gate_candidate_done",
                message=(
                    f"Qwen3-VL 视觉门槛复核完成：第 {pending_index + 1}/{len(pending)} 个候选，"
                    f"结果 {gate_review.verdict}/{gate_review.decision}。"
                ),
            )
            if self._request_failed(gate_review):
                failed_requests += 1
            else:
                failed_requests = 0
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

    def _emit_progress(self, *, diff_id: str, page_no: int, index: int, total: int, event: str, message: str) -> None:
        callback = self.progress_callback
        if not callable(callback):
            return
        try:
            callback(
                {
                    "event": event,
                    "builder": "qwen_vl_gate",
                    "diff_id": str(diff_id or ""),
                    "page_no": int(page_no or 0),
                    "item_current": int(index),
                    "item_total": int(total),
                    "message": message,
                }
            )
        except Exception:
            return

    def _write_partial_reviews(self, *, rows: Sequence[QwenVlGateReview], status: str = "running") -> None:
        write_partial_review_payload(
            work_dir=self.work_dir,
            filename="qwen_vl_reviews.partial.json",
            version="qwen_vl_gate_v1",
            reviews=rows,
            extra={"status": status, "model": self.model, "resume_key": self._resume_key()},
        )

    def _load_partial_reviews(self, *, selected_diff_ids: Sequence[str]) -> List[QwenVlGateReview]:
        loaded, _payload = load_partial_review_payload(
            work_dir=self.work_dir,
            filename="qwen_vl_reviews.partial.json",
            review_type=QwenVlGateReview,
            expected_resume_key=self._resume_key(),
        )
        if not loaded:
            return []
        order = {str(diff_id): index for index, diff_id in enumerate(selected_diff_ids) if str(diff_id)}
        by_diff: Dict[str, QwenVlGateReview] = {}
        for row in loaded:
            diff_id = str(row.diff_id or "")
            if diff_id in order:
                by_diff[diff_id] = row
        return [by_diff[diff_id] for diff_id in selected_diff_ids if diff_id in by_diff]

    def _resume_key(self) -> str:
        return (
            "qwen_vl_gate_v2:"
            f"model={self.model}:"
            f"max_candidates={self.max_candidates}:"
            f"min_confidence={self.min_confidence:.3f}:"
            f"fast_ocr_gate={int(self.fast_ocr_gate)}:"
            "strict_visual_gate_v1"
        )

    def _skipped_review(
        self,
        *,
        index: int,
        review: FocusedCandidateReview,
        error: str,
        reason: str,
        flag: str,
    ) -> QwenVlGateReview:
        return QwenVlGateReview(
            vl_id=f"qwen_vl_{index:04d}",
            diff_id=review.diff_id,
            page_no=int(review.page_no or 0),
            attempted=False,
            available=False,
            model=self.model,
            verdict="unreadable",
            decision="defer",
            confidence=0.0,
            reason=reason,
            next_route=review.next_route or "needs_human_visual_review",
            crop_path=review.crop_path,
            error=error,
            flags=["qwen_vl_skipped", flag],
        )

    def _request_failed(self, review: QwenVlGateReview) -> bool:
        if not review.attempted or review.available:
            return False
        error = str(review.error or "")
        return bool(
            not error
            or "Timeout" in error
            or "Connection" in error
            or error in {"qwen_vl_failed", "ReadTimeout", "OllamaWallTimeout"}
        )

    def _fast_ocr_gate_review(
        self,
        *,
        index: int,
        review: FocusedCandidateReview,
        diff: ConversionDiffCandidate | None,
    ) -> QwenVlGateReview | None:
        if not self.fast_ocr_gate:
            return None
        visual_text = dict(review.visual_text or {})
        if not visual_text.get("stable") or str(visual_text.get("support") or "") != "pdf":
            return None
        category = str((diff.category if diff else review.category) or "")
        if category not in {"critical_field_changed", "text_substitution"}:
            return None
        quality = str(review.crop_ocr_quality or visual_text.get("crop_ocr_quality") or "").lower()
        if quality == "low":
            return None
        preferred_text = self._clip_text((diff.pdf_value if diff else "") or (diff.pdf_text if diff else "") or review.pdf_text, limit=220)
        visible_text = self._clip_text(review.crop_ocr_text or preferred_text, limit=260)
        if not preferred_text or not self._visible_supports_candidate(
            diff=diff,
            review=review,
            preferred_text=preferred_text,
            visible_text=visible_text,
        ):
            return None
        confidence = max(float(review.confidence or 0.0), 0.78)
        if confidence < self.min_confidence:
            return None
        return QwenVlGateReview(
            vl_id=f"qwen_vl_{index:04d}",
            diff_id=review.diff_id,
            page_no=int(review.page_no or 0),
            attempted=False,
            available=True,
            model="focused_crop_ocr",
            verdict="supports_pdf",
            decision="allow_report_candidate",
            confidence=round(min(0.88, confidence), 4),
            reason="局部 crop OCR 已稳定支持 PDF 侧差异文本，跳过 Qwen3-VL 调用并进入批注门槛。",
            visible_text=visible_text,
            preferred_text=preferred_text,
            next_route="",
            crop_path=review.crop_path,
            flags=[
                "qwen_vl_fast_ocr_gate",
                "qwen_vl_call_skipped",
                "focused_crop_ocr_supports_pdf",
                f"crop_ocr_quality={quality or 'unknown'}",
            ],
        )

    def _selected_reviews(self, *, preflight_result: ConversionPreflightResult) -> List[FocusedCandidateReview]:
        diff_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        priority = {
            "text_substitution": 0,
            "critical_field_changed": 1,
            "unlocated_hard_field": 2,
            "page_coverage_gap": 3,
            "visual_region_unresolved": 4,
        }
        rows = [
            item
            for item in preflight_result.focused_reviews
            if item.crop_path
            and (
                item.next_route == "needs_qwen_vl"
                or item.decision == "needs_visual_review"
                or item.status == "recall_guard"
            )
            and "crop_ocr_supports_docx" not in set(item.flags)
        ]
        if self.strict_selection:
            rows = [item for item in rows if self._worth_visual_model_call(review=item, diff=diff_by_id.get(item.diff_id))]
        rows.sort(
            key=lambda item: (
                1 if "model_recall_candidate" in set(item.flags or []) else 0,
                priority.get(item.category, 99),
                item.page_no or 9999,
                item.diff_id,
            )
        )
        rows = self._dedupe_overlapping_model_recall_reviews(rows=rows, diff_by_id=diff_by_id)
        return rows[: self.max_candidates]

    def _dedupe_overlapping_model_recall_reviews(
        self,
        *,
        rows: Sequence[FocusedCandidateReview],
        diff_by_id: Dict[str, ConversionDiffCandidate],
    ) -> List[FocusedCandidateReview]:
        kept: List[FocusedCandidateReview] = []
        for review in rows:
            if not self._is_model_recall_review(review=review, diff=diff_by_id.get(review.diff_id)):
                kept.append(review)
                continue
            replaced = False
            for index, existing in enumerate(kept):
                if not self._model_recall_reviews_overlap(
                    left=existing,
                    right=review,
                    left_diff=diff_by_id.get(existing.diff_id),
                    right_diff=diff_by_id.get(review.diff_id),
                ):
                    continue
                if self._model_recall_review_score(review=review) > self._model_recall_review_score(review=existing):
                    kept[index] = review
                replaced = True
                break
            if not replaced:
                kept.append(review)
        return kept

    def _is_model_recall_review(
        self,
        *,
        review: FocusedCandidateReview,
        diff: ConversionDiffCandidate | None,
    ) -> bool:
        flags = set(review.flags or [])
        if diff is not None:
            flags.update(diff.flags or [])
        return "model_recall_candidate" in flags

    def _model_recall_reviews_overlap(
        self,
        *,
        left: FocusedCandidateReview,
        right: FocusedCandidateReview,
        left_diff: ConversionDiffCandidate | None,
        right_diff: ConversionDiffCandidate | None,
    ) -> bool:
        if not self._is_model_recall_review(review=left, diff=left_diff) or not self._is_model_recall_review(review=right, diff=right_diff):
            return False
        if int(left.page_no or 0) != int(right.page_no or 0):
            return False
        if str(left.crop_path or "") and str(right.crop_path or "") and str(left.crop_path or "") != str(right.crop_path or ""):
            return False
        left_docx = self._semantic_compact((left_diff.docx_text if left_diff else "") or left.docx_text)
        right_docx = self._semantic_compact((right_diff.docx_text if right_diff else "") or right.docx_text)
        left_pdf = self._semantic_compact((left_diff.pdf_text if left_diff else "") or left.pdf_text)
        right_pdf = self._semantic_compact((right_diff.pdf_text if right_diff else "") or right.pdf_text)
        return bool(
            self._contains_substantive_text(left_docx, right_docx)
            and self._contains_substantive_text(left_pdf, right_pdf)
        )

    def _contains_substantive_text(self, left: str, right: str) -> bool:
        left = str(left or "")
        right = str(right or "")
        if len(left) < 6 or len(right) < 6:
            return False
        return bool(left in right or right in left)

    def _model_recall_review_score(self, *, review: FocusedCandidateReview) -> tuple[int, float, int]:
        text_len = len(self._semantic_compact(review.pdf_text)) + len(self._semantic_compact(review.docx_text))
        quality_bonus = 1 if str(review.crop_ocr_quality or "") == "unit_text_alignment" else 0
        return (text_len, float(review.confidence or 0.0), quality_bonus)

    def _worth_visual_model_call(
        self,
        *,
        review: FocusedCandidateReview,
        diff: ConversionDiffCandidate | None,
    ) -> bool:
        quality = str(review.crop_ocr_quality or "").strip().lower()
        visual_text = dict(review.visual_text or {})
        support = str(visual_text.get("support") or "").strip()
        stable = bool(visual_text.get("stable"))
        if quality == "low" and not (stable and support == "pdf"):
            if not self._high_value_low_quality_visual_candidate(review=review, diff=diff):
                return False
        flags = set(getattr(diff, "flags", []) or [])
        category = str((diff.category if diff else review.category) or "")
        if category in {"unlocated_hard_field", "page_coverage_gap"}:
            pdf_text = str((diff.pdf_text if diff else "") or review.pdf_text or "").strip()
            pdf_value = str((diff.pdf_value if diff else "") or "").strip() if diff else ""
            if "recall_pdf_field_not_matched" in flags and not pdf_text and not pdf_value:
                return False
        return True

    def _high_value_low_quality_visual_candidate(
        self,
        *,
        review: FocusedCandidateReview,
        diff: ConversionDiffCandidate | None,
    ) -> bool:
        category = str((diff.category if diff else review.category) or "")
        if category not in {"text_substitution", "critical_field_changed"}:
            return False
        pdf_text = str((diff.pdf_text if diff else "") or review.pdf_text or "").strip()
        docx_text = str((diff.docx_text if diff else "") or review.docx_text or "").strip()
        if not pdf_text or not docx_text:
            return False
        pdf_tokens = set(self._high_value_tokens(pdf_text))
        docx_tokens = set(self._high_value_tokens(docx_text))
        if not pdf_tokens or not docx_tokens:
            return False
        return bool(pdf_tokens.symmetric_difference(docx_tokens))

    def _high_value_tokens(self, text: str) -> List[str]:
        value = str(text or "")
        patterns = [
            r"(?<!\d)\d{17}[\dXx](?!\d)",
            r"(?<!\d)\d{15}(?!\d)",
            r"[（(]?\d{4}[）)]?[\u4e00-\u9fff]{1,8}\d{2,8}[\u4e00-\u9fff]{0,8}\d+号",
            r"[A-Za-z]{0,4}\d[\w./\\-]{4,24}",
            r"[¥￥]\s*\d+(?:,\d{3})*(?:\.\d+)?",
        ]
        tokens: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, value):
                token = self._semantic_compact(match.group(0))
                if token and token not in tokens:
                    tokens.append(token)
        return tokens

    def _preflight(self) -> tuple[bool, str]:
        preflight = getattr(self.client, "preflight", None)
        if not callable(preflight):
            return True, ""
        result = preflight(timeout=min(30, self.timeout))
        if bool(result.get("available")):
            return True, ""
        return False, str(result.get("error") or result.get("reason") or "qwen_vl_preflight_failed")

    def _review(
        self,
        *,
        index: int,
        review: FocusedCandidateReview,
        diff: ConversionDiffCandidate | None,
        image_path: Path,
        timeout: int,
    ) -> QwenVlGateReview:
        result = self.client.structured_chat_with_image(
            image_path=image_path,
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(review=review, diff=diff),
            schema=qwen_vl_gate_schema(),
            timeout=max(1, int(timeout or self.timeout)),
            num_predict=260,
            temperature=0.0,
        )
        if not getattr(result, "ok", False):
            return QwenVlGateReview(
                vl_id=f"qwen_vl_{index:04d}",
                diff_id=review.diff_id,
                page_no=int(review.page_no or 0),
                attempted=True,
                available=False,
                model=self.model,
                verdict="unreadable",
                decision="defer",
                next_route=review.next_route or "needs_human_visual_review",
                crop_path=review.crop_path,
                error=str(getattr(result, "error", "") or "qwen_vl_failed"),
                flags=["qwen_vl_failed"],
            )

        parsed = dict(getattr(result, "parsed", {}) or {})
        verdict = self._choice(parsed.get("verdict"), {"supports_pdf", "supports_docx", "conflict", "unreadable", "not_relevant"}, "unreadable")
        decision = self._choice(parsed.get("decision"), {"allow_report_candidate", "block_candidate", "defer"}, "defer")
        confidence = self._confidence(parsed.get("confidence"))
        visible_text = self._clip_text(parsed.get("visible_text"), limit=260)
        preferred_text = self._clip_text(parsed.get("preferred_text"), limit=220)
        next_route = str(parsed.get("next_route") or "").strip()
        reason = self._clip_reason(parsed.get("reason"))
        flags: List[str] = ["qwen_vl_gate"]

        if verdict == "supports_docx":
            decision = "block_candidate"
            next_route = ""
            flags.append("qwen_vl_supports_docx")
        elif verdict in {"conflict", "unreadable", "not_relevant"}:
            if decision == "allow_report_candidate":
                decision = "defer"
                flags.append("qwen_vl_allow_blocked_by_unclear_verdict")
            next_route = next_route or ("needs_region_ocr" if verdict == "unreadable" else "needs_human_visual_review")
        elif decision == "allow_report_candidate":
            if confidence < self.min_confidence:
                decision = "defer"
                next_route = next_route or "needs_human_visual_review"
                flags.append("qwen_vl_confidence_below_threshold")
            elif not preferred_text and not visible_text:
                decision = "defer"
                next_route = next_route or "needs_human_visual_review"
                flags.append("qwen_vl_missing_visible_text")
            elif not self._visible_supports_candidate(
                diff=diff,
                review=review,
                preferred_text=preferred_text,
                visible_text=visible_text,
            ) and not self._model_recall_preferred_text_safe(
                diff=diff,
                review=review,
                preferred_text=preferred_text,
                confidence=confidence,
            ):
                decision = "defer"
                next_route = next_route or "needs_human_visual_review"
                flags.append("qwen_vl_preferred_text_not_in_candidate_context")
            elif not self._visible_supports_candidate(diff=diff, review=review, preferred_text=preferred_text, visible_text=visible_text):
                flags.append("qwen_vl_model_recall_preferred_text_supported")
        elif decision == "defer" and not next_route:
            next_route = "needs_human_visual_review"

        rescue = self._visible_text_rescue(diff=diff, review=review, visible_text=visible_text)
        if decision == "defer" and rescue:
            verdict = "supports_pdf"
            decision = "allow_report_candidate"
            preferred_text = str(rescue["preferred_text"])
            confidence = max(confidence, float(rescue["confidence"]))
            next_route = ""
            flags = [
                flag
                for flag in flags
                if flag
                not in {
                    "qwen_vl_allow_blocked_by_unclear_verdict",
                    "qwen_vl_confidence_below_threshold",
                    "qwen_vl_missing_visible_text",
                    "qwen_vl_preferred_text_not_in_candidate_context",
                }
            ]
            flags.extend(str(flag) for flag in rescue["flags"])
            rescue_reason = str(rescue["reason"])
            reason = self._clip_reason(f"{rescue_reason} 原门槛说明：{reason}" if reason else rescue_reason)

        if decision == "allow_report_candidate":
            next_route = ""

        return QwenVlGateReview(
            vl_id=f"qwen_vl_{index:04d}",
            diff_id=review.diff_id,
            page_no=int(review.page_no or 0),
            attempted=True,
            available=True,
            model=self.model,
            verdict=verdict,
            decision=decision,
            confidence=confidence,
            reason=reason,
            visible_text=visible_text,
            preferred_text=preferred_text,
            next_route=next_route,
            crop_path=review.crop_path,
            flags=flags,
        )

    def _visible_supports_candidate(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        preferred_text: str,
        visible_text: str,
    ) -> bool:
        candidate = self._semantic_compact(preferred_text or visible_text)
        if not candidate:
            return False
        pdf_text = self._semantic_compact((diff.pdf_text if diff else "") or review.pdf_text)
        pdf_value = self._semantic_compact((diff.pdf_value if diff else "") or "")
        docx_text = self._semantic_compact((diff.docx_text if diff else "") or review.docx_text)
        docx_value = self._semantic_compact((diff.docx_value if diff else "") or "")
        if self._model_recall_overexpanded_candidate(
            diff=diff,
            review=review,
            candidate=candidate,
            docx_text=docx_text or docx_value,
            pdf_text=pdf_text or pdf_value,
        ):
            return False
        if (pdf_text and (candidate in pdf_text or pdf_text in candidate)) or (pdf_value and (candidate in pdf_value or pdf_value in candidate)):
            return bool(not docx_text or candidate != docx_text or pdf_text != docx_text)
        if self._model_recall_visible_supports_preferred(
            diff=diff,
            review=review,
            candidate=candidate,
            visible_text=visible_text,
            docx_text=docx_text or docx_value,
        ):
            return True
        if diff and diff.category in {"unlocated_hard_field", "page_coverage_gap"}:
            if docx_value and candidate and candidate != docx_value and (docx_value not in candidate):
                return True
            if docx_text and candidate and candidate != docx_text and (docx_text not in candidate):
                return True
        return False

    def _model_recall_overexpanded_candidate(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        candidate: str,
        docx_text: str,
        pdf_text: str,
    ) -> bool:
        if diff is None or "model_recall_candidate" not in set(diff.flags or []):
            return False
        if "full_page_visual_context" not in set(review.flags or []):
            return False
        max_context = max(len(docx_text), len(pdf_text), 1)
        return bool(len(candidate) > max_context * 1.45)

    def _model_recall_visible_supports_preferred(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        candidate: str,
        visible_text: str,
        docx_text: str,
    ) -> bool:
        if diff is None or "model_recall_candidate" not in set(diff.flags or []):
            return False
        if not candidate or not visible_text or not docx_text or candidate == docx_text:
            return False
        if len(candidate) < 3 or len(candidate) > 80:
            return False
        if any(marker in candidate for marker in ("?", "_", "无法", "不清", "不确定", "疑似")):
            return False
        visible = self._semantic_compact(visible_text)
        if not visible:
            return False
        return bool(candidate in visible or visible in candidate)

    def _model_recall_preferred_text_safe(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        preferred_text: str,
        confidence: float,
    ) -> bool:
        if diff is None:
            return False
        flags = set(diff.flags or []) | set(review.flags or [])
        if "model_recall_candidate" not in flags and "full_page_visual_context" not in flags:
            return False
        if confidence < max(self.min_confidence + 0.12, 0.84):
            return False
        candidate = self._semantic_compact(preferred_text)
        docx_text = self._semantic_compact(diff.docx_value or diff.docx_text or review.docx_text)
        pdf_text = self._semantic_compact(diff.pdf_text or diff.pdf_value or review.pdf_text)
        if not candidate or not docx_text or candidate == docx_text:
            return False
        if len(candidate) < 3 or len(candidate) > 80:
            return False
        if any(marker in candidate for marker in ("?", "？", "_", "无法", "不清", "不确定", "疑似", "参考")):
            return False
        if self._model_recall_overexpanded_candidate(
            diff=diff,
            review=review,
            candidate=candidate,
            docx_text=docx_text,
            pdf_text=pdf_text,
        ):
            return False
        if docx_text in candidate and len(candidate) > len(docx_text) * 1.6:
            return False
        return True

    def _visible_text_rescue(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        visible_text: str,
    ) -> Dict[str, Any] | None:
        """Conservatively recover a clear visual value when the old OCR candidate is noisy."""

        category = str((diff.category if diff else review.category) or "")
        if category not in {"text_substitution", "critical_field_changed"}:
            return None
        if not self._visible_text_is_clean_for_rescue(visible_text):
            return None

        docx_text = str((diff.docx_value if diff else "") or (diff.docx_text if diff else "") or review.docx_text or "")
        visible_dates = self._extract_chinese_dates(visible_text, require_four_digit_year=True)
        docx_dates = self._extract_chinese_dates(docx_text, require_four_digit_year=False)
        if visible_dates and docx_dates:
            for visible in visible_dates:
                for docx in docx_dates:
                    if not self._date_rescue_match(visible=visible, docx=docx):
                        continue
                    preferred = f"{visible['year']}年{int(visible['month'])}月{int(visible['day'])}日"
                    if self._semantic_compact(preferred) == self._semantic_compact(docx["raw"]):
                        continue
                    return {
                        "preferred_text": preferred,
                        "confidence": max(self.min_confidence + 0.02, 0.74),
                        "reason": "Qwen-VL visible_text 清楚读出同月同日日期，DOCX 年份疑似少一位或错一位；旧 PDF OCR 候选冲突不再阻断该局部视觉证据。",
                        "flags": ["qwen_vl_visible_text_rescue", "qwen_vl_date_rescue"],
                    }
        structured_identifier_rescue = self._structured_identifier_rescue(
            diff=diff,
            review=review,
            visible_text=visible_text,
            docx_text=docx_text,
        )
        if structured_identifier_rescue:
            return structured_identifier_rescue
        short_text_rescue = self._short_text_substitution_rescue(
            diff=diff,
            review=review,
            visible_text=visible_text,
            docx_text=docx_text,
        )
        if short_text_rescue:
            return short_text_rescue
        return None

    def _visible_text_is_clean_for_rescue(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        if any(marker in value for marker in ["?", "？", "_", "＿", "无法", "不清", "模糊", "看不清", "不确定"]):
            return False
        return True

    def _extract_chinese_dates(self, text: str, *, require_four_digit_year: bool) -> List[Dict[str, str]]:
        value = str(text or "")
        pattern = re.compile(r"(?<!\d)(\d{4}|\d{3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,3})\s*(?:日|号)?")
        dates: List[Dict[str, str]] = []
        for match in pattern.finditer(value):
            year, month, day = match.group(1), match.group(2), match.group(3)
            if require_four_digit_year and len(year) != 4:
                continue
            try:
                month_i = int(month)
                day_i = int(day)
            except ValueError:
                continue
            if not (1 <= month_i <= 12 and 1 <= day_i <= 31):
                continue
            dates.append({"raw": match.group(0), "year": year, "month": str(month_i), "day": str(day_i)})
        return dates

    def _date_rescue_match(self, *, visible: Dict[str, str], docx: Dict[str, str]) -> bool:
        if visible["month"] != docx["month"] or visible["day"] != docx["day"]:
            return False
        visible_year = visible["year"]
        docx_year = docx["year"]
        if visible_year == docx_year:
            return False
        if len(visible_year) != 4:
            return False
        if len(docx_year) == 3:
            return visible_year.startswith(docx_year)
        if len(docx_year) == 4:
            mismatch_count = sum(1 for left, right in zip(visible_year, docx_year) if left != right)
            return mismatch_count == 1 and visible_year[:2] == docx_year[:2]
        return False

    def _structured_identifier_rescue(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        visible_text: str,
        docx_text: str,
    ) -> Dict[str, Any] | None:
        visible_identifiers = self._extract_structured_identifiers(visible_text)
        if not visible_identifiers:
            return None
        docx_semantic = self._semantic_compact(docx_text)
        if not docx_semantic:
            return None
        pdf_semantic = self._semantic_compact((diff.pdf_text if diff else "") or review.pdf_text or "")
        for item in visible_identifiers:
            preferred = self._normalize_structured_identifier(str(item["preferred"]), docx_text=docx_text)
            preferred_semantic = self._semantic_compact(preferred)
            if not preferred_semantic or self._semantic_compact(docx_text) == preferred_semantic:
                continue
            if not self._structured_identifier_related(
                docx_semantic=docx_semantic,
                preferred_semantic=preferred_semantic,
                digits=str(item["digits"]),
            ):
                continue
            if pdf_semantic:
                pdf_digits = self._digits_only(pdf_semantic)
                if pdf_digits and not self._shared_identifier_digits(pdf_digits, str(item["digits"])):
                    continue
            if preferred_semantic in pdf_semantic and preferred_semantic == docx_semantic:
                continue
            return {
                "preferred_text": preferred,
                "confidence": max(self.min_confidence + 0.02, 0.74),
                "reason": "Qwen-VL visible_text 清楚读出结构化标识符文本，DOCX 同一编号/编码存在漏字、拆字或格式污染；旧 PDF OCR 候选噪声不再阻断该视觉证据。",
                "flags": ["qwen_vl_visible_text_rescue", "qwen_vl_structured_identifier_rescue"],
            }
        return None

    def _short_text_substitution_rescue(
        self,
        *,
        diff: ConversionDiffCandidate | None,
        review: FocusedCandidateReview,
        visible_text: str,
        docx_text: str,
    ) -> Dict[str, Any] | None:
        flags = set((diff.flags if diff else []) or []) | set(review.flags or [])
        if "model_recall_candidate" not in flags:
            return None
        visible = self._semantic_compact(visible_text)
        docx = self._semantic_compact(docx_text)
        if not visible or not docx or visible == docx:
            return None
        if len(visible) < 4 or len(visible) > 24 or len(docx) > 24:
            return None
        if any(ch.isdigit() for ch in visible) and len(re.findall(r"\d", visible)) >= max(4, len(visible) // 2):
            return None
        if not any("\u4e00" <= ch <= "\u9fff" for ch in visible):
            return None
        score = float(similarity(visible, docx))
        if score < 0.62 or score >= 0.995:
            return None
        if abs(len(visible) - len(docx)) > 6:
            return None
        return {
            "preferred_text": visible_text.strip(),
            "confidence": max(self.min_confidence + 0.02, 0.74),
            "reason": "Qwen-VL visible_text 对短文本替换给出清晰可见值，适合作为模型召回候选的最终视觉确认。",
            "flags": ["qwen_vl_visible_text_rescue", "qwen_vl_short_text_rescue"],
        }

    def _extract_structured_identifiers(self, text: str) -> List[Dict[str, str]]:
        value = str(text or "")
        rows: List[Dict[str, str]] = []
        patterns = (
            re.compile(r"(?:编号|编码|案号|证号|账号|账户|房号|楼号|单元号)\s*[:：]?\s*[\w\-（）()第号/]{4,40}", flags=re.IGNORECASE),
            re.compile(r"[\u4e00-\u9fffA-Za-z]{0,12}[（(]?\d{2,4}[）)]?\s*第\s*\d{2,10}(?:\s*号)?"),
            re.compile(r"(?:No\.?|NO\.?)\s*[:：]?\s*[A-Za-z0-9\-]{4,24}", flags=re.IGNORECASE),
        )
        seen: set[str] = set()
        for pattern in patterns:
            for match in pattern.finditer(value):
                preferred = re.sub(r"\s+", "", match.group(0))
                digits = self._digits_only(preferred)
                if len(digits) < 4 or len(preferred) > 40 or preferred in seen:
                    continue
                seen.add(preferred)
                rows.append({"raw": match.group(0), "preferred": preferred, "digits": digits})
        return rows

    def _normalize_structured_identifier(self, text: str, *, docx_text: str = "") -> str:
        preferred = re.sub(r"\s+", "", str(text or ""))
        if docx_text and "号" in self._semantic_compact(docx_text) and not preferred.endswith("号") and re.search(r"第\d{2,10}$", preferred):
            preferred += "号"
        return preferred

    def _structured_identifier_related(self, *, docx_semantic: str, preferred_semantic: str, digits: str) -> bool:
        docx_digits = self._digits_only(docx_semantic)
        preferred_digits = self._digits_only(preferred_semantic)
        if not self._shared_identifier_digits(docx_digits, preferred_digits or digits):
            return False
        score = float(similarity(docx_semantic, preferred_semantic))
        if score >= 0.995 or score < 0.4:
            return False
        if abs(len(docx_semantic) - len(preferred_semantic)) > 12:
            return False
        docx_alpha = re.sub(r"\d", "", docx_semantic)
        preferred_alpha = re.sub(r"\d", "", preferred_semantic)
        if docx_alpha and preferred_alpha and docx_alpha not in preferred_alpha and preferred_alpha not in docx_alpha:
            if similarity(docx_alpha, preferred_alpha) < 0.35:
                return False
        return True

    def _shared_identifier_digits(self, left_digits: str, right_digits: str) -> bool:
        if not left_digits or not right_digits:
            return False
        if left_digits in right_digits or right_digits in left_digits:
            return True
        if min(len(left_digits), len(right_digits)) >= 4 and left_digits[-4:] == right_digits[-4:]:
            return True
        return False

    def _digits_only(self, text: str) -> str:
        return re.sub(r"\D+", "", str(text or ""))

    def _resolve_crop(self, crop_path: str) -> Path | None:
        if not crop_path:
            return None
        path = Path(crop_path)
        if path.is_absolute():
            return path
        return self.work_dir / crop_path

    def _system_prompt(self) -> str:
        return (
            "你是 WPS PDF 转 DOCX 忠实度审查的局部视觉门槛。"
            "你只能根据用户提供的一张 PDF 局部截图判断截图可见文字更支持 PDF 候选还是 DOCX 文本。"
            "不要做法律分析，不要猜测截图外内容。看不清就返回 unreadable/defer。必须只输出 JSON。"
        )

    def _user_prompt(self, *, review: FocusedCandidateReview, diff: ConversionDiffCandidate | None) -> str:
        if diff and "model_recall_candidate" in set(diff.flags or []):
            payload = {
                "candidate_type": "model_recall_from_page_text_gap",
                "page_no": review.page_no or diff.pdf_page_no,
                "docx_unit_id": diff.docx_unit_id,
                "docx_text": diff.docx_text or review.docx_text,
                "pdf_ocr_hint": diff.pdf_text or review.pdf_text,
                "task": "只判断 PDF 图片中同一位置可见文字是否支持一个不同于 docx_text 的清晰文本。",
                "rules": [
                    "必须优先看图片；pdf_ocr_hint 只是定位线索，可能有 OCR 错字。",
                    "如果图片清楚显示 DOCX 是对的，verdict=supports_docx，decision=block_candidate。",
                    "如果图片清楚显示 DOCX 错了，verdict=supports_pdf，decision=allow_report_candidate，preferred_text 填图片中真实可见文本，不必照抄 pdf_ocr_hint。",
                    "如果看不清、位置不确定、只能猜，verdict=unreadable 或 conflict，decision=defer。",
                    "只输出 JSON，不要解释 JSON 外内容。",
                ],
            }
            return (
                "请对照 PDF 页面图片审查一个 WPS 转换候选。输出字段固定为 verdict, decision, "
                "visible_text, preferred_text, reason, confidence, next_route。\n"
                f"候选 JSON：{json.dumps(payload, ensure_ascii=False)[:2600]}"
            )
        payload = {
            "diff": diff.to_dict() if diff else {"diff_id": review.diff_id},
            "focused_review": review.to_dict(),
            "rules": [
                "先逐字读取截图里的可见文字，写入 visible_text；不确定的字用 ?，不要脑补。",
                "如果截图可见文字稳定支持 PDF 候选值/文本，并且与 DOCX 文本直接冲突，verdict=supports_pdf。",
                "如果截图可见文字稳定支持 DOCX 文本，verdict=supports_docx 且 decision=block_candidate。",
                "如果 PDF 候选、DOCX 文本和截图文字三者互相冲突，verdict=conflict 且 decision=defer。",
                "如果截图太小、模糊、被截断、只看到局部而无法判断，verdict=unreadable 且 decision=defer。",
                "unlocated_hard_field/page_coverage_gap 是漏检兜底：只有截图明确显示同一字段的可见值与 DOCX 不一致时，才允许 allow_report_candidate；否则 defer 或 block_candidate。",
                "格式、空格、换行、全半角、普通标点差异不得 allow_report_candidate。",
                "preferred_text 只填截图支持的 PDF 正确文本；不确定则留空。",
            ],
        }
        return (
            "请审查这一个候选及配套截图。输出字段固定为 verdict, decision, visible_text, preferred_text, reason, confidence, next_route。\n"
            "decision=allow_report_candidate 表示视觉证据足够写入 reviewed.docx 批注，但仍不自动修改正文。\n"
            "next_route 可为空，或填写 needs_region_ocr / needs_human_visual_review / needs_human_mapping_review。\n"
            f"候选 JSON：{json.dumps(payload, ensure_ascii=False)[:6200]}"
        )

    def _choice(self, value: Any, allowed: Sequence[str] | set[str], fallback: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else fallback

    def _confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _semantic_compact(self, text: Any) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _clip_text(self, value: Any, *, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _clip_reason(self, value: Any, *, limit: int = 90) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        first_sentence = text.split("。", 1)[0].strip()
        if first_sentence and len(first_sentence) <= limit:
            return first_sentence + "。"
        return text[:limit].rstrip() + "..."

    def _unload(self) -> None:
        unload = getattr(self.client, "unload", None)
        if callable(unload):
            unload()


class OllamaQwenVlClient(OllamaQwenGateClient):
    @property
    def show_endpoint(self) -> str:
        return f"{self.base_url}/api/show"

    def preflight(self, *, timeout: int = 30) -> Dict[str, Any]:
        if not self.base_url:
            return {"available": False, "error": "ollama_base_url_missing"}
        if not self.model:
            return {"available": False, "error": "qwen_vl_model_missing"}
        try:
            response = requests.post(self.show_endpoint, json={"model": self.model}, timeout=max(1, int(timeout or 1)))
            response.raise_for_status()
            return {"available": True, "model": self.model, "endpoint": self.show_endpoint}
        except Exception as exc:
            return {"available": False, "model": self.model, "endpoint": self.show_endpoint, "error": type(exc).__name__}

    def _post_chat(self, *, chat_payload: Dict[str, Any], timeout: int) -> QwenStructuredResult:
        timeout_seconds = max(1, int(timeout or 1))
        raw, error = self._post_json_hard_timeout(
            endpoint=self.chat_endpoint,
            payload=chat_payload,
            timeout_seconds=timeout_seconds,
        )
        if error:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.chat_endpoint,
                error=error,
                metadata={"transport": "chat", "timeout_seconds": timeout_seconds, "hard_timeout": True},
            )
        content = self._chat_content(raw)
        parsed, parse_error = parse_json_object(content)
        if parse_error:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.chat_endpoint,
                raw_content=content,
                error=parse_error,
                metadata={"transport": "chat", "timeout_seconds": timeout_seconds, "hard_timeout": True},
            )
        return QwenStructuredResult(
            ok=True,
            model=self.model,
            endpoint=self.chat_endpoint,
            parsed=parsed,
            raw_content=content,
            metadata={"transport": "chat", "timeout_seconds": timeout_seconds, "hard_timeout": True},
        )

    def _post_generate(self, *, generate_payload: Dict[str, Any], timeout: int) -> QwenStructuredResult:
        timeout_seconds = max(1, int(timeout or 1))
        raw, error = self._post_json_hard_timeout(
            endpoint=self.generate_endpoint,
            payload=generate_payload,
            timeout_seconds=timeout_seconds,
        )
        if error:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.generate_endpoint,
                error=error,
                metadata={"transport": "generate", "timeout_seconds": timeout_seconds, "hard_timeout": True},
            )
        content = self._generate_content(raw)
        parsed, parse_error = parse_json_object(content)
        if parse_error:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.generate_endpoint,
                raw_content=content,
                error=parse_error,
                metadata={"transport": "generate", "timeout_seconds": timeout_seconds, "hard_timeout": True},
            )
        return QwenStructuredResult(
            ok=True,
            model=self.model,
            endpoint=self.generate_endpoint,
            parsed=parsed,
            raw_content=content,
            metadata={"transport": "generate", "timeout_seconds": timeout_seconds, "hard_timeout": True},
        )

    def _post_json_hard_timeout(
        self,
        *,
        endpoint: str,
        payload: Dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[Dict[str, Any], str]:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_vl_post_json_worker,
            args=(queue, endpoint, payload, max(1, int(timeout_seconds or 1))),
        )
        process.daemon = True
        try:
            process.start()
            process.join(max(1, int(timeout_seconds or 1)) + 2)
            if process.is_alive():
                process.terminate()
                process.join(2)
                return {}, "OllamaWallTimeout"
            if queue.empty():
                return {}, "empty_response"
            row = queue.get_nowait()
        except Exception as exc:
            if process.is_alive():
                process.terminate()
                process.join(2)
            return {}, type(exc).__name__
        finally:
            try:
                queue.close()
            except Exception:
                pass
        if not isinstance(row, dict):
            return {}, "invalid_response"
        if row.get("error"):
            return {}, str(row.get("error"))
        status_code = int(row.get("status_code") or 0)
        if status_code >= 400:
            return {}, "HTTPError"
        try:
            return json.loads(str(row.get("text") or "{}")), ""
        except Exception:
            return {}, "JSONDecodeError"

    def structured_chat_with_image(
        self,
        *,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        timeout: int,
        num_predict: int = 256,
        temperature: float = 0.0,
        allow_generate_fallback: bool = True,
    ) -> QwenStructuredResult:
        if not self.base_url:
            return QwenStructuredResult(ok=False, model=self.model, endpoint="", error="ollama_base_url_missing")
        if not self.model:
            return QwenStructuredResult(ok=False, model="", endpoint=self.chat_endpoint, error="qwen_vl_model_missing")
        try:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        except Exception as exc:
            return QwenStructuredResult(ok=False, model=self.model, endpoint=self.chat_endpoint, error=f"image_read_failed:{type(exc).__name__}")

        messages = [
            {"role": "system", "content": "/no_think\n关闭思考。只输出合法 JSON。\n" + str(system_prompt or "")},
            {"role": "user", "content": "/no_think\n" + str(user_prompt or ""), "images": [encoded]},
        ]
        keep_alive = str(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_KEEP_ALIVE", "5m") or "5m")
        chat_payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": float(temperature),
                "num_predict": max(1, int(num_predict or 256)),
            },
        }
        result = self._post_chat(chat_payload=chat_payload, timeout=timeout)
        if result.ok:
            return result
        if self._timeout_like_error(result.error):
            result.metadata["generate_fallback_skipped"] = "chat_timeout"
            return result
        if not allow_generate_fallback:
            result.metadata["generate_fallback_skipped"] = "disabled_for_call"
            return result
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_GENERATE_FALLBACK_ENABLED", False)):
            result.metadata["generate_fallback_skipped"] = "disabled_for_visual_timeout_control"
            return result

        generate_payload = {
            "model": self.model,
            "prompt": messages[0]["content"] + "\n" + messages[1]["content"],
            "images": [encoded],
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": float(temperature),
                "num_predict": max(1, int(num_predict or 256)),
            },
        }
        fallback = self._post_generate(generate_payload=generate_payload, timeout=timeout)
        if fallback.ok:
            fallback.attempts = 2
            fallback.metadata["chat_error"] = result.error
        return fallback

    def _timeout_like_error(self, error: Any) -> bool:
        text = str(error or "")
        return bool("Timeout" in text or text in {"ReadTimeout", "OllamaWallTimeout"})
