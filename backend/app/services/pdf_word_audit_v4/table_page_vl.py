from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw

from app.core.config import settings

from .common import looks_like_paragraphized_table_fragment, looks_like_table_title, table_text_artifact_replacement
from .models import ContentCoverageReview, ConversionPreflightResult, DocxEvidenceUnit, TablePageVlReview
from .partial_artifacts import load_partial_review_payload, write_partial_review_payload
from .qwen_vl_gate import OllamaQwenVlClient


def table_page_vl_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["table_risk", "no_obvious_issue", "mapping_uncertain", "unreadable"],
            },
            "visible_text_excerpt": {"type": "string"},
            "suspicious_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sample_id": {"type": "string"},
                        "unit_id": {"type": "string"},
                        "docx_text": {"type": "string"},
                        "visible_text": {"type": "string"},
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "digit_letter_confusion",
                                "decimal_point_missing",
                                "decimal_or_punctuation_pollution",
                                "value_mismatch",
                                "missing_or_extra_cell",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "row": {"type": "string"},
                        "col": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["docx_text", "visible_text", "issue_type", "severity", "reason"],
                },
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
            "next_route": {"type": "string"},
        },
        "required": ["verdict", "visible_text_excerpt", "suspicious_values", "reason", "confidence", "next_route"],
    }


def table_page_orientation_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rotation_degrees": {"type": "string", "enum": ["0", "90", "180", "270"]},
            "readability": {"type": "string", "enum": ["good", "partial", "unreadable"]},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["rotation_degrees", "readability", "reason", "confidence"],
    }


class TablePageVlBuilder:
    """Use Qwen-VL on a tiny set of high-risk table pages.

    This is intentionally page-level and quality-first. It does not try to parse every
    table cell. It asks the visual model to inspect the rendered PDF page against
    unresolved DOCX table samples and surface high-risk value examples.
    """

    TARGET_STATUSES = {"table_pending", "mapping_uncertain", "diff_candidate"}

    def __init__(
        self,
        *,
        work_dir: Path,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_qwen_pages: Optional[int] = None,
        max_samples: Optional[int] = None,
        orientation_enabled: Optional[bool] = None,
        client: Any = None,
        unload_after: Optional[bool] = None,
        progress_callback: Any = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_ENABLED", True) if enabled is None else enabled)
        self.model = str(model or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MODEL", "qwen3-vl:8b") or "qwen3-vl:8b").strip()
        self.timeout = max(
            1,
            int(
                timeout
                or getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_TIMEOUT", None)
                or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_TIMEOUT", 180)
                or 180
            ),
        )
        self.max_pages = max(0, int(max_pages if max_pages is not None else getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_MAX_PAGES", 9999) or 9999))
        raw_max_qwen_pages = (
            max_qwen_pages
            if max_qwen_pages is not None
            else getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_MAX_QWEN_PAGES", 9999)
        )
        self.max_qwen_pages = max(0, int(raw_max_qwen_pages if raw_max_qwen_pages is not None else 9999))
        self.max_samples = max(1, int(max_samples if max_samples is not None else getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_MAX_SAMPLES", 36) or 36))
        raw_max_total_seconds = getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_MAX_TOTAL_SECONDS", 0)
        self.max_total_seconds = max(0, int(0 if raw_max_total_seconds is None else raw_max_total_seconds))
        self.prompt_max_chars = max(
            3200,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_PROMPT_MAX_CHARS", 6000) or 6000),
        )
        self.num_predict = max(
            720,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_NUM_PREDICT", 1600) or 1600),
        )
        self.fast_ocr_bypass = bool(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_FAST_OCR_BYPASS_ENABLED", True))
        self.fast_ocr_min_values = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_FAST_OCR_MIN_VALUES", 2) or 2))
        self.deep_min_unresolved = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_DEEP_MIN_UNRESOLVED", 6) or 6))
        self.orientation_enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_VL_ORIENTATION_ENABLED", True)
            if orientation_enabled is None
            else orientation_enabled
        )
        self.client = client or OllamaQwenVlClient(model=self.model)
        self.unload_after = bool(True if unload_after is None else unload_after)
        self.progress_callback = progress_callback

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        rendered_pages: Dict[int, Path],
    ) -> List[TablePageVlReview]:
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
        started_at = time.perf_counter()
        pending: List[Dict[str, Any]] = []
        deferred_without_deep_signal: List[Dict[str, Any]] = []
        for page in selected:
            image_path = rendered_pages.get(page["page_no"])
            fast_review = self._fast_ocr_review(index=len(rows) + 1, page=page, image_path=Path(image_path) if image_path else None)
            needs_deep = self.max_qwen_pages > 0 and self._needs_deep_qwen_review(page=page, fast_review=fast_review)
            if needs_deep:
                page_with_fast = dict(page)
                page_with_fast["_fast_review"] = fast_review
                pending.append(page_with_fast)
            elif fast_review is not None:
                rows.append(fast_review)
                self._write_partial_reviews(rows=rows)
            else:
                deferred_without_deep_signal.append(page)
        if self.max_qwen_pages <= 0:
            for page in pending:
                fast_review = page.get("_fast_review")
                if fast_review is not None:
                    rows.append(fast_review)
                else:
                    rows.append(
                        self._skipped_review(
                            index=len(rows) + 1,
                            page=page,
                            reason="表格页 Qwen3-VL 深度复核页数预算为 0，该页保留为表格人工复核路由。",
                            flag="table_page_vl_qwen_page_budget_deferred",
                        )
                    )
                self._write_partial_reviews(rows=rows)
            self._append_deep_gate_deferred_reviews(rows=rows, pages=deferred_without_deep_signal)
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        qwen_pending = pending[: self.max_qwen_pages]
        overflow_pending = pending[self.max_qwen_pages :]
        pending = qwen_pending
        if not pending:
            self._append_budget_deferred_reviews(rows=rows, pages=overflow_pending)
            self._append_deep_gate_deferred_reviews(rows=rows, pages=deferred_without_deep_signal)
            self._write_partial_reviews(rows=rows, status="done")
            return rows
        available, error = self._preflight()
        if not available:
            fallback_pages: List[Dict[str, Any]] = []
            for page in pending:
                fast_review = page.get("_fast_review")
                if fast_review is not None:
                    rows.append(fast_review)
                else:
                    fallback_pages.append(page)
            if self.unload_after:
                self._unload()
            base_index = len(rows)
            rows.extend(
                TablePageVlReview(
                    review_id=f"table_vl_{base_index + index:04d}",
                    page_no=page["page_no"],
                    attempted=False,
                    available=False,
                    model=self.model,
                    verdict="unreadable",
                    reason="Qwen-VL 当前不可用，无法执行表格页视觉复核。",
                    docx_samples=list(page["samples"]),
                    next_route="needs_human_table_review",
                    error=error or "qwen_vl_unavailable",
                    flags=["table_page_vl_preflight_failed"],
                )
                for index, page in enumerate(fallback_pages, start=1)
            )
            self._append_budget_deferred_reviews(rows=rows, pages=overflow_pending)
            self._append_deep_gate_deferred_reviews(rows=rows, pages=deferred_without_deep_signal)
            self._write_partial_reviews(rows=rows, status="done")
            return rows

        for pending_offset, page in enumerate(pending):
            index = len(rows) + 1
            self._emit_progress(
                page_no=int(page["page_no"]),
                index=pending_offset + 1,
                total=len(pending),
                event="table_page_vl_page_start",
                message=f"表格页 Qwen3-VL 视觉复核：第 {pending_offset + 1}/{len(pending)} 个候选页，PDF 第 {int(page['page_no'])} 页。",
            )
            if self._budget_exhausted(started_at):
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset + 1,
                        page=skip_page,
                        reason="表格页 Qwen3-VL 已达到总耗时预算，跳过剩余页，保留表格证据和人工复核路由。",
                        flag="table_page_vl_total_budget_exhausted",
                    )
                    for skip_offset, skip_page in enumerate(pending[pending_offset:])
                )
                self._write_partial_reviews(rows=rows)
                break
            request_timeout = self._remaining_budget_seconds(started_at)
            if request_timeout <= 0:
                base_index = len(rows)
                rows.extend(
                    self._skipped_review(
                        index=base_index + skip_offset + 1,
                        page=skip_page,
                        reason="表格页 Qwen3-VL 已达到总耗时预算，跳过剩余页，保留表格证据和人工复核路由。",
                        flag="table_page_vl_total_budget_exhausted",
                    )
                    for skip_offset, skip_page in enumerate(pending[pending_offset:])
                )
                self._write_partial_reviews(rows=rows)
                break
            image_path = rendered_pages.get(page["page_no"])
            if image_path is None or not Path(image_path).exists():
                rows.append(
                    TablePageVlReview(
                        review_id=f"table_vl_{index:04d}",
                        page_no=page["page_no"],
                        attempted=False,
                        available=False,
                        model=self.model,
                        verdict="unreadable",
                        reason="未找到 PDF 页面渲染图，无法执行表格页视觉复核。",
                        docx_samples=list(page["samples"]),
                        next_route="needs_human_table_review",
                        error="missing_page_image",
                        flags=["table_page_vl_missing_page_image"],
                    )
                )
                self._write_partial_reviews(rows=rows)
                continue
            review = self._review(index=index, page=page, image_path=Path(image_path), timeout=request_timeout)
            rows.append(review)
            self._emit_progress(
                page_no=int(page["page_no"]),
                index=pending_offset + 1,
                total=len(pending),
                event="table_page_vl_page_done",
                message=(
                    f"表格页 Qwen3-VL 视觉复核完成：第 {pending_offset + 1}/{len(pending)} 个候选页，"
                    f"PDF 第 {int(page['page_no'])} 页，结果 {review.verdict}。"
                ),
            )
            self._write_partial_reviews(rows=rows)
        if self.unload_after:
            self._unload()
        self._append_budget_deferred_reviews(rows=rows, pages=overflow_pending)
        self._append_deep_gate_deferred_reviews(rows=rows, pages=deferred_without_deep_signal)
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
                    "builder": "table_page_vl",
                    "page_no": int(page_no),
                    "item_current": int(index),
                    "item_total": int(total),
                    "message": message,
                }
            )
        except Exception:
            return

    def _write_partial_reviews(self, *, rows: Sequence[TablePageVlReview], status: str = "running") -> None:
        write_partial_review_payload(
            work_dir=self.work_dir,
            filename="table_page_vl_reviews.partial.json",
            version="table_page_vl_v1",
            reviews=rows,
            extra={"status": status, "model": self.model, "resume_key": self._resume_key()},
        )

    def _load_partial_reviews(self, *, selected_page_nos: Sequence[int]) -> List[TablePageVlReview]:
        loaded, _payload = load_partial_review_payload(
            work_dir=self.work_dir,
            filename="table_page_vl_reviews.partial.json",
            review_type=TablePageVlReview,
            expected_resume_key=self._resume_key(),
        )
        if not loaded:
            return []
        order = {int(page_no): index for index, page_no in enumerate(selected_page_nos)}
        by_page: Dict[int, TablePageVlReview] = {}
        for row in loaded:
            page_no = int(row.page_no or 0)
            if page_no in order:
                by_page[page_no] = row
        return [by_page[page_no] for page_no in selected_page_nos if page_no in by_page]

    def _resume_key(self) -> str:
        return (
            "table_page_vl_v2:"
            f"model={self.model}:"
            f"max_samples={self.max_samples}:"
            f"fast_ocr={int(self.fast_ocr_bypass)}:"
            "strict_suspicious_filter_v2"
        )

    def _skipped_review(self, *, index: int, page: Dict[str, Any], reason: str, flag: str) -> TablePageVlReview:
        return TablePageVlReview(
            review_id=f"table_vl_{index:04d}",
            page_no=int(page["page_no"]),
            attempted=False,
            available=False,
            model=self.model,
            verdict="unreadable",
            confidence=0.0,
            reason=reason,
            docx_samples=list(page["samples"]),
            next_route="needs_human_table_review",
            error=flag,
            flags=["table_page_vl_skipped", flag],
        )

    def _append_budget_deferred_reviews(self, *, rows: List[TablePageVlReview], pages: Sequence[Dict[str, Any]]) -> None:
        for page in pages:
            fast_review = page.get("_fast_review")
            if fast_review is not None:
                rows.append(fast_review)
            else:
                rows.append(
                    self._skipped_review(
                        index=len(rows) + 1,
                        page=page,
                        reason="表格页 Qwen3-VL 已达到页数预算，该页保留为表格人工复核路由。",
                        flag="table_page_vl_qwen_page_budget_deferred",
                    )
                )

    def _append_deep_gate_deferred_reviews(self, *, rows: List[TablePageVlReview], pages: Sequence[Dict[str, Any]]) -> None:
        for page in pages:
            rows.append(
                self._skipped_review(
                    index=len(rows) + 1,
                    page=page,
                    reason="该表格页被选中但没有快速 OCR 支持值，也未达到深度 Qwen3-VL 触发阈值，保留为表格人工复核路由。",
                    flag="table_page_vl_deep_gate_deferred",
                )
            )

    def _fast_ocr_review(self, *, index: int, page: Dict[str, Any], image_path: Path | None) -> TablePageVlReview | None:
        if not self.fast_ocr_bypass:
            return None
        suspicious_values = self._ocr_supported_suspicious_values(page=page, existing=[])
        if len(suspicious_values) < self.fast_ocr_min_values:
            return None
        relative = self._relative_path(image_path) if image_path else ""
        flags = [
            "table_page_vl",
            "table_page_vl_fast_ocr_bypass",
            "table_page_ocr_supported_suspicious_values",
            "table_page_vl_suspicious_values",
        ]
        return TablePageVlReview(
            review_id=f"table_vl_{index:04d}",
            page_no=int(page["page_no"]),
            attempted=False,
            available=True,
            model="pdf_page_ocr",
            verdict="table_risk",
            confidence=0.74,
            reason=(
                "PDF 页级 OCR 已保守匹配出多个 DOCX 表格值的数字/字母混淆或小数点异常，"
                "跳过整页 Qwen3-VL 等待并直接进入人工表格复核提示。"
            ),
            page_image_path=relative,
            visible_text_excerpt=self._ocr_excerpt(suspicious_values),
            suspicious_values=suspicious_values,
            docx_samples=list(page["samples"]),
            next_route="needs_human_table_review",
            flags=flags,
        )

    def _selected_pages(self, *, preflight_result: ConversionPreflightResult) -> List[Dict[str, Any]]:
        docx_by_id = {item.unit_id: item for item in preflight_result.docx_units}
        pdf_signals = self._pdf_table_signals(preflight_result=preflight_result)
        grid_context_by_page = self._table_grid_context_by_page(preflight_result=preflight_result)
        evidence_samples_by_page = self._table_cell_evidence_samples_by_page(
            preflight_result=preflight_result,
            docx_by_id=docx_by_id,
        )
        has_pdf_units = bool(preflight_result.pdf_units)
        grouped: Dict[int, List[ContentCoverageReview]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.side != "docx" or review.status not in self.TARGET_STATUSES or review.decision == "covered":
                continue
            unit = docx_by_id.get(review.unit_id)
            profile = dict(preflight_result.page_profiles.get(str(int(review.page_no or 0))) or {})
            is_table = (
                bool(unit and unit.container_type == "table_cell")
                or "container=table_cell" in set(review.flags)
                or bool(unit and self._is_paragraphized_table_fragment(unit=unit, profile=profile))
            )
            if not is_table:
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            grouped.setdefault(page_no, []).append(review)

        pages: List[Dict[str, Any]] = []
        for page_no in sorted(set(grouped) | set(evidence_samples_by_page)):
            rows = grouped.get(page_no, [])
            priority_samples = evidence_samples_by_page.get(page_no, [])
            signal = pdf_signals.get(page_no, {})
            profile = dict(preflight_result.page_profiles.get(str(page_no)) or {})
            if has_pdf_units and not self._pdf_supports_table_page(
                signal,
                profile=profile,
                docx_row_count=len(rows) + len(priority_samples),
            ):
                continue
            samples = self._samples(rows=rows, docx_by_id=docx_by_id)
            samples = self._merge_priority_samples(samples=samples, priority_samples=priority_samples)
            if not samples:
                continue
            suspicious_count = sum(1 for item in samples if self._has_confusable_letter(item.get("text", "")))
            digit_count = sum(1 for item in samples if any(ch.isdigit() for ch in str(item.get("text", ""))))
            high_signal_count = sum(1 for item in samples if self._high_signal(item.get("text", "")))
            priority_sample_count = sum(1 for item in samples if str(item.get("ocr_candidate_text") or "").strip())
            grid_context = grid_context_by_page.get(page_no, [])
            grid_anomaly_count = sum(int(item.get("anomaly_count") or 0) for item in grid_context)
            grid_confirmed_count = sum(int(item.get("confirmed_error_count") or 0) for item in grid_context)
            grid_suspected_count = sum(int(item.get("suspected_error_count") or 0) for item in grid_context)
            score = (
                suspicious_count * 100
                + priority_sample_count * 90
                + grid_suspected_count * 55
                + grid_confirmed_count * 35
                + grid_anomaly_count * 10
                + digit_count * 4
                + high_signal_count * 2
                + min(len(rows) + len(priority_samples), 80)
            )
            pages.append(
                {
                    "page_no": page_no,
                    "score": score,
                    "unresolved_count": len(rows) + len(priority_samples),
                    "suspicious_count": suspicious_count,
                    "digit_count": digit_count,
                    "pdf_table_signal": signal,
                    "table_grid_context": grid_context,
                    "samples": samples,
                }
            )
        pages.sort(key=lambda item: (-int(item["score"]), int(item["page_no"])))
        return pages

    def _table_cell_evidence_samples_by_page(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        docx_by_id: Dict[str, DocxEvidenceUnit],
    ) -> Dict[int, List[Dict[str, Any]]]:
        rows: Dict[int, List[Dict[str, Any]]] = {}
        seen: set[tuple[int, str, str]] = set()
        reviewable_statuses = {
            "pdf_ocr_ambiguous_mismatch",
            "pdf_ocr_decimal_context_needs_review",
            "pdf_ocr_context_confirmed_mismatch",
        }
        reviewable_issue_types = {
            "digit_letter_confusion",
            "decimal_point_missing",
            "decimal_or_punctuation_pollution",
            "value_mismatch",
        }
        for review in preflight_result.table_cell_evidence_reviews:
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            docx_text = " ".join(str(review.docx_text or "").split())[:80]
            visible_text = " ".join(str(review.visible_text or "").split())[:80]
            if not docx_text or not visible_text:
                continue
            if self._compact(docx_text) == self._compact(visible_text):
                continue
            issue_type = str(review.issue_type or "").strip()
            flags = set(review.flags or [])
            needs_visual_confirmation = (
                review.decision == "suspected_error"
                or review.status in reviewable_statuses
                or "requires_human_decimal_review" in flags
                or "table_context_confirmed_decimal_missing" in flags
            )
            if not needs_visual_confirmation or issue_type not in reviewable_issue_types:
                continue
            unit = docx_by_id.get(review.docx_unit_id)
            signature = (page_no, review.docx_unit_id or review.review_id, visible_text)
            if signature in seen:
                continue
            seen.add(signature)
            rows.setdefault(page_no, []).append(
                {
                    "coverage_review_id": review.review_id,
                    "sample_id": review.review_id,
                    "unit_id": review.docx_unit_id,
                    "text": docx_text,
                    "status": review.status,
                    "sample_category": self._sample_category(docx_text),
                    "table_index": unit.table_index if unit else review.table_index,
                    "row_index": unit.row_index if unit else review.row_index,
                    "col_index": unit.col_index if unit else review.col_index,
                    "row_text": self._row_context_text(unit=unit, docx_by_id=docx_by_id) if unit else "",
                    "issue_type_hint": self._clip(issue_type, limit=32),
                    "ocr_candidate_text": visible_text,
                    "review_hint": self._clip(review.reason, limit=140),
                }
            )
        for samples in rows.values():
            samples.sort(
                key=lambda item: (
                    0 if item.get("sample_category") == "numeric_confusable" else 1,
                    0 if item.get("issue_type_hint") == "digit_letter_confusion" else 1,
                    str(item.get("row_index") or ""),
                    str(item.get("col_index") or ""),
                )
            )
        return rows

    def _merge_priority_samples(
        self,
        *,
        samples: Sequence[Dict[str, Any]],
        priority_samples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not priority_samples:
            return list(samples)
        merged: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for source in (priority_samples, samples):
            for item in source:
                signature = (
                    str(item.get("unit_id") or ""),
                    str(item.get("coverage_review_id") or item.get("sample_id") or ""),
                    self._clip(item.get("text"), limit=80),
                    self._clip(item.get("ocr_candidate_text"), limit=80),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                merged.append(dict(item))
        return merged

    def _table_grid_context_by_page(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, List[Dict[str, Any]]]:
        rows: Dict[int, List[Dict[str, Any]]] = {}
        for grid in preflight_result.table_grid_evidence:
            page_no = int(grid.page_no or 0)
            if page_no <= 0:
                continue
            anomalies = [
                {
                    "unit_id": self._clip(item.get("unit_id"), limit=48),
                    "row": self._clip(item.get("row_index"), limit=12),
                    "col": self._clip(item.get("col_index"), limit=12),
                    "docx_text": self._clip(item.get("text"), limit=42),
                    "expected_hint": self._clip(item.get("expected_text"), limit=42),
                    "type": self._clip(item.get("anomaly_type"), limit=40),
                    "decision": self._clip(item.get("decision"), limit=28),
                    "reason": self._clip(item.get("reason"), limit=90),
                }
                for item in list(grid.anomaly_cells or [])[:12]
            ]
            columns = [
                {
                    "col": int(item.get("col_index") or 0),
                    "type": self._clip(item.get("dominant_type"), limit=24),
                    "unresolved": int(item.get("unresolved_count") or 0),
                    "confirmed": int(item.get("confirmed_evidence_count") or 0),
                    "suspected": int(item.get("suspected_evidence_count") or 0),
                    "decimal_evidence": int(item.get("decimal_evidence_count") or 0),
                    "headers": [self._clip(value, limit=24) for value in list(item.get("header_samples") or [])[:4]],
                    "values": [self._clip(value, limit=24) for value in list(item.get("value_samples") or [])[:4]],
                }
                for item in list(grid.column_profiles or [])[:10]
                if int(item.get("unresolved_count") or 0) > 0
                or int(item.get("confirmed_evidence_count") or 0) > 0
                or int(item.get("suspected_evidence_count") or 0) > 0
                or int(item.get("decimal_evidence_count") or 0) > 0
            ]
            rows.setdefault(page_no, []).append(
                {
                    "grid_id": grid.grid_id,
                    "table_index": int(grid.table_index or 0),
                    "status": grid.status,
                    "rows": int(grid.row_count or 0),
                    "cols": int(grid.col_count or 0),
                    "cells": int(grid.cell_count or 0),
                    "unresolved_cell_count": int(grid.unresolved_cell_count or 0),
                    "confirmed_error_count": int(grid.confirmed_error_count or 0),
                    "suspected_error_count": int(grid.suspected_error_count or 0),
                    "no_issue_sample_count": int(grid.no_issue_sample_count or 0),
                    "anomaly_count": len(grid.anomaly_cells or []),
                    "route_hints": [self._clip(value, limit=40) for value in list(grid.route_hints or [])[:8]],
                    "columns": columns,
                    "anomaly_cells": anomalies,
                }
            )
        for page_rows in rows.values():
            page_rows.sort(
                key=lambda item: (
                    -int(item.get("confirmed_error_count") or 0),
                    -int(item.get("suspected_error_count") or 0),
                    -int(item.get("anomaly_count") or 0),
                    int(item.get("table_index") or 0),
                )
            )
        return rows

    def _samples(
        self,
        *,
        rows: Sequence[ContentCoverageReview],
        docx_by_id: Dict[str, DocxEvidenceUnit],
    ) -> List[Dict[str, Any]]:
        ordered = sorted(rows, key=lambda item: (0 if self._high_signal(item.text) else 1, item.review_id))
        buckets: Dict[str, List[ContentCoverageReview]] = {
            "text_artifact": [],
            "numeric_confusable": [],
            "numeric": [],
            "text": [],
            "other": [],
        }
        for review in ordered:
            buckets.setdefault(self._sample_category(review.text), []).append(review)
        samples: List[Dict[str, Any]] = []
        seen: set[str] = set()
        quotas = [
            ("text_artifact", min(12, self.max_samples)),
            ("numeric_confusable", min(14, self.max_samples)),
            ("numeric", max(8, self.max_samples // 3)),
            ("text", max(8, self.max_samples // 4)),
            ("other", self.max_samples),
        ]
        for category, quota in quotas:
            added = 0
            for review in buckets.get(category, []):
                if added >= quota or len(samples) >= self.max_samples:
                    break
                if self._append_sample(samples=samples, seen=seen, review=review, docx_by_id=docx_by_id):
                    added += 1
            if len(samples) >= self.max_samples:
                break
        if len(samples) < self.max_samples:
            for review in ordered:
                if self._append_sample(samples=samples, seen=seen, review=review, docx_by_id=docx_by_id):
                    if len(samples) >= self.max_samples:
                        break
        return samples

    def _append_sample(
        self,
        *,
        samples: List[Dict[str, Any]],
        seen: set[str],
        review: ContentCoverageReview,
        docx_by_id: Dict[str, DocxEvidenceUnit],
    ) -> bool:
        text = " ".join(str(review.text or "").split())[:80]
        if not text or text in seen:
            return False
        seen.add(text)
        unit = docx_by_id.get(review.unit_id)
        samples.append(
            {
                "coverage_review_id": review.review_id,
                "unit_id": review.unit_id,
                "text": text,
                "status": review.status,
                "sample_category": self._sample_category(text),
                "table_index": unit.table_index if unit else None,
                "row_index": unit.row_index if unit else None,
                "col_index": unit.col_index if unit else None,
                "row_text": self._row_context_text(unit=unit, docx_by_id=docx_by_id) if unit else "",
            }
        )
        return True

    def _row_context_text(self, *, unit: DocxEvidenceUnit, docx_by_id: Dict[str, DocxEvidenceUnit]) -> str:
        if unit.table_index in (None, "") or unit.row_index in (None, ""):
            return ""
        rows = [
            item
            for item in docx_by_id.values()
            if int(item.estimated_page_no or 0) == int(unit.estimated_page_no or 0)
            and int(item.table_index or 0) == int(unit.table_index or 0)
            and int(item.row_index or 0) == int(unit.row_index or 0)
            and str(item.text or "").strip()
        ]
        rows.sort(key=lambda item: (int(item.col_index or 9999), int(item.order_index or 0)))
        values = [" ".join(str(item.text or "").split())[:24] for item in rows[:10]]
        return " | ".join(value for value in values if value)[:180]

    def _sample_category(self, text: Any) -> str:
        compact = "".join(str(text or "").split())
        if not compact:
            return "other"
        if table_text_artifact_replacement(compact):
            return "text_artifact"
        if self._has_confusable_letter(compact):
            return "numeric_confusable"
        if any(ch.isdigit() for ch in compact):
            return "numeric"
        if any("\u4e00" <= ch <= "\u9fff" for ch in compact):
            return "text"
        return "other"

    def _needs_deep_qwen_review(self, *, page: Dict[str, Any], fast_review: TablePageVlReview | None) -> bool:
        samples = list(page.get("samples") or [])
        if not samples:
            return False
        suspicious_count = len((fast_review.suspicious_values if fast_review is not None else []) or [])
        has_text_artifact = any(table_text_artifact_replacement(sample.get("text")) for sample in samples)
        grid_context = list(page.get("table_grid_context") or [])
        grid_has_suspected_pattern = any(
            int(item.get("suspected_error_count") or 0) > 0
            or any(str(hint) in {"needs_decimal_column_sweep", "needs_table_column_pattern_review", "needs_qwen_vl_table_cell_review"} for hint in item.get("route_hints") or [])
            for item in grid_context
        )
        fast_review_enough = suspicious_count >= max(3, self.fast_ocr_min_values)
        if fast_review is not None and fast_review_enough and not has_text_artifact and not grid_has_suspected_pattern:
            return False
        if grid_has_suspected_pattern:
            return True
        if has_text_artifact:
            return True
        has_table_position = any(sample.get("table_index") not in (None, "") for sample in samples)
        unresolved_count = int(page.get("unresolved_count") or 0)
        confusable_count = sum(1 for sample in samples if self._sample_category(sample.get("text")) == "numeric_confusable")
        text_count = sum(1 for sample in samples if self._sample_category(sample.get("text")) in {"text_artifact", "text"})
        if fast_review is not None and suspicious_count > 0 and confusable_count >= 1:
            return True
        if fast_review is None and unresolved_count >= self.deep_min_unresolved:
            return True
        if not has_table_position:
            return False
        if fast_review is None and confusable_count >= 1:
            return True
        if unresolved_count >= 12 and (confusable_count >= 1 or text_count >= 3):
            return True
        if unresolved_count >= 20 and confusable_count >= 2 and suspicious_count < 3:
            return True
        return False

    def _is_paragraphized_table_fragment(self, *, unit: DocxEvidenceUnit, profile: Dict[str, Any]) -> bool:
        if unit.container_type == "table_cell":
            return True
        labels = set(profile.get("labels") or [])
        primary = str(profile.get("primary_route") or "")
        table_route = bool(
            profile.get("needs_table_parser")
            or profile.get("table_like")
            or "table_heavy" in labels
            or primary in {"image_table_cell_compare", "native_table_compare"}
        )
        if not table_route:
            return False
        return looks_like_paragraphized_table_fragment(unit.text)

    def _review(self, *, index: int, page: Dict[str, Any], image_path: Path, timeout: Optional[int] = None) -> TablePageVlReview:
        page_started_at = time.perf_counter()
        request_timeout = max(1, min(self.timeout, int(timeout or self.timeout)))
        prepared = self._prepare_review_image(
            index=index,
            page=page,
            page_no=int(page["page_no"]),
            image_path=image_path,
            timeout=request_timeout,
        )
        elapsed = int(time.perf_counter() - page_started_at)
        request_timeout = max(1, request_timeout - max(0, elapsed))
        result = self.client.structured_chat_with_image(
            image_path=prepared["path"],
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(page=page, prepared=prepared),
            schema=table_page_vl_schema(),
            timeout=request_timeout,
            num_predict=self.num_predict,
            temperature=0.0,
            allow_generate_fallback=True,
        )
        relative = self._relative_path(image_path)
        review_relative = self._relative_path(Path(prepared["path"]))
        if not getattr(result, "ok", False):
            ocr_suspicious_values = self._ocr_supported_suspicious_values(page=page, existing=[])
            if ocr_suspicious_values:
                return TablePageVlReview(
                    review_id=f"table_vl_{index:04d}",
                    page_no=int(page["page_no"]),
                    attempted=True,
                    available=True,
                    model=self.model,
                    verdict="table_risk",
                    confidence=0.72,
                    reason="Qwen-VL 表格页输出未形成可解析结论，已保留 PDF 页 OCR 支持的表格疑似值作为待复核证据。",
                    page_image_path=relative,
                    review_image_path=review_relative,
                    orientation_degrees=int(prepared.get("rotation_degrees") or 0),
                    orientation_confidence=float(prepared.get("orientation_confidence") or 0.0),
                    orientation_reason=str(prepared.get("orientation_reason") or ""),
                    visible_text_excerpt=self._ocr_excerpt(ocr_suspicious_values),
                    suspicious_values=ocr_suspicious_values,
                    docx_samples=list(page["samples"]),
                    next_route="needs_human_table_review",
                    error=str(getattr(result, "error", "") or "qwen_vl_failed"),
                    model_parse_error=str(getattr(result, "error", "") or ""),
                    model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), limit=600),
                    flags=[
                        "table_page_vl",
                        "table_page_vl_failed",
                        "table_page_ocr_supported_suspicious_values",
                        "table_page_vl_suspicious_values",
                    ],
                )
            return TablePageVlReview(
                review_id=f"table_vl_{index:04d}",
                page_no=int(page["page_no"]),
                attempted=True,
                available=False,
                model=self.model,
                verdict="unreadable",
                page_image_path=relative,
                review_image_path=review_relative,
                orientation_degrees=int(prepared.get("rotation_degrees") or 0),
                orientation_confidence=float(prepared.get("orientation_confidence") or 0.0),
                orientation_reason=str(prepared.get("orientation_reason") or ""),
                docx_samples=list(page["samples"]),
                reason="Qwen-VL 表格页视觉复核请求失败。",
                next_route="needs_human_table_review",
                error=str(getattr(result, "error", "") or "qwen_vl_failed"),
                model_parse_error=str(getattr(result, "error", "") or ""),
                model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), limit=600),
                flags=["table_page_vl_failed"],
            )
        raw_parsed = dict(getattr(result, "parsed", {}) or {})
        parsed, payload_normalized = self._normalized_review_payload(raw_parsed)
        verdict = self._choice(parsed.get("verdict"), {"table_risk", "no_obvious_issue", "mapping_uncertain", "unreadable"}, "unreadable")
        suspicious_values = self._suspicious_values(
            parsed.get("suspicious_values"),
            samples=page.get("samples") or [],
            table_grid_context=page.get("table_grid_context") or [],
        )
        ocr_suspicious_values = self._ocr_supported_suspicious_values(page=page, existing=suspicious_values)
        if ocr_suspicious_values:
            suspicious_values.extend(ocr_suspicious_values)
            if verdict in {"no_obvious_issue", "mapping_uncertain", "unreadable"}:
                verdict = "table_risk"
        filtered_model_values = False
        if verdict == "table_risk" and not suspicious_values and isinstance(parsed.get("suspicious_values"), list):
            filtered_model_values = bool(parsed.get("suspicious_values"))
            if filtered_model_values:
                verdict = "no_obvious_issue"
        flags = ["table_page_vl"]
        if int(prepared.get("rotation_degrees") or 0):
            flags.append("table_page_orientation_normalized")
        if str(prepared.get("review_image_type") or "") == "table_crop_sheet":
            flags.append("table_page_vl_crop_sheet")
        if payload_normalized:
            flags.append("table_page_vl_payload_normalized")
        if not parsed.get("verdict"):
            flags.append("table_page_vl_schema_incomplete")
        if ocr_suspicious_values:
            flags.append("table_page_ocr_supported_suspicious_values")
        if suspicious_values:
            flags.append("table_page_vl_suspicious_values")
        if filtered_model_values:
            flags.append("table_page_vl_values_filtered")
        reason = self._clip(parsed.get("reason"), limit=180)
        if filtered_model_values:
            reason = "模型返回的表格项与 DOCX 样例一致或缺少可定位差异，未形成可批注错误。"
        next_route = str(parsed.get("next_route") or ("needs_human_table_review" if verdict != "no_obvious_issue" else "")).strip()
        if verdict == "no_obvious_issue":
            next_route = ""
        return TablePageVlReview(
            review_id=f"table_vl_{index:04d}",
            page_no=int(page["page_no"]),
            attempted=True,
            available=True,
            model=self.model,
            verdict=verdict,
            confidence=max(self._confidence(parsed.get("confidence")), 0.72 if ocr_suspicious_values else 0.0),
            reason=reason,
            page_image_path=relative,
            review_image_path=review_relative,
            orientation_degrees=int(prepared.get("rotation_degrees") or 0),
            orientation_confidence=float(prepared.get("orientation_confidence") or 0.0),
            orientation_reason=str(prepared.get("orientation_reason") or ""),
            visible_text_excerpt=self._clip(parsed.get("visible_text_excerpt") or self._ocr_excerpt(ocr_suspicious_values), limit=420),
            suspicious_values=suspicious_values,
            docx_samples=list(page["samples"]),
            next_route=next_route,
            model_raw_content_excerpt=self._clip(getattr(result, "raw_content", ""), limit=600),
            model_parsed_keys=sorted(str(key) for key in raw_parsed.keys()),
            flags=flags,
        )

    def _normalized_review_payload(self, parsed: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        payload = dict(parsed or {})
        changed = False
        if self._looks_like_suspicious_value(payload):
            return (
                {
                    "verdict": "table_risk",
                    "visible_text_excerpt": self._clip(payload.get("visible_text"), limit=180),
                    "suspicious_values": [payload],
                    "reason": self._clip(payload.get("reason") or "模型返回了单条可疑表格值，已归一化为表格页复核结果。", limit=180),
                    "confidence": payload.get("confidence") or 0.78,
                    "next_route": "needs_human_table_review",
                },
                True,
            )
        values = payload.get("suspicious_values")
        if not isinstance(values, list):
            for key in ("suspicious", "values", "issues", "mismatches"):
                if isinstance(payload.get(key), list):
                    payload["suspicious_values"] = payload.get(key)
                    changed = True
                    break
        if isinstance(payload.get("suspicious_values"), list) and payload.get("suspicious_values") and not payload.get("verdict"):
            payload["verdict"] = "table_risk"
            changed = True
        if payload.get("verdict") == "table_risk":
            payload.setdefault("next_route", "needs_human_table_review")
            payload.setdefault("confidence", 0.78)
        if not payload.get("reason") and payload.get("verdict"):
            payload["reason"] = "模型返回字段不完整，已按可疑值和 verdict 做保守归一化。"
            changed = True
        return payload, changed

    def _looks_like_suspicious_value(self, value: Dict[str, Any]) -> bool:
        if not isinstance(value, dict):
            return False
        has_source = bool(
            str(value.get("docx_text") or value.get("text") or value.get("unit_id") or value.get("sample_id") or value.get("id") or "").strip()
        )
        return bool(has_source and str(value.get("visible_text") or "").strip())

    def _preflight(self) -> tuple[bool, str]:
        preflight = getattr(self.client, "preflight", None)
        if not callable(preflight):
            return True, ""
        result = preflight(timeout=min(30, self.timeout))
        if bool(result.get("available")):
            return True, ""
        return False, str(result.get("error") or result.get("reason") or "qwen_vl_preflight_failed")

    def _system_prompt(self) -> str:
        return (
            "你是 WPS PDF 转 DOCX 审查的表格页视觉复核模型。"
            "任务是读取这一页 PDF 图片，把给定 DOCX 表格样例逐项核对。"
            "只报告能从图片中清楚定位并确认的转换错误，重点是数字、金额、小数点、字母混入数字、表头、姓名/备注、漏格和行列错位。"
            "不要法律分析，不要猜图片外内容。看不清或无法定位就不要列入 suspicious_values。必须只输出 JSON。"
        )

    def _user_prompt(self, *, page: Dict[str, Any], prepared: Dict[str, Any]) -> str:
        samples = self._prompt_samples(page.get("samples") or [])
        payload = {
            "page_no": page["page_no"],
            "unresolved_table_cell_count": page["unresolved_count"],
            "pdf_table_signal": self._prompt_table_signal(page.get("pdf_table_signal") or {}),
            "table_grid_context": self._prompt_table_grid_context(page.get("table_grid_context") or []),
            "image_orientation": {
                "rotation_degrees_counterclockwise": int(prepared.get("rotation_degrees") or 0),
                "confidence": round(float(prepared.get("orientation_confidence") or 0.0), 4),
                "reason": str(prepared.get("orientation_reason") or "")[:160],
                "review_image_path": self._relative_path(Path(prepared["path"])),
                "review_image_type": str(prepared.get("review_image_type") or "full_page"),
                "review_image_note": str(prepared.get("review_image_note") or "")[:160],
            },
            "docx_samples": samples,
            "output_limits": {"max_suspicious_values": 10, "row_col_type": "string"},
            "rules": [
                "输入图已正向化；按图片可见内容判断。",
                "如果输入图是表格区域切片图，每个切片都是同一页 PDF 的局部放大区域，按切片内可见表格判断。",
                "visible_text_excerpt 写页面中能读出的关键表格文本摘要。",
                "suspicious_values 只放可定位、可逐字确认的样例，最多 10 条。",
                "数字/金额/小数点/字母混入数字优先；短中文和姓名只有清楚可读才报告。",
                "每条尽量回填 sample_id、unit_id、row、col；visible_text 填 PDF 可见值。",
                "table_grid_context 是程序根据整表/整列模式整理的待核查提示，不是最终证据；必须以图片可见内容为准。",
                "如果同列出现多个小数点缺失、字母数字混淆或表头污染提示，优先核对该列相邻单元。",
                "无法定位或不能确认替换文本时 verdict=mapping_uncertain，不要输出该样例。",
            ],
        }
        while len(json.dumps(payload, ensure_ascii=False)) > self.prompt_max_chars and len(payload["docx_samples"]) > 16:
            payload["docx_samples"] = payload["docx_samples"][:-4]
        while len(json.dumps(payload, ensure_ascii=False)) > self.prompt_max_chars and payload["table_grid_context"]:
            payload["table_grid_context"] = payload["table_grid_context"][:-1]
        return "复核这一页表格，只输出符合 schema 的 JSON。上下文：" + json.dumps(payload, ensure_ascii=False)

    def _prompt_table_grid_context(self, grids: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for grid in list(grids)[:4]:
            rows.append(
                {
                    "grid_id": self._clip(grid.get("grid_id"), limit=28),
                    "table_index": int(grid.get("table_index") or 0),
                    "status": self._clip(grid.get("status"), limit=40),
                    "shape": {
                        "rows": int(grid.get("rows") or 0),
                        "cols": int(grid.get("cols") or 0),
                        "cells": int(grid.get("cells") or 0),
                    },
                    "risk_counts": {
                        "unresolved": int(grid.get("unresolved_cell_count") or 0),
                        "confirmed": int(grid.get("confirmed_error_count") or 0),
                        "suspected": int(grid.get("suspected_error_count") or 0),
                        "anomalies": int(grid.get("anomaly_count") or 0),
                    },
                    "route_hints": [self._clip(value, limit=36) for value in list(grid.get("route_hints") or [])[:6]],
                    "column_profiles": list(grid.get("columns") or [])[:6],
                    "anomaly_cells": list(grid.get("anomaly_cells") or [])[:8],
                }
            )
        return rows

    def _prompt_table_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "table_region_count": int(signal.get("table_region_count") or 0),
            "line_band_count": int(signal.get("line_band_count") or 0),
            "table_like_region_count": int(signal.get("table_like_region_count") or 0),
            "anchor_ocr_line_count": int(signal.get("anchor_ocr_line_count") or 0),
            "anchor_ocr_page_line_count": int(signal.get("anchor_ocr_page_line_count") or 0),
            "digit_token_count": int(signal.get("digit_token_count") or 0),
            "text_chars": int(signal.get("text_chars") or 0),
            "title_line_text": self._clip(signal.get("title_line_text"), limit=60),
            "table_crop_candidate_count": len(signal.get("table_crop_bboxes") or []),
        }

    def _prompt_samples(self, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in list(samples)[: self.max_samples]:
            row = {
                "id": self._clip(item.get("coverage_review_id"), limit=28),
                "sample_id": self._clip(item.get("coverage_review_id"), limit=28),
                "unit_id": self._clip(item.get("unit_id"), limit=48),
                "text": self._clip(item.get("text"), limit=40),
                "cat": self._clip(item.get("sample_category"), limit=24),
                "row": self._clip(item.get("row_index"), limit=12),
                "col": self._clip(item.get("col_index"), limit=12),
                "row_text": self._clip(item.get("row_text"), limit=80),
            }
            status = self._clip(item.get("status"), limit=36)
            if status:
                row["status"] = status
            issue_hint = self._clip(item.get("issue_type_hint"), limit=32)
            if issue_hint:
                row["issue_hint"] = issue_hint
            candidate_text = self._clip(item.get("ocr_candidate_text"), limit=40)
            if candidate_text:
                row["ocr_candidate"] = candidate_text
            review_hint = self._clip(item.get("review_hint"), limit=96)
            if review_hint:
                row["review_hint"] = review_hint
            rows.append(row)
        return rows

    def _prepare_review_image(self, *, index: int, page: Dict[str, Any], page_no: int, image_path: Path, timeout: int) -> Dict[str, Any]:
        if not self.orientation_enabled:
            orientation = {"rotation_degrees": 0, "confidence": 0.0, "reason": "orientation_disabled"}
        else:
            orientation = self._heuristic_orientation(page=page, image_path=image_path)
            if orientation is None:
                orientation = self._detect_orientation(
                    index=index,
                    page_no=page_no,
                    image_path=image_path,
                    timeout=max(1, min(60, int(timeout or self.timeout))),
                )
        degrees = int(orientation.get("rotation_degrees") or 0)
        if degrees not in {0, 90, 180, 270}:
            degrees = 0
        output_path = Path(image_path)
        if degrees:
            output_path = self._rotated_image_path(page_no=page_no, degrees=degrees)
            if not output_path.exists():
                try:
                    with Image.open(image_path) as image:
                        image.convert("RGB").rotate(degrees, expand=True).save(output_path, quality=92)
                except Exception:
                    output_path = Path(image_path)
                    degrees = 0
        review_path = output_path
        review_type = "full_page"
        review_note = "使用正向化后的整页图片。"
        crop_bboxes = self._table_review_crop_bboxes(page=page, image_path=Path(image_path), rotation_degrees=degrees)
        if crop_bboxes:
            sheet_path = self._table_crop_sheet_path(page_no=page_no, degrees=degrees)
            if not sheet_path.exists():
                self._write_table_crop_sheet(image_path=output_path, bboxes=crop_bboxes, output_path=sheet_path)
            if sheet_path.exists():
                review_path = sheet_path
                review_type = "table_crop_sheet"
                review_note = f"使用 {len(crop_bboxes)} 个表格/文字密集区域切片放大图。"
        return {
            "path": review_path,
            "oriented_page_path": output_path,
            "rotation_degrees": degrees,
            "orientation_confidence": self._confidence(orientation.get("confidence")),
            "orientation_reason": self._clip(orientation.get("reason"), limit=180),
            "review_image_type": review_type,
            "review_image_note": review_note,
        }

    def _table_review_crop_bboxes(self, *, page: Dict[str, Any], image_path: Path, rotation_degrees: int) -> List[List[float]]:
        signal = dict(page.get("pdf_table_signal") or {})
        raw_items = list(signal.get("table_crop_bboxes") or [])
        if not raw_items:
            return []
        try:
            with Image.open(image_path) as image:
                width, height = int(image.width), int(image.height)
        except Exception:
            return []
        if width <= 0 or height <= 0:
            return []
        page_area = float(width * height)
        candidates: List[Dict[str, Any]] = []
        for item in raw_items:
            bbox = list(item.get("bbox") or [])
            if len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except Exception:
                continue
            x1, y1, x2, y2 = self._expanded_bbox([x1, y1, x2, y2], width=width, height=height, margin=36)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ratio = area / page_area if page_area else 0.0
            if ratio < 0.015 or ratio > 0.92:
                continue
            candidates.append({"bbox": [x1, y1, x2, y2], "ratio": ratio, "image_like": bool(item.get("image_like"))})
        if not candidates:
            return []
        preferred = [item for item in candidates if 0.04 <= float(item["ratio"]) <= 0.72]
        rows = preferred or candidates
        rows.sort(key=lambda item: (not bool(item.get("image_like")), -float(item.get("ratio") or 0.0)))
        selected = self._dedupe_overlapping_bboxes([item["bbox"] for item in rows], limit=4)
        return [
            self._transform_bbox_for_rotation(bbox=bbox, width=width, height=height, degrees=rotation_degrees)
            for bbox in selected
        ]

    def _expanded_bbox(self, bbox: Sequence[float], *, width: int, height: int, margin: int) -> List[float]:
        x1, y1, x2, y2 = [float(value) for value in list(bbox)[:4]]
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        return [
            max(0.0, left - margin),
            max(0.0, top - margin),
            min(float(width), right + margin),
            min(float(height), bottom + margin),
        ]

    def _dedupe_overlapping_bboxes(self, bboxes: Sequence[Sequence[float]], *, limit: int) -> List[List[float]]:
        selected: List[List[float]] = []
        for bbox in bboxes:
            current = [float(value) for value in list(bbox)[:4]]
            if any(self._bbox_iou(current, existing) >= 0.82 for existing in selected):
                continue
            selected.append(current)
            if len(selected) >= limit:
                break
        return selected

    def _bbox_iou(self, left: Sequence[float], right: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in list(left)[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in list(right)[:4]]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_left + area_right - intersection
        return intersection / union if union > 0 else 0.0

    def _transform_bbox_for_rotation(self, *, bbox: Sequence[float], width: int, height: int, degrees: int) -> List[float]:
        x1, y1, x2, y2 = [float(value) for value in list(bbox)[:4]]
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        normalized = int(degrees or 0) % 360
        transformed: List[tuple[float, float]] = []
        for x, y in points:
            if normalized == 90:
                transformed.append((y, width - x))
            elif normalized == 180:
                transformed.append((width - x, height - y))
            elif normalized == 270:
                transformed.append((height - y, x))
            else:
                transformed.append((x, y))
        xs = [point[0] for point in transformed]
        ys = [point[1] for point in transformed]
        return [max(0.0, min(xs)), max(0.0, min(ys)), max(xs), max(ys)]

    def _write_table_crop_sheet(self, *, image_path: Path, bboxes: Sequence[Sequence[float]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                tiles: List[tuple[str, Image.Image]] = []
                for index, bbox in enumerate(list(bboxes)[:4], start=1):
                    x1, y1, x2, y2 = [int(round(float(value))) for value in list(bbox)[:4]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(source.width, x2), min(source.height, y2)
                    if x2 - x1 < 80 or y2 - y1 < 24:
                        continue
                    crop = source.crop((x1, y1, x2, y2))
                    crop.thumbnail((1760, 720), Image.Resampling.LANCZOS)
                    tiles.append((f"区域{index}", crop))
                if not tiles:
                    return
                label_h = 34
                gap = 14
                width = max(900, max(tile.width for _, tile in tiles) + 40)
                height = 20 + sum(label_h + tile.height + gap for _, tile in tiles)
                sheet = Image.new("RGB", (width, height), "white")
                draw = ImageDraw.Draw(sheet)
                y = 12
                for label, tile in tiles:
                    draw.rectangle([0, y, width - 1, y + label_h - 1], fill=(245, 245, 245), outline=(180, 180, 180))
                    draw.text((14, y + 9), label, fill=(0, 0, 0))
                    y += label_h
                    x = (width - tile.width) // 2
                    sheet.paste(tile, (x, y))
                    y += tile.height + gap
                sheet.save(output_path, quality=94)
        except Exception:
            return

    def _heuristic_orientation(self, *, page: Dict[str, Any], image_path: Path) -> Optional[Dict[str, Any]]:
        signal = dict(page.get("pdf_table_signal") or {})
        title_y = signal.get("title_line_center_y")
        if title_y in (None, ""):
            return None
        try:
            with Image.open(image_path) as image:
                height = max(1, int(image.height or 1))
            ratio = float(title_y) / float(height)
        except Exception:
            return None
        if ratio >= 0.62:
            return {
                "rotation_degrees": 180,
                "confidence": 0.96,
                "reason": "PDF OCR 行位置显示表格标题位于页面下方，原图为 180 度倒置，已先旋转正向。",
            }
        if ratio <= 0.38:
            return {
                "rotation_degrees": 0,
                "confidence": 0.88,
                "reason": "PDF OCR 行位置显示表格标题位于页面上方，原图方向可直接阅读。",
            }
        return None

    def _detect_orientation(self, *, index: int, page_no: int, image_path: Path, timeout: int) -> Dict[str, Any]:
        sheet_path = self._orientation_sheet_path(page_no=page_no)
        if not sheet_path.exists():
            self._write_orientation_sheet(image_path=image_path, output_path=sheet_path)
        result = self.client.structured_chat_with_image(
            image_path=sheet_path,
            system_prompt=(
                "你是文档图片方向识别模型。你只判断哪一个旋转版本最适合阅读中文文档。"
                "rotation_degrees 表示需要把原始图逆时针旋转多少度才能正向阅读。必须只输出 JSON。"
            ),
            user_prompt=(
                "这张图是同一 PDF 页面四个方向的对照图：A=0 度，B=90 度，C=180 度，D=270 度。"
                "请忽略 A/B/C/D 标签本身，只看文档文字方向，选择文字最正、最适合后续表格审查的一格。"
                "输出字段固定为 rotation_degrees, readability, reason, confidence。"
            ),
            schema=table_page_orientation_schema(),
            timeout=max(1, min(self.timeout, int(timeout or self.timeout))),
            num_predict=160,
            temperature=0.0,
            allow_generate_fallback=False,
        )
        if not getattr(result, "ok", False):
            return {"rotation_degrees": 0, "confidence": 0.0, "reason": str(getattr(result, "error", "") or "orientation_failed")}
        parsed = dict(getattr(result, "parsed", {}) or {})
        degrees = self._choice(parsed.get("rotation_degrees"), {"0", "90", "180", "270"}, "0")
        return {
            "rotation_degrees": int(degrees),
            "confidence": self._confidence(parsed.get("confidence")),
            "reason": self._clip(parsed.get("reason"), limit=180),
        }

    def _write_orientation_sheet(self, *, image_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            variants = [(0, source), (90, source.rotate(90, expand=True)), (180, source.rotate(180, expand=True)), (270, source.rotate(270, expand=True))]
            cell_w, cell_h = 820, 620
            label_h = 34
            sheet = Image.new("RGB", (cell_w * 2, (cell_h + label_h) * 2), "white")
            draw = ImageDraw.Draw(sheet)
            labels = ["A", "B", "C", "D"]
            for idx, (degrees, image) in enumerate(variants):
                x = (idx % 2) * cell_w
                y = (idx // 2) * (cell_h + label_h)
                label = f"{labels[idx]} = {degrees}"
                draw.rectangle([x, y, x + cell_w - 1, y + label_h - 1], fill=(245, 245, 245), outline=(180, 180, 180))
                draw.text((x + 10, y + 9), label, fill=(0, 0, 0))
                fitted = image.copy()
                fitted.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                px = x + (cell_w - fitted.width) // 2
                py = y + label_h + (cell_h - fitted.height) // 2
                sheet.paste(fitted, (px, py))
            sheet.save(output_path, quality=88)

    def _orientation_sheet_path(self, *, page_no: int) -> Path:
        return self._image_dir() / f"page_{page_no:04d}_orientation_sheet.jpg"

    def _rotated_image_path(self, *, page_no: int, degrees: int) -> Path:
        return self._image_dir() / f"page_{page_no:04d}_rot{int(degrees):03d}.jpg"

    def _table_crop_sheet_path(self, *, page_no: int, degrees: int) -> Path:
        return self._image_dir() / f"page_{page_no:04d}_rot{int(degrees):03d}_table_crops.jpg"

    def _image_dir(self) -> Path:
        path = self.work_dir / "evidence" / "table_page_vl_images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _pdf_table_signals(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, Dict[str, Any]]:
        signals: Dict[int, Dict[str, Any]] = {}
        for unit in preflight_result.pdf_units:
            page_no = int(getattr(unit, "page_no", 0) or 0)
            if page_no <= 0:
                continue
            signal = signals.setdefault(
                page_no,
                {
                    "table_region_count": 0,
                    "line_band_count": 0,
                    "table_like_region_count": 0,
                    "anchor_ocr_line_count": 0,
                    "anchor_ocr_page_line_count": 0,
                    "digit_token_count": 0,
                    "text_chars": 0,
                    "title_line_center_y": None,
                    "title_line_text": "",
                    "title_line_width": 0.0,
                    "ocr_numeric_tokens": [],
                    "table_crop_bboxes": [],
                },
            )
            unit_type = str(getattr(unit, "unit_type", "") or "")
            region_type = str(getattr(unit, "region_type", "") or "")
            flags = set(getattr(unit, "flags", []) or [])
            text = str(getattr(unit, "text", "") or "")
            bbox = list(getattr(unit, "bbox", []) or [])
            if unit_type == "table_region" or region_type == "table":
                signal["table_region_count"] += 1
                self._append_table_crop_bbox(signal=signal, bbox=bbox, image_like=True)
            if region_type == "line_band":
                signal["line_band_count"] += 1
                self._append_table_crop_bbox(signal=signal, bbox=bbox, image_like=False)
            if "table_like" in flags:
                signal["table_like_region_count"] += 1
                self._append_table_crop_bbox(signal=signal, bbox=bbox, image_like=True)
            if unit_type == "visual_region" and region_type in {"large_visual_region", "text_band"}:
                self._append_table_crop_bbox(signal=signal, bbox=bbox, image_like=(region_type == "large_visual_region"))
            if unit_type == "anchor_ocr_line":
                signal["anchor_ocr_line_count"] += 1
            if unit_type == "anchor_ocr_page":
                signal["anchor_ocr_page_line_count"] += len([line for line in text.splitlines() if line.strip()])
            if self._looks_like_table_title(text) and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = [float(item) for item in bbox[:4]]
                    width = abs(x2 - x1)
                    if width >= float(signal.get("title_line_width") or 0.0):
                        signal["title_line_width"] = width
                        signal["title_line_center_y"] = (y1 + y2) / 2.0
                        signal["title_line_text"] = self._clip(text, limit=90)
                except Exception:
                    pass
            signal["digit_token_count"] += len(re.findall(r"\d+(?:[.,]\d+)?", text))
            self._extend_numeric_tokens(signal=signal, text=text)
            signal["text_chars"] += len("".join(text.split()))
        return signals

    def _append_table_crop_bbox(self, *, signal: Dict[str, Any], bbox: Sequence[Any], image_like: bool) -> None:
        if len(list(bbox or [])) < 4:
            return
        try:
            x1, y1, x2, y2 = [float(item) for item in list(bbox)[:4]]
        except Exception:
            return
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width < 160 or height < 24:
            return
        rows = signal.setdefault("table_crop_bboxes", [])
        candidate = {
            "bbox": [round(min(x1, x2), 2), round(min(y1, y2), 2), round(max(x1, x2), 2), round(max(y1, y2), 2)],
            "area": round(width * height, 2),
            "image_like": bool(image_like),
        }
        key = tuple(candidate["bbox"])
        if any(tuple(item.get("bbox", [])) == key for item in rows):
            return
        rows.append(candidate)
        rows.sort(key=lambda item: (not bool(item.get("image_like")), -float(item.get("area") or 0.0)))
        del rows[16:]

    def _looks_like_table_title(self, text: Any) -> bool:
        return looks_like_table_title(text)

    def _extend_numeric_tokens(self, *, signal: Dict[str, Any], text: Any) -> None:
        tokens = signal.setdefault("ocr_numeric_tokens", [])
        seen = set(tokens)
        for match in re.finditer(r"[A-Za-z]?\d[\dA-Za-z.,:：]{1,16}", str(text or "")):
            token = match.group(0).strip(".,:：")
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= 260:
                break

    def _pdf_supports_table_page(
        self,
        signal: Dict[str, Any],
        *,
        profile: Optional[Dict[str, Any]] = None,
        docx_row_count: int = 0,
    ) -> bool:
        if not signal:
            return False
        profile = dict(profile or {})
        labels = set(profile.get("labels") or [])
        primary_route = str(profile.get("primary_route") or "")
        canonical_page_type = str(profile.get("audit_canonical_page_type") or "")
        recognition_strategy = str(profile.get("recognition_strategy") or "")
        anchor_lines = int(signal.get("anchor_ocr_line_count") or 0) + int(signal.get("anchor_ocr_page_line_count") or 0)
        page_ocr_lines = int(signal.get("anchor_ocr_page_line_count") or 0)
        digit_tokens = int(signal.get("digit_token_count") or 0)
        table_regions = int(signal.get("table_region_count") or 0)
        line_bands = int(signal.get("line_band_count") or 0)
        table_like_regions = int(signal.get("table_like_region_count") or 0)
        profile_line_count = int(profile.get("horizontal_line_count") or 0) + int(profile.get("vertical_line_count") or 0)
        profile_table_like = bool(
            profile.get("table_like")
            or profile.get("needs_table_parser")
            or canonical_page_type == "table_image_page"
            or recognition_strategy == "table_structure_and_cell_ocr"
            or "table_heavy" in labels
            or primary_route in {"image_table_cell_compare", "native_table_compare"}
        )
        if table_regions > 0 and (page_ocr_lines >= 40 or digit_tokens >= 80):
            return True
        if table_regions > 0 and anchor_lines >= 35 and digit_tokens >= 50:
            return True
        if line_bands >= 3 and digit_tokens >= 60:
            return True
        if profile_table_like and docx_row_count >= 6:
            if table_regions > 0 or table_like_regions >= 1 or line_bands >= 1 or profile_line_count >= 8:
                return True
        if profile_table_like and docx_row_count >= 20 and digit_tokens >= 20 and int(signal.get("text_chars") or 0) >= 120:
            return True
        if profile_table_like and docx_row_count >= 20 and int(signal.get("text_chars") or 0) >= 180:
            return True
        return False

    def _relative_path(self, image_path: Path) -> str:
        try:
            return Path(image_path).relative_to(self.work_dir).as_posix()
        except Exception:
            return str(image_path)

    def _high_signal(self, text: Any) -> bool:
        compact = "".join(str(text or "").split())
        if any(ch.isdigit() for ch in compact):
            return len(compact) >= 3
        return len(compact) >= 4

    def _has_confusable_letter(self, text: Any) -> bool:
        compact = "".join(str(text or "").split()).upper()
        if len(compact) < 3 or not any(ch.isdigit() for ch in compact):
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in compact):
            return False
        return any(ch in set("BDEGILOQSZ") for ch in compact)

    def _ocr_supported_suspicious_values(self, *, page: Dict[str, Any], existing: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        signal = dict(page.get("pdf_table_signal") or {})
        tokens = [str(item or "").strip() for item in signal.get("ocr_numeric_tokens", []) if str(item or "").strip()]
        if not tokens:
            return []
        existing_keys = {(self._confusable_compact(item.get("docx_text")), self._confusable_compact(item.get("visible_text"))) for item in existing}
        rows: List[Dict[str, Any]] = []
        for sample in list(page.get("samples") or []):
            docx_text = self._clip(sample.get("text"), limit=80)
            if not docx_text:
                continue
            best = self._best_ocr_token_match(docx_text=docx_text, tokens=tokens)
            visible_text = str(best.get("visible_text") or "")
            if not visible_text:
                continue
            issue_type = str(best.get("issue_type") or "")
            if not issue_type:
                issue_type = "digit_letter_confusion" if self._has_confusable_letter(docx_text) else "value_mismatch"
            severity = str(best.get("severity") or "")
            if not severity:
                severity = "high" if issue_type in {"digit_letter_confusion", "decimal_point_missing"} else "medium"
            key = (self._confusable_compact(docx_text), self._confusable_compact(visible_text))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            rows.append(
                {
                    "sample_id": self._clip(sample.get("coverage_review_id"), limit=28),
                    "unit_id": self._clip(sample.get("unit_id"), limit=48),
                    "docx_text": docx_text,
                    "visible_text": visible_text,
                    "issue_type": issue_type,
                    "severity": severity,
                    "row": self._clip(sample.get("row_index"), limit=12),
                    "col": self._clip(sample.get("col_index"), limit=12),
                    "reason": "PDF 页 OCR 文本中存在与 DOCX 表格值高度相近的可见值，疑似数字/字母混淆或小数点丢失。",
                    "source": "pdf_page_ocr",
                }
            )
            if len(rows) >= 8:
                break
        return rows

    def _best_ocr_token_match(self, *, docx_text: str, tokens: Sequence[str]) -> Dict[str, str]:
        docx_key = self._confusable_compact(docx_text)
        if len(docx_key) < 3:
            return {}
        numeric_like = bool(re.fullmatch(r"\d{4,}", docx_key))
        confusable = self._has_confusable_letter(docx_text)
        docx_has_decimal = "." in docx_text or "。" in docx_text
        if not confusable and not numeric_like:
            return {}
        for token in tokens:
            token_key = self._confusable_compact(token)
            if len(token_key) < 3:
                continue
            if token_key == docx_key:
                token_has_decimal = "." in str(token) or "。" in str(token)
                if confusable or (numeric_like and token_has_decimal and not docx_has_decimal):
                    return {
                        "visible_text": str(token),
                        "issue_type": self._ocr_issue_type(docx_text=docx_text, visible_text=str(token)),
                        "severity": self._ocr_issue_severity(docx_text=docx_text, visible_text=str(token)),
                    }
                continue
        if not confusable:
            return {}
        best_token = ""
        best_distance = 99
        for token in tokens:
            token_key = self._confusable_compact(token)
            if token_key == docx_key:
                continue
            # Page OCR support is only a conservative shortcut. Do not turn a
            # near match into a suggested table value if it drops a non-zero
            # digit after letter/digit normalization.
            if abs(len(token_key) - len(docx_key)) > 1:
                continue
            if len(token_key) == len(docx_key):
                continue
            if len(token_key) != len(docx_key) and not self._single_zero_insertion_or_deletion(docx_key, token_key):
                continue
            distance = self._edit_distance_at_most_one(docx_key, token_key)
            if distance >= 0 and distance < best_distance and (confusable or "." in str(token) or "。" in str(token)):
                best_distance = distance
                best_token = str(token)
        if not best_token:
            return {}
        return {
            "visible_text": best_token,
            "issue_type": self._ocr_issue_type(docx_text=docx_text, visible_text=best_token),
            "severity": self._ocr_issue_severity(docx_text=docx_text, visible_text=best_token),
        }

    def _ocr_issue_type(self, *, docx_text: str, visible_text: str) -> str:
        docx_digits = re.sub(r"\D+", "", str(docx_text or ""))
        visible_digits = re.sub(r"\D+", "", str(visible_text or ""))
        if self._has_confusable_letter(docx_text):
            return "digit_letter_confusion"
        if docx_digits and docx_digits == visible_digits and len(docx_digits) >= 4:
            if "." not in str(docx_text or "") and "." in str(visible_text or ""):
                return "decimal_point_missing"
            if not re.search(r"[A-Za-z]", str(docx_text or "")) and re.search(r"[/:：]", str(docx_text or "")) and "." in str(visible_text or ""):
                return "decimal_or_punctuation_pollution"
        return "value_mismatch"

    def _ocr_issue_severity(self, *, docx_text: str, visible_text: str) -> str:
        issue_type = self._ocr_issue_type(docx_text=docx_text, visible_text=visible_text)
        if issue_type in {"digit_letter_confusion", "decimal_point_missing"}:
            return "high"
        if issue_type == "decimal_or_punctuation_pollution":
            return "medium"
        return "medium"

    def _confusable_compact(self, text: Any) -> str:
        mapping = str.maketrans(
            {
                "B": "8",
                "D": "0",
                "O": "0",
                "Q": "0",
                "C": "0",
                "G": "6",
                "E": "6",
                "I": "1",
                "L": "1",
                "S": "5",
                "Z": "2",
            }
        )
        value = str(text or "").upper().translate(mapping)
        return re.sub(r"[^0-9]", "", value)

    def _edit_distance_at_most_one(self, left: str, right: str) -> int:
        if left == right:
            return 0
        if abs(len(left) - len(right)) > 1:
            return -1
        if len(left) == len(right):
            return 1 if sum(1 for a, b in zip(left, right) if a != b) == 1 else -1
        short, long = (left, right) if len(left) < len(right) else (right, left)
        i = j = edits = 0
        while i < len(short) and j < len(long):
            if short[i] == long[j]:
                i += 1
                j += 1
            else:
                edits += 1
                if edits > 1:
                    return -1
                j += 1
        return 1

    def _single_zero_insertion_or_deletion(self, left: str, right: str) -> bool:
        if abs(len(left) - len(right)) != 1:
            return False
        short, long = (left, right) if len(left) < len(right) else (right, left)
        i = j = edits = 0
        while i < len(short) and j < len(long):
            if short[i] == long[j]:
                i += 1
                j += 1
                continue
            edits += 1
            if edits > 1 or long[j] != "0":
                return False
            j += 1
        if j < len(long):
            edits += 1
            if long[j] != "0":
                return False
        return edits == 1

    def _ocr_excerpt(self, values: Sequence[Dict[str, Any]]) -> str:
        visible = [str(item.get("visible_text") or "").strip() for item in values if str(item.get("visible_text") or "").strip()]
        if not visible:
            return ""
        return "PDF页OCR匹配值：" + "，".join(visible[:12])

    def _suspicious_values(
        self,
        value: Any,
        *,
        samples: Sequence[Dict[str, Any]] = (),
        table_grid_context: Sequence[Dict[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        rows = value if isinstance(value, list) else []
        sample_by_unit = {str(item.get("unit_id") or ""): item for item in samples if str(item.get("unit_id") or "")}
        sample_by_id = {
            key: item
            for item in samples
            for key in [str(item.get("coverage_review_id") or item.get("sample_id") or item.get("id") or "")]
            if key
        }
        samples_by_position: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for sample in samples:
            row_key = self._position_key(sample.get("row_index") or sample.get("row"))
            col_key = self._position_key(sample.get("col_index") or sample.get("col"))
            if row_key and col_key:
                samples_by_position.setdefault((row_key, col_key), []).append(sample)
        grid_by_unit: Dict[str, Dict[str, Any]] = {}
        grid_by_position: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for grid in table_grid_context:
            grid_id = self._clip(grid.get("grid_id"), limit=28)
            for anomaly in list(grid.get("anomaly_cells") or []):
                row = self._position_key(anomaly.get("row") or anomaly.get("row_index"))
                col = self._position_key(anomaly.get("col") or anomaly.get("col_index"))
                row_payload = dict(anomaly)
                if grid_id:
                    row_payload["grid_id"] = grid_id
                unit = str(row_payload.get("unit_id") or "")
                if unit:
                    grid_by_unit[unit] = row_payload
                if row and col:
                    grid_by_position.setdefault((row, col), []).append(row_payload)
        cleaned: List[Dict[str, Any]] = []
        for item in rows[:12]:
            if not isinstance(item, dict):
                continue
            unit_id = self._clip(item.get("unit_id"), limit=48)
            sample_id = self._clip(item.get("sample_id") or item.get("coverage_review_id") or item.get("id"), limit=28)
            sample = sample_by_id.get(sample_id) or sample_by_unit.get(unit_id) or {}
            anchor_status = "sample_id" if sample_by_id.get(sample_id) else "unit_id" if sample_by_unit.get(unit_id) else ""
            row_key = self._position_key(item.get("row"))
            col_key = self._position_key(item.get("col"))
            if not sample and row_key and col_key:
                positional = samples_by_position.get((row_key, col_key), [])
                if len(positional) == 1:
                    sample = positional[0]
                    anchor_status = "row_col_sample"
            grid_anchor: Dict[str, Any] = {}
            if unit_id and unit_id in grid_by_unit:
                grid_anchor = grid_by_unit[unit_id]
            elif row_key and col_key:
                positional_grid = grid_by_position.get((row_key, col_key), [])
                if len(positional_grid) == 1:
                    grid_anchor = positional_grid[0]
            if not sample and grid_anchor:
                sample = {
                    "coverage_review_id": grid_anchor.get("coverage_review_id"),
                    "unit_id": grid_anchor.get("unit_id"),
                    "text": grid_anchor.get("docx_text") or grid_anchor.get("text"),
                    "row_index": grid_anchor.get("row") or grid_anchor.get("row_index"),
                    "col_index": grid_anchor.get("col") or grid_anchor.get("col_index"),
                    "grid_id": grid_anchor.get("grid_id"),
                }
                anchor_status = "grid_anomaly"
            if not sample:
                continue
            resolved_sample_id = sample_id or self._clip(sample.get("coverage_review_id") or sample.get("sample_id") or sample.get("id"), limit=28)
            resolved_unit_id = self._clip(sample.get("unit_id") or unit_id, limit=48)
            docx_text = self._clip(item.get("docx_text") or item.get("text") or sample.get("text"), limit=80)
            visible_text = self._clip(item.get("visible_text"), limit=80)
            if not resolved_sample_id and not resolved_unit_id and not (row_key and col_key):
                continue
            source = self._clip(item.get("source") or "qwen_vl", limit=40)
            issue_type = self._choice(
                item.get("issue_type"),
                {
                    "digit_letter_confusion",
                    "decimal_point_missing",
                    "decimal_or_punctuation_pollution",
                    "value_mismatch",
                    "missing_or_extra_cell",
                    "unreadable",
                    "other",
                },
                "",
            )
            reason = self._clip(item.get("reason"), limit=140)
            if not docx_text or not visible_text:
                continue
            if self._compact(docx_text) == self._compact(visible_text):
                continue
            if self._label_to_plain_number(docx_text=docx_text, visible_text=visible_text):
                continue
            if not issue_type and self._direct_numeric_pair(docx_text=docx_text, visible_text=visible_text):
                inferred_issue = "digit_letter_confusion" if self._has_confusable_letter(docx_text) else "value_mismatch"
                if inferred_issue == "value_mismatch":
                    inferred_issue = self._ocr_issue_type(docx_text=docx_text, visible_text=visible_text)
                if self._safe_direct_numeric_replacement(
                    issue_type=inferred_issue,
                    docx_text=docx_text,
                    visible_text=visible_text,
                ):
                    issue_type = inferred_issue
            if not reason and source != "pdf_page_ocr" and issue_type and self._direct_numeric_pair(docx_text=docx_text, visible_text=visible_text):
                if self._safe_direct_numeric_replacement(
                    issue_type=issue_type,
                    docx_text=docx_text,
                    visible_text=visible_text,
                ):
                    reason = "模型返回可定位表格数字，数字/字符归一校验通过。"
            if not self._valid_model_suspicious_value(
                source=source,
                issue_type=issue_type,
                docx_text=docx_text,
                visible_text=visible_text,
                reason=reason,
            ):
                continue
            cleaned.append(
                {
                    "sample_id": resolved_sample_id,
                    "coverage_review_id": resolved_sample_id,
                    "unit_id": resolved_unit_id,
                    "docx_text": docx_text,
                    "visible_text": visible_text,
                    "issue_type": issue_type,
                    "severity": self._choice(item.get("severity"), {"high", "medium", "low"}, "medium"),
                    "row": self._clip(item.get("row") or sample.get("row_index") or sample.get("row"), limit=12),
                    "col": self._clip(item.get("col") or sample.get("col_index") or sample.get("col"), limit=12),
                    "reason": reason,
                    "source": source,
                    "anchor_status": anchor_status or "anchored",
                    "grid_id": self._clip(sample.get("grid_id") or grid_anchor.get("grid_id"), limit=28),
                }
            )
        return self._filter_repeated_direct_model_targets(cleaned)

    def _valid_model_suspicious_value(
        self,
        *,
        source: str,
        issue_type: str,
        docx_text: str,
        visible_text: str,
        reason: str,
    ) -> bool:
        if str(source or "") == "pdf_page_ocr":
            if issue_type not in {
                "digit_letter_confusion",
                "decimal_point_missing",
                "decimal_or_punctuation_pollution",
                "value_mismatch",
            }:
                return False
            if self._direct_numeric_pair(docx_text=docx_text, visible_text=visible_text):
                return self._safe_direct_numeric_replacement(
                    issue_type=issue_type,
                    docx_text=docx_text,
                    visible_text=visible_text,
                )
            return True
        if issue_type not in {
            "digit_letter_confusion",
            "decimal_point_missing",
            "decimal_or_punctuation_pollution",
            "value_mismatch",
            "missing_or_extra_cell",
        }:
            return False
        if not str(reason or "").strip():
            return False
        if self._noisy_direct_model_value(docx_text) or self._noisy_direct_model_value(visible_text):
            return False
        if self._direct_numeric_pair(docx_text=docx_text, visible_text=visible_text):
            return self._safe_direct_numeric_replacement(
                issue_type=issue_type,
                docx_text=docx_text,
                visible_text=visible_text,
            )
        return True

    def _noisy_direct_model_value(self, value: Any) -> bool:
        text = " ".join(str(value or "").split())
        compact = self._compact(text)
        if not compact:
            return True
        if len(text) > 80:
            return True
        if re.search(r"(\d)\1{7,}", compact):
            return True
        if re.search(r"([\u4e00-\u9fffA-Za-z])\1{5,}", compact):
            return True
        if compact.count("0") >= 10 and len(set(compact)) <= 4:
            return True
        if any(marker in text for marker in ("?", "？", "_", "看不清", "无法", "不确定", "疑似")):
            return True
        return False

    def _direct_numeric_pair(self, *, docx_text: str, visible_text: str) -> bool:
        value = f"{docx_text}{visible_text}"
        if re.search(r"[\u4e00-\u9fff]", value):
            return False
        return len(re.sub(r"\D+", "", value)) >= 2

    def _safe_direct_numeric_replacement(self, *, issue_type: str, docx_text: str, visible_text: str) -> bool:
        visible = str(visible_text or "").replace(",", "").replace("，", "").strip()
        if not re.fullmatch(r"[¥￥]?\d{1,8}(?:\.\d{1,4})?", visible):
            return False
        old_key = self._confusable_numeric_key(docx_text)
        new_key = self._confusable_numeric_key(visible_text)
        if len(old_key) < 2 or len(new_key) < 2:
            return False
        if old_key != new_key:
            return False
        if issue_type == "digit_letter_confusion":
            return bool(re.search(r"[A-Za-z]", str(docx_text or "")))
        return True

    def _confusable_numeric_key(self, value: Any) -> str:
        mapping = str.maketrans(
            {
                "B": "8",
                "b": "8",
                "D": "0",
                "d": "0",
                "O": "0",
                "o": "0",
                "Q": "0",
                "q": "0",
                "C": "0",
                "c": "0",
                "G": "6",
                "g": "6",
                "E": "6",
                "e": "6",
                "I": "1",
                "i": "1",
                "L": "1",
                "l": "1",
                "S": "5",
                "s": "5",
                "Z": "2",
                "z": "2",
            }
        )
        return re.sub(r"\D+", "", str(value or "").translate(mapping))

    def _filter_repeated_direct_model_targets(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if str(row.get("source") or "") == "pdf_page_ocr":
                continue
            visible = str(row.get("visible_text") or "")
            if not visible or self._direct_numeric_pair(docx_text=str(row.get("docx_text") or ""), visible_text=visible):
                continue
            key = self._compact(visible)
            if len(key) < 4:
                continue
            grouped.setdefault(key, []).append(row)
        noisy_targets = {key for key, values in grouped.items() if len(values) > 2}
        if not noisy_targets:
            return list(rows)
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            if str(row.get("source") or "") == "pdf_page_ocr":
                filtered.append(row)
                continue
            key = self._compact(row.get("visible_text"))
            if key in noisy_targets:
                continue
            filtered.append(row)
        return filtered

    def _position_key(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"\d+", text)
        return str(int(match.group(0))) if match else ""

    def _choice(self, value: Any, allowed: Sequence[str] | set[str], fallback: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else fallback

    def _confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _clip(self, value: Any, *, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _compact(self, value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "")).strip().lower()

    def _label_to_plain_number(self, *, docx_text: str, visible_text: str) -> bool:
        old_value = self._compact(docx_text)
        new_value = self._compact(visible_text)
        if not old_value or not new_value:
            return False
        if not any("\u4e00" <= ch <= "\u9fff" for ch in old_value):
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in new_value):
            return False
        normalized_number = new_value.replace(",", "").replace("，", "")
        return bool(re.fullmatch(r"[¥￥]?\d{1,8}(?:[./]\d{1,4})?", normalized_number))

    def _unload(self) -> None:
        unload = getattr(self.client, "unload", None)
        if callable(unload):
            unload()
