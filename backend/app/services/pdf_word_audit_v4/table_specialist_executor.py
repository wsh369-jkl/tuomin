from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings

from .common import GENERIC_CANONICAL_TABLE_HEADERS, GENERIC_TABLE_HEADER_MARKERS, looks_like_table_header, similarity, table_text_artifact_replacement
from .models import ConversionPreflightResult, DocxEvidenceUnit, SpecialistReviewResult, SpecialistReviewTask, TableCellEvidenceReview


class TableSpecialistExecutor:
    """Promote table specialist evidence into structured results.

    The executor is deliberately conservative: it does not treat page-level
    table risk as a commentable error unless it can anchor the issue to a DOCX
    table cell and produce an exact replacement value. Rule-only schema
    inference is kept as a candidate for report/model routing, not as a
    confirmed PDF conversion error.
    """

    TABLE_TASK_TYPES = {"table_cell_specialist_review", "table_visual_specialist_review"}
    COMMENT_POLICY = "comment_if_exact_replacement"
    CANONICAL_TABLE_HEADERS = GENERIC_CANONICAL_TABLE_HEADERS

    def __init__(self, *, enabled: Optional[bool] = None, max_results: Optional[int] = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_results = max(0, int(max_results if max_results is not None else getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_RESULTS", 9999) or 9999))
        self.max_per_page = max(
            1,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_RESULTS_PER_PAGE", 9999) or 9999),
        )

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[SpecialistReviewResult]:
        if not self.enabled or self.max_results <= 0:
            return []
        tasks = [task for task in preflight_result.specialist_review_tasks if task.task_type in self.TABLE_TASK_TYPES]
        if not tasks:
            return []
        self._table_cell_exact_count_by_page = Counter(
            int(review.page_no or 0)
            for review in preflight_result.table_cell_evidence_reviews
            if review.decision == "confirmed_error" and review.visible_text and review.docx_text and int(review.page_no or 0) > 0
        )
        self._task_by_page_type = {(int(task.page_no or 0), task.task_type): task for task in tasks}
        self._fallback_task_by_page = self._group_tasks_by_page(tasks)
        self._docx_by_page = self._table_docx_units_by_page(preflight_result)
        self._docx_by_unit = {unit.unit_id: unit for unit in preflight_result.docx_units}
        (
            self._table_page_vl_support_by_unit,
            self._table_page_vl_support_by_position,
        ) = self._build_table_page_vl_corroboration(preflight_result=preflight_result)
        self._page_text_qwen_support_by_unit = self._build_page_text_qwen_corroboration(
            preflight_result=preflight_result
        )
        self._coverage_by_unit = {review.unit_id: review for review in preflight_result.content_coverage_reviews if review.side == "docx"}
        self._coverage_by_id = {review.review_id: review for review in preflight_result.content_coverage_reviews if review.side == "docx"}
        self._used_units: set[str] = set()
        self._result_signatures: set[Tuple[int, str, str]] = set()
        self._grouped_exact_cell_results: Dict[Tuple[int, int, int, str, str], SpecialistReviewResult] = {}
        self._grouped_text_results: Dict[Tuple[int, int, int, str], SpecialistReviewResult] = {}
        self._completed_no_issue_task_ids: set[str] = set()
        results: List[SpecialistReviewResult] = []

        self._append_table_page_vl_results(preflight_result=preflight_result, results=results)
        self._append_table_cell_evidence_results(preflight_result=preflight_result, results=results)
        self._append_page_text_qwen_results(preflight_result=preflight_result, results=results)
        self._append_rule_based_table_cell_results(preflight_result=preflight_result, results=results)
        self._append_deferred_task_results(tasks=tasks, results=results)

        for index, result in enumerate(results[: self.max_results], start=1):
            result.result_id = f"table_specialist_{index:04d}"
        return results[: self.max_results]

    def _append_table_page_vl_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        for review in preflight_result.table_page_vl_reviews:
            page_no = int(review.page_no or 0)
            task = self._task_for(page_no=page_no, task_type="table_visual_specialist_review")
            if task is None:
                continue
            if review.available and review.verdict == "no_obvious_issue" and not review.suspicious_values:
                self._completed_no_issue_task_ids.add(task.task_id)
                continue
            if not review.available or not review.suspicious_values:
                self._append_deferred_result(
                    results=results,
                    task=task,
                    status="model_unavailable" if not review.available else "no_exact_table_replacement",
                    reason=review.reason or "表格页视觉复核没有给出可定位的精确替换值。",
                    evidence_ref={"source": "table_page_vl_review", "id": review.review_id, "raw": "table_page_vl_reviews.json"},
                    flags=review.flags + [f"table_vl_verdict={review.verdict}"],
                )
                continue
            for value in review.suspicious_values:
                if len(results) >= self.max_results:
                    return
                docx_text = self._clean_cell_text(value.get("docx_text"))
                visible_text = self._clean_cell_text(value.get("visible_text"))
                if not self._valid_exact_replacement(old_text=docx_text, new_text=visible_text):
                    continue
                if self._seen_result(page_no=page_no, old_text=docx_text, new_text=visible_text):
                    continue
                if not self._table_page_vl_value_has_anchor(value=value):
                    continue
                anchor = self._pick_docx_unit_for_table_vl_value(page_no=page_no, value=value, docx_text=docx_text)
                if anchor is None:
                    continue
                if not self._table_page_vl_value_commentable(value=value, anchor=anchor):
                    continue
                page_ocr_only = bool(str(value.get("source") or "") == "pdf_page_ocr" or review.model == "pdf_page_ocr" or not review.attempted)
                decision = "suspected_error" if page_ocr_only else "confirmed_error"
                comment_policy = "report_only_until_confirmed" if page_ocr_only else self.COMMENT_POLICY
                self._used_units.add(anchor.unit_id)
                results.append(
                    SpecialistReviewResult(
                        result_id="",
                        task_id=task.task_id,
                        task_type=task.task_type,
                        page_no=page_no,
                        status="page_ocr_candidate_needs_cell_confirmation" if page_ocr_only else "executed",
                        decision=decision,
                        confidence=min(0.68, float(review.confidence or 0.0)) if page_ocr_only else max(float(review.confidence or 0.0), 0.74),
                        reason=self._reason(
                            value.get("reason")
                            or review.reason
                            or "表格专项视觉复核确认 DOCX 表格单元与 PDF 可见值不一致。"
                        ),
                        issue_type=str(value.get("issue_type") or "table_value_mismatch"),
                        wps_unit_id=anchor.unit_id,
                        old_text=docx_text or anchor.text,
                        new_text=visible_text,
                        next_route="needs_table_cell_confirmation" if page_ocr_only else "",
                        model=review.model,
                        comment_policy=comment_policy,
                        evidence_refs=[
                            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
                            {"source": "table_page_vl_review", "id": review.review_id, "raw": "table_page_vl_reviews.json"},
                        ],
                        context=self._table_context(unit=anchor),
                        flags=list(review.flags)
                        + ["table_specialist", "exact_replacement"]
                        + (["page_ocr_candidate_only", "needs_table_cell_confirmation"] if page_ocr_only else []),
                    )
                )

    def _append_table_cell_evidence_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        rows = [
            review
            for review in preflight_result.table_cell_evidence_reviews
            if review.decision == "confirmed_error" and review.visible_text and review.docx_text
        ]
        rows.sort(key=lambda item: (int(item.page_no or 9999), int(item.table_index or 9999), int(item.row_index or 9999), int(item.col_index or 9999), item.review_id))
        per_page_count: Dict[int, int] = {}
        for review in rows:
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            page_limit = max(self.max_per_page, int(self._table_cell_exact_count_by_page.get(page_no, 0) or 0))
            if per_page_count.get(page_no, 0) >= page_limit:
                continue
            if review.docx_unit_id in self._used_units:
                continue
            unit = self._docx_by_unit.get(review.docx_unit_id)
            if unit is None:
                continue
            paragraphized_fragment = "paragraphized_table_fragment" in set(review.flags)
            if unit.container_type != "table_cell" and not paragraphized_fragment:
                continue
            old_text = self._clean_cell_text(review.docx_text or unit.text)
            new_text = self._clean_cell_text(review.visible_text)
            if not self._valid_exact_replacement(old_text=old_text, new_text=new_text):
                continue
            task = self._task_for(page_no=page_no, task_type="table_cell_specialist_review")
            if task is None:
                task = self._task_for(page_no=page_no, task_type="table_visual_specialist_review")
            if task is None:
                continue
            outcome = self._table_cell_evidence_outcome(
                task=task,
                review=review,
                unit=unit,
                old_text=old_text,
                new_text=new_text,
            )
            self._used_units.add(review.docx_unit_id)
            group_key = self._table_cell_evidence_group_key(page_no=page_no, review=review, unit=unit)
            if group_key is not None and group_key in self._grouped_exact_cell_results:
                self._merge_related_exact_table_cell_result(
                    result=self._grouped_exact_cell_results[group_key],
                    task=task,
                    review=review,
                    unit=unit,
                    old_text=old_text,
                    new_text=new_text,
                    outcome=outcome,
                )
                continue
            if len(results) >= self.max_results:
                return
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            context = self._table_context(unit=unit)
            if outcome["corroboration"]:
                context["corroboration"] = list(outcome["corroboration"])
            result = SpecialistReviewResult(
                result_id="",
                task_id=task.task_id,
                task_type="table_cell_specialist_review",
                page_no=page_no,
                status=str(outcome["status"]),
                decision=str(outcome["decision"]),
                confidence=float(outcome["confidence"]),
                reason=str(outcome["reason"]),
                issue_type=review.issue_type or "table_value_mismatch",
                wps_unit_id=review.docx_unit_id,
                old_text=old_text,
                new_text=new_text,
                next_route="",
                model=str(outcome["model"]),
                comment_policy=str(outcome["comment_policy"]),
                evidence_refs=list(outcome["evidence_refs"]),
                context=context,
                flags=list(review.flags)
                + ["table_specialist", "exact_replacement"]
                + (
                    ["table_cell_evidence_confirmed"]
                    if outcome["decision"] == "confirmed_error"
                    else ["table_cell_evidence_needs_corroboration", "report_only_until_confirmed"]
                )
                + list(outcome["flags"]),
            )
            if group_key is not None:
                self._grouped_exact_cell_results[group_key] = result
                result.context["group_summary"] = {
                    "kind": "same_table_column_exact_replacements",
                    "count": 1,
                    "page_no": page_no,
                    "table_index": int(unit.table_index or 0) or None,
                    "col_index": int(unit.col_index or 0) or None,
                    "issue_type": result.issue_type,
                    "evidence_source": review.evidence_source,
                }
            results.append(result)

    def _append_page_text_qwen_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        per_page_count: Dict[int, int] = {}
        for review in preflight_result.page_text_qwen_reviews:
            if len(results) >= self.max_results:
                return
            page_no = int(review.page_no or 0)
            if page_no <= 0 or not review.available or review.decision != "allow_exact_replacements":
                continue
            task = self._task_for(page_no=page_no, task_type="table_cell_specialist_review")
            if task is None:
                task = self._task_for(page_no=page_no, task_type="table_visual_specialist_review")
            if task is None:
                continue
            for value in review.suspicious_values:
                if len(results) >= self.max_results:
                    return
                if per_page_count.get(page_no, 0) >= self.max_per_page:
                    break
                unit_id = str(value.get("unit_id") or "").strip()
                unit = self._docx_by_unit.get(unit_id)
                if unit is None or unit.unit_id in self._used_units:
                    continue
                if not self._page_text_qwen_value_targets_table(unit=unit, value=value):
                    continue
                support_score = self._float(value.get("ocr_support_score"))
                if support_score < 0.9:
                    continue
                old_text = self._clean_cell_text(value.get("docx_text") or unit.text)
                new_text = self._clean_cell_text(value.get("pdf_text") or value.get("visible_text") or "")
                if not self._valid_exact_replacement(old_text=old_text, new_text=new_text):
                    continue
                if self._short_plain_number(old_text) and self._short_plain_number(new_text):
                    continue
                self._used_units.add(unit.unit_id)
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
                results.append(
                    SpecialistReviewResult(
                        result_id="",
                        task_id=task.task_id,
                        task_type="table_cell_specialist_review",
                        page_no=page_no,
                        status="executed",
                        decision="confirmed_error",
                        confidence=max(float(review.confidence or 0.0), support_score, 0.78),
                        reason=self._reason_for_page_text_qwen(review=review, value=value),
                        issue_type=str(value.get("issue_type") or "table_text_qwen_replacement"),
                        wps_unit_id=unit.unit_id,
                        old_text=old_text,
                        new_text=new_text,
                        next_route="",
                        model=review.model,
                        comment_policy=self.COMMENT_POLICY,
                        evidence_refs=[
                            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
                            {"source": "page_text_qwen_review", "id": review.review_id, "raw": "page_text_qwen_reviews.json"},
                        ],
                        context=self._table_context(unit=unit),
                        flags=list(review.flags)
                        + [
                            "table_specialist",
                            "page_text_qwen_confirmed",
                            "table_page_text_qwen_confirmed",
                            "exact_replacement",
                            f"ocr_support_score={support_score:.2f}",
                        ],
                    )
                )

    def _append_rule_based_table_cell_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        docx_by_unit = {unit.unit_id: unit for unit in preflight_result.docx_units}
        rows = [
            review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx"
            and review.unit_id not in self._used_units
            and review.status in {"covered", "covered_by_page_ocr", "covered_by_nearby_page_ocr", "table_pending", "mapping_uncertain", "diff_candidate", "uncovered_docx_content"}
        ]
        rows.sort(
            key=lambda review: (
                int(review.page_no or 9999),
                self._coverage_priority(review.status),
                review.review_id,
            )
        )
        per_page_count: Dict[int, int] = {}
        for review in rows:
            if len(results) >= self.max_results:
                return
            page_no = int(review.page_no or 0)
            task = self._task_for(page_no=page_no, task_type="table_cell_specialist_review")
            if task is None:
                continue
            if per_page_count.get(page_no, 0) >= self.max_per_page:
                continue
            unit = docx_by_unit.get(review.unit_id)
            if unit is None or unit.container_type != "table_cell":
                continue
            target, issue_type, confidence, reason = self._rule_replacement(review.text, unit=unit)
            if not self._valid_exact_replacement(old_text=review.text, new_text=target):
                continue
            if self._seen_result(page_no=page_no, old_text=review.text, new_text=target):
                continue
            group_key = self._table_text_group_key(page_no=page_no, unit=unit, issue_type=issue_type, target=target)
            if group_key is not None and group_key in self._grouped_text_results:
                grouped = self._grouped_text_results[group_key]
                self._merge_related_table_text_result(result=grouped, review=review, unit=unit, target=target)
                self._used_units.add(review.unit_id)
                continue
            self._used_units.add(review.unit_id)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            result = SpecialistReviewResult(
                result_id="",
                task_id=task.task_id,
                task_type=task.task_type,
                page_no=page_no,
                status="rule_candidate_needs_model_confirmation",
                decision="suspected_error",
                confidence=min(0.59, max(confidence, float(review.confidence or 0.0))),
                reason=self._rule_candidate_reason(reason),
                issue_type=issue_type,
                wps_unit_id=review.unit_id,
                old_text=self._clean_cell_text(review.text),
                new_text=target,
                next_route=task.next_route or "needs_table_parser",
                model="table_candidate_rules",
                comment_policy="report_only_until_confirmed",
                evidence_refs=[
                    {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
                    {"source": "content_coverage_review", "id": review.review_id, "raw": "full_content_coverage_reviews.json"},
                ],
                context=self._table_context(unit=unit),
                flags=list(review.flags)
                + [
                    "table_specialist",
                    f"coverage_status={review.status}",
                    "rule_candidate_only",
                    "needs_pdf_or_model_confirmation",
                ]
                + (["contextual_table_header_inference"] if issue_type == "table_header_schema_inference" else []),
            )
            if group_key is not None:
                self._grouped_text_results[group_key] = result
            results.append(result)

    def _append_deferred_task_results(
        self,
        *,
        tasks: Sequence[SpecialistReviewTask],
        results: List[SpecialistReviewResult],
    ) -> None:
        completed = {result.task_id for result in results if result.task_id}
        for task in tasks:
            if len(results) >= self.max_results:
                return
            if task.task_id in completed:
                continue
            if task.task_id in self._completed_no_issue_task_ids:
                continue
            if task.task_type not in self.TABLE_TASK_TYPES:
                continue
            self._append_deferred_result(
                results=results,
                task=task,
                status="deferred_no_exact_replacement",
                reason=task.reason or "表格专项暂无可定位的精确替换值，保持报告记录，不写批注。",
                evidence_ref={"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
                flags=task.flags,
            )

    def _append_deferred_result(
        self,
        *,
        results: List[SpecialistReviewResult],
        task: SpecialistReviewTask,
        status: str,
        reason: str,
        evidence_ref: Dict[str, Any],
        flags: Sequence[str],
    ) -> None:
        results.append(
            SpecialistReviewResult(
                result_id="",
                task_id=task.task_id,
                task_type=task.task_type,
                page_no=task.page_no,
                status=status,
                decision="defer",
                confidence=0.0,
                reason=self._reason(reason),
                next_route=task.next_route or "needs_human_table_review",
                model=task.model_hint,
                comment_policy="report_only_until_confirmed",
                evidence_refs=[dict(evidence_ref)],
                flags=list(flags) + ["table_specialist", "no_exact_replacement"],
            )
        )

    def _table_cell_evidence_refs(
        self,
        *,
        task: SpecialistReviewTask,
        review: TableCellEvidenceReview,
    ) -> List[Dict[str, Any]]:
        refs = [
            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
            {"source": "table_cell_evidence_review", "id": review.review_id, "raw": "table_cell_evidence_reviews.json"},
        ]
        if review.coverage_review_id:
            refs.append(
                {
                    "source": "content_coverage_review",
                    "id": review.coverage_review_id,
                    "raw": "full_content_coverage_reviews.json",
                }
            )
        return refs

    def _table_cell_evidence_outcome(
        self,
        *,
        task: SpecialistReviewTask,
        review: TableCellEvidenceReview,
        unit: DocxEvidenceUnit,
        old_text: str,
        new_text: str,
    ) -> Dict[str, Any]:
        base_reason = self._reason(review.reason or "表格单元证据确认 DOCX 表格值与 PDF 页 OCR 可见值不一致。")
        base_confidence = max(float(review.confidence or 0.0), 0.76)
        refs = self._table_cell_evidence_refs(task=task, review=review)
        corroboration = self._table_cell_exact_corroboration(
            review=review,
            unit=unit,
            old_text=old_text,
            new_text=new_text,
        )
        if corroboration["required"] and not corroboration["confirmed"]:
            return {
                "status": "needs_additional_corroboration",
                "decision": "suspected_error",
                "confidence": min(base_confidence, 0.73),
                "reason": self._reason(
                    f"{base_reason} 当前仅有页 OCR 的列/行上下文支持，缺少局部视觉或独立文本证据交叉确认，"
                    "先作为疑似错误保留，不进入最终确认批注。"
                ),
                "model": "table_cell_evidence_rules",
                "comment_policy": "report_only_until_confirmed",
                "evidence_refs": refs,
                "flags": [
                    "requires_cross_evidence_confirmation",
                    "context_only_decimal_not_final",
                ],
                "corroboration": [],
            }
        if corroboration["confirmed"]:
            for ref in corroboration["evidence_refs"]:
                if ref not in refs:
                    refs.append(ref)
            return {
                "status": "executed",
                "decision": "confirmed_error",
                "confidence": max(base_confidence, float(corroboration["confidence"] or 0.0)),
                "reason": self._reason(f"{base_reason} {corroboration['reason_suffix']}"),
                "model": "table_cell_evidence_rules",
                "comment_policy": self.COMMENT_POLICY,
                "evidence_refs": refs,
                "flags": list(corroboration["flags"]),
                "corroboration": list(corroboration["details"]),
            }
        return {
            "status": "executed",
            "decision": "confirmed_error",
            "confidence": base_confidence,
            "reason": base_reason,
            "model": "table_cell_evidence_rules",
            "comment_policy": self.COMMENT_POLICY,
            "evidence_refs": refs,
            "flags": [],
            "corroboration": [],
        }

    def _table_cell_exact_corroboration(
        self,
        *,
        review: TableCellEvidenceReview,
        unit: DocxEvidenceUnit,
        old_text: str,
        new_text: str,
    ) -> Dict[str, Any]:
        required = self._table_cell_evidence_requires_corroboration(review=review)
        if not required:
            return {
                "required": False,
                "confirmed": False,
                "confidence": 0.0,
                "reason_suffix": "",
                "flags": [],
                "evidence_refs": [],
                "details": [],
            }
        supports: List[Dict[str, Any]] = []
        for item in self._table_page_vl_support_candidates(review=review, unit=unit):
            if self._table_support_matches(item=item, old_text=old_text, new_text=new_text, review_issue_type=review.issue_type):
                supports.append(item)
        for item in self._page_text_qwen_support_candidates(unit=unit):
            if self._page_text_qwen_support_matches(item=item, old_text=old_text, new_text=new_text):
                supports.append(item)
        if not supports:
            return {
                "required": True,
                "confirmed": False,
                "confidence": 0.0,
                "reason_suffix": "",
                "flags": [],
                "evidence_refs": [],
                "details": [],
            }
        labels: List[str] = []
        refs: List[Dict[str, Any]] = []
        flags: List[str] = []
        details: List[Dict[str, Any]] = []
        best_confidence = 0.0
        for item in supports:
            label = str(item.get("label") or "")
            if label and label not in labels:
                labels.append(label)
            ref = {
                "source": str(item.get("source") or ""),
                "id": str(item.get("id") or ""),
                "raw": str(item.get("raw") or ""),
            }
            for key in ("page_image_path", "review_image_path", "crop_path", "image_path"):
                value = str(item.get(key) or "").strip()
                if value:
                    ref[key] = value
            if ref not in refs:
                refs.append(ref)
            for flag in list(item.get("flags") or []):
                if flag not in flags:
                    flags.append(flag)
            details.append(
                {
                    "source": str(item.get("source") or ""),
                    "label": label,
                    "confidence": round(float(item.get("confidence") or 0.0), 4),
                    "issue_type": str(item.get("issue_type") or ""),
                    "visible_text": self._clean_cell_text(item.get("visible_text")),
                }
            )
            best_confidence = max(best_confidence, float(item.get("confidence") or 0.0))
        label_text = "、".join(labels) if labels else "其他独立证据"
        return {
            "required": True,
            "confirmed": True,
            "confidence": max(best_confidence, 0.8),
            "reason_suffix": f"已由{label_text}交叉支持，可作为最终确认错误。",
            "flags": flags,
            "evidence_refs": refs,
            "details": details,
        }

    def _table_cell_evidence_requires_corroboration(self, *, review: TableCellEvidenceReview) -> bool:
        if str(review.issue_type or "") not in {"decimal_point_missing", "decimal_or_punctuation_pollution"}:
            return False
        flags = set(review.flags or [])
        return bool(
            review.status == "pdf_ocr_context_confirmed_mismatch"
            or "table_context_confirmed_decimal_missing" in flags
        )

    def _build_table_page_vl_corroboration(
        self,
        *,
        preflight_result: ConversionPreflightResult,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[int, str, str], List[Dict[str, Any]]]]:
        by_unit: Dict[str, List[Dict[str, Any]]] = {}
        by_position: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
        for review in preflight_result.table_page_vl_reviews:
            if not review.available or not review.attempted or review.model == "pdf_page_ocr":
                continue
            for value in review.suspicious_values:
                if str(value.get("source") or "") == "pdf_page_ocr":
                    continue
                item = {
                    "source": "table_page_vl_review",
                    "id": review.review_id,
                    "raw": "table_page_vl_reviews.json",
                    "label": "表格页视觉复核",
                    "confidence": max(float(review.confidence or 0.0), 0.78),
                    "issue_type": str(value.get("issue_type") or ""),
                    "docx_text": self._clean_cell_text(value.get("docx_text")),
                    "visible_text": self._clean_cell_text(value.get("visible_text")),
                    "review_image_path": getattr(review, "review_image_path", ""),
                    "page_image_path": getattr(review, "page_image_path", ""),
                    "flags": ["corroborated_by_table_page_vl"],
                }
                unit_id = str(value.get("unit_id") or value.get("docx_unit_id") or "").strip()
                if unit_id:
                    by_unit.setdefault(unit_id, []).append(item)
                row_key = self._table_position_key(value.get("row") or value.get("row_index"))
                col_key = self._table_position_key(value.get("col") or value.get("col_index"))
                if int(review.page_no or 0) > 0 and row_key and col_key:
                    by_position.setdefault((int(review.page_no or 0), row_key, col_key), []).append(item)
        return by_unit, by_position

    def _build_page_text_qwen_corroboration(
        self,
        *,
        preflight_result: ConversionPreflightResult,
    ) -> Dict[str, List[Dict[str, Any]]]:
        rows: Dict[str, List[Dict[str, Any]]] = {}
        for review in preflight_result.page_text_qwen_reviews:
            if not review.available or review.decision != "allow_exact_replacements":
                continue
            for value in review.suspicious_values:
                if str(value.get("page_kind") or "") != "table_page":
                    continue
                unit_id = str(value.get("unit_id") or "").strip()
                if not unit_id:
                    continue
                support_score = self._float(value.get("ocr_support_score"))
                if support_score < 0.9:
                    continue
                rows.setdefault(unit_id, []).append(
                    {
                        "source": "page_text_qwen_review",
                        "id": review.review_id,
                        "raw": "page_text_qwen_reviews.json",
                        "label": "页级 OCR 文本复核",
                        "confidence": max(float(review.confidence or 0.0), support_score),
                        "issue_type": str(value.get("issue_type") or ""),
                        "docx_text": self._clean_cell_text(value.get("docx_text")),
                        "visible_text": self._clean_cell_text(value.get("pdf_text") or value.get("visible_text")),
                        "flags": ["corroborated_by_page_text_qwen"],
                    }
                )
        return rows

    def _table_page_vl_support_candidates(
        self,
        *,
        review: TableCellEvidenceReview,
        unit: DocxEvidenceUnit,
    ) -> List[Dict[str, Any]]:
        rows = list(self._table_page_vl_support_by_unit.get(review.docx_unit_id, []))
        position_key = self._table_support_position_key(review=review, unit=unit)
        if position_key is not None:
            for item in self._table_page_vl_support_by_position.get(position_key, []):
                if item not in rows:
                    rows.append(item)
        return rows

    def _page_text_qwen_support_candidates(self, *, unit: DocxEvidenceUnit) -> List[Dict[str, Any]]:
        return list(self._page_text_qwen_support_by_unit.get(unit.unit_id, []))

    def _table_support_position_key(
        self,
        *,
        review: TableCellEvidenceReview,
        unit: DocxEvidenceUnit,
    ) -> Optional[Tuple[int, str, str]]:
        page_no = int(review.page_no or unit.estimated_page_no or 0)
        row_key = self._table_position_key(unit.row_index or review.row_index)
        col_key = self._table_position_key(unit.col_index or review.col_index)
        if page_no <= 0 or not row_key or not col_key:
            return None
        return page_no, row_key, col_key

    def _table_support_matches(
        self,
        *,
        item: Dict[str, Any],
        old_text: str,
        new_text: str,
        review_issue_type: str,
    ) -> bool:
        candidate_old = self._clean_cell_text(item.get("docx_text"))
        candidate_new = self._clean_cell_text(item.get("visible_text"))
        if not candidate_new:
            return False
        if candidate_new != self._clean_cell_text(new_text):
            return False
        if candidate_old and not self._old_text_matches_for_support(old_text=old_text, candidate_old=candidate_old):
            return False
        issue_type = str(item.get("issue_type") or "")
        if issue_type and issue_type not in {review_issue_type, "value_mismatch"}:
            return False
        return self._safe_direct_table_vl_numeric_replacement(
            issue_type=review_issue_type,
            old_text=old_text,
            new_text=new_text,
        )

    def _page_text_qwen_support_matches(
        self,
        *,
        item: Dict[str, Any],
        old_text: str,
        new_text: str,
    ) -> bool:
        candidate_old = self._clean_cell_text(item.get("docx_text"))
        candidate_new = self._clean_cell_text(item.get("visible_text"))
        if not candidate_new or candidate_new != self._clean_cell_text(new_text):
            return False
        if candidate_old and not self._old_text_matches_for_support(old_text=old_text, candidate_old=candidate_old):
            return False
        return self._valid_exact_replacement(old_text=old_text, new_text=new_text)

    def _old_text_matches_for_support(self, *, old_text: str, candidate_old: str) -> bool:
        old_value = self._clean_cell_text(old_text)
        candidate_value = self._clean_cell_text(candidate_old)
        if old_value == candidate_value:
            return True
        if self._numeric_like(old_value) or self._numeric_like(candidate_value):
            return self._confusable_numeric_key(old_value) == self._confusable_numeric_key(candidate_value)
        return self._compact_for_match(old_value) == self._compact_for_match(candidate_value)

    def _table_position_key(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"\d+", text)
        return str(int(match.group(0))) if match else ""

    def _rule_candidate_reason(self, reason: str) -> str:
        base = self._reason(reason)
        suffix = "该结果来自表格 schema/上下文规则，只能作为候选；需表格解析、PDF OCR 或 Qwen/VL 确认后才能作为必须修改项。"
        if base:
            return f"{base} {suffix}"[:220]
        return suffix

    def _reason_for_page_text_qwen(self, *, review: Any, value: Dict[str, Any]) -> str:
        base = self._reason(
            value.get("reason")
            or getattr(review, "reason", "")
            or "页级 OCR 文本 Qwen 复核确认表格单元与 PDF OCR 可见内容不一致。"
        )
        support = self._float(value.get("ocr_support_score"))
        if support > 0:
            return f"{base} OCR支持分 {support:.2f}。"[:180]
        return base

    def _task_for(self, *, page_no: int, task_type: str) -> Optional[SpecialistReviewTask]:
        task = self._task_by_page_type.get((int(page_no or 0), task_type))
        if task is not None:
            return task
        tasks = self._fallback_task_by_page.get(int(page_no or 0), [])
        for candidate in tasks:
            if candidate.task_type == task_type:
                return candidate
        return tasks[0] if tasks else None

    def _group_tasks_by_page(self, tasks: Sequence[SpecialistReviewTask]) -> Dict[int, List[SpecialistReviewTask]]:
        rows: Dict[int, List[SpecialistReviewTask]] = {}
        for task in tasks:
            page = int(task.page_no or 0)
            if page > 0:
                rows.setdefault(page, []).append(task)
        return rows

    def _table_docx_units_by_page(self, preflight_result: ConversionPreflightResult) -> Dict[int, List[DocxEvidenceUnit]]:
        rows: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in preflight_result.docx_units:
            if unit.container_type != "table_cell" and not self._looks_like_paragraphized_table_fragment(unit.text):
                continue
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            rows.setdefault(page_no, []).append(unit)
        for units in rows.values():
            units.sort(key=lambda unit: (int(unit.table_index or 9999), int(unit.row_index or 9999), int(unit.col_index or 9999), unit.order_index))
        return rows

    def _looks_like_paragraphized_table_fragment(self, text: Any) -> bool:
        value = self._clean_cell_text(text)
        if len(value) < 3 or len(value) > 40:
            return False
        if not any(ch.isdigit() for ch in value):
            return False
        return bool(
            re.search(r"[A-Za-z]", value)
            or re.search(r"\d{4,}", value)
            or re.search(r"\d+[/:：]\d+", value)
            or re.search(r"\d+\.\d+", value)
        )

    def _page_text_qwen_value_targets_table(self, *, unit: DocxEvidenceUnit, value: Dict[str, Any]) -> bool:
        if str(value.get("page_kind") or "") != "table_page":
            return False
        if unit.container_type == "table_cell":
            return True
        if str(value.get("container_type") or "") == "table_cell":
            return True
        return self._looks_like_paragraphized_table_fragment(unit.text)

    def _pick_docx_unit(self, *, page_no: int, docx_text: str) -> Optional[DocxEvidenceUnit]:
        wanted = self._compact_for_match(docx_text)
        candidates = [unit for unit in self._docx_by_page.get(int(page_no or 0), []) if unit.unit_id not in self._used_units]
        if wanted:
            for unit in candidates:
                current = self._compact_for_match(unit.text)
                if current and (current == wanted or wanted in current or current in wanted):
                    return unit
        return None

    def _pick_docx_unit_for_table_vl_value(
        self,
        *,
        page_no: int,
        value: Dict[str, Any],
        docx_text: str,
    ) -> Optional[DocxEvidenceUnit]:
        if not self._table_page_vl_value_has_anchor(value=value):
            return None
        for key in ("unit_id", "docx_unit_id"):
            unit_id = str(value.get(key) or "")
            unit = self._docx_by_unit.get(unit_id)
            if unit is not None and unit.unit_id not in self._used_units:
                return unit
        sample_id = str(value.get("sample_id") or value.get("coverage_review_id") or value.get("id") or "")
        coverage = self._coverage_by_id.get(sample_id)
        if coverage is not None:
            unit = self._docx_by_unit.get(coverage.unit_id)
            if unit is not None and unit.unit_id not in self._used_units:
                return unit
        row_key = self._table_position_key(value.get("row") or value.get("row_index"))
        col_key = self._table_position_key(value.get("col") or value.get("col_index"))
        if row_key and col_key:
            positional = [
                unit
                for unit in self._docx_by_page.get(int(page_no or 0), [])
                if unit.unit_id not in self._used_units
                and self._table_position_key(unit.row_index) == row_key
                and self._table_position_key(unit.col_index) == col_key
            ]
            if len(positional) == 1:
                return positional[0]
        if not any(str(value.get(key) or "").strip() for key in ("unit_id", "docx_unit_id", "coverage_review_id", "sample_id", "id")):
            return None
        return self._pick_docx_unit(page_no=page_no, docx_text=docx_text)

    def _table_page_vl_value_has_anchor(self, *, value: Dict[str, Any]) -> bool:
        if any(str(value.get(key) or "").strip() for key in ("unit_id", "docx_unit_id", "coverage_review_id", "sample_id", "id", "anchor_status", "grid_id")):
            return True
        return bool(self._table_position_key(value.get("row") or value.get("row_index")) and self._table_position_key(value.get("col") or value.get("col_index")))

    def _table_page_vl_value_commentable(self, *, value: Dict[str, Any], anchor: DocxEvidenceUnit) -> bool:
        source = str(value.get("source") or "").strip()
        issue_type = str(value.get("issue_type") or "").strip()
        if source == "pdf_page_ocr":
            old_text = self._clean_cell_text(value.get("docx_text") or anchor.text)
            new_text = self._clean_cell_text(value.get("visible_text"))
            if self._numeric_like(old_text) or self._numeric_like(new_text):
                return self._safe_direct_table_vl_numeric_replacement(
                    issue_type=issue_type,
                    old_text=old_text,
                    new_text=new_text,
                )
            return True
        if anchor.container_type != "table_cell":
            return False
        if not (int(anchor.table_index or 0) and int(anchor.row_index or 0) and int(anchor.col_index or 0)):
            return False
        if issue_type not in {
            "digit_letter_confusion",
            "value_mismatch",
            "missing_or_extra_cell",
            "table_value_mismatch",
            "decimal_point_missing",
            "decimal_or_punctuation_pollution",
        }:
            return False
        reason = self._clean_cell_text(value.get("reason"))
        if not reason:
            return False
        old_text = self._clean_cell_text(value.get("docx_text") or anchor.text)
        new_text = self._clean_cell_text(value.get("visible_text"))
        if self._has_repeated_noise_run(old_text) or self._has_repeated_noise_run(new_text):
            return False
        if self._looks_like_noisy_visible_text(new_text):
            return False
        if self._numeric_like(old_text) or self._numeric_like(new_text):
            return self._safe_direct_table_vl_numeric_replacement(
                issue_type=issue_type,
                old_text=old_text,
                new_text=new_text,
            )
        return True

    def _safe_direct_table_vl_numeric_replacement(self, *, issue_type: str, old_text: str, new_text: str) -> bool:
        visible = self._clean_cell_text(new_text).replace(",", "").replace("，", "")
        if not re.fullmatch(r"[¥￥]?\d{1,8}(?:\.\d{1,4})?", visible):
            return False
        old_key = self._confusable_numeric_key(old_text)
        new_key = self._confusable_numeric_key(new_text)
        if len(old_key) < 2 or len(new_key) < 2:
            return False
        if old_key != new_key:
            return False
        if issue_type == "digit_letter_confusion":
            return bool(re.search(r"[A-Za-z]", str(old_text or "")))
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

    def _table_context(self, *, unit: DocxEvidenceUnit) -> Dict[str, Any]:
        position = {
            "table_index": int(unit.table_index or 0) or None,
            "row_index": int(unit.row_index or 0) or None,
            "col_index": int(unit.col_index or 0) or None,
        }
        row_units = [
            item
            for item in self._docx_by_page.get(int(unit.estimated_page_no or 0), [])
            if int(item.table_index or 0) == int(unit.table_index or 0)
            and int(item.row_index or 0) == int(unit.row_index or 0)
        ]
        row_units.sort(key=lambda item: (int(item.col_index or 9999), item.order_index))
        row_context = [
            {
                "unit_id": item.unit_id,
                "col_index": int(item.col_index or 0) or None,
                "text": self._clip(item.text, 40),
            }
            for item in row_units
            if item.unit_id != unit.unit_id and self._clean_cell_text(item.text)
        ][:8]
        return {
            "table_position": position,
            "row_context": row_context,
            "related_units": [],
        }

    def _table_cell_evidence_group_key(
        self,
        *,
        page_no: int,
        review: TableCellEvidenceReview,
        unit: DocxEvidenceUnit,
    ) -> Optional[Tuple[int, int, int, str, str]]:
        if unit.container_type != "table_cell":
            return None
        issue_type = str(review.issue_type or "").strip()
        if issue_type not in {
            "decimal_point_missing",
            "decimal_or_punctuation_pollution",
            "digit_letter_confusion",
            "table_value_mismatch",
        }:
            return None
        table_index = int(unit.table_index or review.table_index or 0)
        col_index = int(unit.col_index or review.col_index or 0)
        if page_no <= 0 or table_index <= 0 or col_index <= 0:
            return None
        source = str(review.evidence_source or "pdf_page_ocr").strip() or "pdf_page_ocr"
        return int(page_no), table_index, col_index, issue_type, source

    def _merge_related_exact_table_cell_result(
        self,
        *,
        result: SpecialistReviewResult,
        task: SpecialistReviewTask,
        review: TableCellEvidenceReview,
        unit: DocxEvidenceUnit,
        old_text: str,
        new_text: str,
        outcome: Dict[str, Any],
    ) -> None:
        related = result.context.setdefault("related_units", [])
        related.append(
            {
                "unit_id": unit.unit_id,
                "old_text": old_text,
                "new_text": new_text,
                "table_index": int(unit.table_index or 0) or None,
                "row_index": int(unit.row_index or 0) or None,
                "col_index": int(unit.col_index or 0) or None,
                "review_id": review.review_id,
            }
        )
        summary = result.context.setdefault(
            "group_summary",
            {
                "kind": "same_table_column_exact_replacements",
                "count": 1,
                "page_no": int(result.page_no or 0) or None,
                "table_index": int(unit.table_index or 0) or None,
                "col_index": int(unit.col_index or 0) or None,
                "issue_type": result.issue_type,
                "evidence_source": review.evidence_source,
            },
        )
        summary["count"] = int(summary.get("count") or 1) + 1
        result.confidence = max(float(result.confidence or 0.0), max(float(review.confidence or 0.0), 0.76))
        for ref in self._table_cell_evidence_refs(task=task, review=review):
            if ref not in result.evidence_refs:
                result.evidence_refs.append(ref)
        for ref in list(outcome.get("evidence_refs") or []):
            if ref not in result.evidence_refs:
                result.evidence_refs.append(ref)
        for flag in ("grouped_table_column_exact_replacements", "same_column_table_cell_evidence"):
            if flag not in result.flags:
                result.flags.append(flag)
        for flag in list(outcome.get("flags") or []):
            if flag not in result.flags:
                result.flags.append(flag)
        if outcome.get("corroboration"):
            existing = result.context.setdefault("corroboration", [])
            for item in list(outcome.get("corroboration") or []):
                if item not in existing:
                    existing.append(item)
        if outcome.get("decision") == "confirmed_error" and result.decision != "confirmed_error":
            previous_primary = {
                "unit_id": result.wps_unit_id,
                "old_text": result.old_text,
                "new_text": result.new_text,
                "table_index": ((result.context or {}).get("table_position") or {}).get("table_index"),
                "row_index": ((result.context or {}).get("table_position") or {}).get("row_index"),
                "col_index": ((result.context or {}).get("table_position") or {}).get("col_index"),
            }
            related[:] = [item for item in related if str(item.get("unit_id") or "") != unit.unit_id]
            if previous_primary["unit_id"]:
                if not any(str(item.get("unit_id") or "") == previous_primary["unit_id"] for item in related):
                    related.insert(0, previous_primary)
            result.wps_unit_id = unit.unit_id
            result.old_text = old_text
            result.new_text = new_text
            refreshed_context = self._table_context(unit=unit)
            result.context["table_position"] = dict(refreshed_context.get("table_position") or {})
            result.context["row_context"] = list(refreshed_context.get("row_context") or [])
            result.decision = "confirmed_error"
            result.status = str(outcome.get("status") or "executed")
            result.comment_policy = self.COMMENT_POLICY
            result.confidence = max(float(result.confidence or 0.0), float(outcome.get("confidence") or 0.0))
            result.reason = str(outcome.get("reason") or result.reason)
            result.flags = [
                flag
                for flag in result.flags
                if flag not in {"table_cell_evidence_needs_corroboration", "report_only_until_confirmed"}
            ]
            if "table_cell_evidence_confirmed" not in result.flags:
                result.flags.append("table_cell_evidence_confirmed")
        if "同页同表同列还存在" not in result.reason:
            suffix = (
                "同页同表同列还存在同类确认错误，已合并为一个代表批注。"
                if result.decision == "confirmed_error"
                else "同页同表同列还存在同类疑似错误，已合并为一个代表项。"
            )
            result.reason = f"{result.reason} {suffix}"

    def _table_text_group_key(
        self,
        *,
        page_no: int,
        unit: DocxEvidenceUnit,
        issue_type: str,
        target: str,
    ) -> Optional[Tuple[int, int, int, str]]:
        if issue_type not in {"table_text_artifact", "table_header_schema_inference"}:
            return None
        table_index = int(unit.table_index or 0)
        row_index = int(unit.row_index or 0)
        target_key = self._compact_for_match(target)
        if page_no <= 0 or table_index <= 0 or row_index <= 0 or not target_key:
            return None
        return int(page_no), table_index, row_index, target_key

    def _merge_related_table_text_result(
        self,
        *,
        result: SpecialistReviewResult,
        review: Any,
        unit: DocxEvidenceUnit,
        target: str,
    ) -> None:
        related = result.context.setdefault("related_units", [])
        related.append(
            {
                "unit_id": unit.unit_id,
                "old_text": self._clean_cell_text(review.text),
                "new_text": target,
                "table_index": int(unit.table_index or 0) or None,
                "row_index": int(unit.row_index or 0) or None,
                "col_index": int(unit.col_index or 0) or None,
            }
        )
        result.evidence_refs.append(
            {"source": "related_content_coverage_review", "id": review.review_id, "raw": "full_content_coverage_reviews.json"}
        )
        if "grouped_table_text_artifact" not in result.flags:
            result.flags.append("grouped_table_text_artifact")
        if "同页同一表格行内还存在" not in result.reason:
            result.reason = f"{result.reason} 同页同一表格行内还存在同类文字错误，已合并为一个代表批注。"

    def _rule_replacement(self, text: str, *, unit: Optional[DocxEvidenceUnit] = None) -> Tuple[str, str, float, str]:
        numeric = self._numeric_confusable_hint(text)
        if numeric:
            return (
                numeric,
                "digit_letter_confusion",
                0.68,
                "表格专项规则确认数字类单元含易混字母，替换后形成稳定数字格式。",
            )
        if unit is not None:
            contextual = self._contextual_table_header_replacement(unit=unit, text=text)
            if contextual[0]:
                return contextual
        table_text = self._table_text_artifact_hint(text)
        if table_text and (unit is None or self._allow_table_text_artifact(unit=unit, target=table_text)):
            return (
                table_text,
                "table_text_artifact",
                0.62,
                "表格专项在表格行上下文中确认短表头/单元格存在 OCR 形近字或污染字符。",
            )
        return "", "", 0.0, ""

    def _allow_table_text_artifact(self, *, unit: DocxEvidenceUnit, target: str) -> bool:
        row_units = self._row_units(unit=unit)
        if self._looks_like_header_row(unit=unit, row_units=row_units):
            return True
        target_key = self._compact_for_match(target)
        if not target_key:
            return False
        same_target_count = 0
        for item in row_units:
            mapped = table_text_artifact_replacement(item.text)
            if mapped and self._compact_for_match(mapped) == target_key:
                same_target_count += 1
        return same_target_count >= 2

    def _contextual_table_header_replacement(
        self,
        *,
        unit: DocxEvidenceUnit,
        text: str,
    ) -> Tuple[str, str, float, str]:
        value = self._clean_cell_text(text)
        compact = self._compact_for_match(value)
        if unit.container_type != "table_cell" or not compact:
            return "", "", 0.0, ""
        if len(compact) < 2 or len(compact) > 14:
            return "", "", 0.0, ""
        if self._mostly_numeric_or_symbol(compact):
            return "", "", 0.0, ""
        row_units = self._row_units(unit=unit)
        if not self._looks_like_header_row(unit=unit, row_units=row_units):
            return "", "", 0.0, ""
        target, score, match_reason = self._infer_header_target(value)
        if not target:
            return "", "", 0.0, ""
        if self._compact_for_match(target) == compact:
            return "", "", 0.0, ""
        context_hits = self._row_header_context_hits(unit=unit, row_units=row_units)
        if context_hits < 2 and score < 0.82:
            return "", "", 0.0, ""
        confidence = min(0.74, max(0.58, 0.50 + score * 0.18 + min(context_hits, 4) * 0.025))
        return (
            target,
            "table_header_schema_inference",
            confidence,
            f"表格专项根据同一行表头结构和字段 schema 推断该短表头应为“{target}”；依据：{match_reason}。",
        )

    def _row_units(self, *, unit: DocxEvidenceUnit) -> List[DocxEvidenceUnit]:
        rows = [
            item
            for item in self._docx_by_page.get(int(unit.estimated_page_no or 0), [])
            if int(item.table_index or 0) == int(unit.table_index or 0)
            and int(item.row_index or 0) == int(unit.row_index or 0)
        ]
        rows.sort(key=lambda item: (int(item.col_index or 9999), item.order_index))
        return rows

    def _looks_like_header_row(self, *, unit: DocxEvidenceUnit, row_units: Sequence[DocxEvidenceUnit]) -> bool:
        if len(row_units) < 3:
            return False
        values = [self._clean_cell_text(item.text) for item in row_units if self._clean_cell_text(item.text)]
        if not values:
            return False
        short_count = sum(1 for value in values if 1 <= len(self._compact_for_match(value)) <= 14)
        if short_count < 3:
            return False
        context_hits = self._row_header_context_hits(unit=unit, row_units=row_units)
        if context_hits >= 2:
            return True
        row_index = int(unit.row_index or 0)
        if row_index > 5 or len(row_units) < 5:
            return False
        keyword_hits = 0
        for value in values:
            headerish, _, _ = self._looks_like_header_candidate_text(value)
            if headerish:
                keyword_hits += 1
        return keyword_hits >= 3 and short_count >= max(4, int(len(row_units) * 0.55))

    def _row_header_context_hits(self, *, unit: DocxEvidenceUnit, row_units: Sequence[DocxEvidenceUnit]) -> int:
        hits = 0
        seen_targets: set[str] = set()
        for item in row_units:
            if item.unit_id == unit.unit_id:
                continue
            headerish, target, _ = self._looks_like_header_candidate_text(item.text)
            if headerish:
                key = target or self._compact_for_match(item.text)
                if key not in seen_targets:
                    seen_targets.add(key)
                    hits += 1
        return hits

    def _looks_like_header_candidate_text(self, text: Any) -> Tuple[bool, str, float]:
        value = self._clean_cell_text(text)
        if not value:
            return False, "", 0.0
        mapped = table_text_artifact_replacement(value)
        if mapped:
            target, score, _ = self._infer_header_target(mapped)
            if target:
                return True, target, max(score, 0.9)
        target, score, _ = self._infer_header_target(value)
        if target and score >= 0.72:
            return True, target, score
        compact = self._compact_for_match(value)
        if looks_like_table_header(value) or any(keyword in compact for keyword in GENERIC_TABLE_HEADER_MARKERS):
            return True, target, score
        return False, "", 0.0

    def _infer_header_target(self, text: Any) -> Tuple[str, float, str]:
        value = self._clean_cell_text(text)
        if not value:
            return "", 0.0, ""
        key = self._header_match_key(value)
        if not key:
            return "", 0.0, ""
        if key in {"单元", "单元号"}:
            return "", 0.0, ""
        mapped = table_text_artifact_replacement(value)
        if mapped:
            return mapped, 0.84, "通用表头 schema 与表格行上下文匹配"
        if self._looks_like_unit_price_header(key):
            return "单价/m²", 0.9, "含单价/平方米单位模式"
        if key.endswith("姓名") and 2 <= len(key) <= 6:
            return "姓名", 0.82, "姓名类表头与通用姓名字段匹配"
        if key.endswith("面积") and 2 <= len(key) <= 6:
            return "面积", 0.82, "面积类表头与通用面积字段匹配"
        best_target = ""
        best_score = 0.0
        best_reason = ""
        for target in self.CANONICAL_TABLE_HEADERS:
            target_key = self._header_match_key(target)
            if key == target_key:
                return "", 1.0, ""
            score = max(similarity(key, target_key), self._edit_similarity(key, target_key))
            if self._one_edit_away(key, target_key):
                score = max(score, 0.84)
            if score > best_score:
                best_target = target
                best_score = score
                best_reason = f"与标准表头“{target}”形近，匹配分 {best_score:.2f}"
        if best_target and best_score >= 0.74 and self._header_category_compatible(key, self._header_match_key(best_target)):
            return best_target, best_score, best_reason
        if key.endswith("号") and len(key) <= 4 and ("楼" in key or "层" in key):
            return "楼层", 0.78, "楼层编号类表头与通用楼层字段匹配"
        if key.endswith("号") and len(key) <= 4 and "房" in key:
            return "房号", 0.78, "房间编号类表头与通用房号字段匹配"
        if len(key) <= 3 and key.endswith("注") and key != self._header_match_key("备注"):
            return "备注", 0.78, "备注类短表头与备注 schema 匹配"
        return "", 0.0, ""

    def _header_category_compatible(self, source_key: str, target_key: str) -> bool:
        if not source_key or not target_key:
            return False
        if target_key == "单价/m²":
            return self._looks_like_unit_price_header(source_key)
        source_markers = {marker for marker in GENERIC_TABLE_HEADER_MARKERS if marker in source_key}
        target_markers = {marker for marker in GENERIC_TABLE_HEADER_MARKERS if marker in target_key}
        if source_markers and target_markers:
            return bool(source_markers & target_markers)
        if target_key in {"备注", "摘要", "说明"}:
            return source_key.endswith(("注", "要", "明")) or any(token in source_key for token in ("备注", "摘要", "说明"))
        if target_key in {"楼层", "房号"}:
            return any(token in source_key for token in ("楼", "层", "房", "号"))
        if target_key == "姓名":
            return any(token in source_key for token in ("名", "姓", "人"))
        if target_key == "面积":
            return any(token in source_key for token in ("面", "积"))
        return True

    def _header_match_key(self, value: Any) -> str:
        text = self._compact_for_match(value)
        text = text.replace("／", "/").replace("㎡", "m²").replace("m2", "m²").replace("m²²", "m²")
        text = re.sub(r"^[!！|丨:：;；,.，。]+", "", text)
        text = re.sub(r"m[?？3iiln]+", "m²", text)
        text = text.replace("/m²²", "/m²").replace("im²", "/m²")
        return text

    def _looks_like_unit_price_header(self, key: str) -> bool:
        if "m²" not in key and "/m" not in key:
            return False
        if len(key) > 10:
            return False
        return key.startswith("单") or "价" in key or self._edit_similarity(key, "单价/m²") >= 0.72

    def _edit_similarity(self, left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        distance = self._levenshtein(left, right)
        return max(0.0, 1.0 - distance / max(len(left), len(right), 1))

    def _one_edit_away(self, left: str, right: str) -> bool:
        if abs(len(left) - len(right)) > 1:
            return False
        return self._levenshtein(left, right, max_distance=1) <= 1

    def _levenshtein(self, left: str, right: str, *, max_distance: Optional[int] = None) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for index, left_char in enumerate(left, start=1):
            current = [index]
            row_min = current[0]
            for right_index, right_char in enumerate(right, start=1):
                insert_cost = current[right_index - 1] + 1
                delete_cost = previous[right_index] + 1
                replace_cost = previous[right_index - 1] + (0 if left_char == right_char else 1)
                value = min(insert_cost, delete_cost, replace_cost)
                current.append(value)
                row_min = min(row_min, value)
            if max_distance is not None and row_min > max_distance:
                return max_distance + 1
            previous = current
        return previous[-1]

    def _mostly_numeric_or_symbol(self, text: str) -> bool:
        value = self._compact_for_match(text)
        if not value:
            return True
        chinese_count = sum("\u4e00" <= ch <= "\u9fff" for ch in value)
        digit_count = sum(ch.isdigit() for ch in value)
        return chinese_count == 0 and digit_count >= max(1, len(value) // 2)

    def _valid_exact_replacement(self, *, old_text: str, new_text: str) -> bool:
        old_value = self._clean_cell_text(old_text)
        new_value = self._clean_cell_text(new_text)
        if not old_value or not new_value:
            return False
        if old_value == new_value or self._compact_for_match(old_value) == self._compact_for_match(new_value):
            return False
        if len(new_value) > 90:
            return False
        if "?" in new_value or "？" in new_value or "_" in new_value:
            return False
        if any(marker in new_value for marker in ("看不清", "无法", "不确定", "参考", "仍需", "核对", "疑似")):
            return False
        if self._has_repeated_noise_run(old_value) or self._has_repeated_noise_run(new_value):
            return False
        if self._looks_like_noisy_visible_text(new_value):
            return False
        if self._label_to_plain_number(old_value=old_value, new_value=new_value):
            return False
        if self._numeric_like(old_value) or self._numeric_like(new_value):
            if not self._valid_numeric_replacement(old_value=old_value, new_value=new_value):
                return False
        if re.search(r"[A-Za-z]", old_value) and not (re.search(r"\d", new_value) or re.search(r"[A-Za-z]", new_value) or "²" in new_value):
            return False
        return True

    def _has_repeated_noise_run(self, value: str) -> bool:
        compact = self._compact_for_match(value)
        return bool(re.search(r"([0-9A-Za-z])\1{7,}", compact))

    def _label_to_plain_number(self, *, old_value: str, new_value: str) -> bool:
        old_key = self._compact_for_match(old_value)
        new_key = self._compact_for_match(new_value)
        if not old_key or not new_key:
            return False
        if not any("\u4e00" <= ch <= "\u9fff" for ch in old_key):
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in new_key):
            return False
        normalized_number = new_key.replace(",", "").replace("，", "")
        return bool(re.fullmatch(r"[¥￥]?\d{1,8}(?:[./]\d{1,4})?", normalized_number))

    def _numeric_like(self, value: str) -> bool:
        text = self._clean_cell_text(value)
        if not text:
            return False
        digit_count = sum(ch.isdigit() for ch in text)
        return bool(digit_count >= 2 and not any("\u4e00" <= ch <= "\u9fff" for ch in text))

    def _valid_numeric_replacement(self, *, old_value: str, new_value: str) -> bool:
        if re.search(r"[A-Za-z]", new_value):
            return False
        if not re.fullmatch(r"[¥￥]?\d{1,8}(?:\.\d{1,4})?", new_value.replace(",", "")):
            return False
        old_digits = re.sub(r"\D+", "", old_value)
        new_digits = re.sub(r"\D+", "", new_value)
        if len(new_digits) < 2:
            return False
        if old_digits and new_digits and abs(len(old_digits) - len(new_digits)) > 2:
            return False
        return True

    def _looks_like_noisy_visible_text(self, text: str) -> bool:
        value = "".join(str(text or "").split())
        if len(value) >= 16:
            ascii_ratio = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in value) / max(1, len(value))
            punct_ratio = sum(not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff") for ch in value) / max(1, len(value))
            if ascii_ratio > 0.25 or punct_ratio > 0.25:
                return True
        if re.search(r"[A-Za-z]{3,}", value) and re.search(r"[\u4e00-\u9fff]", value):
            return True
        return False

    def _numeric_confusable_hint(self, text: str) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) < 3 or not any(ch.isdigit() for ch in compact):
            return ""
        if any("\u4e00" <= ch <= "\u9fff" for ch in compact):
            return ""
        if not self._confusable_letters_are_fractional(compact):
            return ""
        replacements = str.maketrans(
            {
                "O": "0",
                "o": "0",
                "D": "0",
                "B": "8",
                "b": "8",
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
        hinted = compact.translate(replacements)
        if re.search(r"[A-Za-z]", hinted):
            return ""
        if hinted == compact:
            return ""
        if not re.search(r"\d", hinted):
            return ""
        return hinted

    def _confusable_letters_are_fractional(self, text: str) -> bool:
        compact = "".join(str(text or "").split())
        if not compact or not re.search(r"[A-Za-z]", compact):
            return False
        if "." not in compact:
            return False
        integer_part, fractional_part = compact.rsplit(".", 1)
        if re.search(r"[A-Za-z]", integer_part):
            return False
        return bool(re.search(r"[A-Za-z]", fractional_part))

    def _table_text_artifact_hint(self, text: str) -> str:
        return table_text_artifact_replacement(text)

    def _coverage_priority(self, status: str) -> int:
        return {
            "table_pending": 0,
            "diff_candidate": 1,
            "mapping_uncertain": 2,
            "covered_by_page_ocr": 3,
            "covered_by_nearby_page_ocr": 4,
            "covered": 5,
            "uncovered_docx_content": 6,
        }.get(str(status or ""), 9)

    def _seen_result(self, *, page_no: int, old_text: str, new_text: str) -> bool:
        key = (int(page_no or 0), self._compact_for_match(old_text), self._compact_for_match(new_text))
        if key in self._result_signatures:
            return True
        self._result_signatures.add(key)
        return False

    def _clean_cell_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _compact_for_match(self, value: Any) -> str:
        return "".join(str(value or "").split()).lower()

    def _short_plain_number(self, value: Any) -> bool:
        text = self._clean_cell_text(value).replace(",", "")
        return bool(re.fullmatch(r"\d{1,3}", text))

    def _float(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _clip(self, value: Any, limit: int) -> str:
        text = self._clean_cell_text(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _reason(self, value: Any) -> str:
        return self._clean_cell_text(value)[:180]
