from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings

from .models import ConversionDiffCandidate, ConversionPreflightResult, ReviewTask


class CoverageGapScheduler:
    """Turn unresolved v4 coverage gaps into explicit follow-up review tasks.

    This scheduler is intentionally lightweight. It does not run OCR or models;
    it only converts existing evidence gaps into deterministic tasks so the
    report can explain what remains unresolved and which engine should handle it
    next.
    """

    def __init__(self, *, enabled: Optional[bool] = None, max_tasks: Optional[int] = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_SCHEDULER_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_tasks = max(
            0,
            int(max_tasks if max_tasks is not None else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_SCHEDULER_MAX_TASKS", 9999) or 9999),
        )

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        if not self.enabled or self.max_tasks <= 0:
            return []
        tasks: List[ReviewTask] = []
        tasks.extend(self._high_risk_page_tasks(preflight_result=preflight_result))
        tasks.extend(self._page_text_tasks(preflight_result=preflight_result))
        tasks.extend(self._fragment_tasks(preflight_result=preflight_result))
        tasks.extend(self._mapping_and_recall_tasks(preflight_result=preflight_result))
        tasks.extend(self._table_grid_tasks(preflight_result=preflight_result))
        tasks.extend(self._image_page_tasks(preflight_result=preflight_result))
        tasks = self._dedupe(tasks)
        tasks.sort(key=lambda item: (-int(item.priority or 0), int(item.page_no or 999999), item.task_type))
        for index, task in enumerate(tasks[: self.max_tasks], start=1):
            task.task_id = f"coverage_task_{index:04d}"
        return tasks[: self.max_tasks]

    def summary(self, *, tasks: Sequence[ReviewTask]) -> Dict[str, Any]:
        type_counts = Counter(task.task_type for task in tasks)
        status_counts = Counter(task.status for task in tasks)
        engine_counts = Counter(task.next_engine for task in tasks if task.next_engine)
        planned_engine_counts = Counter(task.planned_engine for task in tasks if task.planned_engine)
        executor_counts = Counter(task.executor for task in tasks if task.executor)
        execution_outcome_counts = Counter(task.execution_outcome for task in tasks if task.execution_outcome)
        pages = sorted({int(task.page_no or 0) for task in tasks if int(task.page_no or 0) > 0})
        open_tasks = [task for task in tasks if task.requires_human_review()]
        resolved_tasks = [task for task in tasks if task.is_resolved()]
        open_pages = sorted({int(task.page_no or 0) for task in open_tasks if int(task.page_no or 0) > 0})
        resolved_pages = sorted({int(task.page_no or 0) for task in resolved_tasks if int(task.page_no or 0) > 0})
        return {
            "enabled": self.enabled,
            "version": "coverage_gap_scheduler_v1",
            "task_count": len(tasks),
            "page_count": len(pages),
            "pages": pages[:80],
            "open_task_count": len(open_tasks),
            "resolved_task_count": len(resolved_tasks),
            "open_page_count": len(open_pages),
            "resolved_page_count": len(resolved_pages),
            "open_pages": open_pages[:80],
            "resolved_pages": resolved_pages[:80],
            "high_priority_task_count": sum(1 for task in tasks if int(task.priority or 0) >= 85),
            "human_required_count": sum(1 for task in tasks if task.requires_human_review()),
            "type_counts": dict(sorted(type_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "next_engine_counts": dict(sorted(engine_counts.items())),
            "planned_engine_counts": dict(sorted(planned_engine_counts.items())),
            "executor_counts": dict(sorted(executor_counts.items())),
            "execution_outcome_counts": dict(sorted(execution_outcome_counts.items())),
        }

    def human_review_queue(self, *, tasks: Sequence[ReviewTask], limit: int = 120) -> List[Dict[str, Any]]:
        rows = [task for task in tasks if task.requires_human_review()]
        rows.sort(key=lambda item: (-int(item.priority or 0), int(item.page_no or 999999), item.task_type))
        return [task.to_dict() for task in rows[: max(0, int(limit or 0))]]

    def _high_risk_page_tasks(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        rows: List[ReviewTask] = []
        for review in preflight_result.high_risk_page_coverage_reviews:
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            if int(review.table_unresolved_count or 0) > 0 or review.next_route == "needs_table_parser":
                rows.append(
                    self._task(
                        task_type="table_row_cell_review",
                        page_no=page_no,
                        priority=max(int(review.priority or 0), 88),
                        reason=f"表格页仍有 {int(review.table_unresolved_count or 0)} 个未闭环单元/片段，需要行列和单元格专项复核。",
                        source_gap_id=review.review_id,
                        source_payload_ref="evidence/raw/high_risk_page_coverage_reviews.json",
                        next_engine="table_audit_engine",
                        coverage_review_ids=review.coverage_review_ids,
                        evidence_refs=[self._ref("high_risk_page_coverage", review.review_id)],
                        flags=["from_high_risk_page", "dominant=table"],
                    )
                )
            if int(review.mapping_uncertain_count or 0) > 0:
                rows.append(
                    self._task(
                        task_type="page_anchor_remap_review",
                        page_no=page_no,
                        priority=max(int(review.priority or 0), 86),
                        reason=f"页内仍有 {int(review.mapping_uncertain_count or 0)} 个映射不稳定项，需要重新做页锚点/片段映射。",
                        source_gap_id=review.review_id,
                        source_payload_ref="evidence/raw/high_risk_page_coverage_reviews.json",
                        next_engine="mapping_stabilizer",
                        coverage_review_ids=review.coverage_review_ids,
                        evidence_refs=[self._ref("high_risk_page_coverage", review.review_id)],
                        flags=["from_high_risk_page", "dominant=mapping"],
                    )
                )
            if int(review.visual_unresolved_count or 0) > 0 or int(review.needs_ocr_count or 0) > 0:
                rows.append(
                    self._task(
                        task_type="local_visual_crop_review",
                        page_no=page_no,
                        priority=max(int(review.priority or 0), 84),
                        reason=f"页内仍有 {int(review.visual_unresolved_count or 0)} 个视觉/OCR 未闭环项，需要局部截图复核。",
                        source_gap_id=review.review_id,
                        source_payload_ref="evidence/raw/high_risk_page_coverage_reviews.json",
                        next_engine="focused_crop_review",
                        coverage_review_ids=review.coverage_review_ids,
                        evidence_refs=[self._ref("high_risk_page_coverage", review.review_id)],
                        flags=["from_high_risk_page", "dominant=visual"],
                    )
                )
            if int(review.unresolved_count or 0) > 0 and not any(
                item.page_no == page_no and item.source_gap_id == review.review_id for item in rows
            ):
                rows.append(
                    self._task(
                        task_type="human_review_required",
                        page_no=page_no,
                        priority=int(review.priority or 70),
                        status="human_required",
                        reason=review.reason or "高风险页仍存在未闭环覆盖缺口，需要人工整页复核。",
                        source_gap_id=review.review_id,
                        source_payload_ref="evidence/raw/high_risk_page_coverage_reviews.json",
                        next_engine="human_review",
                        coverage_review_ids=review.coverage_review_ids,
                        evidence_refs=[self._ref("high_risk_page_coverage", review.review_id)],
                        flags=["from_high_risk_page", "fallback_only"],
                    )
                )
        return rows

    def _page_text_tasks(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        rows: List[ReviewTask] = []
        for profile in preflight_result.page_text_coverage_profiles:
            if profile.risk_level not in {"high", "medium"}:
                continue
            page_no = int(profile.page_no or 0)
            if page_no <= 0:
                continue
            task_type = "local_visual_crop_review" if profile.risk_level == "high" else "focused_qwen_vl_review"
            rows.append(
                self._task(
                    task_type=task_type,
                    page_no=page_no,
                    priority=82 if profile.risk_level == "high" else 68,
                    reason=profile.reason or "页级 PDF/DOCX 文本覆盖率偏低，需要按缺口样例做局部视觉复核。",
                    source_gap_id=f"page_text_coverage_{page_no}",
                    source_payload_ref="evidence/raw/page_text_coverage_profiles.json",
                    next_engine="focused_crop_review" if task_type == "local_visual_crop_review" else "qwen_vl_local_gate",
                    evidence_refs=[self._ref("page_text_coverage_profile", f"page_{page_no}")],
                    flags=[f"risk={profile.risk_level}", f"status={profile.status}"],
                )
            )
        return rows

    def _fragment_tasks(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        rows: List[ReviewTask] = []
        for review in preflight_result.fragment_anomaly_reviews:
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            rows.append(
                self._task(
                    task_type="docx_fragment_merge_review",
                    page_no=page_no,
                    priority=78,
                    reason=review.reason or "DOCX 存在碎片/重复片段异常，需要先合并片段再重新对齐。",
                    source_gap_id=review.review_id,
                    source_payload_ref="evidence/raw/fragment_anomaly_reviews.json",
                    target_unit_id=review.anchor_unit_id,
                    next_engine="docx_fragment_merger",
                    evidence_refs=[self._ref("fragment_anomaly", review.review_id)],
                    flags=list(review.flags),
                )
            )
        return rows

    def _mapping_and_recall_tasks(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        rows: List[ReviewTask] = []
        for diff in preflight_result.diff_candidates:
            page_no = int(diff.pdf_page_no or diff.docx_estimated_page_no or 0)
            if page_no <= 0:
                continue
            if diff.category == "mapping_uncertain":
                rows.append(self._diff_task(diff=diff, task_type="page_anchor_remap_review", next_engine="mapping_stabilizer", priority=76))
            if diff.category == "unlocated_hard_field" or "recall_guard" in set(diff.flags):
                rows.append(self._diff_task(diff=diff, task_type="critical_field_visual_review", next_engine="qwen_vl_local_gate", priority=84))
        return rows

    def _table_grid_tasks(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        rows: List[ReviewTask] = []
        for grid in preflight_result.table_grid_evidence:
            page_no = int(grid.page_no or 0)
            if page_no <= 0:
                continue
            unresolved_cell_count = int(grid.unresolved_cell_count or 0)
            has_open_anomalies = any(
                str((cell or {}).get("decision") or "") != "confirmed_error"
                for cell in list(grid.anomaly_cells or [])
            )
            if unresolved_cell_count <= 0 and not has_open_anomalies:
                continue
            rows.append(
                self._task(
                    task_type="table_pattern_cluster_review",
                    page_no=page_no,
                    priority=83 if unresolved_cell_count else 72,
                    reason=f"表格网格仍有 {unresolved_cell_count} 个未闭环单元和 {len(grid.anomaly_cells)} 个异常单元，需要列模式聚类复核。",
                    source_gap_id=grid.grid_id,
                    source_payload_ref="evidence/raw/table_grid_evidence.json",
                    next_engine="table_audit_engine",
                    evidence_refs=[self._ref("table_grid_evidence", grid.grid_id)],
                    flags=[f"table_index={grid.table_index}", f"status={grid.status}"],
                )
            )
        return rows

    def _image_page_tasks(self, *, preflight_result: ConversionPreflightResult) -> List[ReviewTask]:
        rows: List[ReviewTask] = []
        for review in preflight_result.image_page_reviews:
            if review.risk_level not in {"high", "medium"}:
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            rows.append(
                self._task(
                    task_type="local_visual_crop_review",
                    page_no=page_no,
                    priority=80 if review.risk_level == "high" else 66,
                    reason=review.reason or "图片型 PDF 页存在转换风险，需要局部视觉复核。",
                    source_gap_id=review.review_id,
                    source_payload_ref="evidence/raw/image_pdf_page_reviews.json",
                    target_unit_id=review.anchor_unit_id,
                    next_engine="focused_crop_review",
                    evidence_refs=[self._ref("image_pdf_page_review", review.review_id)],
                    flags=[f"risk={review.risk_level}", f"page_kind={review.page_kind}"],
                )
            )
        return rows

    def _diff_task(self, *, diff: ConversionDiffCandidate, task_type: str, next_engine: str, priority: int) -> ReviewTask:
        page_no = int(diff.pdf_page_no or diff.docx_estimated_page_no or 0)
        return self._task(
            task_type=task_type,
            page_no=page_no,
            priority=priority,
            reason=diff.reason or f"{diff.category} 需要进入 {task_type}。",
            source_gap_id=diff.diff_id,
            source_payload_ref="evidence/raw/conversion_diff_candidates.json",
            target_unit_id=diff.docx_unit_id,
            next_engine=next_engine,
            candidate_ids=[diff.diff_id],
            evidence_refs=[self._ref("conversion_diff_candidate", diff.diff_id)],
            flags=[f"category={diff.category}", *list(diff.flags)[:8]],
        )

    def _task(
        self,
        *,
        task_type: str,
        page_no: Optional[int],
        priority: int,
        reason: str,
        source_gap_id: str,
        source_payload_ref: str,
        next_engine: str,
        status: str = "planned",
        target_unit_id: str = "",
        coverage_review_ids: Sequence[str] = (),
        candidate_ids: Sequence[str] = (),
        evidence_refs: Sequence[Dict[str, Any]] = (),
        flags: Sequence[str] = (),
    ) -> ReviewTask:
        return ReviewTask(
            task_id="",
            task_type=task_type,
            page_no=page_no,
            priority=max(0, min(99, int(priority or 0))),
            status=status,
            reason=str(reason or "")[:500],
            source_gap_id=source_gap_id,
            source_payload_ref=source_payload_ref,
            target_unit_id=target_unit_id,
            next_engine=next_engine,
            planned_engine=next_engine,
            budget=self._budget(task_type=task_type),
            fallback="human_review_required",
            evidence_refs=[dict(item) for item in evidence_refs],
            candidate_ids=list(candidate_ids),
            coverage_review_ids=list(coverage_review_ids)[:9999],
            flags=list(flags),
        )

    def _budget(self, *, task_type: str) -> Dict[str, Any]:
        if task_type in {"table_row_cell_review", "table_pattern_cluster_review"}:
            return {"max_cells": 80, "max_crops": 40, "max_model_calls": 8}
        if task_type in {"local_visual_crop_review", "critical_field_visual_review", "focused_qwen_vl_review"}:
            return {"max_crops": 12, "max_model_calls": 4}
        if task_type in {"page_anchor_remap_review", "docx_fragment_merge_review"}:
            return {"max_units": 120, "max_windows": 40, "max_model_calls": 0}
        return {"max_model_calls": 0}

    def _ref(self, source: str, ref_id: str) -> Dict[str, Any]:
        return {"source": source, "id": ref_id}

    def _dedupe(self, tasks: Sequence[ReviewTask]) -> List[ReviewTask]:
        best: Dict[Tuple[str, int, str], ReviewTask] = {}
        for task in tasks:
            key = (task.task_type, int(task.page_no or 0), task.source_gap_id or "")
            existing = best.get(key)
            if existing is None or int(task.priority or 0) > int(existing.priority or 0):
                best[key] = task
        return list(best.values())
