from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings

from .models import ConversionPreflightResult, ReviewTask


class CoverageTaskExecutionBridge:
    """Finalize coverage review tasks into real end-state records.

    Coverage tasks are created after the main v4 automation chain has already
    run. Without a final execution bridge they remain stuck in "planned", even
    when the underlying gap was already auto-closed or when all relevant
    automated stages have already been exhausted and the only truthful next step
    is human review.
    """

    PAGE_SCOPE_SOURCE_REFS = {
        "evidence/raw/page_text_coverage_profiles.json",
        "evidence/raw/image_pdf_page_reviews.json",
    }
    TABLE_TASK_TYPES = {"table_row_cell_review", "table_pattern_cluster_review"}
    VISUAL_TASK_TYPES = {"local_visual_crop_review", "critical_field_visual_review", "focused_qwen_vl_review"}
    ALIGNMENT_TASK_TYPES = {"page_anchor_remap_review", "docx_fragment_merge_review"}
    CLOSED_STATUS_DETAILS = {
        "resolved_by_confirmed_review": (
            "specialist_review_chain",
            "resolved_by_confirmed_review",
            "对应覆盖缺口已被更严格的专项/视觉证据确认并闭环，不再进入人工队列。",
        ),
        "resolved_by_linked_review_pair": (
            "specialist_review_chain",
            "resolved_by_linked_review_pair",
            "对应覆盖缺口已沿同一对齐/差异链路被成对闭环，不再进入人工队列。",
        ),
        "resolved_by_table_page_vl_no_issue": (
            "table_page_vl_review",
            "resolved_by_table_page_vl_no_issue",
            "对应表格页样例已完成整页视觉复核且未发现问题，不再进入人工队列。",
        ),
        "resolved_by_page_text_qwen_no_issue": (
            "full_page_review_closure",
            "resolved_by_page_text_qwen_no_issue",
            "对应页级 OCR 文本整页复核已判定无明显问题，不再进入人工队列。",
        ),
        "resolved_by_image_text_vl_no_issue": (
            "full_page_review_closure",
            "resolved_by_image_text_vl_no_issue",
            "对应图片页整页视觉复核已判定无明显问题，不再进入人工队列。",
        ),
        "covered_by_table_page_text_support": (
            "table_heavy_page_closure",
            "resolved_by_table_heavy_page_text_support",
            "对应表格重灾页缺口已被页级 OCR/回填文本重新命中，不再进入人工队列。",
        ),
        "covered_by_nearby_table_page_text_support": (
            "table_heavy_page_closure",
            "resolved_by_nearby_table_page_text_support",
            "对应表格重灾页缺口已被邻页 OCR/回填文本支持并按页映射偏移闭环，不再进入人工队列。",
        ),
    }

    def __init__(self, *, enabled: Optional[bool] = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_TASK_EXECUTION_ENABLED", True)
            if enabled is None
            else enabled
        )

    def apply(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "version": "coverage_task_execution_v1",
            "task_count": len(preflight_result.coverage_review_tasks),
            "resolved_task_count": 0,
            "human_required_task_count": 0,
            "redundant_task_count": 0,
            "executor_counts": {},
            "execution_outcome_counts": {},
            "open_pages": [],
            "resolved_pages": [],
        }
        if not self.enabled or not preflight_result.coverage_review_tasks:
            return summary

        self._coverage_by_id = {
            review.review_id: review
            for review in preflight_result.content_coverage_reviews
            if review.review_id
        }
        self._open_coverage_ids = {
            review.review_id
            for review in preflight_result.content_coverage_reviews
            if review.review_id and review.decision != "covered"
        }
        self._coverage_ids_by_page: Dict[int, List[str]] = {}
        self._coverage_ids_by_diff: Dict[str, List[str]] = {}
        self._coverage_ids_by_unit: Dict[str, List[str]] = {}
        self._covered_statuses_by_page: Dict[int, Counter[str]] = {}
        for review in preflight_result.content_coverage_reviews:
            page_no = int(review.page_no or 0)
            if page_no > 0 and review.review_id:
                self._coverage_ids_by_page.setdefault(page_no, []).append(review.review_id)
            if review.related_diff_id and review.review_id:
                self._coverage_ids_by_diff.setdefault(str(review.related_diff_id), []).append(review.review_id)
            if review.unit_id and review.review_id:
                self._coverage_ids_by_unit.setdefault(str(review.unit_id), []).append(review.review_id)
            if review.decision == "covered" and page_no > 0:
                self._covered_statuses_by_page.setdefault(page_no, Counter())[str(review.status or "covered")] += 1

        self._high_risk_by_id = {
            item.review_id: item
            for item in preflight_result.high_risk_page_coverage_reviews
            if item.review_id
        }
        self._table_grid_by_id = {
            item.grid_id: item
            for item in preflight_result.table_grid_evidence
            if item.grid_id
        }
        self._page_text_qwen_no_issue_by_page = {
            int(item.page_no or 0): item
            for item in preflight_result.page_text_qwen_reviews
            if int(item.page_no or 0) > 0
            and item.available
            and str(item.verdict or "") == "no_obvious_issue"
            and not list(item.suspicious_values or [])
        }
        self._image_text_vl_no_issue_by_page = {
            int(item.page_no or 0): item
            for item in preflight_result.image_text_vl_reviews
            if int(item.page_no or 0) > 0
            and item.available
            and str(item.verdict or "") == "no_obvious_issue"
            and not list(item.suspicious_values or [])
        }
        self._table_page_vl_no_issue_by_page = {
            int(item.page_no or 0): item
            for item in preflight_result.table_page_vl_reviews
            if int(item.page_no or 0) > 0
            and item.available
            and str(item.verdict or "") == "no_obvious_issue"
            and not list(item.suspicious_values or [])
        }
        self._specialist_tasks_by_page: Dict[int, List[Any]] = {}
        for task in preflight_result.specialist_review_tasks:
            page_no = int(task.page_no or 0)
            if page_no > 0:
                self._specialist_tasks_by_page.setdefault(page_no, []).append(task)
        self._specialist_results_by_page: Dict[int, List[Any]] = {}
        for result in preflight_result.specialist_review_results:
            page_no = int(result.page_no or 0)
            if page_no > 0:
                self._specialist_results_by_page.setdefault(page_no, []).append(result)

        for task in preflight_result.coverage_review_tasks:
            if not task.planned_engine:
                task.planned_engine = str(task.next_engine or "")
            related_ids = self._related_coverage_ids(task=task)
            open_ids = [review_id for review_id in related_ids if review_id in self._open_coverage_ids]
            closed_ids = [
                review_id
                for review_id in related_ids
                if review_id in self._coverage_by_id and review_id not in self._open_coverage_ids
            ]
            task.open_coverage_review_ids = open_ids[:9999]
            task.closed_coverage_review_ids = closed_ids[:9999]
            task.open_coverage_count = len(open_ids)
            task.resolved_coverage_count = len(closed_ids)

        self._resolve_closed_tasks(tasks=preflight_result.coverage_review_tasks)
        summary["redundant_task_count"] = self._resolve_redundant_page_scope_tasks(
            tasks=preflight_result.coverage_review_tasks
        )

        executor_counts: Counter[str] = Counter()
        outcome_counts: Counter[str] = Counter()
        open_pages: set[int] = set()
        resolved_pages: set[int] = set()
        for task in preflight_result.coverage_review_tasks:
            if task.is_resolved():
                summary["resolved_task_count"] += 1
                if int(task.page_no or 0) > 0:
                    resolved_pages.add(int(task.page_no or 0))
            else:
                self._escalate_open_task(task=task)
                summary["human_required_task_count"] += 1
                if int(task.page_no or 0) > 0:
                    open_pages.add(int(task.page_no or 0))
            if task.executor:
                executor_counts[task.executor] += 1
            if task.execution_outcome:
                outcome_counts[task.execution_outcome] += 1

        summary["executor_counts"] = dict(sorted(executor_counts.items()))
        summary["execution_outcome_counts"] = dict(sorted(outcome_counts.items()))
        summary["open_pages"] = sorted(open_pages)[:120]
        summary["resolved_pages"] = sorted(resolved_pages)[:120]
        return summary

    def _related_coverage_ids(self, *, task: ReviewTask) -> List[str]:
        rows: List[str] = []
        for review_id in list(task.coverage_review_ids or []):
            self._append_unique(rows, review_id)

        source_ref = str(task.source_payload_ref or "")
        source_gap_id = str(task.source_gap_id or "")
        page_no = int(task.page_no or 0)
        target_unit_id = str(task.target_unit_id or "")

        if source_ref.endswith("high_risk_page_coverage_reviews.json"):
            review = self._high_risk_by_id.get(source_gap_id)
            for review_id in list(getattr(review, "coverage_review_ids", []) or []):
                self._append_unique(rows, review_id)

        if source_ref.endswith("table_grid_evidence.json"):
            grid = self._table_grid_by_id.get(source_gap_id)
            for cell in list(getattr(grid, "anomaly_cells", []) or []):
                review_id = str((cell or {}).get("coverage_review_id") or "").strip()
                if review_id:
                    self._append_unique(rows, review_id)

        if source_ref.endswith("conversion_diff_candidates.json"):
            for review_id in self._coverage_ids_by_diff.get(source_gap_id, []):
                self._append_unique(rows, review_id)
            for review_id in self._coverage_ids_by_unit.get(target_unit_id, []):
                self._append_unique(rows, review_id)

        if source_ref in self.PAGE_SCOPE_SOURCE_REFS or source_ref.endswith("fragment_anomaly_reviews.json"):
            for review_id in self._coverage_ids_by_page.get(page_no, []):
                self._append_unique(rows, review_id)

        return rows

    def _resolve_closed_tasks(self, *, tasks: Sequence[ReviewTask]) -> None:
        for task in tasks:
            if task.is_resolved():
                continue
            related_ids = list(task.open_coverage_review_ids or []) + list(task.closed_coverage_review_ids or [])
            if task.coverage_review_ids and not task.open_coverage_review_ids and task.closed_coverage_review_ids:
                self._mark_resolved(
                    task=task,
                    outcome=self._resolution_outcome_from_reviews(review_ids=related_ids),
                    reason=self._resolution_reason_from_reviews(review_ids=related_ids),
                    executor=self._resolution_executor_from_reviews(review_ids=related_ids),
                    refs=self._resolution_refs(review_ids=related_ids, page_no=int(task.page_no or 0)),
                )
                continue
            if (
                str(task.source_payload_ref or "") in self.PAGE_SCOPE_SOURCE_REFS
                and int(task.page_no or 0) > 0
                and not task.open_coverage_review_ids
            ):
                page_review_ids = self._coverage_ids_by_page.get(int(task.page_no or 0), [])
                self._mark_resolved(
                    task=task,
                    outcome=self._resolution_outcome_from_reviews(review_ids=page_review_ids),
                    reason=self._resolution_reason_from_reviews(review_ids=page_review_ids),
                    executor=self._resolution_executor_from_reviews(review_ids=page_review_ids),
                    refs=self._resolution_refs(review_ids=page_review_ids, page_no=int(task.page_no or 0)),
                )

    def _resolve_redundant_page_scope_tasks(self, *, tasks: Sequence[ReviewTask]) -> int:
        task_count = 0
        open_tasks_by_page: Dict[int, List[ReviewTask]] = {}
        for task in tasks:
            if task.is_open() and int(task.page_no or 0) > 0:
                open_tasks_by_page.setdefault(int(task.page_no or 0), []).append(task)
        for task in tasks:
            if (
                task.is_resolved()
                or str(task.source_payload_ref or "") not in self.PAGE_SCOPE_SOURCE_REFS
                or not task.open_coverage_review_ids
            ):
                continue
            page_no = int(task.page_no or 0)
            task_open_ids = set(task.open_coverage_review_ids)
            for other in open_tasks_by_page.get(page_no, []):
                if other is task or other.is_resolved():
                    continue
                if int(other.priority or 0) < int(task.priority or 0):
                    continue
                other_open_ids = set(other.open_coverage_review_ids or [])
                if not other_open_ids or not task_open_ids.issubset(other_open_ids):
                    continue
                if other.task_type == task.task_type and other.source_gap_id == task.source_gap_id:
                    continue
                self._mark_resolved(
                    task=task,
                    outcome="superseded_by_more_specific_open_task",
                    reason=f"同页更高优先级任务 {other.task_type} 已覆盖相同开放缺口，本任务转为已闭环记录，不再重复进入人工队列。",
                    executor="coverage_task_execution",
                    refs=[
                        {
                            "source": "coverage_review_task",
                            "id": str(other.task_id or other.source_gap_id or ""),
                            "task_type": other.task_type,
                            "raw": "coverage_review_tasks.json",
                        }
                    ],
                )
                task_count += 1
                break
        return task_count

    def _escalate_open_task(self, *, task: ReviewTask) -> None:
        family = self._task_family(task=task)
        task.status = "human_required"
        task.next_engine = "human_review"
        task.executor = self._executor_for_family(family=family, page_no=int(task.page_no or 0))
        task.execution_outcome = "automation_exhausted_requires_human_review"
        task.execution_reason = self._open_task_reason(family=family)
        refs = self._open_task_refs(task=task, family=family)
        if refs:
            task.execution_refs = refs[:20]
        if "automation_exhausted" not in task.flags:
            task.flags.append("automation_exhausted")

    def _task_family(self, *, task: ReviewTask) -> str:
        if task.task_type in self.TABLE_TASK_TYPES or str(task.planned_engine or "") == "table_audit_engine":
            return "table"
        if task.task_type in self.VISUAL_TASK_TYPES or str(task.planned_engine or "") in {"focused_crop_review", "qwen_vl_local_gate"}:
            return "visual"
        if task.task_type in self.ALIGNMENT_TASK_TYPES or str(task.planned_engine or "") in {"mapping_stabilizer", "docx_fragment_merger"}:
            return "alignment"
        if task.task_type == "human_review_required" or str(task.planned_engine or "") == "human_review":
            return "human"
        return "coverage"

    def _executor_for_family(self, *, family: str, page_no: int) -> str:
        if family == "table":
            if any(str(item.task_type or "").startswith("table_") for item in self._specialist_tasks_by_page.get(page_no, [])):
                return "table_specialist_executor"
            return "table_audit_engine"
        if family == "visual":
            if any(str(item.task_type or "") == "image_page_specialist_review" for item in self._specialist_tasks_by_page.get(page_no, [])):
                return "image_page_specialist_executor"
            if page_no in self._page_text_qwen_no_issue_by_page or page_no in self._image_text_vl_no_issue_by_page:
                return "full_page_review_chain"
            return "focused_visual_review_chain"
        if family == "alignment":
            return "mapping_stabilizer"
        if family == "human":
            return "human_review_queue"
        return "coverage_task_execution"

    def _open_task_reason(self, *, family: str) -> str:
        if family == "table":
            return "表格自动复核链路已执行，但仍存在未闭环单元/异常，下一步只能进入人工复核。"
        if family == "visual":
            return "局部/整页视觉复核链路已执行，但仍存在未闭环差异，下一步只能进入人工复核。"
        if family == "alignment":
            return "映射/碎片自动稳定链路已执行，但仍未恢复可靠对齐，下一步只能进入人工复核。"
        if family == "human":
            return "当前缺口没有安全的自动闭环器，直接进入人工复核。"
        return "自动闭环链路已穷尽，仍存在开放缺口，下一步只能进入人工复核。"

    def _open_task_refs(self, *, task: ReviewTask, family: str) -> List[Dict[str, Any]]:
        page_no = int(task.page_no or 0)
        rows: List[Dict[str, Any]] = []
        if family == "table":
            for item in self._specialist_tasks_by_page.get(page_no, []):
                if not str(item.task_type or "").startswith("table_"):
                    continue
                rows.append(
                    {
                        "source": "specialist_review_task",
                        "id": str(item.task_id or ""),
                        "task_type": str(item.task_type or ""),
                        "raw": "specialist_review_tasks.json",
                    }
                )
            for item in self._specialist_results_by_page.get(page_no, []):
                if not str(item.task_type or "").startswith("table_"):
                    continue
                rows.append(
                    {
                        "source": "specialist_review_result",
                        "id": str(item.result_id or ""),
                        "decision": str(item.decision or ""),
                        "raw": "specialist_review_results.json",
                    }
                )
            return rows[:20]
        if family == "visual":
            page_text = self._page_text_qwen_no_issue_by_page.get(page_no)
            if page_text is not None:
                rows.append({"source": "page_text_qwen_review", "id": str(page_text.review_id or ""), "raw": "page_text_qwen_reviews.json"})
            image_text = self._image_text_vl_no_issue_by_page.get(page_no)
            if image_text is not None:
                rows.append({"source": "image_text_vl_review", "id": str(image_text.review_id or ""), "raw": "image_text_vl_reviews.json"})
            for item in self._specialist_tasks_by_page.get(page_no, []):
                if str(item.task_type or "") != "image_page_specialist_review":
                    continue
                rows.append(
                    {
                        "source": "specialist_review_task",
                        "id": str(item.task_id or ""),
                        "task_type": str(item.task_type or ""),
                        "raw": "specialist_review_tasks.json",
                    }
                )
            for item in self._specialist_results_by_page.get(page_no, []):
                if str(item.task_type or "") != "image_page_specialist_review":
                    continue
                rows.append(
                    {
                        "source": "specialist_review_result",
                        "id": str(item.result_id or ""),
                        "decision": str(item.decision or ""),
                        "raw": "specialist_review_results.json",
                    }
                )
            return rows[:20]
        return rows

    def _resolution_outcome_from_reviews(self, *, review_ids: Sequence[str]) -> str:
        status = self._dominant_closed_status(review_ids=review_ids)
        if status in self.CLOSED_STATUS_DETAILS:
            return self.CLOSED_STATUS_DETAILS[status][1]
        return "resolved_by_closed_coverage"

    def _resolution_reason_from_reviews(self, *, review_ids: Sequence[str]) -> str:
        status = self._dominant_closed_status(review_ids=review_ids)
        if status in self.CLOSED_STATUS_DETAILS:
            return self.CLOSED_STATUS_DETAILS[status][2]
        return "对应覆盖缺口已被后续自动复核链路关闭，不再进入人工队列。"

    def _resolution_executor_from_reviews(self, *, review_ids: Sequence[str]) -> str:
        status = self._dominant_closed_status(review_ids=review_ids)
        if status in self.CLOSED_STATUS_DETAILS:
            return self.CLOSED_STATUS_DETAILS[status][0]
        return "coverage_task_execution"

    def _resolution_refs(self, *, review_ids: Sequence[str], page_no: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for review_id in review_ids[:20]:
            review = self._coverage_by_id.get(review_id)
            if review is None or review.decision != "covered":
                continue
            rows.append(
                {
                    "source": "content_coverage_review",
                    "id": review_id,
                    "status": str(review.status or ""),
                    "raw": "full_content_coverage_reviews.json",
                }
            )
        page_text = self._page_text_qwen_no_issue_by_page.get(page_no)
        if page_text is not None:
            rows.append({"source": "page_text_qwen_review", "id": str(page_text.review_id or ""), "raw": "page_text_qwen_reviews.json"})
        image_text = self._image_text_vl_no_issue_by_page.get(page_no)
        if image_text is not None:
            rows.append({"source": "image_text_vl_review", "id": str(image_text.review_id or ""), "raw": "image_text_vl_reviews.json"})
        table_page = self._table_page_vl_no_issue_by_page.get(page_no)
        if table_page is not None:
            rows.append({"source": "table_page_vl_review", "id": str(table_page.review_id or ""), "raw": "table_page_vl_reviews.json"})
        return rows[:20]

    def _dominant_closed_status(self, *, review_ids: Sequence[str]) -> str:
        counts: Counter[str] = Counter()
        for review_id in review_ids:
            review = self._coverage_by_id.get(review_id)
            if review is None or review.decision != "covered":
                continue
            counts[str(review.status or "covered")] += 1
        if counts:
            return counts.most_common(1)[0][0]
        return ""

    def _mark_resolved(
        self,
        *,
        task: ReviewTask,
        outcome: str,
        reason: str,
        executor: str,
        refs: Sequence[Dict[str, Any]],
    ) -> None:
        task.status = "resolved"
        task.next_engine = ""
        task.executor = executor
        task.execution_outcome = outcome
        task.execution_reason = reason
        task.execution_refs = [dict(item) for item in refs][:20]
        if "coverage_task_resolved" not in task.flags:
            task.flags.append("coverage_task_resolved")

    def _append_unique(self, rows: List[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in rows:
            rows.append(text)
