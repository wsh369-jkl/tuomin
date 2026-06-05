from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .common import CorrectionCandidate
from .coverage_scheduler import CoverageGapScheduler
from .models import ConversionPreflightResult
from .page_terminal_state import PageTerminalStateMachine


class ProductReportBuilder:
    """Build the stable product-facing report contract for v4.

    The raw report still exposes detailed intermediate payloads for debugging.
    This builder creates the smaller structure that API clients and the review
    dashboard can rely on without understanding every internal stage.
    """

    VERSION = "product_report_v1"

    def build(
        self,
        *,
        audit_id: str,
        preflight_result: ConversionPreflightResult,
        findings: Sequence[Dict[str, Any]],
        corrections: Sequence[CorrectionCandidate],
        reviewed_docx_path: Path,
        report_path: Path,
        evidence_zip_path: Path,
    ) -> Dict[str, Any]:
        scheduler = CoverageGapScheduler()
        tasks = list(preflight_result.coverage_review_tasks)
        open_tasks = [task for task in tasks if task.requires_human_review()]
        page_risk_summary = self._page_risk_summary(preflight_result=preflight_result, findings=findings)
        page_terminal_summary = PageTerminalStateMachine().summary(pages=page_risk_summary)
        table_summary = self._table_summary(preflight_result=preflight_result)
        coverage_summary = self._coverage_summary(
            preflight_result=preflight_result,
            page_terminal_summary=page_terminal_summary,
        )
        return {
            "enabled": True,
            "version": self.VERSION,
            "audit_id": audit_id,
            "status": self._status(preflight_result=preflight_result, findings=findings),
            "summary": self._summary(
                preflight_result=preflight_result,
                findings=findings,
                corrections=corrections,
                page_terminal_summary=page_terminal_summary,
                open_tasks=open_tasks,
            ),
            "page_risk_summary": page_risk_summary,
            "page_terminal_summary": page_terminal_summary,
            "table_summary": table_summary,
            "coverage_summary": coverage_summary,
            "model_guard_summary": dict(
                preflight_result.model_output_guard
                or {
                    "enabled": False,
                    "version": "model_output_guard_v1",
                }
            ),
            "human_review_queue": scheduler.human_review_queue(tasks=tasks),
            "artifact_manifest": self._artifact_manifest(
                reviewed_docx_path=reviewed_docx_path,
                report_path=report_path,
                evidence_zip_path=evidence_zip_path,
            ),
        }

    def _status(self, *, preflight_result: ConversionPreflightResult, findings: Sequence[Dict[str, Any]]) -> str:
        quality_status = str((preflight_result.quality_inspection or {}).get("overall_status") or "")
        open_tasks = [task for task in preflight_result.coverage_review_tasks if task.requires_human_review()]
        if quality_status in {"needs_pipeline_improvement", "needs_more_review"}:
            return "needs_human_review"
        if any(str(item.get("status") or "") == "coverage_gap" for item in findings):
            return "needs_human_review"
        if open_tasks:
            return "needs_human_review"
        if any(str(item.get("status") or "") == "model_conflict" for item in findings):
            return "needs_human_review"
        if any(str(item.get("status") or "") == "confirmed_error" for item in findings):
            return "confirmed_errors_found"
        return "no_confirmed_error"

    def _summary(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        findings: Sequence[Dict[str, Any]],
        corrections: Sequence[CorrectionCandidate],
        page_terminal_summary: Dict[str, Any],
        open_tasks: Sequence[Any],
    ) -> Dict[str, Any]:
        finding_status_counts = Counter(str(item.get("status") or "unknown") for item in findings)
        finding_category_counts = Counter(str(item.get("category") or "unknown") for item in findings)
        return {
            "page_count": int(preflight_result.pdf_page_count or 0),
            "finding_count": len(findings),
            "comment_count": len(corrections),
            "confirmed_count": int(finding_status_counts.get("confirmed_error", 0)),
            "suspected_count": int(finding_status_counts.get("suspected_error", 0)),
            "model_conflict_count": int(finding_status_counts.get("model_conflict", 0)),
            "coverage_gap_count": int(finding_status_counts.get("coverage_gap", 0)),
            "finding_status_counts": dict(sorted(finding_status_counts.items())),
            "finding_category_counts": dict(sorted(finding_category_counts.items())),
            "human_review_task_count": len(open_tasks),
            "resolved_review_task_count": sum(1 for item in preflight_result.coverage_review_tasks if item.is_resolved()),
            "model_guarded_count": int((preflight_result.model_output_guard or {}).get("guarded_count") or 0),
            "page_terminal_state_counts": dict(page_terminal_summary.get("terminal_state_counts") or {}),
            "resolved_page_count": int(page_terminal_summary.get("resolved_page_count") or 0),
            "review_required_page_count": int(page_terminal_summary.get("review_required_page_count") or 0),
            "high_risk_review_required_page_count": int(
                page_terminal_summary.get("high_risk_review_required_page_count") or 0
            ),
        }

    def _page_risk_summary(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        findings: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pages: Dict[int, Dict[str, Any]] = {
            page_no: {
                "page_no": page_no,
                "risk_level": "low",
                "labels": list((preflight_result.page_profiles.get(str(page_no)) or {}).get("labels") or []),
                "finding_count": 0,
                "confirmed_count": 0,
                "suspected_count": 0,
                "model_conflict_count": 0,
                "coverage_gap_count": 0,
                "review_task_count": 0,
                "table_grid_count": 0,
                "unresolved_coverage_count": 0,
                "reasons": [],
            }
            for page_no in range(1, int(preflight_result.pdf_page_count or 0) + 1)
        }

        for finding in findings:
            page_no = int(finding.get("page_no") or 0)
            page = pages.get(page_no)
            if page is None:
                continue
            status = str(finding.get("status") or "")
            page["finding_count"] += 1
            if status == "confirmed_error":
                page["confirmed_count"] += 1
            elif status == "suspected_error":
                page["suspected_count"] += 1
            elif status == "model_conflict":
                page["model_conflict_count"] += 1
            elif status == "coverage_gap":
                page["coverage_gap_count"] += 1
            reason = str(finding.get("category") or status or "")
            if reason and reason not in page["reasons"]:
                page["reasons"].append(reason)

        for task in preflight_result.coverage_review_tasks:
            if not task.requires_human_review():
                continue
            page = pages.get(int(task.page_no or 0))
            if page is None:
                continue
            page["review_task_count"] += 1
            if task.task_type not in page["reasons"]:
                page["reasons"].append(task.task_type)

        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered":
                continue
            page = pages.get(int(review.page_no or 0))
            if page is not None:
                page["unresolved_coverage_count"] += 1

        for grid in preflight_result.table_grid_evidence:
            page = pages.get(int(grid.page_no or 0))
            if page is not None:
                page["table_grid_count"] += 1

        rows = list(pages.values())
        for page in rows:
            page["risk_level"] = self._page_risk_level(page)
            page["reasons"] = page["reasons"][:12]
        return PageTerminalStateMachine().apply(pages=rows)

    def _page_risk_level(self, page: Dict[str, Any]) -> str:
        if page["confirmed_count"] or page["coverage_gap_count"] or page["model_conflict_count"]:
            return "high"
        if page["suspected_count"] or page["review_task_count"] or page["unresolved_coverage_count"]:
            return "medium"
        labels = set(page.get("labels") or [])
        if labels & {"image_pdf_page", "scan_like", "table_heavy", "table_image_page"}:
            return "medium"
        return "low"

    def _table_summary(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        audit = dict(preflight_result.table_audit_summary or {})
        grid_status_counts = Counter(str(grid.status or "unknown") for grid in preflight_result.table_grid_evidence)
        return {
            "enabled": bool(preflight_result.table_audit_summary or preflight_result.table_grid_evidence),
            "reviewed_table_count": int(audit.get("reviewed_table_count") or len(preflight_result.table_grid_evidence)),
            "reviewed_cell_count": int(audit.get("reviewed_cell_count") or sum(int(grid.cell_count or 0) for grid in preflight_result.table_grid_evidence)),
            "confirmed_cell_count": int(audit.get("confirmed_cell_count") or sum(int(grid.confirmed_error_count or 0) for grid in preflight_result.table_grid_evidence)),
            "suspected_cell_count": int(audit.get("suspected_cell_count") or sum(int(grid.suspected_error_count or 0) for grid in preflight_result.table_grid_evidence)),
            "unresolved_cell_count": int(audit.get("unresolved_cell_count") or sum(int(grid.unresolved_cell_count or 0) for grid in preflight_result.table_grid_evidence)),
            "pattern_cluster_count": int(audit.get("cluster_count") or 0),
            "merged_comment_count": int(audit.get("merged_comment_count") or 0),
            "grid_status_counts": dict(sorted(grid_status_counts.items())),
            "high_risk_pages": sorted({int(grid.page_no or 0) for grid in preflight_result.table_grid_evidence if int(grid.unresolved_cell_count or 0) or int(grid.confirmed_error_count or 0)}),
        }

    def _coverage_summary(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        page_terminal_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        coverage_status_counts = Counter(str(item.status or "unknown") for item in preflight_result.content_coverage_reviews)
        coverage_decision_counts = Counter(str(item.decision or "unknown") for item in preflight_result.content_coverage_reviews)
        task_type_counts = Counter(str(item.task_type or "unknown") for item in preflight_result.coverage_review_tasks)
        task_status_counts = Counter(str(item.status or "unknown") for item in preflight_result.coverage_review_tasks)
        open_tasks = [item for item in preflight_result.coverage_review_tasks if item.requires_human_review()]
        unresolved_pages = sorted(
            {
                int(item.page_no or 0)
                for item in preflight_result.content_coverage_reviews
                if item.decision != "covered" and int(item.page_no or 0) > 0
            }
        )
        return {
            "coverage_review_count": len(preflight_result.content_coverage_reviews),
            "unresolved_count": sum(1 for item in preflight_result.content_coverage_reviews if item.decision != "covered"),
            "unresolved_pages": unresolved_pages,
            "coverage_status_counts": dict(sorted(coverage_status_counts.items())),
            "coverage_decision_counts": dict(sorted(coverage_decision_counts.items())),
            "review_task_count": len(preflight_result.coverage_review_tasks),
            "open_review_task_count": len(open_tasks),
            "resolved_review_task_count": sum(1 for item in preflight_result.coverage_review_tasks if item.is_resolved()),
            "review_task_type_counts": dict(sorted(task_type_counts.items())),
            "review_task_status_counts": dict(sorted(task_status_counts.items())),
            "quality_bottlenecks": list((preflight_result.quality_inspection or {}).get("bottlenecks") or [])[:80],
            "page_terminal_state_counts": dict(page_terminal_summary.get("terminal_state_counts") or {}),
            "high_risk_review_required_page_count": int(
                page_terminal_summary.get("high_risk_review_required_page_count") or 0
            ),
        }

    def _artifact_manifest(
        self,
        *,
        reviewed_docx_path: Path,
        report_path: Path,
        evidence_zip_path: Path,
    ) -> Dict[str, Any]:
        return {
            "reviewed_docx_path": str(reviewed_docx_path),
            "audit_report_path": str(report_path),
            "evidence_zip_path": str(evidence_zip_path),
            "raw_payload_paths": [
                "evidence/raw/product_report.json",
                "evidence/raw/conversion_preflight.json",
                "evidence/raw/quality_inspection.json",
                "evidence/raw/comment_corrections.json",
                "evidence/raw/model_output_guard.json",
            ],
        }
