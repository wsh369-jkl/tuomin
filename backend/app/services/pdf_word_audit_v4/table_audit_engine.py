from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

from .models import ConversionPreflightResult
from .table_cell_cluster import TableCellClusterer


class TableAuditEngine:
    """Build product-level table audit summaries from existing v4 evidence."""

    VERSION = "table_audit_engine_v1"

    def build(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        grids = list(preflight_result.table_grid_evidence or [])
        cell_reviews = list(preflight_result.table_cell_evidence_reviews or [])
        table_results = [
            item
            for item in preflight_result.specialist_review_results
            if str(item.task_type or "").startswith("table_")
        ]
        clusters = TableCellClusterer().cluster(grids=grids)
        page_rows = self._page_rows(
            preflight_result=preflight_result,
            clusters=clusters,
            table_results=table_results,
        )
        representative_findings = self._representative_findings(results=table_results, limit=80)
        reviewed_cell_count = sum(int(getattr(grid, "cell_count", 0) or 0) for grid in grids)
        unresolved_cell_count = sum(int(getattr(grid, "unresolved_cell_count", 0) or 0) for grid in grids)
        confirmed_cell_count = sum(int(getattr(grid, "confirmed_error_count", 0) or 0) for grid in grids)
        suspected_cell_count = sum(int(getattr(grid, "suspected_error_count", 0) or 0) for grid in grids)
        decision_counts = Counter(str(item.decision or "unknown") for item in cell_reviews)
        cluster_issue_counts = Counter(str(item.get("issue_type") or "unknown") for item in clusters)
        return {
            "enabled": bool(grids or cell_reviews or table_results),
            "version": self.VERSION,
            "reviewed_table_count": len(grids),
            "reviewed_cell_count": reviewed_cell_count,
            "confirmed_cell_count": confirmed_cell_count,
            "suspected_cell_count": suspected_cell_count,
            "unresolved_cell_count": unresolved_cell_count,
            "merged_comment_count": self._merged_comment_count(table_results),
            "representative_finding_count": len(representative_findings),
            "cluster_count": len(clusters),
            "cluster_issue_counts": dict(sorted(cluster_issue_counts.items())),
            "cell_decision_counts": dict(sorted(decision_counts.items())),
            "pages": page_rows,
            "pattern_clusters": clusters[:120],
            "representative_findings": representative_findings,
            "cell_evidence_refs": [
                {
                    "source": "table_cell_evidence_review",
                    "id": item.review_id,
                    "page_no": item.page_no,
                    "docx_unit_id": item.docx_unit_id,
                    "decision": item.decision,
                    "issue_type": item.issue_type,
                }
                for item in cell_reviews[:300]
            ],
        }

    def _page_rows(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        clusters: Sequence[Dict[str, Any]],
        table_results: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        grids_by_page: Dict[int, List[Any]] = defaultdict(list)
        for grid in preflight_result.table_grid_evidence:
            page_no = int(grid.page_no or 0)
            if page_no > 0:
                grids_by_page[page_no].append(grid)
        clusters_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for cluster in clusters:
            page_no = int(cluster.get("page_no") or 0)
            if page_no > 0:
                clusters_by_page[page_no].append(dict(cluster))
        result_counts_by_page: Dict[int, Counter[str]] = defaultdict(Counter)
        merged_comments_by_page: Counter[int] = Counter()
        for result in table_results:
            page_no = int(getattr(result, "page_no", 0) or 0)
            if page_no <= 0:
                continue
            result_counts_by_page[page_no][str(getattr(result, "decision", "") or "unknown")] += 1
            group_count = int(((getattr(result, "context", {}) or {}).get("group_summary") or {}).get("count") or 1)
            if group_count > 1:
                merged_comments_by_page[page_no] += 1
        rows: List[Dict[str, Any]] = []
        for page_no in sorted(set(grids_by_page) | set(clusters_by_page) | set(result_counts_by_page)):
            grids = grids_by_page.get(page_no, [])
            rows.append(
                {
                    "page_no": page_no,
                    "reviewed_table_count": len(grids),
                    "reviewed_cell_count": sum(int(grid.cell_count or 0) for grid in grids),
                    "confirmed_cell_count": sum(int(grid.confirmed_error_count or 0) for grid in grids),
                    "suspected_cell_count": sum(int(grid.suspected_error_count or 0) for grid in grids),
                    "unresolved_cell_count": sum(int(grid.unresolved_cell_count or 0) for grid in grids),
                    "cluster_count": len(clusters_by_page.get(page_no, [])),
                    "merged_comment_count": int(merged_comments_by_page.get(page_no, 0)),
                    "result_decision_counts": dict(sorted(result_counts_by_page.get(page_no, Counter()).items())),
                    "dominant_issue_types": self._dominant_issue_types(clusters_by_page.get(page_no, [])),
                    "needs_human_review": any(cluster.get("needs_human_review") for cluster in clusters_by_page.get(page_no, []))
                    or any(int(grid.unresolved_cell_count or 0) > 0 for grid in grids),
                }
            )
        return rows

    def _representative_findings(self, *, results: Sequence[Any], limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for result in results:
            if result.decision not in {"confirmed_error", "suspected_error"}:
                continue
            context = dict(result.context or {})
            group_summary = dict(context.get("group_summary") or {})
            rows.append(
                {
                    "result_id": result.result_id,
                    "task_id": result.task_id,
                    "page_no": result.page_no,
                    "decision": result.decision,
                    "issue_type": result.issue_type,
                    "wps_unit_id": result.wps_unit_id,
                    "old_text": result.old_text,
                    "new_text": result.new_text,
                    "confidence": round(float(result.confidence or 0.0), 4),
                    "merged_cell_count": int(group_summary.get("count") or 1),
                    "table_position": dict(context.get("table_position") or {}),
                    "related_units": list(context.get("related_units") or [])[:30],
                    "evidence_refs": [dict(item) for item in result.evidence_refs],
                    "reason": result.reason,
                }
            )
            if len(rows) >= max(0, int(limit or 0)):
                break
        return rows

    def _merged_comment_count(self, results: Sequence[Any]) -> int:
        count = 0
        for result in results:
            group_count = int(((result.context or {}).get("group_summary") or {}).get("count") or 1)
            if group_count > 1:
                count += 1
        return count

    def _dominant_issue_types(self, clusters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counts = Counter(str(item.get("issue_type") or "unknown") for item in clusters)
        return [{"issue_type": key, "count": value} for key, value in counts.most_common(5)]
