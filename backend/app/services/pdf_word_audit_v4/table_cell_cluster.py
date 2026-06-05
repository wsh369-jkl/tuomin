from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple


class TableCellClusterer:
    """Cluster table anomalies by page/table/column/issue type."""

    def cluster(self, *, grids: Sequence[Any], max_samples: int = 8) -> List[Dict[str, Any]]:
        rows: Dict[Tuple[int, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
        for grid in grids:
            page_no = int(getattr(grid, "page_no", 0) or 0)
            table_index = int(getattr(grid, "table_index", 0) or 0)
            for cell in getattr(grid, "anomaly_cells", []) or []:
                if not isinstance(cell, dict):
                    continue
                col_index = int(cell.get("col_index") or 0)
                issue_type = str(cell.get("anomaly_type") or cell.get("issue_type") or "table_cell_anomaly")
                rows[(page_no, table_index, col_index, issue_type)].append(dict(cell))

        clusters: List[Dict[str, Any]] = []
        for index, ((page_no, table_index, col_index, issue_type), cells) in enumerate(
            sorted(rows.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])),
            start=1,
        ):
            decisions = Counter(str(cell.get("decision") or "unknown") for cell in cells)
            evidence_ids = [
                str(cell.get("evidence_review_id") or "")
                for cell in cells
                if str(cell.get("evidence_review_id") or "")
            ]
            coverage_ids = [
                str(cell.get("coverage_review_id") or "")
                for cell in cells
                if str(cell.get("coverage_review_id") or "")
            ]
            samples = []
            for cell in cells[: max(1, int(max_samples or 1))]:
                samples.append(
                    {
                        "unit_id": str(cell.get("unit_id") or ""),
                        "row_index": cell.get("row_index"),
                        "col_index": cell.get("col_index"),
                        "text": str(cell.get("text") or "")[:80],
                        "expected_text": str(cell.get("expected_text") or "")[:80],
                        "decision": str(cell.get("decision") or ""),
                        "confidence": float(cell.get("confidence") or 0.0),
                    }
                )
            clusters.append(
                {
                    "cluster_id": f"table_cluster_{index:04d}",
                    "page_no": page_no,
                    "table_index": table_index or None,
                    "col_index": col_index or None,
                    "issue_type": issue_type,
                    "cell_count": len(cells),
                    "confirmed_cell_count": int(decisions.get("confirmed_error", 0)),
                    "suspected_cell_count": int(decisions.get("suspected_error", 0)),
                    "decision_counts": dict(sorted(decisions.items())),
                    "evidence_review_ids": evidence_ids[:80],
                    "coverage_review_ids": coverage_ids[:80],
                    "samples": samples,
                    "needs_human_review": int(decisions.get("confirmed_error", 0)) < len(cells),
                }
            )
        clusters.sort(
            key=lambda item: (
                -int(item.get("confirmed_cell_count") or 0),
                -int(item.get("cell_count") or 0),
                int(item.get("page_no") or 0),
                int(item.get("col_index") or 0),
            )
        )
        return clusters
