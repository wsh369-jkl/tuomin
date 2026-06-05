from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from app.core.config import settings

from .common import normalize_text
from .full_content_coverage import FullContentCoverageBuilder
from .models import ConversionPreflightResult, DocxEvidenceUnit


class TableHeavyPageCoverageCloser:
    """Conservatively collapse false-open coverage gaps on table-heavy pages.

    This runs after specialist evidence is available. It only resolves DOCX
    table-cell coverage reviews when the page is clearly table-heavy, no table
    anomaly already targets the unit, and the cell text is directly supported by
    same-page or nearby-page OCR/backfill text. It does not downgrade any real
    mismatch candidate into "covered".
    """

    TARGET_STATUSES = {"mapping_uncertain", "table_pending", "uncovered_docx_content"}
    TABLE_PRIMARY_ROUTES = {"image_table_cell_compare", "native_table_compare"}

    def __init__(self, *, enabled: Optional[bool] = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_HEAVY_PAGE_CLOSURE_ENABLED", True)
            if enabled is None
            else enabled
        )

    def apply(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "version": "table_heavy_page_coverage_closure_v1",
            "eligible_review_count": 0,
            "resolved_review_count": 0,
            "same_page_resolved_count": 0,
            "nearby_page_resolved_count": 0,
            "blocked_by_existing_anomaly_count": 0,
            "page_count": 0,
            "pages": [],
            "status_counts": {},
            "resolved_review_ids": [],
        }
        if not self.enabled:
            self._store_summary(preflight_result=preflight_result, summary=summary)
            return summary

        coverage_builder = FullContentCoverageBuilder()
        page_texts = self._combined_page_texts(
            preflight_result=preflight_result,
            coverage_builder=coverage_builder,
        )
        if not page_texts:
            self._store_summary(preflight_result=preflight_result, summary=summary)
            return summary

        docx_by_id = {unit.unit_id: unit for unit in preflight_result.docx_units}
        blocking_unit_ids = self._blocking_unit_ids(preflight_result=preflight_result)
        resolved_pages: set[int] = set()
        status_counts: Counter[str] = Counter()

        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered" or review.side != "docx" or review.status not in self.TARGET_STATUSES:
                continue
            unit = docx_by_id.get(review.unit_id)
            if unit is None or not self._eligible_table_review(review=review, unit=unit, page_profiles=preflight_result.page_profiles):
                continue
            summary["eligible_review_count"] += 1
            if review.related_diff_id:
                continue
            if unit.unit_id in blocking_unit_ids:
                summary["blocked_by_existing_anomaly_count"] += 1
                continue

            page_no = int(review.page_no or unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            coverage = coverage_builder._page_text_coverage(
                unit=unit,
                page_no=page_no,
                page_texts=page_texts,
            )
            if not coverage:
                continue

            matched_page_no = int(coverage.get("page_no") or page_no)
            same_page = matched_page_no == page_no
            original_status = str(review.status or "")
            review.decision = "covered"
            review.status = "covered_by_table_page_text_support" if same_page else "covered_by_nearby_table_page_text_support"
            review.confidence = max(
                float(review.confidence or 0.0),
                float(coverage.get("confidence") or 0.0),
                0.76 if same_page else 0.7,
            )
            review.reason = (
                "表格重灾页专项复核在页级 OCR/回填文本中重新命中该 DOCX 表格单元，"
                "且该单元没有独立异常证据，故不再计入未处理覆盖缺口。"
                if same_page
                else f"表格重灾页专项复核在相邻 PDF 第 {matched_page_no} 页页级 OCR/回填文本中重新命中该 DOCX 表格单元，"
                "按页映射偏移记录，不再计入未处理覆盖缺口。"
            )
            for flag in (
                "table_heavy_page_closure",
                f"closure_original_status={original_status or 'unknown'}",
                "closure_source=same_page_text" if same_page else f"closure_source=nearby_page_text:{matched_page_no}",
            ):
                if flag not in review.flags:
                    review.flags.append(flag)

            resolved_pages.add(page_no)
            status_counts[original_status or "unknown"] += 1
            summary["resolved_review_count"] += 1
            summary["same_page_resolved_count" if same_page else "nearby_page_resolved_count"] += 1
            summary["resolved_review_ids"].append(review.review_id)

        summary["page_count"] = len(resolved_pages)
        summary["pages"] = sorted(resolved_pages)[:120]
        summary["status_counts"] = dict(sorted(status_counts.items()))
        summary["resolved_review_ids"] = summary["resolved_review_ids"][:500]
        self._store_summary(preflight_result=preflight_result, summary=summary)
        return summary

    def _combined_page_texts(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        coverage_builder: FullContentCoverageBuilder,
    ) -> Dict[int, str]:
        page_texts = dict(coverage_builder._page_texts(preflight_result.pdf_units))
        min_chars = max(1, int(getattr(coverage_builder, "min_text_chars", 4) or 4))
        for backfill in preflight_result.content_coverage_backfills:
            if not backfill.available:
                continue
            page_no = int(backfill.page_no or 0)
            if page_no <= 0:
                continue
            text = str(backfill.normalized_text or normalize_text(backfill.extracted_text or ""))
            if len(text) < min_chars:
                continue
            existing = page_texts.get(page_no, "")
            page_texts[page_no] = f"{existing}\n{text}".strip() if existing else text
        return page_texts

    def _blocking_unit_ids(self, *, preflight_result: ConversionPreflightResult) -> set[str]:
        rows: set[str] = set()
        for review in preflight_result.table_cell_evidence_reviews:
            unit_id = str(review.docx_unit_id or "").strip()
            if unit_id and review.decision in {"confirmed_error", "suspected_error"}:
                rows.add(unit_id)
        for grid in preflight_result.table_grid_evidence:
            for cell in list(getattr(grid, "anomaly_cells", []) or []):
                unit_id = str((cell or {}).get("unit_id") or "").strip()
                if unit_id:
                    rows.add(unit_id)
        for result in preflight_result.specialist_review_results:
            if result.decision not in {"confirmed_error", "suspected_error"}:
                continue
            unit_id = str(result.wps_unit_id or "").strip()
            if unit_id:
                rows.add(unit_id)
            for related in list((result.context or {}).get("related_units") or []):
                related_unit_id = str((related or {}).get("unit_id") or "").strip()
                if related_unit_id:
                    rows.add(related_unit_id)
        return rows

    def _eligible_table_review(
        self,
        *,
        review: Any,
        unit: DocxEvidenceUnit,
        page_profiles: Dict[str, Dict[str, Any]],
    ) -> bool:
        if unit.container_type != "table_cell":
            return False
        page_no = int(review.page_no or unit.estimated_page_no or 0)
        profile = dict(page_profiles.get(str(page_no)) or {})
        labels = set(profile.get("labels") or [])
        flags = set(review.flags or [])
        canonical = str(profile.get("audit_canonical_page_type") or "")
        primary_route = str(profile.get("primary_route") or "")
        return bool(
            int(unit.table_index or 0) > 0
            and (
                profile.get("needs_table_parser")
                or profile.get("table_like")
                or "table_heavy" in labels
                or canonical in {"table_image_page", "mixed_layout_page"}
                or primary_route in self.TABLE_PRIMARY_ROUTES
                or review.next_route == "needs_table_parser"
                or "container=table_cell" in flags
            )
        )

    def _store_summary(self, *, preflight_result: ConversionPreflightResult, summary: Dict[str, Any]) -> None:
        reconciliation = dict(preflight_result.content_coverage_reconciliation or {})
        reconciliation["table_heavy_page_closure"] = dict(summary)
        preflight_result.content_coverage_reconciliation = reconciliation
