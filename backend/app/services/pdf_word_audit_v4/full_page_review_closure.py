from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from app.core.config import settings

from .models import ConversionPreflightResult


class FullPageReviewCoverageCloser:
    """Resolve sampled non-table coverage gaps after full-page review.

    Page-level text/VL reviewers only inspect selected high-signal DOCX samples.
    Before this closure, a "no_obvious_issue" page review did not feed back into
    content coverage, so the same sampled gaps kept reappearing in
    high-risk-page aggregation and follow-up tasks. This closer resolves only
    the sampled DOCX coverage items that were explicitly sent into full-page
    review and passed without exact replacement findings.
    """

    def __init__(self, *, enabled: Optional[bool] = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_FULL_PAGE_REVIEW_CLOSURE_ENABLED", True)
            if enabled is None
            else enabled
        )

    def apply(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "version": "full_page_review_closure_v1",
            "resolved_review_count": 0,
            "page_text_qwen_resolved_count": 0,
            "image_text_vl_resolved_count": 0,
            "skipped_missing_coverage_id_count": 0,
            "skipped_non_docx_sample_count": 0,
            "page_count": 0,
            "pages": [],
            "status_counts": {},
            "resolved_review_ids": [],
        }
        if not self.enabled:
            self._store_summary(preflight_result=preflight_result, summary=summary)
            return summary

        coverage_by_id = {review.review_id: review for review in preflight_result.content_coverage_reviews}
        coverage_by_unit = {
            review.unit_id: review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx" and review.unit_id
        }
        resolved_pages: set[int] = set()
        original_status_counts: Counter[str] = Counter()

        for review in preflight_result.page_text_qwen_reviews:
            if not self._review_has_no_issue(review=review):
                continue
            for sample in list(review.docx_samples or []):
                resolved = self._resolve_sample(
                    sample=sample,
                    coverage_by_id=coverage_by_id,
                    coverage_by_unit=coverage_by_unit,
                    status="resolved_by_page_text_qwen_no_issue",
                    reason="该 DOCX 样例已进入页级 OCR 文本整页复核，未发现可定位差异，不再计入未处理覆盖缺口。",
                    confidence=max(float(review.confidence or 0.0), 0.72),
                    source_flag="page_text_qwen_no_issue",
                )
                if resolved is None:
                    summary["skipped_missing_coverage_id_count"] += 1
                    continue
                if resolved is False:
                    summary["skipped_non_docx_sample_count"] += 1
                    continue
                page_no, review_id, original_status = resolved
                resolved_pages.add(page_no)
                original_status_counts[original_status] += 1
                summary["resolved_review_count"] += 1
                summary["page_text_qwen_resolved_count"] += 1
                summary["resolved_review_ids"].append(review_id)

        for review in preflight_result.image_text_vl_reviews:
            if not self._review_has_no_issue(review=review):
                continue
            for sample in list(review.docx_samples or []):
                resolved = self._resolve_sample(
                    sample=sample,
                    coverage_by_id=coverage_by_id,
                    coverage_by_unit=coverage_by_unit,
                    status="resolved_by_image_text_vl_no_issue",
                    reason="该 DOCX 样例已进入图片页整页视觉复核，未发现可定位差异，不再计入未处理覆盖缺口。",
                    confidence=max(float(review.confidence or 0.0), 0.74),
                    source_flag="image_text_vl_no_issue",
                )
                if resolved is None:
                    summary["skipped_missing_coverage_id_count"] += 1
                    continue
                if resolved is False:
                    summary["skipped_non_docx_sample_count"] += 1
                    continue
                page_no, review_id, original_status = resolved
                resolved_pages.add(page_no)
                original_status_counts[original_status] += 1
                summary["resolved_review_count"] += 1
                summary["image_text_vl_resolved_count"] += 1
                summary["resolved_review_ids"].append(review_id)

        summary["page_count"] = len(resolved_pages)
        summary["pages"] = sorted(resolved_pages)[:120]
        summary["status_counts"] = dict(sorted(original_status_counts.items()))
        summary["resolved_review_ids"] = summary["resolved_review_ids"][:500]
        self._store_summary(preflight_result=preflight_result, summary=summary)
        return summary

    def _review_has_no_issue(self, *, review: Any) -> bool:
        if not bool(getattr(review, "available", False)):
            return False
        if str(getattr(review, "verdict", "") or "") != "no_obvious_issue":
            return False
        if list(getattr(review, "suspicious_values", []) or []):
            return False
        return True

    def _resolve_sample(
        self,
        *,
        sample: Dict[str, Any],
        coverage_by_id: Dict[str, Any],
        coverage_by_unit: Dict[str, Any],
        status: str,
        reason: str,
        confidence: float,
        source_flag: str,
    ) -> tuple[int, str, str] | bool | None:
        coverage_review_id = str(sample.get("coverage_review_id") or "").strip()
        unit_id = str(sample.get("unit_id") or "").strip()
        review = coverage_by_id.get(coverage_review_id) if coverage_review_id else None
        if review is None and unit_id:
            review = coverage_by_unit.get(unit_id)
        if review is None:
            return None
        if review.side != "docx":
            return False
        if review.decision == "covered":
            return False
        original_status = str(review.status or "")
        review.decision = "covered"
        review.status = status
        review.confidence = max(float(review.confidence or 0.0), float(confidence or 0.0))
        review.reason = reason
        for flag in ("full_page_review_closure", source_flag):
            if flag not in review.flags:
                review.flags.append(flag)
        return int(review.page_no or 0), review.review_id, original_status or "unknown"

    def _store_summary(self, *, preflight_result: ConversionPreflightResult, summary: Dict[str, Any]) -> None:
        reconciliation = dict(preflight_result.content_coverage_reconciliation or {})
        reconciliation["full_page_review_closure"] = dict(summary)
        preflight_result.content_coverage_reconciliation = reconciliation
