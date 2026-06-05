from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings

from .models import ContentCoverageReview, ConversionPreflightResult, DocxEvidenceUnit, HighRiskPageCoverageReview


class HighRiskPageCoverageBuilder:
    """Group unresolved coverage evidence by high-risk page.

    This is a recall layer, not an error-confirmation layer. It records where
    full-page review is still needed so later comments and quality inspection
    can point to pages that may contain unlocated WPS conversion errors.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        max_pages: Optional[int] = None,
        min_unresolved: Optional[int] = None,
        min_table_unresolved: Optional[int] = None,
        min_visual_unresolved: Optional[int] = None,
        min_mapping_uncertain: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_pages = max(0, int(max_pages if max_pages is not None else getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MAX_PAGES", 9999) or 9999))
        self.min_unresolved = max(1, int(min_unresolved if min_unresolved is not None else getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_UNRESOLVED", 12) or 12))
        self.min_table_unresolved = max(1, int(min_table_unresolved if min_table_unresolved is not None else getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_TABLE_UNRESOLVED", 8) or 8))
        self.min_visual_unresolved = max(1, int(min_visual_unresolved if min_visual_unresolved is not None else getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_VISUAL_UNRESOLVED", 6) or 6))
        self.min_mapping_uncertain = max(1, int(min_mapping_uncertain if min_mapping_uncertain is not None else getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_MAPPING_UNCERTAIN", 6) or 6))
        self.max_samples = max(1, int(max_samples if max_samples is not None else getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MAX_SAMPLES", 10) or 10))

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[HighRiskPageCoverageReview]:
        if not self.enabled or self.max_pages <= 0:
            return []
        self._docx_by_unit = {unit.unit_id: unit for unit in preflight_result.docx_units}
        resolved_coverage_ids = self._resolved_coverage_ids(preflight_result=preflight_result)
        by_page: Dict[int, List[ContentCoverageReview]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered" or review.review_id in resolved_coverage_ids:
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            by_page.setdefault(page_no, []).append(review)

        image_risk_by_page = {int(item.page_no or 0): item.risk_level for item in preflight_result.image_page_reviews}
        text_profile_by_page = {
            int(item.page_no or 0): item
            for item in preflight_result.page_text_coverage_profiles
            if int(item.page_no or 0) > 0
        }
        backfill_counts = self._backfill_counts(preflight_result=preflight_result)
        rows: List[HighRiskPageCoverageReview] = []
        for page_no, reviews in by_page.items():
            row = self._build_page_review(
                page_no=page_no,
                reviews=reviews,
                page_profile=preflight_result.page_profiles.get(str(page_no)) or {},
                image_risk=image_risk_by_page.get(page_no, ""),
                text_profile=text_profile_by_page.get(page_no),
                backfilled_count=backfill_counts.get(page_no, 0),
            )
            if row:
                rows.append(row)

        rows.sort(key=lambda item: (-int(item.priority or 0), int(item.page_no or 0)))
        for index, row in enumerate(rows[: self.max_pages], start=1):
            row.review_id = f"high_risk_page_coverage_{index:04d}"
        return rows[: self.max_pages]

    def _build_page_review(
        self,
        *,
        page_no: int,
        reviews: Sequence[ContentCoverageReview],
        page_profile: Dict[str, Any],
        image_risk: str,
        text_profile: Any,
        backfilled_count: int,
    ) -> Optional[HighRiskPageCoverageReview]:
        status_counts: Dict[str, int] = {}
        route_counts: Dict[str, int] = {}
        for review in reviews:
            key = f"{review.side}/{review.status}"
            status_counts[key] = status_counts.get(key, 0) + 1
            if review.next_route:
                route_counts[review.next_route] = route_counts.get(review.next_route, 0) + 1
        docx_reviews = [item for item in reviews if item.side == "docx"]
        pdf_reviews = [item for item in reviews if item.side == "pdf"]
        unresolved_count = len(reviews)
        table_count = sum(1 for item in reviews if item.status == "table_pending" or "container=table_cell" in set(item.flags))
        mapping_count = sum(1 for item in reviews if item.status == "mapping_uncertain")
        visual_routes = {
            "needs_region_ocr",
            "needs_qwen_vl",
            "needs_human_visual_review",
            "needs_full_page_ocr",
            "needs_region_segmentation",
            "needs_qwen_vl_page_gate",
        }
        visual_statuses = {"needs_pdf_ocr", "visual_pending", "needs_full_page_ocr", "needs_region_segmentation", "low_confidence_page_review"}
        visual_count = sum(1 for item in reviews if item.status in visual_statuses or item.next_route in visual_routes)
        needs_ocr_count = sum(1 for item in reviews if item.next_route in visual_routes or item.status in visual_statuses)
        labels = set(page_profile.get("labels") or [])
        canonical = str(page_profile.get("audit_canonical_page_type") or "")
        page_is_image = bool(
            canonical in {"scan_text_page", "table_image_page", "mixed_layout_page", "low_confidence_page"}
            or labels & {"scan_like", "image_text_heavy", "mixed_layout"}
            or page_profile.get("needs_ocr")
            or image_risk in {"high", "medium"}
        )
        text_profile_risk = str(getattr(text_profile, "risk_level", "") or "")
        text_profile_status = str(getattr(text_profile, "status", "") or "")
        low_text_coverage = text_profile_risk in {"high", "medium"} and text_profile_status not in {
            "page_text_coverage_supported",
            "no_text_evidence",
        }
        if not (
            unresolved_count >= self.min_unresolved
            or table_count >= self.min_table_unresolved
            or visual_count >= self.min_visual_unresolved
            or mapping_count >= self.min_mapping_uncertain
            or (page_is_image and unresolved_count >= max(3, self.min_unresolved // 2))
            or (page_is_image and low_text_coverage and unresolved_count >= 3)
        ):
            return None

        priority = self._priority(
            unresolved_count=unresolved_count,
            table_count=table_count,
            mapping_count=mapping_count,
            visual_count=visual_count,
            image_risk=image_risk,
            page_is_image=page_is_image,
        )
        if text_profile_risk == "high":
            priority = min(99, priority + 8)
        elif text_profile_risk == "medium":
            priority = min(99, priority + 4)
        risk_level = "high" if priority >= 85 or unresolved_count >= self.min_unresolved * 2 else "medium"
        anchor = self._comment_anchor(page_no=page_no, docx_reviews=docx_reviews)
        dominant = self._dominant(table_count=table_count, mapping_count=mapping_count, visual_count=visual_count, unresolved_count=unresolved_count)
        reason = self._reason(
            unresolved_count=unresolved_count,
            table_count=table_count,
            mapping_count=mapping_count,
            visual_count=visual_count,
            image_risk=image_risk,
            dominant=dominant,
        )
        flags = [f"dominant={dominant}", f"image_risk={image_risk or 'none'}"]
        if canonical:
            flags.append(f"canonical_page_type={canonical}")
        if text_profile_status:
            flags.append(f"page_text_coverage={text_profile_status}")
        if text_profile_risk:
            flags.append(f"page_text_risk={text_profile_risk}")
        flags.extend(f"label={label}" for label in sorted(labels)[:8])
        return HighRiskPageCoverageReview(
            review_id="",
            page_no=page_no,
            status="high_risk_page_unresolved",
            decision="review_required",
            risk_level=risk_level,
            priority=priority,
            unresolved_count=unresolved_count,
            docx_unresolved_count=len(docx_reviews),
            pdf_unresolved_count=len(pdf_reviews),
            table_unresolved_count=table_count,
            mapping_uncertain_count=mapping_count,
            visual_unresolved_count=visual_count,
            needs_ocr_count=needs_ocr_count,
            backfilled_count=backfilled_count,
            comment_anchor_unit_id=anchor.unit_id if anchor else "",
            comment_anchor_text=anchor.text if anchor else "",
            reason=reason,
            next_route="needs_table_parser" if dominant == "table" else "needs_full_page_review",
            coverage_review_ids=[item.review_id for item in reviews],
            docx_samples=self._samples(docx_reviews),
            pdf_samples=self._samples(pdf_reviews),
            status_counts=dict(sorted(status_counts.items())),
            route_counts=dict(sorted(route_counts.items())),
            page_text_coverage=self._text_profile_summary(text_profile),
            flags=flags,
        )

    def _text_profile_summary(self, text_profile: Any) -> Dict[str, Any]:
        if text_profile is None:
            return {}
        return {
            "status": str(getattr(text_profile, "status", "") or ""),
            "risk_level": str(getattr(text_profile, "risk_level", "") or ""),
            "docx_token_coverage_ratio": round(float(getattr(text_profile, "docx_token_coverage_ratio", 0.0) or 0.0), 4),
            "pdf_token_coverage_ratio": round(float(getattr(text_profile, "pdf_token_coverage_ratio", 0.0) or 0.0), 4),
            "page_text_similarity": round(float(getattr(text_profile, "page_text_similarity", 0.0) or 0.0), 4),
            "docx_gap_samples": list(getattr(text_profile, "docx_gap_samples", []) or [])[:6],
            "pdf_gap_samples": list(getattr(text_profile, "pdf_gap_samples", []) or [])[:6],
            "reason": str(getattr(text_profile, "reason", "") or "")[:220],
        }

    def _resolved_coverage_ids(self, *, preflight_result: ConversionPreflightResult) -> set[str]:
        resolved_unit_ids: set[str] = set()
        resolved_coverage_ids: set[str] = set()
        for result in preflight_result.specialist_review_results:
            if result.decision != "confirmed_error" or result.comment_policy != "comment_if_exact_replacement":
                continue
            if result.wps_unit_id:
                resolved_unit_ids.add(result.wps_unit_id)
            for ref in result.evidence_refs:
                if ref.get("source") == "content_coverage_review" and ref.get("id"):
                    resolved_coverage_ids.add(str(ref["id"]))
        for review in preflight_result.table_cell_evidence_reviews:
            if review.decision != "confirmed_error":
                continue
            if review.docx_unit_id:
                resolved_unit_ids.add(review.docx_unit_id)
            if review.coverage_review_id:
                resolved_coverage_ids.add(review.coverage_review_id)
        for review in preflight_result.page_ocr_text_evidence_reviews:
            if review.decision == "confirmed_error" and review.coverage_review_id:
                resolved_coverage_ids.add(review.coverage_review_id)
                if review.docx_unit_id:
                    resolved_unit_ids.add(review.docx_unit_id)
        for review in preflight_result.content_coverage_reviews:
            if review.unit_id in resolved_unit_ids:
                resolved_coverage_ids.add(review.review_id)
        return resolved_coverage_ids

    def _backfill_counts(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, int]:
        rows: Dict[int, int] = {}
        for item in preflight_result.content_coverage_backfills:
            if item.available:
                page_no = int(item.page_no or 0)
                rows[page_no] = rows.get(page_no, 0) + 1
        return rows

    def _priority(
        self,
        *,
        unresolved_count: int,
        table_count: int,
        mapping_count: int,
        visual_count: int,
        image_risk: str,
        page_is_image: bool,
    ) -> int:
        score = 60
        score += min(20, unresolved_count // 2)
        score += min(16, table_count)
        score += min(12, mapping_count)
        score += min(12, visual_count)
        if image_risk == "high":
            score += 10
        elif image_risk == "medium" or page_is_image:
            score += 5
        return min(99, score)

    def _dominant(self, *, table_count: int, mapping_count: int, visual_count: int, unresolved_count: int) -> str:
        ranked = {
            "table": table_count,
            "mapping": mapping_count,
            "visual": visual_count,
            "coverage": unresolved_count,
        }
        return max(ranked.items(), key=lambda item: item[1])[0]

    def _reason(
        self,
        *,
        unresolved_count: int,
        table_count: int,
        mapping_count: int,
        visual_count: int,
        image_risk: str,
        dominant: str,
    ) -> str:
        parts = [f"本页仍有 {unresolved_count} 个内容单元未被可靠覆盖"]
        if table_count:
            parts.append(f"表格/类表格 {table_count} 个")
        if mapping_count:
            parts.append(f"映射不确定 {mapping_count} 个")
        if visual_count:
            parts.append(f"需 OCR/视觉复核 {visual_count} 个")
        if image_risk:
            parts.append(f"图片页风险 {image_risk}")
        return "；".join(parts) + f"。主阻塞：{dominant}。"

    def _comment_anchor(self, *, page_no: int, docx_reviews: Sequence[ContentCoverageReview]) -> Optional[DocxEvidenceUnit]:
        for review in docx_reviews:
            unit = self._docx_by_unit.get(review.unit_id)
            if unit and unit.container_type != "table_cell" and str(unit.text or "").strip():
                return unit
        for review in docx_reviews:
            unit = self._docx_by_unit.get(review.unit_id)
            if unit and str(unit.text or "").strip():
                return unit
        for unit in self._docx_by_unit.values():
            if int(unit.estimated_page_no or 0) == page_no and str(unit.text or "").strip():
                return unit
        return None

    def _samples(self, reviews: Sequence[ContentCoverageReview]) -> List[Dict[str, Any]]:
        priority = {
            "diff_candidate": 0,
            "mapping_uncertain": 1,
            "table_pending": 2,
            "needs_pdf_ocr": 3,
            "uncovered_docx_content": 4,
            "uncovered_pdf_content": 5,
        }
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        ordered = sorted(reviews, key=lambda item: (priority.get(item.status, 9), item.review_id))
        for review in ordered:
            text = " ".join(str(review.text or "").split()).strip()
            if not text:
                text = review.unit_id
            compact = text[:80]
            key = f"{review.side}:{compact}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "review_id": review.review_id,
                    "unit_id": review.unit_id,
                    "status": review.status,
                    "next_route": review.next_route,
                    "text": compact,
                }
            )
            if len(rows) >= self.max_samples:
                break
        return rows
