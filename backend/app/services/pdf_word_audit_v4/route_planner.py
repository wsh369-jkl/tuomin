from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from .common import has_high_value_field_content, looks_like_paragraphized_table_fragment, looks_like_table_header, looks_like_table_title
from .models import AlignmentLink, ConversionDiffCandidate, ConversionPreflightResult, DocxEvidenceUnit, PdfEvidenceUnit


class AuditRoutePlanner:
    """Plan type-specific v4 review routes before specialist modules run.

    This layer is intentionally separate from the older candidate-level
    ``review_routes``.  It classifies the document and every page, then writes a
    primary route back to page_profiles so downstream modules can choose the
    right evidence workflow instead of re-inferring page type independently.
    """

    VERSION = "audit_route_plan_v2"

    TEXT_ROUTE = "native_text_compare"
    NATIVE_TABLE_ROUTE = "native_table_compare"
    IMAGE_TEXT_ROUTE = "image_text_compare"
    IMAGE_TABLE_ROUTE = "image_table_cell_compare"
    IMAGE_FORM_ROUTE = "image_form_field_compare"
    MIXED_ROUTE = "mixed_region_compare"
    HUMAN_ROUTE = "human_review_only"

    def build(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        pages = self._build_pages(preflight_result=preflight_result)
        summary = self._summary(preflight_result=preflight_result, pages=pages)
        plan = {
            "enabled": True,
            "version": self.VERSION,
            "summary": summary,
            "pages": pages,
        }
        self._write_back_page_profiles(preflight_result=preflight_result, pages=pages, document_type=summary["document_type"])
        preflight_result.audit_route_plan = plan
        return plan

    def _build_pages(self, *, preflight_result: ConversionPreflightResult) -> List[Dict[str, Any]]:
        links_by_page = self._links_by_page(preflight_result.alignment_links)
        diffs_by_page = self._diffs_by_page(preflight_result.diff_candidates)
        docx_stats_by_page = self._docx_stats_by_page(preflight_result.docx_units)
        pdf_stats_by_page = self._pdf_stats_by_page(preflight_result.pdf_units)
        rows: List[Dict[str, Any]] = []
        for page_no in range(1, int(preflight_result.pdf_page_count or 0) + 1):
            profile = dict(preflight_result.page_profiles.get(str(page_no)) or {})
            docx_stats = docx_stats_by_page.get(page_no, {})
            pdf_stats = pdf_stats_by_page.get(page_no, {})
            link_stats = links_by_page.get(page_no, {})
            diff_stats = diffs_by_page.get(page_no, {})
            page_type = self._page_type(profile=profile, docx_stats=docx_stats, pdf_stats=pdf_stats)
            primary_route = self._primary_route(page_type=page_type, profile=profile, docx_stats=docx_stats, link_stats=link_stats)
            taxonomy = self._page_taxonomy(
                page_type=page_type,
                primary_route=primary_route,
                profile=profile,
                docx_stats=docx_stats,
                pdf_stats=pdf_stats,
                link_stats=link_stats,
                diff_stats=diff_stats,
            )
            secondary_routes = self._secondary_routes(
                page_type=page_type,
                primary_route=primary_route,
                taxonomy=taxonomy,
                profile=profile,
                docx_stats=docx_stats,
                link_stats=link_stats,
                diff_stats=diff_stats,
            )
            confidence = self._route_confidence(
                page_type=page_type,
                primary_route=primary_route,
                profile=profile,
                docx_stats=docx_stats,
                link_stats=link_stats,
            )
            rows.append(
                {
                    "page_no": page_no,
                    "page_type": page_type,
                    "canonical_page_type": taxonomy["canonical_page_type"],
                    "pdf_source_type": taxonomy["pdf_source_type"],
                    "recognition_strategy": taxonomy["recognition_strategy"],
                    "recognition_risk": taxonomy["recognition_risk"],
                    "quality_first": taxonomy["quality_first"],
                    "primary_route": primary_route,
                    "secondary_routes": secondary_routes,
                    "route_confidence": confidence,
                    "execution_policy": self._execution_policy(primary_route=primary_route, page_type=page_type),
                    "labels": list(profile.get("labels") or []),
                    "signals": self._signals(profile=profile, docx_stats=docx_stats, pdf_stats=pdf_stats, link_stats=link_stats, diff_stats=diff_stats, taxonomy=taxonomy),
                    "blockers": self._blockers(primary_route=primary_route, profile=profile, link_stats=link_stats, diff_stats=diff_stats, taxonomy=taxonomy),
                    "reasons": self._reasons(page_type=page_type, primary_route=primary_route, profile=profile, docx_stats=docx_stats, link_stats=link_stats, taxonomy=taxonomy),
                }
            )
        return rows

    def _summary(self, *, preflight_result: ConversionPreflightResult, pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        page_type_counts = Counter(str(item.get("page_type") or "unknown") for item in pages)
        canonical_page_type_counts = Counter(str(item.get("canonical_page_type") or "unknown") for item in pages)
        source_type_counts = Counter(str(item.get("pdf_source_type") or "unknown") for item in pages)
        recognition_strategy_counts = Counter(str(item.get("recognition_strategy") or "unknown") for item in pages)
        recognition_risk_counts = Counter(str(item.get("recognition_risk") or "unknown") for item in pages)
        route_counts = Counter(str(item.get("primary_route") or "unknown") for item in pages)
        image_pages = sum(
            page_type_counts.get(key, 0)
            for key in ("image_text_page", "image_table_page", "image_form_field_page", "mixed_complex_page")
        )
        native_pages = page_type_counts.get("native_text_page", 0) + page_type_counts.get("native_table_page", 0)
        table_pages = page_type_counts.get("image_table_page", 0) + page_type_counts.get("native_table_page", 0)
        page_count = max(1, len(pages))
        if native_pages / page_count >= 0.8 and image_pages / page_count <= 0.2:
            document_type = "text_pdf"
        elif image_pages / page_count >= 0.8 and native_pages / page_count <= 0.2:
            document_type = "image_pdf"
        else:
            document_type = "mixed_pdf"
        if table_pages / page_count >= 0.55:
            layout_family = "table_heavy"
        elif image_pages / page_count >= 0.55:
            layout_family = "image_text_heavy"
        elif native_pages / page_count >= 0.55:
            layout_family = "native_text_heavy"
        else:
            layout_family = "mixed_layout"
        source_type = self._document_source_type(source_type_counts=source_type_counts, page_count=page_count)
        return {
            "document_type": document_type,
            "document_source_type": source_type,
            "layout_family": layout_family,
            "page_count": len(pages),
            "page_type_counts": dict(sorted(page_type_counts.items())),
            "canonical_page_type_counts": dict(sorted(canonical_page_type_counts.items())),
            "pdf_source_type_counts": dict(sorted(source_type_counts.items())),
            "recognition_strategy_counts": dict(sorted(recognition_strategy_counts.items())),
            "recognition_risk_counts": dict(sorted(recognition_risk_counts.items())),
            "primary_route_counts": dict(sorted(route_counts.items())),
            "image_page_count": image_pages,
            "native_page_count": native_pages,
            "table_page_count": table_pages,
            "quality_first_page_count": sum(1 for item in pages if item.get("quality_first")),
            "low_confidence_page_count": canonical_page_type_counts.get("low_confidence_page", 0),
            "table_image_page_count": canonical_page_type_counts.get("table_image_page", 0),
            "scan_text_page_count": canonical_page_type_counts.get("scan_text_page", 0),
            "safe_default_route": self.TEXT_ROUTE if document_type == "text_pdf" else self.IMAGE_TEXT_ROUTE,
            "exact_replacement_required": True,
            "safe_to_generate_findings": False,
            "reason": self._document_reason(document_type=document_type, layout_family=layout_family, page_type_counts=page_type_counts),
        }

    def _write_back_page_profiles(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        pages: Sequence[Dict[str, Any]],
        document_type: str,
    ) -> None:
        for page in pages:
            page_no = int(page.get("page_no") or 0)
            if page_no <= 0:
                continue
            profile = preflight_result.page_profiles.setdefault(str(page_no), {})
            signals = dict(page.get("signals") or {})
            if signals.get("visual_table_signal_suppressed"):
                profile.setdefault("raw_table_like", bool(profile.get("table_like")))
                profile.setdefault("raw_needs_table_parser", bool(profile.get("needs_table_parser")))
                profile["table_like"] = False
                profile["needs_table_parser"] = False
                labels = [str(item) for item in (profile.get("labels") or []) if str(item) != "table_heavy"]
                labels.append("visual_table_signal_suppressed")
                profile["labels"] = sorted(set(labels))
            profile["audit_document_type"] = document_type
            profile["audit_page_type"] = page.get("page_type") or ""
            profile["audit_canonical_page_type"] = page.get("canonical_page_type") or ""
            profile["pdf_source_type"] = page.get("pdf_source_type") or ""
            profile["recognition_strategy"] = page.get("recognition_strategy") or ""
            profile["recognition_risk"] = page.get("recognition_risk") or ""
            profile["quality_first"] = bool(page.get("quality_first"))
            profile["primary_route"] = page.get("primary_route") or ""
            profile["secondary_routes"] = list(page.get("secondary_routes") or [])
            profile["route_confidence"] = page.get("route_confidence") or 0.0
            profile["route_blockers"] = list(page.get("blockers") or [])
            profile["route_reasons"] = list(page.get("reasons") or [])

    def _page_type(self, *, profile: Dict[str, Any], docx_stats: Dict[str, Any], pdf_stats: Dict[str, Any]) -> str:
        labels = set(profile.get("labels") or [])
        native_reliable = bool(profile.get("native_text_reliable")) and not bool(profile.get("needs_ocr"))
        table_like = self._table_like(profile=profile, docx_stats=docx_stats)
        image_like = bool(
            profile.get("needs_ocr")
            or not profile.get("native_text_reliable", True)
            or labels & {"scan_like", "image_text_heavy"}
        )
        if native_reliable and table_like:
            return "native_table_page"
        if native_reliable and not image_like:
            return "native_text_page"
        if image_like and table_like:
            return "image_table_page"
        if image_like and self._form_like(profile=profile, docx_stats=docx_stats, pdf_stats=pdf_stats):
            return "image_form_field_page"
        if image_like:
            return "image_text_page"
        if table_like:
            return "native_table_page"
        return "mixed_complex_page"

    def _page_taxonomy(
        self,
        *,
        page_type: str,
        primary_route: str,
        profile: Dict[str, Any],
        docx_stats: Dict[str, Any],
        pdf_stats: Dict[str, Any],
        link_stats: Dict[str, Any],
        diff_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = set(profile.get("labels") or [])
        native_reliable = bool(profile.get("native_text_reliable")) and not bool(profile.get("needs_ocr"))
        needs_ocr = bool(profile.get("needs_ocr"))
        table_like = self._table_like(profile=profile, docx_stats=docx_stats)
        image_area = float(profile.get("image_area_ratio") or 0.0)
        native_chars = int(profile.get("native_text_chars") or 0)
        anchor_chars = int(profile.get("anchor_ocr_text_chars") or 0)
        anchor_quality = str(profile.get("anchor_ocr_quality") or "").lower()
        uncertain_links = int(link_stats.get("uncertain_count") or 0)
        matched_links = int(link_stats.get("matched_count") or 0)
        diff_total = sum(int(value or 0) for value in diff_stats.values())
        mixed_signal = bool("mixed_layout" in labels or (native_reliable and (image_area >= 0.2 or table_like)))
        image_like = bool(needs_ocr or labels & {"scan_like", "image_text_heavy"} or (not native_reliable and image_area >= 0.2))

        low_confidence_reasons: List[str] = []
        if needs_ocr and anchor_chars < 40:
            low_confidence_reasons.append("weak_or_missing_page_ocr")
        if not native_reliable and not image_like and native_chars < 24:
            low_confidence_reasons.append("missing_native_and_visual_signal")
        if uncertain_links >= 3 and matched_links == 0:
            low_confidence_reasons.append("mapping_uncertain_without_matched_anchor")
        if "mapping_risk" in labels:
            low_confidence_reasons.append("mapping_risk_label")
        if profile.get("orientation_confidence") is not None and float(profile.get("orientation_confidence") or 0.0) < 0.35:
            low_confidence_reasons.append("low_orientation_confidence")
        if diff_total >= 12 and matched_links == 0:
            low_confidence_reasons.append("many_unresolved_candidates_without_mapping")

        if table_like and image_like:
            canonical = "table_image_page"
            source_type = "image_pdf_page"
            strategy = "table_structure_and_cell_ocr"
        elif table_like and native_reliable:
            canonical = "native_table_page"
            source_type = "native_text_pdf_page"
            strategy = "native_table_structure_compare"
        elif native_reliable and not mixed_signal:
            canonical = "native_text_page"
            source_type = "native_text_pdf_page"
            strategy = "native_text_compare"
        elif image_like:
            canonical = "scan_text_page"
            source_type = "image_pdf_page"
            strategy = "full_page_ocr_then_text_alignment"
        elif mixed_signal:
            canonical = "mixed_layout_page"
            source_type = "mixed_pdf_page"
            strategy = "region_segmentation_then_ocr"
        else:
            canonical = "low_confidence_page"
            source_type = "unknown_pdf_page"
            strategy = "low_confidence_full_page_review"

        if canonical == "scan_text_page" and "mixed_layout" in labels:
            canonical = "mixed_layout_page"
            source_type = "mixed_pdf_page"
            strategy = "region_segmentation_then_ocr"
        if canonical not in {"table_image_page", "native_table_page"} and low_confidence_reasons and not native_reliable and not table_like and anchor_chars < 20:
            canonical = "low_confidence_page"
            source_type = "unknown_pdf_page" if not image_like else "image_pdf_page"
            strategy = "low_confidence_full_page_review"

        risk = "low"
        if canonical in {"table_image_page", "low_confidence_page"} or len(low_confidence_reasons) >= 2:
            risk = "high"
        elif canonical in {"scan_text_page", "mixed_layout_page"} or low_confidence_reasons or anchor_quality in {"low", "failed"}:
            risk = "medium"
        quality_first = bool(canonical in {"scan_text_page", "table_image_page", "mixed_layout_page", "low_confidence_page"} or risk == "high")
        return {
            "canonical_page_type": canonical,
            "pdf_source_type": source_type,
            "recognition_strategy": strategy,
            "recognition_risk": risk,
            "quality_first": quality_first,
            "low_confidence_reasons": low_confidence_reasons,
            "legacy_page_type": page_type,
            "legacy_primary_route": primary_route,
        }

    def _primary_route(self, *, page_type: str, profile: Dict[str, Any], docx_stats: Dict[str, Any], link_stats: Dict[str, Any]) -> str:
        if page_type == "native_text_page":
            return self.TEXT_ROUTE
        if page_type == "native_table_page":
            return self.NATIVE_TABLE_ROUTE
        if page_type == "image_table_page":
            return self.IMAGE_TABLE_ROUTE
        if page_type == "image_form_field_page":
            return self.IMAGE_FORM_ROUTE
        if page_type == "image_text_page":
            return self.IMAGE_TEXT_ROUTE
        if page_type == "mixed_complex_page":
            return self.MIXED_ROUTE
        return self.HUMAN_ROUTE

    def _secondary_routes(
        self,
        *,
        page_type: str,
        primary_route: str,
        taxonomy: Dict[str, Any],
        profile: Dict[str, Any],
        docx_stats: Dict[str, Any],
        link_stats: Dict[str, Any],
        diff_stats: Dict[str, Any],
    ) -> List[str]:
        routes: List[str] = []
        if primary_route in {self.IMAGE_TABLE_ROUTE, self.NATIVE_TABLE_ROUTE}:
            routes.extend(["needs_table_parser", "needs_cell_ocr"])
            if primary_route == self.IMAGE_TABLE_ROUTE:
                routes.append("needs_qwen_vl_table_gate")
        elif primary_route == self.IMAGE_TEXT_ROUTE:
            routes.extend(["needs_region_ocr", "needs_qwen_text_gate"])
        elif primary_route == self.IMAGE_FORM_ROUTE:
            routes.extend(["needs_field_anchor_ocr", "needs_qwen_vl_field_gate"])
        elif primary_route == self.MIXED_ROUTE:
            routes.extend(["needs_region_segmentation", "needs_region_ocr"])
            if self._table_like(profile=profile, docx_stats=docx_stats):
                routes.append("needs_table_parser")
        elif primary_route == self.TEXT_ROUTE:
            routes.append("native_text_alignment")
        canonical = str(taxonomy.get("canonical_page_type") or "")
        if canonical == "scan_text_page":
            routes.extend(["needs_full_page_ocr", "needs_text_alignment"])
        elif canonical == "table_image_page":
            routes.extend(["needs_table_parser", "needs_cell_ocr", "needs_qwen_vl_table_gate"])
        elif canonical == "mixed_layout_page":
            routes.extend(["needs_region_segmentation", "needs_region_ocr", "needs_full_page_coverage_review"])
        elif canonical == "low_confidence_page":
            routes.extend(["needs_full_page_ocr", "needs_qwen_vl_page_gate", "needs_human_mapping_review"])
        if taxonomy.get("quality_first"):
            routes.append("quality_first_review")
        if int(link_stats.get("uncertain_count") or 0) >= 3 or int(diff_stats.get("mapping_uncertain") or 0) > 0:
            routes.append("needs_human_mapping_review")
        if int(diff_stats.get("unlocated_hard_field") or 0) > 0:
            routes.append("needs_recall_guard_review")
        return self._unique(routes)

    def _execution_policy(self, *, primary_route: str, page_type: str) -> Dict[str, Any]:
        source_truth = "pdf_native_text" if primary_route in {self.TEXT_ROUTE, self.NATIVE_TABLE_ROUTE} else "pdf_rendered_image"
        if primary_route in {self.IMAGE_TABLE_ROUTE, self.NATIVE_TABLE_ROUTE}:
            allowed = ["table_cell_comment", "table_text_comment", "table_page_review"]
        elif primary_route == self.IMAGE_TEXT_ROUTE:
            allowed = ["body_text_comment", "field_comment", "page_review"]
        elif primary_route == self.IMAGE_FORM_ROUTE:
            allowed = ["field_comment", "visual_field_review"]
        elif primary_route == self.TEXT_ROUTE:
            allowed = ["text_diff_comment", "field_comment"]
        else:
            allowed = ["page_review"]
        return {
            "source_truth": source_truth,
            "allowed_comment_types": allowed,
            "exact_replacement_required": True,
            "mapping_required": primary_route not in {self.IMAGE_TEXT_ROUTE, self.IMAGE_TABLE_ROUTE, self.MIXED_ROUTE},
            "can_auto_replace": False,
            "can_generate_field_finding": primary_route in {self.TEXT_ROUTE, self.NATIVE_TABLE_ROUTE},
            "page_type": page_type,
        }

    def _route_confidence(
        self,
        *,
        page_type: str,
        primary_route: str,
        profile: Dict[str, Any],
        docx_stats: Dict[str, Any],
        link_stats: Dict[str, Any],
    ) -> float:
        confidence = 0.62
        if profile.get("native_text_reliable") and primary_route in {self.TEXT_ROUTE, self.NATIVE_TABLE_ROUTE}:
            confidence += 0.22
        if primary_route in {self.IMAGE_TABLE_ROUTE, self.NATIVE_TABLE_ROUTE} and self._table_like(profile=profile, docx_stats=docx_stats):
            confidence += 0.18
        if primary_route in {self.IMAGE_TEXT_ROUTE, self.IMAGE_FORM_ROUTE} and profile.get("needs_ocr"):
            confidence += 0.12
        if int(link_stats.get("matched_count") or 0) >= 2:
            confidence += 0.04
        if int(link_stats.get("uncertain_count") or 0) >= 3:
            confidence -= 0.08
        return round(max(0.1, min(0.98, confidence)), 4)

    def _signals(
        self,
        *,
        profile: Dict[str, Any],
        docx_stats: Dict[str, Any],
        pdf_stats: Dict[str, Any],
        link_stats: Dict[str, Any],
        diff_stats: Dict[str, Any],
        taxonomy: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "canonical_page_type": str(taxonomy.get("canonical_page_type") or ""),
            "pdf_source_type": str(taxonomy.get("pdf_source_type") or ""),
            "recognition_strategy": str(taxonomy.get("recognition_strategy") or ""),
            "recognition_risk": str(taxonomy.get("recognition_risk") or ""),
            "quality_first": bool(taxonomy.get("quality_first")),
            "low_confidence_reasons": list(taxonomy.get("low_confidence_reasons") or []),
            "native_text_reliable": bool(profile.get("native_text_reliable")),
            "needs_ocr": bool(profile.get("needs_ocr")),
            "needs_table_parser": bool(profile.get("needs_table_parser")),
            "table_like": bool(profile.get("table_like")),
            "native_text_chars": int(profile.get("native_text_chars") or 0),
            "anchor_ocr_text_chars": int(profile.get("anchor_ocr_text_chars") or 0),
            "anchor_ocr_quality": str(profile.get("anchor_ocr_quality") or ""),
            "image_area_ratio": float(profile.get("image_area_ratio") or 0.0),
            "horizontal_line_count": int(profile.get("horizontal_line_count") or 0),
            "vertical_line_count": int(profile.get("vertical_line_count") or 0),
            "docx_unit_count": int(docx_stats.get("unit_count") or 0),
            "docx_table_cell_count": int(docx_stats.get("table_cell_count") or 0),
            "docx_paragraph_count": int(docx_stats.get("paragraph_count") or 0),
            "docx_hard_field_count": int(docx_stats.get("hard_field_count") or 0),
            "docx_table_keyword_count": int(docx_stats.get("table_keyword_count") or 0),
            "docx_digit_unit_count": int(docx_stats.get("digit_unit_count") or 0),
            "pdf_table_region_count": int(pdf_stats.get("table_region_count") or 0),
            "pdf_visual_region_count": int(pdf_stats.get("visual_region_count") or 0),
            "matched_link_count": int(link_stats.get("matched_count") or 0),
            "uncertain_link_count": int(link_stats.get("uncertain_count") or 0),
            "visual_table_signal_suppressed": self._visual_table_signal_suppressed(profile=profile, docx_stats=docx_stats),
            "diff_counts": dict(diff_stats),
        }

    def _blockers(
        self,
        *,
        primary_route: str,
        profile: Dict[str, Any],
        link_stats: Dict[str, Any],
        diff_stats: Dict[str, Any],
        taxonomy: Dict[str, Any],
    ) -> List[str]:
        blockers: List[str] = []
        if int(link_stats.get("uncertain_count") or 0) >= 3:
            blockers.append("mapping_uncertain")
        if primary_route in {self.IMAGE_TABLE_ROUTE, self.NATIVE_TABLE_ROUTE} and not bool(profile.get("needs_table_parser") or profile.get("table_like") or "table_heavy" in set(profile.get("labels") or [])):
            blockers.append("weak_table_signal")
        if primary_route in {self.IMAGE_TEXT_ROUTE, self.IMAGE_FORM_ROUTE, self.IMAGE_TABLE_ROUTE} and not profile.get("anchor_ocr_text_chars"):
            blockers.append("no_anchor_ocr_text")
        if int(diff_stats.get("page_coverage_gap") or 0) > 0:
            blockers.append("page_coverage_gap")
        blockers.extend(str(item) for item in taxonomy.get("low_confidence_reasons") or [])
        return blockers

    def _reasons(
        self,
        *,
        page_type: str,
        primary_route: str,
        profile: Dict[str, Any],
        docx_stats: Dict[str, Any],
        link_stats: Dict[str, Any],
        taxonomy: Dict[str, Any],
    ) -> List[str]:
        reasons = [
            f"page_type={page_type}",
            f"canonical_page_type={taxonomy.get('canonical_page_type') or ''}",
            f"recognition_strategy={taxonomy.get('recognition_strategy') or ''}",
            f"primary_route={primary_route}",
        ]
        if profile.get("native_text_reliable"):
            reasons.append("pdf_native_text_reliable")
        if profile.get("needs_ocr"):
            reasons.append("pdf_native_text_unreliable_or_image_based")
        if self._table_like(profile=profile, docx_stats=docx_stats):
            reasons.append("table_structure_or_docx_table_cells_present")
        if int(link_stats.get("uncertain_count") or 0) > 0:
            reasons.append(f"uncertain_links={int(link_stats.get('uncertain_count') or 0)}")
        return reasons

    def _document_reason(self, *, document_type: str, layout_family: str, page_type_counts: Counter[str]) -> str:
        return (
            f"document_type={document_type}; layout_family={layout_family}; "
            f"page_type_counts={dict(sorted(page_type_counts.items()))}"
        )

    def _document_source_type(self, *, source_type_counts: Counter[str], page_count: int) -> str:
        native = source_type_counts.get("native_text_pdf_page", 0)
        image = source_type_counts.get("image_pdf_page", 0)
        mixed = source_type_counts.get("mixed_pdf_page", 0)
        unknown = source_type_counts.get("unknown_pdf_page", 0)
        denominator = max(1, page_count)
        if native / denominator >= 0.8 and image / denominator <= 0.2 and unknown == 0:
            return "text_pdf"
        if image / denominator >= 0.8 and native / denominator <= 0.2:
            return "image_pdf"
        if unknown / denominator >= 0.4:
            return "low_confidence_pdf"
        if mixed or image:
            return "mixed_pdf"
        return "text_pdf"

    def _table_like(self, *, profile: Dict[str, Any], docx_stats: Dict[str, Any]) -> bool:
        labels = set(profile.get("labels") or [])
        docx_table_cells = int(docx_stats.get("table_cell_count") or 0)
        if docx_table_cells >= 8:
            return True
        if self._paragraphized_table_like(profile=profile, docx_stats=docx_stats):
            return True
        if self._visual_table_signal_suppressed(profile=profile, docx_stats=docx_stats):
            return False
        return bool(
            profile.get("needs_table_parser")
            or profile.get("table_like")
            or "table_heavy" in labels
        )

    def _visual_table_signal_suppressed(self, *, profile: Dict[str, Any], docx_stats: Dict[str, Any]) -> bool:
        labels = set(profile.get("labels") or [])
        has_visual_table_signal = bool(profile.get("needs_table_parser") or profile.get("table_like") or "table_heavy" in labels)
        if not has_visual_table_signal:
            return False
        if int(docx_stats.get("table_cell_count") or 0) > 0:
            return False
        anchor_chars = int(profile.get("anchor_ocr_text_chars") or 0)
        anchor_lines = int(profile.get("anchor_ocr_line_count") or 0)
        anchor_count = int(profile.get("anchor_ocr_anchor_count") or 0)
        visual_regions = int(profile.get("visual_region_count") or 0)
        horizontal = int(profile.get("horizontal_line_count") or 0)
        vertical = int(profile.get("vertical_line_count") or 0)
        if anchor_chars >= 80 and 4 <= anchor_lines <= 35 and anchor_count <= 8 and visual_regions <= 18:
            return True
        if anchor_chars >= 120 and anchor_lines <= 45 and anchor_count <= 10 and (horizontal + vertical) <= 24:
            return True
        return False

    def _form_like(self, *, profile: Dict[str, Any], docx_stats: Dict[str, Any], pdf_stats: Dict[str, Any]) -> bool:
        docx_hard_fields = int(docx_stats.get("hard_field_count") or 0)
        hard_fields = docx_hard_fields + int(pdf_stats.get("hard_field_count") or 0)
        labels = set(profile.get("labels") or [])
        return bool(
            docx_hard_fields >= 1
            and hard_fields >= 2
            and "table_heavy" not in labels
            and int(docx_stats.get("table_cell_count") or 0) < 8
        )

    def _paragraphized_table_like(self, *, profile: Dict[str, Any], docx_stats: Dict[str, Any]) -> bool:
        if int(docx_stats.get("table_cell_count") or 0) > 0:
            return False
        unit_count = int(docx_stats.get("unit_count") or 0)
        paragraph_count = int(docx_stats.get("paragraph_count") or 0)
        table_keywords = int(docx_stats.get("table_keyword_count") or 0)
        digit_units = int(docx_stats.get("digit_unit_count") or 0)
        anchor_lines = int(profile.get("anchor_ocr_line_count") or 0)
        anchor_count = int(profile.get("anchor_ocr_anchor_count") or 0)
        anchor_chars = int(profile.get("anchor_ocr_text_chars") or 0)
        return bool(
            unit_count >= 25
            and paragraph_count >= 20
            and table_keywords >= 3
            and digit_units >= 8
            and (anchor_lines >= 50 or anchor_count >= 10 or anchor_chars >= 450)
        )

    def _docx_stats_by_page(self, docx_units: Sequence[DocxEvidenceUnit]) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for unit in docx_units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            stats = rows.setdefault(
                page_no,
                {
                    "unit_count": 0,
                    "table_cell_count": 0,
                    "paragraph_count": 0,
                    "hard_field_count": 0,
                    "text_chars": 0,
                    "table_keyword_count": 0,
                    "digit_unit_count": 0,
                },
            )
            stats["unit_count"] += 1
            stats["text_chars"] += len(unit.normalized_text or "")
            if unit.container_type == "table_cell":
                stats["table_cell_count"] += 1
            if unit.container_type == "paragraph":
                stats["paragraph_count"] += 1
            if self._table_keyword(unit.text):
                stats["table_keyword_count"] += 1
            if re.search(r"\d", str(unit.text or "")):
                stats["digit_unit_count"] += 1
            if self._has_hard_field(unit.text):
                stats["hard_field_count"] += 1
        return rows

    def _pdf_stats_by_page(self, pdf_units: Sequence[PdfEvidenceUnit]) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for unit in pdf_units:
            page_no = int(unit.page_no or 0)
            if page_no <= 0:
                continue
            stats = rows.setdefault(page_no, {"table_region_count": 0, "visual_region_count": 0, "hard_field_count": 0, "text_chars": 0})
            stats["text_chars"] += len(unit.normalized_text or "")
            if unit.unit_type == "table_region" or unit.region_type == "table":
                stats["table_region_count"] += 1
            if "region" in unit.unit_type or unit.region_type:
                stats["visual_region_count"] += 1
            if self._has_hard_field(unit.text):
                stats["hard_field_count"] += 1
        return rows

    def _links_by_page(self, links: Sequence[AlignmentLink]) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for link in links:
            page_no = int(link.pdf_page_no or link.docx_estimated_page_no or 0)
            if page_no <= 0:
                continue
            stats = rows.setdefault(page_no, {"matched_count": 0, "uncertain_count": 0})
            if link.status == "matched":
                stats["matched_count"] += 1
            elif link.status == "mapping_uncertain":
                stats["uncertain_count"] += 1
        return rows

    def _diffs_by_page(self, diffs: Sequence[ConversionDiffCandidate]) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for diff in diffs:
            page_no = int(diff.pdf_page_no or diff.docx_estimated_page_no or 0)
            if page_no <= 0:
                continue
            stats = rows.setdefault(page_no, {})
            stats[diff.category] = int(stats.get(diff.category, 0)) + 1
        return rows

    def _has_hard_field(self, text: Any) -> bool:
        value = str(text or "")
        if not value:
            return False
        if has_high_value_field_content(value):
            return True
        return bool(
            re.search(r"(?<!\d)\d{17}[\dXx](?!\d)", value)
            or re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", value)
            or re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", value)
            or re.search(r"[¥￥]\s*\d+(?:,\d{3})*(?:\.\d+)?", value)
        )

    def _table_keyword(self, text: Any) -> bool:
        value = str(text or "")
        if not value:
            return False
        if looks_like_table_title(value) or looks_like_table_header(value):
            return True
        compact = "".join(value.split())
        if len(compact) <= 72 and looks_like_paragraphized_table_fragment(compact):
            return True
        return False

    def _unique(self, values: Sequence[str]) -> List[str]:
        rows: List[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            rows.append(value)
        return rows
