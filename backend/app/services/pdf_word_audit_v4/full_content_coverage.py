from __future__ import annotations

import re
from typing import Dict, List, Sequence

from app.core.config import settings

from .common import normalize_text
from .models import (
    AlignmentLink,
    ContentCoverageReview,
    ConversionDiffCandidate,
    ConversionPreflightResult,
    DocxEvidenceUnit,
    PdfEvidenceUnit,
)


class FullContentCoverageBuilder:
    """Track full-document coverage, not just hard-field candidates."""

    def __init__(self, *, enabled: bool | None = None, min_text_chars: int | None = None) -> None:
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_FULL_CONTENT_COVERAGE_ENABLED", True) if enabled is None else enabled)
        self.min_text_chars = max(1, int(min_text_chars if min_text_chars is not None else getattr(settings, "PDF_WORD_AUDIT_V4_FULL_CONTENT_MIN_TEXT_CHARS", 4) or 4))

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[ContentCoverageReview]:
        if not self.enabled:
            return []
        links_by_docx = {link.docx_unit_id: link for link in preflight_result.alignment_links}
        links_by_pdf = {link.pdf_unit_id: link for link in preflight_result.alignment_links}
        diffs_by_docx = self._diffs_by_docx(preflight_result.diff_candidates)
        diffs_by_pdf = self._diffs_by_pdf(preflight_result.diff_candidates)
        page_routes = self._page_routes(preflight_result)
        page_contexts = self._page_contexts(preflight_result=preflight_result, page_routes=page_routes)
        page_texts = self._page_texts(preflight_result.pdf_units)
        docx_page_texts = self._docx_page_texts(preflight_result.docx_units)
        comparable_pages = self._comparable_pdf_pages(preflight_result.pdf_units)

        reviews: List[ContentCoverageReview] = []
        for unit in sorted(preflight_result.docx_units, key=lambda item: item.order_index):
            if not self._docx_unit_in_scope(unit):
                continue
            link = links_by_docx.get(unit.unit_id)
            diff = diffs_by_docx.get(unit.unit_id)
            reviews.append(
                self._docx_review(
                    reviews=reviews,
                    unit=unit,
                    link=link,
                    diff=diff,
                    page_routes=page_routes.get(int(unit.estimated_page_no or 0), set()),
                    page_context=page_contexts.get(int(unit.estimated_page_no or 0), {}),
                    page_texts=page_texts,
                    comparable_pages=comparable_pages,
                )
            )

        for unit in sorted(preflight_result.pdf_units, key=lambda item: (item.page_no, item.order_index, item.unit_id)):
            if not self._pdf_unit_in_scope(unit):
                continue
            link = links_by_pdf.get(unit.unit_id)
            diff = diffs_by_pdf.get(unit.unit_id)
            reviews.append(
                self._pdf_review(
                    reviews=reviews,
                    unit=unit,
                    link=link,
                    diff=diff,
                    page_routes=page_routes.get(int(unit.page_no or 0), set()),
                    page_context=page_contexts.get(int(unit.page_no or 0), {}),
                    docx_page_texts=docx_page_texts,
                )
            )
        return reviews

    def _docx_review(
        self,
        *,
        reviews: Sequence[ContentCoverageReview],
        unit: DocxEvidenceUnit,
        link: AlignmentLink | None,
        diff: ConversionDiffCandidate | None,
        page_routes: set[str],
        page_context: Dict[str, str],
        page_texts: Dict[int, str],
        comparable_pages: set[int],
    ) -> ContentCoverageReview:
        flags = self._coverage_flags(unit.flags, page_context=page_context)
        page_no = int(unit.estimated_page_no or 0)
        if unit.container_type:
            flags.append(f"container={unit.container_type}")
        if link and link.status == "matched":
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status="covered",
                decision="covered",
                confidence=link.confidence,
                reason="DOCX 内容单元已找到高置信 PDF 对应内容。",
                text=unit.text,
                related_link_id=link.link_id,
                flags=flags,
            )
        if diff:
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status="diff_candidate",
                decision="review_required",
                confidence=max(float(diff.confidence or 0.0), float(diff.alignment_confidence or 0.0)),
                reason=f"DOCX 内容已进入差异候选：{diff.category}。",
                text=unit.text,
                next_route=self._route_for_diff(diff.category, page_routes=page_routes, page_context=page_context),
                related_diff_id=diff.diff_id,
                flags=[*flags, f"diff_category={diff.category}"],
            )
        if link and link.status == "mapping_uncertain":
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status="mapping_uncertain",
                decision="review_required",
                confidence=link.confidence,
                reason="DOCX 内容只找到低置信 PDF 映射，不能认为已经审查通过。",
                text=unit.text,
                next_route="needs_human_mapping_review",
                related_link_id=link.link_id,
                flags=flags,
            )
        page_coverage = self._page_text_coverage(unit=unit, page_no=page_no, page_texts=page_texts)
        if page_coverage:
            coverage_flags = [*flags, "covered_by_page_ocr"]
            covered_page_no = int(page_coverage.get("page_no") or 0)
            if page_coverage.get("status") == "covered_by_nearby_page_ocr":
                coverage_flags.extend(["page_mapping_shift_suspect", f"covered_pdf_page={covered_page_no}"])
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status=str(page_coverage.get("status") or "covered_by_page_ocr"),
                decision="covered",
                confidence=float(page_coverage.get("confidence") or 0.74),
                reason=str(page_coverage.get("reason") or "DOCX 内容在 PDF 页级 OCR 文本中保守命中，视为已覆盖。"),
                text=unit.text,
                flags=coverage_flags,
            )
        if unit.container_type == "table_cell" or "needs_table_parser" in page_routes:
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status="table_pending",
                decision="review_required",
                confidence=0.54,
                reason="DOCX 表格/类表格内容尚未建立单元格级 PDF 覆盖。",
                text=unit.text,
                next_route="needs_table_parser",
                flags=flags,
            )
        if page_no > 0 and page_no not in comparable_pages:
            route = self._route_for_uncovered_docx(page_context=page_context, page_routes=page_routes)
            status = self._status_for_route(route=route, side="docx", fallback="needs_pdf_ocr")
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status=status,
                decision="review_required",
                confidence=0.5,
                reason=self._reason_for_uncovered_docx(route=route, page_context=page_context),
                text=unit.text,
                next_route=route,
                flags=flags,
            )
        if unit.container_type in {"header", "footer", "footnote", "endnote"}:
            return self._review(
                reviews,
                side="docx",
                unit_id=unit.unit_id,
                page_no=unit.estimated_page_no,
                status="low_priority_uncovered",
                decision="review_required",
                confidence=0.46,
                reason="页眉页脚/脚注内容未找到 PDF 对应内容，需要在全内容审查中确认。",
                text=unit.text,
                next_route="needs_human_mapping_review",
                flags=flags,
            )
        route = self._route_for_uncovered_docx(page_context=page_context, page_routes=page_routes)
        return self._review(
            reviews,
            side="docx",
            unit_id=unit.unit_id,
            page_no=unit.estimated_page_no,
            status=self._status_for_route(route=route, side="docx", fallback="uncovered_docx_content"),
            decision="review_required",
            confidence=0.56,
            reason=self._reason_for_uncovered_docx(route=route, page_context=page_context),
            text=unit.text,
            next_route=route,
            flags=flags,
        )

    def _pdf_review(
        self,
        *,
        reviews: Sequence[ContentCoverageReview],
        unit: PdfEvidenceUnit,
        link: AlignmentLink | None,
        diff: ConversionDiffCandidate | None,
        page_routes: set[str],
        page_context: Dict[str, str],
        docx_page_texts: Dict[int, str],
    ) -> ContentCoverageReview:
        flags = self._coverage_flags(unit.flags, page_context=page_context)
        if unit.unit_type:
            flags.append(f"unit_type={unit.unit_type}")
        if link and link.status == "matched":
            return self._review(
                reviews,
                side="pdf",
                unit_id=unit.unit_id,
                page_no=unit.page_no,
                status="covered",
                decision="covered",
                confidence=link.confidence,
                reason="PDF 内容单元已找到高置信 DOCX 对应内容。",
                text=unit.text,
                related_link_id=link.link_id,
                flags=flags,
            )
        if diff:
            return self._review(
                reviews,
                side="pdf",
                unit_id=unit.unit_id,
                page_no=unit.page_no,
                status="diff_candidate",
                decision="review_required",
                confidence=max(float(diff.confidence or 0.0), float(diff.alignment_confidence or 0.0)),
                reason=f"PDF 内容已进入差异候选：{diff.category}。",
                text=unit.text,
                next_route=self._route_for_diff(diff.category, page_routes=page_routes, page_context=page_context),
                related_diff_id=diff.diff_id,
                flags=[*flags, f"diff_category={diff.category}"],
            )
        if link and link.status == "mapping_uncertain":
            return self._review(
                reviews,
                side="pdf",
                unit_id=unit.unit_id,
                page_no=unit.page_no,
                status="mapping_uncertain",
                decision="review_required",
                confidence=link.confidence,
                reason="PDF 内容只找到低置信 DOCX 映射，不能认为已经审查通过。",
                text=unit.text,
                next_route="needs_human_mapping_review",
                related_link_id=link.link_id,
                flags=flags,
            )
        page_text_coverage = self._pdf_page_text_coverage(
            unit=unit,
            page_routes=page_routes,
            docx_page_texts=docx_page_texts,
        )
        if page_text_coverage:
            return self._review(
                reviews,
                side="pdf",
                unit_id=unit.unit_id,
                page_no=unit.page_no,
                status=str(page_text_coverage.get("status") or "covered_by_docx_page_text"),
                decision="covered",
                confidence=float(page_text_coverage.get("confidence") or 0.76),
                reason=str(page_text_coverage.get("reason") or "PDF 文本行在同页 DOCX 页级文本中保守命中，视为已覆盖。"),
                text=unit.text,
                flags=[*flags, "covered_by_docx_page_text"],
            )
        if unit.unit_type == "table_region" or "needs_table_parser" in page_routes:
            return self._review(
                reviews,
                side="pdf",
                unit_id=unit.unit_id,
                page_no=unit.page_no,
                status="table_pending",
                decision="review_required",
                confidence=float(unit.confidence or 0.52),
                reason="PDF 表格/类表格区域尚未建立 DOCX 单元格级覆盖。",
                text=unit.text,
                next_route="needs_table_parser",
                flags=flags,
            )
        if unit.unit_type == "visual_region":
            route = self._route_for_uncovered_pdf(unit=unit, page_context=page_context, page_routes=page_routes)
            return self._review(
                reviews,
                side="pdf",
                unit_id=unit.unit_id,
                page_no=unit.page_no,
                status="visual_pending",
                decision="review_required",
                confidence=float(unit.confidence or 0.5),
                reason=self._reason_for_uncovered_pdf(route=route, page_context=page_context),
                text=unit.text,
                next_route=route,
                flags=flags,
            )
        route = self._route_for_uncovered_pdf(unit=unit, page_context=page_context, page_routes=page_routes)
        return self._review(
            reviews,
            side="pdf",
            unit_id=unit.unit_id,
            page_no=unit.page_no,
            status=self._status_for_route(route=route, side="pdf", fallback="uncovered_pdf_content"),
            decision="review_required",
            confidence=0.56,
            reason=self._reason_for_uncovered_pdf(route=route, page_context=page_context),
            text=unit.text,
            next_route=route,
            flags=flags,
        )

    def _docx_unit_in_scope(self, unit: DocxEvidenceUnit) -> bool:
        text_chars = len(unit.normalized_text or "")
        if unit.container_type == "table_cell":
            return text_chars >= 1
        return text_chars >= self.min_text_chars

    def _pdf_unit_in_scope(self, unit: PdfEvidenceUnit) -> bool:
        if unit.unit_type in {"visual_region", "table_region"}:
            return bool(unit.bbox)
        return len(unit.normalized_text or "") >= self.min_text_chars

    def _diffs_by_docx(self, diffs: Sequence[ConversionDiffCandidate]) -> Dict[str, ConversionDiffCandidate]:
        rows = [item for item in diffs if item.docx_unit_id]
        rows.sort(key=lambda item: self._diff_priority(item.category))
        return {item.docx_unit_id: item for item in rows}

    def _diffs_by_pdf(self, diffs: Sequence[ConversionDiffCandidate]) -> Dict[str, ConversionDiffCandidate]:
        rows = [item for item in diffs if item.pdf_unit_id]
        rows.sort(key=lambda item: self._diff_priority(item.category))
        return {item.pdf_unit_id: item for item in rows}

    def _page_routes(self, preflight_result: ConversionPreflightResult) -> Dict[int, set[str]]:
        rows: Dict[int, set[str]] = {}
        for route in preflight_result.review_routes:
            if route.page_no:
                rows.setdefault(int(route.page_no), set()).add(route.route)
        for page_key, profile in preflight_result.page_profiles.items():
            try:
                page_no = int(page_key)
            except Exception:
                continue
            primary = str(profile.get("primary_route") or "")
            secondary = {str(item) for item in profile.get("secondary_routes") or [] if str(item)}
            page_routes = rows.setdefault(page_no, set())
            if primary == "native_text_compare":
                page_routes.add("text_compare")
            elif primary in {"image_table_cell_compare", "native_table_compare"}:
                page_routes.add("needs_table_parser")
            elif primary in {"image_text_compare", "image_form_field_compare", "mixed_region_compare"}:
                page_routes.add("needs_region_ocr")
            page_routes.update(secondary)
        return rows

    def _page_contexts(self, *, preflight_result: ConversionPreflightResult, page_routes: Dict[int, set[str]]) -> Dict[int, Dict[str, str]]:
        rows: Dict[int, Dict[str, str]] = {}
        for page_key, profile in preflight_result.page_profiles.items():
            try:
                page_no = int(page_key)
            except Exception:
                continue
            canonical = str(profile.get("audit_canonical_page_type") or "")
            if not canonical:
                labels = set(profile.get("labels") or [])
                if profile.get("needs_table_parser") or "table_heavy" in labels:
                    canonical = "table_image_page" if profile.get("needs_ocr") else "native_table_page"
                elif profile.get("needs_ocr") or labels & {"scan_like", "image_text_heavy"}:
                    canonical = "scan_text_page"
                elif profile.get("native_text_reliable") or "simple_native_text" in labels:
                    canonical = "native_text_page"
                else:
                    canonical = "low_confidence_page"
            routes = page_routes.get(page_no, set())
            rows[page_no] = {
                "canonical_page_type": canonical,
                "pdf_source_type": str(profile.get("pdf_source_type") or ""),
                "recognition_strategy": str(profile.get("recognition_strategy") or ""),
                "recognition_risk": str(profile.get("recognition_risk") or ""),
                "quality_first": "1" if bool(profile.get("quality_first") or canonical in {"scan_text_page", "table_image_page", "mixed_layout_page", "low_confidence_page"}) else "0",
                "has_table_route": "1" if "needs_table_parser" in routes else "0",
                "has_full_page_ocr_route": "1" if "needs_full_page_ocr" in routes else "0",
                "has_region_segmentation_route": "1" if "needs_region_segmentation" in routes else "0",
            }
        return rows

    def _comparable_pdf_pages(self, pdf_units: Sequence[PdfEvidenceUnit]) -> set[int]:
        pages: set[int] = set()
        for unit in pdf_units:
            if unit.unit_type in {"native_text_block", "anchor_ocr_line"} and len(unit.normalized_text or "") >= self.min_text_chars:
                pages.add(int(unit.page_no))
            elif unit.unit_type == "anchor_ocr_page" and len(unit.normalized_text or "") >= max(20, self.min_text_chars):
                pages.add(int(unit.page_no))
        return pages

    def _page_texts(self, pdf_units: Sequence[PdfEvidenceUnit]) -> Dict[int, str]:
        rows: Dict[int, List[str]] = {}
        useful_types = {"native_text_block", "anchor_ocr_page", "anchor_ocr_line"}
        for unit in pdf_units:
            if unit.unit_type not in useful_types:
                continue
            text = unit.normalized_text or normalize_text(unit.text)
            if len(text) < self.min_text_chars:
                continue
            rows.setdefault(int(unit.page_no), []).append(text)
        return {page_no: "\n".join(parts) for page_no, parts in rows.items()}

    def _docx_page_texts(self, docx_units: Sequence[DocxEvidenceUnit]) -> Dict[int, str]:
        rows: Dict[int, List[str]] = {}
        for unit in sorted(docx_units, key=lambda item: (int(item.estimated_page_no or 0), int(item.order_index or 0))):
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            text = unit.normalized_text or normalize_text(unit.text)
            if len(text) < self.min_text_chars:
                continue
            rows.setdefault(page_no, []).append(text)
        return {page_no: "\n".join(parts) for page_no, parts in rows.items()}

    def _covered_by_page_text(self, *, unit: DocxEvidenceUnit, page_text: str) -> bool:
        if not page_text:
            return False
        value = self._semantic_compact(unit.normalized_text or unit.text)
        page_value = self._semantic_compact(page_text)
        if not value or not page_value:
            return False
        if unit.container_type == "table_cell":
            return self._table_cell_covered_by_page_text(value=value, page_value=page_value)
        if len(value) >= max(4, self.min_text_chars) and value in page_value:
            return True
        return False

    def _page_text_coverage(self, *, unit: DocxEvidenceUnit, page_no: int, page_texts: Dict[int, str]) -> Dict[str, object]:
        if page_no <= 0:
            return {}
        if self._covered_by_page_text(unit=unit, page_text=page_texts.get(page_no, "")):
            return {
                "status": "covered_by_page_ocr",
                "page_no": page_no,
                "confidence": 0.74 if unit.container_type == "table_cell" else 0.78,
                "reason": "DOCX 内容在同页 PDF 页级 OCR 文本中保守命中，视为已覆盖。",
            }
        if unit.container_type != "table_cell":
            return {}
        for nearby_page_no in self._nearby_page_order(page_no=page_no, page_texts=page_texts, window=3):
            if self._covered_by_page_text(unit=unit, page_text=page_texts.get(nearby_page_no, "")):
                return {
                    "status": "covered_by_nearby_page_ocr",
                    "page_no": nearby_page_no,
                    "confidence": 0.68,
                    "reason": f"DOCX 表格单元未命中估算页，但在相邻 PDF 第 {nearby_page_no} 页页级 OCR 中保守命中；按页映射偏移记录，不作为字段错误。",
                }
        return {}

    def _pdf_page_text_coverage(
        self,
        *,
        unit: PdfEvidenceUnit,
        page_routes: set[str],
        docx_page_texts: Dict[int, str],
    ) -> Dict[str, object]:
        page_no = int(unit.page_no or 0)
        if page_no <= 0 or "needs_table_parser" in page_routes:
            return {}
        if unit.unit_type not in {"native_text_block", "anchor_ocr_line"}:
            return {}
        value = self._semantic_compact(unit.normalized_text or unit.text)
        if len(value) < max(6, self.min_text_chars):
            return {}
        docx_page = self._semantic_compact(docx_page_texts.get(page_no, ""))
        if not docx_page or value not in docx_page:
            return {}
        return {
            "status": "covered_by_docx_page_text",
            "confidence": 0.78 if unit.unit_type == "native_text_block" else 0.74,
            "reason": "PDF 文本行在同页 DOCX 页级文本中保守命中，视为已覆盖；不再作为漏转候选。",
        }

    def _nearby_page_order(self, *, page_no: int, page_texts: Dict[int, str], window: int) -> List[int]:
        rows: List[int] = []
        for distance in range(1, max(0, int(window)) + 1):
            for candidate in (page_no - distance, page_no + distance):
                if candidate > 0 and candidate in page_texts:
                    rows.append(candidate)
        return rows

    def _table_cell_covered_by_page_text(self, *, value: str, page_value: str) -> bool:
        if len(value) >= 4 and value in page_value:
            return True
        if re.search(r"[a-zA-Z]", value):
            return False
        tokens = [item for item in re.findall(r"\d+", value) if len(item) >= 3]
        if not tokens:
            return False
        return all(item in page_value for item in tokens)

    def _semantic_compact(self, text: str) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _route_for_diff(self, category: str, *, page_routes: set[str], page_context: Dict[str, str] | None = None) -> str:
        if category in {"table_structure_suspect", "table_cell_mismatch_suspect"} or "needs_table_parser" in page_routes:
            return "needs_table_parser"
        page_context = page_context or {}
        if category in {"mapping_uncertain", "missing_content", "extra_content", "duplicate_content", "reading_order_changed"}:
            if category in {"missing_content", "extra_content"}:
                return self._route_for_uncovered_docx(page_context=page_context, page_routes=page_routes)
            return "needs_human_mapping_review"
        if category in {"text_substitution", "visual_region_unresolved"}:
            if str(page_context.get("canonical_page_type") or "") == "scan_text_page":
                return "needs_full_page_ocr"
            if str(page_context.get("canonical_page_type") or "") == "mixed_layout_page":
                return "needs_region_segmentation"
            if str(page_context.get("canonical_page_type") or "") == "low_confidence_page":
                return "needs_qwen_vl_page_gate"
            return "needs_qwen_vl"
        if category in {"unlocated_hard_field", "page_coverage_gap"}:
            return "needs_recall_guard_review"
        return "text_compare"

    def _coverage_flags(self, flags: Sequence[str], *, page_context: Dict[str, str]) -> List[str]:
        rows = list(flags)
        for key in ("canonical_page_type", "pdf_source_type", "recognition_strategy", "recognition_risk"):
            value = str(page_context.get(key) or "")
            if value:
                rows.append(f"{key}={value}")
        if page_context.get("quality_first") == "1":
            rows.append("quality_first_page")
        return rows

    def _route_for_uncovered_docx(self, *, page_context: Dict[str, str], page_routes: set[str]) -> str:
        canonical = str(page_context.get("canonical_page_type") or "")
        if "needs_table_parser" in page_routes or canonical in {"table_image_page", "native_table_page"}:
            return "needs_table_parser"
        if canonical == "scan_text_page":
            return "needs_full_page_ocr"
        if canonical == "mixed_layout_page":
            return "needs_region_segmentation"
        if canonical == "low_confidence_page":
            return "needs_qwen_vl_page_gate"
        if canonical == "native_text_page":
            return "needs_text_alignment"
        return "needs_human_mapping_review"

    def _route_for_uncovered_pdf(self, *, unit: PdfEvidenceUnit, page_context: Dict[str, str], page_routes: set[str]) -> str:
        canonical = str(page_context.get("canonical_page_type") or "")
        if unit.unit_type == "table_region" or "needs_table_parser" in page_routes or canonical in {"table_image_page", "native_table_page"}:
            return "needs_table_parser"
        if canonical == "scan_text_page":
            return "needs_full_page_ocr"
        if canonical == "mixed_layout_page":
            return "needs_region_segmentation"
        if canonical == "low_confidence_page":
            return "needs_qwen_vl_page_gate"
        if unit.unit_type == "visual_region":
            return "needs_region_ocr"
        if canonical == "native_text_page":
            return "needs_text_alignment"
        return "needs_human_mapping_review"

    def _status_for_route(self, *, route: str, side: str, fallback: str) -> str:
        if route == "needs_full_page_ocr":
            return "needs_full_page_ocr"
        if route == "needs_region_segmentation":
            return "needs_region_segmentation"
        if route == "needs_qwen_vl_page_gate":
            return "low_confidence_page_review"
        if route == "needs_text_alignment":
            return "needs_text_alignment"
        if route == "needs_table_parser":
            return "table_pending"
        if side == "docx" and route == "needs_region_ocr":
            return "needs_pdf_ocr"
        return fallback

    def _reason_for_uncovered_docx(self, *, route: str, page_context: Dict[str, str]) -> str:
        canonical = str(page_context.get("canonical_page_type") or "unknown")
        if route == "needs_full_page_ocr":
            return f"该页属于 {canonical}，PDF 侧需要整页 OCR 后再做 DOCX 段落覆盖对齐，当前不能视为已审查通过。"
        if route == "needs_table_parser":
            return f"该页属于 {canonical}，DOCX 表格/类表格内容尚未建立单元格级 PDF 覆盖。"
        if route == "needs_region_segmentation":
            return f"该页属于 {canonical}，需要先做区域切分和局部 OCR，再判断 DOCX 内容是否多识别或错位。"
        if route == "needs_qwen_vl_page_gate":
            return f"该页属于 {canonical}，现有 PDF 文本证据低置信，需要页级视觉门槛复核后再判断。"
        if route == "needs_text_alignment":
            return "普通 DOCX 内容单元未找到可靠 PDF 覆盖，需要文本级对齐复核，不能直接放行。"
        return "普通 DOCX 内容单元未找到可靠 PDF 覆盖，不能直接放行。"

    def _reason_for_uncovered_pdf(self, *, route: str, page_context: Dict[str, str]) -> str:
        canonical = str(page_context.get("canonical_page_type") or "unknown")
        if route == "needs_full_page_ocr":
            return f"该页属于 {canonical}，PDF 可见区域需要整页 OCR 和文本对齐后确认是否漏转。"
        if route == "needs_table_parser":
            return f"该页属于 {canonical}，PDF 表格/类表格区域尚未建立 DOCX 单元格级覆盖。"
        if route == "needs_region_segmentation":
            return f"该页属于 {canonical}，PDF 可见区域需要先切分再 OCR，当前不能判断 DOCX 是否覆盖。"
        if route == "needs_qwen_vl_page_gate":
            return f"该页属于 {canonical}，页面证据低置信，需要页级视觉门槛复核。"
        if route == "needs_text_alignment":
            return "PDF 文本单元未找到可靠 DOCX 覆盖，需要文本级对齐复核，可能存在漏转。"
        return "PDF 可见区域尚未完成文本化和 DOCX 覆盖核查。"

    def _diff_priority(self, category: str) -> int:
        order = {
            "critical_field_changed": 0,
            "text_substitution": 1,
            "table_cell_mismatch_suspect": 2,
            "missing_content": 3,
            "extra_content": 4,
            "mapping_uncertain": 5,
            "unlocated_hard_field": 6,
            "page_coverage_gap": 7,
        }
        return order.get(category, 99)

    def _review(
        self,
        reviews: Sequence[ContentCoverageReview],
        *,
        side: str,
        unit_id: str,
        page_no: int | None,
        status: str,
        decision: str,
        confidence: float,
        reason: str,
        text: str,
        next_route: str = "",
        related_link_id: str = "",
        related_diff_id: str = "",
        flags: Sequence[str] = (),
    ) -> ContentCoverageReview:
        return ContentCoverageReview(
            review_id=f"coverage_{len(reviews)+1:04d}",
            side=side,
            unit_id=unit_id,
            page_no=page_no,
            status=status,
            decision=decision,
            confidence=confidence,
            reason=reason,
            text=text,
            next_route=next_route,
            related_link_id=related_link_id,
            related_diff_id=related_diff_id,
            flags=list(dict.fromkeys(str(item) for item in flags if str(item).strip())),
        )
