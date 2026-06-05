from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.core.config import settings

from .common import normalize_text
from .models import ContentCoverageReview, ConversionPreflightResult, DocxEvidenceUnit, ImagePdfPageReview


class ImagePdfPageReviewBuilder:
    """Page-level review for scan/photo/image PDF pages.

    WPS conversion errors on image PDFs often do not first appear as a clean
    field mismatch. They appear as weak OCR, scattered DOCX fragments, missing
    page coverage, or table-like blocks without stable cell mapping. This
    builder makes that page risk explicit and commentable.
    """

    IMAGE_LABELS = {"scan_like", "image_text_heavy"}

    def __init__(self, *, enabled: bool | None = None, max_pages: int | None = None) -> None:
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_PAGE_REVIEW_ENABLED", True) if enabled is None else enabled)
        self.max_pages = max(0, int(max_pages if max_pages is not None else getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_PAGE_REVIEW_MAX_PAGES", 9999) or 9999))

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[ImagePdfPageReview]:
        if not self.enabled or self.max_pages <= 0:
            return []
        docx_by_page = self._docx_by_page(preflight_result.docx_units)
        coverage_by_page = self._coverage_by_page(preflight_result.content_coverage_reviews)
        anchor_ocr_by_page = self._anchor_ocr_by_page(preflight_result.anchor_ocr)
        fragment_counts = self._fragment_counts(preflight_result)

        reviews: List[ImagePdfPageReview] = []
        for page_no in range(1, int(preflight_result.pdf_page_count or 0) + 1):
            profile = preflight_result.page_profiles.get(str(page_no)) or {}
            labels = set(profile.get("labels") or [])
            if not self._is_image_pdf_page(profile=profile, labels=labels):
                continue
            review = self._review_page(
                page_no=page_no,
                profile=profile,
                labels=labels,
                docx_units=docx_by_page.get(page_no, []),
                coverage_reviews=coverage_by_page.get(page_no, []),
                anchor_ocr=anchor_ocr_by_page.get(page_no, {}),
                fragment_anomaly_count=fragment_counts.get(page_no, 0),
            )
            if review.risk_level == "low" and review.unresolved_count < 8:
                continue
            reviews.append(review)

        reviews.sort(key=lambda item: (-self._risk_sort(item.risk_level), -int(item.unresolved_count or 0), item.page_no))
        limited = reviews[: self.max_pages]
        for index, review in enumerate(limited, start=1):
            review.review_id = f"image_page_{index:04d}"
        return limited

    def _review_page(
        self,
        *,
        page_no: int,
        profile: Dict[str, Any],
        labels: set[str],
        docx_units: Sequence[DocxEvidenceUnit],
        coverage_reviews: Sequence[ContentCoverageReview],
        anchor_ocr: Dict[str, Any],
        fragment_anomaly_count: int,
    ) -> ImagePdfPageReview:
        status_counts = self._status_counts(coverage_reviews)
        unresolved = sum(1 for item in coverage_reviews if item.decision != "covered")
        ocr_quality = str(anchor_ocr.get("quality") or profile.get("anchor_ocr_quality") or "").lower()
        ocr_confidence = float(anchor_ocr.get("confidence") or 0.0)
        ocr_text_chars = int(anchor_ocr.get("text_chars") or profile.get("anchor_ocr_text_chars") or 0)
        ocr_line_count = int(anchor_ocr.get("line_count") or profile.get("anchor_ocr_line_count") or 0)
        ocr_anchor_count = int(anchor_ocr.get("anchor_count") or profile.get("anchor_ocr_anchor_count") or 0)
        table_like = bool(profile.get("table_like") or "table_heavy" in labels)
        page_kind = self._page_kind(profile=profile, labels=labels, table_like=table_like)
        risk_level = self._risk_level(
            labels=labels,
            unresolved=unresolved,
            ocr_quality=ocr_quality,
            ocr_text_chars=ocr_text_chars,
            fragment_anomaly_count=fragment_anomaly_count,
            table_like=table_like,
            status_counts=status_counts,
        )
        verdict = {
            "high": "image_pdf_page_high_risk",
            "medium": "image_pdf_page_needs_review",
            "low": "image_pdf_page_trace_only",
        }.get(risk_level, "image_pdf_page_needs_review")
        anchor_unit = self._anchor_unit(docx_units)
        confidence = self._confidence(
            risk_level=risk_level,
            unresolved=unresolved,
            ocr_quality=ocr_quality,
            fragment_anomaly_count=fragment_anomaly_count,
            table_like=table_like,
        )
        reason = self._reason(
            page_no=page_no,
            page_kind=page_kind,
            risk_level=risk_level,
            unresolved=unresolved,
            ocr_quality=ocr_quality,
            ocr_text_chars=ocr_text_chars,
            fragment_anomaly_count=fragment_anomaly_count,
            table_like=table_like,
        )
        flags = ["image_pdf_page_review", f"page_kind={page_kind}", f"risk={risk_level}"]
        flags.extend(f"page_label={label}" for label in sorted(labels))
        if ocr_quality in {"low", "failed", ""} or ocr_text_chars < 80:
            flags.append("weak_page_ocr")
        if unresolved >= 25:
            flags.append("large_coverage_gap")
        if fragment_anomaly_count:
            flags.append("has_fragment_anomaly")
        if table_like:
            flags.append("table_like_image_page")
        return ImagePdfPageReview(
            review_id="image_page_pending",
            page_no=page_no,
            page_kind=page_kind,
            verdict=verdict,
            risk_level=risk_level,
            confidence=confidence,
            anchor_unit_id=anchor_unit.unit_id if anchor_unit else "",
            reason=reason,
            labels=sorted(labels),
            native_text_reliable=bool(profile.get("native_text_reliable")),
            image_area_ratio=float(profile.get("image_area_ratio") or 0.0),
            dark_pixel_ratio=float(profile.get("dark_pixel_ratio") or 0.0),
            ocr_quality=ocr_quality,
            ocr_confidence=ocr_confidence,
            ocr_text_chars=ocr_text_chars,
            ocr_line_count=ocr_line_count,
            ocr_anchor_count=ocr_anchor_count,
            docx_unit_count=len(docx_units),
            docx_text_chars=sum(len(unit.normalized_text or normalize_text(unit.text)) for unit in docx_units),
            unresolved_count=unresolved,
            coverage_status_counts=status_counts,
            fragment_anomaly_count=fragment_anomaly_count,
            table_like=table_like,
            docx_samples=self._docx_samples(docx_units),
            pdf_ocr_excerpt=self._clip(anchor_ocr.get("text") or "", 420),
            next_route="needs_full_page_review" if risk_level != "low" else "trace_only",
            flags=flags,
        )

    def _is_image_pdf_page(self, *, profile: Dict[str, Any], labels: set[str]) -> bool:
        canonical = str(profile.get("audit_canonical_page_type") or "")
        if canonical in {"scan_text_page", "table_image_page", "mixed_layout_page", "low_confidence_page"}:
            return True
        if canonical in {"native_text_page", "native_table_page"}:
            return False
        if labels & self.IMAGE_LABELS:
            return True
        if bool(profile.get("needs_ocr")) and float(profile.get("dark_pixel_ratio") or 0.0) > 0.002:
            return True
        return bool(not profile.get("native_text_reliable") and float(profile.get("image_area_ratio") or 0.0) >= 0.2)

    def _page_kind(self, *, profile: Dict[str, Any], labels: set[str], table_like: bool) -> str:
        canonical = str(profile.get("audit_canonical_page_type") or "")
        if canonical == "table_image_page":
            return "table_image_pdf"
        if canonical == "scan_text_page":
            return "scan_text_pdf"
        if canonical == "mixed_layout_page":
            return "mixed_layout_image_pdf"
        if canonical == "low_confidence_page":
            return "low_confidence_image_pdf"
        if table_like and not profile.get("native_text_reliable"):
            return "table_image_pdf"
        if "scan_like" in labels and "image_text_heavy" in labels:
            return "scan_or_photo_pdf"
        if "scan_like" in labels:
            return "scan_like_pdf"
        if "image_text_heavy" in labels:
            return "image_text_pdf"
        return "image_pdf"

    def _risk_level(
        self,
        *,
        labels: set[str],
        unresolved: int,
        ocr_quality: str,
        ocr_text_chars: int,
        fragment_anomaly_count: int,
        table_like: bool,
        status_counts: Dict[str, int],
    ) -> str:
        visual_unresolved = status_counts.get("pdf/visual_pending", 0) + status_counts.get("docx/needs_pdf_ocr", 0)
        table_unresolved = status_counts.get("docx/table_pending", 0) + status_counts.get("pdf/table_pending", 0)
        mapping_uncertain = status_counts.get("docx/mapping_uncertain", 0) + status_counts.get("pdf/mapping_uncertain", 0)
        weak_ocr = ocr_quality in {"", "failed", "low"} or ocr_text_chars < 60
        if fragment_anomaly_count or unresolved >= 40 or (table_like and table_unresolved >= 20) or mapping_uncertain >= 20:
            return "high"
        if weak_ocr and unresolved >= 8:
            return "high"
        if unresolved >= 15 or table_unresolved >= 10 or visual_unresolved >= 8:
            return "medium"
        if labels & self.IMAGE_LABELS:
            return "medium" if weak_ocr else "low"
        return "low"

    def _confidence(
        self,
        *,
        risk_level: str,
        unresolved: int,
        ocr_quality: str,
        fragment_anomaly_count: int,
        table_like: bool,
    ) -> float:
        score = {"high": 0.68, "medium": 0.56, "low": 0.42}.get(risk_level, 0.52)
        score += min(0.08, max(0, unresolved - 10) * 0.002)
        if fragment_anomaly_count:
            score += 0.08
        if table_like:
            score += 0.03
        if ocr_quality in {"high", "medium"}:
            score += 0.03
        return round(min(0.86, score), 4)

    def _reason(
        self,
        *,
        page_no: int,
        page_kind: str,
        risk_level: str,
        unresolved: int,
        ocr_quality: str,
        ocr_text_chars: int,
        fragment_anomaly_count: int,
        table_like: bool,
    ) -> str:
        parts = [
            f"第 {page_no} 页属于 {page_kind}，风险等级 {risk_level}。",
            f"页级 OCR 质量 {ocr_quality or 'unknown'}，OCR 文本 {ocr_text_chars} 字；未可靠覆盖内容单元 {unresolved} 个。",
        ]
        if fragment_anomaly_count:
            parts.append("该页还存在扫描页碎片/重复异常，WPS 可能将同一区域拆碎或重复识别。")
        if table_like:
            parts.append("该页包含表格/网格结构，图片型 PDF 的表格单元最容易发生错位、漏行或数字误识别。")
        parts.append("图片型 PDF 不能按普通文字 PDF 的通过标准处理，需要整页核查可见内容与 DOCX 是否一致。")
        return " ".join(parts)

    def _docx_by_page(self, units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[DocxEvidenceUnit]]:
        rows: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            if not str(unit.text or "").strip():
                continue
            rows.setdefault(page_no, []).append(unit)
        for page_no, page_units in rows.items():
            rows[page_no] = sorted(page_units, key=lambda item: item.order_index)
        return rows

    def _coverage_by_page(self, reviews: Sequence[ContentCoverageReview]) -> Dict[int, List[ContentCoverageReview]]:
        rows: Dict[int, List[ContentCoverageReview]] = {}
        for review in reviews:
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            rows.setdefault(page_no, []).append(review)
        return rows

    def _anchor_ocr_by_page(self, anchor_ocr: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for page in anchor_ocr.get("pages") or []:
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            if page_no <= 0:
                continue
            rows[page_no] = dict(page)
        return rows

    def _fragment_counts(self, preflight_result: ConversionPreflightResult) -> Dict[int, int]:
        rows: Dict[int, int] = {}
        for review in preflight_result.fragment_anomaly_reviews:
            page_no = int(review.page_no or 0)
            if page_no > 0:
                rows[page_no] = rows.get(page_no, 0) + 1
        return rows

    def _status_counts(self, reviews: Sequence[ContentCoverageReview]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for review in reviews:
            if review.decision == "covered":
                continue
            key = f"{review.side}/{review.status}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _anchor_unit(self, units: Sequence[DocxEvidenceUnit]) -> DocxEvidenceUnit | None:
        candidates = [unit for unit in units if str(unit.text or "").strip()]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                0 if item.container_type == "paragraph" else 1,
                -min(len(item.normalized_text or normalize_text(item.text)), 120),
                item.order_index,
            )
        )
        return candidates[0]

    def _docx_samples(self, units: Sequence[DocxEvidenceUnit]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        candidates = sorted(units, key=lambda item: (0 if item.container_type == "paragraph" else 1, item.order_index))
        for unit in candidates:
            text = " ".join(str(unit.text or "").split())
            if not text:
                continue
            compact = normalize_text(text)
            if compact in seen:
                continue
            seen.add(compact)
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "container_type": unit.container_type,
                    "text": self._clip(text, 90),
                    "order_index": int(unit.order_index or 0),
                }
            )
            if len(rows) >= 10:
                break
        return rows

    def _risk_sort(self, risk_level: str) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(risk_level, 0)

    def _clip(self, text: Any, limit: int) -> str:
        value = " ".join(str(text or "").split())
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "…"
