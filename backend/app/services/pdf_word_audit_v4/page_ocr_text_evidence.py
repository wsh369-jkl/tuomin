from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.core.config import settings

from .common import has_high_value_field_content, looks_like_document_title, looks_like_organization_name, looks_like_table_title, normalize_text
from .models import ConversionPreflightResult, PageOcrTextEvidenceReview


class PageOcrTextEvidenceBuilder:
    """Build exact text replacement evidence from page OCR and DOCX artifacts.

    This layer is deliberately lightweight: it reuses the existing page OCR
    text, applies conservative DOCX artifact normalizers, and records whether
    the proposed replacement is actually supported by same-page PDF OCR.
    """

    TARGET_STATUSES = {
        "covered",
        "covered_by_page_ocr",
        "covered_by_nearby_page_ocr",
        "covered_by_backfill_ocr",
        "covered_by_backfill_ocr_fuzzy",
        "uncovered_docx_content",
        "mapping_uncertain",
        "needs_pdf_ocr",
        "diff_candidate",
        "table_pending",
    }

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        max_reviews: Optional[int] = None,
    ) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_OCR_TEXT_EVIDENCE_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_reviews = max(
            0,
            int(max_reviews if max_reviews is not None else getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_OCR_TEXT_EVIDENCE_MAX_REVIEWS", 9999) or 9999),
        )

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        text_hint_fn: Callable[..., str],
        priority_fn: Callable[..., int],
        context_terms: Sequence[str],
    ) -> List[PageOcrTextEvidenceReview]:
        if not self.enabled or self.max_reviews <= 0:
            return []
        docx_by_unit = {unit.unit_id: unit for unit in preflight_result.docx_units}
        image_pages = self._image_pages(preflight_result=preflight_result)
        page_ocr_texts = self._page_ocr_texts(preflight_result=preflight_result)
        rows: List[PageOcrTextEvidenceReview] = []
        seen_units: set[str] = set()

        candidates = [
            review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx"
            and review.status in self.TARGET_STATUSES
            and int(review.page_no or 0) in image_pages
            and review.unit_id not in seen_units
        ]
        candidates.sort(key=lambda item: (int(item.page_no or 9999), self._coverage_priority(item.status), item.review_id))
        for review in candidates:
            if len(rows) >= self.max_reviews:
                break
            unit = docx_by_unit.get(review.unit_id)
            if unit is None or unit.container_type == "table_cell":
                continue
            hint = self._call_text_hint(text_hint_fn=text_hint_fn, text=review.text, context_terms=context_terms)
            suggested_text = self._clean_suggestion(hint)
            if not self._valid_replacement(old_text=review.text, new_text=suggested_text):
                continue
            priority = self._call_priority(priority_fn=priority_fn, text=review.text, hint=hint)
            if priority > 3:
                continue
            page_no = int(review.page_no or 0)
            ocr_text = page_ocr_texts.get(page_no, "")
            score = self._ocr_support_score(suggested_text=suggested_text, current_text=review.text, page_ocr_text=ocr_text)
            decision = "confirmed_error" if score >= 0.78 else "review_required"
            status = "page_ocr_supported_replacement" if decision == "confirmed_error" else "rule_suggestion_unverified_by_page_ocr"
            flags = [
                "page_ocr_text_evidence",
                f"coverage_status={review.status}",
                f"artifact_priority={priority}",
            ]
            if decision == "confirmed_error":
                flags.append("page_ocr_supports_suggestion")
            else:
                flags.append("page_ocr_support_missing")
            rows.append(
                PageOcrTextEvidenceReview(
                    review_id=f"page_ocr_text_ev_{len(rows) + 1:04d}",
                    page_no=page_no,
                    docx_unit_id=review.unit_id,
                    coverage_review_id=review.review_id,
                    docx_text=review.text,
                    suggested_text=suggested_text,
                    status=status,
                    decision=decision,
                    confidence=self._confidence(priority=priority, score=score, fallback=float(review.confidence or 0.0)),
                    issue_type=self._issue_type(old_text=review.text, new_text=suggested_text),
                    reason=self._reason(decision=decision, priority=priority, score=score),
                    ocr_text_excerpt=self._ocr_excerpt(suggested_text=suggested_text, page_ocr_text=ocr_text),
                    normalized_docx=self._semantic_compact(review.text),
                    normalized_suggestion=self._semantic_compact(suggested_text),
                    ocr_support_score=score,
                    flags=flags,
                )
            )
            seen_units.add(review.unit_id)
        return rows

    def _image_pages(self, *, preflight_result: ConversionPreflightResult) -> set[int]:
        pages = {
            int(review.page_no or 0)
            for review in preflight_result.image_page_reviews
            if int(review.page_no or 0) > 0 and review.risk_level in {"high", "medium"}
        }
        for page_key, profile in preflight_result.page_profiles.items():
            try:
                page_no = int(page_key)
            except Exception:
                continue
            labels = set(profile.get("labels") or [])
            if labels & {"scan_like", "image_text_heavy"} or bool(profile.get("needs_ocr")):
                pages.add(page_no)
        return pages

    def _page_ocr_texts(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, str]:
        rows: Dict[int, List[str]] = {}
        for page in preflight_result.anchor_ocr.get("pages") or []:
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            text = str(page.get("text") or "")
            if page_no > 0 and text.strip():
                rows.setdefault(page_no, []).append(text)
        for unit in preflight_result.pdf_units:
            unit_type = str(getattr(unit, "unit_type", "") or "")
            if unit_type not in {"anchor_ocr_page", "anchor_ocr_line", "native_text_block"}:
                continue
            text = str(getattr(unit, "text", "") or "")
            page_no = int(getattr(unit, "page_no", 0) or 0)
            if page_no > 0 and text.strip():
                rows.setdefault(page_no, []).append(text)
        for review in preflight_result.content_coverage_backfills:
            if not review.available:
                continue
            page_no = int(review.page_no or 0)
            text = str(review.extracted_text or review.normalized_text or "")
            if page_no > 0 and text.strip():
                rows.setdefault(page_no, []).append(text)
        return {page_no: "\n".join(parts) for page_no, parts in rows.items()}

    def _ocr_support_score(self, *, suggested_text: str, current_text: str, page_ocr_text: str) -> float:
        suggestion = self._semantic_compact(suggested_text)
        current = self._semantic_compact(current_text)
        page = self._semantic_compact(page_ocr_text)
        if len(suggestion) < 4 or not page:
            return 0.0
        if suggestion in page:
            return 0.96
        if current and current in page and len(current) >= len(suggestion) - 2:
            return 0.0
        if len(suggestion) >= 10:
            score = self._best_window_ratio(needle=suggestion, haystack=page)
            if score >= 0.88:
                return max(0.78, min(0.92, score))
        date_relaxed = suggestion.replace("日", "")
        if "年" in suggestion and "月" in suggestion and date_relaxed and date_relaxed in page:
            return 0.82
        return 0.0

    def _best_window_ratio(self, *, needle: str, haystack: str) -> float:
        if not needle or not haystack:
            return 0.0
        size = len(needle)
        if len(haystack) <= size + 8:
            return SequenceMatcher(None, needle, haystack).ratio()
        step = max(1, size // 4)
        best = 0.0
        for start in range(0, max(1, len(haystack) - size + 1), step):
            window = haystack[start : start + size + 8]
            best = max(best, SequenceMatcher(None, needle, window).ratio())
            if best >= 0.94:
                break
        return best

    def _confidence(self, *, priority: int, score: float, fallback: float) -> float:
        if score >= 0.78:
            return round(max(fallback, 0.7, min(0.9, 0.64 + score * 0.24 - priority * 0.015)), 4)
        return round(max(0.5, min(0.68, fallback)), 4)

    def _issue_type(self, *, old_text: str, new_text: str) -> str:
        old_value = str(old_text or "")
        new_value = str(new_text or "")
        if re.search(r"\d\s+\d.*年|\d{4}年.*\d{2,3}.*时", old_value) or re.search(r"\d{4}年\d{1,2}月\d{1,2}日", new_value):
            return "date_or_digit_spacing_artifact"
        if any(token in old_value + new_value for token in ("身份证", "身份号码", "住址", "地址", "账号", "账户", "电话", "手机号")):
            return "identity_or_address_text_artifact"
        compact = normalize_text(new_value)
        if looks_like_document_title(compact) or looks_like_organization_name(compact) or looks_like_table_title(compact):
            return "title_or_organization_text_artifact"
        if has_high_value_field_content(old_value) or has_high_value_field_content(new_value):
            return "high_value_field_text_artifact"
        return "body_text_artifact"

    def _reason(self, *, decision: str, priority: int, score: float) -> str:
        if decision == "confirmed_error":
            return f"DOCX 正文疑似 OCR 污染，规则建议值在同页 PDF OCR 中得到支持；支持分 {score:.2f}。"
        return f"DOCX 正文疑似 OCR 污染，但同页 PDF OCR 尚未充分支持建议值；优先级 {priority}，仅进入报告。"

    def _ocr_excerpt(self, *, suggested_text: str, page_ocr_text: str) -> str:
        text = " ".join(str(page_ocr_text or "").split())
        if not text:
            return ""
        compact_suggestion = self._semantic_compact(suggested_text)
        compact_text = self._semantic_compact(text)
        index = compact_text.find(compact_suggestion)
        if index < 0:
            return text[:260]
        return text[max(0, index - 80) : index + len(suggested_text) + 140][:260]

    def _call_text_hint(self, *, text_hint_fn: Callable[..., str], text: str, context_terms: Sequence[str]) -> str:
        try:
            return str(text_hint_fn(text, context_terms=context_terms) or "")
        except TypeError:
            return str(text_hint_fn(text) or "")

    def _call_priority(self, *, priority_fn: Callable[..., int], text: str, hint: str) -> int:
        try:
            value = priority_fn(text=text, hint=hint)
        except TypeError:
            value = priority_fn(text, hint)
        if value is None or value == "":
            return 5
        return int(value)

    def _clean_suggestion(self, value: Any) -> str:
        text = " ".join(str(value or "").split()).strip()
        for prefix in ("疑似应核对为：", "建议核对为：", "建议改为：", "疑似存在前缀污染字符，建议核对："):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        return self._clean_cjk_digit_spacing(text)

    def _clean_cjk_digit_spacing(self, text: str) -> str:
        value = str(text or "")
        value = re.sub(r"\s+([，。；：、）)])", r"\1", value)
        value = re.sub(r"(?<=[，。；：、])\s+(?=[\u4e00-\u9fff\d])", "", value)
        value = re.sub(r"([（(])\s+", r"\1", value)
        value = re.sub(r"(?<=[\u4e00-\u9fff\d])\s+(?=[\u4e00-\u9fff\d])", "", value)
        return " ".join(value.split()).strip()

    def _valid_replacement(self, *, old_text: str, new_text: str) -> bool:
        old_value = " ".join(str(old_text or "").split()).strip()
        new_value = " ".join(str(new_text or "").split()).strip()
        if not old_value or not new_value or old_value == new_value:
            return False
        if self._semantic_compact(old_value) == self._semantic_compact(new_value) and " " not in old_value:
            return False
        if len(new_value) > 180:
            return False
        if any(marker in new_value for marker in ("?", "？", "_", "看不清", "无法", "不确定", "参考", "仍需", "核对", "疑似")):
            return False
        return True

    def _semantic_compact(self, text: Any) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _coverage_priority(self, status: str) -> int:
        order = {
            "diff_candidate": 0,
            "uncovered_docx_content": 1,
            "mapping_uncertain": 2,
            "needs_pdf_ocr": 3,
            "covered_by_page_ocr": 4,
            "covered_by_nearby_page_ocr": 5,
            "covered": 6,
            "table_pending": 7,
        }
        return order.get(status, 99)
