from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from app.core.config import settings

from .common import normalize_text
from .models import ConversionPreflightResult, DocxEvidenceUnit, FragmentAnomalyReview


class FragmentAnomalyBuilder:
    """Detect scan-page OCR fragmentation and repetition in the WPS DOCX side.

    This layer is deliberately page-level. It does not decide the corrected
    value. It marks pages where WPS appears to have split, duplicated, or
    garbled one visible PDF region into many small DOCX paragraphs, which is a
    common failure mode on photo/scanned PDF conversions.
    """

    TEXT_CONTAINERS = {"paragraph", "textbox"}
    COMPLEX_LABELS = {"scan_like", "image_text_heavy", "mixed_layout", "mapping_risk"}

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_pages: int | None = None,
        min_fragments: int = 10,
        min_repeated_hits: int = 5,
    ) -> None:
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_FRAGMENT_ANOMALY_ENABLED", True) if enabled is None else enabled)
        self.max_pages = max(0, int(max_pages if max_pages is not None else getattr(settings, "PDF_WORD_AUDIT_V4_FRAGMENT_ANOMALY_MAX_PAGES", 9999) or 9999))
        self.min_fragments = max(4, int(min_fragments or 10))
        self.min_repeated_hits = max(3, int(min_repeated_hits or 5))

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[FragmentAnomalyReview]:
        if not self.enabled or self.max_pages <= 0:
            return []
        grouped = self._docx_units_by_page(preflight_result.docx_units)
        page_ocr = self._anchor_ocr_by_page(preflight_result.anchor_ocr)
        scored: List[tuple[float, FragmentAnomalyReview]] = []
        for page_no, units in grouped.items():
            review = self._review_page(
                page_no=page_no,
                units=units,
                profile=preflight_result.page_profiles.get(str(page_no)) or {},
                pdf_anchor_text=page_ocr.get(page_no, ""),
            )
            if review is None:
                continue
            score = float(review.confidence or 0.0) + min(0.3, review.repeated_fragment_count / 100.0)
            scored.append((score, review))
        scored.sort(key=lambda item: (-item[0], item[1].page_no))
        reviews: List[FragmentAnomalyReview] = []
        for index, (_score, review) in enumerate(scored[: self.max_pages], start=1):
            review.review_id = f"fragment_anomaly_{index:04d}"
            reviews.append(review)
        return reviews

    def _docx_units_by_page(self, units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[DocxEvidenceUnit]]:
        grouped: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            if unit.container_type not in self.TEXT_CONTAINERS:
                continue
            text = str(unit.text or "").strip()
            compact = self._compact(text)
            if len(compact) < 3:
                continue
            if unit.container_type == "paragraph" and self._looks_like_page_number(compact):
                continue
            grouped.setdefault(page_no, []).append(unit)
        return grouped

    def _review_page(
        self,
        *,
        page_no: int,
        units: Sequence[DocxEvidenceUnit],
        profile: Dict[str, Any],
        pdf_anchor_text: str,
    ) -> FragmentAnomalyReview | None:
        if len(units) < self.min_fragments:
            return None
        labels = set(profile.get("labels") or [])
        complex_page = bool(labels & self.COMPLEX_LABELS)
        compact_values = [self._compact(unit.text) for unit in units]
        short_fragment_count = sum(1 for value in compact_values if len(value) <= 30)
        unique_fragment_count = len({value for value in compact_values if value})
        repeated_terms = self._repeated_terms(units)
        if not repeated_terms:
            return None
        dominant = repeated_terms[0]
        repeated_fragment_count = int(dominant.get("unit_count") or 0)
        strong_repetition = repeated_fragment_count >= self.min_repeated_hits
        severe_fragmentation = short_fragment_count >= max(self.min_fragments, 12) and repeated_fragment_count >= 4
        if not ((complex_page and strong_repetition) or severe_fragmentation):
            return None
        pdf_support = self._pdf_supports_term(pdf_anchor_text=pdf_anchor_text, term=dominant)
        confidence = self._confidence(
            complex_page=complex_page,
            repeated_fragment_count=repeated_fragment_count,
            fragment_count=len(units),
            short_fragment_count=short_fragment_count,
            pdf_support=pdf_support,
        )
        examples = self._examples(units=units, repeated_terms=repeated_terms[:3])
        anchor_unit_id = self._anchor_unit_id(examples=examples, units=units)
        flags = ["fragment_anomaly", "page_level_conversion_risk"]
        if complex_page:
            flags.append("complex_page")
        if labels:
            flags.extend(f"page_label={label}" for label in sorted(labels))
        if pdf_support:
            flags.append("pdf_anchor_supports_repeated_term")
        else:
            flags.append("pdf_anchor_support_missing")
        if short_fragment_count >= max(self.min_fragments, 12):
            flags.append("many_short_fragments")
        reason = (
            f"第 {page_no} 页 DOCX 出现 {len(units)} 个普通文本段落，其中 {short_fragment_count} 个为短碎片；"
            f"片段“{dominant.get('display') or dominant.get('term')}”覆盖 {repeated_fragment_count} 个段落。"
        )
        if pdf_support:
            reason += " PDF 页级 OCR 也出现同类文本，说明 WPS 可能把同一可见区域拆碎或重复识别。"
        else:
            reason += " 当前 PDF 页级 OCR 未能稳定确认该片段，需要人工按原图整页核查。"
        return FragmentAnomalyReview(
            review_id="fragment_anomaly_pending",
            page_no=page_no,
            anchor_unit_id=anchor_unit_id,
            verdict="fragmented_repetition_risk",
            confidence=confidence,
            reason=reason,
            repeated_terms=repeated_terms[:8],
            docx_examples=examples[:12],
            pdf_anchor_excerpt=self._clip(pdf_anchor_text, 420),
            fragment_count=len(units),
            repeated_fragment_count=repeated_fragment_count,
            short_fragment_count=short_fragment_count,
            unique_fragment_count=unique_fragment_count,
            next_route="needs_human_page_review",
            flags=flags,
        )

    def _repeated_terms(self, units: Sequence[DocxEvidenceUnit]) -> List[Dict[str, Any]]:
        buckets: Dict[str, Dict[str, Any]] = {}
        for unit in units:
            terms = self._terms(unit.text)
            seen_in_unit: set[str] = set()
            for term in terms:
                key = self._term_key(term)
                if not key or key in seen_in_unit:
                    continue
                seen_in_unit.add(key)
                bucket = buckets.setdefault(
                    key,
                    {
                        "term": key,
                        "display": term,
                        "unit_ids": [],
                        "examples": [],
                        "source": "docx_fragment",
                    },
                )
                bucket["unit_ids"].append(unit.unit_id)
                if len(term) > len(str(bucket.get("display") or "")):
                    bucket["display"] = term
                if len(bucket["examples"]) < 5:
                    bucket["examples"].append(self._clip(unit.text, 80))
        rows: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            unit_ids = list(bucket.get("unit_ids") or [])
            if len(unit_ids) < 3:
                continue
            rows.append(
                {
                    "term": bucket.get("term") or "",
                    "display": bucket.get("display") or bucket.get("term") or "",
                    "unit_count": len(unit_ids),
                    "unit_ids": unit_ids[:30],
                    "examples": list(bucket.get("examples") or []),
                    "source": "docx_fragment",
                }
            )
        rows.sort(key=lambda item: (-int(item.get("unit_count") or 0), -len(str(item.get("display") or "")), str(item.get("term") or "")))
        return rows

    def _terms(self, text: str) -> List[str]:
        compact = self._compact(text)
        if not compact:
            return []
        terms: List[str] = []
        patterns = [
            r"[\u4e00-\u9fff]{0,12}\d{1,4}号楼\d{2,6}",
            r"\d{1,4}号楼\d{2,6}",
            r"[\u4e00-\u9fff]{2,12}\d[A-Za-z0-9\u4e00-\u9fff]{3,14}",
            r"[A-Z]{0,4}\d[A-Z0-9]{5,14}",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
                term = match.group(0).strip()
                if self._valid_term(term):
                    terms.append(term)
        return self._dedupe(terms)

    def _valid_term(self, term: str) -> bool:
        value = str(term or "").strip()
        if len(value) < 5:
            return False
        if self._looks_like_date(value):
            return False
        if re.fullmatch(r"\d+(\.\d+)?", value):
            return False
        has_digit = any(ch.isdigit() for ch in value)
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in value)
        has_alpha = any(ch.isalpha() for ch in value)
        return has_digit and (has_cjk or has_alpha)

    def _term_key(self, term: str) -> str:
        value = str(term or "").upper()
        building = re.search(r"\d{1,4}号楼\d{2,6}", value)
        if building:
            return building.group(0)
        account = re.search(r"[A-Z]{0,4}\d[A-Z0-9]{5,14}", value)
        if account:
            return account.group(0)
        return value

    def _examples(self, *, units: Sequence[DocxEvidenceUnit], repeated_terms: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keys = {str(item.get("term") or "") for item in repeated_terms}
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for unit in units:
            unit_terms = [self._term_key(term) for term in self._terms(unit.text)]
            matched = [term for term in unit_terms if term in keys]
            if not matched:
                continue
            compact = self._compact(unit.text)
            dedupe_key = f"{matched[0]}:{compact}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "text": self._clip(unit.text, 120),
                    "terms": self._dedupe(matched)[:5],
                    "order_index": int(unit.order_index or 0),
                }
            )
            if len(rows) >= 16:
                break
        rows.sort(key=lambda item: (int(item.get("order_index") or 0), item.get("unit_id") or ""))
        return rows

    def _anchor_unit_id(self, *, examples: Sequence[Dict[str, Any]], units: Sequence[DocxEvidenceUnit]) -> str:
        if examples:
            rows = sorted(
                examples,
                key=lambda item: (
                    -len(self._compact(item.get("text") or "")),
                    int(item.get("order_index") or 0),
                    item.get("unit_id") or "",
                ),
            )
            return str(rows[0].get("unit_id") or "")
        for unit in units:
            if str(unit.text or "").strip():
                return unit.unit_id
        return ""

    def _pdf_supports_term(self, *, pdf_anchor_text: str, term: Dict[str, Any]) -> bool:
        pdf = self._semantic_compact(pdf_anchor_text)
        if not pdf:
            return False
        candidates = [
            str(term.get("term") or ""),
            str(term.get("display") or ""),
        ]
        examples = term.get("examples") or []
        candidates.extend(str(item or "") for item in examples[:2])
        for candidate in candidates:
            value = self._semantic_compact(candidate)
            if len(value) >= 5 and (value in pdf or pdf.find(value[-min(len(value), 8):]) >= 0):
                return True
        core = self._term_key(str(term.get("display") or term.get("term") or ""))
        return bool(core and self._semantic_compact(core) in pdf)

    def _anchor_ocr_by_page(self, anchor_ocr: Dict[str, Any]) -> Dict[int, str]:
        rows: Dict[int, str] = {}
        for page in anchor_ocr.get("pages") or []:
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            if page_no <= 0:
                continue
            rows[page_no] = str(page.get("text") or "")
        return rows

    def _confidence(
        self,
        *,
        complex_page: bool,
        repeated_fragment_count: int,
        fragment_count: int,
        short_fragment_count: int,
        pdf_support: bool,
    ) -> float:
        score = 0.48
        if complex_page:
            score += 0.1
        score += min(0.16, max(0, repeated_fragment_count - self.min_repeated_hits) * 0.012)
        score += min(0.08, max(0, fragment_count - self.min_fragments) * 0.003)
        if short_fragment_count >= max(12, self.min_fragments):
            score += 0.05
        if pdf_support:
            score += 0.08
        return round(min(0.82, score), 4)

    def _compact(self, text: Any) -> str:
        return normalize_text(str(text or "")).upper()

    def _semantic_compact(self, text: Any) -> str:
        value = self._compact(text)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _looks_like_date(self, value: str) -> bool:
        return bool(re.fullmatch(r"\d{4}[-/.年]\d{1,2}([-/.\u6708]\d{1,2}\u65e5?)?", value))

    def _looks_like_page_number(self, value: str) -> bool:
        return bool(re.fullmatch(r"第?\d{1,3}页", value))

    def _clip(self, text: Any, limit: int) -> str:
        value = " ".join(str(text or "").split())
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "…"

    def _dedupe(self, values: Sequence[str]) -> List[str]:
        rows: List[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            rows.append(value)
        return rows
