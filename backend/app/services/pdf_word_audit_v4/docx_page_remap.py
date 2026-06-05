from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from app.core.config import settings

from .common import has_high_value_field_content, looks_like_paragraphized_table_fragment, looks_like_table_header, looks_like_table_title, normalize_text
from .models import DocxEvidenceUnit, PdfEvidenceUnit


class DocxPageRemapper:
    """Conservatively remap DOCX table blocks to PDF pages using PDF text evidence.

    DOCX XML page numbers are only an estimate unless a rendered-DOCX pipeline is
    available. Large WPS-created tables can drift by one or more pages, and that
    bad estimate pollutes route selection. This layer only moves an entire table
    block when the current page has weak support and another nearby PDF page has
    strong OCR/native-text support.
    """

    VERSION = "docx_page_remap_v1"

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_window = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_WINDOW", 4) or 4))
        self.min_table_units = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_MIN_TABLE_UNITS", 20) or 20))
        self.max_samples = max(20, int(getattr(settings, "PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_MAX_SAMPLES", 9999) or 9999))

    def remap(
        self,
        *,
        docx_units: Sequence[DocxEvidenceUnit],
        pdf_units: Sequence[PdfEvidenceUnit],
        page_profiles: Dict[str, Dict[str, Any]],
        pdf_page_count: int,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self._payload(enabled=False, table_reviews=[], remapped_units=0, warnings=[])

        page_texts = self._page_texts(pdf_units)
        if not page_texts:
            return self._payload(
                enabled=True,
                table_reviews=[],
                remapped_units=0,
                warnings=["docx_page_remap_no_pdf_text_evidence"],
            )

        table_groups = self._table_groups(docx_units)
        table_reviews: List[Dict[str, Any]] = []
        remapped_units = 0
        remapped_companion_units = 0
        for table_index, units in sorted(table_groups.items()):
            if len(units) < self.min_table_units:
                continue
            review = self._review_table(
                table_index=table_index,
                units=units,
                page_texts=page_texts,
                page_profiles=page_profiles,
                pdf_page_count=pdf_page_count,
            )
            if not review:
                continue
            table_reviews.append(review)
            if not review.get("remapped"):
                continue
            old_page = int(review.get("from_page") or 0)
            new_page = int(review.get("to_page") or 0)
            if old_page <= 0 or new_page <= 0 or old_page == new_page:
                continue
            companion_units = self._companion_units(
                docx_units=docx_units,
                table_units=units,
                from_page=old_page,
                to_page=new_page,
                page_texts=page_texts,
            )
            review["companion_unit_count"] = len(companion_units)
            review["companion_unit_ids"] = [unit.unit_id for unit in companion_units[:40]]
            for unit in units:
                unit.estimated_page_no = new_page
                self._append_flags(
                    unit,
                    [
                        "docx_page_remapped",
                        "table_block_remapped_by_pdf_ocr",
                        f"page_remapped_from={old_page}",
                        f"page_remap_confidence={float(review.get('confidence') or 0.0):.2f}",
                    ],
                )
            remapped_units += len(units)
            for unit in companion_units:
                unit.estimated_page_no = new_page
                self._append_flags(
                    unit,
                    [
                        "docx_page_remapped",
                        "table_companion_remapped_by_pdf_ocr",
                        f"page_remapped_from={old_page}",
                        f"page_remap_confidence={float(review.get('confidence') or 0.0):.2f}",
                        f"page_remap_companion_for_table={int(table_index)}",
                    ],
                )
            remapped_companion_units += len(companion_units)

        warnings: List[str] = []
        if remapped_units:
            warnings.append("docx_page_remap_applied")
        return self._payload(
            enabled=True,
            table_reviews=table_reviews,
            remapped_units=remapped_units,
            remapped_companion_units=remapped_companion_units,
            warnings=warnings,
        )

    def _review_table(
        self,
        *,
        table_index: int,
        units: Sequence[DocxEvidenceUnit],
        page_texts: Dict[int, str],
        page_profiles: Dict[str, Dict[str, Any]],
        pdf_page_count: int,
    ) -> Dict[str, Any]:
        from_page = self._dominant_page(units)
        if from_page <= 0:
            return {}
        samples = self._samples(units)
        if not samples:
            return {}
        candidate_pages = self._candidate_pages(from_page=from_page, pdf_page_count=pdf_page_count)
        scores: List[Dict[str, Any]] = []
        for page_no in candidate_pages:
            score, matched_count, matched_samples = self._score_page(samples=samples, page_text=page_texts.get(page_no, ""))
            scores.append(
                {
                    "page_no": page_no,
                    "score": score,
                    "matched_count": matched_count,
                    "matched_samples": matched_samples[:12],
                    "pdf_text_chars": len(normalize_text(page_texts.get(page_no, ""))),
                    "page_labels": list((page_profiles.get(str(page_no)) or {}).get("labels") or [])[:8],
                }
            )
        scores.sort(key=lambda item: (-int(item["score"]), abs(int(item["page_no"]) - from_page), int(item["page_no"])))
        if not scores:
            return {}
        best = scores[0]
        current = next((item for item in scores if int(item["page_no"]) == from_page), {"page_no": from_page, "score": 0, "matched_count": 0})
        second = scores[1] if len(scores) > 1 else {"score": 0, "matched_count": 0}
        min_score = self._min_score(sample_count=len(samples))
        best_score = int(best.get("score") or 0)
        current_score = int(current.get("score") or 0)
        score_gap = best_score - current_score
        remapped = self._should_remap(
            from_page=from_page,
            best_page=int(best.get("page_no") or 0),
            best_score=best_score,
            current_score=current_score,
            best_match_count=int(best.get("matched_count") or 0),
            min_score=min_score,
        )
        confidence = self._confidence(
            best_score=best_score,
            current_score=current_score,
            second_score=int(second.get("score") or 0),
            matched_count=int(best.get("matched_count") or 0),
            sample_count=len(samples),
        )
        reason = (
            f"表格块当前估页第 {from_page} 页得分 {current_score}，"
            f"最佳 PDF 页第 {int(best.get('page_no') or 0)} 页得分 {best_score}。"
        )
        if remapped:
            reason += "当前页支持弱且最佳页 OCR/原生文本强匹配，执行整体归页校正。"
        else:
            reason += "未达到保守归页阈值，保持原估页。"
        return {
            "table_index": int(table_index),
            "from_page": int(from_page),
            "to_page": int(best.get("page_no") or from_page),
            "unit_count": len(units),
            "sample_count": len(samples),
            "current_score": current_score,
            "best_score": best_score,
            "second_score": int(second.get("score") or 0),
            "score_gap": score_gap,
            "matched_count": int(best.get("matched_count") or 0),
            "min_score": min_score,
            "confidence": confidence,
            "remapped": bool(remapped),
            "reason": reason,
            "top_pages": scores[:6],
        }

    def _should_remap(
        self,
        *,
        from_page: int,
        best_page: int,
        best_score: int,
        current_score: int,
        best_match_count: int,
        min_score: int,
    ) -> bool:
        if best_page <= 0 or best_page == from_page:
            return False
        if best_score < min_score:
            return False
        if best_match_count < 6:
            return False
        if current_score >= min_score and current_score >= best_score * 0.82:
            return False
        if best_score - current_score < max(14, int(best_score * 0.18)):
            return False
        return True

    def _confidence(
        self,
        *,
        best_score: int,
        current_score: int,
        second_score: int,
        matched_count: int,
        sample_count: int,
    ) -> float:
        gap = max(0, best_score - current_score)
        second_gap = max(0, best_score - second_score)
        score_part = min(0.28, gap / 260.0)
        match_part = min(0.16, matched_count / max(30.0, sample_count * 0.65))
        second_part = min(0.08, second_gap / 260.0)
        return round(max(0.0, min(0.98, 0.5 + score_part + match_part + second_part)), 4)

    def _min_score(self, *, sample_count: int) -> int:
        return max(20, min(120, int(sample_count * 1.6)))

    def _score_page(self, *, samples: Sequence[Dict[str, str]], page_text: str) -> Tuple[int, int, List[str]]:
        page_norm = self._semantic_compact(page_text)
        page_digits = self._digits_key(page_text)
        score = 0
        matched_count = 0
        matched_samples: List[str] = []
        for sample in samples:
            text = sample["text"]
            norm = sample["norm"]
            digits = sample["digits"]
            added = 0
            if norm and len(norm) >= 4 and norm in page_norm:
                added = 6 + min(8, len(norm) // 6)
            elif digits and len(digits) >= 3 and digits in page_digits:
                added = 3 if len(digits) <= 4 else 4 + min(5, len(digits) // 4)
            elif norm and len(norm) >= 2 and sample.get("has_cjk") == "1" and norm in page_norm:
                added = 2
            if not added:
                continue
            score += added
            matched_count += 1
            if len(matched_samples) < 12:
                matched_samples.append(text)
        return score, matched_count, matched_samples

    def _samples(self, units: Sequence[DocxEvidenceUnit]) -> List[Dict[str, str]]:
        rows: List[Tuple[int, Dict[str, str]]] = []
        seen: set[str] = set()
        for unit in units:
            text = " ".join(str(unit.text or "").split()).strip()
            if not text:
                continue
            norm = self._semantic_compact(text)
            digits = self._digits_key(text)
            if len(norm) < 2 and len(digits) < 2:
                continue
            key = norm or digits
            if key in seen:
                continue
            seen.add(key)
            has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
            priority = 0
            if len(norm) >= 4:
                priority += 4
            if len(digits) >= 3:
                priority += 3
            if has_cjk:
                priority += 2
            if re.search(r"[A-Za-z]", text) and any(char.isdigit() for char in text):
                priority += 2
            rows.append(
                (
                    priority,
                    {
                        "unit_id": unit.unit_id,
                        "text": text[:100],
                        "norm": norm,
                        "digits": digits,
                        "has_cjk": "1" if has_cjk else "0",
                    },
                )
            )
        rows.sort(key=lambda item: (-item[0], item[1]["unit_id"]))
        return [item for _priority, item in rows[: self.max_samples]]

    def _candidate_pages(self, *, from_page: int, pdf_page_count: int) -> List[int]:
        start = max(1, from_page - self.max_window)
        end = min(max(1, int(pdf_page_count or 1)), from_page + self.max_window)
        return list(range(start, end + 1))

    def _dominant_page(self, units: Sequence[DocxEvidenceUnit]) -> int:
        counts = Counter(int(unit.estimated_page_no or 0) for unit in units if int(unit.estimated_page_no or 0) > 0)
        return int(counts.most_common(1)[0][0]) if counts else 0

    def _table_groups(self, docx_units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[DocxEvidenceUnit]]:
        rows: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in docx_units:
            if unit.container_type != "table_cell" or not unit.table_index:
                continue
            rows.setdefault(int(unit.table_index), []).append(unit)
        return rows

    def _companion_units(
        self,
        *,
        docx_units: Sequence[DocxEvidenceUnit],
        table_units: Sequence[DocxEvidenceUnit],
        from_page: int,
        to_page: int,
        page_texts: Dict[int, str],
    ) -> List[DocxEvidenceUnit]:
        if not table_units:
            return []
        table_ids = {id(unit) for unit in table_units}
        min_order = min(int(unit.order_index or 0) for unit in table_units)
        part_names = {str(unit.part_name or "") for unit in table_units if str(unit.part_name or "")}
        new_text = page_texts.get(to_page, "")
        old_text = page_texts.get(from_page, "")
        candidates: List[Tuple[int, DocxEvidenceUnit, str, int, int, bool]] = []
        rows: List[Tuple[int, DocxEvidenceUnit]] = []
        for unit in docx_units:
            if id(unit) in table_ids or unit.container_type == "table_cell":
                continue
            if part_names and str(unit.part_name or "") not in part_names:
                continue
            if int(unit.estimated_page_no or 0) != int(from_page):
                continue
            order = int(unit.order_index or 0)
            distance = min_order - order
            if distance <= 0 or distance > 42:
                continue
            text = " ".join(str(unit.text or "").split()).strip()
            if not text:
                continue
            new_score = self._unit_score(text=text, page_text=new_text)
            old_score = self._unit_score(text=text, page_text=old_text)
            title_like = self._looks_like_table_companion(text)
            candidates.append((distance, unit, text, new_score, old_score, title_like))
            if new_score >= 4 and new_score > old_score:
                rows.append((distance, unit))
            elif title_like and distance <= 12 and new_score >= old_score:
                rows.append((distance, unit))
        rows.sort(key=lambda item: (-int(item[0]), int(item[1].order_index or 0)))
        selected = [unit for _distance, unit in rows[:36]]
        if len(selected) >= 3:
            selected_ids = {id(unit) for unit in selected}
            min_selected_order = min(int(unit.order_index or 0) for unit in selected)
            max_selected_order = max(int(unit.order_index or 0) for unit in selected)
            for _distance, unit, text, new_score, old_score, _title_like in candidates:
                if id(unit) in selected_ids:
                    continue
                order = int(unit.order_index or 0)
                if not (min_selected_order <= order <= max_selected_order):
                    continue
                if self._looks_like_table_fragment(text) or (new_score >= 2 and new_score >= old_score):
                    selected.append(unit)
                    selected_ids.add(id(unit))
        selected.sort(key=lambda unit: int(unit.order_index or 0))
        return selected[:36]

    def _unit_score(self, *, text: str, page_text: str) -> int:
        norm = self._semantic_compact(text)
        digits = self._digits_key(text)
        if not norm and not digits:
            return 0
        score = 0
        page_norm = self._semantic_compact(page_text)
        page_digits = self._digits_key(page_text)
        if norm and len(norm) >= 4 and norm in page_norm:
            score += 8 + min(8, len(norm) // 8)
        elif norm and len(norm) >= 2 and any("\u4e00" <= char <= "\u9fff" for char in text) and norm in page_norm:
            score += 3
        if digits and len(digits) >= 3 and digits in page_digits:
            score += 4 + min(6, len(digits) // 4)
        return score

    def _looks_like_table_companion(self, text: str) -> bool:
        compact = self._semantic_compact(text)
        if not compact:
            return False
        if looks_like_table_title(compact) or looks_like_table_header(compact):
            return True
        return bool("表" in compact and has_high_value_field_content(compact))

    def _looks_like_table_fragment(self, text: str) -> bool:
        compact = self._semantic_compact(text)
        if not compact or not re.search(r"\d", compact):
            return False
        if looks_like_paragraphized_table_fragment(compact):
            return True
        if has_high_value_field_content(compact) and len(compact) <= 36:
            return True
        return bool(len(compact) <= 36 and (re.search(r"[A-Za-z]", compact) or re.search(r"\d{4,}", compact) or re.search(r"\d+[/:：]\d+", compact)))

    def _page_texts(self, pdf_units: Sequence[PdfEvidenceUnit]) -> Dict[int, str]:
        rows: Dict[int, List[str]] = {}
        useful_types = {"anchor_ocr_page", "native_text_block", "anchor_ocr_line"}
        for unit in pdf_units:
            if unit.unit_type not in useful_types:
                continue
            text = str(unit.text or "")
            if not text.strip():
                continue
            rows.setdefault(int(unit.page_no or 0), []).append(text)
        return {page_no: "\n".join(parts) for page_no, parts in rows.items() if page_no > 0}

    def _append_flags(self, unit: DocxEvidenceUnit, flags: Sequence[str]) -> None:
        seen = set(unit.flags)
        for flag in flags:
            value = str(flag or "").strip()
            if value and value not in seen:
                unit.flags.append(value)
                seen.add(value)

    def _semantic_compact(self, value: Any) -> str:
        text = normalize_text(str(value or "")).lower()
        text = text.replace("〇", "0").replace("○", "0")
        text = re.sub(r"\s+", "", text)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)

    def _digits_key(self, value: Any) -> str:
        return re.sub(r"\D+", "", str(value or ""))

    def _payload(
        self,
        *,
        enabled: bool,
        table_reviews: Sequence[Dict[str, Any]],
        remapped_units: int,
        remapped_companion_units: int = 0,
        warnings: Sequence[str],
    ) -> Dict[str, Any]:
        remapped_tables = [item for item in table_reviews if item.get("remapped")]
        return {
            "enabled": bool(enabled),
            "version": self.VERSION,
            "summary": {
                "enabled": bool(enabled),
                "version": self.VERSION,
                "reviewed_table_count": len(table_reviews),
                "remapped_table_count": len(remapped_tables),
                "remapped_unit_count": int(remapped_units or 0),
                "remapped_companion_unit_count": int(remapped_companion_units or 0),
                "warnings": [str(item) for item in warnings if str(item)],
            },
            "table_reviews": [dict(item) for item in table_reviews],
        }
