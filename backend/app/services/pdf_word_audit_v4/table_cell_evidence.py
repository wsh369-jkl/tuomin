from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from app.core.config import settings

from .models import ContentCoverageReview, ConversionPreflightResult, DocxEvidenceUnit, PdfEvidenceUnit, TableCellEvidenceReview


class TableCellEvidenceBuilder:
    """Build cell-level table evidence from page OCR tokens.

    This is a deterministic evidence layer between page-level OCR and the table
    specialist. It does not parse every grid cell visually yet; it links DOCX
    table cells to high-confidence PDF page OCR tokens when the mismatch is
    mechanically explainable, such as letters mixed into numbers or missing
    decimal points.
    """

    TARGET_STATUSES = {
        "table_pending",
        "mapping_uncertain",
        "diff_candidate",
        "uncovered_docx_content",
        "covered_by_page_ocr",
        "covered_by_nearby_page_ocr",
    }

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[TableCellEvidenceReview]:
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_CELL_EVIDENCE_ENABLED", True)):
            return []
        max_reviews = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_CELL_EVIDENCE_MAX_REVIEWS", 9999) or 9999))
        if max_reviews <= 0:
            return []
        docx_by_unit = {unit.unit_id: unit for unit in preflight_result.docx_units}
        tokens_by_page = self._tokens_by_page(preflight_result=preflight_result)
        rows: List[TableCellEvidenceReview] = []
        seen: set[tuple[int, str, str]] = set()
        for review in self._candidate_reviews(preflight_result.content_coverage_reviews):
            unit = docx_by_unit.get(review.unit_id)
            if unit is None:
                continue
            page_no = int(review.page_no or unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            profile = dict(preflight_result.page_profiles.get(str(page_no)) or {})
            if not self._is_table_route(profile=profile, unit=unit):
                continue
            paragraphized_fragment = unit.container_type != "table_cell"
            if paragraphized_fragment and not self._is_paragraphized_table_fragment(unit=unit, profile=profile):
                continue
            match = self._match_cell(text=unit.text or review.text, tokens=tokens_by_page.get(page_no, []))
            if not match:
                continue
            docx_exact_seen = self._page_ocr_supports_docx_value(unit.text or review.text, tokens=tokens_by_page.get(page_no, []))
            if docx_exact_seen and not self._allow_decimal_missing_with_exact_noise(text=unit.text or review.text, match=match):
                continue
            key = (page_no, review.unit_id, self._compact(match["visible_text"]))
            if key in seen:
                continue
            seen.add(key)
            confidence = float(match["confidence"] or 0.0)
            flags = [
                "table_cell_evidence",
                "exact_replacement",
                f"coverage_status={review.status}",
                f"issue_type={match['issue_type']}",
            ]
            flags.extend(str(flag) for flag in match.get("flags", []) if str(flag or "").strip())
            if paragraphized_fragment:
                flags.append("paragraphized_table_fragment")
            if docx_exact_seen:
                confidence = min(confidence, 0.76)
                flags.append("docx_exact_ocr_also_seen")
            decision = "suspected_error" if docx_exact_seen else "confirmed_error"
            status = "pdf_ocr_ambiguous_mismatch" if docx_exact_seen else "pdf_ocr_supported_mismatch"
            rows.append(
                TableCellEvidenceReview(
                    review_id=f"table_cell_ev_{len(rows)+1:04d}",
                    page_no=page_no,
                    docx_unit_id=unit.unit_id,
                    table_index=unit.table_index,
                    row_index=unit.row_index,
                    col_index=unit.col_index,
                    docx_text=self._clean(unit.text or review.text),
                    visible_text=match["visible_text"],
                    status=status,
                    decision=decision,
                    confidence=confidence,
                    issue_type=match["issue_type"],
                    evidence_source=match["source"],
                    reason=match["reason"],
                    normalized_docx=match["docx_key"],
                    normalized_visible=match["visible_key"],
                    matched_token=match["matched_token"],
                    coverage_review_id=review.review_id,
                    flags=flags,
                )
            )
            if len(rows) >= max_reviews:
                break
        self._promote_contextual_decimal_matches(rows=rows, docx_by_unit=docx_by_unit)
        self._downgrade_confusable_values_missing_decimal_context(rows=rows, docx_by_unit=docx_by_unit)
        return rows

    def _candidate_reviews(self, reviews: Sequence[ContentCoverageReview]) -> List[ContentCoverageReview]:
        rows = [
            review
            for review in reviews
            if review.side == "docx"
            and review.status in self.TARGET_STATUSES
            and str(review.text or "").strip()
        ]
        rows.sort(key=lambda item: (int(item.page_no or 9999), self._status_priority(item.status), item.review_id))
        return rows

    def _status_priority(self, status: str) -> int:
        return {
            "table_pending": 0,
            "diff_candidate": 1,
            "mapping_uncertain": 2,
            "uncovered_docx_content": 3,
            "covered_by_page_ocr": 4,
            "covered_by_nearby_page_ocr": 5,
        }.get(str(status or ""), 9)

    def _is_table_route(self, *, profile: Dict[str, Any], unit: DocxEvidenceUnit) -> bool:
        labels = set(profile.get("labels") or [])
        primary = str(profile.get("primary_route") or "")
        return bool(
            unit.container_type == "table_cell"
            or primary in {"image_table_cell_compare", "native_table_compare"}
            or profile.get("needs_table_parser")
            or profile.get("table_like")
            or "table_heavy" in labels
        )

    def _is_paragraphized_table_fragment(self, *, unit: DocxEvidenceUnit, profile: Dict[str, Any]) -> bool:
        if unit.container_type == "table_cell":
            return True
        labels = set(profile.get("labels") or [])
        primary = str(profile.get("primary_route") or "")
        if not (
            profile.get("needs_table_parser")
            or profile.get("table_like")
            or "table_heavy" in labels
            or primary in {"image_table_cell_compare", "native_table_compare"}
        ):
            return False
        text = self._clean(unit.text)
        if len(text) < 3 or len(text) > 40:
            return False
        if not any(ch.isdigit() for ch in text):
            return False
        return bool(
            re.search(r"[A-Za-z]", text)
            or re.search(r"\d{4,}", text)
            or re.search(r"\d+[/:：]\d+", text)
            or re.search(r"\d+\.\d+", text)
        )

    def _tokens_by_page(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, List[Dict[str, Any]]]:
        rows: Dict[int, List[Dict[str, Any]]] = {}
        seen: set[tuple[int, str, str]] = set()
        for unit in preflight_result.pdf_units:
            unit_type = str(unit.unit_type or "")
            if unit_type not in {"anchor_ocr_page", "anchor_ocr_line", "native_text_block"}:
                continue
            text = str(unit.text or "")
            if not text:
                continue
            page_no = int(unit.page_no or 0)
            if page_no <= 0:
                continue
            for token in self._extract_value_tokens(text):
                self._append_token(
                    rows=rows,
                    seen=seen,
                    page_no=page_no,
                    token=token,
                    source=unit.source or "pdf_page_ocr",
                    evidence_id=unit.unit_id,
                )
        for review in preflight_result.content_coverage_backfills:
            if not review.available:
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            text = str(review.normalized_text or review.extracted_text or "")
            if not text:
                continue
            source = f"{review.method or 'ocr'}_backfill_ocr"
            for token in self._extract_value_tokens(text):
                self._append_token(
                    rows=rows,
                    seen=seen,
                    page_no=page_no,
                    token=token,
                    source=source,
                    evidence_id=review.backfill_id,
                )
        return rows

    def _append_token(
        self,
        *,
        rows: Dict[int, List[Dict[str, Any]]],
        seen: set[tuple[int, str, str]],
        page_no: int,
        token: str,
        source: str,
        evidence_id: str,
    ) -> None:
        visible = self._clean_visible_token(token)
        raw = self._clean(token)
        if not visible and not raw:
            return
        key = (int(page_no or 0), self._compact(visible or raw), str(source or ""))
        if key in seen:
            return
        seen.add(key)
        rows.setdefault(int(page_no or 0), []).append(
            {
                "text": token,
                "visible_text": visible,
                "source": source or "pdf_page_ocr",
                "evidence_id": evidence_id,
            }
        )

    def _extract_value_tokens(self, text: str) -> List[str]:
        rows: List[str] = []
        for match in re.finditer(r"[A-Za-z¥￥]?\d[\dA-Za-z.,:：，]{1,18}", str(text or "")):
            token = match.group(0).strip(".,:：，")
            if len(self._digits_key(token)) < 3:
                continue
            rows.append(token)
        return rows

    def _match_cell(self, *, text: str, tokens: Sequence[Any]) -> Dict[str, Any]:
        docx_text = self._clean(text)
        if not self._candidate_cell_text(docx_text):
            return {}
        docx_key = self._confusable_key(docx_text)
        docx_digits = self._digits_key(docx_text)
        if len(docx_key) < 3 and len(docx_digits) < 3:
            return {}
        best: Dict[str, Any] = {}
        for record in tokens:
            token = str(record.get("text") or "") if isinstance(record, dict) else str(record or "")
            visible = str(record.get("visible_text") or "") if isinstance(record, dict) else self._clean_visible_token(token)
            source = str(record.get("source") or "pdf_page_ocr") if isinstance(record, dict) else "pdf_page_ocr"
            if not visible:
                continue
            visible_key = self._confusable_key(visible)
            visible_digits = self._digits_key(visible)
            if not visible_digits:
                continue
            issue_type = ""
            confidence = 0.0
            if self._has_confusable_letter(docx_text) and visible_digits == docx_key and visible != docx_text:
                issue_type = "digit_letter_confusion"
                confidence = 0.84
            elif self._decimal_missing(docx_text=docx_text, visible_text=visible, docx_digits=docx_digits, visible_digits=visible_digits):
                issue_type = "decimal_point_missing"
                confidence = 0.8
            elif self._punctuation_or_decimal_pollution(docx_text=docx_text, visible_text=visible, docx_digits=docx_digits, visible_digits=visible_digits):
                issue_type = "decimal_or_punctuation_pollution"
                confidence = 0.78
            if not issue_type:
                continue
            if self._same_semantic_value(docx_text, visible):
                continue
            candidate = {
                "visible_text": visible,
                "matched_token": token,
                "docx_key": docx_key or docx_digits,
                "visible_key": visible_key or visible_digits,
                "issue_type": issue_type,
                "confidence": confidence,
                "source": source,
                "reason": "PDF 页级 OCR 中存在与 DOCX 表格单元高度对应的可见值，可解释为 WPS 数字/字母混淆或小数点丢失。",
                "flags": ["confusable_decimal_candidate"] if issue_type == "digit_letter_confusion" and "." in visible else [],
            }
            if not best or self._better_match_candidate(candidate=candidate, current=best):
                best = candidate
        return best

    def _better_match_candidate(self, *, candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
        candidate_confidence = float(candidate.get("confidence") or 0.0)
        current_confidence = float(current.get("confidence") or 0.0)
        if candidate_confidence > current_confidence:
            return True
        if candidate_confidence < current_confidence:
            return False
        candidate_visible = self._clean(candidate.get("visible_text"))
        current_visible = self._clean(current.get("visible_text"))
        if "." in candidate_visible and "." not in current_visible:
            return True
        if len(candidate_visible) > len(current_visible) and self._digits_key(candidate_visible) == self._digits_key(current_visible):
            return True
        return False

    def _promote_contextual_decimal_matches(
        self,
        *,
        rows: Sequence[TableCellEvidenceReview],
        docx_by_unit: Dict[str, DocxEvidenceUnit],
    ) -> None:
        """Upgrade noisy decimal evidence only when table context agrees.

        Page OCR on dense table images often contains both the WPS-converted
        value without a decimal point and the visible PDF value with a decimal
        point. A single occurrence remains suspected. Multiple same-column or
        same-row occurrences in the same table are strong evidence that WPS
        systematically dropped decimal points, so those can be promoted.
        """

        candidates = [
            review
            for review in rows
            if self._context_promotable_decimal(review=review, docx_by_unit=docx_by_unit)
        ]
        if not candidates:
            return
        column_counts: Dict[tuple[int, int, int], int] = {}
        row_counts: Dict[tuple[int, int, int], int] = {}
        for review in candidates:
            table_key = self._table_key(review)
            if table_key is None:
                continue
            page_no, table_index, row_index, col_index = table_key
            column_key = (page_no, table_index, col_index)
            row_key = (page_no, table_index, row_index)
            column_counts[column_key] = column_counts.get(column_key, 0) + 1
            row_counts[row_key] = row_counts.get(row_key, 0) + 1

        for review in candidates:
            if review.decision != "suspected_error" or review.status != "pdf_ocr_ambiguous_mismatch":
                continue
            table_key = self._table_key(review)
            if table_key is None:
                continue
            page_no, table_index, row_index, col_index = table_key
            column_count = column_counts.get((page_no, table_index, col_index), 0)
            row_count = row_counts.get((page_no, table_index, row_index), 0)
            if column_count < 2 and row_count < 3:
                continue
            review.status = "pdf_ocr_context_confirmed_mismatch"
            review.decision = "confirmed_error"
            review.confidence = max(float(review.confidence or 0.0), 0.79 if column_count >= 2 else 0.77)
            review.reason = (
                "PDF 页级 OCR 中存在带小数点的可见值；同一表格的同列/同行存在多个小数点缺失证据，"
                "可排除孤立 OCR 噪声并确认该单元为 WPS 小数点漏识别。"
            )
            review.flags = list(
                dict.fromkeys(
                    [
                        *review.flags,
                        "table_context_confirmed_decimal_missing",
                        f"same_column_decimal_evidence={column_count}",
                        f"same_row_decimal_evidence={row_count}",
                    ]
                )
            )

    def _downgrade_confusable_values_missing_decimal_context(
        self,
        *,
        rows: Sequence[TableCellEvidenceReview],
        docx_by_unit: Dict[str, DocxEvidenceUnit],
    ) -> None:
        """Avoid confirming incomplete amount-like replacements.

        Dense fee tables often contain OCR candidates such as ``89220`` near
        already-decimal values. If the DOCX cell contains a confusable letter
        (``B9220``) and the PDF-side candidate only normalizes that letter to an
        integer-like value, the system has not proven whether the real visible
        value is ``89220`` or ``892.20``. In decimal-heavy row/table context we
        must keep it as human review instead of writing a "must change" comment.
        """

        for review in rows:
            if review.issue_type != "digit_letter_confusion":
                continue
            if review.decision != "confirmed_error":
                continue
            docx_text = self._clean(review.docx_text)
            visible_text = self._clean(review.visible_text)
            if "." in visible_text or "." in docx_text:
                continue
            if not re.search(r"[A-Za-z]", docx_text):
                continue
            visible_digits = self._digits_key(visible_text)
            if not re.fullmatch(r"\d{4,8}", visible_digits):
                continue
            unit = docx_by_unit.get(review.docx_unit_id)
            if unit is None or unit.container_type != "table_cell":
                continue
            decimal_context = self._decimal_context_count(unit=unit, docx_by_unit=docx_by_unit)
            if decimal_context < 2:
                continue
            review.status = "pdf_ocr_decimal_context_needs_review"
            review.decision = "suspected_error"
            review.confidence = min(float(review.confidence or 0.0), 0.68)
            review.reason = (
                "PDF 页级 OCR 只支持字母/数字归一后的整数形态；但同一表格行/表内存在金额小数语境，"
                "当前证据不足以确认是否还缺失小数点，需人工核对单元格截图。"
            )
            review.flags = list(
                dict.fromkeys(
                    [
                        *review.flags,
                        "confusable_integer_in_decimal_context",
                        "requires_human_decimal_review",
                        f"decimal_context_count={decimal_context}",
                    ]
                )
            )

    def _decimal_context_count(self, *, unit: DocxEvidenceUnit, docx_by_unit: Dict[str, DocxEvidenceUnit]) -> int:
        page_no = int(unit.estimated_page_no or 0)
        table_index = int(unit.table_index or 0)
        row_index = int(unit.row_index or 0)
        if page_no <= 0 or table_index <= 0 or row_index <= 0:
            return 0
        row_count = 0
        table_amount_like_count = 0
        for item in docx_by_unit.values():
            if item.unit_id == unit.unit_id:
                continue
            if int(item.estimated_page_no or 0) != page_no or int(item.table_index or 0) != table_index:
                continue
            text = self._clean(item.text)
            if self._decimal_amount_like(text):
                table_amount_like_count += 1
                if int(item.row_index or 0) == row_index:
                    row_count += 1
        return row_count * 2 + min(table_amount_like_count, 6)

    def _decimal_amount_like(self, text: str) -> bool:
        value = self._clean(text).replace(",", "").replace("，", "")
        return bool(re.fullmatch(r"[¥￥]?\d{1,6}\.\d{1,4}", value))

    def _context_promotable_decimal(
        self,
        *,
        review: TableCellEvidenceReview,
        docx_by_unit: Dict[str, DocxEvidenceUnit],
    ) -> bool:
        if review.issue_type != "decimal_point_missing":
            return False
        if self._table_key(review) is None:
            return False
        unit = docx_by_unit.get(review.docx_unit_id)
        if unit is None or unit.container_type != "table_cell":
            return False
        docx_text = self._clean(review.docx_text)
        visible_text = self._clean(review.visible_text)
        if not docx_text or not visible_text:
            return False
        if "." in docx_text or "." not in visible_text:
            return False
        if re.search(r"[A-Za-z]", docx_text):
            return False
        docx_digits = self._digits_key(docx_text)
        visible_digits = self._digits_key(visible_text)
        if docx_digits != visible_digits:
            return False
        if not re.fullmatch(r"\d{4,8}", docx_digits):
            return False
        return bool(re.fullmatch(r"\d{1,6}\.\d{2,4}", visible_text))

    def _table_key(self, review: TableCellEvidenceReview) -> tuple[int, int, int, int] | None:
        page_no = int(review.page_no or 0)
        table_index = int(review.table_index or 0)
        row_index = int(review.row_index or 0)
        col_index = int(review.col_index or 0)
        if page_no <= 0 or table_index <= 0 or row_index <= 0 or col_index <= 0:
            return None
        return page_no, table_index, row_index, col_index

    def _page_ocr_supports_docx_value(self, text: str, *, tokens: Sequence[Any]) -> bool:
        docx_text = self._clean(text)
        docx_compact = self._compact(docx_text)
        if not docx_compact:
            return False
        for record in tokens:
            token = str(record.get("text") or "") if isinstance(record, dict) else str(record or "")
            visible = str(record.get("visible_text") or "") if isinstance(record, dict) else self._clean_visible_token(token)
            raw = self._clean(token)
            if visible and self._compact(visible) == docx_compact:
                return True
            if raw and self._compact(raw) == docx_compact:
                return True
        return False

    def _allow_decimal_missing_with_exact_noise(self, *, text: str, match: Dict[str, Any]) -> bool:
        if str(match.get("issue_type") or "") != "decimal_point_missing":
            return False
        docx_text = self._clean(text)
        docx_digits = self._digits_key(docx_text)
        if not re.fullmatch(r"\d{5,8}", docx_digits):
            return False
        if re.search(r"[A-Za-z]", docx_text):
            return False
        visible_text = self._clean(match.get("visible_text"))
        if "." not in visible_text:
            return False
        return self._digits_key(visible_text) == docx_digits

    def _candidate_cell_text(self, text: str) -> bool:
        value = self._clean(text)
        if len(value) < 3 or len(value) > 40:
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in value):
            return False
        if not any(ch.isdigit() for ch in value):
            return False
        return bool(re.search(r"[A-Za-z]", value) or re.fullmatch(r"\d{4,}", self._digits_key(value)) or re.search(r"[/:：]", value))

    def _decimal_missing(self, *, docx_text: str, visible_text: str, docx_digits: str, visible_digits: str) -> bool:
        if docx_digits != visible_digits or len(docx_digits) < 4:
            return False
        return "." not in docx_text and "." in visible_text

    def _punctuation_or_decimal_pollution(self, *, docx_text: str, visible_text: str, docx_digits: str, visible_digits: str) -> bool:
        if docx_digits != visible_digits or len(docx_digits) < 4:
            return False
        if re.search(r"[A-Za-z]", docx_text):
            return False
        return bool(re.search(r"[/:：]", docx_text) and "." in visible_text)

    def _has_confusable_letter(self, text: str) -> bool:
        value = self._clean(text).upper()
        if not any(ch.isdigit() for ch in value):
            return False
        return any(ch in set("BCDEGILOQSZ") for ch in value)

    def _confusable_key(self, text: str) -> str:
        mapping = str.maketrans(
            {
                "B": "8",
                "C": "0",
                "D": "0",
                "E": "6",
                "G": "6",
                "I": "1",
                "L": "1",
                "O": "0",
                "Q": "0",
                "S": "5",
                "Z": "2",
                "b": "8",
                "c": "0",
                "d": "0",
                "e": "6",
                "g": "6",
                "i": "1",
                "l": "1",
                "o": "0",
                "q": "0",
                "s": "5",
                "z": "2",
            }
        )
        return self._digits_key(str(text or "").translate(mapping))

    def _digits_key(self, text: str) -> str:
        return re.sub(r"\D+", "", str(text or ""))

    def _same_semantic_value(self, left: str, right: str) -> bool:
        return self._compact(left) == self._compact(right)

    def _clean_visible_token(self, value: Any) -> str:
        text = self._clean(value).replace("，", ".").replace("：", ".")
        text = re.sub(r"\.{2,}", ".", text)
        text = text.lstrip("¥￥")
        text = text.strip(".")
        if len(text) > 24:
            return ""
        if re.search(r"[A-Za-z]", text):
            return ""
        return text

    def _clean(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _compact(self, value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "")).lower()
