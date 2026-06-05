from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from .common import normalize_text, similarity
from .models import (
    AlignmentLink,
    ConversionDiffCandidate,
    ConversionPreflightResult,
    DocxEvidenceUnit,
    PdfEvidenceUnit,
    ReviewRoute,
)


MATCH_THRESHOLD = 0.75
UNCERTAIN_THRESHOLD = 0.45

ID_RE = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")
MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
DATE_RE = re.compile(r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})")
AMOUNT_RE = re.compile(r"(?:[¥￥]\s*)?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:元|万元|人民币|RMB)?")
CASE_NO_RE = re.compile(r"[（(]?\d{4}[）)]?[\u4e00-\u9fff]{1,6}\d{2,8}[\u4e00-\u9fff]{1,8}\d+号")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{12,25}(?!\d)")
CONTRACT_RE = re.compile(r"(?:合同|协议|编号|No\.?|NO\.?)[:：]?\s*[\w\-（）()第号]{4,32}")
PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


class ConversionPreflightBuilder:
    """Build report-only conversion fidelity evidence from extracted PDF/DOCX units."""

    def build(
        self,
        *,
        pdf_page_count: int,
        pdf_units: Sequence[PdfEvidenceUnit],
        docx_units: Sequence[DocxEvidenceUnit],
        page_profiles: Dict[str, Dict[str, Any]],
        warnings: Sequence[str] = (),
        anchor_ocr_payload: Dict[str, Any] | None = None,
    ) -> ConversionPreflightResult:
        links = self._align_units(pdf_units=pdf_units, docx_units=docx_units)
        diffs = self._build_diff_candidates(pdf_units=pdf_units, docx_units=docx_units, links=links, page_profiles=page_profiles)
        routes = self._build_routes(pdf_units=pdf_units, docx_units=docx_units, links=links, diffs=diffs, page_profiles=page_profiles)
        return ConversionPreflightResult(
            pdf_page_count=pdf_page_count,
            pdf_units=list(pdf_units),
            docx_units=list(docx_units),
            alignment_links=links,
            diff_candidates=diffs,
            review_routes=routes,
            page_profiles=page_profiles,
            warnings=list(warnings),
            anchor_ocr=dict(anchor_ocr_payload or {}),
        )

    def _align_units(
        self,
        *,
        pdf_units: Sequence[PdfEvidenceUnit],
        docx_units: Sequence[DocxEvidenceUnit],
    ) -> List[AlignmentLink]:
        pdf_text_units = [
            item
            for item in pdf_units
            if item.normalized_text and item.unit_type in {"native_text_block", "anchor_ocr_line"}
        ]
        docx_text_units = [item for item in docx_units if item.normalized_text]
        candidates: List[Tuple[float, PdfEvidenceUnit, DocxEvidenceUnit, List[Dict[str, str]], List[str]]] = []
        for pdf_unit in pdf_text_units:
            pdf_anchors = self._anchors(pdf_unit.text)
            for docx_unit in docx_text_units:
                score, anchors, reasons = self._alignment_score(pdf_unit, docx_unit, pdf_anchors=pdf_anchors)
                if score >= UNCERTAIN_THRESHOLD:
                    candidates.append((score, pdf_unit, docx_unit, anchors, reasons))
        candidates.sort(key=lambda item: item[0], reverse=True)
        used_pdf: set[str] = set()
        used_docx: set[str] = set()
        links: List[AlignmentLink] = []
        for score, pdf_unit, docx_unit, anchors, reasons in candidates:
            if pdf_unit.unit_id in used_pdf or docx_unit.unit_id in used_docx:
                continue
            used_pdf.add(pdf_unit.unit_id)
            used_docx.add(docx_unit.unit_id)
            status = "matched" if score >= MATCH_THRESHOLD else "mapping_uncertain"
            alignment_type = "anchor_text_match" if anchors else "text_similarity_match"
            links.append(
                AlignmentLink(
                    link_id=f"align_{len(links)+1:04d}",
                    pdf_unit_id=pdf_unit.unit_id,
                    docx_unit_id=docx_unit.unit_id,
                    pdf_page_no=pdf_unit.page_no,
                    docx_estimated_page_no=docx_unit.estimated_page_no,
                    alignment_type=alignment_type,
                    confidence=score,
                    status=status,
                    anchors=anchors[:12],
                    reasons=reasons,
                )
            )
        return links

    def _alignment_score(
        self,
        pdf_unit: PdfEvidenceUnit,
        docx_unit: DocxEvidenceUnit,
        *,
        pdf_anchors: List[Dict[str, str]],
    ) -> Tuple[float, List[Dict[str, str]], List[str]]:
        left = pdf_unit.normalized_text
        right = docx_unit.normalized_text
        base = similarity(left, right)
        docx_anchor_values = {item["value"] for item in self._anchors(docx_unit.text)}
        matched_anchors = [item for item in pdf_anchors if item["value"] in docx_anchor_values]
        hard_matches = [item for item in matched_anchors if item["kind"] in {"id_number", "mobile", "date", "case_no", "long_number", "contract_no"}]
        reasons: List[str] = []
        score = base
        if matched_anchors:
            score = max(score, min(0.98, 0.55 + len(matched_anchors) * 0.08))
            reasons.append("shared_anchors")
        if hard_matches:
            score = max(score, min(0.99, 0.72 + len(hard_matches) * 0.07))
            reasons.append("shared_hard_fields")
        if left and right and (left in right or right in left):
            score = max(score, min(len(left), len(right)) / max(len(left), len(right)))
            reasons.append("text_containment")
        if pdf_unit.page_no and docx_unit.estimated_page_no and abs(int(pdf_unit.page_no) - int(docx_unit.estimated_page_no)) <= 1:
            score = min(1.0, score + 0.03)
            reasons.append("page_proximity")
        return round(float(score), 4), matched_anchors, reasons or ["text_similarity"]

    def _build_diff_candidates(
        self,
        *,
        pdf_units: Sequence[PdfEvidenceUnit],
        docx_units: Sequence[DocxEvidenceUnit],
        links: Sequence[AlignmentLink],
        page_profiles: Dict[str, Dict[str, Any]] | None = None,
    ) -> List[ConversionDiffCandidate]:
        pdf_by_id = {item.unit_id: item for item in pdf_units}
        docx_by_id = {item.unit_id: item for item in docx_units}
        diffs: List[ConversionDiffCandidate] = []
        linked_pdf = {link.pdf_unit_id for link in links}
        linked_docx = {link.docx_unit_id for link in links}
        uncertain_docx_pages = {
            int(link.docx_estimated_page_no)
            for link in links
            if link.status == "mapping_uncertain" and link.docx_estimated_page_no is not None
        }
        matched_docx_pages = {
            int(link.docx_estimated_page_no)
            for link in links
            if link.status == "matched" and link.docx_estimated_page_no is not None
        }
        anchor_ocr_pages = {
            int(unit.page_no)
            for unit in pdf_units
            if unit.unit_type == "anchor_ocr_page" and len(unit.normalized_text) > 0
        }
        uncertain_by_page: Dict[int, List[AlignmentLink]] = {}

        for link in links:
            pdf_unit = pdf_by_id.get(link.pdf_unit_id)
            docx_unit = docx_by_id.get(link.docx_unit_id)
            if not pdf_unit or not docx_unit:
                continue
            if link.status == "mapping_uncertain":
                uncertain_by_page.setdefault(int(link.pdf_page_no), []).append(link)
                continue
            self._append_text_diffs(diffs, pdf_unit=pdf_unit, docx_unit=docx_unit, alignment_confidence=link.confidence)

        for page_no, page_links in sorted(uncertain_by_page.items()):
            matched_on_page = sum(1 for link in links if link.status == "matched" and int(link.pdf_page_no) == page_no)
            if len(page_links) < 3 and matched_on_page >= 2:
                continue
            sample = page_links[0]
            pdf_unit = pdf_by_id.get(sample.pdf_unit_id)
            docx_unit = docx_by_id.get(sample.docx_unit_id)
            if not pdf_unit or not docx_unit:
                continue
            diff = self._diff(
                diffs,
                "mapping_uncertain",
                "medium",
                pdf_unit=pdf_unit,
                docx_unit=docx_unit,
                alignment_confidence=max(link.confidence for link in page_links),
                confidence=max(link.confidence for link in page_links),
                reason=f"本页存在 {len(page_links)} 个不确定 PDF-DOCX 对齐链接，禁止字段级判断；完整链接见 alignment_links.json。",
            )
            diff.flags.append(f"uncertain_link_count={len(page_links)}")
            diffs.append(diff)

        for unit in pdf_units:
            if unit.unit_type == "native_text_block" and unit.unit_id not in linked_pdf and len(unit.normalized_text) >= 12:
                diffs.append(
                    self._diff(
                        diffs,
                        "missing_content",
                        "high",
                        pdf_unit=unit,
                        alignment_confidence=0.0,
                        confidence=0.72,
                        reason="PDF 原生文本块未找到可靠 DOCX 对应内容。",
                    )
                )
            elif unit.unit_type in {"visual_region", "table_region"}:
                if "needs_table_parser" in unit.flags:
                    diffs.append(
                        self._diff(
                            diffs,
                            "table_structure_suspect",
                            "medium",
                            pdf_unit=unit,
                            confidence=unit.confidence,
                            reason="PDF 视觉区域呈现表格特征，当前版本仅标记为后续表格结构解析。",
                        )
                    )
                elif "needs_ocr" in unit.flags and unit.region_type == "visual_content":
                    if int(unit.page_no) in anchor_ocr_pages:
                        continue
                    diffs.append(
                        self._diff(
                            diffs,
                            "visual_region_unresolved",
                            "medium",
                            pdf_unit=unit,
                            confidence=unit.confidence,
                            reason="PDF 视觉区域存在内容但原生文本不可靠，需要局部 OCR 或视觉复核。",
                        )
                    )

        comparable_pdf_pages = self._comparable_pdf_pages(pdf_units)
        for unit in docx_units:
            if unit.unit_id not in linked_docx and len(unit.normalized_text) >= 12:
                if unit.estimated_page_no is not None and int(unit.estimated_page_no) not in comparable_pdf_pages:
                    continue
                if unit.estimated_page_no is not None and int(unit.estimated_page_no) in uncertain_docx_pages:
                    continue
                if unit.estimated_page_no is not None and int(unit.estimated_page_no) not in matched_docx_pages:
                    continue
                diffs.append(
                    self._diff(
                        diffs,
                        "extra_content",
                        "medium",
                        docx_unit=unit,
                        confidence=0.68,
                        reason="DOCX 文本单元未找到可靠 PDF 对应内容。",
                    )
                )

        self._append_duplicate_diffs(diffs, docx_units, uncertain_docx_pages=uncertain_docx_pages)
        self._append_reading_order_diff(diffs, links=links, pdf_by_id=pdf_by_id, docx_by_id=docx_by_id)
        self._append_recall_guard_diffs(
            diffs,
            pdf_units=pdf_units,
            docx_units=docx_units,
            links=links,
            page_profiles=page_profiles or {},
        )
        return diffs

    def _append_recall_guard_diffs(
        self,
        diffs: List[ConversionDiffCandidate],
        *,
        pdf_units: Sequence[PdfEvidenceUnit],
        docx_units: Sequence[DocxEvidenceUnit],
        links: Sequence[AlignmentLink],
        page_profiles: Dict[str, Dict[str, Any]],
    ) -> None:
        """Add conservative recall-safety candidates for things the main matcher missed.

        These candidates are not conversion findings. They exist so later review
        stages can see high-risk hard fields or complex pages that never became
        a normal text/table diff.
        """
        matched_pdf_ids = {link.pdf_unit_id for link in links if link.status == "matched"}
        matched_docx_ids = {link.docx_unit_id for link in links if link.status == "matched"}
        diff_pdf_ids = {item.pdf_unit_id for item in diffs if item.pdf_unit_id}
        diff_docx_ids = {item.docx_unit_id for item in diffs if item.docx_unit_id}
        pdf_by_page: Dict[int, List[PdfEvidenceUnit]] = {}
        docx_by_page: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in pdf_units:
            pdf_by_page.setdefault(int(unit.page_no), []).append(unit)
        for unit in docx_units:
            if unit.estimated_page_no is not None:
                docx_by_page.setdefault(int(unit.estimated_page_no), []).append(unit)

        for docx_unit in sorted(docx_units, key=lambda item: item.order_index):
            if not docx_unit.normalized_text or docx_unit.unit_id in matched_docx_ids or docx_unit.unit_id in diff_docx_ids:
                continue
            fields = self._fields(docx_unit.text)
            hard_fields = self._hard_field_items(fields)
            if not hard_fields:
                continue
            page_no = int(docx_unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            profile = page_profiles.get(str(page_no)) or {}
            if not self._recall_guard_page_enabled(profile):
                continue
            pdf_unit, field = self._best_recall_pdf_unit(page_no=page_no, field_items=hard_fields, pdf_by_page=pdf_by_page)
            diff = self._diff(
                diffs,
                "unlocated_hard_field",
                "medium",
                pdf_unit=pdf_unit,
                docx_unit=docx_unit,
                confidence=0.58,
                reason="DOCX 高风险字段未进入高置信对齐或差异候选，作为漏检兜底任务进入复核。",
                field_type=field.get("field_type") or "",
                field_role=field.get("role") or "recall_guard",
                pdf_value=field.get("pdf_value") or "",
                docx_value=field.get("value") or "",
            )
            diff.flags.extend(["recall_guard", "unmatched_docx_hard_field", f"field_count={len(hard_fields)}"])
            if field.get("pdf_value"):
                diff.flags.append("recall_matched_pdf_field")
            else:
                diff.pdf_text = ""
                diff.flags.append("recall_pdf_field_not_matched")
            if pdf_unit is not None:
                diff.flags.append(f"recall_pdf_source={pdf_unit.source}")
            diffs.append(diff)
            diff_docx_ids.add(docx_unit.unit_id)

        matched_pages = {int(link.pdf_page_no) for link in links if link.status == "matched"}
        existing_gap_pages = {int(item.pdf_page_no or 0) for item in diffs if item.category in {"mapping_uncertain", "page_coverage_gap"}}
        for page_key, profile in sorted(page_profiles.items(), key=lambda item: int(item[0])):
            page_no = int(page_key)
            if page_no in matched_pages or page_no in existing_gap_pages:
                continue
            if not self._recall_guard_page_enabled(profile):
                continue
            pdf_unit = self._best_visual_or_text_unit(pdf_by_page.get(page_no) or [])
            docx_unit = self._best_docx_anchor_unit(docx_by_page.get(page_no) or [])
            if pdf_unit is None and docx_unit is None:
                continue
            diff = self._diff(
                diffs,
                "page_coverage_gap",
                "medium",
                pdf_unit=pdf_unit,
                docx_unit=docx_unit,
                confidence=0.54,
                reason="复杂页没有建立高置信 PDF-DOCX 覆盖链路，可能存在前序候选漏召回。",
            )
            diff.flags.extend(["recall_guard", "page_level_coverage_gap"])
            if profile.get("labels"):
                diff.flags.append("page_labels=" + ",".join(str(item) for item in profile.get("labels") or []))
            diffs.append(diff)

    def _hard_field_items(self, fields: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        hard_order = ("id_number", "mobile", "case_no", "contract_no", "long_number", "amount", "date")
        items: List[Dict[str, str]] = []
        for field_type in hard_order:
            for item in fields.get(field_type) or []:
                row = dict(item)
                row["field_type"] = field_type
                items.append(row)
        return items

    def _recall_guard_page_enabled(self, profile: Dict[str, Any]) -> bool:
        labels = set(profile.get("labels") or [])
        if profile.get("needs_ocr") or profile.get("needs_table_parser") or profile.get("table_like"):
            return True
        return bool(labels.intersection({"scan_like", "mixed_layout", "table_heavy", "image_text_heavy", "mapping_risk"}))

    def _best_recall_pdf_unit(
        self,
        *,
        page_no: int,
        field_items: Sequence[Dict[str, str]],
        pdf_by_page: Dict[int, List[PdfEvidenceUnit]],
    ) -> Tuple[PdfEvidenceUnit | None, Dict[str, str]]:
        units = pdf_by_page.get(page_no) or []
        if not units:
            return None, dict(field_items[0] if field_items else {})
        for field in field_items:
            value = str(field.get("normalized_value") or field.get("value") or "")
            if not value:
                continue
            for unit in units:
                if value and value in normalize_text(unit.text):
                    row = dict(field)
                    row["pdf_value"] = value
                    return unit, row
        return self._best_visual_or_text_unit(units), dict(field_items[0] if field_items else {})

    def _best_visual_or_text_unit(self, units: Sequence[PdfEvidenceUnit]) -> PdfEvidenceUnit | None:
        if not units:
            return None
        priority = {
            "native_text_block": 0,
            "anchor_ocr_line": 1,
            "anchor_ocr_page": 2,
            "table_region": 3,
            "visual_region": 4,
        }
        ordered = sorted(
            units,
            key=lambda item: (
                priority.get(item.unit_type, 99),
                -len(item.normalized_text or ""),
                item.order_index,
            ),
        )
        return ordered[0]

    def _best_docx_anchor_unit(self, units: Sequence[DocxEvidenceUnit]) -> DocxEvidenceUnit | None:
        if not units:
            return None
        hard = [unit for unit in units if self._hard_field_items(self._fields(unit.text))]
        candidates = hard or [unit for unit in units if len(unit.normalized_text) >= 8] or list(units)
        return sorted(candidates, key=lambda item: (-len(item.normalized_text or ""), item.order_index))[0]

    def _comparable_pdf_pages(self, pdf_units: Sequence[PdfEvidenceUnit]) -> set[int]:
        pages: set[int] = set()
        for unit in pdf_units:
            if unit.unit_type == "native_text_block" and len(unit.normalized_text) >= 12:
                pages.add(int(unit.page_no))
            elif unit.unit_type == "anchor_ocr_page" and len(unit.normalized_text) >= 50 and "low_confidence_ocr" not in set(unit.flags):
                pages.add(int(unit.page_no))
        return pages

    def _append_text_diffs(
        self,
        diffs: List[ConversionDiffCandidate],
        *,
        pdf_unit: PdfEvidenceUnit,
        docx_unit: DocxEvidenceUnit,
        alignment_confidence: float,
    ) -> None:
        text_score = similarity(pdf_unit.normalized_text, docx_unit.normalized_text)
        pdf_fields = self._fields(pdf_unit.text)
        docx_fields = self._fields(docx_unit.text)
        for field_type in sorted(set(pdf_fields) & set(docx_fields)):
            pdf_values = {item["normalized_value"] for item in pdf_fields[field_type]}
            docx_values = {item["normalized_value"] for item in docx_fields[field_type]}
            if not self._should_emit_critical_field_diff(field_type=field_type, pdf_values=pdf_values, docx_values=docx_values):
                continue
            first_pdf = pdf_fields[field_type][0]
            first_docx = docx_fields[field_type][0]
            diffs.append(
                self._diff(
                    diffs,
                    "critical_field_changed",
                    "high",
                    pdf_unit=pdf_unit,
                    docx_unit=docx_unit,
                    alignment_confidence=alignment_confidence,
                    confidence=max(0.76, alignment_confidence),
                    reason="同一高置信对齐上下文内关键字段值不同，且两侧没有共同字段值。",
                    field_type=field_type,
                    field_role=first_pdf.get("role") or first_docx.get("role") or "same_aligned_unit",
                    pdf_value=first_pdf.get("value") or "",
                    docx_value=first_docx.get("value") or "",
                )
            )
        if (
            text_score < 0.98
            and len(pdf_unit.normalized_text) >= 8
            and len(docx_unit.normalized_text) >= 8
            and self._should_emit_text_diff(pdf_unit.text, docx_unit.text, pdf_fields=pdf_fields, docx_fields=docx_fields)
        ):
            category = "table_cell_mismatch_suspect" if docx_unit.container_type == "table_cell" else "text_substitution"
            diffs.append(
                self._diff(
                    diffs,
                    category,
                    "medium",
                    pdf_unit=pdf_unit,
                    docx_unit=docx_unit,
                    alignment_confidence=alignment_confidence,
                    confidence=max(0.55, alignment_confidence * (1.0 - text_score)),
                    reason="PDF 与 DOCX 在同一对齐单元内文本不完全一致。",
                )
            )

    def _should_emit_text_diff(
        self,
        pdf_text: str,
        docx_text: str,
        *,
        pdf_fields: Dict[str, List[Dict[str, str]]],
        docx_fields: Dict[str, List[Dict[str, str]]],
    ) -> bool:
        left = self._semantic_compact(pdf_text)
        right = self._semantic_compact(docx_text)
        if not left or not right or left == right:
            return False
        if left in right or right in left:
            return False
        for field_type in sorted(set(pdf_fields) & set(docx_fields)):
            pdf_values = {item["normalized_value"] for item in pdf_fields[field_type]}
            docx_values = {item["normalized_value"] for item in docx_fields[field_type]}
            if pdf_values and docx_values and pdf_values.intersection(docx_values):
                if min(len(left), len(right)) <= 32 or similarity(left, right) >= 0.86:
                    return False
        if self._numeric_tokens(left) and self._numeric_tokens(left).issubset(self._numeric_tokens(right)):
            return False
        if self._numeric_tokens(right) and self._numeric_tokens(right).issubset(self._numeric_tokens(left)):
            return False
        if not (self._numeric_tokens(left) or self._numeric_tokens(right) or set(pdf_fields) or set(docx_fields)):
            return False
        return True

    def _should_emit_critical_field_diff(self, *, field_type: str, pdf_values: set[str], docx_values: set[str]) -> bool:
        if not pdf_values or not docx_values:
            return False
        if pdf_values == docx_values or pdf_values.intersection(docx_values):
            return False
        if field_type in {"date", "amount"}:
            return len(pdf_values) == 1 and len(docx_values) == 1
        return True

    def _semantic_compact(self, text: str) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("（", "(").replace("）", ")")
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"(\d{4})部(?=\d{1,2}月)", r"\1年", value)
        value = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日", lambda m: f"{m.group(1)}/{int(m.group(2))}/{int(m.group(3))}", value)
        value = re.sub(r"(\d{4})[-.](\d{1,2})[-.](\d{1,2})", lambda m: f"{m.group(1)}/{int(m.group(2))}/{int(m.group(3))}", value)
        value = value.replace("年", "").replace("月", "").replace("日", "")
        value = re.sub(r"\s+", "", value)
        return PUNCT_RE.sub("", value)

    def _numeric_tokens(self, text: str) -> set[str]:
        return {item for item in re.findall(r"\d+(?:\.\d+)?", str(text or "")) if len(re.sub(r"\D", "", item)) >= 2}


    def _append_duplicate_diffs(
        self,
        diffs: List[ConversionDiffCandidate],
        docx_units: Sequence[DocxEvidenceUnit],
        *,
        uncertain_docx_pages: set[int],
    ) -> None:
        eligible = [
            item
            for item in docx_units
            if item.container_type != "table_cell"
            and len(item.normalized_text) >= 24
            and not self._looks_like_numeric_table_text(item.normalized_text)
            and (item.estimated_page_no is None or int(item.estimated_page_no) not in uncertain_docx_pages)
        ]
        counts = Counter(item.normalized_text for item in eligible)
        seen: set[str] = set()
        for unit in eligible:
            if counts.get(unit.normalized_text, 0) <= 1 or unit.normalized_text in seen:
                continue
            seen.add(unit.normalized_text)
            diffs.append(
                self._diff(
                    diffs,
                    "duplicate_content",
                    "medium",
                    docx_unit=unit,
                    confidence=0.7,
                    reason=f"DOCX 中存在重复文本单元，重复次数 {counts[unit.normalized_text]}。",
                )
            )

    def _looks_like_numeric_table_text(self, text: str) -> bool:
        compact = normalize_text(text)
        if not compact:
            return False
        digits = sum(1 for char in compact if char.isdigit())
        chinese = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
        return digits >= 6 and digits > chinese * 2

    def _append_reading_order_diff(
        self,
        diffs: List[ConversionDiffCandidate],
        *,
        links: Sequence[AlignmentLink],
        pdf_by_id: Dict[str, PdfEvidenceUnit],
        docx_by_id: Dict[str, DocxEvidenceUnit],
    ) -> None:
        matched = [link for link in links if link.status == "matched"]
        if any((pdf_by_id.get(link.pdf_unit_id) and pdf_by_id[link.pdf_unit_id].unit_type.startswith("anchor_ocr")) for link in matched):
            return
        ordered = sorted(
            matched,
            key=lambda link: (pdf_by_id.get(link.pdf_unit_id).page_no if pdf_by_id.get(link.pdf_unit_id) else 0, pdf_by_id.get(link.pdf_unit_id).order_index if pdf_by_id.get(link.pdf_unit_id) else 0),
        )
        previous_order = -1
        for link in ordered:
            docx = docx_by_id.get(link.docx_unit_id)
            pdf = pdf_by_id.get(link.pdf_unit_id)
            if not docx or not pdf:
                continue
            if previous_order >= 0 and docx.order_index + 3 < previous_order:
                diffs.append(
                    self._diff(
                        diffs,
                        "reading_order_changed",
                        "medium",
                        pdf_unit=pdf,
                        docx_unit=docx,
                        alignment_confidence=link.confidence,
                        confidence=0.66,
                        reason="已匹配内容在 DOCX 中的顺序相对 PDF 出现明显倒置。",
                    )
                )
                return
            previous_order = max(previous_order, docx.order_index)

    def _build_routes(
        self,
        *,
        pdf_units: Sequence[PdfEvidenceUnit],
        docx_units: Sequence[DocxEvidenceUnit],
        links: Sequence[AlignmentLink],
        diffs: Sequence[ConversionDiffCandidate],
        page_profiles: Dict[str, Dict[str, Any]],
    ) -> List[ReviewRoute]:
        routes: List[ReviewRoute] = []
        uncertain_links_by_page: Dict[int, List[AlignmentLink]] = {}
        matched_count_by_page: Counter[int] = Counter()
        for link in links:
            if link.status == "mapping_uncertain":
                uncertain_links_by_page.setdefault(int(link.pdf_page_no), []).append(link)
            elif link.status == "matched":
                matched_count_by_page[int(link.pdf_page_no)] += 1
        for page_key, profile in sorted(page_profiles.items(), key=lambda item: int(item[0])):
            page_no = int(page_key)
            if profile.get("needs_table_parser"):
                routes.append(self._route(routes, "needs_table_parser", "PDF 页面呈现表格/网格特征，后续需要表格结构解析。", page_no=page_no, priority=80))
            anchor_chars = int(profile.get("anchor_ocr_text_chars") or 0)
            if profile.get("needs_ocr") and anchor_chars <= 0:
                routes.append(self._route(routes, "needs_anchor_ocr", "PDF 原生文本不可靠，后续先抽取页级标题、案号、身份证号、金额、日期等轻量锚点。", page_no=page_no, priority=70))
            elif profile.get("needs_ocr") and anchor_chars < 40 and not profile.get("needs_table_parser"):
                routes.append(self._route(routes, "needs_region_ocr", "本页轻量 OCR 文本过少，后续需要局部 OCR 或视觉模型补证。", page_no=page_no, priority=75))
            if "mapping_risk" in set(profile.get("labels") or []):
                routes.append(self._route(routes, "needs_human_mapping_review", "页面证据不足或类型不明，需人工确认映射。", page_no=page_no, priority=60))
        matched_route_pages: set[int] = set()
        uncertain_route_pages: set[int] = set()
        for link in links:
            if link.status == "matched":
                if int(link.pdf_page_no) in matched_route_pages:
                    continue
                matched_route_pages.add(int(link.pdf_page_no))
                routes.append(
                    self._route(
                        routes,
                        "text_compare",
                        "PDF 与 DOCX 已建立高置信锚点对齐，可进行本页文本一致性比较。",
                        page_no=link.pdf_page_no,
                        priority=40,
                    )
            )
            elif link.status == "mapping_uncertain":
                continue
        for page_no, page_links in sorted(uncertain_links_by_page.items()):
            if page_no in uncertain_route_pages:
                continue
            if len(page_links) < 3 and matched_count_by_page.get(page_no, 0) >= 2:
                continue
            uncertain_route_pages.add(page_no)
            routes.append(
                self._route(
                    routes,
                    "needs_human_mapping_review",
                    f"本页存在 {len(page_links)} 个材料性不确定对齐链接，禁止自动字段判断。",
                    page_no=page_no,
                    priority=90,
                )
            )
        for diff in diffs:
            if diff.category == "critical_field_changed":
                routes.append(self._route(routes, "text_compare", "高置信对齐内关键字段不同，当前版本记录为候选差异。", page_no=diff.pdf_page_no, pdf_unit_id=diff.pdf_unit_id, docx_unit_id=diff.docx_unit_id, diff_id=diff.diff_id, priority=95))
            elif diff.category in {"table_structure_suspect", "table_cell_mismatch_suspect", "visual_region_unresolved"}:
                continue
            elif diff.category == "mapping_uncertain":
                continue
            elif diff.category == "unlocated_hard_field":
                routes.append(
                    self._route(
                        routes,
                        "needs_recall_guard_review",
                        "高风险字段未被主候选链路覆盖，进入漏检兜底复核。",
                        page_no=diff.pdf_page_no or diff.docx_estimated_page_no,
                        pdf_unit_id=diff.pdf_unit_id,
                        docx_unit_id=diff.docx_unit_id,
                        diff_id=diff.diff_id,
                        priority=88,
                    )
                )
            elif diff.category == "page_coverage_gap":
                routes.append(
                    self._route(
                        routes,
                        "needs_recall_guard_review",
                        "复杂页缺少高置信覆盖链路，进入页级漏检兜底复核。",
                        page_no=diff.pdf_page_no or diff.docx_estimated_page_no,
                        pdf_unit_id=diff.pdf_unit_id,
                        docx_unit_id=diff.docx_unit_id,
                        diff_id=diff.diff_id,
                        priority=82,
                    )
                )
        return routes

    def _route(
        self,
        routes: Sequence[ReviewRoute],
        route: str,
        reason: str,
        *,
        page_no: int | None = None,
        pdf_unit_id: str = "",
        docx_unit_id: str = "",
        diff_id: str = "",
        priority: int = 0,
    ) -> ReviewRoute:
        return ReviewRoute(
            route_id=f"route_{len(routes)+1:04d}",
            route=route,
            reason=reason,
            page_no=page_no,
            pdf_unit_id=pdf_unit_id,
            docx_unit_id=docx_unit_id,
            diff_id=diff_id,
            priority=priority,
        )

    def _diff(
        self,
        diffs: Sequence[ConversionDiffCandidate],
        category: str,
        risk: str,
        *,
        pdf_unit: PdfEvidenceUnit | None = None,
        docx_unit: DocxEvidenceUnit | None = None,
        alignment_confidence: float = 0.0,
        confidence: float = 0.0,
        reason: str = "",
        field_type: str = "",
        field_role: str = "",
        pdf_value: str = "",
        docx_value: str = "",
    ) -> ConversionDiffCandidate:
        return ConversionDiffCandidate(
            diff_id=f"diff_{len(diffs)+1:04d}",
            category=category,
            risk=risk,
            pdf_unit_id=pdf_unit.unit_id if pdf_unit else "",
            docx_unit_id=docx_unit.unit_id if docx_unit else "",
            pdf_page_no=pdf_unit.page_no if pdf_unit else None,
            docx_estimated_page_no=docx_unit.estimated_page_no if docx_unit else None,
            pdf_text=pdf_unit.text if pdf_unit else "",
            docx_text=docx_unit.text if docx_unit else "",
            field_type=field_type,
            field_role=field_role,
            pdf_value=pdf_value,
            docx_value=docx_value,
            alignment_confidence=alignment_confidence,
            confidence=confidence,
            reason=reason,
        )

    def _anchors(self, text: str) -> List[Dict[str, str]]:
        value = str(text or "")
        anchors: List[Dict[str, str]] = []
        for kind, pattern in (
            ("id_number", ID_RE),
            ("mobile", MOBILE_RE),
            ("date", DATE_RE),
            ("case_no", CASE_NO_RE),
            ("contract_no", CONTRACT_RE),
            ("long_number", LONG_NUMBER_RE),
        ):
            for match in pattern.finditer(value):
                token = normalize_text(match.group(0))
                if token:
                    anchors.append({"kind": kind, "value": token})
        for amount in self._amount_tokens(value):
            anchors.append({"kind": "amount", "value": amount})
        compact = PUNCT_RE.sub("", value)
        if len(compact) >= 12:
            chunks = [compact[:12], compact[-12:]]
            midpoint = max(0, len(compact) // 2 - 6)
            chunks.append(compact[midpoint : midpoint + 12])
            for chunk in chunks:
                token = normalize_text(chunk)
                if len(token) >= 6:
                    if re.fullmatch(r"\d+", token):
                        continue
                    anchors.append({"kind": "phrase", "value": token})
        result: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in anchors:
            key = (item["kind"], item["value"])
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result[:40]

    def _fields(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        value = str(text or "")
        fields: Dict[str, List[Dict[str, str]]] = {}
        for field_type, pattern in (
            ("id_number", ID_RE),
            ("mobile", MOBILE_RE),
            ("date", DATE_RE),
            ("case_no", CASE_NO_RE),
            ("contract_no", CONTRACT_RE),
        ):
            for match in pattern.finditer(value):
                self._add_field(fields, field_type, match.group(0), role=self._role(value, match.start()))
        for amount in self._amount_tokens(value):
            self._add_field(fields, "amount", amount, role=self._role(value, value.find(amount)))
        return fields

    def _add_field(self, fields: Dict[str, List[Dict[str, str]]], field_type: str, value: str, *, role: str) -> None:
        normalized = normalize_text(value)
        if not normalized:
            return
        fields.setdefault(field_type, []).append({"value": value.strip(), "normalized_value": normalized, "role": role or "same_aligned_unit"})

    def _amount_tokens(self, text: str) -> List[str]:
        result: List[str] = []
        for match in AMOUNT_RE.finditer(text):
            token = match.group(0).strip()
            digits = re.sub(r"\D", "", token)
            if len(digits) < 2:
                continue
            has_money_marker = any(marker in token for marker in ("元", "万", "¥", "￥", "人民币", "RMB"))
            context = text[max(0, match.start() - 16) : match.start()]
            has_amount_context = bool(re.search(r"(金额|价款|款项|合计|小计|总计|amount|total)", context, flags=re.IGNORECASE))
            if not has_money_marker and not has_amount_context:
                continue
            result.append(normalize_text(token))
        return result[:20]

    def _role(self, text: str, offset: int) -> str:
        if offset < 0:
            return "same_aligned_unit"
        before = text[max(0, offset - 18) : offset]
        before = re.sub(r"\s+", "", before)
        before = re.sub(r"^[,，。；;:：、\s]+", "", before)
        return before[-16:] or "same_aligned_unit"
