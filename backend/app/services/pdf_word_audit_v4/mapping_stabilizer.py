from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .common import normalize_text, similarity
from .docx_fragment_merger import DocxFragmentMerger
from .models import AlignmentLink, ConversionDiffCandidate, ConversionPreflightResult, DocxEvidenceUnit, PdfEvidenceUnit


HARD_ANCHOR_RE = re.compile(
    r"(?P<id>\d{17}[\dXx]|\d{15})"
    r"|(?P<mobile>1[3-9]\d{9})"
    r"|(?P<date>\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
    r"|(?P<long_number>\d{8,25})"
    r"|(?P<case_no>[（(]?\d{4}[）)]?[\u4e00-\u9fff]{1,6}\d{2,8}[\u4e00-\u9fff]{1,8}\d+号)"
)


class MappingStabilizer:
    """Conservatively stabilize mapping-uncertain links before downstream review."""

    VERSION = "mapping_stabilizer_v1"

    def __init__(self, *, min_score: float = 0.82) -> None:
        self.min_score = max(0.0, min(1.0, float(min_score or 0.0)))
        self.fragment_merger = DocxFragmentMerger()

    def apply(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        fragment_payload = self.fragment_merger.build(docx_units=preflight_result.docx_units)
        pdf_by_id = {unit.unit_id: unit for unit in preflight_result.pdf_units}
        docx_by_id = {unit.unit_id: unit for unit in preflight_result.docx_units}
        links_by_pair = {(link.pdf_unit_id, link.docx_unit_id): link for link in preflight_result.alignment_links}
        reviews: List[Dict[str, Any]] = []
        resolved_count = 0
        unresolved_count = 0
        for diff in preflight_result.diff_candidates:
            if diff.category != "mapping_uncertain":
                continue
            pdf_unit = pdf_by_id.get(diff.pdf_unit_id)
            docx_unit = docx_by_id.get(diff.docx_unit_id)
            if pdf_unit is None or docx_unit is None:
                unresolved_count += 1
                reviews.append(self._review(diff=diff, decision="unresolved", reason="缺少 PDF 或 DOCX 对齐单元，不能稳定映射。"))
                continue
            review = self._evaluate(diff=diff, pdf_unit=pdf_unit, docx_unit=docx_unit, fragment_payload=fragment_payload)
            reviews.append(review)
            if review["decision"] != "resolved":
                unresolved_count += 1
                diff.flags.append("mapping_stabilizer_unresolved")
                continue
            resolved_count += 1
            diff.category = "mapping_stabilized"
            diff.risk = "low"
            diff.confidence = max(float(diff.confidence or 0.0), float(review.get("confidence") or 0.0))
            diff.reason = f"{diff.reason} 映射稳定器补充判断：{review['reason']}"[:500]
            diff.flags.extend(
                [
                    "mapping_stabilized",
                    f"mapping_stabilizer_confidence={float(review.get('confidence') or 0.0):.2f}",
                    f"mapping_stabilizer_signal={review.get('signal') or 'unknown'}",
                ]
            )
            link = links_by_pair.get((diff.pdf_unit_id, diff.docx_unit_id))
            if link is not None:
                link.status = "matched"
                link.confidence = max(float(link.confidence or 0.0), float(review.get("confidence") or 0.0))
                if "mapping_stabilized" not in link.reasons:
                    link.reasons.append("mapping_stabilized")

        payload = {
            "enabled": True,
            "version": self.VERSION,
            "resolved_count": resolved_count,
            "unresolved_count": unresolved_count,
            "review_count": len(reviews),
            "docx_fragment_merger": fragment_payload,
            "reviews": reviews,
        }
        preflight_result.mapping_stabilization = payload
        return payload

    def _evaluate(
        self,
        *,
        diff: ConversionDiffCandidate,
        pdf_unit: PdfEvidenceUnit,
        docx_unit: DocxEvidenceUnit,
        fragment_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        pdf_text = str(pdf_unit.text or "")
        docx_text = str(docx_unit.text or "")
        fragments = self.fragment_merger.fragments_for_unit(payload=fragment_payload, unit_id=docx_unit.unit_id)
        best_fragment = max(fragments, key=lambda item: similarity(normalize_text(pdf_text), str(item.get("normalized_text") or "")), default={})
        merged_text = str(best_fragment.get("text") or docx_text)
        base_similarity = similarity(normalize_text(pdf_text), normalize_text(docx_text))
        merged_similarity = similarity(normalize_text(pdf_text), normalize_text(merged_text))
        pdf_anchors = self._anchors(pdf_text)
        docx_anchors = self._anchors(docx_text)
        merged_anchors = self._anchors(merged_text)
        shared_docx = sorted(pdf_anchors & docx_anchors)
        shared_merged = sorted(pdf_anchors & merged_anchors)
        page_distance = abs(int(pdf_unit.page_no or 0) - int(docx_unit.estimated_page_no or 0))
        signal = "none"
        confidence = max(base_similarity, merged_similarity)
        if shared_docx:
            signal = "shared_hard_anchor"
            confidence = max(confidence, min(0.96, 0.84 + len(shared_docx) * 0.04))
        elif shared_merged:
            signal = "shared_merged_fragment_anchor"
            confidence = max(confidence, min(0.94, 0.82 + len(shared_merged) * 0.04))
        elif merged_similarity >= 0.88 and page_distance <= 1:
            signal = "merged_fragment_similarity"
            confidence = max(confidence, 0.86)
        elif base_similarity >= 0.86 and page_distance <= 1:
            signal = "same_unit_similarity"
            confidence = max(confidence, 0.84)

        duplicate_risk = self._duplicate_risk(docx_text)
        if duplicate_risk:
            confidence = min(confidence, 0.74)
        if page_distance > 2 and not (shared_docx or shared_merged):
            confidence = min(confidence, 0.72)

        decision = "resolved" if confidence >= self.min_score and not duplicate_risk else "unresolved"
        reason = self._reason(
            signal=signal,
            confidence=confidence,
            base_similarity=base_similarity,
            merged_similarity=merged_similarity,
            shared_count=max(len(shared_docx), len(shared_merged)),
            duplicate_risk=duplicate_risk,
            page_distance=page_distance,
        )
        return self._review(
            diff=diff,
            decision=decision,
            reason=reason,
            confidence=confidence,
            signal=signal,
            pdf_unit_id=pdf_unit.unit_id,
            docx_unit_id=docx_unit.unit_id,
            page_no=pdf_unit.page_no,
            docx_page_no=docx_unit.estimated_page_no,
            fragment_id=str(best_fragment.get("fragment_id") or ""),
            shared_anchors=shared_docx or shared_merged,
            base_similarity=base_similarity,
            merged_similarity=merged_similarity,
        )

    def _review(self, *, diff: ConversionDiffCandidate, decision: str, reason: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "review_id": f"mapping_stabilizer_{diff.diff_id}",
            "diff_id": diff.diff_id,
            "decision": decision,
            "reason": reason,
            "confidence": round(float(kwargs.get("confidence") or 0.0), 4),
            "signal": str(kwargs.get("signal") or ""),
            "pdf_unit_id": str(kwargs.get("pdf_unit_id") or diff.pdf_unit_id or ""),
            "docx_unit_id": str(kwargs.get("docx_unit_id") or diff.docx_unit_id or ""),
            "page_no": kwargs.get("page_no", diff.pdf_page_no),
            "docx_page_no": kwargs.get("docx_page_no", diff.docx_estimated_page_no),
            "fragment_id": str(kwargs.get("fragment_id") or ""),
            "shared_anchors": list(kwargs.get("shared_anchors") or []),
            "base_similarity": round(float(kwargs.get("base_similarity") or 0.0), 4),
            "merged_similarity": round(float(kwargs.get("merged_similarity") or 0.0), 4),
        }

    def _anchors(self, text: str) -> set[str]:
        anchors: set[str] = set()
        for match in HARD_ANCHOR_RE.finditer(str(text or "")):
            value = match.group(0)
            if value:
                anchors.add(normalize_text(value))
        return anchors

    def _duplicate_risk(self, text: str) -> bool:
        compact = normalize_text(text)
        if len(compact) <= 4:
            return True
        low_value_tokens = {"身份证号码", "住址", "金额", "日期", "备注", "合同编号", "页眉合同资料"}
        return compact in low_value_tokens

    def _reason(
        self,
        *,
        signal: str,
        confidence: float,
        base_similarity: float,
        merged_similarity: float,
        shared_count: int,
        duplicate_risk: bool,
        page_distance: int,
    ) -> str:
        parts = [
            f"signal={signal}",
            f"confidence={confidence:.2f}",
            f"base_similarity={base_similarity:.2f}",
            f"merged_similarity={merged_similarity:.2f}",
            f"shared_hard_anchor_count={shared_count}",
            f"page_distance={page_distance}",
        ]
        if duplicate_risk:
            parts.append("duplicate_or_low_value_anchor_risk")
        if confidence >= self.min_score and not duplicate_risk:
            parts.append("达到保守映射稳定阈值，可解除 mapping_uncertain 阻塞。")
        else:
            parts.append("未达到保守映射稳定阈值，仍需人工或局部视觉复核。")
        return "；".join(parts)
