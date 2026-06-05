from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

from app.services.macos_vision_ocr import MacOSVisionOCRService
from .common import normalize_text
from .models import ConversionDiffCandidate, ConversionPreflightResult, FocusedCandidateReview, PdfEvidenceUnit

logger = logging.getLogger(__name__)


ACTIONABLE_CATEGORIES = {
    "critical_field_changed",
    "text_substitution",
    "table_cell_mismatch_suspect",
    "missing_content",
    "extra_content",
    "duplicate_content",
    "reading_order_changed",
    "unlocated_hard_field",
    "page_coverage_gap",
}


class FocusedReviewBuilder:
    """Prepare small, traceable review tasks after v4 preflight.

    This layer is intentionally conservative: it does not confirm WPS errors.
    It separates candidates that can later enter a visual/Qwen/table review
    from candidates blocked by mapping uncertainty or table structure.
    """

    MAX_REVIEWS = 9999

    def __init__(self, *, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.crop_dir = work_dir / "evidence" / "focused_crops"

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        rendered_pages: Dict[int, Path],
    ) -> List[FocusedCandidateReview]:
        pdf_by_id = {unit.unit_id: unit for unit in preflight_result.pdf_units}
        reviews: List[FocusedCandidateReview] = []
        for diff in self._reviewable_diffs(preflight_result.diff_candidates):
            pdf_unit = pdf_by_id.get(diff.pdf_unit_id)
            crop_path = self._crop_candidate(diff=diff, pdf_unit=pdf_unit, rendered_pages=rendered_pages, index=len(reviews) + 1)
            crop_ocr = self._ocr_crop(crop_path)
            reviews.append(self._review(diff=diff, pdf_unit=pdf_unit, crop_path=crop_path, crop_ocr=crop_ocr, index=len(reviews) + 1))
            if len(reviews) >= self.MAX_REVIEWS:
                break
        return reviews

    def _reviewable_diffs(self, diffs: Sequence[ConversionDiffCandidate]) -> List[ConversionDiffCandidate]:
        priority = {
            "critical_field_changed": 0,
            "text_substitution": 1,
            "table_cell_mismatch_suspect": 2,
            "table_structure_suspect": 3,
            "mapping_uncertain": 4,
            "visual_region_unresolved": 5,
            "missing_content": 6,
            "extra_content": 7,
            "unlocated_hard_field": 8,
            "page_coverage_gap": 9,
            "duplicate_content": 10,
            "reading_order_changed": 11,
        }
        return sorted(list(diffs), key=lambda item: (priority.get(item.category, 99), item.pdf_page_no or 9999, item.diff_id))

    def _review(
        self,
        *,
        diff: ConversionDiffCandidate,
        pdf_unit: PdfEvidenceUnit | None,
        crop_path: str,
        crop_ocr: Dict[str, str],
        index: int,
    ) -> FocusedCandidateReview:
        status, decision, next_route, reason, confidence, flags = self._classify(diff=diff, pdf_unit=pdf_unit, crop_path=crop_path, crop_ocr=crop_ocr)
        return FocusedCandidateReview(
            review_id=f"focused_{index:04d}",
            diff_id=diff.diff_id,
            category=diff.category,
            page_no=diff.pdf_page_no,
            status=status,
            decision=decision,
            confidence=confidence,
            reason=reason,
            next_route=next_route,
            crop_path=crop_path,
            crop_ocr_text=str(crop_ocr.get("text") or ""),
            crop_ocr_quality=str(crop_ocr.get("quality") or ""),
            pdf_text=diff.pdf_text,
            docx_text=diff.docx_text,
            flags=flags,
            visual_text=self._visual_text_evidence(diff=diff, crop_ocr=crop_ocr),
        )

    def _classify(
        self,
        *,
        diff: ConversionDiffCandidate,
        pdf_unit: PdfEvidenceUnit | None,
        crop_path: str,
        crop_ocr: Dict[str, str],
    ) -> tuple[str, str, str, str, float, List[str]]:
        flags: List[str] = []
        source = str(getattr(pdf_unit, "source", "") or "")
        if source:
            flags.append(f"pdf_source={source}")
        if crop_path:
            flags.append("has_local_crop")
        elif diff.pdf_page_no:
            flags.append("missing_local_crop")

        crop_signal = self._crop_signal(diff=diff, crop_ocr=crop_ocr)
        if crop_signal:
            flags.append(crop_signal)
        if crop_signal == "crop_ocr_supports_docx":
            return (
                "blocked_ocr_conflict",
                "no_confirmed_error",
                "",
                "局部 crop OCR 更支持 DOCX 文本，当前候选按 OCR 冲突处理，不确认 WPS 错误。",
                min(0.66, float(diff.alignment_confidence or diff.confidence or 0.0)),
                flags,
            )

        if diff.category == "critical_field_changed":
            if pdf_unit and pdf_unit.unit_type.startswith("anchor_ocr"):
                return (
                    "needs_visual_gate",
                    "needs_visual_review",
                    "needs_qwen_vl",
                    "关键字段差异来自 OCR 行，必须用局部截图或视觉模型复核后才能确认。",
                    min(0.74, float(diff.confidence or 0.0)),
                    flags,
                )
            return (
                "ready_for_final_gate",
                "possible_conversion_error",
                "qwen_text_gate",
                "关键字段差异来自可比较文本证据，可进入最终文本门槛。",
                max(0.76, float(diff.confidence or 0.0)),
                flags,
            )
        if diff.category == "text_substitution":
            return (
                "needs_visual_gate",
                "needs_visual_review",
                "needs_qwen_vl" if crop_path else "needs_region_ocr",
                "文本差异来自 OCR 对齐结果，当前只作为局部视觉复核任务，不直接确认 WPS 错误。",
                min(0.7, float(diff.alignment_confidence or diff.confidence or 0.0)),
                flags,
            )
        if diff.category == "table_cell_mismatch_suspect":
            return (
                "needs_table_parser",
                "needs_table_review",
                "needs_table_parser",
                "差异位于表格/类表格内容，必须先做单元格结构解析再比较。",
                min(0.68, float(diff.alignment_confidence or diff.confidence or 0.0)),
                flags,
            )
        if diff.category == "table_structure_suspect":
            return (
                "needs_table_parser",
                "needs_table_review",
                "needs_table_parser",
                "PDF 页面呈现表格特征，当前只进入表格解析路由。",
                0.62,
                flags,
            )
        if diff.category == "mapping_uncertain":
            return (
                "blocked_mapping_uncertain",
                "needs_mapping_review",
                "needs_human_mapping_review",
                "PDF-DOCX 映射不确定，禁止字段级或文本级错误确认。",
                min(0.62, float(diff.alignment_confidence or diff.confidence or 0.0)),
                flags,
            )
        if diff.category in {"missing_content", "extra_content", "duplicate_content", "reading_order_changed"}:
            return (
                "needs_context_gate",
                "needs_mapping_review",
                "needs_human_mapping_review",
                "内容级差异需要更稳定的页/区域上下文后再确认。",
                min(0.62, float(diff.confidence or 0.0)),
                flags,
            )
        if diff.category == "unlocated_hard_field":
            return (
                "recall_guard",
                "needs_recall_review",
                "needs_qwen_vl" if crop_path else "needs_human_mapping_review",
                "高风险字段没有进入高置信候选链路，作为漏检兜底批注任务，不直接判断字段错误。",
                min(0.6, float(diff.confidence or 0.0)),
                flags,
            )
        if diff.category == "page_coverage_gap":
            return (
                "recall_guard",
                "needs_recall_review",
                "needs_qwen_vl" if crop_path else "needs_human_mapping_review",
                "复杂页缺少高置信覆盖链路，提示可能存在前序候选漏召回。",
                min(0.56, float(diff.confidence or 0.0)),
                flags,
            )
        return (
            "deferred",
            "needs_review",
            "needs_human_mapping_review",
            "候选类别暂未进入自动确认路径。",
            min(0.5, float(diff.confidence or 0.0)),
            flags,
        )

    def _crop_signal(self, *, diff: ConversionDiffCandidate, crop_ocr: Dict[str, str]) -> str:
        crop_text = str(crop_ocr.get("text") or "")
        if not crop_text.strip():
            return ""
        crop = self._semantic_compact(crop_text)
        pdf = self._semantic_compact(diff.pdf_text)
        docx = self._semantic_compact(diff.docx_text)
        if not crop:
            return ""
        if docx and (docx in crop or crop in docx) and not (pdf and (pdf in crop or crop in pdf)):
            return "crop_ocr_supports_docx"
        if pdf and (pdf in crop or crop in pdf) and not (docx and (docx in crop or crop in docx)):
            return "crop_ocr_supports_pdf"
        return ""

    def _visual_text_evidence(self, *, diff: ConversionDiffCandidate, crop_ocr: Dict[str, str]) -> Dict[str, object]:
        if diff.category not in {"text_substitution", "critical_field_changed"}:
            return {}
        crop_text = str(crop_ocr.get("text") or "")
        quality = str(crop_ocr.get("quality") or "")
        pdf_tokens = self._number_tokens(diff.pdf_text)
        docx_tokens = self._number_tokens(diff.docx_text)
        crop_tokens = self._number_tokens(crop_text)
        shared = set(pdf_tokens).intersection(docx_tokens)
        pdf_unique = [item for item in pdf_tokens if item not in shared]
        docx_unique = [item for item in docx_tokens if item not in shared]
        crop_set = set(crop_tokens)
        pdf_hits = [item for item in pdf_unique if item in crop_set]
        docx_hits = [item for item in docx_unique if item in crop_set]
        raw_has_split_digits = self._has_split_numeric_token(crop_text, pdf_unique + docx_unique)
        flags: List[str] = []
        if quality:
            flags.append(f"crop_ocr_quality={quality}")
        if raw_has_split_digits:
            flags.append("split_numeric_token")
        if pdf_hits:
            flags.append("crop_numeric_hits_pdf")
        if docx_hits:
            flags.append("crop_numeric_hits_docx")
        stable = False
        support = "ambiguous"
        reason = "局部 OCR 未稳定支持 PDF 或 DOCX 的差异数字。"
        if pdf_unique and docx_unique and pdf_hits and not docx_hits and quality != "low" and not raw_has_split_digits:
            support = "pdf"
            stable = True
            reason = "局部 OCR 稳定命中 PDF 侧差异数字，且未命中 DOCX 侧差异数字。"
        elif pdf_unique and docx_unique and docx_hits and not pdf_hits and quality != "low" and not raw_has_split_digits:
            support = "docx"
            stable = True
            reason = "局部 OCR 稳定命中 DOCX 侧差异数字，且未命中 PDF 侧差异数字。"
        elif pdf_hits and docx_hits:
            support = "conflict"
            reason = "局部 OCR 同时命中 PDF 和 DOCX 的差异数字，不能确认转换错误。"
        elif quality == "low" or raw_has_split_digits:
            support = "low_quality_ambiguous"
            reason = "局部 OCR 质量低或差异数字被拆分/污染，不能作为确认错误依据。"
        return {
            "support": support,
            "stable": stable,
            "reason": reason,
            "crop_ocr_quality": quality,
            "pdf_number_tokens": pdf_tokens,
            "docx_number_tokens": docx_tokens,
            "crop_number_tokens": crop_tokens,
            "pdf_unique_tokens": pdf_unique,
            "docx_unique_tokens": docx_unique,
            "pdf_hits": pdf_hits,
            "docx_hits": docx_hits,
            "flags": flags,
        }

    def _number_tokens(self, text: str) -> List[str]:
        result: List[str] = []
        for token in re.findall(r"\d+", str(text or "")):
            if len(token) < 2:
                continue
            if token not in result:
                result.append(token)
        return result

    def _has_split_numeric_token(self, text: str, target_tokens: Sequence[str]) -> bool:
        raw = str(text or "")
        for token in target_tokens:
            if len(token) < 4 or token in raw:
                continue
            pattern = r""
            for index, char in enumerate(token):
                if index:
                    pattern += r"[\s_/\-\\.·]*"
                pattern += re.escape(char)
            if re.search(pattern, raw):
                return True
        return False

    def _crop_candidate(
        self,
        *,
        diff: ConversionDiffCandidate,
        pdf_unit: PdfEvidenceUnit | None,
        rendered_pages: Dict[int, Path],
        index: int,
    ) -> str:
        if not pdf_unit or not pdf_unit.bbox or not diff.pdf_page_no:
            return ""
        image_path = rendered_pages.get(int(diff.pdf_page_no))
        if image_path is None or not image_path.exists():
            return ""
        try:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image:
                left, top, right, bottom = self._padded_bbox(pdf_unit.bbox, width=image.width, height=image.height)
                if right <= left or bottom <= top:
                    return ""
                crop = image.crop((left, top, right, bottom))
                target = self.crop_dir / f"{diff.diff_id}_{index:04d}_p{int(diff.pdf_page_no):04d}.jpg"
                crop.convert("RGB").save(target, format="JPEG", quality=94)
            try:
                target.chmod(0o600)
            except Exception:
                pass
            return target.relative_to(self.work_dir).as_posix()
        except Exception:
            logger.debug("Failed to crop focused candidate %s.", diff.diff_id, exc_info=True)
            return ""

    def _ocr_crop(self, crop_path: str) -> Dict[str, str]:
        if not crop_path:
            return {}
        path = self.work_dir / crop_path
        if not path.exists() or not MacOSVisionOCRService.is_supported():
            return {}
        try:
            service = MacOSVisionOCRService(timeout=30)
            result = asyncio.run(service.extract_document_text_from_image_async(path.read_bytes()))
            return {
                "text": str(result.get("text") or "").strip(),
                "quality": str(result.get("quality") or "").strip(),
            }
        except Exception:
            logger.debug("Failed to OCR focused crop %s.", crop_path, exc_info=True)
            return {}

    def _semantic_compact(self, text: str) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("（", "(").replace("）", ")")
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"(\d{4})部(?=\d{1,2}月)", r"\1年", value)
        value = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日", lambda m: f"{m.group(1)}/{int(m.group(2))}/{int(m.group(3))}", value)
        value = re.sub(r"(\d{4})[-.](\d{1,2})[-.](\d{1,2})", lambda m: f"{m.group(1)}/{int(m.group(2))}/{int(m.group(3))}", value)
        value = value.replace("年", "").replace("月", "").replace("日", "")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _padded_bbox(self, bbox: Sequence[float], *, width: int, height: int) -> tuple[int, int, int, int]:
        left, top, right, bottom = [float(item) for item in bbox[:4]]
        pad_x = max(24.0, (right - left) * 0.35)
        pad_y = max(18.0, (bottom - top) * 1.2)
        return (
            max(0, int(left - pad_x)),
            max(0, int(top - pad_y)),
            min(width, int(right + pad_x)),
            min(height, int(bottom + pad_y)),
        )
