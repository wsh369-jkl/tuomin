from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from app.services.macos_vision_ocr import MacOSVisionOCRService

from .common import normalize_text
from .models import ConversionDiffCandidate, ConversionPreflightResult, DocxEvidenceUnit, FocusedCandidateReview, PdfEvidenceUnit, TableParseReview

logger = logging.getLogger(__name__)


class TableReviewBuilder:
    """Lightweight table-structure evidence for v4 report-only review.

    This is deliberately not a full OCR table recognizer. It answers a narrower
    question first: whether the PDF page has a parseable grid and whether DOCX
    contains table cells on the same estimated page. That lets the workflow
    route table candidates without trusting noisy full-page OCR.
    """

    def __init__(self, *, work_dir: Path | None = None) -> None:
        self.work_dir = work_dir
        self.cell_crop_dir = work_dir / "evidence" / "table_cells" if work_dir else None

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        rendered_pages: Dict[int, Path],
    ) -> List[TableParseReview]:
        target_pages = self._target_pages(preflight_result)
        if not target_pages:
            return []
        docx_table_stats = self._docx_table_stats(preflight_result.docx_units)
        diffs_by_page: Dict[int, List[str]] = defaultdict(list)
        diffs_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        pdf_by_id = {item.unit_id: item for item in preflight_result.pdf_units}
        focused_by_id = {item.diff_id: item for item in preflight_result.focused_reviews}
        for diff in preflight_result.diff_candidates:
            if diff.category in {"table_cell_mismatch_suspect", "table_structure_suspect"} and diff.pdf_page_no:
                diffs_by_page[int(diff.pdf_page_no)].append(diff.diff_id)

        reviews: List[TableParseReview] = []
        for page_no in sorted(target_pages):
            image_path = rendered_pages.get(page_no)
            if image_path is None or not image_path.exists():
                reviews.append(
                    TableParseReview(
                        table_id=f"table_p{page_no:04d}",
                        page_no=page_no,
                        status="missing_page_image",
                        confidence=0.0,
                        related_diff_ids=diffs_by_page.get(page_no, []),
                        docx_table_candidates=docx_table_stats.get(page_no, []),
                        flags=["missing_page_image"],
                    )
                )
                continue
            try:
                review = self._analyze_page(
                    page_no=page_no,
                    image_path=image_path,
                    related_diff_ids=diffs_by_page.get(page_no, []),
                    docx_tables=docx_table_stats.get(page_no, []),
                )
                review.cell_evidence = self._cell_evidence_for_page(
                    page_no=page_no,
                    image_path=image_path,
                    related_diff_ids=diffs_by_page.get(page_no, []),
                    diffs_by_id=diffs_by_id,
                    pdf_by_id=pdf_by_id,
                    focused_by_id=focused_by_id,
                )
            except Exception:
                logger.debug("Failed to analyze table page %s.", page_no, exc_info=True)
                review = TableParseReview(
                    table_id=f"table_p{page_no:04d}",
                    page_no=page_no,
                    status="table_parse_failed",
                    confidence=0.0,
                    related_diff_ids=diffs_by_page.get(page_no, []),
                    docx_table_candidates=docx_table_stats.get(page_no, []),
                    flags=["table_parse_failed"],
                )
            reviews.append(review)
        self._apply_focused_gate(preflight_result=preflight_result, table_reviews=reviews)
        return reviews

    def _target_pages(self, preflight_result: ConversionPreflightResult) -> set[int]:
        pages: set[int] = set()
        for page_key, profile in preflight_result.page_profiles.items():
            try:
                page_no = int(page_key)
            except Exception:
                continue
            primary_route = str(profile.get("primary_route") or "")
            if (
                profile.get("needs_table_parser")
                or "table_heavy" in set(profile.get("labels") or [])
                or primary_route in {"image_table_cell_compare", "native_table_compare"}
            ):
                pages.add(page_no)
        for diff in preflight_result.diff_candidates:
            if diff.pdf_page_no and diff.category in {"table_cell_mismatch_suspect", "table_structure_suspect"}:
                pages.add(int(diff.pdf_page_no))
        for review in preflight_result.focused_reviews:
            if review.page_no and review.next_route == "needs_table_parser":
                pages.add(int(review.page_no))
        return pages

    def _analyze_page(
        self,
        *,
        page_no: int,
        image_path: Path,
        related_diff_ids: Sequence[str],
        docx_tables: Sequence[Dict[str, Any]],
    ) -> TableParseReview:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            arr = np.asarray(gray)
        mask = self._foreground_mask(arr)
        height, width = mask.shape
        bbox = self._content_bbox(mask)
        horizontal = self._line_positions(mask.mean(axis=1), threshold=max(0.18, float(mask.mean()) * 4.0), min_gap=max(3, height // 360))
        vertical = self._line_positions(mask.mean(axis=0), threshold=max(0.12, float(mask.mean()) * 3.5), min_gap=max(3, width // 360))
        row_count = max(0, len(horizontal) - 1)
        col_count = max(0, len(vertical) - 1)
        flags: List[str] = []
        if docx_tables:
            flags.append("docx_table_cells_present")
        if row_count >= 1 and col_count >= 1:
            status = "parseable_grid"
            confidence = min(0.9, 0.55 + min(row_count, 10) * 0.025 + min(col_count, 8) * 0.025)
        elif len(horizontal) >= 3 or len(vertical) >= 3:
            status = "partial_grid"
            confidence = 0.5
            flags.append("partial_table_lines")
        else:
            status = "no_grid_detected"
            confidence = 0.28
            flags.append("no_grid_lines")
        return TableParseReview(
            table_id=f"table_p{page_no:04d}",
            page_no=page_no,
            status=status,
            confidence=confidence,
            bbox=bbox,
            row_count=row_count,
            col_count=col_count,
            horizontal_line_count=len(horizontal),
            vertical_line_count=len(vertical),
            related_diff_ids=list(related_diff_ids),
            docx_table_candidates=[dict(item) for item in docx_tables],
            flags=flags,
        )

    def _apply_focused_gate(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        table_reviews: Sequence[TableParseReview],
    ) -> None:
        table_by_page = {item.page_no: item for item in table_reviews}
        related_diff_ids = {diff_id for item in table_reviews for diff_id in item.related_diff_ids}
        for review in preflight_result.focused_reviews:
            if review.diff_id not in related_diff_ids or review.category != "table_cell_mismatch_suspect":
                continue
            flags = set(review.flags)
            if "crop_ocr_supports_docx" in flags:
                continue
            if "crop_ocr_supports_pdf" not in flags:
                continue
            table = table_by_page.get(int(review.page_no or 0))
            if table is None:
                continue
            cell_evidence = self._table_cell_evidence(table, review.diff_id)
            if cell_evidence:
                review.table_cell = dict(cell_evidence)
                for flag in self._cell_evidence_flags(cell_evidence):
                    if flag not in review.flags:
                        review.flags.append(flag)
            table_ready = table.status == "parseable_grid"
            table_partial_with_docx = table.status == "partial_grid" and bool(table.docx_table_candidates)
            table_cell_stable = bool(cell_evidence.get("stable_text_match")) if cell_evidence else False
            if not table_ready and not table_partial_with_docx and not table_cell_stable:
                continue
            review.status = "ready_for_table_gate"
            review.decision = "possible_conversion_error"
            review.next_route = "qwen_text_gate"
            review.confidence = max(float(review.confidence or 0.0), 0.82 if table_cell_stable else 0.72)
            structural_flag = ""
            if table_ready:
                structural_flag = "table_structure_parseable"
            elif table_partial_with_docx:
                structural_flag = "table_structure_partial_with_docx"
            if table_cell_stable:
                review.reason = "PDF 表格单元格局部证据与 PDF 候选值一致，且与 DOCX 单元直接冲突，可进入 Qwen 表格最终门槛。"
                flag = "table_cell_evidence_stable"
            elif table_ready:
                review.reason = "表格页已有可解析网格，局部 crop OCR 支持 PDF 值，可进入 Qwen 表格门槛。"
                flag = "table_structure_parseable"
            else:
                review.reason = "表格页有部分网格且 DOCX 存在表格单元，局部 crop OCR 支持 PDF 值，可进入 Qwen 表格门槛。"
                flag = "table_structure_partial_with_docx"
            if structural_flag and structural_flag not in review.flags:
                review.flags.append(structural_flag)
            if flag not in review.flags:
                review.flags.append(flag)

    def _cell_evidence_for_page(
        self,
        *,
        page_no: int,
        image_path: Path,
        related_diff_ids: Sequence[str],
        diffs_by_id: Dict[str, ConversionDiffCandidate],
        pdf_by_id: Dict[str, PdfEvidenceUnit],
        focused_by_id: Dict[str, FocusedCandidateReview],
    ) -> List[Dict[str, Any]]:
        if not related_diff_ids:
            return []
        try:
            with Image.open(image_path) as image:
                gray = image.convert("L")
                arr = np.asarray(gray)
                rgb_image = image.convert("RGB")
        except Exception:
            return []
        mask = self._foreground_mask(arr)
        rows: List[Dict[str, Any]] = []
        for diff_id in related_diff_ids:
            diff = diffs_by_id.get(diff_id)
            if diff is None or diff.category != "table_cell_mismatch_suspect":
                continue
            pdf_unit = pdf_by_id.get(diff.pdf_unit_id)
            focused = focused_by_id.get(diff_id)
            if pdf_unit is None or not pdf_unit.bbox:
                continue
            cell_bbox = self._local_cell_bbox(mask=mask, unit_bbox=pdf_unit.bbox)
            crop_path = self._save_cell_crop(image=rgb_image, bbox=cell_bbox, diff_id=diff_id, page_no=page_no)
            cell_ocr_text = self._ocr_cell_crop(crop_path)
            fallback_ocr = str(getattr(focused, "crop_ocr_text", "") or "")
            effective_ocr = " ".join(item for item in (cell_ocr_text, fallback_ocr) if item.strip())
            pdf_norm = self._semantic_compact(diff.pdf_text)
            docx_norm = self._semantic_compact(diff.docx_text)
            cell_norm = self._semantic_compact(effective_ocr)
            stable = bool(pdf_norm and cell_norm and pdf_norm in cell_norm and docx_norm and docx_norm != pdf_norm)
            confusable = self._confusable_substitution(pdf_norm, docx_norm)
            flags: List[str] = []
            if crop_path:
                flags.append("has_table_cell_crop")
            if cell_ocr_text:
                flags.append("has_table_cell_ocr")
            elif fallback_ocr:
                flags.append("uses_focused_crop_ocr")
            if stable:
                flags.append("table_cell_ocr_supports_pdf")
            if confusable:
                flags.append("confusable_digit_substitution")
            rows.append(
                {
                    "diff_id": diff_id,
                    "page_no": int(page_no),
                    "bbox": [round(float(v), 2) for v in cell_bbox],
                    "crop_path": crop_path,
                    "cell_ocr_text": cell_ocr_text,
                    "fallback_ocr_text": fallback_ocr,
                    "pdf_text": diff.pdf_text,
                    "docx_text": diff.docx_text,
                    "pdf_normalized": pdf_norm,
                    "docx_normalized": docx_norm,
                    "cell_ocr_normalized": cell_norm,
                    "stable_text_match": stable,
                    "confusable_substitution": confusable,
                    "confidence": round(0.84 if stable and confusable else 0.78 if stable else 0.52, 4),
                    "flags": flags,
                }
            )
        return rows

    def _table_cell_evidence(self, table: TableParseReview, diff_id: str) -> Dict[str, Any]:
        for item in table.cell_evidence:
            if item.get("diff_id") == diff_id:
                return dict(item)
        return {}

    def _cell_evidence_flags(self, cell_evidence: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        if cell_evidence.get("crop_path"):
            flags.append("has_table_cell_crop")
        if cell_evidence.get("stable_text_match"):
            flags.append("table_cell_evidence_stable")
        if cell_evidence.get("confusable_substitution"):
            flags.append("table_cell_confusable_substitution")
        return flags

    def _local_cell_bbox(self, *, mask: np.ndarray, unit_bbox: Sequence[float]) -> List[float]:
        height, width = mask.shape
        left, top, right, bottom = self._clip_bbox(unit_bbox, width=width, height=height)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        margin_x = max(90, int((right - left) * 3.0))
        margin_y = max(90, int((bottom - top) * 4.0))
        roi_left = max(0, int(left - margin_x))
        roi_right = min(width, int(right + margin_x))
        roi_top = max(0, int(top - margin_y))
        roi_bottom = min(height, int(bottom + margin_y))
        roi = mask[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            return [left, top, right, bottom]
        row_positions = [roi_top + item for item in self._local_line_positions(roi.mean(axis=1), axis_length=roi.shape[1])]
        col_positions = [roi_left + item for item in self._local_line_positions(roi.mean(axis=0), axis_length=roi.shape[0])]
        cell_left, cell_right = self._nearest_bounds(col_positions, center=cx, fallback_min=left, fallback_max=right, page_min=0, page_max=width, min_size=28)
        cell_top, cell_bottom = self._nearest_bounds(row_positions, center=cy, fallback_min=top, fallback_max=bottom, page_min=0, page_max=height, min_size=20)
        pad_x = 3
        pad_y = 3
        return [
            float(max(0, cell_left + pad_x)),
            float(max(0, cell_top + pad_y)),
            float(min(width, cell_right - pad_x)),
            float(min(height, cell_bottom - pad_y)),
        ]

    def _local_line_positions(self, density: np.ndarray, *, axis_length: int) -> List[int]:
        if density.size == 0:
            return []
        q90 = float(np.quantile(density, 0.90))
        q97 = float(np.quantile(density, 0.97))
        threshold = max(0.08, min(0.26, max(q90 * 0.85, q97 * 0.55)))
        min_gap = max(2, axis_length // 260)
        return self._line_positions(density, threshold=threshold, min_gap=min_gap)

    def _nearest_bounds(
        self,
        positions: Sequence[int],
        *,
        center: float,
        fallback_min: float,
        fallback_max: float,
        page_min: int,
        page_max: int,
        min_size: int,
    ) -> tuple[float, float]:
        before = [float(item) for item in positions if float(item) < float(fallback_min) - 3]
        after = [float(item) for item in positions if float(item) > float(fallback_max) + 3]
        left = max(before) if before else max(float(page_min), float(fallback_min) - min_size)
        right = min(after) if after else min(float(page_max), float(fallback_max) + min_size)
        if right - left < min_size:
            half = max(min_size / 2.0, (float(fallback_max) - float(fallback_min)) * 1.2)
            left = max(float(page_min), center - half)
            right = min(float(page_max), center + half)
        return left, right

    def _clip_bbox(self, bbox: Sequence[float], *, width: int, height: int) -> tuple[float, float, float, float]:
        values = list(bbox)[:4]
        if len(values) < 4:
            return 0.0, 0.0, float(width), float(height)
        left = max(0.0, min(float(width), float(values[0])))
        top = max(0.0, min(float(height), float(values[1])))
        right = max(0.0, min(float(width), float(values[2])))
        bottom = max(0.0, min(float(height), float(values[3])))
        if right <= left:
            right = min(float(width), left + 1)
        if bottom <= top:
            bottom = min(float(height), top + 1)
        return left, top, right, bottom

    def _save_cell_crop(self, *, image: Image.Image, bbox: Sequence[float], diff_id: str, page_no: int) -> str:
        if self.work_dir is None or self.cell_crop_dir is None:
            return ""
        try:
            self.cell_crop_dir.mkdir(parents=True, exist_ok=True)
            left, top, right, bottom = [int(round(float(value))) for value in bbox[:4]]
            if right <= left or bottom <= top:
                return ""
            target = self.cell_crop_dir / f"{diff_id}_cell_p{page_no:04d}.jpg"
            image.crop((left, top, right, bottom)).save(target, format="JPEG", quality=94)
            try:
                target.chmod(0o600)
            except Exception:
                pass
            return target.relative_to(self.work_dir).as_posix()
        except Exception:
            logger.debug("Failed to save table cell crop for %s.", diff_id, exc_info=True)
            return ""

    def _ocr_cell_crop(self, crop_path: str) -> str:
        if not crop_path or self.work_dir is None:
            return ""
        path = self.work_dir / crop_path
        if not path.exists() or not MacOSVisionOCRService.is_supported():
            return ""
        try:
            result = asyncio.run(MacOSVisionOCRService(timeout=30).extract_document_text_from_image_async(path.read_bytes()))
            return str(result.get("text") or "").strip()
        except Exception:
            logger.debug("Failed to OCR table cell crop %s.", crop_path, exc_info=True)
            return ""

    def _semantic_compact(self, text: str) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _confusable_substitution(self, pdf_value: str, docx_value: str) -> bool:
        if not pdf_value or not docx_value or pdf_value == docx_value:
            return False
        if len(pdf_value) != len(docx_value):
            return False
        pairs = [(left, right) for left, right in zip(pdf_value, docx_value) if left != right]
        if len(pairs) != 1:
            return False
        left, right = pairs[0]
        if left.isdigit() and not right.isdigit():
            return True
        confusable = {("0", "o"), ("0", "g"), ("0", "q"), ("1", "l"), ("1", "i"), ("5", "s"), ("8", "b"), ("6", "g")}
        return (left, right) in confusable or (right, left) in confusable

    def _docx_table_stats(self, docx_units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[tuple[int, int], Dict[str, Any]] = {}
        for unit in docx_units:
            if unit.container_type != "table_cell" or unit.estimated_page_no is None or unit.table_index is None:
                continue
            key = (int(unit.estimated_page_no), int(unit.table_index))
            row = grouped.setdefault(
                key,
                {
                    "estimated_page_no": int(unit.estimated_page_no),
                    "table_index": int(unit.table_index),
                    "row_count": 0,
                    "col_count": 0,
                    "cell_count": 0,
                    "text_chars": 0,
                },
            )
            row["row_count"] = max(int(row["row_count"]), int(unit.row_index or 0))
            row["col_count"] = max(int(row["col_count"]), int(unit.col_index or 0))
            row["cell_count"] = int(row["cell_count"]) + 1
            row["text_chars"] = int(row["text_chars"]) + len(unit.normalized_text)
        by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for (page_no, _table_index), item in grouped.items():
            by_page[page_no].append(dict(item))
        return dict(by_page)

    def _foreground_mask(self, arr: np.ndarray) -> np.ndarray:
        q01, q05, q50, q99 = np.quantile(arr, [0.01, 0.05, 0.50, 0.99])
        threshold = float(q05 + 0.35 * max(0.0, q50 - q05))
        if q99 - q01 < 12:
            threshold = float(q01 - 1)
        threshold = max(0.0, min(245.0, threshold))
        return arr < threshold

    def _line_positions(self, density: np.ndarray, *, threshold: float, min_gap: int) -> List[int]:
        if density.size == 0:
            return []
        indices = np.where(density >= threshold)[0]
        if indices.size == 0:
            return []
        positions: List[int] = []
        start = previous = int(indices[0])
        for raw in indices[1:]:
            value = int(raw)
            if value - previous <= min_gap:
                previous = value
                continue
            positions.append((start + previous) // 2)
            start = previous = value
        positions.append((start + previous) // 2)
        return positions

    def _content_bbox(self, mask: np.ndarray) -> List[float]:
        if mask.size == 0 or not bool(mask.any()):
            return []
        ys, xs = np.where(mask)
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
