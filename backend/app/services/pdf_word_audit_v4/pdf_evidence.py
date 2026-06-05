from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .common import normalize_text
from .models import PdfEvidenceUnit

logger = logging.getLogger(__name__)

try:  # PyMuPDF is optional at import time; v4 falls back to visual regions.
    import fitz  # type: ignore
except Exception:  # pragma: no cover - exercised only on minimal deployments.
    fitz = None  # type: ignore


NOISE_RE = re.compile(r"[�□■◆◇]+")


class PdfEvidenceExtractor:
    """Extract lightweight PDF ground-truth evidence for conversion preflight."""

    MIN_NATIVE_CHARS = 24
    MAX_NOISE_RATIO = 0.08
    MAX_VISUAL_REGIONS_PER_PAGE = 30

    def extract(self, *, pdf_path: Path, rendered_pages: Dict[int, Path], page_count: int) -> tuple[List[PdfEvidenceUnit], Dict[str, Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        native_units: List[PdfEvidenceUnit] = []
        native_metrics: Dict[int, Dict[str, Any]] = {}
        if fitz is None:
            warnings.append("pymupdf_unavailable_native_text_skipped")
        else:
            try:
                native_units, native_metrics = self._extract_native_units(pdf_path=pdf_path, page_count=page_count)
            except Exception:
                logger.debug("Failed to extract PDF native text.", exc_info=True)
                warnings.append("pdf_native_text_extract_failed")

        visual_units, visual_profiles = self._extract_visual_units(rendered_pages=rendered_pages, native_metrics=native_metrics)
        units = [*native_units, *visual_units]
        page_profiles: Dict[str, Dict[str, Any]] = {}
        for page_no in range(1, page_count + 1):
            native = native_metrics.get(page_no) or {}
            visual = visual_profiles.get(page_no) or {}
            page_profiles[str(page_no)] = self._page_profile(page_no=page_no, native=native, visual=visual)
        return units, page_profiles, warnings

    def _extract_native_units(self, *, pdf_path: Path, page_count: int) -> tuple[List[PdfEvidenceUnit], Dict[int, Dict[str, Any]]]:
        assert fitz is not None
        document = fitz.open(str(pdf_path))
        units: List[PdfEvidenceUnit] = []
        metrics: Dict[int, Dict[str, Any]] = {}
        try:
            for page_index in range(min(page_count, len(document))):
                page_no = page_index + 1
                page = document[page_index]
                page_rect = page.rect
                page_area = max(1.0, float(page_rect.width * page_rect.height))
                payload = page.get_text("dict") or {}
                blocks = [block for block in payload.get("blocks") or [] if isinstance(block, dict)]
                text_chars = 0
                bad_chars = 0
                text_area = 0.0
                image_area = 0.0
                order_index = 0
                for block_index, block in enumerate(blocks):
                    bbox = self._bbox(block.get("bbox"))
                    block_type = int(block.get("type") or 0)
                    if block_type == 1:
                        image_area += self._area(bbox)
                        continue
                    text = self._block_text(block)
                    normalized = normalize_text(text)
                    if not normalized:
                        continue
                    text_chars += len(normalized)
                    bad_chars += len(NOISE_RE.findall(text))
                    text_area += self._area(bbox)
                    order_index += 1
                    confidence = 0.96 if len(normalized) >= 8 else 0.82
                    units.append(
                        PdfEvidenceUnit(
                            unit_id=f"pdf_p{page_no}_native_{block_index}",
                            page_no=page_no,
                            unit_type="native_text_block",
                            text=text,
                            normalized_text=normalized,
                            bbox=bbox,
                            source="pdf_native_text",
                            confidence=confidence,
                            order_index=order_index,
                            region_type="text",
                        )
                    )
                noise_ratio = bad_chars / max(text_chars, 1)
                metrics[page_no] = {
                    "native_text_chars": text_chars,
                    "native_noise_ratio": round(noise_ratio, 4),
                    "native_text_area_ratio": round(min(1.0, text_area / page_area), 4),
                    "image_area_ratio": round(min(1.0, image_area / page_area), 4),
                    "native_text_reliable": bool(text_chars >= self.MIN_NATIVE_CHARS and noise_ratio <= self.MAX_NOISE_RATIO),
                }
        finally:
            document.close()
        return units, metrics

    def _extract_visual_units(
        self,
        *,
        rendered_pages: Dict[int, Path],
        native_metrics: Dict[int, Dict[str, Any]],
    ) -> tuple[List[PdfEvidenceUnit], Dict[int, Dict[str, Any]]]:
        units: List[PdfEvidenceUnit] = []
        profiles: Dict[int, Dict[str, Any]] = {}
        for page_no, image_path in sorted(rendered_pages.items()):
            try:
                profile = self._visual_profile(image_path)
            except Exception:
                logger.debug("Failed to profile rendered PDF page %s.", page_no, exc_info=True)
                profile = {
                    "dark_pixel_ratio": 0.0,
                    "visual_content_bbox": [],
                    "horizontal_line_count": 0,
                    "vertical_line_count": 0,
                    "table_like": False,
                    "visual_region_count": 0,
                    "regions": [],
                }
            native = native_metrics.get(page_no) or {}
            flags: List[str] = []
            if not native.get("native_text_reliable"):
                flags.append("needs_ocr")
            if profile.get("table_like"):
                flags.append("table_like")
            bbox = list(profile.get("visual_content_bbox") or [])
            if bbox:
                units.append(
                    PdfEvidenceUnit(
                        unit_id=f"pdf_p{page_no}_visual_content",
                        page_no=page_no,
                        unit_type="visual_region",
                        bbox=bbox,
                        source="pdf_render_visual",
                        confidence=0.72,
                        order_index=10_000,
                        region_type="visual_content",
                        flags=flags,
                        metrics=profile,
                    )
                )
            for index, region in enumerate(profile.get("regions") or [], start=1):
                region_flags = list(flags)
                if region.get("region_type") == "text_band":
                    region_flags.append("needs_anchor_ocr")
                units.append(
                    PdfEvidenceUnit(
                        unit_id=f"pdf_p{page_no}_region_{index:03d}",
                        page_no=page_no,
                        unit_type="visual_region",
                        bbox=list(region.get("bbox") or []),
                        source="pdf_render_visual",
                        confidence=float(region.get("confidence") or 0.62),
                        order_index=10_000 + index,
                        region_type=str(region.get("region_type") or "visual_region"),
                        flags=region_flags,
                        metrics={key: value for key, value in region.items() if key != "bbox"},
                    )
                )
            if profile.get("table_like") and bbox:
                units.append(
                    PdfEvidenceUnit(
                        unit_id=f"pdf_p{page_no}_table_suspect",
                        page_no=page_no,
                        unit_type="table_region",
                        bbox=bbox,
                        source="pdf_render_visual",
                        confidence=0.68,
                        order_index=10_001,
                        region_type="table",
                        flags=["needs_table_parser"],
                        metrics=profile,
                    )
                )
            profiles[page_no] = profile
        return units, profiles

    def _visual_profile(self, image_path: Path) -> Dict[str, Any]:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            arr = np.asarray(gray)
        foreground, threshold, background = self._foreground_mask(arr)
        height, width = foreground.shape
        foreground_count = int(foreground.sum())
        if foreground_count:
            ys, xs = np.where(foreground)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        else:
            bbox = []
        foreground_ratio = foreground_count / max(1, width * height)
        row_density = foreground.mean(axis=1) if height else np.array([])
        col_density = foreground.mean(axis=0) if width else np.array([])
        horizontal_lines = self._line_count(row_density, threshold=0.32)
        vertical_lines = self._line_count(col_density, threshold=0.28)
        table_like = bool(
            (horizontal_lines >= 4 and vertical_lines >= 4)
            or (vertical_lines >= 8 and foreground_ratio >= 0.08)
            or (horizontal_lines >= 8 and vertical_lines >= 2 and foreground_ratio >= 0.08)
        )
        regions = self._region_bboxes(foreground, max_regions=self.MAX_VISUAL_REGIONS_PER_PAGE)
        return {
            "image_width": int(width),
            "image_height": int(height),
            "dark_pixel_ratio": round(float(foreground_ratio), 6),
            "foreground_pixel_ratio": round(float(foreground_ratio), 6),
            "foreground_threshold": round(float(threshold), 2),
            "background_level": round(float(background), 2),
            "visual_content_bbox": bbox,
            "horizontal_line_count": int(horizontal_lines),
            "vertical_line_count": int(vertical_lines),
            "table_like": table_like,
            "visual_region_count": len(regions),
            "regions": regions,
        }

    def _page_profile(self, *, page_no: int, native: Dict[str, Any], visual: Dict[str, Any]) -> Dict[str, Any]:
        native_chars = int(native.get("native_text_chars") or 0)
        native_reliable = bool(native.get("native_text_reliable"))
        image_area_ratio = float(native.get("image_area_ratio") or 0.0)
        table_like = bool(visual.get("table_like"))
        dark_ratio = float(visual.get("dark_pixel_ratio") or 0.0)
        labels: List[str] = []
        if native_reliable and not table_like and image_area_ratio < 0.2:
            labels.append("simple_native_text")
        if table_like:
            labels.append("table_heavy")
        if not native_reliable and dark_ratio > 0.002:
            labels.append("scan_like")
        if native_reliable and (table_like or image_area_ratio >= 0.2):
            labels.append("mixed_layout")
        if image_area_ratio >= 0.35:
            labels.append("image_text_heavy")
        if not labels:
            labels.append("mapping_risk")
        return {
            "page_no": int(page_no),
            "labels": sorted(set(labels)),
            "native_text_reliable": native_reliable,
            "native_text_chars": native_chars,
            "native_noise_ratio": native.get("native_noise_ratio", 0.0),
            "native_text_area_ratio": native.get("native_text_area_ratio", 0.0),
            "image_area_ratio": round(image_area_ratio, 4),
            "dark_pixel_ratio": dark_ratio,
            "foreground_pixel_ratio": visual.get("foreground_pixel_ratio", dark_ratio),
            "foreground_threshold": visual.get("foreground_threshold"),
            "background_level": visual.get("background_level"),
            "visual_region_count": int(visual.get("visual_region_count") or 0),
            "horizontal_line_count": int(visual.get("horizontal_line_count") or 0),
            "vertical_line_count": int(visual.get("vertical_line_count") or 0),
            "table_like": table_like,
            "needs_ocr": bool(not native_reliable and dark_ratio > 0.002),
            "needs_table_parser": table_like,
        }

    def _foreground_mask(self, arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        q01, q05, q50, q99 = np.quantile(arr, [0.01, 0.05, 0.50, 0.99])
        # Scanned legal PDFs often have gray paper backgrounds. A fixed
        # threshold like 220 treats the whole page as foreground, so use the
        # page's own lower/mid histogram band to isolate ink-like pixels.
        threshold = float(q05 + 0.35 * max(0.0, q50 - q05))
        if q99 - q01 < 12:
            threshold = float(q01 - 1)
        threshold = max(0.0, min(245.0, threshold))
        foreground = arr < threshold
        return foreground, threshold, float(q99)

    def _region_bboxes(self, foreground: np.ndarray, *, max_regions: int) -> List[Dict[str, Any]]:
        if foreground.size == 0 or not bool(foreground.any()):
            return []
        height, width = foreground.shape
        row_density = foreground.mean(axis=1)
        row_threshold = max(0.006, min(0.04, float(foreground.mean()) * 0.30))
        row_runs = self._runs(row_density > row_threshold, min_len=max(8, height // 220), gap=max(6, height // 180))
        regions: List[Dict[str, Any]] = []
        min_area = max(80.0, width * height * 0.00025)
        for start_y, end_y in row_runs:
            slice_mask = foreground[start_y:end_y, :]
            if not bool(slice_mask.any()):
                continue
            ys, xs = np.where(slice_mask)
            start_x, end_x = int(xs.min()), int(xs.max()) + 1
            bbox = [float(start_x), float(start_y), float(end_x), float(end_y)]
            area = max(0.0, (end_x - start_x) * (end_y - start_y))
            if area < min_area:
                continue
            region_type = "text_band"
            if (end_x - start_x) > width * 0.75 and (end_y - start_y) < height * 0.08:
                region_type = "line_band"
            elif (end_y - start_y) > height * 0.22:
                region_type = "large_visual_region"
            regions.append(
                {
                    "bbox": bbox,
                    "region_type": region_type,
                    "confidence": 0.64,
                    "foreground_ratio": round(float(slice_mask[:, start_x:end_x].mean()), 6),
                }
            )
            if len(regions) >= max_regions:
                return regions
        return regions

    def _runs(self, values: Sequence[bool] | np.ndarray, *, min_len: int, gap: int) -> List[Tuple[int, int]]:
        indices = np.where(np.asarray(values))[0]
        if indices.size == 0:
            return []
        result: List[Tuple[int, int]] = []
        start = previous = int(indices[0])
        for raw in indices[1:]:
            value = int(raw)
            if value - previous <= gap:
                previous = value
                continue
            if previous - start + 1 >= min_len:
                result.append((start, previous + 1))
            start = previous = value
        if previous - start + 1 >= min_len:
            result.append((start, previous + 1))
        return result

    def _block_text(self, block: Dict[str, Any]) -> str:
        lines: List[str] = []
        for line in block.get("lines") or []:
            spans = line.get("spans") or [] if isinstance(line, dict) else []
            text = "".join(str(span.get("text") or "") for span in spans if isinstance(span, dict))
            if text.strip():
                lines.append(text.strip())
        return "\n".join(lines).strip()

    def _bbox(self, value: Any) -> List[float]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return []
        try:
            return [round(float(value[0]), 2), round(float(value[1]), 2), round(float(value[2]), 2), round(float(value[3]), 2)]
        except Exception:
            return []

    def _area(self, bbox: List[float]) -> float:
        if len(bbox) < 4:
            return 0.0
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))

    def _line_count(self, density: np.ndarray, *, threshold: float) -> int:
        if density.size == 0:
            return 0
        mask = density >= threshold
        count = 0
        in_run = False
        for value in mask:
            if bool(value) and not in_run:
                count += 1
                in_run = True
            elif not bool(value):
                in_run = False
        return count
