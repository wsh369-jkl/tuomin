from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from app.core.config import settings
from app.services.macos_vision_ocr import MacOSVisionOCRService

from .common import normalize_text
from .models import PdfEvidenceUnit

logger = logging.getLogger(__name__)


ANCHOR_RE = re.compile(
    r"(?P<id_number>(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d))"
    r"|(?P<mobile>(?<!\d)1[3-9]\d{9}(?!\d))"
    r"|(?P<date>\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
    r"|(?P<case_no>[（(]?\d{4}[）)]?[\u4e00-\u9fff]{1,6}\d{2,8}[\u4e00-\u9fff]{1,8}\d+号)"
    r"|(?P<long_number>(?<!\d)\d{12,25}(?!\d))"
)


class AnchorOcrExtractor:
    """Run a serial, lightweight OCR pass only to create alignment anchors.

    This is deliberately narrow: it produces page/line
    text evidence and anchor counts for PDF-DOCX mapping, but it does not decide
    conversion errors and does not call Qwen or vision LLMs.
    """

    MAX_LINES_PER_PAGE = 90

    def __init__(self, *, work_dir: Path) -> None:
        self.work_dir = work_dir

    def extract(
        self,
        *,
        rendered_pages: Dict[int, Path],
        page_profiles: Dict[str, Dict[str, Any]],
        page_count: int,
    ) -> tuple[List[PdfEvidenceUnit], Dict[str, Any], List[str]]:
        target_pages = [
            page_no
            for page_no in range(1, page_count + 1)
            if page_profiles.get(str(page_no), {}).get("needs_ocr") and page_no in rendered_pages
        ]
        if not target_pages:
            summary = {
                "enabled": True,
                "version": "anchor_ocr_v1",
                "engine": "not_needed",
                "attempted_page_count": 0,
                "succeeded_page_count": 0,
                "line_count": 0,
                "anchor_count": 0,
                "text_chars": 0,
            }
            return [], {"enabled": True, "version": "anchor_ocr_v1", "summary": summary, "pages": []}, []

        units, payload, warnings = self._extract_with_macos_vision(
            rendered_pages={page: rendered_pages[page] for page in target_pages},
            page_profiles=page_profiles,
            page_count=page_count,
        )
        succeeded_pages = {int(page.get("page") or 0) for page in payload.get("pages") or [] if isinstance(page, dict) and page.get("ok")}
        missing_pages = [page for page in target_pages if page not in succeeded_pages]
        if units and not missing_pages:
            return units, payload, warnings
        if units:
            if missing_pages:
                warnings.append("macos_vision_anchor_ocr_partial")
            return units, payload, warnings

        summary = {
            "enabled": False,
            "version": "anchor_ocr_v1",
            "engine": "unavailable",
            "attempted_page_count": 0,
            "succeeded_page_count": 0,
            "line_count": 0,
            "anchor_count": 0,
            "text_chars": 0,
            "warnings": ["anchor_ocr_unavailable"],
        }
        return [], {"enabled": False, "version": "anchor_ocr_v1", "summary": summary, "pages": []}, ["anchor_ocr_unavailable"]

    def _extract_with_macos_vision(
        self,
        *,
        rendered_pages: Dict[int, Path],
        page_profiles: Dict[str, Dict[str, Any]],
        page_count: int,
    ) -> tuple[List[PdfEvidenceUnit], Dict[str, Any], List[str]]:
        if not MacOSVisionOCRService.is_supported():
            return [], {"enabled": False, "summary": {}}, ["macos_vision_anchor_ocr_unavailable"]

        service = MacOSVisionOCRService(timeout=max(20, int(getattr(settings, "PDF_OCR_TIMEOUT", 90) or 90)))
        units: List[PdfEvidenceUnit] = []
        pages: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for page_no, image_path in sorted(rendered_pages.items()):
            try:
                image_bytes = image_path.read_bytes()
                result = asyncio.run(
                    service.extract_document_text_from_image_async(
                        image_bytes,
                        page_number=page_no,
                        total_pages=page_count,
                    )
                )
                page_units, page_payload = self._units_from_page_result(
                    page_no=page_no,
                    image_path=image_path,
                    result=result,
                    source="macos_vision_anchor_ocr",
                )
                units.extend(page_units)
                pages.append(page_payload)
                page_profiles.setdefault(str(page_no), {})["anchor_ocr_text_chars"] = page_payload.get("text_chars", 0)
                page_profiles.setdefault(str(page_no), {})["anchor_ocr_line_count"] = page_payload.get("line_count", 0)
                page_profiles.setdefault(str(page_no), {})["anchor_ocr_anchor_count"] = page_payload.get("anchor_count", 0)
                page_profiles.setdefault(str(page_no), {})["anchor_ocr_quality"] = page_payload.get("quality", "low")
            except Exception as exc:
                logger.debug("macOS Vision anchor OCR failed for page %s.", page_no, exc_info=True)
                warning = f"macos_vision_anchor_ocr_failed_page_{page_no}"
                warnings.append(warning)
                pages.append(
                    {
                        "page": page_no,
                        "ok": False,
                        "engine": "macos_vision_anchor_ocr",
                        "text": "",
                        "lines": [],
                        "quality": "failed",
                        "text_chars": 0,
                        "line_count": 0,
                        "anchor_count": 0,
                        "warnings": [warning, f"{type(exc).__name__}: {exc}"],
                    }
                )
        return units, self._payload(engine="macos_vision_anchor_ocr", pages=pages, warnings=warnings), warnings

    def _units_from_page_result(
        self,
        *,
        page_no: int,
        image_path: Path | None,
        result: Dict[str, Any],
        source: str,
    ) -> tuple[List[PdfEvidenceUnit], Dict[str, Any]]:
        text = str(result.get("text") or "").strip()
        normalized = normalize_text(text)
        quality = str(result.get("quality") or "medium").lower()
        confidence = self._quality_confidence(quality)
        raw_lines = [line for line in result.get("lines") or [] if isinstance(line, dict)]
        image_size = self._image_size(image_path)
        anchors = self._anchors(text)
        flags = ["anchor_ocr"]
        if confidence < 0.6:
            flags.append("low_confidence_ocr")
        if not anchors:
            flags.append("no_anchor_tokens")

        units: List[PdfEvidenceUnit] = []
        if normalized:
            units.append(
                PdfEvidenceUnit(
                    unit_id=f"pdf_p{page_no}_anchor_ocr_page",
                    page_no=page_no,
                    unit_type="anchor_ocr_page",
                    text=text,
                    normalized_text=normalized,
                    source=source,
                    confidence=confidence,
                    order_index=20_000,
                    region_type="ocr_page_text",
                    flags=flags,
                    metrics={
                        "engine": source,
                        "quality": quality,
                        "line_count": len(raw_lines),
                        "anchor_count": len(anchors),
                        "anchors": anchors[:30],
                    },
                )
            )

        selected_lines = self._selected_lines(raw_lines)
        for index, line in enumerate(selected_lines, start=1):
            line_text = str(line.get("text") or "").strip()
            line_normalized = normalize_text(line_text)
            if not line_normalized:
                continue
            line_anchors = self._anchors(line_text)
            line_flags = ["anchor_ocr_line"]
            if line_anchors:
                line_flags.append("has_anchor_tokens")
            if confidence < 0.6:
                line_flags.append("low_confidence_ocr")
            units.append(
                PdfEvidenceUnit(
                    unit_id=f"pdf_p{page_no}_anchor_line_{index:03d}",
                    page_no=page_no,
                    unit_type="anchor_ocr_line",
                    text=line_text,
                    normalized_text=line_normalized,
                    bbox=self._line_bbox(line.get("bbox"), image_size=image_size),
                    source=source,
                    confidence=confidence,
                    order_index=20_000 + index,
                    region_type="ocr_line_text",
                    flags=line_flags,
                    metrics={"engine": source, "quality": quality, "anchors": line_anchors[:12]},
                )
            )

        page_payload = {
            "page": page_no,
            "ok": bool(normalized),
            "engine": source,
            "text": text,
            "quality": quality,
            "confidence": confidence,
            "text_chars": len(normalized),
            "line_count": len(raw_lines),
            "selected_line_count": len(selected_lines),
            "anchor_count": len(anchors),
            "anchors": anchors[:40],
            "lines": [
                {
                    "text": str(line.get("text") or ""),
                    "bbox": self._line_bbox(line.get("bbox"), image_size=image_size),
                    "anchors": self._anchors(str(line.get("text") or ""))[:12],
                }
                for line in selected_lines
            ],
            "warnings": [str(item) for item in result.get("warnings") or [] if str(item)],
        }
        return units, page_payload

    def _payload(self, *, engine: str, pages: List[Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
        succeeded = [page for page in pages if page.get("ok")]
        summary = {
            "enabled": True,
            "version": "anchor_ocr_v1",
            "engine": engine,
            "attempted_page_count": len(pages),
            "succeeded_page_count": len(succeeded),
            "line_count": sum(int(page.get("line_count") or 0) for page in pages),
            "selected_line_count": sum(int(page.get("selected_line_count") or 0) for page in pages),
            "anchor_count": sum(int(page.get("anchor_count") or 0) for page in pages),
            "text_chars": sum(int(page.get("text_chars") or 0) for page in pages),
            "warnings": sorted({str(item) for item in warnings if str(item)}),
        }
        return {"enabled": True, "version": "anchor_ocr_v1", "summary": summary, "pages": pages}

    def _merge_payloads(self, *, primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        pages_by_no: Dict[int, Dict[str, Any]] = {}
        for payload in (primary, fallback):
            for page in payload.get("pages") or []:
                if not isinstance(page, dict):
                    continue
                page_no = int(page.get("page") or 0)
                if page_no <= 0:
                    continue
                if page_no not in pages_by_no or (page.get("ok") and not pages_by_no[page_no].get("ok")):
                    pages_by_no[page_no] = page
        warnings = [
            *list((primary.get("summary") or {}).get("warnings") or []),
            *list((fallback.get("summary") or {}).get("warnings") or []),
        ]
        merged = self._payload(
            engine=f"{(primary.get('summary') or {}).get('engine') or 'primary'}+{(fallback.get('summary') or {}).get('engine') or 'fallback'}",
            pages=[pages_by_no[key] for key in sorted(pages_by_no)],
            warnings=[str(item) for item in warnings if str(item)],
        )
        return merged

    def _selected_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for line in lines:
            text = str(line.get("text") or "").strip()
            normalized = normalize_text(text)
            if not normalized:
                continue
            if self._anchors(text) or len(normalized) >= 8:
                selected.append(line)
            if len(selected) >= self.MAX_LINES_PER_PAGE:
                break
        return selected

    def _anchors(self, text: str) -> List[Dict[str, str]]:
        anchors: List[Dict[str, str]] = []
        value = str(text or "")
        for match in ANCHOR_RE.finditer(value):
            kind = next((name for name, item in match.groupdict().items() if item), "")
            token = normalize_text(match.group(0))
            if kind and token:
                anchors.append({"kind": kind, "value": token})
        compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", value)
        if len(compact) >= 12:
            for token in (compact[:12], compact[-12:]):
                normalized = normalize_text(token)
                if len(normalized) >= 6:
                    if re.fullmatch(r"\d+", normalized):
                        continue
                    anchors.append({"kind": "phrase", "value": normalized})
        result: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in anchors:
            key = (item["kind"], item["value"])
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result[:40]

    def _line_bbox(self, value: Any, *, image_size: tuple[int, int] | None) -> List[float]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return []
        try:
            bbox = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception:
            return []
        if image_size and max(bbox) <= 1000:
            width, height = image_size
            return [
                round(bbox[0] / 1000.0 * width, 2),
                round(bbox[1] / 1000.0 * height, 2),
                round(bbox[2] / 1000.0 * width, 2),
                round(bbox[3] / 1000.0 * height, 2),
            ]
        return [round(item, 2) for item in bbox]

    def _image_size(self, image_path: Path | None) -> tuple[int, int] | None:
        if image_path is None:
            return None
        try:
            with Image.open(image_path) as image:
                return int(image.width), int(image.height)
        except Exception:
            return None

    def _quality_confidence(self, quality: str) -> float:
        if quality == "high":
            return 0.86
        if quality == "medium":
            return 0.72
        if quality == "low":
            return 0.48
        return 0.38
