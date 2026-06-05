from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from app.core.config import settings
from app.core.runtime_security import ensure_private_directory, ensure_private_file
from app.services.macos_vision_ocr import MacOSVisionOCRService

from .common import normalize_text
from .models import (
    ContentCoverageBackfillReview,
    ContentCoverageReview,
    ConversionPreflightResult,
    DocxEvidenceUnit,
    PdfEvidenceUnit,
)
from .qwen_vl_gate import OllamaQwenVlClient

logger = logging.getLogger(__name__)


def coverage_backfill_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "readability": {
                "type": "string",
                "enum": ["readable", "partially_readable", "unreadable", "not_text"],
            },
            "extracted_text": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["readability", "extracted_text", "reason", "confidence"],
    }


class ContentCoverageBackfillBuilder:
    """Read unresolved PDF visual regions before they reach human/Qwen review.

    This layer is evidence-only. It crops unresolved PDF regions, tries local
    macOS Vision OCR first, then optionally uses Qwen-VL as a small serial
    fallback. It does not confirm conversion errors or change coverage
    decisions; it only supplies readable text evidence for reports/comments.
    """

    PDF_TARGET_STATUSES = {
        "visual_pending",
        "table_pending",
        "uncovered_pdf_content",
        "diff_candidate",
        "mapping_uncertain",
        "needs_full_page_ocr",
        "needs_region_segmentation",
        "low_confidence_page_review",
        "needs_text_alignment",
    }
    DOCX_TARGET_STATUSES = {
        "needs_pdf_ocr",
        "needs_full_page_ocr",
        "needs_region_segmentation",
        "low_confidence_page_review",
    }

    def __init__(
        self,
        *,
        work_dir: Path,
        enabled: Optional[bool] = None,
        max_regions: Optional[int] = None,
        max_per_page: Optional[int] = None,
        min_text_chars: Optional[int] = None,
        qwen_vl_enabled: Optional[bool] = None,
        qwen_vl_max_regions: Optional[int] = None,
        timeout: Optional[int] = None,
        ocr_service: Any = None,
        qwen_client: Any = None,
        progress_callback: Any = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.crop_dir = self.work_dir / "evidence" / "content_coverage_crops"
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_ENABLED", True) if enabled is None else enabled)
        self.max_regions = max(0, int(max_regions if max_regions is not None else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MAX_REGIONS", 9999) or 9999))
        self.max_per_page = max(
            1,
            int(
                max_per_page
                if max_per_page is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MAX_PER_PAGE", 9999)
                or 9999
            ),
        )
        self.min_text_chars = max(1, int(min_text_chars if min_text_chars is not None else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MIN_TEXT_CHARS", 4) or 4))
        self.max_text_chars = max(
            1200,
            int(getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MAX_TEXT_CHARS", 6000) or 6000),
        )
        self.qwen_vl_enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_VL_ENABLED", True)
            if qwen_vl_enabled is None
            else qwen_vl_enabled
        )
        self.qwen_vl_max_regions = max(
            0,
            int(
                qwen_vl_max_regions
                if qwen_vl_max_regions is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_VL_MAX_REGIONS", 9999)
                or 9999
            ),
        )
        self.timeout = max(1, int(timeout or getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_TIMEOUT", 60) or 60))
        self.ocr_service = ocr_service
        self.qwen_client = qwen_client
        self.progress_callback = progress_callback
        self._qwen_preflight: Optional[tuple[bool, str]] = None
        self._qwen_attempts = 0

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        rendered_pages: Dict[int, Path],
    ) -> List[ContentCoverageBackfillReview]:
        if not self.enabled or self.max_regions <= 0:
            return []
        pdf_by_id = {item.unit_id: item for item in preflight_result.pdf_units}
        targets = self._targets(preflight_result=preflight_result, pdf_by_id=pdf_by_id)
        rows: List[ContentCoverageBackfillReview] = []
        crop_cache: Dict[tuple[int, str], str] = {}
        for target_index, review in enumerate(targets, start=1):
            self._emit_progress(
                review=review,
                index=target_index,
                total=len(targets),
                event="coverage_backfill_region_start",
                message=f"全内容 OCR 回填：第 {target_index}/{len(targets)} 个区域，PDF 第 {int(review.page_no or 0)} 页。",
            )
            row = self._review(
                index=len(rows) + 1,
                review=review,
                pdf_unit=pdf_by_id.get(review.unit_id),
                rendered_pages=rendered_pages,
                crop_cache=crop_cache,
            )
            rows.append(row)
            self._emit_progress(
                review=review,
                index=target_index,
                total=len(targets),
                event="coverage_backfill_region_done",
                message=(
                    f"全内容 OCR 回填完成：第 {target_index}/{len(targets)} 个区域，"
                    f"PDF 第 {int(review.page_no or 0)} 页，结果 {row.status}。"
                ),
            )
        self._unload_qwen()
        return rows

    def _emit_progress(self, *, review: ContentCoverageReview, index: int, total: int, event: str, message: str) -> None:
        callback = self.progress_callback
        if not callable(callback):
            return
        try:
            callback(
                {
                    "event": event,
                    "builder": "coverage_backfill",
                    "review_id": str(review.review_id or ""),
                    "unit_id": str(review.unit_id or ""),
                    "page_no": int(review.page_no or 0),
                    "item_current": int(index),
                    "item_total": int(total),
                    "qwen_attempts": int(self._qwen_attempts),
                    "qwen_max_regions": int(self.qwen_vl_max_regions),
                    "message": message,
                }
            )
        except Exception:
            return

    def _targets(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        pdf_by_id: Dict[str, PdfEvidenceUnit],
    ) -> List[ContentCoverageReview]:
        eligible: List[ContentCoverageReview] = []
        seen_page_status: set[tuple[int, str]] = set()
        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered":
                continue
            if review.side == "pdf":
                unit = pdf_by_id.get(review.unit_id)
                if review.status not in self.PDF_TARGET_STATUSES:
                    continue
                if not self._pdf_unit_needs_backfill(unit=unit, review=review):
                    continue
                eligible.append(review)
                continue
            if review.side == "docx" and review.status in self.DOCX_TARGET_STATUSES and review.page_no:
                key = (int(review.page_no), review.status)
                if key in seen_page_status:
                    continue
                seen_page_status.add(key)
                eligible.append(review)
        eligible.sort(key=lambda item: self._target_sort_key(item, pdf_by_id=pdf_by_id))
        return self._balanced_targets(eligible)

    def _balanced_targets(self, eligible: Sequence[ContentCoverageReview]) -> List[ContentCoverageReview]:
        by_page: Dict[int, List[ContentCoverageReview]] = {}
        no_page: List[ContentCoverageReview] = []
        for review in eligible:
            page_no = int(review.page_no or 0)
            if page_no > 0:
                by_page.setdefault(page_no, []).append(review)
            else:
                no_page.append(review)
        rows: List[ContentCoverageReview] = []
        for round_index in range(self.max_per_page):
            for page_no in sorted(by_page):
                if len(rows) >= self.max_regions:
                    return rows
                page_rows = by_page.get(page_no) or []
                if round_index < len(page_rows):
                    rows.append(page_rows[round_index])
        remaining: List[ContentCoverageReview] = []
        for page_no in sorted(by_page):
            remaining.extend(by_page[page_no][self.max_per_page :])
        remaining.extend(no_page)
        for review in remaining:
            if len(rows) >= self.max_regions:
                break
            rows.append(review)
        return rows

    def _target_sort_key(
        self,
        review: ContentCoverageReview,
        *,
        pdf_by_id: Dict[str, PdfEvidenceUnit],
    ) -> tuple[int, int, int, str]:
        status_priority = {
            "visual_pending": 0,
            "needs_full_page_ocr": 1,
            "needs_region_segmentation": 2,
            "low_confidence_page_review": 3,
            "table_pending": 4,
            "uncovered_pdf_content": 5,
            "diff_candidate": 6,
            "mapping_uncertain": 7,
            "needs_pdf_ocr": 8,
        }
        unit = pdf_by_id.get(review.unit_id)
        unit_priority = 9
        if unit is not None:
            if unit.unit_type == "visual_region" and unit.unit_id.endswith("_visual_content"):
                unit_priority = 0
            elif unit.unit_type == "table_region":
                unit_priority = 1
            elif unit.region_type == "text_band":
                unit_priority = 2
            elif unit.region_type == "line_band":
                unit_priority = 3
            elif unit.unit_type == "visual_region":
                unit_priority = 4
        return (status_priority.get(review.status, 99), unit_priority, int(review.page_no or 9999), review.review_id)

    def _pdf_unit_needs_backfill(self, *, unit: PdfEvidenceUnit | None, review: ContentCoverageReview) -> bool:
        if unit is None:
            return False
        unit_type = str(unit.unit_type or "")
        route = str(review.next_route or "")
        if unit_type in {"visual_region", "table_region"} and bool(unit.bbox):
            return True
        if route in {
            "needs_region_ocr",
            "needs_qwen_vl",
            "needs_table_parser",
            "needs_full_page_ocr",
            "needs_region_segmentation",
            "needs_qwen_vl_page_gate",
        } and bool(unit.bbox):
            return True
        return not str(review.text or "").strip() and bool(unit.bbox)

    def _review(
        self,
        *,
        index: int,
        review: ContentCoverageReview,
        pdf_unit: PdfEvidenceUnit | None,
        rendered_pages: Dict[int, Path],
        crop_cache: Dict[tuple[int, str], str],
    ) -> ContentCoverageBackfillReview:
        flags = self._base_flags(review=review, pdf_unit=pdf_unit)
        page_no = int(review.page_no or 0)
        image_path = rendered_pages.get(page_no)
        if page_no <= 0 or image_path is None or not image_path.exists():
            return self._row(
                index=index,
                review=review,
                attempted=False,
                available=False,
                method="",
                status="missing_page_image",
                reason="未找到可用于局部回填的 PDF 页面渲染图。",
                next_route="needs_human_visual_review",
                flags=[*flags, "missing_page_image"],
            )
        crop_path = self._crop_for_review(
            review=review,
            pdf_unit=pdf_unit,
            image_path=image_path,
            crop_cache=crop_cache,
        )
        if not crop_path:
            return self._row(
                index=index,
                review=review,
                attempted=False,
                available=False,
                method="",
                status="crop_failed",
                reason="未能生成局部 PDF 截图，无法执行文本回填。",
                next_route="needs_human_visual_review",
                flags=[*flags, "crop_failed"],
            )
        absolute_crop = self.work_dir / crop_path
        ocr = self._vision_ocr(absolute_crop)
        if ocr.get("text"):
            text = str(ocr.get("text") or "").strip()
            normalized = normalize_text(text)
            confidence = self._quality_confidence(str(ocr.get("quality") or "medium"))
            status = "ocr_succeeded" if len(normalized) >= self.min_text_chars else "ocr_text_too_short"
            if (
                status == "ocr_succeeded"
                and confidence < 0.6
                and self.qwen_vl_enabled
                and self._qwen_attempts < self.qwen_vl_max_regions
                and self._should_use_qwen_for_backfill(review=review, pdf_unit=pdf_unit)
            ):
                qwen = self._qwen_ocr(absolute_crop=absolute_crop, review=review)
                qwen_text = str(qwen.get("text") or "").strip()
                qwen_normalized = normalize_text(qwen_text)
                if qwen_text and len(qwen_normalized) >= self.min_text_chars:
                    return self._row(
                        index=index,
                        review=review,
                        attempted=True,
                        available=True,
                        method="qwen_vl",
                        status="qwen_vl_refined_low_quality_ocr",
                        confidence=max(confidence, self._confidence(qwen.get("confidence"))),
                        quality=str(ocr.get("quality") or ""),
                        readability=str(qwen.get("readability") or "readable"),
                        extracted_text=qwen_text,
                        crop_path=crop_path,
                        reason=str(qwen.get("reason") or "本地 OCR 低置信，已用 Qwen-VL 重读局部截图。"),
                        next_route="needs_text_alignment",
                        flags=[*flags, "macos_vision_low_quality", "qwen_vl_refined_low_quality_ocr"],
                    )
            next_route = "needs_text_alignment" if status == "ocr_succeeded" else "needs_human_visual_review"
            return self._row(
                index=index,
                review=review,
                attempted=True,
                available=status == "ocr_succeeded",
                method="macos_vision",
                status=status,
                confidence=confidence,
                quality=str(ocr.get("quality") or ""),
                readability="readable" if status == "ocr_succeeded" else "partially_readable",
                extracted_text=text,
                crop_path=crop_path,
                reason="已用本地 macOS Vision 对未覆盖 PDF 区域完成局部 OCR 回填。",
                next_route=next_route,
                flags=[*flags, "macos_vision_backfill", *(["short_ocr_text"] if status != "ocr_succeeded" else [])],
            )
        if (
            not self.qwen_vl_enabled
            or self._qwen_attempts >= self.qwen_vl_max_regions
            or not self._should_use_qwen_for_backfill(review=review, pdf_unit=pdf_unit)
        ):
            return self._row(
                index=index,
                review=review,
                attempted=True,
                available=False,
                method="macos_vision",
                status="ocr_empty",
                confidence=0.0,
                quality=str(ocr.get("quality") or ""),
                crop_path=crop_path,
                reason="本地 OCR 未读出稳定文本，Qwen-VL 回填未启用或达到上限。",
                next_route="needs_qwen_vl" if self.qwen_vl_enabled else "needs_human_visual_review",
                error=str(ocr.get("error") or ""),
                flags=[*flags, "macos_vision_empty"],
            )
        qwen = self._qwen_ocr(absolute_crop=absolute_crop, review=review)
        text = str(qwen.get("text") or "").strip()
        normalized = normalize_text(text)
        if text and len(normalized) >= self.min_text_chars:
            return self._row(
                index=index,
                review=review,
                attempted=True,
                available=True,
                method="qwen_vl",
                status="qwen_vl_succeeded",
                confidence=self._confidence(qwen.get("confidence")),
                readability=str(qwen.get("readability") or "readable"),
                extracted_text=text,
                crop_path=crop_path,
                reason=str(qwen.get("reason") or "已用 Qwen-VL 对未覆盖 PDF 区域完成局部文本回填。"),
                next_route="needs_text_alignment",
                flags=[*flags, "qwen_vl_backfill"],
            )
        return self._row(
            index=index,
            review=review,
            attempted=True,
            available=False,
            method="qwen_vl" if qwen.get("attempted") else "macos_vision",
            status=str(qwen.get("status") or "qwen_vl_empty"),
            confidence=self._confidence(qwen.get("confidence")),
            readability=str(qwen.get("readability") or "unreadable"),
            crop_path=crop_path,
            reason=str(qwen.get("reason") or "局部截图仍未读出稳定文本。"),
            next_route=str(qwen.get("next_route") or "needs_human_visual_review"),
            error=str(qwen.get("error") or ocr.get("error") or ""),
            flags=[*flags, "coverage_backfill_unresolved"],
        )

    def _crop_for_review(
        self,
        *,
        review: ContentCoverageReview,
        pdf_unit: PdfEvidenceUnit | None,
        image_path: Path,
        crop_cache: Dict[tuple[int, str], str],
    ) -> str:
        page_no = int(review.page_no or 0)
        if review.side == "docx":
            key = (page_no, "full_page")
            cached = crop_cache.get(key)
            if cached:
                return cached
        bbox = list(getattr(pdf_unit, "bbox", []) or [])
        cache_key = (page_no, ",".join(str(round(float(value), 2)) for value in bbox[:4]) if bbox else "full_page")
        if cache_key in crop_cache:
            return crop_cache[cache_key]
        try:
            ensure_private_directory(self.crop_dir)
            with Image.open(image_path) as image:
                if bbox:
                    left, top, right, bottom = self._padded_bbox(
                        bbox,
                        width=image.width,
                        height=image.height,
                        unit_type=str(getattr(pdf_unit, "unit_type", "") or ""),
                    )
                else:
                    left, top, right, bottom = 0, 0, image.width, image.height
                if right <= left or bottom <= top:
                    return ""
                target = self.crop_dir / f"{review.review_id}_{review.unit_id}_p{page_no:04d}.jpg"
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.name)
                target = target.with_name(safe_name)
                image.crop((left, top, right, bottom)).convert("RGB").save(target, format="JPEG", quality=94)
            ensure_private_file(target)
            relative = target.relative_to(self.work_dir).as_posix()
            crop_cache[cache_key] = relative
            if review.side == "docx":
                crop_cache[(page_no, "full_page")] = relative
            return relative
        except Exception:
            logger.debug("Failed to crop content coverage review %s.", review.review_id, exc_info=True)
            return ""

    def _vision_ocr(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"text": "", "quality": "", "error": "crop_missing"}
        service = self.ocr_service
        if service is None:
            if not MacOSVisionOCRService.is_supported():
                return {"text": "", "quality": "", "error": "macos_vision_unavailable"}
            service = MacOSVisionOCRService(timeout=min(max(10, self.timeout), 90))
        try:
            if callable(service) and not hasattr(service, "extract_document_text_from_image_async"):
                payload = service(path)
            else:
                payload = asyncio.run(service.extract_document_text_from_image_async(path.read_bytes()))
            if not isinstance(payload, dict):
                return {"text": "", "quality": "", "error": "invalid_ocr_payload"}
            return {
                "text": str(payload.get("text") or "").strip(),
                "quality": str(payload.get("quality") or "").strip(),
                "warnings": list(payload.get("warnings") or []) if isinstance(payload.get("warnings"), list) else [],
            }
        except Exception as exc:
            logger.debug("Content coverage macOS Vision OCR failed for %s.", path, exc_info=True)
            return {"text": "", "quality": "", "error": f"{type(exc).__name__}: {exc}"}

    def _qwen_ocr(self, *, absolute_crop: Path, review: ContentCoverageReview) -> Dict[str, Any]:
        self._qwen_attempts += 1
        available, error = self._qwen_available()
        if not available:
            return {
                "attempted": False,
                "status": "qwen_vl_unavailable",
                "error": error or "qwen_vl_unavailable",
                "next_route": "needs_human_visual_review",
                "reason": "Qwen-VL 当前不可用，无法执行视觉文本回填。",
            }
        client = self._qwen_client()
        result = client.structured_chat_with_image(
            image_path=absolute_crop,
            system_prompt=self._qwen_system_prompt(),
            user_prompt=self._qwen_user_prompt(review=review),
            schema=coverage_backfill_schema(),
            timeout=self._qwen_timeout(review=review),
            num_predict=self._qwen_num_predict(review=review),
            temperature=0.0,
            allow_generate_fallback=False,
        )
        if not getattr(result, "ok", False):
            return {
                "attempted": True,
                "status": "qwen_vl_failed",
                "error": str(getattr(result, "error", "") or "qwen_vl_failed"),
                "next_route": "needs_human_visual_review",
                "reason": "Qwen-VL 回填请求失败。",
            }
        parsed = dict(getattr(result, "parsed", {}) or {})
        readability = self._choice(parsed.get("readability"), {"readable", "partially_readable", "unreadable", "not_text"}, "unreadable")
        text = self._clip_text(parsed.get("extracted_text"), limit=self._text_limit_for_review(review=review))
        if readability in {"unreadable", "not_text"}:
            text = ""
        return {
            "attempted": True,
            "status": "qwen_vl_empty" if not text else "qwen_vl_succeeded",
            "text": text,
            "readability": readability,
            "confidence": self._confidence(parsed.get("confidence")),
            "reason": self._clip_text(parsed.get("reason"), limit=110),
            "next_route": "needs_human_visual_review" if not text else "needs_text_alignment",
        }

    def _qwen_available(self) -> tuple[bool, str]:
        if self._qwen_preflight is not None:
            return self._qwen_preflight
        client = self._qwen_client()
        preflight = getattr(client, "preflight", None)
        if not callable(preflight):
            self._qwen_preflight = (True, "")
            return self._qwen_preflight
        result = preflight(timeout=min(30, max(1, self.timeout)))
        available = bool(result.get("available"))
        error = "" if available else str(result.get("error") or result.get("reason") or "qwen_vl_unavailable")
        self._qwen_preflight = (available, error)
        return self._qwen_preflight

    def _qwen_client(self) -> Any:
        if self.qwen_client is None:
            model = str(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MODEL", "qwen3-vl:8b") or "qwen3-vl:8b").strip()
            self.qwen_client = OllamaQwenVlClient(model=model)
        return self.qwen_client

    def _unload_qwen(self) -> None:
        client = self.qwen_client
        if client is None:
            return
        unload = getattr(client, "unload", None)
        if callable(unload):
            unload()

    def _qwen_system_prompt(self) -> str:
        return (
            "你是 WPS PDF 转 DOCX 审查的局部 OCR 回填模型。"
            "你只负责逐字读取截图中的可见文字，不做结论判断，不比较 DOCX，不补全截图外内容。"
            "看不清的字用 ? 标记；如果不是文字或无法阅读，返回 unreadable/not_text。必须只输出 JSON。"
        )

    def _qwen_user_prompt(self, *, review: ContentCoverageReview) -> str:
        payload = {
            "coverage_review": review.to_dict(),
            "rules": [
                "只读取截图内真实可见文字；不要根据上下文猜测。",
                "保留原始数字、金额、日期、案号、姓名、单位名和换行顺序。",
                "表格区域尽量按行输出，单元格之间可用 | 分隔。",
                "无法确定的字符用 ?；不要把 ? 替换成猜测字符。",
                "如果文字太模糊或截图不是文字，extracted_text 留空。",
            ],
        }
        return "请读取这张 PDF 局部截图中的全部可见文字。上下文 JSON：" + json.dumps(payload, ensure_ascii=False)[:4200]

    def _should_use_qwen_for_backfill(
        self,
        *,
        review: ContentCoverageReview,
        pdf_unit: PdfEvidenceUnit | None,
    ) -> bool:
        if pdf_unit is None:
            return True
        if pdf_unit.unit_type == "visual_region" and pdf_unit.unit_id.endswith("_visual_content"):
            return self._allow_full_page_qwen_backfill(review=review, pdf_unit=pdf_unit)
        if pdf_unit.region_type == "visual_content" and review.status == "visual_pending":
            return self._allow_full_page_qwen_backfill(review=review, pdf_unit=pdf_unit)
        return True

    def _allow_full_page_qwen_backfill(
        self,
        *,
        review: ContentCoverageReview,
        pdf_unit: PdfEvidenceUnit,
    ) -> bool:
        flags = set(str(item) for item in list(review.flags or []) + list(pdf_unit.flags or []))
        canonical = ""
        for flag in flags:
            if flag.startswith("canonical_page_type="):
                canonical = flag.split("=", 1)[1]
                break
        if canonical in {"table_image_page", "native_table_page"} or "table_like" in flags:
            return False
        return bool(canonical in {"scan_text_page", "mixed_layout_page", "low_confidence_page"} or "needs_ocr" in flags)

    def _qwen_timeout(self, *, review: ContentCoverageReview) -> int:
        base = max(self.timeout, int(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_TIMEOUT", 180) or 180))
        if self._is_full_page_review(review=review):
            return max(base, int(getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_FULL_PAGE_TIMEOUT", 300) or 300))
        return base

    def _qwen_num_predict(self, *, review: ContentCoverageReview) -> int:
        if self._is_full_page_review(review=review):
            return max(800, int(getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_FULL_PAGE_NUM_PREDICT", 1800) or 1800))
        return 640

    def _is_full_page_review(self, *, review: ContentCoverageReview) -> bool:
        unit_id = str(review.unit_id or "")
        flags = set(str(item) for item in review.flags or [])
        return bool(unit_id.endswith("_visual_content") or "region_type=visual_content" in flags)

    def _text_limit_for_review(self, *, review: ContentCoverageReview) -> int:
        if self._is_full_page_review(review=review):
            return self.max_text_chars
        return min(self.max_text_chars, 1800)

    def _padded_bbox(self, bbox: Sequence[float], *, width: int, height: int, unit_type: str) -> tuple[int, int, int, int]:
        values = list(bbox)[:4]
        if len(values) < 4:
            return 0, 0, width, height
        left, top, right, bottom = [float(item) for item in values]
        box_width = max(1.0, right - left)
        box_height = max(1.0, bottom - top)
        if unit_type == "table_region":
            pad_x = max(8.0, box_width * 0.02)
            pad_y = max(8.0, box_height * 0.02)
        elif box_width * box_height > width * height * 0.45:
            pad_x = max(2.0, box_width * 0.01)
            pad_y = max(2.0, box_height * 0.01)
        else:
            pad_x = max(24.0, box_width * 0.18)
            pad_y = max(18.0, box_height * 0.35)
        return (
            max(0, int(left - pad_x)),
            max(0, int(top - pad_y)),
            min(width, int(right + pad_x)),
            min(height, int(bottom + pad_y)),
        )

    def _base_flags(
        self,
        *,
        review: ContentCoverageReview,
        pdf_unit: PdfEvidenceUnit | None,
    ) -> List[str]:
        flags = [f"coverage_status={review.status}", f"coverage_side={review.side}"]
        flags.extend(str(item) for item in review.flags if str(item).strip())
        if pdf_unit is not None:
            if pdf_unit.unit_type:
                flags.append(f"unit_type={pdf_unit.unit_type}")
            if pdf_unit.region_type:
                flags.append(f"region_type={pdf_unit.region_type}")
        return list(dict.fromkeys(flags))

    def _row(
        self,
        *,
        index: int,
        review: ContentCoverageReview,
        attempted: bool,
        available: bool,
        method: str,
        status: str,
        confidence: float = 0.0,
        quality: str = "",
        readability: str = "",
        extracted_text: str = "",
        crop_path: str = "",
        reason: str = "",
        next_route: str = "",
        error: str = "",
        flags: Sequence[str] = (),
    ) -> ContentCoverageBackfillReview:
        text = self._clip_text(extracted_text, limit=self._text_limit_for_review(review=review))
        return ContentCoverageBackfillReview(
            backfill_id=f"coverage_backfill_{index:04d}",
            coverage_review_id=review.review_id,
            side=review.side,
            unit_id=review.unit_id,
            page_no=review.page_no,
            attempted=attempted,
            available=available,
            method=method,
            status=status,
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            quality=quality,
            readability=readability,
            extracted_text=text,
            normalized_text=normalize_text(text),
            crop_path=crop_path,
            reason=reason,
            next_route=next_route,
            error=error,
            flags=list(dict.fromkeys(str(item) for item in flags if str(item).strip())),
        )

    def _quality_confidence(self, quality: str) -> float:
        value = str(quality or "").lower()
        if value == "high":
            return 0.88
        if value == "medium":
            return 0.72
        if value == "low":
            return 0.48
        return 0.62

    def _confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _choice(self, value: Any, allowed: Sequence[str] | set[str], fallback: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else fallback

    def _clip_text(self, value: Any, *, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]


class ContentCoverageBackfillResolver:
    """Feed successful OCR backfills back into full-content coverage decisions.

    The backfill builder only creates evidence. This resolver is still
    conservative: it covers ordinary DOCX paragraphs only when same-page OCR
    text contains the DOCX text exactly or with a high window similarity. Table
    cells stay in table-specific review, because whole-page OCR is not reliable
    enough to prove cell-level fidelity.
    """

    def __init__(
        self,
        *,
        min_page_text_chars: Optional[int] = None,
        min_docx_chars: Optional[int] = None,
        fuzzy_threshold: Optional[float] = None,
    ) -> None:
        self.min_page_text_chars = max(
            1,
            int(
                min_page_text_chars
                if min_page_text_chars is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_RESOLVE_MIN_PAGE_TEXT_CHARS", 20)
                or 20
            ),
        )
        self.min_docx_chars = max(
            1,
            int(
                min_docx_chars
                if min_docx_chars is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_RESOLVE_MIN_DOCX_CHARS", 4)
                or 4
            ),
        )
        self.fuzzy_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    fuzzy_threshold
                    if fuzzy_threshold is not None
                    else getattr(settings, "PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_RESOLVE_FUZZY_THRESHOLD", 0.9)
                    or 0.9
                ),
            ),
        )

    def apply(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        docx_by_id = {unit.unit_id: unit for unit in preflight_result.docx_units}
        pdf_by_id = {unit.unit_id: unit for unit in preflight_result.pdf_units}
        page_backfill_texts, backfill_by_review_id, full_page_backfill_by_page = self._page_backfill_texts(
            preflight_result.content_coverage_backfills
        )
        page_docx_texts = self._page_docx_texts(preflight_result.docx_units)
        summary: Dict[str, Any] = {
            "enabled": True,
            "version": "content_coverage_backfill_reconciliation_v1",
            "page_text_count": len(page_backfill_texts),
            "page_text_chars": sum(len(text) for text in page_backfill_texts.values()),
            "resolved_review_count": 0,
            "resolved_docx_count": 0,
            "resolved_docx_exact_count": 0,
            "resolved_docx_fuzzy_count": 0,
            "resolved_pdf_visual_carrier_count": 0,
            "resolved_pdf_visual_subregion_count": 0,
            "resolved_pdf_text_count": 0,
            "table_cell_skipped_count": 0,
            "resolved_review_ids": [],
        }
        if not page_backfill_texts:
            preflight_result.content_coverage_reconciliation = summary
            return summary

        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered":
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            if review.side == "docx":
                unit = docx_by_id.get(review.unit_id)
                if unit is not None and unit.container_type == "table_cell":
                    summary["table_cell_skipped_count"] += 1
                    continue
                match = self._docx_backfill_match(review=review, page_text=page_backfill_texts.get(page_no, ""))
                if match is None:
                    continue
                status, confidence, match_kind = match
                self._mark_covered(
                    review,
                    status=status,
                    confidence=confidence,
                    reason="DOCX 内容已在同页 OCR 回填文本中命中，未覆盖状态降级为已覆盖证据。",
                    flags=["covered_by_backfill_ocr", f"backfill_match={match_kind}"],
                )
                summary["resolved_docx_count"] += 1
                summary["resolved_review_count"] += 1
                summary[f"resolved_docx_{match_kind}_count"] += 1
                summary["resolved_review_ids"].append(review.review_id)
                continue

            pdf_unit = pdf_by_id.get(review.unit_id)
            backfill = backfill_by_review_id.get(review.review_id)
            if backfill is not None and self._is_visual_carrier(review=review, unit=pdf_unit):
                self._mark_covered(
                    review,
                    status="covered_by_backfill_ocr",
                    confidence=max(float(review.confidence or 0.0), float(backfill.confidence or 0.0), 0.72),
                    reason="PDF 图片/视觉区域已经被 OCR 回填读取，后续由同页 DOCX 单元对齐结果承接内容核查。",
                    flags=["covered_by_backfill_ocr", f"backfill_id={backfill.backfill_id}"],
                )
                summary["resolved_pdf_visual_carrier_count"] += 1
                summary["resolved_review_count"] += 1
                summary["resolved_review_ids"].append(review.review_id)
                continue
            page_full_backfill = full_page_backfill_by_page.get(page_no)
            if page_full_backfill is not None and self._is_visual_subregion_carrier(review=review, unit=pdf_unit):
                self._mark_covered(
                    review,
                    status="covered_by_page_backfill_ocr",
                    confidence=max(float(review.confidence or 0.0), float(page_full_backfill.confidence or 0.0), 0.68),
                    reason="同页整页 PDF 图片区域已经 OCR 回填成功，该视觉子区域作为承载区域不再单独计入未覆盖缺口。",
                    flags=[
                        "covered_by_page_backfill_ocr",
                        f"backfill_id={page_full_backfill.backfill_id}",
                    ],
                )
                summary["resolved_pdf_visual_subregion_count"] += 1
                summary["resolved_review_count"] += 1
                summary["resolved_review_ids"].append(review.review_id)
                continue
            if page_full_backfill is not None and self._is_superseded_pdf_ocr_text(review=review, unit=pdf_unit):
                self._mark_covered(
                    review,
                    status="covered_by_page_backfill_ocr",
                    confidence=max(float(review.confidence or 0.0), float(page_full_backfill.confidence or 0.0), 0.68),
                    reason="同页整页 OCR 回填已提供更完整的 PDF 文本证据，该旧 OCR 文本单元不再单独计入未覆盖缺口。",
                    flags=[
                        "covered_by_page_backfill_ocr",
                        "superseded_pdf_ocr_text",
                        f"backfill_id={page_full_backfill.backfill_id}",
                    ],
                )
                summary["resolved_pdf_text_count"] += 1
                summary["resolved_review_count"] += 1
                summary["resolved_review_ids"].append(review.review_id)
                continue
            if self._pdf_text_covered_by_docx(review=review, unit=pdf_unit, page_docx_text=page_docx_texts.get(page_no, "")):
                self._mark_covered(
                    review,
                    status="covered_by_docx_text",
                    confidence=max(float(review.confidence or 0.0), 0.74),
                    reason="PDF 文本片段已在同页 DOCX 文本中命中，未覆盖状态降级为已覆盖证据。",
                    flags=["covered_by_docx_text"],
                )
                summary["resolved_pdf_text_count"] += 1
                summary["resolved_review_count"] += 1
                summary["resolved_review_ids"].append(review.review_id)

        summary["resolved_review_ids"] = summary["resolved_review_ids"][:500]
        preflight_result.content_coverage_reconciliation = summary
        return summary

    def _page_backfill_texts(
        self,
        backfills: Sequence[ContentCoverageBackfillReview],
    ) -> tuple[Dict[int, str], Dict[str, ContentCoverageBackfillReview], Dict[int, ContentCoverageBackfillReview]]:
        by_page: Dict[int, List[str]] = {}
        by_review_id: Dict[str, ContentCoverageBackfillReview] = {}
        full_page_by_page: Dict[int, ContentCoverageBackfillReview] = {}
        for backfill in backfills:
            if not backfill.available:
                continue
            page_no = int(backfill.page_no or 0)
            text = normalize_text(backfill.extracted_text or backfill.normalized_text or "")
            if page_no <= 0 or len(text) < self.min_page_text_chars:
                continue
            by_page.setdefault(page_no, []).append(text)
            if backfill.coverage_review_id:
                by_review_id[backfill.coverage_review_id] = backfill
            if self._is_full_page_backfill(backfill):
                existing = full_page_by_page.get(page_no)
                if existing is None or float(backfill.confidence or 0.0) > float(existing.confidence or 0.0):
                    full_page_by_page[page_no] = backfill
        return {page_no: "\n".join(parts) for page_no, parts in by_page.items()}, by_review_id, full_page_by_page

    def _page_docx_texts(self, docx_units: Sequence[DocxEvidenceUnit]) -> Dict[int, str]:
        by_page: Dict[int, List[str]] = {}
        for unit in docx_units:
            page_no = int(unit.estimated_page_no or 0)
            text = normalize_text(unit.text)
            if page_no > 0 and text:
                by_page.setdefault(page_no, []).append(text)
        return {page_no: "\n".join(parts) for page_no, parts in by_page.items()}

    def _docx_backfill_match(self, *, review: ContentCoverageReview, page_text: str) -> tuple[str, float, str] | None:
        needle = normalize_text(review.text)
        haystack = normalize_text(page_text)
        if len(needle) < self.min_docx_chars or not haystack:
            return None
        if needle in haystack:
            return ("covered_by_backfill_ocr", max(float(review.confidence or 0.0), 0.82), "exact")
        if len(needle) < 14:
            return None
        score = self._best_window_ratio(needle=needle, haystack=haystack)
        if score >= self.fuzzy_threshold:
            return ("covered_by_backfill_ocr_fuzzy", max(float(review.confidence or 0.0), min(0.8, 0.58 + score * 0.22)), "fuzzy")
        return None

    def _pdf_text_covered_by_docx(
        self,
        *,
        review: ContentCoverageReview,
        unit: PdfEvidenceUnit | None,
        page_docx_text: str,
    ) -> bool:
        if unit is None or str(unit.unit_type or "") in {"visual_region", "table_region"}:
            return False
        text = normalize_text(review.text or unit.text)
        if len(text) < self.min_docx_chars or not page_docx_text:
            return False
        return text in normalize_text(page_docx_text)

    def _is_visual_carrier(self, *, review: ContentCoverageReview, unit: PdfEvidenceUnit | None) -> bool:
        if unit is None:
            return False
        unit_type = str(unit.unit_type or "")
        region_type = str(unit.region_type or "")
        if unit_type != "visual_region":
            return False
        if review.status not in {
            "visual_pending",
            "needs_full_page_ocr",
            "needs_region_segmentation",
            "low_confidence_page_review",
            "needs_text_alignment",
        }:
            return False
        return bool(unit.unit_id.endswith("_visual_content") or region_type == "visual_content")

    def _is_visual_subregion_carrier(self, *, review: ContentCoverageReview, unit: PdfEvidenceUnit | None) -> bool:
        if unit is None:
            return False
        if str(unit.unit_type or "") != "visual_region":
            return False
        if self._is_visual_carrier(review=review, unit=unit):
            return False
        if review.status not in {
            "visual_pending",
            "needs_full_page_ocr",
            "needs_region_segmentation",
            "low_confidence_page_review",
            "needs_text_alignment",
        }:
            return False
        return bool(unit.bbox)

    def _is_full_page_backfill(self, backfill: ContentCoverageBackfillReview) -> bool:
        unit_id = str(backfill.unit_id or "")
        flags = set(str(item) for item in backfill.flags)
        return bool(
            unit_id.endswith("_visual_content")
            or "region_type=visual_content" in flags
        )

    def _is_superseded_pdf_ocr_text(self, *, review: ContentCoverageReview, unit: PdfEvidenceUnit | None) -> bool:
        if unit is None:
            return False
        if review.status not in {"needs_full_page_ocr", "needs_text_alignment"}:
            return False
        return str(unit.unit_type or "") in {"anchor_ocr_page", "anchor_ocr_line", "native_text_block"}

    def _mark_covered(
        self,
        review: ContentCoverageReview,
        *,
        status: str,
        confidence: float,
        reason: str,
        flags: Sequence[str],
    ) -> None:
        review.status = status
        review.decision = "covered"
        review.confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        review.reason = reason
        review.next_route = ""
        review.flags = list(dict.fromkeys([*review.flags, *[str(flag) for flag in flags if str(flag).strip()]]))

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
            if best >= 0.96:
                break
        return best
