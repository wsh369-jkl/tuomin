from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from app.core.config import settings
from app.core.runtime_security import ensure_private_directory, ensure_private_file
from app.services.macos_vision_ocr import MacOSVisionOCRService

from .common import has_high_value_field_content, looks_like_document_title, looks_like_organization_name, looks_like_table_title, normalize_text


class PageOrientationNormalizer:
    """Normalize rendered PDF page images to upright orientation before OCR.

    The source PDF is not modified. Downstream v4 stages receive rotated page
    images when a high-confidence orientation decision exists, and the decision
    is recorded as raw evidence.
    """

    VERSION = "page_orientation_v1"
    ROTATIONS = (0, 90, 180, 270)

    def __init__(
        self,
        *,
        work_dir: Path,
        enabled: bool | None = None,
        client: Any = None,
        timeout: int | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_NORMALIZE_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_edge = max(600, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MAX_EDGE", 1000) or 1000))
        self.timeout = max(10, int(timeout or getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_TIMEOUT", 30) or 30))
        self.max_ocr_pages = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MAX_OCR_PAGES", 30) or 30))
        self.max_ocr_candidates = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MAX_OCR_CANDIDATES", 4) or 4))
        self.min_confidence = max(
            0.0,
            min(1.0, float(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MIN_CONFIDENCE", 0.62) or 0.62)),
        )
        self.client = client
        self._ocr_pages_used = 0

    def normalize(self, *, rendered_pages: Dict[int, Path]) -> Tuple[Dict[int, Path], Dict[str, Any]]:
        if not self.enabled:
            return dict(rendered_pages), self._payload(enabled=False, pages=[], warnings=["page_orientation_disabled"])
        if not rendered_pages:
            return {}, self._payload(enabled=True, pages=[], warnings=[])
        if self.client is None:
            if not MacOSVisionOCRService.is_supported():
                return dict(rendered_pages), self._payload(
                    enabled=True,
                    pages=[],
                    warnings=["page_orientation_macos_vision_unavailable"],
                )
            self.client = MacOSVisionOCRService(timeout=self.timeout)

        normalized: Dict[int, Path] = dict(rendered_pages)
        page_reviews: List[Dict[str, Any]] = []
        warnings: List[str] = []
        output_dir = self.work_dir / "pages_normalized"
        ensure_private_directory(output_dir)
        self._ocr_pages_used = 0
        for page_no, image_path in sorted(rendered_pages.items()):
            review = self._review_page(page_no=int(page_no), image_path=Path(image_path), output_dir=output_dir)
            page_reviews.append(review)
            if review.get("normalized_path"):
                normalized_path = Path(str(review["normalized_path"]))
                if not normalized_path.is_absolute():
                    normalized_path = self.work_dir / normalized_path
                normalized[int(page_no)] = normalized_path
            if review.get("warning"):
                warnings.append(str(review["warning"]))
        return normalized, self._payload(enabled=True, pages=page_reviews, warnings=warnings)

    def _review_page(self, *, page_no: int, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        if not image_path.exists():
            return {
                "page_no": page_no,
                "attempted": False,
                "rotation_degrees": 0,
                "confidence": 0.0,
                "normalized": False,
                "original_path": str(image_path),
                "normalized_path": "",
                "reason": "page_image_missing",
                "warning": f"page_orientation_missing_image_{page_no}",
                "candidates": [],
            }
        try:
            visual_candidates = [
                self._visual_orientation_candidate(page_no=page_no, image_path=image_path, degrees=degrees)
                for degrees in self.ROTATIONS
            ]
        except Exception as exc:
            return {
                "page_no": page_no,
                "attempted": True,
                "rotation_degrees": 0,
                "confidence": 0.0,
                "normalized": False,
                "original_path": str(image_path),
                "normalized_path": str(image_path),
                "reason": f"orientation_failed:{type(exc).__name__}",
                "warning": f"page_orientation_failed_{page_no}",
                "candidates": [],
            }
        visual_best, visual_second = self._best_two(visual_candidates)
        visual_confidence = self._visual_confidence(best=visual_best, second=visual_second)
        visual_degrees = int(visual_best.get("rotation_degrees") or 0)
        if self._can_keep_original_from_visual(best=visual_best, confidence=visual_confidence):
            return self._orientation_review(
                page_no=page_no,
                image_path=image_path,
                output_dir=output_dir,
                best=visual_best,
                second=visual_second,
                confidence=visual_confidence,
                candidates=visual_candidates,
                ocr_attempted=False,
                warning="",
            )
        if visual_degrees == 0 or self._ocr_pages_used >= self.max_ocr_pages:
            warning = "" if visual_degrees == 0 else f"page_orientation_ocr_budget_deferred_{page_no}"
            return self._orientation_review(
                page_no=page_no,
                image_path=image_path,
                output_dir=output_dir,
                best={**visual_best, "rotation_degrees": 0, "score": max(float(visual_best.get("score") or 0.0), 0.0)},
                second=visual_second,
                confidence=min(visual_confidence, self.min_confidence - 0.01),
                candidates=visual_candidates,
                ocr_attempted=False,
                warning=warning,
            )

        self._ocr_pages_used += 1
        try:
            degrees_to_ocr = self._ocr_candidate_degrees(visual_candidates)
            candidates = [
                self._orientation_candidate(page_no=page_no, image_path=image_path, degrees=degrees)
                for degrees in degrees_to_ocr
            ]
        except Exception as exc:
            return self._orientation_review(
                page_no=page_no,
                image_path=image_path,
                output_dir=output_dir,
                best={**visual_best, "rotation_degrees": 0},
                second=visual_second,
                confidence=min(visual_confidence, self.min_confidence - 0.01),
                candidates=visual_candidates,
                ocr_attempted=True,
                warning=f"page_orientation_ocr_failed_{page_no}:{type(exc).__name__}",
            )
        best, second = self._best_two(candidates)
        confidence = self._confidence(best=best, second=second)
        return self._orientation_review(
            page_no=page_no,
            image_path=image_path,
            output_dir=output_dir,
            best=best,
            second=second,
            confidence=confidence,
            candidates=candidates,
            ocr_attempted=True,
            warning="",
        )

    def _orientation_review(
        self,
        *,
        page_no: int,
        image_path: Path,
        output_dir: Path,
        best: Dict[str, Any],
        second: Dict[str, Any],
        confidence: float,
        candidates: Sequence[Dict[str, Any]],
        ocr_attempted: bool,
        warning: str,
    ) -> Dict[str, Any]:
        degrees = int(best.get("rotation_degrees") or 0)
        force_sideways = self._sideways_rotation_supported(best=best, candidates=candidates)
        should_rotate = bool(degrees and (confidence >= self.min_confidence or force_sideways))
        if should_rotate and force_sideways and confidence < self.min_confidence:
            best = {**best, "forced_sideways_rotation": True}
            confidence = self.min_confidence
        normalized_path = image_path
        if should_rotate:
            normalized_path = output_dir / f"page_{page_no:04d}_upright.jpg"
            self._write_rotated(image_path=image_path, output_path=normalized_path, degrees=degrees)
        reason = self._reason(best=best, second=second, confidence=confidence, rotated=should_rotate)
        return {
            "page_no": page_no,
            "attempted": True,
            "rotation_degrees": degrees if should_rotate else 0,
            "detected_rotation_degrees": degrees,
            "confidence": confidence,
            "normalized": should_rotate,
            "original_path": self._relative_path(image_path),
            "normalized_path": self._relative_path(normalized_path),
            "reason": reason,
            "ocr_attempted": bool(ocr_attempted),
            "engine": str(best.get("engine") or ("ocr_score" if ocr_attempted else "visual_projection")),
            "warning": warning,
            "candidates": candidates,
        }

    def _sideways_rotation_supported(self, *, best: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> bool:
        """Rotate when OCR strongly says the original page is sideways.

        Some photographed pages produce nearly tied 90/270 scores because both
        rotations contain readable text, while the original 0-degree image has
        vertical text columns. In that case the best-vs-second confidence is
        intentionally low, but keeping the original sideways page is worse for
        downstream OCR/VL. This check compares the best rotated candidate with
        the original orientation rather than only with the opposite rotation.
        """

        degrees = int(best.get("rotation_degrees") or 0)
        if degrees not in {90, 270}:
            return False
        best_score = float(best.get("score") or 0.0)
        if best_score < 45.0:
            return False
        if int(best.get("text_chars") or 0) < 24 and int(best.get("wide_line_count") or 0) < 4:
            return False
        original = next((item for item in candidates if int(item.get("rotation_degrees") or 0) == 0), {})
        original_score = float(original.get("score") or 0.0)
        original_tall = int(original.get("tall_line_count") or 0)
        original_wide = int(original.get("wide_line_count") or 0)
        best_wide = int(best.get("wide_line_count") or 0)
        best_tall = int(best.get("tall_line_count") or 0)
        score_gap = best_score - original_score
        original_looks_sideways = original_tall >= max(4, original_wide * 2)
        best_looks_horizontal = best_wide >= max(3, best_tall + 2)
        return bool(score_gap >= 35.0 and (original_looks_sideways or best_looks_horizontal))

    def _visual_orientation_candidate(self, *, page_no: int, image_path: Path, degrees: int) -> Dict[str, Any]:
        with Image.open(image_path) as source:
            image = source.convert("L")
            if degrees:
                image = image.rotate(int(degrees), expand=True)
            image.thumbnail((min(self.max_edge, 700), min(self.max_edge, 700)), Image.Resampling.LANCZOS)
            width, height = image.size
            pixels = image.load()
            row_counts = [0 for _ in range(height)]
            col_counts = [0 for _ in range(width)]
            for y in range(height):
                row_total = 0
                for x in range(width):
                    if pixels[x, y] < 245:
                        row_total += 1
                        col_counts[x] += 1
                row_counts[y] = row_total
        dark_pixels = sum(row_counts)
        row_peak = self._projection_peak(row_counts, dark_pixels)
        col_peak = self._projection_peak(col_counts, dark_pixels)
        wide_line_count = sum(1 for value in row_counts if value >= max(4, int(width * 0.035)))
        tall_line_count = sum(1 for value in col_counts if value >= max(4, int(height * 0.035)))
        score = max(0.0, (row_peak - col_peak) * 100.0 + wide_line_count * 0.12 - tall_line_count * 0.05)
        if int(degrees) == 0:
            score += 0.8
        return {
            "rotation_degrees": int(degrees),
            "score": round(float(score), 4),
            "text_chars": 0,
            "line_count": wide_line_count,
            "wide_line_count": wide_line_count,
            "tall_line_count": tall_line_count,
            "quality": "visual",
            "text_excerpt": "",
            "engine": "visual_projection",
            "dark_pixel_ratio": round(dark_pixels / max(1, width * height), 6),
            "row_peak": round(row_peak, 6),
            "col_peak": round(col_peak, 6),
        }

    def _projection_peak(self, values: Sequence[int], total: int) -> float:
        if not values or total <= 0:
            return 0.0
        top_n = max(3, int(len(values) * 0.08))
        return sum(sorted((int(value) for value in values), reverse=True)[:top_n]) / max(1, int(total))

    def _visual_confidence(self, *, best: Dict[str, Any], second: Dict[str, Any]) -> float:
        best_score = max(0.0, float(best.get("score") or 0.0))
        second_score = max(0.0, float(second.get("score") or 0.0))
        if best_score <= 0:
            return 0.0
        margin = max(0.0, best_score - second_score)
        line_count = int(best.get("line_count") or 0)
        return round(max(0.0, min(0.92, 0.50 + (margin / max(1.0, best_score)) * 0.34 + min(0.08, line_count / 500.0))), 4)

    def _can_keep_original_from_visual(self, *, best: Dict[str, Any], confidence: float) -> bool:
        if int(best.get("rotation_degrees") or 0) != 0:
            return False
        if int(best.get("line_count") or 0) < 3:
            return False
        return bool(confidence >= max(0.64, self.min_confidence))

    def _ocr_candidate_degrees(self, visual_candidates: Sequence[Dict[str, Any]]) -> List[int]:
        ordered = sorted(visual_candidates, key=lambda item: (-float(item.get("score") or 0.0), int(item.get("rotation_degrees") or 0)))
        degrees: List[int] = []
        for candidate in ordered:
            degree = int(candidate.get("rotation_degrees") or 0)
            if degree not in degrees:
                degrees.append(degree)
            if degree in {90, 270}:
                opposite = 270 if degree == 90 else 90
                if opposite not in degrees:
                    degrees.append(opposite)
            if degree in {0, 180}:
                opposite = 180 if degree == 0 else 0
                if opposite not in degrees:
                    degrees.append(opposite)
            if len(degrees) >= self.max_ocr_candidates:
                break
        if 0 not in degrees and len(degrees) < self.max_ocr_candidates:
            degrees.append(0)
        if 0 in degrees and 180 not in degrees and len(degrees) < self.max_ocr_candidates:
            degrees.append(180)
        return degrees[: self.max_ocr_candidates]

    def _orientation_candidate(self, *, page_no: int, image_path: Path, degrees: int) -> Dict[str, Any]:
        variant_path = self._variant_path(page_no=page_no, degrees=degrees)
        self._write_variant(image_path=image_path, output_path=variant_path, degrees=degrees)
        result = self._ocr_image(variant_path)
        lines = [line for line in result.get("lines") or [] if isinstance(line, dict)]
        text = str(result.get("text") or "")
        metrics = self._line_metrics(lines)
        score = self._score(text=text, metrics=metrics)
        return {
            "rotation_degrees": int(degrees),
            "score": round(float(score), 4),
            "text_chars": len(normalize_text(text)),
            "line_count": len(lines),
            "wide_line_count": metrics["wide_line_count"],
            "tall_line_count": metrics["tall_line_count"],
            "quality": str(result.get("quality") or ""),
            "text_excerpt": " ".join(text.split())[:180],
        }

    def _ocr_image(self, path: Path) -> Dict[str, Any]:
        extract = getattr(self.client, "extract_document_text_from_image_async", None)
        if not callable(extract):
            return {"text": "", "quality": "low", "lines": [], "warnings": ["orientation_ocr_client_unavailable"]}
        return asyncio.run(extract(Path(path).read_bytes(), page_number=None, total_pages=None))

    def _score(self, *, text: str, metrics: Dict[str, int]) -> float:
        compact = normalize_text(text)
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
        digit_count = sum(char.isdigit() for char in text)
        keyword_bonus = self._keyword_bonus(text)
        return (
            len(compact) * 0.2
            + cjk_count * 0.2
            + digit_count * 0.05
            + metrics["wide_line_count"] * 8.0
            - metrics["tall_line_count"] * 8.0
            + keyword_bonus
        )

    def _keyword_bonus(self, text: str) -> float:
        value = str(text or "")
        lines = [normalize_text(line) for line in re.split(r"[\r\n]+", value) if str(line).strip()]
        compact = normalize_text(value)
        bonus = 0.0
        title_line_index = next(
            (
                index
                for index, line in enumerate(lines[:4])
                if looks_like_document_title(line) or looks_like_table_title(line)
            ),
            -1,
        )
        if title_line_index == 0:
            bonus += 120.0
        elif title_line_index > 0:
            bonus += 70.0 if title_line_index <= 2 else 28.0
        elif re.search(r"^\d{1,4}[年月日/-]|^20\d{2}年", compact[:24]):
            bonus -= 18.0
        if any(looks_like_organization_name(line) for line in lines[:6]):
            bonus += 16.0
        if any(looks_like_organization_name(line) for line in lines[-4:]):
            bonus += 12.0
        bonus += self._proof_order_bonus(compact)
        if has_high_value_field_content(value):
            bonus += 8.0
        return bonus

    def _proof_order_bonus(self, compact: str) -> float:
        if not compact:
            return 0.0
        chunks = [chunk for chunk in re.split(r"[。；;\n]+", compact) if chunk]
        title_idx = next(
            (
                index
                for index, chunk in enumerate(chunks[:5])
                if looks_like_document_title(chunk) or looks_like_table_title(chunk)
            ),
            -1,
        )
        org_idx = next(
            (index for index, chunk in enumerate(chunks) if looks_like_organization_name(chunk)),
            -1,
        )
        if title_idx == 0 and org_idx > title_idx:
            return 24.0
        if title_idx > 0 and org_idx >= 0 and org_idx < title_idx:
            return -16.0
        return 0.0

    def _line_metrics(self, lines: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        wide = 0
        tall = 0
        for line in lines:
            bbox = line.get("bbox") or []
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            try:
                width = max(0, int(bbox[2]) - int(bbox[0]))
                height = max(0, int(bbox[3]) - int(bbox[1]))
            except Exception:
                continue
            if width >= height * 1.45:
                wide += 1
            if height >= width * 1.45:
                tall += 1
        return {"wide_line_count": wide, "tall_line_count": tall}

    def _best_two(self, candidates: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rows = sorted(candidates, key=lambda item: (-float(item.get("score") or 0.0), int(item.get("rotation_degrees") or 0)))
        if not rows:
            return {"rotation_degrees": 0, "score": 0.0}, {"rotation_degrees": 0, "score": 0.0}
        if len(rows) == 1:
            return rows[0], {"rotation_degrees": 0, "score": 0.0}
        return rows[0], rows[1]

    def _confidence(self, *, best: Dict[str, Any], second: Dict[str, Any]) -> float:
        best_score = max(0.0, float(best.get("score") or 0.0))
        second_score = max(0.0, float(second.get("score") or 0.0))
        if best_score <= 0:
            return 0.0
        margin = max(0.0, best_score - second_score)
        margin_ratio = margin / max(1.0, best_score)
        line_count = int(best.get("line_count") or 0)
        evidence_part = min(0.18, line_count / 220.0)
        return round(max(0.0, min(0.98, 0.48 + margin_ratio * 0.34 + evidence_part)), 4)

    def _reason(self, *, best: Dict[str, Any], second: Dict[str, Any], confidence: float, rotated: bool) -> str:
        action = "已正向化" if rotated else "保持原方向"
        reason = (
            f"{action}。最佳旋转 {int(best.get('rotation_degrees') or 0)} 度得分 {float(best.get('score') or 0.0):.1f}，"
            f"次优 {int(second.get('rotation_degrees') or 0)} 度得分 {float(second.get('score') or 0.0):.1f}，"
            f"置信度 {confidence:.2f}。"
        )
        if best.get("forced_sideways_rotation"):
            reason += "原方向呈明显侧向文本，已按最佳 OCR 方向强制正向化。"
        return reason

    def _write_variant(self, *, image_path: Path, output_path: Path, degrees: int) -> None:
        if output_path.exists():
            return
        ensure_private_directory(output_path.parent)
        with Image.open(image_path) as image:
            output = image.convert("RGB")
            if degrees:
                output = output.rotate(int(degrees), expand=True)
            output.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
            output.save(output_path, format="JPEG", quality=88)
        ensure_private_file(output_path)

    def _write_rotated(self, *, image_path: Path, output_path: Path, degrees: int) -> None:
        ensure_private_directory(output_path.parent)
        with Image.open(image_path) as image:
            output = image.convert("RGB")
            if degrees:
                output = output.rotate(int(degrees), expand=True)
            output.save(output_path, format="JPEG", quality=92)
        ensure_private_file(output_path)

    def _variant_path(self, *, page_no: int, degrees: int) -> Path:
        path = self.work_dir / "evidence" / "page_orientation"
        ensure_private_directory(path)
        return path / f"page_{page_no:04d}_rot{int(degrees):03d}.jpg"

    def _relative_path(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.work_dir.resolve()))
        except Exception:
            return str(path)

    def _payload(self, *, enabled: bool, pages: Sequence[Dict[str, Any]], warnings: Sequence[str]) -> Dict[str, Any]:
        normalized_pages = [page for page in pages if page.get("normalized")]
        return {
            "enabled": bool(enabled),
            "version": self.VERSION,
            "summary": {
                "enabled": bool(enabled),
                "version": self.VERSION,
                "attempted_page_count": sum(1 for page in pages if page.get("attempted")),
                "normalized_page_count": len(normalized_pages),
                "rotation_counts": self._rotation_counts(normalized_pages),
                "warnings": [str(item) for item in warnings if str(item)],
            },
            "pages": [dict(page) for page in pages],
        }

    def _rotation_counts(self, pages: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        rows: Dict[str, int] = {}
        for page in pages:
            key = str(int(page.get("rotation_degrees") or 0))
            rows[key] = rows.get(key, 0) + 1
        return dict(sorted(rows.items()))
