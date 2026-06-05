from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings

from .models import ConversionDiffCandidate, ConversionPreflightResult, DocxEvidenceUnit, FocusedCandidateReview, PageTextCoverageProfile


class ModelCandidateRecallBuilder:
    """Lift generic page-level text gaps into model-gated focused candidates.

    Page coverage profiles are intentionally broad: they catch "something on
    this page did not line up", but they are too noisy to become comments
    directly. This builder only turns stable, same-page near matches into
    standard diff/focused-review records so Qwen-VL can verify them against the
    rendered PDF page before any user-facing comment is written.
    """

    TARGET_STATUSES = {
        "page_text_coverage_gap",
        "page_text_coverage_uncertain",
        "table_page_text_coverage_uncertain",
    }

    def __init__(
        self,
        *,
        work_dir: Path,
        max_candidates: Optional[int] = None,
        max_per_page: Optional[int] = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.max_candidates = max(
            0,
            int(
                max_candidates
                if max_candidates is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_MODEL_RECALL_MAX_CANDIDATES", 9999)
                or 9999
            ),
        )
        self.max_per_page = max(
            1,
            int(
                max_per_page
                if max_per_page is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_MODEL_RECALL_MAX_PER_PAGE", 9999)
                or 9999
            ),
        )

    def apply(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        rendered_pages: Dict[int, Path],
    ) -> Dict[str, Any]:
        if self.max_candidates <= 0:
            return {"added_diff_count": 0, "added_focused_count": 0, "reason": "disabled"}
        docx_by_page = self._docx_by_page(preflight_result.docx_units)
        pdf_segments_by_page = self._pdf_segments_by_page(preflight_result=preflight_result)
        existing = self._existing_signatures(preflight_result)
        added = 0
        per_page: Dict[int, int] = {}
        next_index = len(preflight_result.diff_candidates) + 1
        for profile in self._ordered_profiles(preflight_result.page_text_coverage_profiles):
            if added >= self.max_candidates:
                break
            page_no = int(profile.page_no or 0)
            if page_no <= 0 or per_page.get(page_no, 0) >= self.max_per_page:
                continue
            page_image = self._relative_page_image(rendered_pages.get(page_no))
            if not page_image:
                continue
            candidates = self._page_text_gap_candidates(profile=profile)
            candidates.extend(
                self._unit_alignment_candidates(
                    profile=profile,
                    docx_units=docx_by_page.get(page_no, []),
                    pdf_segments=pdf_segments_by_page.get(page_no, []),
                )
            )
            candidates = self._dedupe_candidates(candidates)
            for candidate in candidates:
                if added >= self.max_candidates or per_page.get(page_no, 0) >= self.max_per_page:
                    break
                unit = self._candidate_docx_unit(candidate=candidate, page_no=page_no, docx_by_page=docx_by_page)
                if unit is None:
                    continue
                signature = (page_no, unit.unit_id, self._compact(candidate.get("docx_text")), self._compact(candidate.get("pdf_text")))
                if signature in existing:
                    continue
                existing.add(signature)
                diff_id = f"model_recall_{next_index:04d}"
                next_index += 1
                confidence = max(0.5, min(0.68, float(candidate.get("confidence") or 0.0)))
                preflight_result.diff_candidates.append(
                    ConversionDiffCandidate(
                        diff_id=diff_id,
                        category="text_substitution",
                        risk="high" if profile.risk_level == "high" else "medium",
                        docx_unit_id=unit.unit_id,
                        pdf_page_no=page_no,
                        docx_estimated_page_no=unit.estimated_page_no or page_no,
                        pdf_text=str(candidate.get("pdf_text") or ""),
                        docx_text=str(candidate.get("docx_text") or unit.text or ""),
                        pdf_value=str(candidate.get("pdf_text") or ""),
                        docx_value=str(candidate.get("docx_text") or unit.text or ""),
                        alignment_confidence=0.0,
                        confidence=confidence,
                        reason=str(candidate.get("reason") or "页级覆盖画像召回同页 DOCX/PDF 近似缺口，需模型对照 PDF 页面确认。"),
                        flags=[
                            "model_recall_candidate",
                            "page_text_gap_candidate",
                            "requires_qwen_vl_confirmation",
                            f"page_text_status={profile.status}",
                            f"page_text_risk={profile.risk_level}",
                            *list(candidate.get("flags") or []),
                        ],
                    )
                )
                preflight_result.focused_reviews.append(
                    FocusedCandidateReview(
                        review_id=f"model_recall_focused_{next_index - 1:04d}",
                        diff_id=diff_id,
                        category="text_substitution",
                        page_no=page_no,
                        status="needs_visual_gate",
                        decision="needs_visual_review",
                        confidence=confidence,
                        reason="页级覆盖缺口已定位到 DOCX 单元；必须由 Qwen-VL 对照 PDF 页面确认后才允许写批注。",
                        next_route="needs_qwen_vl",
                        crop_path=page_image,
                        crop_ocr_text=str(candidate.get("pdf_text") or ""),
                        crop_ocr_quality=str(candidate.get("quality") or "page_text_gap"),
                        pdf_text=str(candidate.get("pdf_text") or ""),
                        docx_text=str(candidate.get("docx_text") or unit.text or ""),
                        flags=[
                            "model_recall_candidate",
                            "page_text_gap_candidate",
                            "full_page_visual_context",
                            "not_auto_commented",
                            *list(candidate.get("flags") or []),
                        ],
                        visual_text={
                            "support": "candidate_requires_visual_confirmation",
                            "stable": False,
                            "reason": "PDF 文本来自页级覆盖缺口样例，不能直接确认，只能作为 Qwen-VL 定位线索。",
                            "page_text_similarity": candidate.get("similarity", 0.0),
                            "common_span": candidate.get("common_span", 0),
                        },
                    )
                )
                added += 1
                per_page[page_no] = per_page.get(page_no, 0) + 1
        return {
            "enabled": True,
            "added_diff_count": added,
            "added_focused_count": added,
            "per_page": dict(sorted(per_page.items())),
        }

    def _candidate_docx_unit(
        self,
        *,
        candidate: Dict[str, Any],
        page_no: int,
        docx_by_page: Dict[int, List[DocxEvidenceUnit]],
    ) -> Optional[DocxEvidenceUnit]:
        unit_id = str(candidate.get("docx_unit_id") or "")
        if unit_id:
            for unit in docx_by_page.get(page_no, []):
                if unit.unit_id == unit_id:
                    return unit
            return None
        return self._locate_docx_unit(
            page_no=page_no,
            docx_text=str(candidate.get("docx_text") or ""),
            docx_by_page=docx_by_page,
        )

    def _ordered_profiles(self, profiles: Sequence[PageTextCoverageProfile]) -> List[PageTextCoverageProfile]:
        rows = [
            profile
            for profile in profiles
            if profile.risk_level in {"high", "medium"} and profile.status in self.TARGET_STATUSES
        ]
        rows.sort(key=lambda item: (0 if item.risk_level == "high" else 1, int(item.page_no or 9999)))
        return rows

    def _docx_by_page(self, units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[DocxEvidenceUnit]]:
        rows: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0 or unit.container_type == "table_cell":
                continue
            rows.setdefault(page_no, []).append(unit)
        for page_units in rows.values():
            page_units.sort(key=lambda item: (int(item.order_index or 0), item.unit_id))
        return rows

    def _existing_signatures(self, preflight_result: ConversionPreflightResult) -> set[tuple[int, str, str, str]]:
        signatures: set[tuple[int, str, str, str]] = set()
        for diff in preflight_result.diff_candidates:
            if not diff.docx_unit_id:
                continue
            signatures.add(
                (
                    int(diff.pdf_page_no or diff.docx_estimated_page_no or 0),
                    diff.docx_unit_id,
                    self._compact(diff.docx_text or diff.docx_value),
                    self._compact(diff.pdf_text or diff.pdf_value),
                )
            )
        return signatures

    def _relative_page_image(self, image_path: Optional[Path]) -> str:
        if image_path is None or not Path(image_path).exists():
            return ""
        try:
            return Path(image_path).resolve().relative_to(self.work_dir.resolve()).as_posix()
        except Exception:
            return str(image_path)

    def _page_text_gap_candidates(self, *, profile: PageTextCoverageProfile) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for docx_text in profile.docx_gap_samples:
            for pdf_text in profile.pdf_gap_samples:
                candidate = self._page_text_gap_pair(docx_text=docx_text, pdf_text=pdf_text)
                if candidate:
                    rows.append(candidate)
        rows = self._drop_unstable_candidates(rows)
        rows.sort(
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                -len(self._compact(item.get("docx_text") or "")),
                str(item.get("docx_text") or ""),
            )
        )
        deduped: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (self._compact(row.get("docx_text") or ""), self._compact(row.get("pdf_text") or ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= self.max_per_page:
                break
        return deduped

    def _unit_alignment_candidates(
        self,
        *,
        profile: PageTextCoverageProfile,
        docx_units: Sequence[DocxEvidenceUnit],
        pdf_segments: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not docx_units or not pdf_segments:
            return rows
        page_pdf_compact = "".join(self._compact(item.get("text") or "") for item in pdf_segments)
        for unit in docx_units:
            old_text = self._clean_text(unit.text)
            old_key = self._compact(old_text)
            if not (4 <= len(old_key) <= 120):
                continue
            if old_key and old_key in page_pdf_compact:
                continue
            if self._noise(old_text):
                continue
            best = self._best_segment_candidate(docx_text=old_text, pdf_segments=pdf_segments)
            if not best:
                continue
            best["docx_unit_id"] = unit.unit_id
            best["flags"] = [
                "unit_text_alignment_recall",
                f"unit_alignment_source={best.get('source') or 'pdf_ocr'}",
            ]
            best["reason"] = (
                "同页 DOCX 文本单元与 PDF OCR 片段高度近似但不一致，疑似 WPS 转换文字错误；"
                "必须由 Qwen-VL 对照原 PDF 页面确认。"
            )
            rows.append(best)
        rows.sort(
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                int(item.get("page_order") or 999999),
                str(item.get("docx_unit_id") or ""),
            )
        )
        return rows[: self.max_per_page]

    def _best_segment_candidate(self, *, docx_text: str, pdf_segments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        old_text = self._clean_text(docx_text)
        old_key = self._compact(old_text)
        best: Dict[str, Any] = {}
        for segment in pdf_segments:
            new_text = self._clean_text(segment.get("text") or "")
            candidate = self._unit_alignment_pair(docx_text=old_text, pdf_text=new_text)
            if not candidate:
                continue
            candidate["source"] = segment.get("source") or ""
            candidate["page_order"] = segment.get("order_index") or 999999
            if not best or float(candidate.get("confidence") or 0.0) > float(best.get("confidence") or 0.0):
                best = candidate
        if best and self._compact(best.get("pdf_text") or "") == old_key:
            return {}
        return best

    def _unit_alignment_pair(self, *, docx_text: Any, pdf_text: Any) -> Dict[str, Any]:
        old_text = self._clean_text(docx_text)
        new_text = self._clean_text(pdf_text)
        old_key = self._compact(old_text)
        new_key = self._compact(new_text)
        if not old_key or not new_key or old_key == new_key:
            return {}
        if not (4 <= len(old_key) <= 120 and 4 <= len(new_key) <= 180):
            return {}
        old_semantic = self._semantic_compact(old_key)
        new_semantic = self._semantic_compact(new_key)
        if old_semantic == new_semantic:
            return {}
        if self._semantic_containment(old_semantic, new_semantic):
            return {}
        # A shorter OCR segment contained in the DOCX text is usually OCR
        # truncation, not a reliable conversion-error candidate.
        if old_key in new_key or new_key in old_key:
            return {}
        if self._noise(old_text) or self._noise(new_text):
            return {}
        length_ratio = min(len(old_key), len(new_key)) / max(len(old_key), len(new_key))
        if length_ratio < 0.55:
            return {}
        if self._weak_numeric_fragment(old_key=old_key, new_key=new_key):
            return {}
        if self._date_fragment(old_text=old_text, new_text=new_text):
            return {}
        ratio = SequenceMatcher(None, old_key, new_key).ratio()
        common = self._longest_common_substring_length(old_key, new_key)
        threshold = 0.86 if min(len(old_key), len(new_key)) <= 7 else 0.78
        if any(ch.isdigit() for ch in old_key + new_key):
            threshold = max(threshold, 0.82)
        if ratio < threshold:
            return {}
        required_common = min(6, max(4, min(len(old_key), len(new_key)) - 1))
        if common < required_common and ratio < threshold + 0.01:
            return {}
        return {
            "docx_text": old_text,
            "pdf_text": new_text,
            "confidence": max(0.52, min(0.7, 0.42 + ratio * 0.28)),
            "similarity": round(ratio, 4),
            "common_span": common,
            "quality": "unit_text_alignment",
        }

    def _dedupe_candidates(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = self._drop_unstable_candidates(list(rows))
        rows.sort(
            key=lambda item: (
                0 if "unit_text_alignment_recall" in set(item.get("flags") or []) else 1,
                -float(item.get("confidence") or 0.0),
                -len(self._compact(item.get("docx_text") or "")),
                str(item.get("docx_unit_id") or ""),
            )
        )
        deduped: List[Dict[str, Any]] = []
        seen_units: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for row in rows:
            unit_id = str(row.get("docx_unit_id") or "")
            pair = (self._compact(row.get("docx_text") or ""), self._compact(row.get("pdf_text") or ""))
            if unit_id and unit_id in seen_units:
                continue
            if pair in seen_pairs:
                continue
            if unit_id:
                seen_units.add(unit_id)
            seen_pairs.add(pair)
            deduped.append(row)
        return deduped

    def _page_text_gap_pair(self, *, docx_text: Any, pdf_text: Any) -> Dict[str, Any]:
        old_text = self._clean_text(docx_text)
        new_text = self._clean_text(pdf_text)
        old_key = self._compact(old_text)
        new_key = self._compact(new_text)
        if not old_key or not new_key or old_key == new_key:
            return {}
        if not (4 <= len(old_key) <= 48 and 4 <= len(new_key) <= 48):
            return {}
        if self._noise(old_text) or self._noise(new_text):
            return {}
        if self._weak_numeric_fragment(old_key=old_key, new_key=new_key):
            return {}
        if self._date_fragment(old_text=old_text, new_text=new_text):
            return {}
        if min(len(old_key), len(new_key)) / max(len(old_key), len(new_key)) < 0.72:
            return {}
        if not (re.search(r"[\u4e00-\u9fff]", old_key + new_key) or re.search(r"\d", old_key + new_key)):
            return {}
        ratio = SequenceMatcher(None, old_key, new_key).ratio()
        common = self._longest_common_substring_length(old_key, new_key)
        digit_old = re.sub(r"\D+", "", old_key)
        digit_new = re.sub(r"\D+", "", new_key)
        same_digits = bool(len(digit_old) >= 3 and digit_old == digit_new)
        threshold = 0.52 if same_digits else (0.68 if any(ch.isdigit() for ch in old_key + new_key) else 0.72)
        if ratio < threshold and common < min(6, max(4, min(len(old_key), len(new_key)) - 1)):
            return {}
        return {
            "docx_text": old_text,
            "pdf_text": new_text,
            "confidence": max(0.5, min(0.66, 0.38 + ratio * 0.24 + (0.06 if same_digits else 0.0))),
            "similarity": round(ratio, 4),
            "common_span": common,
            "reason": "页级覆盖画像发现同页 DOCX 缺口与 PDF OCR 缺口高度相似，疑似转换文字需要视觉模型确认。",
        }

    def _drop_unstable_candidates(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_old: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("docx_unit_id") or "") or self._compact(row.get("docx_text") or "")
            by_old.setdefault(key, []).append(row)
        stable: List[Dict[str, Any]] = []
        for old_key, group in by_old.items():
            new_keys = {self._compact(item.get("pdf_text") or "") for item in group}
            if len(new_keys) <= 1:
                stable.extend(group)
                continue
            # Model-gated recall should be high precision. If one DOCX span
            # maps to multiple different PDF OCR spans, treat the OCR side as
            # unstable and keep it out of the model queue. The broader report
            # still records page-level gaps separately.
            continue
        return stable

    def _pdf_segments_by_page(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, List[Dict[str, Any]]]:
        rows: Dict[int, List[Dict[str, Any]]] = {}
        seen: set[tuple[int, str]] = set()

        def append(page_no: int, text: Any, *, source: str, order_index: int = 999999) -> None:
            value = self._clean_text(text)
            key = self._compact(value)
            if page_no <= 0 or len(key) < 4:
                return
            dedupe_key = (page_no, key[:160])
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            rows.setdefault(page_no, []).append({"text": value, "source": source, "order_index": order_index})

        for page in preflight_result.anchor_ocr.get("pages") or []:
            if not isinstance(page, dict):
                continue
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            for line_index, line in enumerate(page.get("lines") or []):
                text = line.get("text") if isinstance(line, dict) else line
                append(page_no, text, source="anchor_ocr_line", order_index=line_index)
            for part in self._split_pdf_text(page.get("text")):
                append(page_no, part, source="anchor_ocr_page", order_index=500000)
        for unit in preflight_result.pdf_units:
            unit_type = str(getattr(unit, "unit_type", "") or "")
            if unit_type not in {"anchor_ocr_page", "anchor_ocr_line", "native_text_block"}:
                continue
            page_no = int(getattr(unit, "page_no", 0) or 0)
            source = str(getattr(unit, "source", "") or unit_type)
            for part in self._split_pdf_text(getattr(unit, "text", "")):
                append(page_no, part, source=source, order_index=int(getattr(unit, "order_index", 999999) or 999999))
        for backfill in preflight_result.content_coverage_backfills:
            if not backfill.available:
                continue
            page_no = int(backfill.page_no or 0)
            source = f"{backfill.method or 'ocr'}_backfill"
            for part in self._split_pdf_text(backfill.extracted_text or backfill.normalized_text):
                append(page_no, part, source=source, order_index=800000)
        for page_no, page_rows in rows.items():
            rows[page_no] = sorted(page_rows, key=lambda item: int(item.get("order_index") or 999999))
        return rows

    def _split_pdf_text(self, value: Any) -> List[str]:
        text = self._clean_text(value)
        if not text:
            return []
        parts: List[str] = []
        for segment in re.split(r"[\n\r。；;]+", text):
            segment = self._clean_text(segment)
            key = self._compact(segment)
            if 4 <= len(key) <= 180:
                parts.append(segment)
            elif len(key) > 180:
                for start in range(0, len(key), 60):
                    chunk = key[start : start + 100]
                    if len(chunk) >= 8:
                        parts.append(chunk)
        return parts

    def _locate_docx_unit(
        self,
        *,
        page_no: int,
        docx_text: str,
        docx_by_page: Dict[int, List[DocxEvidenceUnit]],
    ) -> Optional[DocxEvidenceUnit]:
        needle = self._compact(docx_text)
        if not needle:
            return None
        exact: List[DocxEvidenceUnit] = []
        fuzzy: List[tuple[float, DocxEvidenceUnit]] = []
        for unit in docx_by_page.get(page_no, []):
            key = self._compact(unit.text)
            if not key:
                continue
            if needle in key or key in needle:
                exact.append(unit)
                continue
            if len(needle) >= 6:
                score = SequenceMatcher(None, needle, key[: max(len(needle) + 8, 16)]).ratio()
                if score >= 0.72:
                    fuzzy.append((score, unit))
        if exact:
            exact.sort(key=lambda item: (len(self._compact(item.text)), int(item.order_index or 0)))
            return exact[0]
        if fuzzy:
            fuzzy.sort(key=lambda item: (-item[0], int(item[1].order_index or 0)))
            return fuzzy[0][1]
        return None

    def _noise(self, text: str) -> bool:
        value = self._clean_text(text)
        if any(marker in value for marker in ("?", "？", "_", "*", "%", "[", "]", "［", "］", "无法", "看不清")):
            return True
        compact = self._compact(value)
        if re.search(r"[\u4e00-\u9fff][A-Za-z]|[A-Za-z][\u4e00-\u9fff]", compact):
            return True
        if len(compact) >= 18 and not re.search(r"[\u4e00-\u9fff]", compact):
            return True
        return bool(re.fullmatch(r"\d{8,}", compact))

    def _weak_numeric_fragment(self, *, old_key: str, new_key: str) -> bool:
        old_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", old_key))
        new_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", new_key))
        old_digits = re.sub(r"\D+", "", old_key)
        new_digits = re.sub(r"\D+", "", new_key)
        if not old_has_cjk and not new_has_cjk:
            return True
        if old_digits and new_digits and old_digits != new_digits:
            old_digit_ratio = len(old_digits) / max(1, len(old_key))
            new_digit_ratio = len(new_digits) / max(1, len(new_key))
            if old_digit_ratio >= 0.45 or new_digit_ratio >= 0.45:
                return True
        return bool(old_has_cjk != new_has_cjk and (old_digits or new_digits))

    def _date_fragment(self, *, old_text: str, new_text: str) -> bool:
        full_date = r"\d{4}年\d{1,2}月\d{1,2}日"
        old_has_full_date = bool(re.search(full_date, old_text))
        new_has_full_date = bool(re.search(full_date, new_text))
        if old_has_full_date != new_has_full_date:
            return True
        if re.search(r"\d{4}\s*年|\d{4}\s*月", old_text + new_text) and not (old_has_full_date and new_has_full_date):
            return re.sub(r"\D+", "", old_text) != re.sub(r"\D+", "", new_text)
        return False

    def _longest_common_substring_length(self, left: str, right: str) -> int:
        if not left or not right:
            return 0
        previous = [0] * (len(right) + 1)
        best = 0
        for left_char in left:
            current = [0]
            for index, right_char in enumerate(right, start=1):
                value = previous[index - 1] + 1 if left_char == right_char else 0
                current.append(value)
                if value > best:
                    best = value
            previous = current
        return best

    def _clean_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _compact(self, value: Any) -> str:
        return "".join(str(value or "").split()).lower()

    def _semantic_compact(self, value: Any) -> str:
        text = self._compact(value)
        text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
        return text.lower()

    def _semantic_containment(self, left: str, right: str) -> bool:
        if len(left) < 4 or len(right) < 4:
            return False
        shorter = min(len(left), len(right))
        longer = max(len(left), len(right))
        if shorter / max(1, longer) < 0.45:
            return False
        return bool(left in right or right in left)
