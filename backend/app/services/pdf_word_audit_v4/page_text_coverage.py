from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

from app.core.config import settings

from .common import normalize_text
from .models import ConversionPreflightResult, DocxEvidenceUnit, PageTextCoverageProfile


class PageTextCoverageProfileBuilder:
    """Build page-level text coverage profiles for image/scan PDF audits.

    Full-content coverage is unit based. On photographed or table-heavy pages,
    WPS often splits one visual line into many DOCX fragments, so unit-level
    matching alone leaves a large unresolved count. This profile adds a page
    view: how much DOCX text is supported by PDF OCR, how much PDF OCR appears
    in DOCX, and which generic tokens remain uncovered.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_gap_samples: int | None = None,
    ) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_COVERAGE_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_gap_samples = max(
            1,
            int(
                max_gap_samples
                if max_gap_samples is not None
                else getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_COVERAGE_MAX_GAP_SAMPLES", 24)
                or 12
            ),
        )

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[PageTextCoverageProfile]:
        if not self.enabled:
            return []
        docx_by_page = self._docx_by_page(preflight_result.docx_units)
        pdf_texts_by_page = self._pdf_texts_by_page(preflight_result=preflight_result)
        rows: List[PageTextCoverageProfile] = []
        for page_no in range(1, int(preflight_result.pdf_page_count or 0) + 1):
            profile = self._build_page(
                page_no=page_no,
                docx_units=docx_by_page.get(page_no, []),
                pdf_texts=pdf_texts_by_page.get(page_no, []),
                page_profile=preflight_result.page_profiles.get(str(page_no)) or {},
            )
            rows.append(profile)
        return rows

    def _build_page(
        self,
        *,
        page_no: int,
        docx_units: Sequence[DocxEvidenceUnit],
        pdf_texts: Sequence[str],
        page_profile: Dict[str, Any],
    ) -> PageTextCoverageProfile:
        docx_parts = [str(unit.text or "") for unit in docx_units if str(unit.text or "").strip()]
        pdf_parts = [str(text or "") for text in pdf_texts if str(text or "").strip()]
        docx_text = "\n".join(docx_parts)
        pdf_text = "\n".join(pdf_parts)
        docx_compact = self._compact(docx_text)
        pdf_compact = self._compact(pdf_text)
        docx_tokens = self._tokens(docx_text)
        pdf_tokens = self._tokens(pdf_text)
        pdf_token_set = {token["key"] for token in pdf_tokens}
        docx_token_set = {token["key"] for token in docx_tokens}
        docx_covered = [token for token in docx_tokens if token["key"] in pdf_token_set or token["key"] in pdf_compact]
        pdf_covered = [token for token in pdf_tokens if token["key"] in docx_token_set or token["key"] in docx_compact]
        docx_ratio = len(docx_covered) / max(1, len(docx_tokens))
        pdf_ratio = len(pdf_covered) / max(1, len(pdf_tokens))
        similarity = self._similarity(docx_compact, pdf_compact)
        table_units = sum(1 for unit in docx_units if unit.container_type == "table_cell")
        table_ratio = table_units / max(1, len(docx_units))
        status, risk_level, reason = self._verdict(
            page_profile=page_profile,
            docx_ratio=docx_ratio,
            pdf_ratio=pdf_ratio,
            similarity=similarity,
            docx_token_count=len(docx_tokens),
            pdf_token_count=len(pdf_tokens),
            table_ratio=table_ratio,
        )
        flags = self._flags(page_profile=page_profile, table_ratio=table_ratio)
        return PageTextCoverageProfile(
            page_no=page_no,
            status=status,
            risk_level=risk_level,
            docx_text_chars=len(docx_compact),
            pdf_text_chars=len(pdf_compact),
            docx_unit_count=len(docx_units),
            pdf_text_source_count=len(pdf_parts),
            docx_token_count=len(docx_tokens),
            pdf_token_count=len(pdf_tokens),
            docx_token_coverage_ratio=docx_ratio,
            pdf_token_coverage_ratio=pdf_ratio,
            page_text_similarity=similarity,
            table_cell_ratio=table_ratio,
            reason=reason,
            docx_gap_samples=self._gap_samples(tokens=docx_tokens, covered_keys={token["key"] for token in docx_covered}),
            pdf_gap_samples=self._gap_samples(tokens=pdf_tokens, covered_keys={token["key"] for token in pdf_covered}),
            flags=flags,
        )

    def _docx_by_page(self, units: Sequence[DocxEvidenceUnit]) -> Dict[int, List[DocxEvidenceUnit]]:
        rows: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in units:
            page_no = int(unit.estimated_page_no or 0)
            text = str(unit.text or "").strip()
            if page_no <= 0 or not text:
                continue
            rows.setdefault(page_no, []).append(unit)
        for page_no, page_units in rows.items():
            rows[page_no] = sorted(page_units, key=lambda item: int(item.order_index or 0))
        return rows

    def _pdf_texts_by_page(self, *, preflight_result: ConversionPreflightResult) -> Dict[int, List[str]]:
        rows: Dict[int, List[str]] = {}
        seen: set[tuple[int, str]] = set()

        def append(page_no: int, text: Any) -> None:
            value = str(text or "").strip()
            if page_no <= 0 or not value:
                return
            key = (page_no, self._compact(value)[:300])
            if key in seen:
                return
            seen.add(key)
            rows.setdefault(page_no, []).append(value)

        for page in preflight_result.anchor_ocr.get("pages") or []:
            try:
                page_no = int(page.get("page") or page.get("page_no") or 0)
            except Exception:
                page_no = 0
            append(page_no, page.get("text"))
            for line in page.get("lines") or []:
                if isinstance(line, dict):
                    append(page_no, line.get("text"))
                else:
                    append(page_no, line)
        for unit in preflight_result.pdf_units:
            unit_type = str(getattr(unit, "unit_type", "") or "")
            if unit_type not in {"anchor_ocr_page", "anchor_ocr_line", "native_text_block"}:
                continue
            append(int(getattr(unit, "page_no", 0) or 0), getattr(unit, "text", ""))
        for backfill in preflight_result.content_coverage_backfills:
            if not backfill.available:
                continue
            append(int(backfill.page_no or 0), backfill.extracted_text or backfill.normalized_text)
        return rows

    def _tokens(self, text: str) -> List[Dict[str, str]]:
        compact = self._compact(text)
        if not compact:
            return []
        tokens: List[Dict[str, str]] = []
        seen: set[str] = set()

        def add(raw: str, kind: str) -> None:
            key = self._compact(raw)
            if len(key) < 3 or key in seen:
                return
            seen.add(key)
            tokens.append({"key": key, "text": raw, "kind": kind})

        for segment in re.split(r"[\n\r\t ,，。；;：:、()（）]+", str(text or "")):
            key = self._compact(segment)
            if 4 <= len(key) <= 40:
                add(segment, "phrase")
            elif len(key) > 40:
                for start in range(0, len(key), 18):
                    chunk = key[start : start + 18]
                    if len(chunk) >= 6:
                        add(chunk, "phrase")
        for match in re.finditer(r"[A-Za-z0-9]{3,}|[￥¥]?\d[\d.,:：/-]{2,}", compact):
            add(match.group(0), "number")
        chinese = re.sub(r"[^\u4e00-\u9fff]+", "", compact)
        for size in (8, 6, 4):
            if len(chinese) < size:
                continue
            step = max(1, size // 2)
            for start in range(0, len(chinese) - size + 1, step):
                add(chinese[start : start + size], "cjk")
        return tokens

    def _gap_samples(self, *, tokens: Sequence[Dict[str, str]], covered_keys: set[str]) -> List[str]:
        samples: List[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = token["key"]
            if key in covered_keys or key in seen:
                continue
            seen.add(key)
            samples.append(token["text"][:40])
            if len(samples) >= self.max_gap_samples:
                break
        return samples

    def _verdict(
        self,
        *,
        page_profile: Dict[str, Any],
        docx_ratio: float,
        pdf_ratio: float,
        similarity: float,
        docx_token_count: int,
        pdf_token_count: int,
        table_ratio: float,
    ) -> tuple[str, str, str]:
        labels = set(page_profile.get("labels") or [])
        canonical = str(page_profile.get("audit_canonical_page_type") or "")
        image_like = bool(
            canonical in {"scan_text_page", "table_image_page", "mixed_layout_page", "low_confidence_page"}
            or labels & {"scan_like", "image_text_heavy", "table_heavy"}
            or page_profile.get("needs_ocr")
        )
        if docx_token_count == 0 and pdf_token_count == 0:
            return "no_text_evidence", "low", "本页没有足够 DOCX/PDF OCR 文本 token 可做页级覆盖画像。"
        if docx_token_count == 0 or pdf_token_count == 0:
            risk = "high" if image_like else "medium"
            return "missing_side_text", risk, "DOCX 或 PDF OCR 一侧文本不足，无法完成页级覆盖验证。"
        if docx_ratio < 0.55 or pdf_ratio < 0.45:
            return "page_text_coverage_gap", "high", "页级 DOCX/PDF OCR token 覆盖率过低，存在漏识别、错识别或映射错位风险。"
        if image_like and (docx_ratio < 0.72 or pdf_ratio < 0.62 or similarity < 0.62):
            return "page_text_coverage_uncertain", "medium", "图片型/扫描型页面的页级覆盖未达到稳定放行阈值，需要继续复核。"
        if table_ratio >= 0.5 and (docx_ratio < 0.82 or pdf_ratio < 0.72):
            return "table_page_text_coverage_uncertain", "medium", "表格页文本 token 有覆盖缺口，需结合单元格级复核判断。"
        return "page_text_coverage_supported", "low", "页级 DOCX/PDF OCR token 覆盖基本稳定。"

    def _flags(self, *, page_profile: Dict[str, Any], table_ratio: float) -> List[str]:
        rows: List[str] = []
        for key in ("audit_canonical_page_type", "pdf_source_type", "recognition_strategy", "recognition_risk"):
            value = str(page_profile.get(key) or "")
            if value:
                rows.append(f"{key}={value}")
        for label in sorted(set(page_profile.get("labels") or []))[:8]:
            rows.append(f"label={label}")
        if table_ratio >= 0.5:
            rows.append("table_heavy_docx_units")
        return rows

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if len(left) > 4000:
            left = left[:4000]
        if len(right) > 4000:
            right = right[:4000]
        return SequenceMatcher(None, left, right).ratio()

    def _compact(self, text: Any) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        return re.sub(r"[^\w\u4e00-\u9fff￥¥]+", "", value)
