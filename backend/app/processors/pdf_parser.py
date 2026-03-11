"""PDF parser with native text extraction and OCR fallback for scanned pages."""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
from io import BytesIO
from typing import Any, Dict, List, Optional

import pdfplumber
import pypdfium2 as pdfium

from app.core.config import settings
from app.processors.base_parser import BaseParser
from app.services.ollama_service import OllamaLLMService

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    async def parse(self, file_path: str, **kwargs: Any) -> Dict:
        logger.info("Start parsing PDF: %s", file_path)
        try:
            return await asyncio.to_thread(
                self._parse_sync,
                file_path,
                bool(kwargs.get("use_llm", False)),
                kwargs.get("llm_model"),
            )
        except Exception as exc:
            logger.error("PDF parsing failed: %s", exc)
            raise

    def _parse_sync(
        self,
        file_path: str,
        use_llm: bool = False,
        llm_model: Optional[str] = None,
    ) -> Dict:
        ocr_service = self._build_ocr_service(use_llm=use_llm, llm_model=llm_model)
        pdfium_document = self._open_pdfium_document(file_path, ocr_service)
        page_entries: List[Dict[str, Any]] = []
        parser_warnings: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_index, page in enumerate(pdf.pages):
                native_text = self._normalize_page_text(page.extract_text() or "")
                native_blocks = self._extract_native_blocks(page, native_text)
                page_entry: Dict[str, Any] = {
                    "page_number": page_index + 1,
                    "width": float(getattr(page, "width", 595.0) or 595.0),
                    "height": float(getattr(page, "height", 842.0) or 842.0),
                    "text": native_text,
                    "source": "native",
                    "char_count": self._count_readable_chars(native_text),
                    "blocks": native_blocks,
                }

                needs_ocr = self._should_run_ocr(native_text)
                if needs_ocr:
                    page_entry["source"] = "native_low_text" if native_text else "empty"

                    if ocr_service is not None and pdfium_document is not None:
                        try:
                            image_bytes = self._render_page_image(pdfium_document, page_index)
                            ocr_result = ocr_service.extract_document_text_from_image(
                                image_bytes,
                                page_number=page_index + 1,
                                total_pages=total_pages,
                            )
                            ocr_text = self._normalize_page_text(ocr_result.get("text", ""))
                            if ocr_text:
                                page_entry["text"] = ocr_text
                                page_entry["source"] = "ocr"
                                page_entry["char_count"] = self._count_readable_chars(ocr_text)
                                page_entry["ocr_quality"] = ocr_result.get("quality")
                                page_entry["ocr_layout"] = ocr_result.get("layout")
                                page_entry["blocks"] = self._normalize_blocks(
                                    ocr_result.get("blocks"),
                                    fallback_text=ocr_text,
                                )
                                if ocr_result.get("warnings"):
                                    page_entry["warnings"] = list(ocr_result["warnings"])
                            elif native_text:
                                page_entry["source"] = "native_low_text"
                            else:
                                page_entry["source"] = "ocr_empty"
                                page_entry["blocks"] = []
                        except Exception as exc:
                            logger.warning(
                                "PDF OCR failed on page %s: %s",
                                page_index + 1,
                                exc,
                            )
                            page_entry["ocr_error"] = str(exc)
                            if native_text:
                                page_entry["source"] = "native_ocr_failed"
                            else:
                                page_entry["source"] = "ocr_failed"
                                page_entry["blocks"] = []
                    else:
                        parser_warnings.append("ocr_unavailable_for_scan_pages")

                page_entries.append(page_entry)

        if pdfium_document is not None:
            pdfium_document.close()

        text = self._join_page_texts(page_entries)
        native_text_pages = sum(
            1 for page in page_entries if str(page.get("source", "")).startswith("native")
        )
        ocr_pages = sum(1 for page in page_entries if page.get("source") == "ocr")
        empty_pages = sum(1 for page in page_entries if not str(page.get("text", "")).strip())

        metadata = {
            "pages": len(page_entries),
            "format": "pdf",
            "file_path": file_path,
            "native_text_pages": native_text_pages,
            "ocr_pages": ocr_pages,
            "empty_pages": empty_pages,
            "ocr_enabled": bool(ocr_service),
            "ocr_model": ocr_service.model if ocr_service is not None else None,
            "normalized_export": "docx" if ocr_pages > 0 else None,
        }
        if parser_warnings:
            metadata["warnings"] = sorted(set(parser_warnings))

        logger.info(
            "PDF parsing complete, pages=%s, text_length=%s, ocr_pages=%s",
            metadata["pages"],
            len(text),
            ocr_pages,
        )
        return {
            "text": text,
            "metadata": metadata,
            "structure": {
                "pages": page_entries,
            },
        }

    def _build_ocr_service(
        self,
        *,
        use_llm: bool,
        llm_model: Optional[str],
    ) -> Optional[OllamaLLMService]:
        if not use_llm or not settings.PDF_OCR_ENABLED:
            return None

        service = OllamaLLMService(
            base_url=settings.OLLAMA_BASE_URL,
            model=llm_model or settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            num_ctx=settings.OLLAMA_NUM_CTX,
        )
        if not service.available:
            logger.warning("Skip PDF OCR because Ollama is unavailable.")
            return None
        return service

    def _open_pdfium_document(
        self,
        file_path: str,
        ocr_service: Optional[OllamaLLMService],
    ) -> Optional[pdfium.PdfDocument]:
        if ocr_service is None:
            return None

        try:
            return pdfium.PdfDocument(file_path)
        except Exception as exc:
            logger.warning("Unable to open PDFium document for OCR fallback: %s", exc)
            return None

    def _should_run_ocr(self, native_text: str) -> bool:
        compact = re.sub(r"\s+", "", native_text)
        if not compact:
            return True

        readable_chars = self._count_readable_chars(compact)
        readable_ratio = readable_chars / max(len(compact), 1)
        return (
            len(compact) < settings.PDF_OCR_TEXT_THRESHOLD
            or readable_ratio < settings.PDF_OCR_MIN_READABLE_RATIO
        )

    def _count_readable_chars(self, text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))

    def _normalize_page_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u3000", " ")
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _extract_native_blocks(self, page, native_text: str) -> List[Dict[str, Any]]:
        try:
            raw_lines = page.extract_text_lines(strip=False, return_chars=True, layout=True)
        except TypeError:
            try:
                raw_lines = page.extract_text_lines(strip=False, return_chars=True)
            except Exception:
                raw_lines = []
        except Exception:
            raw_lines = []

        if not isinstance(raw_lines, list) or not raw_lines:
            return self._build_basic_blocks_from_text(native_text)

        line_entries: List[Dict[str, Any]] = []
        page_width = float(getattr(page, "width", 595.0) or 595.0)

        for raw_line in raw_lines:
            if not isinstance(raw_line, dict):
                continue

            text = self._normalize_page_text(str(raw_line.get("text", "")))
            if not text:
                continue

            top = float(raw_line.get("top", 0.0) or 0.0)
            bottom = float(raw_line.get("bottom", top + 12.0) or (top + 12.0))
            x0 = float(raw_line.get("x0", 0.0) or 0.0)
            x1 = float(raw_line.get("x1", page_width) or page_width)
            chars = raw_line.get("chars")
            font_size_hint = None
            if isinstance(chars, list):
                sizes = []
                for char in chars:
                    if not isinstance(char, dict):
                        continue
                    size = char.get("size")
                    try:
                        if size is not None:
                            sizes.append(float(size))
                    except (TypeError, ValueError):
                        continue
                if sizes:
                    font_size_hint = round(statistics.median(sizes), 1)

            line_entries.append(
                {
                    "text": text,
                    "top": top,
                    "bottom": bottom,
                    "x0": x0,
                    "x1": x1,
                    "font_size_hint": font_size_hint,
                }
            )

        if not line_entries:
            return self._build_basic_blocks_from_text(native_text)

        line_entries.sort(key=lambda item: (item["top"], item["x0"]))
        min_x0 = min(item["x0"] for item in line_entries)
        line_heights = [max(1.0, item["bottom"] - item["top"]) for item in line_entries]
        baseline_height = statistics.median(line_heights) if line_heights else 12.0

        blocks: List[Dict[str, Any]] = []
        previous_bottom = None
        for index, item in enumerate(line_entries):
            gap = max(0.0, item["top"] - previous_bottom) if previous_bottom is not None else 0.0
            if gap > max(10.0, baseline_height * 0.85):
                spacer_count = 1 if gap < baseline_height * 1.8 else 2
                blocks.append({"type": "spacer", "count": spacer_count})

            align = self._infer_alignment(item["x0"], item["x1"], page_width, item["text"])
            block_type = self._infer_block_type(item["text"], align, index)
            blocks.append(
                {
                    "type": block_type,
                    "text": item["text"],
                    "align": align,
                    "indent_pt": round(max(0.0, min(item["x0"] - min_x0, 144.0)), 1),
                    "space_before_pt": round(min(max(gap - baseline_height * 0.25, 0.0), 18.0), 1),
                    "font_size_hint": item["font_size_hint"],
                }
            )
            previous_bottom = item["bottom"]

        return self._collapse_spacers(blocks)

    def _normalize_blocks(
        self,
        blocks: Any,
        *,
        fallback_text: str,
    ) -> List[Dict[str, Any]]:
        if isinstance(blocks, list) and blocks:
            normalized = []
            for item in blocks:
                if not isinstance(item, dict):
                    continue
                block = dict(item)
                block_type = str(block.get("type", "line")).strip().lower()
                if block_type == "spacer":
                    block["count"] = max(1, int(block.get("blank_before", block.get("count", 1)) or 1))
                normalized.append(block)
            if normalized:
                return self._collapse_spacers(normalized)
        return self._build_basic_blocks_from_text(fallback_text)

    def _build_basic_blocks_from_text(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        blocks: List[Dict[str, Any]] = []
        pending_table_rows: List[List[str]] = []
        for index, raw_line in enumerate(text.splitlines()):
            stripped_line = raw_line.strip()
            if not stripped_line:
                if pending_table_rows:
                    blocks.append({"type": "table", "rows": pending_table_rows})
                    pending_table_rows = []
                blocks.append({"type": "spacer", "count": 1})
                continue

            cells = self._split_table_cells(raw_line)
            if len(cells) >= 2:
                pending_table_rows.append(cells)
                continue

            if pending_table_rows:
                blocks.append({"type": "table", "rows": pending_table_rows})
                pending_table_rows = []

            align = "center" if index == 0 and self._looks_like_title(stripped_line) else "left"
            blocks.append(
                {
                    "type": "title" if align == "center" else "line",
                    "text": stripped_line,
                    "align": align,
                    "indent_pt": 0.0,
                    "space_before_pt": 0.0,
                }
            )

        if pending_table_rows:
            blocks.append({"type": "table", "rows": pending_table_rows})

        return self._collapse_spacers(blocks)

    def _split_table_cells(self, line: str) -> List[str]:
        if "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
        else:
            parts = [part.strip() for part in re.split(r"\s{3,}", line)]
        return [part for part in parts if part]

    def _infer_alignment(self, x0: float, x1: float, page_width: float, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return "left"

        left_margin = max(0.0, x0)
        right_margin = max(0.0, page_width - x1)
        centered = abs(left_margin - right_margin) <= page_width * 0.08

        if centered and left_margin > page_width * 0.12 and len(clean_text) <= 40:
            return "center"
        if left_margin > page_width * 0.45 and right_margin < page_width * 0.15:
            return "right"
        return "left"

    def _infer_block_type(self, text: str, align: str, line_index: int) -> str:
        stripped = text.strip()
        if align == "center" and self._looks_like_title(stripped):
            return "title"
        if line_index <= 1 and self._looks_like_title(stripped):
            return "title"
        if len(stripped) > 28:
            return "paragraph"
        return "line"

    def _looks_like_title(self, text: str) -> bool:
        if len(text) > 40:
            return False
        if any(token in text for token in ["：", ":", "。", "；", ";", "，", ","]):
            return False
        return True

    def _collapse_spacers(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        collapsed: List[Dict[str, Any]] = []
        for block in blocks:
            block_type = str(block.get("type", "line")).strip().lower()
            if block_type == "spacer":
                count = max(1, int(block.get("count", block.get("blank_before", 1)) or 1))
                if collapsed and str(collapsed[-1].get("type", "")).lower() == "spacer":
                    collapsed[-1]["count"] = min(
                        3,
                        int(collapsed[-1].get("count", 1) or 1) + count,
                    )
                else:
                    collapsed.append({"type": "spacer", "count": min(3, count)})
                continue

            collapsed.append(block)

        return collapsed

    def _render_page_image(
        self,
        pdfium_document: pdfium.PdfDocument,
        page_index: int,
    ) -> bytes:
        page = pdfium_document[page_index]
        bitmap = page.render(
            scale=settings.PDF_OCR_RENDER_SCALE,
            rev_byteorder=True,
        )
        image = bitmap.to_pil().convert("RGB")

        try:
            max_edge = max(image.size)
            if max_edge > settings.PDF_OCR_IMAGE_MAX_EDGE:
                scale = settings.PDF_OCR_IMAGE_MAX_EDGE / max_edge
                resized_size = (
                    max(1, int(image.size[0] * scale)),
                    max(1, int(image.size[1] * scale)),
                )
                resized_image = image.resize(resized_size)
                image.close()
                image = resized_image

            buffer = BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=settings.PDF_OCR_JPEG_QUALITY,
                optimize=True,
            )
            return buffer.getvalue()
        finally:
            image.close()
            bitmap.close()
            page.close()

    def _join_page_texts(self, page_entries: List[Dict[str, Any]]) -> str:
        page_texts = [
            str(page.get("text", "")).strip()
            for page in page_entries
            if str(page.get("text", "")).strip()
        ]
        return "\n\n".join(page_texts)

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".pdf"
