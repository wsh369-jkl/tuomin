"""Export anonymized content while preserving the source format when possible."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Pt

from app.core.config import settings
from app.core.runtime_security import ensure_private_file
from app.engine.desensitization_engine import get_engine
from app.processors.docx_xml_utils import (
    apply_replacements_to_fragments,
    replace_text_in_docx,
)

logger = logging.getLogger(__name__)

DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_MIME_TYPE = "text/plain; charset=utf-8"
DEFAULT_PDF_PAGE_WIDTH_PT = 595.0
DEFAULT_PDF_PAGE_HEIGHT_PT = 842.0


class DocumentExporter:
    """Create downloadable output files for anonymized content."""

    EXPORT_TEXT_PRIORITY = {
        "PERSON": 1,
        "PERSON_NAME": 1,
        "ORGANIZATION": 2,
        "COMPANY_NAME": 2,
        "BANK_NAME": 3,
        "ACCOUNT_NAME": 4,
        "LOCATION": 5,
        "PROJECT": 6,
        "CONTRACT_NO": 7,
        "POSITION": 8,
        "CN_ID_CARD": 9,
        "CN_PHONE": 9,
        "LANDLINE_PHONE": 9,
        "CN_BANK_CARD": 9,
        "CN_CREDIT_CODE": 9,
        "EMAIL_ADDRESS": 9,
        "AMOUNT": 9,
    }

    def export(
        self,
        *,
        task_id: str,
        source_path: str,
        original_filename: str,
        source_text: str | None,
        source_metadata: Dict | None,
        source_structure: Dict | None,
        entities: List[Dict],
        anonymized_text: str,
        operator_config: Dict | None = None,
    ) -> Dict[str, str | bool | None]:
        suffix = Path(source_path).suffix.lower()

        if suffix == ".docx":
            try:
                return self._export_docx(
                    task_id=task_id,
                    source_path=source_path,
                    original_filename=original_filename,
                    source_text=source_text,
                    entities=entities,
                    operator_config=operator_config,
                )
            except Exception as exc:
                logger.exception("DOCX export failed, falling back to TXT: %s", exc)
                return self._export_text(
                    task_id=task_id,
                    original_filename=original_filename,
                    anonymized_text=anonymized_text,
                    warning="DOCX export failed, so the result was saved as TXT.",
                    preserves_format=False,
                )

        if suffix == ".txt":
            return self._export_text(
                task_id=task_id,
                original_filename=original_filename,
                anonymized_text=anonymized_text,
                warning=None,
                preserves_format=True,
            )

        if suffix == ".pdf":
            try:
                return self._export_pdf_as_docx(
                    task_id=task_id,
                    original_filename=original_filename,
                    source_text=source_text,
                    source_metadata=source_metadata,
                    source_structure=source_structure,
                    entities=entities,
                    operator_config=operator_config,
                )
            except Exception as exc:
                logger.exception("PDF-to-DOCX export failed, falling back to TXT: %s", exc)
                return self._export_text(
                    task_id=task_id,
                    original_filename=original_filename,
                    anonymized_text=anonymized_text,
                    warning=(
                        "PDF normalized DOCX export failed, so the result was saved as TXT."
                    ),
                    preserves_format=False,
                )

        return self._export_text(
            task_id=task_id,
            original_filename=original_filename,
            anonymized_text=anonymized_text,
            warning="Format-preserving export is currently available for DOCX and TXT. PDF currently falls back to TXT.",
            preserves_format=False,
        )

    def _export_text(
        self,
        *,
        task_id: str,
        original_filename: str,
        anonymized_text: str,
        warning: str | None,
        preserves_format: bool,
    ) -> Dict[str, str | bool | None]:
        output_path = Path(settings.OUTPUT_DIR) / f"{task_id}_anonymized.txt"
        output_path.write_text(anonymized_text, encoding="utf-8")
        ensure_private_file(output_path)

        return {
            "output_path": str(output_path),
            "download_name": f"{Path(original_filename).stem}_anonymized.txt",
            "file_type": "txt",
            "media_type": TEXT_MIME_TYPE,
            "preserves_format": preserves_format,
            "warning": warning,
        }

    def _export_docx(
        self,
        *,
        task_id: str,
        source_path: str,
        original_filename: str,
        source_text: str | None,
        entities: List[Dict],
        operator_config: Dict | None,
    ) -> Dict[str, str | bool | None]:
        document = Document(source_path)
        replacements = self._build_replacements(entities, operator_config)

        self._replace_in_container(document, replacements)
        for section in document.sections:
            self._replace_in_container(section.header, replacements)
            self._replace_in_container(section.footer, replacements)

        output_path = Path(settings.OUTPUT_DIR) / f"{task_id}_anonymized.docx"
        document.save(output_path)
        if replacements:
            try:
                replace_text_in_docx(output_path, replacements)
            except Exception:
                logger.exception("Tracked-change DOCX XML replacement failed for %s", output_path)
        ensure_private_file(output_path)

        return {
            "output_path": str(output_path),
            "download_name": f"{Path(original_filename).stem}_anonymized.docx",
            "file_type": "docx",
            "media_type": DOCX_MIME_TYPE,
            "preserves_format": True,
            "warning": None,
        }

    def _export_pdf_as_docx(
        self,
        *,
        task_id: str,
        original_filename: str,
        source_text: str | None,
        source_metadata: Dict | None,
        source_structure: Dict | None,
        entities: List[Dict],
        operator_config: Dict | None,
    ) -> Dict[str, str | bool | None]:
        document = Document()
        self._remove_default_paragraph(document)

        replacements = self._build_replacements(entities, operator_config)
        pages = self._build_pdf_pages(
            source_text=source_text,
            source_metadata=source_metadata,
            source_structure=source_structure,
            replacements=replacements,
        )

        if not pages:
            pages = [
                {
                    "width": DEFAULT_PDF_PAGE_WIDTH_PT,
                    "height": DEFAULT_PDF_PAGE_HEIGHT_PT,
                    "blocks": self._build_blocks_from_text_fallback(
                        self._apply_text_replacements(source_text or "", replacements)
                    ),
                }
            ]

        for page_index, page in enumerate(pages):
            self._configure_pdf_section(document, page, page_index)
            self._append_structured_pdf_page(document, page)

        output_path = Path(settings.OUTPUT_DIR) / f"{task_id}_anonymized.docx"
        document.save(output_path)
        ensure_private_file(output_path)

        ocr_pages = 0
        if isinstance(source_metadata, dict):
            try:
                ocr_pages = int(source_metadata.get("ocr_pages", 0) or 0)
            except (TypeError, ValueError):
                ocr_pages = 0

        warning = (
            "PDF has been reconstructed into an editable DOCX with page-level layout recovery. "
            "Titles, line breaks, spacing, and simple tables are preserved as much as possible."
        )
        if ocr_pages > 0:
            warning += " Scanned pages still depend on OCR, so very complex layouts may differ from the original."

        return {
            "output_path": str(output_path),
            "download_name": f"{Path(original_filename).stem}_anonymized.docx",
            "file_type": "docx",
            "media_type": DOCX_MIME_TYPE,
            "preserves_format": False,
            "warning": warning,
        }

    def _build_replacements(
        self, entities: Iterable[Dict], operator_config: Dict | None
    ) -> List[Tuple[str, str]]:
        engine = get_engine()
        config = copy.deepcopy(engine.pipeline_manager._get_default_operator_config())
        if operator_config:
            config.update(copy.deepcopy(operator_config))

        replacement_map: Dict[str, Dict[str, str]] = {}

        sorted_entities = sorted(
            (entity for entity in entities if entity.get("text")),
            key=lambda item: len(item["text"]),
            reverse=True,
        )

        for entity in sorted_entities:
            entity_type = entity["type"]
            source_text = entity["text"]
            if entity.get("replacement_method") == "preserve":
                continue
            if (
                entity_type in engine.pipeline_manager.contextual_desensitizer.PRESERVE_TYPES
                and not entity.get("replacement")
                and not (operator_config and entity_type in operator_config)
            ):
                continue
            replacement = self._coerce_replacement_text(entity.get("replacement"))
            if not replacement:
                entity_rule = config.get(entity_type, config.get("default", {}))
                operator_name = entity_rule.get("operator", "mask")
                params = entity_rule.get("params", {})
                operator = engine.operator_registry.get_operator(operator_name)
                if operator is None:
                    operator = engine.operator_registry.get_operator("mask")

                replacement = self._coerce_replacement_text(
                    operator.operate(source_text, 0, len(source_text), entity_type, **params)
                )
            if not replacement:
                continue
            if replacement == source_text:
                continue

            existing = replacement_map.get(source_text)
            if existing is not None:
                if existing["replacement"] != replacement:
                    preferred = self._prefer_export_entity(
                        existing["entity_type"],
                        entity_type,
                    )
                    if preferred != existing["entity_type"]:
                        replacement_map[source_text] = {
                            "replacement": replacement,
                            "entity_type": entity_type,
                        }
                    logger.warning(
                        "Conflicting replacements detected for '%s'; using %s instead of %s.",
                        source_text,
                        replacement_map[source_text]["entity_type"],
                        existing["entity_type"],
                    )
                continue

            replacement_map[source_text] = {
                "replacement": replacement,
                "entity_type": entity_type,
            }

        return [
            (source_text, data["replacement"])
            for source_text, data in sorted(
                replacement_map.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        ]

    def _coerce_replacement_text(self, value: Any) -> str:
        if value is None:
            return ""

        replacement = str(value).strip()
        if replacement.lower() == "none":
            return ""
        return replacement

    def _prefer_export_entity(self, existing_type: str, candidate_type: str) -> str:
        existing_priority = self.EXPORT_TEXT_PRIORITY.get(existing_type, 99)
        candidate_priority = self.EXPORT_TEXT_PRIORITY.get(candidate_type, 99)
        if candidate_priority < existing_priority:
            return candidate_type
        return existing_type

    def _replace_in_container(self, container, replacements: List[Tuple[str, str]]) -> None:
        for paragraph in container.paragraphs:
            self._replace_in_paragraph(paragraph, replacements)

        for table in container.tables:
            for row in table.rows:
                for cell in row.cells:
                    self._replace_in_container(cell, replacements)

    def _replace_in_paragraph(self, paragraph, replacements: List[Tuple[str, str]]) -> None:
        if not paragraph.runs:
            return

        run_texts = [run.text for run in paragraph.runs]
        if not any(run_texts):
            return

        updated_texts = apply_replacements_to_fragments(run_texts, replacements)
        if updated_texts == run_texts:
            return

        for run, updated_text in zip(paragraph.runs, updated_texts):
            run.text = updated_text

    def _build_pdf_pages(
        self,
        *,
        source_text: str | None,
        source_metadata: Dict | None,
        source_structure: Dict | None,
        replacements: List[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        if isinstance(source_structure, dict):
            raw_pages = source_structure.get("pages")
            if isinstance(raw_pages, list):
                for raw_page in raw_pages:
                    if not isinstance(raw_page, dict):
                        continue
                    page_text = str(raw_page.get("text", "") or "")
                    page_blocks = raw_page.get("blocks")
                    blocks = self._apply_replacements_to_blocks(page_blocks, replacements)
                    if not blocks:
                        blocks = self._build_blocks_from_text_fallback(
                            self._apply_text_replacements(page_text, replacements)
                        )

                    width = self._coerce_float(
                        raw_page.get("width"),
                        DEFAULT_PDF_PAGE_WIDTH_PT,
                    )
                    height = self._coerce_float(
                        raw_page.get("height"),
                        DEFAULT_PDF_PAGE_HEIGHT_PT,
                    )

                    pages.append(
                        {
                            "width": width,
                            "height": height,
                            "blocks": blocks,
                        }
                    )

        if pages:
            return pages

        fallback_text = self._apply_text_replacements(source_text or "", replacements).strip()
        if not fallback_text:
            return []

        page_count = 1
        if isinstance(source_metadata, dict):
            try:
                page_count = max(1, int(source_metadata.get("pages", 1) or 1))
            except (TypeError, ValueError):
                page_count = 1

        if page_count <= 1:
            return [
                {
                    "width": DEFAULT_PDF_PAGE_WIDTH_PT,
                    "height": DEFAULT_PDF_PAGE_HEIGHT_PT,
                    "blocks": self._build_blocks_from_text_fallback(fallback_text),
                }
            ]

        chunks = [chunk.strip() for chunk in fallback_text.split("\n\n") if chunk.strip()]
        return [
            {
                "width": DEFAULT_PDF_PAGE_WIDTH_PT,
                "height": DEFAULT_PDF_PAGE_HEIGHT_PT,
                "blocks": self._build_blocks_from_text_fallback(chunk),
            }
            for chunk in chunks
        ]

    def _apply_replacements_to_blocks(
        self,
        blocks: Any,
        replacements: List[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(blocks, list):
            return []

        normalized_blocks: List[Dict[str, Any]] = []
        for raw_block in blocks:
            if not isinstance(raw_block, dict):
                continue

            block_type = str(raw_block.get("type", "line")).strip().lower()
            if block_type == "table":
                rows = raw_block.get("rows")
                if not isinstance(rows, list):
                    continue
                normalized_rows: List[List[str]] = []
                for row in rows:
                    if not isinstance(row, list):
                        continue
                    normalized_row = [
                        self._apply_text_replacements(str(cell), replacements).strip()
                        for cell in row
                    ]
                    normalized_rows.append(normalized_row)
                normalized_blocks.append(
                    {
                        "type": "table",
                        "rows": normalized_rows,
                        "align": "left",
                    }
                )
                continue

            if block_type == "spacer":
                normalized_blocks.append(
                    {
                        "type": "spacer",
                        "count": max(
                            1,
                            int(raw_block.get("count", raw_block.get("blank_before", 1)) or 1),
                        ),
                    }
                )
                continue

            text = self._apply_text_replacements(str(raw_block.get("text", "")), replacements).strip()
            if not text:
                continue

            align = str(raw_block.get("align", "left")).strip().lower()
            if align not in {"left", "center", "right"}:
                align = "left"

            normalized_blocks.append(
                {
                    "type": block_type if block_type in {"title", "paragraph", "line"} else "line",
                    "text": text,
                    "align": align,
                    "indent_pt": self._coerce_float(raw_block.get("indent_pt"), 0.0),
                    "indent": int(raw_block.get("indent", 0) or 0),
                    "space_before_pt": self._coerce_float(raw_block.get("space_before_pt"), 0.0),
                    "blank_before": max(0, int(raw_block.get("blank_before", 0) or 0)),
                    "font_size_hint": self._coerce_float(raw_block.get("font_size_hint"), 0.0),
                }
            )

        return self._collapse_spacer_blocks(normalized_blocks)

    def _build_blocks_from_text_fallback(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        blocks: List[Dict[str, Any]] = []
        pending_rows: List[List[str]] = []
        for index, raw_line in enumerate(text.splitlines()):
            stripped = raw_line.strip()
            if not stripped:
                if pending_rows:
                    blocks.append({"type": "table", "rows": pending_rows})
                    pending_rows = []
                blocks.append({"type": "spacer", "count": 1})
                continue

            cells = self._split_fallback_cells(raw_line)
            if len(cells) >= 2:
                pending_rows.append(cells)
                continue

            if pending_rows:
                blocks.append({"type": "table", "rows": pending_rows})
                pending_rows = []

            block_type = "title" if index == 0 and self._looks_like_center_title(stripped) else "line"
            blocks.append(
                {
                    "type": block_type,
                    "text": stripped,
                    "align": "center" if block_type == "title" else "left",
                    "indent_pt": 0.0,
                    "space_before_pt": 0.0,
                }
            )

        if pending_rows:
            blocks.append({"type": "table", "rows": pending_rows})

        return self._collapse_spacer_blocks(blocks)

    def _split_fallback_cells(self, line: str) -> List[str]:
        if "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
        else:
            parts = [part.strip() for part in re.split(r"\s{3,}", line)]
        return [part for part in parts if part]

    def _collapse_spacer_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        collapsed: List[Dict[str, Any]] = []
        for block in blocks:
            if str(block.get("type", "")).lower() == "spacer":
                count = max(1, int(block.get("count", 1) or 1))
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

    def _configure_pdf_section(self, document: Document, page: Dict[str, Any], page_index: int) -> None:
        if page_index == 0:
            section = document.sections[0]
        else:
            section = document.add_section(WD_SECTION.NEW_PAGE)

        section.page_width = Pt(self._coerce_float(page.get("width"), DEFAULT_PDF_PAGE_WIDTH_PT))
        section.page_height = Pt(self._coerce_float(page.get("height"), DEFAULT_PDF_PAGE_HEIGHT_PT))
        section.left_margin = Pt(36)
        section.right_margin = Pt(36)
        section.top_margin = Pt(32)
        section.bottom_margin = Pt(32)

    def _append_structured_pdf_page(self, document: Document, page: Dict[str, Any]) -> None:
        blocks = page.get("blocks") if isinstance(page, dict) else None
        if not isinstance(blocks, list) or not blocks:
            document.add_paragraph("")
            return

        for index, block in enumerate(blocks):
            block_type = str(block.get("type", "line")).strip().lower()
            if block_type == "spacer":
                count = max(1, int(block.get("count", 1) or 1))
                for _ in range(count):
                    document.add_paragraph("")
                continue

            if block_type == "table":
                self._append_table_block(document, block)
                continue

            self._append_text_block(document, block, index)

    def _append_text_block(
        self,
        document: Document,
        block: Dict[str, Any],
        block_index: int,
    ) -> None:
        text = str(block.get("text", "")).strip()
        if not text:
            return

        blank_before = max(0, int(block.get("blank_before", 0) or 0))
        for _ in range(blank_before):
            document.add_paragraph("")

        paragraph = document.add_paragraph()
        paragraph.alignment = self._resolve_alignment(block.get("align"))
        paragraph_format = paragraph.paragraph_format
        paragraph_format.space_after = Pt(0)
        paragraph_format.space_before = Pt(
            min(18.0, self._coerce_float(block.get("space_before_pt"), 0.0))
        )
        paragraph_format.line_spacing = 1.0

        indent_pt = self._coerce_float(block.get("indent_pt"), 0.0)
        if indent_pt <= 0:
            indent_level = max(0, int(block.get("indent", 0) or 0))
            indent_pt = indent_level * 18.0
        if indent_pt > 0:
            paragraph_format.left_indent = Pt(min(indent_pt, 144.0))

        font_size = self._resolve_font_size(block)
        lines = text.splitlines()
        for line_index, line in enumerate(lines):
            run = paragraph.add_run(line)
            run.font.size = Pt(font_size)
            if str(block.get("type", "")).lower() == "title":
                run.bold = True
            if line_index < len(lines) - 1:
                paragraph.add_run().add_break(WD_BREAK.LINE)

        if block_index == 0 and str(block.get("type", "")).lower() == "title":
            paragraph_format.space_after = Pt(6)

    def _append_table_block(self, document: Document, block: Dict[str, Any]) -> None:
        rows = block.get("rows")
        if not isinstance(rows, list) or not rows:
            return

        max_cols = max(len(row) for row in rows if isinstance(row, list)) if rows else 0
        if max_cols <= 0:
            return

        table = document.add_table(rows=len(rows), cols=max_cols)
        table.style = "Table Grid"
        table.autofit = True

        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            for col_index in range(max_cols):
                text = str(row[col_index]).strip() if col_index < len(row) else ""
                cell = table.cell(row_index, col_index)
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.space_before = Pt(0)
                run = paragraph.add_run(text)
                run.font.size = Pt(10.5)

    def _resolve_alignment(self, align: Any) -> WD_ALIGN_PARAGRAPH:
        value = str(align or "left").strip().lower()
        if value == "center":
            return WD_ALIGN_PARAGRAPH.CENTER
        if value == "right":
            return WD_ALIGN_PARAGRAPH.RIGHT
        return WD_ALIGN_PARAGRAPH.LEFT

    def _resolve_font_size(self, block: Dict[str, Any]) -> float:
        block_type = str(block.get("type", "line")).strip().lower()
        font_size_hint = self._coerce_float(block.get("font_size_hint"), 0.0)
        if font_size_hint > 0:
            font_size_hint = max(9.0, min(font_size_hint, 16.0))

        if block_type == "title":
            return max(14.0, font_size_hint or 14.0)
        if block_type == "paragraph":
            return font_size_hint or 11.0
        return font_size_hint or 10.5

    def _looks_like_center_title(self, text: str) -> bool:
        if len(text) > 40:
            return False
        return not any(token in text for token in ["：", ":", "。", "；", ";", "，", ","])

    def _remove_default_paragraph(self, document: Document) -> None:
        if len(document.paragraphs) != 1:
            return
        paragraph = document.paragraphs[0]
        if paragraph.text.strip():
            return
        element = paragraph._element
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    def _apply_text_replacements(
        self,
        text: str,
        replacements: List[Tuple[str, str]],
    ) -> str:
        updated = text
        for original, replacement in replacements:
            updated = updated.replace(original, replacement)
        return updated

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if numeric <= 0:
            return default
        return numeric
