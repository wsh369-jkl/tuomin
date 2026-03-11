"""DOCX parser backed by python-docx."""

import asyncio
import logging
from typing import Any, Dict

from docx import Document

from app.processors.base_parser import BaseParser
from app.processors.docx_xml_utils import extract_docx_text

logger = logging.getLogger(__name__)


class DOCXParser(BaseParser):
    async def parse(self, file_path: str, **kwargs: Any) -> Dict:
        logger.info("Start parsing DOCX: %s", file_path)
        try:
            return await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as exc:
            logger.error("DOCX parsing failed: %s", exc)
            raise

    def _parse_sync(self, file_path: str) -> Dict:
        doc = Document(file_path)
        tracked_metadata: Dict[str, int | bool] = {
            "tracked_changes_detected": False,
            "tracked_insertions": 0,
            "tracked_deletions": 0,
            "tracked_deleted_text_nodes": 0,
        }

        try:
            text, tracked_metadata = extract_docx_text(file_path)
        except Exception as exc:
            logger.warning(
                "DOCX XML extraction failed, falling back to python-docx paragraphs: %s",
                exc,
            )
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)

            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_parts.append(cell.text)

            text = "\n".join(text_parts)

        metadata = {
            "paragraphs": len(doc.paragraphs),
            "tables": len(doc.tables),
            "format": "docx",
            "file_path": file_path,
        }
        metadata.update(tracked_metadata)

        logger.info(
            "DOCX parsing complete, paragraphs=%s, text_length=%s, tracked_changes=%s",
            metadata["paragraphs"],
            len(text),
            metadata["tracked_changes_detected"],
        )
        return {
            "text": text,
            "metadata": metadata,
        }

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".docx"
