"""TXT parser with a small encoding fallback list."""

import asyncio
import logging
from typing import Any, Dict

from app.processors.base_parser import BaseParser

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    async def parse(self, file_path: str, **kwargs: Any) -> Dict:
        logger.info("Start parsing TXT: %s", file_path)
        try:
            return await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as exc:
            logger.error("TXT parsing failed: %s", exc)
            raise

    def _parse_sync(self, file_path: str) -> Dict:
        encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
        text = None
        used_encoding = None

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    text = handle.read()
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("Unable to detect the text file encoding")

        metadata = {
            "encoding": used_encoding,
            "format": "txt",
            "file_path": file_path,
        }

        logger.info(
            "TXT parsing complete, encoding=%s, text_length=%s",
            used_encoding,
            len(text),
        )
        return {
            "text": text,
            "metadata": metadata,
        }

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".txt"
