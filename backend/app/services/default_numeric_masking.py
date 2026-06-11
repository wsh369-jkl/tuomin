"""Text-level numeric masking for the default desensitization line."""

from __future__ import annotations

import re


DATE_VALUE_RE = re.compile(
    r"(?P<year>\d{4})\s*(?:年|[./-])\s*(?P<month>\d{1,2})\s*(?:月|[./-])\s*(?P<day>\d{1,2})?\s*(?:日|号)?"
)


def mask_default_numeric_text(text: str) -> str:
    """Mask every digit with '*', except the year part of date values."""
    if not text:
        return text
    preserve_indexes: set[int] = set()
    for match in DATE_VALUE_RE.finditer(text):
        year_start, year_end = match.span("year")
        preserve_indexes.update(range(year_start, year_end))

    chars = list(text)
    for index, char in enumerate(chars):
        if char.isdigit() and index not in preserve_indexes:
            chars[index] = "*"
    return "".join(chars)
