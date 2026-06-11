"""Semantic text segmentation for Chinese legal-practice recognition passes."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from app.services.contract_structure_backfill_service import ContractStructureBackfillService

SOFT_BOUNDARY_CHARS = "。；;，,、 \t"


def iter_legal_text_segments(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int = 80,
    min_split_chars: int = 80,
) -> Iterable[tuple[str, int]]:
    """Yield recognition segments with offsets while preserving legal field context.

    The previous low-memory recognizers split each physical line into hard
    windows. Chinese legal documents often put several labelled parties or table
    cell values in one long paragraph, so a hard boundary can cut an entity or
    separate a field label from its value. This helper keeps normal short lines
    intact and adds overlapping windows only when a long line must be split.
    """

    if max_chars <= 0:
        yield text, 0
        return

    cursor = 0
    for raw_line in str(text or "").splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw_line)
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        if len(line) <= max_chars:
            yield line, line_start
            continue
        yield from _split_long_line(
            line,
            line_start=line_start,
            max_chars=max_chars,
            overlap_chars=max(0, min(overlap_chars, max_chars // 2)),
            min_split_chars=max(1, min_split_chars),
        )


def _split_long_line(
    line: str,
    *,
    line_start: int,
    max_chars: int,
    overlap_chars: int,
    min_split_chars: int,
) -> Iterable[tuple[str, int]]:
    start = 0
    line_length = len(line)
    while start < line_length:
        end = min(line_length, start + max_chars)
        if end < line_length:
            end = _best_split_position(line, start=start, end=end, min_split_chars=min_split_chars)
        if end <= start:
            end = min(line_length, start + max_chars)
        yield line[start:end], line_start + start
        if end >= line_length:
            break
        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = _trim_overlap_to_boundary(line, next_start, end)


def _best_split_position(line: str, *, start: int, end: int, min_split_chars: int) -> int:
    field_boundary = _last_field_boundary(line, start=start + min_split_chars, end=end)
    if field_boundary > start + min_split_chars:
        return field_boundary
    for char in SOFT_BOUNDARY_CHARS:
        split_at = line.rfind(char, start + min_split_chars, end)
        if split_at > start + min_split_chars:
            return split_at + 1
    return end


def _last_field_boundary(line: str, *, start: int, end: int) -> int:
    best = -1
    for match in _field_boundary_pattern().finditer(line):
        index = match.start()
        if start <= index < end:
            best = index
    return best


@lru_cache(maxsize=1)
def _field_boundary_pattern() -> re.Pattern[str]:
    labels = sorted(
        {str(label or "").strip() for label in ContractStructureBackfillService.FOLLOWING_FIELD_LABELS if str(label or "").strip()},
        key=len,
        reverse=True,
    )
    if not labels:
        return re.compile(r"a^")
    label_pattern = "|".join(re.escape(label) for label in labels)
    return re.compile(rf"(?=(?:{label_pattern})\s*[:：])")


def _trim_overlap_to_boundary(line: str, start: int, previous_end: int) -> int:
    if start <= 0:
        return start
    window = line[start:previous_end]
    for index, char in enumerate(window):
        if char in "。；;，,、 \t":
            return start + index + 1
    return start
