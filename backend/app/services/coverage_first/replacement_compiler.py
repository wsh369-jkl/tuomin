"""Passive replacement allocation summary."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_replacement_compile_summary(
    candidate_ledger: dict[str, Any],
    *,
    directory_compile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = [item for item in candidate_ledger.get("entries") or [] if isinstance(item, dict)]
    by_text: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        text = str(candidate.get("normalized_text") or candidate.get("text") or "").strip()
        replacement = str(candidate.get("replacement") or "").strip()
        if text and replacement:
            by_text[text].add(replacement)
    multi_replacement_count = sum(1 for replacements in by_text.values() if len(replacements) > 1)
    directory_rows = [
        item for item in (directory_compile or {}).get("directory_rows", []) if isinstance(item, dict)
    ]
    directory_replacements = {
        str(item.get("replacement") or "").strip()
        for item in directory_rows
        if str(item.get("replacement") or "").strip()
    }
    return {
        "summary": {
            "preassigned_replacement_subject_count": len(by_text),
            "subject_multi_replacement_count": multi_replacement_count,
            "compiled_replacement_count": len(directory_replacements),
            "requires_final_identity_before_allocation": True,
        },
    }
