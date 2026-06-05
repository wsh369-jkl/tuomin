"""Document workflow routing helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

LARGE_DOCUMENT_PAGE_THRESHOLD = 20
LARGE_DOCUMENT_TEXT_THRESHOLD = 20_000


def resolve_large_document_page_count(
    *,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
) -> int:
    page_count = 0
    if isinstance(source_metadata, dict):
        for key in ("pages", "page_count"):
            try:
                page_count = max(page_count, int(source_metadata.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue
    if isinstance(source_structure, dict):
        for key in ("page_count", "pages_count"):
            try:
                page_count = max(page_count, int(source_structure.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue
        raw_pages = source_structure.get("pages")
        if isinstance(raw_pages, list):
            structured_pages = sum(1 for item in raw_pages if isinstance(item, dict))
            page_count = max(page_count, structured_pages)
    return max(0, page_count)


def classify_document_workflow(
    *,
    text: str,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text_length = len(text or "")
    page_count = resolve_large_document_page_count(
        source_metadata=source_metadata,
        source_structure=source_structure,
    )
    force_standard_workflow = bool(
        isinstance(source_metadata, dict)
        and source_metadata.get("_force_standard_document_workflow")
    ) or bool(
        isinstance(source_structure, dict)
        and source_structure.get("_force_standard_document_workflow")
    )
    triggered_by_page_count = page_count > LARGE_DOCUMENT_PAGE_THRESHOLD
    triggered_by_text_length = text_length >= LARGE_DOCUMENT_TEXT_THRESHOLD
    enabled = (triggered_by_page_count or triggered_by_text_length) and not force_standard_workflow
    return {
        "enabled": enabled,
        "page_count": page_count,
        "text_length": text_length,
        "page_threshold": LARGE_DOCUMENT_PAGE_THRESHOLD,
        "text_threshold": LARGE_DOCUMENT_TEXT_THRESHOLD,
        "triggered_by_page_count": triggered_by_page_count,
        "triggered_by_text_length": triggered_by_text_length,
        "force_standard_workflow": force_standard_workflow,
    }
