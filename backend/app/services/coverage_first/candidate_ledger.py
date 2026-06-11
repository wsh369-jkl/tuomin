"""Passive candidate evidence ledger."""

from __future__ import annotations

from typing import Any, Iterable

from app.core.recognizer_base import RecognizerResult
from app.services.coverage_first.document_graph import DocumentGraph, DocumentUnit


def build_candidate_ledger(
    results: Iterable[RecognizerResult],
    document_graph: DocumentGraph,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    units = list(document_graph.units or [])
    for index, result in enumerate(results or [], start=1):
        unit = _unit_for_result(result, units)
        metadata = dict(result.metadata or {})
        entry = {
            "candidate_id": f"CA{index:05d}",
            "text": str(result.text or ""),
            "normalized_text": str(metadata.get("normalized_text") or "").strip() or "".join(str(result.text or "").split()),
            "entity_type": str(result.entity_type or "").strip().upper(),
            "start": int(result.start),
            "end": int(result.end),
            "unit_id": unit.unit_id if unit else str(metadata.get("docx_unit_id") or ""),
            "source": str(result.source or ""),
            "confidence": float(result.score or 0.0),
            "source_layer": str(metadata.get("source_layer") or result.source or ""),
            "rewrite_policy": str(metadata.get("docx_rewrite_policy") or (unit.rewrite_policy if unit else "")),
            "has_unit": unit is not None or bool(metadata.get("docx_unit_id")),
        }
        entries.append(entry)

    source_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    with_unit_count = 0
    for entry in entries:
        source_counts[entry["source"]] = source_counts.get(entry["source"], 0) + 1
        type_counts[entry["entity_type"]] = type_counts.get(entry["entity_type"], 0) + 1
        if entry["has_unit"]:
            with_unit_count += 1

    return {
        "entries": entries,
        "summary": {
            "candidate_count": len(entries),
            "candidate_with_unit_count": with_unit_count,
            "candidate_without_unit_count": max(0, len(entries) - with_unit_count),
            "source_counts": dict(sorted(source_counts.items())),
            "type_counts": dict(sorted(type_counts.items())),
        },
    }


def _unit_for_result(result: RecognizerResult, units: list[DocumentUnit]) -> DocumentUnit | None:
    metadata = dict(result.metadata or {})
    unit_id = str(metadata.get("docx_unit_id") or "").strip()
    if unit_id:
        for unit in units:
            if unit.unit_id == unit_id:
                return unit
    start = int(result.start)
    end = int(result.end)
    candidates = [unit for unit in units if unit.start <= start and end <= unit.end]
    if not candidates:
        return None
    return min(candidates, key=lambda unit: (unit.end - unit.start, unit.unit_id))
