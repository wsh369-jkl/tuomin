"""Compile final export directory and rewrite entries from prepared entities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from app.services.coverage_first.directory_compiler import REPLACEMENT_PREFIX, TYPE_PRIORITY


NON_REWRITE_TYPES = {"DATE", "AMOUNT", "POSITION"}


def build_coverage_first_final_export_bundle(
    *,
    entities: Iterable[dict[str, Any]],
    source_text: str | None,
) -> dict[str, Any]:
    """Build the final directory and exact rewrite plan from prepared entities.

    This runs after replacement allocation. It does not invent replacements; it
    compiles the already prepared entity layer into a single export authority.
    """
    prepared = [dict(item) for item in entities or [] if isinstance(item, dict)]
    groups = _group_entities(prepared)
    directory_rows = _directory_rows(groups)
    rewrite_entries = _rewrite_entries(
        directory_rows=directory_rows,
        source_text=source_text,
    )
    mapping_entities = _mapping_entities_from_rows(directory_rows)
    summary = _summary(
        directory_rows=directory_rows,
        rewrite_entries=rewrite_entries,
    )
    return {
        "enabled": bool(directory_rows),
        "directory_rows": directory_rows,
        "mapping_entities": mapping_entities,
        "rewrite_entries": rewrite_entries,
        "summary": summary,
    }


def _group_entities(entities: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entity_type = _entity_type(entity)
        source_text = str(entity.get("text") or "").strip()
        replacement = _replacement(entity)
        if not source_text:
            continue
        key = _subject_key(entity=entity, entity_type=entity_type, source_text=source_text, replacement=replacement)
        buckets[key].append(entity)
    return sorted(buckets.values(), key=_group_sort_key)


def _directory_rows(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        subject_type = _canonical_type(group)
        surfaces = _surfaces(group)
        replacement = _replacement(group[0])
        occurrences = [_occurrence(item, index=occurrence_index) for occurrence_index, item in enumerate(group, start=1)]
        rows.append(
            {
                "subject_id": f"CFE{index:05d}",
                "subject_type": subject_type,
                "canonical_text": surfaces[0] if surfaces else "",
                "surfaces": surfaces,
                "replacement": replacement,
                "occurrence_count": len(occurrences),
                "occurrences": occurrences,
                "status": "final_compiled",
                "context_label": _first_non_empty(group, "context_label") or "未命名字段",
                "context_role": _first_non_empty(group, "context_role") or "通用上下文",
                "replacement_family_key": _first_metadata_value(group, "replacement_family_key"),
                "canonical_key": _first_entity_or_metadata_value(group, "canonical_key"),
            }
        )
    return rows


def _rewrite_entries(
    *,
    directory_rows: list[dict[str, Any]],
    source_text: str | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in directory_rows:
        replacement = str(row.get("replacement") or "").strip()
        for occurrence in row.get("occurrences") or []:
            if not isinstance(occurrence, dict):
                continue
            entry = _rewrite_entry(
                row=row,
                occurrence=occurrence,
                replacement=replacement,
                source_text=source_text,
                input_order=len(entries),
            )
            entries.append(entry)
    _mark_overlaps(entries)
    return entries


def _rewrite_entry(
    *,
    row: dict[str, Any],
    occurrence: dict[str, Any],
    replacement: str,
    source_text: str | None,
    input_order: int,
) -> dict[str, Any]:
    source = str(occurrence.get("text") or "")
    start = _as_int(occurrence.get("start"), -1)
    end = _as_int(occurrence.get("end"), -1)
    entity_type = str(occurrence.get("type") or row.get("subject_type") or "")
    status = "pending_prewrite"
    reason = ""
    if entity_type in NON_REWRITE_TYPES:
        status = "blocked"
        reason = "non_rewrite_type"
    elif not replacement:
        status = "blocked"
        reason = "missing_replacement"
    elif start < 0 or end <= start or not source:
        status = "blocked"
        reason = "invalid_range"
    elif source_text is not None and source_text[start:end] != source:
        status = "blocked"
        reason = "source_range_mismatch"
    return {
        "rewrite_id": f"FRW{input_order + 1:05d}",
        "subject_id": str(row.get("subject_id") or ""),
        "source_text": source,
        "replacement": replacement,
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "input_order": input_order,
        "verification_status": status,
        "failure_reason": reason,
    }


def _mapping_entities_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping_entities: list[dict[str, Any]] = []
    for row in rows:
        first_occurrence = next(
            (item for item in row.get("occurrences") or [] if isinstance(item, dict)),
            {},
        )
        mapping_entities.append(
            {
                "type": row.get("subject_type"),
                "text": row.get("canonical_text"),
                "replacement": row.get("replacement"),
                "start": first_occurrence.get("start", 10**12),
                "context_label": row.get("context_label"),
                "context_role": row.get("context_role"),
                "metadata": {
                    "coverage_first_subject_id": row.get("subject_id"),
                    "coverage_first_surfaces": list(row.get("surfaces") or []),
                    "coverage_first_occurrence_count": int(row.get("occurrence_count") or 0),
                },
            }
        )
    return mapping_entities


def _summary(
    *,
    directory_rows: list[dict[str, Any]],
    rewrite_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    replacement_owner: dict[str, set[str]] = defaultdict(set)
    subject_replacements: dict[str, set[str]] = defaultdict(set)
    blocked = [item for item in rewrite_entries if item.get("verification_status") == "blocked"]
    for row in directory_rows:
        subject_id = str(row.get("subject_id") or "")
        replacement = str(row.get("replacement") or "").strip()
        if replacement:
            replacement_owner[replacement].add(subject_id)
            subject_replacements[subject_id].add(replacement)
    replacement_reused_count = sum(1 for owners in replacement_owner.values() if len(owners) > 1)
    subject_multi_replacement_count = sum(1 for values in subject_replacements.values() if len(values) > 1)
    failure_counts = Counter(str(item.get("failure_reason") or "unknown") for item in blocked)
    ready = (
        bool(directory_rows)
        and not blocked
        and replacement_reused_count == 0
        and subject_multi_replacement_count == 0
    )
    return {
        "final_directory_subject_count": len(directory_rows),
        "final_rewrite_entry_count": len(rewrite_entries),
        "final_blocked_rewrite_entry_count": len(blocked),
        "final_rewrite_failure_counts": dict(sorted(failure_counts.items())),
        "final_replacement_reused_by_multi_subject_count": replacement_reused_count,
        "final_subject_multi_replacement_count": subject_multi_replacement_count,
        "final_export_ready": ready,
    }


def _subject_key(
    *,
    entity: dict[str, Any],
    entity_type: str,
    source_text: str,
    replacement: str,
) -> tuple[str, str, str]:
    metadata = dict(entity.get("metadata") or {})
    canonical = str(
        entity.get("canonical_key")
        or metadata.get("canonical_key")
        or metadata.get("replacement_family_key")
        or entity.get("group_id")
        or ""
    ).strip()
    if canonical:
        return (entity_type, "canonical", canonical)
    return (entity_type, "replacement", replacement or source_text)


def _group_sort_key(group: list[dict[str, Any]]) -> tuple[int, int, str]:
    subject_type = _canonical_type(group)
    start = min(_as_int(item.get("start"), 10**12) for item in group) if group else 10**12
    return (TYPE_PRIORITY.get(subject_type, 99), start, _surfaces(group)[0] if _surfaces(group) else "")


def _canonical_type(group: list[dict[str, Any]]) -> str:
    counts = Counter(_entity_type(item) for item in group)
    if not counts:
        return "UNKNOWN"
    return sorted(counts, key=lambda value: (-counts[value], TYPE_PRIORITY.get(value, 99), value))[0]


def _surfaces(group: list[dict[str, Any]]) -> list[str]:
    counter = Counter(str(item.get("text") or "").strip() for item in group if str(item.get("text") or "").strip())
    return [
        value
        for value, _ in sorted(counter.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    ][:12]


def _occurrence(entity: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "occurrence_id": f"O{index:04d}",
        "type": _entity_type(entity),
        "text": str(entity.get("text") or ""),
        "start": _as_int(entity.get("start"), -1),
        "end": _as_int(entity.get("end"), -1),
        "source": str(entity.get("source") or ""),
    }


def _entity_type(entity: dict[str, Any]) -> str:
    return str(entity.get("type") or entity.get("entity_type") or "").strip().upper()


def _replacement(entity: dict[str, Any]) -> str:
    return str(entity.get("replacement") or "").strip()


def _first_non_empty(group: list[dict[str, Any]], key: str) -> str:
    for entity in sorted(group, key=lambda item: _as_int(item.get("start"), 10**12)):
        value = str(entity.get(key) or "").strip()
        if value:
            return value
    return ""


def _first_metadata_value(group: list[dict[str, Any]], key: str) -> str:
    for entity in sorted(group, key=lambda item: _as_int(item.get("start"), 10**12)):
        metadata = dict(entity.get("metadata") or {})
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _first_entity_or_metadata_value(group: list[dict[str, Any]], key: str) -> str:
    for entity in sorted(group, key=lambda item: _as_int(item.get("start"), 10**12)):
        metadata = dict(entity.get("metadata") or {})
        value = str(entity.get(key) or metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _mark_overlaps(entries: list[dict[str, Any]]) -> None:
    active = [
        entry
        for entry in entries
        if entry.get("verification_status") != "blocked"
        and _as_int(entry.get("start"), -1) >= 0
        and _as_int(entry.get("end"), -1) > _as_int(entry.get("start"), -1)
    ]
    previous: dict[str, Any] | None = None
    for entry in sorted(active, key=lambda item: (_as_int(item.get("start"), 0), -_span_len(item))):
        if previous is not None and _as_int(entry.get("start"), 0) < _as_int(previous.get("end"), 0):
            _block_overlap(previous)
            _block_overlap(entry)
        if previous is None or _as_int(entry.get("end"), 0) > _as_int(previous.get("end"), 0):
            previous = entry


def _block_overlap(entry: dict[str, Any]) -> None:
    entry["verification_status"] = "blocked"
    entry["failure_reason"] = "range_overlap"


def _span_len(entry: dict[str, Any]) -> int:
    return max(0, _as_int(entry.get("end"), 0) - _as_int(entry.get("start"), 0))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
