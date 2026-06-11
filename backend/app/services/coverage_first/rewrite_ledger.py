"""Passive rewrite ledger and pre-verification summary."""

from __future__ import annotations

from collections import Counter
from typing import Any


NON_REWRITE_TYPES = {"DATE", "AMOUNT", "POSITION"}


def build_rewrite_ledger_summary(
    *,
    candidate_ledger: dict[str, Any],
    directory_compile: dict[str, Any] | None = None,
    source_text: str | None = None,
) -> dict[str, Any]:
    """Build exact range rewrite entries from compiled directory rows.

    The ledger is intentionally derived from the final directory, not from the
    raw recognizer list. This keeps replacement allocation and file mutation on
    the same subject identity layer.
    """
    candidates = {
        str(item.get("candidate_id") or ""): item
        for item in candidate_ledger.get("entries") or []
        if isinstance(item, dict) and str(item.get("candidate_id") or "")
    }
    directory_rows = [
        item for item in (directory_compile or {}).get("directory_rows", []) if isinstance(item, dict)
    ]
    entries: list[dict[str, Any]] = []
    required_candidate_ids: set[str] = set()
    missing_candidate_ids: list[str] = []

    for row in directory_rows:
        replacement = str(row.get("replacement") or "").strip()
        subject_id = str(row.get("subject_id") or "").strip()
        subject_type = str(row.get("subject_type") or "").strip()
        for candidate_id in row.get("candidate_ids") or []:
            candidate_key = str(candidate_id or "").strip()
            if not candidate_key:
                continue
            required_candidate_ids.add(candidate_key)
            candidate = candidates.get(candidate_key)
            if candidate is None:
                missing_candidate_ids.append(candidate_key)
                continue
            entry = _build_rewrite_entry(
                candidate=candidate,
                row=row,
                subject_id=subject_id,
                subject_type=subject_type,
                replacement=replacement,
                source_text=source_text,
                input_order=len(entries),
            )
            entries.append(entry)

    _mark_overlapping_entries(entries)
    blocked_entries = [entry for entry in entries if entry.get("verification_status") == "blocked"]
    without_unit = sum(1 for entry in entries if entry.get("failure_reason") == "missing_unit")
    unrewritable = sum(1 for entry in entries if entry.get("failure_reason") == "unrewritable_policy")
    source_mismatch = sum(1 for entry in entries if entry.get("failure_reason") == "source_range_mismatch")
    overlap_count = sum(1 for entry in entries if entry.get("failure_reason") == "range_overlap")
    missing_replacement = sum(1 for entry in entries if entry.get("failure_reason") == "missing_replacement")
    invalid_range = sum(1 for entry in entries if entry.get("failure_reason") == "invalid_range")
    candidate_type_counter = Counter(str(entry.get("entity_type") or "") for entry in entries)
    return {
        "rewrite_entries": entries,
        "summary": {
            "required_rewrite_candidate_count": len(required_candidate_ids),
            "rewrite_entry_count": len(entries),
            "required_rewrite_entry_count": len(entries),
            "blocked_rewrite_entry_count": len(blocked_entries),
            "missing_directory_candidate_count": len(missing_candidate_ids),
            "candidate_without_unit_count": without_unit,
            "candidate_unrewritable_count": unrewritable,
            "candidate_source_mismatch_count": source_mismatch,
            "candidate_range_overlap_count": overlap_count,
            "candidate_missing_replacement_count": missing_replacement,
            "candidate_invalid_range_count": invalid_range,
            "rewrite_entry_type_counts": dict(sorted(candidate_type_counter.items())),
            "prewrite_verification_passed": (
                len(blocked_entries) == 0 and len(missing_candidate_ids) == 0
            ),
        },
    }


def _build_rewrite_entry(
    *,
    candidate: dict[str, Any],
    row: dict[str, Any],
    subject_id: str,
    subject_type: str,
    replacement: str,
    source_text: str | None,
    input_order: int,
) -> dict[str, Any]:
    source = str(candidate.get("text") or "")
    start = _coerce_int(candidate.get("start"), -1)
    end = _coerce_int(candidate.get("end"), -1)
    rewrite_policy = str(candidate.get("rewrite_policy") or "exact").strip() or "exact"
    status = "pending_prewrite"
    reason = ""
    if not replacement:
        status = "blocked"
        reason = "missing_replacement"
    elif str(candidate.get("entity_type") or "") in NON_REWRITE_TYPES:
        status = "blocked"
        reason = "non_rewrite_type"
    elif start < 0 or end <= start or not source:
        status = "blocked"
        reason = "invalid_range"
    elif not bool(candidate.get("has_unit")) or not str(candidate.get("unit_id") or "").strip():
        status = "blocked"
        reason = "missing_unit"
    elif rewrite_policy != "exact":
        status = "blocked"
        reason = "unrewritable_policy"
    elif source_text is not None and source_text[start:end] != source:
        status = "blocked"
        reason = "source_range_mismatch"

    return {
        "rewrite_id": f"RW{input_order + 1:05d}",
        "subject_id": subject_id,
        "subject_type": subject_type,
        "canonical_text": str(row.get("canonical_text") or ""),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_text": source,
        "replacement": replacement,
        "entity_type": str(candidate.get("entity_type") or subject_type),
        "start": start,
        "end": end,
        "unit_id": str(candidate.get("unit_id") or ""),
        "rewrite_policy": rewrite_policy,
        "source": str(candidate.get("source") or ""),
        "input_order": input_order,
        "applied": False,
        "verification_status": status,
        "failure_reason": reason,
    }


def _mark_overlapping_entries(entries: list[dict[str, Any]]) -> None:
    candidates = [
        entry
        for entry in entries
        if entry.get("verification_status") != "blocked"
        and _coerce_int(entry.get("start"), -1) >= 0
        and _coerce_int(entry.get("end"), -1) > _coerce_int(entry.get("start"), -1)
    ]
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for entry in candidates:
        by_unit.setdefault(str(entry.get("unit_id") or ""), []).append(entry)
    for unit_entries in by_unit.values():
        ordered = sorted(
            unit_entries,
            key=lambda item: (
                _coerce_int(item.get("start"), 0),
                -(_coerce_int(item.get("end"), 0) - _coerce_int(item.get("start"), 0)),
                str(item.get("rewrite_id") or ""),
            ),
        )
        previous: dict[str, Any] | None = None
        for entry in ordered:
            if previous is not None and _coerce_int(entry.get("start"), 0) < _coerce_int(previous.get("end"), 0):
                _block_overlap(previous)
                _block_overlap(entry)
            if previous is None or _coerce_int(entry.get("end"), 0) > _coerce_int(previous.get("end"), 0):
                previous = entry


def _block_overlap(entry: dict[str, Any]) -> None:
    entry["verification_status"] = "blocked"
    entry["failure_reason"] = "range_overlap"


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
