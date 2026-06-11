"""Rule-layer compaction before dispatching expensive review workers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from app.rules.default_subject_policy import DEFAULT_SUBJECT_TYPES, canonical_default_entity_type
from app.services.lowmem_entity_utils import (
    is_generic_organization_term,
    is_identity_reference_term,
    is_position_title,
    normalize_entity_text,
)


FORMAT_OR_NON_SEMANTIC_TYPES = {"DATE", "AMOUNT", "POSITION"}
MODEL_SOURCE_NAMES = {"uie", "ner", "secondary_ner", "qwen_fragment_review"}
HIGH_VALUE_TYPES = {
    *DEFAULT_SUBJECT_TYPES,
}
NON_SUBJECT_TABLE_TERMS = {
    "证据内容",
    "是否具备",
    "公司提交材料",
    "需要补充侦查",
    "补充侦查",
    "提交材料",
    "情况说明",
}


def compact_review_worker_payload_entities(
    *,
    primary_entities: Iterable[dict[str, Any]],
    review_entities: Iterable[dict[str, Any]],
    max_review_entities: int = 512,
) -> dict[str, Any]:
    """Shrink expensive review-worker input without changing final source entities."""
    primary_rows = [dict(item) for item in primary_entities or [] if isinstance(item, dict)]
    review_rows = [dict(item) for item in review_entities or [] if isinstance(item, dict)]
    compacted_review = _compact_rows(review_rows or primary_rows, max_items=max_review_entities)
    return {
        "entities": primary_rows,
        "review_entities": compacted_review,
        "summary": {
            "primary_input_count": len(primary_rows),
            "primary_output_count": len(primary_rows),
            "review_input_count": len(review_rows),
            "review_output_count": len(compacted_review),
            "primary_filtered_count": 0,
            "review_filtered_count": max(0, len(review_rows) - len(compacted_review)),
            "max_review_entities": int(max_review_entities),
        },
    }


def compact_recognizer_results_for_review(
    results: Iterable[Any],
    *,
    max_items: int = 512,
) -> list[Any]:
    rows = [item for item in results or [] if item is not None]
    selected_indexes = _selected_indexes_for_result_rows(rows, max_items=max_items)
    return [rows[index] for index in selected_indexes]


def _compact_rows(rows: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    selected_indexes = _selected_indexes_for_dict_rows(rows, max_items=max_items)
    return [rows[index] for index in selected_indexes]


def _selected_indexes_for_dict_rows(rows: list[dict[str, Any]], *, max_items: int) -> list[int]:
    must_keep_scored: list[tuple[tuple[int, int, int, int], int]] = []
    optional_scored: list[tuple[tuple[int, int, int, int], int]] = []
    seen: set[tuple[str, str, str]] = set()
    type_surface_counts = Counter(
        (
            _entity_type(row),
            normalize_entity_text(str(row.get("text") or "")),
        )
        for row in rows
    )
    for index, row in enumerate(rows):
        entity_type = canonical_default_entity_type(_entity_type(row), str(row.get("text") or ""))
        if not entity_type:
            continue
        text = str(row.get("text") or "").strip()
        normalized = normalize_entity_text(text)
        must_keep = _must_keep_for_review(row)
        if _rule_drop(entity_type, text, normalized) and not must_keep:
            continue
        key = _dedupe_key(row, entity_type, normalized, force_span=must_keep)
        if key in seen:
            continue
        seen.add(key)
        score = _priority_for_row(
            entity_type=entity_type,
            text=text,
            normalized=normalized,
            source=str(row.get("source") or ""),
            start=_safe_int(row.get("start")),
            duplicate_count=type_surface_counts.get((entity_type, normalized), 0),
            must_keep=must_keep,
        )
        if must_keep:
            must_keep_scored.append((score, index))
        else:
            optional_scored.append((score, index))

    must_keep_indexes = [index for _, index in sorted(must_keep_scored)]
    remaining = max(0, int(max_items or 0) - len(must_keep_indexes))
    optional_indexes = [index for _, index in sorted(optional_scored)[:remaining]]
    return [*must_keep_indexes, *optional_indexes]


def _selected_indexes_for_result_rows(rows: list[Any], *, max_items: int) -> list[int]:
    as_dicts: list[dict[str, Any]] = []
    for item in rows:
        as_dicts.append(
            {
                "type": str(getattr(item, "entity_type", "") or ""),
                "text": str(getattr(item, "text", "") or ""),
                "start": int(getattr(item, "start", -1) or -1),
                "end": int(getattr(item, "end", -1) or -1),
                "score": float(getattr(item, "score", 0.0) or 0.0),
                "source": str(getattr(item, "source", "") or ""),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
            }
        )
    return _selected_indexes_for_dict_rows(as_dicts, max_items=max_items)


def _rule_drop(entity_type: str, text: str, normalized: str) -> bool:
    if not normalized:
        return True
    if entity_type in FORMAT_OR_NON_SEMANTIC_TYPES:
        return True
    if entity_type not in HIGH_VALUE_TYPES:
        return True
    if normalized in NON_SUBJECT_TABLE_TERMS:
        return True
    if is_identity_reference_term(normalized) or is_position_title(normalized):
        return True
    if entity_type in {"ORGANIZATION", "GOVERNMENT"} and is_generic_organization_term(normalized):
        return True
    if len(normalized) > 40 and re.search(r"[。；;，,]", text):
        return True
    return False


def _priority_for_row(
    *,
    entity_type: str,
    text: str,
    normalized: str,
    source: str,
    start: int,
    duplicate_count: int,
    must_keep: bool = False,
) -> tuple[int, int, int, int]:
    risk = 50
    if must_keep:
        risk -= 40
    if source == "qwen_fragment_review":
        risk -= 20
    if entity_type in {"ORGANIZATION", "GOVERNMENT"} and 2 <= len(normalized) <= 6:
        risk -= 15
    if entity_type == "PERSON" and 2 <= len(normalized) <= 4:
        risk -= 8
    if duplicate_count > 1:
        risk -= 5
    if source in MODEL_SOURCE_NAMES:
        risk -= 3
    return (risk, start if start >= 0 else 10**12, -len(normalized), 0)


def _dedupe_key(row: dict[str, Any], entity_type: str, normalized: str, *, force_span: bool = False) -> tuple[str, str, str]:
    metadata = dict(row.get("metadata") or {})
    if force_span:
        occurrence = str(metadata.get("subject_ledger_occurrence_id") or "").strip()
        if occurrence:
            return (entity_type, "ledger_occurrence", occurrence)
        return (entity_type, "span", f"{_safe_int(row.get('start'))}:{_safe_int(row.get('end'))}:{normalized}")
    canonical = str(row.get("canonical_key") or metadata.get("canonical_key") or "").strip()
    if canonical:
        return (entity_type, "canonical", canonical)
    return (entity_type, "surface", normalized)


def _must_keep_for_review(row: dict[str, Any]) -> bool:
    metadata = dict(row.get("metadata") or {})
    review_statuses = {
        "ambiguous_short_subject",
        "unresolved_alias",
        "alias_without_anchor",
        "weak_reference",
        "weak_identity_edge",
        "hard_conflict",
    }
    if str(metadata.get("subject_ledger_status") or "") in review_statuses:
        return True
    if str(metadata.get("subject_ledger_subject_status") or "") in review_statuses:
        return True
    if metadata.get("subject_ledger_edge_id") or metadata.get("subject_ledger_occurrence_id"):
        return True
    if metadata.get("requires_manual_review"):
        return True
    rule_first = metadata.get("rule_first")
    if isinstance(rule_first, dict):
        if canonical_default_entity_type(_entity_type(row), str(row.get("text") or "")) in DEFAULT_SUBJECT_TYPES:
            return True
        if str(rule_first.get("action") or "") == "review":
            return True
        if str(rule_first.get("risk_level") or "") == "high":
            return True
    if metadata.get("candidate_types") or metadata.get("boundary_repaired_from"):
        return True
    if metadata.get("validators_failed"):
        return True
    return False


def _entity_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or row.get("entity_type") or "").strip().upper()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1
