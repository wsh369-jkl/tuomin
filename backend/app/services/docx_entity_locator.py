"""Map recognized entities back to DOCX visible-text units."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def annotate_entities_with_docx_units(
    entities: Iterable[Dict[str, Any]],
    *,
    source_structure: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attach DOCX unit provenance and rewrite-safety metadata to entities."""
    entity_list = [dict(entity) for entity in entities if isinstance(entity, dict)]
    units = _extract_docx_units(source_structure)
    if not units:
        return entity_list, {
            "docx_entity_locator_enabled": False,
            "docx_entity_count": len(entity_list),
            "docx_entity_with_unit_count": 0,
            "docx_entity_without_unit_count": 0,
            "docx_entity_unrewritable_count": 0,
            "docx_entity_in_review_required_story_count": 0,
            "docx_entity_range_crosses_virtual_fragment_count": 0,
            "docx_entity_in_unknown_part_count": 0,
            "docx_entity_quality_flags": [],
        }

    unit_ranges = [
        unit
        for unit in units
        if _coerce_int(unit.get("end"), -1) > _coerce_int(unit.get("start"), -1) >= 0
    ]
    metadata = {
        "docx_entity_locator_enabled": True,
        "docx_entity_count": len(entity_list),
        "docx_entity_with_unit_count": 0,
        "docx_entity_without_unit_count": 0,
        "docx_entity_unrewritable_count": 0,
        "docx_entity_in_review_required_story_count": 0,
        "docx_entity_range_crosses_virtual_fragment_count": 0,
        "docx_entity_in_unknown_part_count": 0,
        "docx_entity_quality_flags": [],
        "docx_entity_problem_samples": [],
    }
    for entity in entity_list:
        start = _coerce_int(entity.get("start"), -1)
        end = _coerce_int(entity.get("end"), -1)
        unit = _find_containing_unit(unit_ranges, start=start, end=end)
        entity_metadata = dict(entity.get("metadata") or {})
        if unit is None:
            entity_metadata["docx_rewritable"] = False
            entity_metadata["docx_unrewritable_reason"] = "entity_without_docx_unit"
            entity["metadata"] = entity_metadata
            metadata["docx_entity_without_unit_count"] += 1
            metadata["docx_entity_unrewritable_count"] += 1
            _append_problem_sample(metadata, entity, "entity_without_docx_unit")
            continue

        unit_start = _coerce_int(unit.get("start"), 0)
        local_start = start - unit_start
        local_end = end - unit_start
        rewrite_policy = str(unit.get("rewrite_policy") or "exact")
        fragment_count = _coerce_int(unit.get("fragment_count"), 0)
        rewritable = rewrite_policy == "exact" and fragment_count > 0
        reason = ""
        fragment_status = _entity_fragment_status(unit, start=start, end=end)
        if fragment_status == "crosses_virtual_fragment":
            rewritable = False
            reason = "entity_range_crosses_virtual_fragment"
            metadata["docx_entity_range_crosses_virtual_fragment_count"] += 1
        elif fragment_status == "no_real_fragment":
            rewritable = False
            reason = "entity_without_real_docx_fragment"
        if rewrite_policy != "exact":
            rewritable = False
            if not reason:
                reason = f"rewrite_policy:{rewrite_policy}"
            metadata["docx_entity_in_review_required_story_count"] += 1

        container_type = str(unit.get("container_type") or "")
        part_name = str(unit.get("part_name") or "")
        flags = list(unit.get("flags") or [])
        if container_type in {"xml_text"} or "unknown" in part_name.lower():
            metadata["docx_entity_in_unknown_part_count"] += 1
            if not reason:
                reason = "unknown_docx_part"
            rewritable = False

        entity_metadata.update(
            {
                "docx_unit_id": unit.get("unit_id"),
                "docx_part_name": part_name,
                "docx_container_type": container_type,
                "docx_rewrite_policy": rewrite_policy,
                "docx_unit_flags": flags,
                "docx_unit_start": unit_start,
                "docx_unit_end": _coerce_int(unit.get("end"), 0),
                "docx_unit_local_start": local_start,
                "docx_unit_local_end": local_end,
                "docx_rewritable": bool(rewritable),
            }
        )
        if reason:
            entity_metadata["docx_unrewritable_reason"] = reason
        else:
            entity_metadata.pop("docx_unrewritable_reason", None)
        entity["metadata"] = entity_metadata
        metadata["docx_entity_with_unit_count"] += 1
        if not rewritable:
            metadata["docx_entity_unrewritable_count"] += 1
            _append_problem_sample(metadata, entity, reason or "docx_entity_unrewritable")

    flags: List[str] = []
    if metadata["docx_entity_without_unit_count"] > 0:
        flags.append("docx_entity_without_unit")
    if metadata["docx_entity_unrewritable_count"] > 0:
        flags.append("docx_entity_unrewritable")
    if metadata["docx_entity_in_review_required_story_count"] > 0:
        flags.append("docx_entity_review_required_story")
    if metadata["docx_entity_range_crosses_virtual_fragment_count"] > 0:
        flags.append("docx_entity_range_crosses_virtual_fragment")
    if metadata["docx_entity_in_unknown_part_count"] > 0:
        flags.append("docx_entity_unknown_part")
    metadata["docx_entity_quality_flags"] = flags
    return entity_list, metadata


def _extract_docx_units(source_structure: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(source_structure, dict):
        return []
    raw_units = source_structure.get("docx_text_units")
    if isinstance(raw_units, list):
        return [dict(unit) for unit in raw_units if isinstance(unit, dict)]
    units: List[Dict[str, Any]] = []
    pages = source_structure.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict) and isinstance(page.get("units"), list):
                units.extend(dict(unit) for unit in page["units"] if isinstance(unit, dict))
    return units


def _find_containing_unit(
    units: List[Dict[str, Any]],
    *,
    start: int,
    end: int,
) -> Dict[str, Any] | None:
    if start < 0 or end <= start:
        return None
    candidates = [
        unit
        for unit in units
        if _coerce_int(unit.get("start"), -1) <= start
        and end <= _coerce_int(unit.get("end"), -1)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda unit: (
            _coerce_int(unit.get("end"), 0) - _coerce_int(unit.get("start"), 0),
            _coerce_int(unit.get("order_index"), 0),
        ),
    )


def _entity_fragment_status(unit: Dict[str, Any], *, start: int, end: int) -> str:
    fragments = unit.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        return "unknown"
    touched = [
        fragment
        for fragment in fragments
        if isinstance(fragment, dict)
        and start < _coerce_int(fragment.get("end"), -1)
        and end > _coerce_int(fragment.get("start"), -1)
    ]
    if not touched:
        return "no_real_fragment"
    if any(bool(fragment.get("virtual")) for fragment in touched):
        return "crosses_virtual_fragment"
    if not any(not bool(fragment.get("virtual")) for fragment in touched):
        return "no_real_fragment"
    return "exact"


def _append_problem_sample(metadata: Dict[str, Any], entity: Dict[str, Any], reason: str) -> None:
    samples = metadata.setdefault("docx_entity_problem_samples", [])
    if len(samples) >= 20:
        return
    samples.append(
        {
            "text": str(entity.get("text") or "")[:80],
            "type": str(entity.get("type") or ""),
            "start": _coerce_int(entity.get("start"), -1),
            "end": _coerce_int(entity.get("end"), -1),
            "reason": reason,
        }
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
