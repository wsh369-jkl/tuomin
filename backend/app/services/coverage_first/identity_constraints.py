"""Build passive identity constraints before final directory compilation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.services.lowmem_entity_utils import (
    is_org_like_text,
    is_probable_person,
    looks_like_organization_short_name,
    normalize_entity_text,
)


def build_identity_constraints(candidate_ledger: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in candidate_ledger.get("entries") or [] if isinstance(item, dict)]
    constraints: list[dict[str, Any]] = []
    constraints.extend(_same_surface_type_conflicts(candidates))
    constraints.extend(_same_surface_must_links(candidates))
    constraints.extend(_org_alias_maybe_links(candidates))
    constraints.extend(_person_org_type_conflicts(candidates))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for constraint in constraints:
        left = str(constraint.get("left_candidate_id") or "")
        right = str(constraint.get("right_candidate_id") or "")
        if left > right:
            left, right = right, left
            constraint = {**constraint, "left_candidate_id": left, "right_candidate_id": right}
        key = (str(constraint.get("constraint_type") or ""), left, right, str(constraint.get("evidence") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(constraint)

    counts: dict[str, int] = {}
    for constraint in deduped:
        kind = str(constraint.get("constraint_type") or "")
        counts[kind] = counts.get(kind, 0) + 1
    unresolved_count = sum(1 for item in deduped if item.get("constraint_type") in {"maybe_link", "type_conflict"})
    return {
        "constraints": deduped,
        "summary": {
            "constraint_count": len(deduped),
            "must_link_count": counts.get("must_link", 0),
            "cannot_link_count": counts.get("cannot_link", 0),
            "maybe_link_count": counts.get("maybe_link", 0),
            "type_conflict_count": counts.get("type_conflict", 0),
            "unresolved_constraint_count": unresolved_count,
            "constraint_type_counts": dict(sorted(counts.items())),
        },
    }


def _same_surface_type_conflicts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        normalized = str(candidate.get("normalized_text") or normalize_entity_text(candidate.get("text"))).strip()
        if normalized:
            by_surface[normalized].append(candidate)
    constraints: list[dict[str, Any]] = []
    for normalized, rows in by_surface.items():
        types = {str(row.get("entity_type") or "") for row in rows}
        if len(types) <= 1:
            continue
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                if left.get("entity_type") == right.get("entity_type"):
                    continue
                constraints.append(_constraint("type_conflict", left, right, f"same_surface:{normalized}"))
    return constraints


def _same_surface_must_links(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type_surface: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        entity_type = str(candidate.get("entity_type") or "")
        normalized = str(candidate.get("normalized_text") or normalize_entity_text(candidate.get("text"))).strip()
        if entity_type and normalized:
            by_type_surface[(entity_type, normalized)].append(candidate)
    constraints: list[dict[str, Any]] = []
    for (_, normalized), rows in by_type_surface.items():
        if len(rows) <= 1:
            continue
        anchor = rows[0]
        for candidate in rows[1:]:
            constraints.append(_constraint("must_link", anchor, candidate, f"same_type_surface:{normalized}"))
    return constraints


def _org_alias_maybe_links(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    orgs = [item for item in candidates if str(item.get("entity_type") or "") in {"ORGANIZATION", "COMPANY_NAME", "ALIAS"}]
    constraints: list[dict[str, Any]] = []
    for left_index, left in enumerate(orgs):
        left_norm = str(left.get("normalized_text") or normalize_entity_text(left.get("text"))).strip()
        if not left_norm:
            continue
        for right in orgs[left_index + 1 :]:
            right_norm = str(right.get("normalized_text") or normalize_entity_text(right.get("text"))).strip()
            if not right_norm or left_norm == right_norm:
                continue
            longer, shorter = (left_norm, right_norm) if len(left_norm) >= len(right_norm) else (right_norm, left_norm)
            if len(shorter) < 2 or len(longer) < 4 or shorter not in longer:
                continue
            if looks_like_organization_short_name(shorter) or is_org_like_text(longer):
                constraints.append(_constraint("maybe_link", left, right, f"full_short_alias:{longer}<->{shorter}"))
    return constraints


def _person_org_type_conflicts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate.get("normalized_text") or normalize_entity_text(candidate.get("text"))).strip()
        entity_type = str(candidate.get("entity_type") or "")
        if not text:
            continue
        if entity_type == "PERSON" and is_org_like_text(text):
            constraints.append(_self_constraint("type_conflict", candidate, "person_candidate_looks_like_organization"))
        elif entity_type in {"ORGANIZATION", "COMPANY_NAME"} and is_probable_person(text) and not is_org_like_text(text):
            constraints.append(_self_constraint("type_conflict", candidate, "organization_candidate_looks_like_person"))
    return constraints


def _constraint(kind: str, left: dict[str, Any], right: dict[str, Any], evidence: str) -> dict[str, Any]:
    return {
        "constraint_type": kind,
        "left_candidate_id": left.get("candidate_id"),
        "right_candidate_id": right.get("candidate_id"),
        "evidence": evidence,
    }


def _self_constraint(kind: str, candidate: dict[str, Any], evidence: str) -> dict[str, Any]:
    return {
        "constraint_type": kind,
        "left_candidate_id": candidate.get("candidate_id"),
        "right_candidate_id": candidate.get("candidate_id"),
        "evidence": evidence,
    }
