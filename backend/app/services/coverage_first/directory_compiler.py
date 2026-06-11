"""Passive final directory compilation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


SUBJECT_TYPES = {"PERSON", "PERSON_NAME", "ORGANIZATION", "COMPANY_NAME", "ALIAS", "BANK_NAME", "ACCOUNT_NAME", "ADDRESS", "LOCATION"}
TYPE_PRIORITY = {
    "ORGANIZATION": 1,
    "COMPANY_NAME": 1,
    "PERSON": 2,
    "PERSON_NAME": 2,
    "ALIAS": 3,
    "BANK_NAME": 4,
    "ACCOUNT_NAME": 5,
    "ADDRESS": 6,
    "LOCATION": 6,
}
REPLACEMENT_PREFIX = {
    "ORGANIZATION": "公司",
    "COMPANY_NAME": "公司",
    "PERSON": "人员",
    "PERSON_NAME": "人员",
    "ALIAS": "简称",
    "BANK_NAME": "银行",
    "ACCOUNT_NAME": "户名",
    "ADDRESS": "地址",
    "LOCATION": "地址",
}


def build_directory_compile_summary(
    *,
    candidate_ledger: dict[str, Any],
    identity_constraints: dict[str, Any],
) -> dict[str, Any]:
    candidates = [item for item in candidate_ledger.get("entries") or [] if isinstance(item, dict)]
    subject_candidates = [
        item for item in candidates if str(item.get("entity_type") or "") in SUBJECT_TYPES
    ]
    rows = _compile_directory_rows(subject_candidates, identity_constraints)
    unresolved_identity = int((identity_constraints.get("summary") or {}).get("unresolved_constraint_count") or 0)
    missing_evidence_count = sum(1 for item in subject_candidates if not item.get("text") or int(item.get("end") or 0) <= int(item.get("start") or 0))
    replacement_counter = Counter(str(row.get("replacement") or "") for row in rows if row.get("replacement"))
    replacement_conflict_count = sum(1 for count in replacement_counter.values() if count > 1)
    return {
        "directory_rows": rows,
        "summary": {
            "compiled_subject_count": len(rows),
            "candidate_subject_count": len(subject_candidates),
            "unresolved_identity_constraint_count": unresolved_identity,
            "missing_evidence_subject_count": missing_evidence_count,
            "replacement_conflict_count": replacement_conflict_count,
            "compilable": unresolved_identity == 0 and missing_evidence_count == 0 and replacement_conflict_count == 0,
        },
    }


def _compile_directory_rows(
    candidates: list[dict[str, Any]],
    identity_constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(item.get("candidate_id") or ""): item for item in candidates if str(item.get("candidate_id") or "")}
    parent = {candidate_id: candidate_id for candidate_id in by_id}

    def find(value: str) -> str:
        while parent.get(value, value) != value:
            parent[value] = parent.get(parent[value], parent[value])
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            return
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for constraint in identity_constraints.get("constraints") or []:
        if not isinstance(constraint, dict) or constraint.get("constraint_type") != "must_link":
            continue
        union(str(constraint.get("left_candidate_id") or ""), str(constraint.get("right_candidate_id") or ""))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, candidate in by_id.items():
        groups[find(candidate_id)].append(candidate)

    ordered_groups = sorted(groups.values(), key=_group_sort_key)
    type_counters: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(ordered_groups, start=1):
        subject_type = _canonical_type(group)
        type_counters[subject_type] += 1
        surfaces = _surfaces(group)
        canonical_text = surfaces[0] if surfaces else ""
        replacement = f"{REPLACEMENT_PREFIX.get(subject_type, '主体')}{type_counters[subject_type]}"
        rows.append(
            {
                "subject_id": f"CFD{index:05d}",
                "subject_type": subject_type,
                "canonical_text": canonical_text,
                "surfaces": surfaces,
                "candidate_ids": sorted(str(item.get("candidate_id") or "") for item in group if item.get("candidate_id")),
                "occurrence_count": len(group),
                "replacement": replacement,
                "status": "compiled",
            }
        )
    return rows


def _group_sort_key(group: list[dict[str, Any]]) -> tuple[int, int, int, str]:
    subject_type = _canonical_type(group)
    start = min(int(item.get("start") or 0) for item in group) if group else 0
    canonical = (_surfaces(group) or [""])[0]
    return (TYPE_PRIORITY.get(subject_type, 99), start, -len(group), canonical)


def _canonical_type(group: list[dict[str, Any]]) -> str:
    counts = Counter(str(item.get("entity_type") or "") for item in group)
    if not counts:
        return "UNKNOWN"
    return sorted(counts, key=lambda value: (-counts[value], TYPE_PRIORITY.get(value, 99), value))[0]


def _surfaces(group: list[dict[str, Any]]) -> list[str]:
    counter = Counter(str(item.get("text") or "").strip() for item in group if str(item.get("text") or "").strip())
    return [
        value
        for value, _ in sorted(counter.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    ][:8]
