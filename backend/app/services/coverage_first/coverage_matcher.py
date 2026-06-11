"""Match candidate evidence to coverage obligations."""

from __future__ import annotations

from typing import Any


TYPE_ALIASES = {
    "PERSON_NAME": "PERSON",
    "LEGAL_REPRESENTATIVE": "PERSON",
    "CONTACT_PERSON": "PERSON",
    "SIGNATORY": "PERSON",
    "COMPANY_NAME": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "LOCATION": "ADDRESS",
}


def match_coverage(
    coverage_plan: dict[str, Any],
    candidate_ledger: dict[str, Any],
) -> dict[str, Any]:
    obligations = [item for item in coverage_plan.get("obligations") or [] if isinstance(item, dict)]
    candidates = [item for item in candidate_ledger.get("entries") or [] if isinstance(item, dict)]
    matched_candidate_ids: set[str] = set()
    obligation_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for obligation in obligations:
        matches = _matches_for_obligation(obligation, candidates)
        for match in matches:
            candidate_id = str(match.get("candidate_id") or "")
            if candidate_id:
                matched_candidate_ids.add(candidate_id)
        status = _status_for_obligation(obligation, matches)
        status_counts[status] = status_counts.get(status, 0) + 1
        obligation_rows.append(
            {
                "obligation_id": obligation.get("obligation_id"),
                "category": obligation.get("category"),
                "priority": obligation.get("priority"),
                "status": status,
                "matched_candidate_ids": [match.get("candidate_id") for match in matches],
                "expected_entity_types": list(obligation.get("expected_entity_types") or []),
                "rewrite_required": bool(obligation.get("rewrite_required")),
            }
        )

    uncovered_required = sum(
        1
        for row in obligation_rows
        if row["status"] == "uncovered" and row["priority"] in {"high", "medium"}
    )
    unrewritable = sum(1 for row in obligation_rows if row["status"] == "unrewritable")
    return {
        "obligations": obligation_rows,
        "summary": {
            "matched_obligation_count": sum(1 for row in obligation_rows if row["matched_candidate_ids"]),
            "uncovered_required_obligation_count": uncovered_required,
            "unrewritable_obligation_count": unrewritable,
            "matched_candidate_count": len(matched_candidate_ids),
            "status_counts": dict(sorted(status_counts.items())),
        },
    }


def _matches_for_obligation(obligation: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unit_ids = {str(item) for item in obligation.get("unit_ids") or [] if str(item)}
    expected = {_normalize_type(item) for item in obligation.get("expected_entity_types") or [] if str(item)}
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_unit_id = str(candidate.get("unit_id") or "")
        if unit_ids and candidate_unit_id not in unit_ids:
            continue
        candidate_type = _normalize_type(candidate.get("entity_type"))
        if expected and candidate_type not in expected:
            continue
        matches.append(candidate)
    return matches


def _status_for_obligation(obligation: dict[str, Any], matches: list[dict[str, Any]]) -> str:
    if bool(obligation.get("rewrite_required")):
        return "unrewritable"
    if matches:
        if all(str(match.get("source") or "") in {"regex", "custom", "contract_structure_backfill"} for match in matches):
            return "satisfied_by_rule"
        return "satisfied_by_candidate"
    if str(obligation.get("priority") or "") == "low":
        return "not_sensitive_by_rule"
    return "uncovered"


def _normalize_type(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    return TYPE_ALIASES.get(normalized, normalized)
