"""Passive deterministic resolutions for coverage-first ledgers."""

from __future__ import annotations

from typing import Any

from app.services.lowmem_entity_utils import (
    P0_ENTITY_TYPES,
    is_identity_reference_term,
    is_position_title,
)


def build_deterministic_resolutions(
    *,
    candidate_ledger: dict[str, Any],
    identity_constraints: dict[str, Any],
) -> dict[str, Any]:
    candidates = [item for item in candidate_ledger.get("entries") or [] if isinstance(item, dict)]
    constraints = [item for item in identity_constraints.get("constraints") or [] if isinstance(item, dict)]
    resolutions: list[dict[str, Any]] = []
    for candidate in candidates:
        entity_type = str(candidate.get("entity_type") or "")
        text = str(candidate.get("text") or "")
        candidate_id = str(candidate.get("candidate_id") or "")
        if entity_type in P0_ENTITY_TYPES:
            resolutions.append(_resolution("confirm_candidate", candidate_id, "format_entity_rule_stable"))
        elif entity_type in {"PERSON", "ORGANIZATION", "COMPANY_NAME", "ALIAS"} and (
            is_identity_reference_term(text) or is_position_title(text)
        ):
            resolutions.append(_resolution("reject_candidate", candidate_id, "role_or_title_not_subject"))

    for constraint in constraints:
        if constraint.get("constraint_type") == "must_link":
            resolutions.append(
                {
                    "operation": "must_link",
                    "target_ids": [constraint.get("left_candidate_id"), constraint.get("right_candidate_id")],
                    "evidence": constraint.get("evidence"),
                    "confidence": "strong",
                }
            )

    operation_counts: dict[str, int] = {}
    for resolution in resolutions:
        operation = str(resolution.get("operation") or "")
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
    return {
        "resolutions": resolutions,
        "summary": {
            "resolution_count": len(resolutions),
            "operation_counts": dict(sorted(operation_counts.items())),
        },
    }


def _resolution(operation: str, candidate_id: str, evidence: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "target_ids": [candidate_id],
        "evidence": evidence,
        "confidence": "strong",
    }
