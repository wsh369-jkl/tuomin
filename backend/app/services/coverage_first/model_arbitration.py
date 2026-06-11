"""Build a passive queue of unresolved model arbitration tasks."""

from __future__ import annotations

from typing import Any


def build_model_arbitration_queue(
    *,
    coverage_match: dict[str, Any],
    identity_constraints: dict[str, Any],
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for obligation in coverage_match.get("obligations") or []:
        if not isinstance(obligation, dict):
            continue
        status = str(obligation.get("status") or "")
        priority = str(obligation.get("priority") or "")
        if status == "uncovered" and priority in {"high", "medium"}:
            tasks.append(
                {
                    "task_id": f"MA{len(tasks) + 1:05d}",
                    "task_type": "coverage_gap",
                    "severity": "hard" if priority == "high" else "medium",
                    "obligation_ids": [obligation.get("obligation_id")],
                    "candidate_ids": [],
                    "constraint_ids": [],
                    "allowed_operations": ["add_missing_candidate", "no_sensitive_entity", "manual_review"],
                }
            )
        elif status == "unrewritable":
            tasks.append(
                {
                    "task_id": f"MA{len(tasks) + 1:05d}",
                    "task_type": "rewrite_addressability",
                    "severity": "hard",
                    "obligation_ids": [obligation.get("obligation_id")],
                    "candidate_ids": obligation.get("matched_candidate_ids") or [],
                    "constraint_ids": [],
                    "allowed_operations": ["manual_review"],
                }
            )
    for index, constraint in enumerate(identity_constraints.get("constraints") or [], start=1):
        if not isinstance(constraint, dict):
            continue
        constraint_type = str(constraint.get("constraint_type") or "")
        if constraint_type not in {"maybe_link", "type_conflict", "boundary_conflict"}:
            continue
        tasks.append(
            {
                "task_id": f"MA{len(tasks) + 1:05d}",
                "task_type": constraint_type,
                "severity": "hard" if constraint_type == "type_conflict" else "medium",
                "obligation_ids": [],
                "candidate_ids": [constraint.get("left_candidate_id"), constraint.get("right_candidate_id")],
                "constraint_ids": [f"IC{index:05d}"],
                "allowed_operations": _allowed_operations_for_constraint(constraint_type),
            }
        )

    type_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for task in tasks:
        task_type = str(task.get("task_type") or "")
        severity = str(task.get("severity") or "")
        type_counts[task_type] = type_counts.get(task_type, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    return {
        "tasks": tasks,
        "summary": {
            "task_count": len(tasks),
            "hard_task_count": severity_counts.get("hard", 0),
            "medium_task_count": severity_counts.get("medium", 0),
            "task_type_counts": dict(sorted(type_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
        },
    }


def _allowed_operations_for_constraint(constraint_type: str) -> list[str]:
    if constraint_type == "maybe_link":
        return ["must_link", "cannot_link", "manual_review"]
    if constraint_type == "type_conflict":
        return ["change_type", "reject_candidate", "manual_review"]
    if constraint_type == "boundary_conflict":
        return ["change_boundary", "reject_candidate", "manual_review"]
    return ["manual_review"]
