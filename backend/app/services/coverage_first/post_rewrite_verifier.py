"""Pre-export and post-rewrite quality verifier for coverage-first artifacts."""

from __future__ import annotations

from typing import Any


def build_post_rewrite_verifier_summary(
    *,
    coverage_match: dict[str, Any],
    identity_constraints: dict[str, Any],
    directory_compile: dict[str, Any],
    replacement_compile: dict[str, Any],
    rewrite_ledger: dict[str, Any],
    model_arbitration_queue: dict[str, Any],
) -> dict[str, Any]:
    """Summarize blocking conditions before the file rewrite/export phase."""
    match_summary = dict(coverage_match.get("summary") or {})
    identity_summary = dict(identity_constraints.get("summary") or {})
    directory_summary = dict(directory_compile.get("summary") or {})
    replacement_summary = dict(replacement_compile.get("summary") or {})
    rewrite_summary = dict(rewrite_ledger.get("summary") or {})
    arbitration_summary = dict(model_arbitration_queue.get("summary") or {})
    issue_counts = {
        "uncovered_required_obligation": _as_int(match_summary.get("uncovered_required_obligation_count")),
        "unrewritable_obligation": _as_int(match_summary.get("unrewritable_obligation_count")),
        "hard_identity_conflict": _as_int(identity_summary.get("type_conflict_count")),
        "hard_arbitration_task": _as_int(arbitration_summary.get("hard_task_count")),
        "directory_replacement_conflict": _as_int(directory_summary.get("replacement_conflict_count")),
        "subject_multi_replacement": _as_int(replacement_summary.get("subject_multi_replacement_count")),
        "blocked_rewrite_entry": _as_int(rewrite_summary.get("blocked_rewrite_entry_count")),
        "missing_directory_candidate": _as_int(rewrite_summary.get("missing_directory_candidate_count")),
    }
    issue_counts = {key: value for key, value in issue_counts.items() if value > 0}
    ready = not issue_counts and bool(directory_summary.get("compilable")) and bool(
        rewrite_summary.get("prewrite_verification_passed")
    )
    return {
        "summary": {
            "post_verifier_ready_to_export": ready,
            "blocking_issue_count": sum(issue_counts.values()),
            "blocking_issue_counts": issue_counts,
            "required_rewrite_entry_count": _as_int(rewrite_summary.get("required_rewrite_entry_count")),
            "blocked_rewrite_entry_count": _as_int(rewrite_summary.get("blocked_rewrite_entry_count")),
            "compiled_subject_count": _as_int(directory_summary.get("compiled_subject_count")),
        }
    }


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
