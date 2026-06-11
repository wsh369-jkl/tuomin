"""Passive coverage-first pipeline orchestration."""

from __future__ import annotations

from typing import Any, Iterable

from app.core.recognizer_base import RecognizerResult
from app.services.coverage_first.candidate_ledger import build_candidate_ledger
from app.services.coverage_first.coverage_matcher import match_coverage
from app.services.coverage_first.coverage_plan import build_coverage_plan
from app.services.coverage_first.deterministic_resolver import build_deterministic_resolutions
from app.services.coverage_first.directory_compiler import build_directory_compile_summary
from app.services.coverage_first.document_graph import build_document_graph
from app.services.coverage_first.identity_constraints import build_identity_constraints
from app.services.coverage_first.model_arbitration import build_model_arbitration_queue
from app.services.coverage_first.post_rewrite_verifier import build_post_rewrite_verifier_summary
from app.services.coverage_first.replacement_compiler import build_replacement_compile_summary
from app.services.coverage_first.rewrite_ledger import build_rewrite_ledger_summary


def build_passive_coverage_first_metadata(
    *,
    source_structure: dict[str, Any] | None,
    candidates: Iterable[RecognizerResult],
    source_text: str | None = None,
) -> dict[str, Any]:
    document_graph = build_document_graph(source_structure)
    coverage_plan = build_coverage_plan(document_graph)
    candidate_ledger = build_candidate_ledger(candidates, document_graph)
    coverage_match = match_coverage(coverage_plan, candidate_ledger)
    identity_constraints = build_identity_constraints(candidate_ledger)
    deterministic_resolutions = build_deterministic_resolutions(
        candidate_ledger=candidate_ledger,
        identity_constraints=identity_constraints,
    )
    model_arbitration_queue = build_model_arbitration_queue(
        coverage_match=coverage_match,
        identity_constraints=identity_constraints,
    )
    directory_compile = build_directory_compile_summary(
        candidate_ledger=candidate_ledger,
        identity_constraints=identity_constraints,
    )
    replacement_compile = build_replacement_compile_summary(
        candidate_ledger,
        directory_compile=directory_compile,
    )
    rewrite_ledger = build_rewrite_ledger_summary(
        candidate_ledger=candidate_ledger,
        directory_compile=directory_compile,
        source_text=source_text,
    )
    post_rewrite_verifier = build_post_rewrite_verifier_summary(
        coverage_match=coverage_match,
        identity_constraints=identity_constraints,
        directory_compile=directory_compile,
        replacement_compile=replacement_compile,
        rewrite_ledger=rewrite_ledger,
        model_arbitration_queue=model_arbitration_queue,
    )

    document_summary = dict(document_graph.summary)
    plan_summary = dict(coverage_plan.get("summary") or {})
    candidate_summary = dict(candidate_ledger.get("summary") or {})
    match_summary = dict(coverage_match.get("summary") or {})
    identity_summary = dict(identity_constraints.get("summary") or {})
    deterministic_summary = dict(deterministic_resolutions.get("summary") or {})
    arbitration_summary = dict(model_arbitration_queue.get("summary") or {})
    directory_summary = dict(directory_compile.get("summary") or {})
    replacement_summary = dict(replacement_compile.get("summary") or {})
    rewrite_summary = dict(rewrite_ledger.get("summary") or {})
    post_rewrite_summary = dict(post_rewrite_verifier.get("summary") or {})
    high_priority_unit_ids = _obligation_unit_ids(
        coverage_plan,
        priorities={"high"},
    )
    review_priority_unit_ids = _obligation_unit_ids(
        coverage_plan,
        priorities={"high", "medium"},
    )
    return {
        "coverage_first_enabled": bool(document_graph.enabled),
        "coverage_first_document_graph": document_summary,
        "coverage_first_coverage_plan": plan_summary,
        "coverage_first_candidate_ledger": candidate_summary,
        "coverage_first_coverage_match": match_summary,
        "coverage_first_identity_constraints": identity_summary,
        "coverage_first_deterministic_resolutions": deterministic_summary,
        "coverage_first_model_arbitration_queue": arbitration_summary,
        "coverage_first_directory_compile": directory_summary,
        "coverage_first_directory_rows": list(directory_compile.get("directory_rows") or [])[:80],
        "coverage_first_replacement_compile": replacement_summary,
        "coverage_first_rewrite_ledger": rewrite_summary,
        "coverage_first_rewrite_entries": list(rewrite_ledger.get("rewrite_entries") or []),
        "coverage_first_rewrite_entry_sample": list(rewrite_ledger.get("rewrite_entries") or [])[:80],
        "coverage_first_post_rewrite_verifier": post_rewrite_summary,
        "coverage_first_unit_count": int(document_summary.get("unit_count") or 0),
        "coverage_first_obligation_count": int(plan_summary.get("obligation_count") or 0),
        "coverage_first_high_priority_obligation_count": int(
            plan_summary.get("high_priority_obligation_count") or 0
        ),
        "coverage_first_candidate_count": int(candidate_summary.get("candidate_count") or 0),
        "coverage_first_matched_candidate_count": int(match_summary.get("matched_candidate_count") or 0),
        "coverage_first_uncovered_required_obligation_count": int(
            match_summary.get("uncovered_required_obligation_count") or 0
        ),
        "coverage_first_unrewritable_obligation_count": int(
            match_summary.get("unrewritable_obligation_count") or 0
        ),
        "coverage_first_identity_constraint_count": int(identity_summary.get("constraint_count") or 0),
        "coverage_first_unresolved_identity_constraint_count": int(
            identity_summary.get("unresolved_constraint_count") or 0
        ),
        "coverage_first_hard_identity_conflict_count": int(
            identity_summary.get("type_conflict_count") or 0
        ),
        "coverage_first_deterministic_resolution_count": int(
            deterministic_summary.get("resolution_count") or 0
        ),
        "coverage_first_model_arbitration_task_count": int(arbitration_summary.get("task_count") or 0),
        "coverage_first_hard_arbitration_task_count": int(arbitration_summary.get("hard_task_count") or 0),
        "coverage_first_directory_compilable": bool(directory_summary.get("compilable")),
        "coverage_first_compiled_subject_count": int(directory_summary.get("compiled_subject_count") or 0),
        "coverage_first_directory_replacement_conflict_count": int(
            directory_summary.get("replacement_conflict_count") or 0
        ),
        "coverage_first_prewrite_verification_passed": bool(rewrite_summary.get("prewrite_verification_passed")),
        "coverage_first_rewrite_entry_count": int(rewrite_summary.get("rewrite_entry_count") or 0),
        "coverage_first_required_rewrite_entry_count": int(
            rewrite_summary.get("required_rewrite_entry_count") or 0
        ),
        "coverage_first_blocked_rewrite_entry_count": int(
            rewrite_summary.get("blocked_rewrite_entry_count") or 0
        ),
        "coverage_first_post_verifier_ready_to_export": bool(
            post_rewrite_summary.get("post_verifier_ready_to_export")
        ),
        "coverage_first_post_verifier_blocking_issue_count": int(
            post_rewrite_summary.get("blocking_issue_count") or 0
        ),
        "coverage_first_status_counts": dict(match_summary.get("status_counts") or {}),
        "coverage_first_high_priority_unit_ids": high_priority_unit_ids,
        "coverage_first_review_priority_unit_ids": review_priority_unit_ids,
    }


def _obligation_unit_ids(coverage_plan: dict[str, Any], *, priorities: set[str]) -> list[str]:
    unit_ids: list[str] = []
    seen: set[str] = set()
    for obligation in coverage_plan.get("obligations") or []:
        if not isinstance(obligation, dict):
            continue
        if str(obligation.get("priority") or "") not in priorities:
            continue
        for unit_id in obligation.get("unit_ids") or []:
            value = str(unit_id or "").strip()
            if value and value not in seen:
                seen.add(value)
                unit_ids.append(value)
    return unit_ids
