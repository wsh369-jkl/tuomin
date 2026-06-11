"""One-shot Qwen fragment review worker.

This worker intentionally does not run the primary UIE/NER stages. It receives
the primary-stage entities from a previous process, loads only the review model,
applies Qwen additions/rejections, then exits so macOS can reclaim MLX memory.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.core.pipeline_manager import PipelineManager
from app.core.recognizer_base import RecognizerResult
from app.core.config import desensitize_mode_context, settings
from app.recognizers.high_quality_lowmem_recognizer import HighQualityLowMemoryRecognizer
from app.services.contextual_desensitization_service import ContextualDesensitizationService
from app.services.qwen_fragment_review_service import QwenFragmentReviewService
from app.services.review_input_compactor import compact_recognizer_results_for_review
from app.services.risk_snippet_scheduler import RiskSnippet

_ledger_adjudication_completion = HighQualityLowMemoryRecognizer._ledger_adjudication_completion
SHORT_ORG_PUBLICATION_FILTER_VERSION = "2026-06-10-structure-ambiguous-short-org-v1"


def _result_from_dict(item: Dict[str, Any]) -> Optional[RecognizerResult]:
    try:
        entity_type = str(item.get("type") or item.get("entity_type") or "").strip()
        text = str(item.get("text") or "")
        start = int(item.get("start"))
        end = int(item.get("end"))
        score = float(item.get("score") or 0.0)
        source = str(item.get("source") or "unknown")
    except Exception:
        return None
    if not entity_type or not text or start < 0 or end <= start:
        return None
    metadata = dict(item.get("metadata") if isinstance(item.get("metadata"), dict) else {})
    for metadata_key in (
        "canonical_key",
        "canonical_role",
        "replacement",
        "replacement_method",
        "source_layer",
    ):
        if metadata_key not in metadata and item.get(metadata_key) is not None:
            metadata[metadata_key] = item.get(metadata_key)
    return RecognizerResult(
        entity_type=entity_type,
        start=start,
        end=end,
        score=score,
        text=text,
        source=source,
        metadata=metadata,
    )


def _results_from_payload(items: Iterable[Dict[str, Any]]) -> List[RecognizerResult]:
    results: list[RecognizerResult] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        result = _result_from_dict(item)
        if result is not None:
            results.append(result)
    return results


def _entity_statistics(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for entity in entities:
        entity_type = str(entity.get("type") or "")
        if not entity_type:
            continue
        bucket = stats.setdefault(entity_type, {"count": 0, "examples": []})
        bucket["count"] = int(bucket["count"]) + 1
        examples = bucket.setdefault("examples", [])
        if isinstance(examples, list) and len(examples) < 3:
            examples.append(entity.get("text"))
    return stats


def _structure_short_org_counts(entities: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total": 0,
        "ambiguous": 0,
        "confirmed": 0,
        "review_backed": 0,
    }
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        metadata = entity.get("metadata") if isinstance(entity.get("metadata"), dict) else {}
        source = str(entity.get("source") or "").strip()
        source_layer = str(metadata.get("source_layer") or "").strip()
        trigger = str(metadata.get("trigger") or "").strip()
        if not metadata.get("short_org_candidate"):
            continue
        if source != "rule_organization_context" and not (source_layer == "structure" and "short_org" in trigger):
            continue
        counts["total"] += 1
        status = str(
            metadata.get("subject_ledger_status")
            or metadata.get("subject_ledger_subject_status")
            or ""
        ).strip()
        if status == "ambiguous_short_subject":
            counts["ambiguous"] += 1
        elif status:
            counts["confirmed"] += 1
        if (
            metadata.get("review")
            or str(metadata.get("source_layer") or "").strip().lower() == "llm_review"
            or source in {"qwen_fragment_review", "qwen_heavy_arbitration"}
        ):
            counts["review_backed"] += 1
    return counts


def _validate_and_expand(text: str, results: List[RecognizerResult]) -> List[RecognizerResult]:
    manager = object.__new__(PipelineManager)
    validated = manager._validate_results(results, text)
    expanded = manager._expand_repeated_mentions(validated, text)
    return manager._validate_results(expanded, text)


def _results_from_entity_dicts(items: Iterable[Dict[str, Any]]) -> List[RecognizerResult]:
    return _results_from_payload(items)


def _write_progress(progress_path: str | None, payload: Dict[str, object]) -> None:
    if not progress_path:
        return
    path = Path(progress_path)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _postprocess_final_entities(
    contextual_service: ContextualDesensitizationService,
    text: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    processed = [dict(entity) for entity in entities]
    sort_and_dedupe = getattr(contextual_service, "_sort_and_deduplicate_entities", None)
    apply_hints = getattr(contextual_service, "_apply_metadata_identity_hints", None)
    prune = getattr(contextual_service, "_prune_invalid_entities", None)

    if callable(sort_and_dedupe):
        processed = sort_and_dedupe(processed)
    if callable(apply_hints):
        processed = apply_hints(processed)
    if callable(prune):
        try:
            processed = prune(processed, full_text=text)
        except TypeError:
            processed = prune(processed)
    if callable(sort_and_dedupe):
        processed = sort_and_dedupe(processed)
    return processed


async def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    with desensitize_mode_context(payload.get("desensitize_mode")):
        text = str(payload.get("text") or "")
        progress_path = str(payload.get("progress_path") or "").strip() or None
        source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
        source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
        entities_filter = payload.get("entities_filter")
        if entities_filter is not None and not isinstance(entities_filter, list):
            entities_filter = None

        primary_entities = _results_from_payload(payload.get("entities") or [])
        review_surface_entities = _results_from_payload(payload.get("review_entities") or [])
        if not review_surface_entities:
            review_surface_entities = list(primary_entities)
        primary_entity_input_count = len(primary_entities)
        review_surface_input_count = len(review_surface_entities)
        review_surface_entities = compact_recognizer_results_for_review(review_surface_entities)
        primary_metadata = dict(payload.get("analysis_metadata") or {})
        recognizer = HighQualityLowMemoryRecognizer()
        ordinary_review_budget_getter = getattr(recognizer, "_ordinary_review_budget", None)
        ordinary_review_budget = (
            int(ordinary_review_budget_getter())
            if callable(ordinary_review_budget_getter)
            else max(1, int(settings.MID_REVIEW_MAX_SNIPPETS or settings.REVIEW_MAX_SNIPPETS or 1))
        )

        snippets = recognizer.snippet_scheduler.build_snippets(
            text,
            review_surface_entities,
            source_structure=source_structure,
            max_snippets=ordinary_review_budget,
        )
        review_snippets = recognizer._select_review_snippets_requiring_semantic_recovery(
            snippets,
            review_surface_entities,
        )
        review_trigger_reasons = sorted({snippet.risk_reason for snippet in review_snippets})
        metadata = dict(primary_metadata)
        recognizer.last_rule_first_metadata = dict(primary_metadata)
        stage_counts = dict(metadata.get("stage_counts") or {})
        stage_counts["review_worker_primary_input_count"] = primary_entity_input_count
        stage_counts["review_worker_primary_compacted_count"] = len(primary_entities)
        stage_counts["review_worker_surface_input_count"] = review_surface_input_count
        stage_counts["review_worker_surface_compacted_count"] = len(review_surface_entities)
        quality_flags = list(metadata.get("quality_flags") or [])
        requires_manual_review = bool(metadata.get("requires_manual_review"))
        review_result = None
        review_skipped_reason = None
        qwen_contribution: dict[str, object] = {
            "qwen_raw_candidates": 0,
            "qwen_materialized_entities": 0,
            "qwen_new_entities_after_merge": 0,
            "qwen_confirmed_overlaps": 0,
            "qwen_discarded_entities": 0,
            "qwen_value_level": "not_run",
            "qwen_rejected_entities": 0,
        }

        results = list(primary_entities)
        review_available = bool(recognizer.review_service.installed)
        review_metadata: dict[str, Any] = {}
        final_adjudication = getattr(recognizer, "_final_deterministic_adjudication_decisions", None)
        deterministic_review_decisions = (
            final_adjudication(results)
            if callable(final_adjudication)
            else {"rejections": [], "entities": []}
        )
        deterministic_review_rejections = list(deterministic_review_decisions.get("rejections") or [])
        deterministic_review_entity_rows = list(deterministic_review_decisions.get("entities") or [])
        deterministic_review_entities = []
        if deterministic_review_entity_rows:
            materialize_candidates = getattr(recognizer.review_service, "_materialize_candidates", None)
            materializer = recognizer.review_service if callable(materialize_candidates) else QwenFragmentReviewService()
            deterministic_review_entities = materializer._materialize_candidates(
                text,
                {"entities": deterministic_review_entity_rows},
                RiskSnippet(
                    "final_subject_adjudication",
                    "final_subject:deterministic_worker",
                    0,
                    len(text),
                    text,
                ),
                source_name="review_deterministic_decision",
            )
        if deterministic_review_rejections:
            before_review_count = len(results)
            results = recognizer._apply_review_rejections(results, deterministic_review_rejections)
            deterministic_rejected_count = max(0, before_review_count - len(results))
            stage_counts["final_deterministic_adjudication_rejected"] = deterministic_rejected_count
            if deterministic_rejected_count:
                quality_flags.append("final_deterministic_adjudication_applied")
        if deterministic_review_entities:
            canonicalize_default_results = getattr(recognizer, "_canonicalize_default_results", None)
            results.extend(
                canonicalize_default_results(deterministic_review_entities)
                if callable(canonicalize_default_results)
                else deterministic_review_entities
            )
            stage_counts["final_deterministic_adjudication_entities"] = len(deterministic_review_entities)
            quality_flags.append("final_deterministic_adjudication_entities_applied")
        if settings.ENABLE_QWEN_REVIEW and review_available and review_snippets:
            review_result = await recognizer.review_service.review(
                text,
                review_snippets,
                existing_entities=review_surface_entities,
                max_snippets=ordinary_review_budget,
                progress_callback=lambda payload: _write_progress(progress_path, payload),
            )
            review_metadata = dict(getattr(review_result, "metadata", {}) or {})
            qwen_contribution = recognizer._measure_qwen_contribution(
                review_surface_entities,
                review_result.entities,
                review_result.raw_candidate_count,
            )
            qwen_contribution["qwen_rejected_entities"] = len(review_result.rejected_entities)
            if review_result.rejected_entities:
                before_review_count = len(results)
                results = recognizer._apply_review_rejections(results, review_result.rejected_entities)
                rejected_review_count = max(0, before_review_count - len(results))
                if rejected_review_count:
                    requires_manual_review = True
                    quality_flags.append("review_rejections_applied")
            ledger_decisions = list(review_metadata.get("ledger_conflict_decisions") or [])
            if ledger_decisions:
                results = recognizer._apply_ledger_adjudication_decisions(results, ledger_decisions)
                recognizer._apply_resolved_subject_ledger_metadata(ledger_decisions)
                metadata.update(recognizer.last_rule_first_metadata)
                quality_flags.append("ledger_conflict_adjudication_applied")
                if any(bool(item.get("requires_manual_review")) for item in ledger_decisions):
                    requires_manual_review = True
                    quality_flags.append("ledger_conflict_manual_review_required")
            if review_result.entities:
                results.extend(review_result.entities)
            requires_manual_review = requires_manual_review or review_result.requires_manual_review
            if review_result.error:
                quality_flags.append(f"review_warning:{review_result.error}")
            if not review_result.model_used:
                requires_manual_review = True
                quality_flags.append("review_model_not_used")
        elif settings.ENABLE_QWEN_REVIEW and review_available:
            review_skipped_reason = "no_review_snippets"
        elif settings.ENABLE_QWEN_REVIEW:
            requires_manual_review = True
            quality_flags.append("review_model_missing")
            review_skipped_reason = "review_model_missing"
        else:
            review_skipped_reason = "review_disabled"

        merged = recognizer.merge_service.merge(results)
        stage_counts["primary_review_surface"] = len(review_surface_entities)
        stage_counts["primary_fallback_entities"] = len(primary_entities)
        stage_counts["risk_snippets"] = len(snippets)
        stage_counts["docx_structure_snippets"] = sum(
            1 for snippet in snippets if snippet.risk_reason.startswith("docx_structure:")
        )
        stage_counts["docx_table_cell_snippets"] = sum(
            1 for snippet in snippets if snippet.snippet_type == "docx_table_cell_block"
        )
        stage_counts["review_snippets_selected"] = len(review_snippets)
        stage_counts["docx_structure_snippets_selected"] = sum(
            1 for snippet in review_snippets if str(snippet.risk_reason or "").startswith("docx_structure:")
        )
        stage_counts["docx_table_cell_snippets_selected"] = sum(
            1 for snippet in review_snippets if str(snippet.snippet_type or "") == "docx_table_cell_block"
        )
        stage_counts["qwen_raw_candidates"] = int(qwen_contribution["qwen_raw_candidates"])
        stage_counts["qwen_review"] = len(review_result.entities) if review_result else 0
        stage_counts["qwen_new_after_merge"] = int(qwen_contribution["qwen_new_entities_after_merge"])
        stage_counts["qwen_rejected"] = int(qwen_contribution.get("qwen_rejected_entities") or 0)
        stage_counts["ledger_conflict_decisions"] = int(
            review_metadata.get("ledger_conflict_decision_count") or 0
        ) if review_result else 0
        stage_counts["ledger_conflict_snippets"] = int(
            review_metadata.get("ledger_conflict_snippet_count") or 0
        ) if review_result else sum(
            1
            for snippet in review_snippets
            if str(snippet.snippet_type or "") == "ledger_conflict_adjudication"
            or str(snippet.risk_reason or "").startswith("subject_ledger:")
        )
        if review_result:
            stage_counts["review_scheduled_ledger_snippets"] = int(
                review_metadata.get("review_scheduled_ledger_snippet_count") or 0
            )
            stage_counts["review_scheduled_standard_snippets"] = int(
                review_metadata.get("review_scheduled_standard_snippet_count") or 0
            )
            stage_counts["review_deterministic_rejections"] = int(
                review_metadata.get("review_deterministic_rejection_count") or 0
            )
            stage_counts["review_deterministic_entity_decisions"] = int(
                review_metadata.get("review_deterministic_entity_decision_count") or 0
            )
            stage_counts["review_deterministic_entities"] = int(
                review_metadata.get("review_deterministic_entity_count") or 0
            )
            stage_counts["review_entity_decision_rejections"] = int(
                review_metadata.get("review_entity_decision_rejection_count") or 0
            )
            stage_counts["review_entity_decision_entities"] = int(
                review_metadata.get("review_entity_decision_entity_count") or 0
            )
        resolved_subject_ledger_summary = dict(
            metadata.get("resolved_subject_ledger_summary") or {}
        )
        stage_counts["resolved_subject_ledger_review_queue"] = int(
            resolved_subject_ledger_summary.get("review_queue_count") or 0
        )
        stage_counts["resolved_subject_ledger_unresolved_subjects"] = int(
            resolved_subject_ledger_summary.get("unresolved_subject_count") or 0
        )
        stage_counts["post_review_merged"] = len(merged)

        if settings.ALIAS_PROPAGATION:
            before_count = len(merged)
            merged = recognizer.merge_service.propagate_aliases(text, merged)
            alias_added = max(0, len(merged) - before_count)
            stage_counts["alias_propagation_added"] = alias_added
            if alias_added:
                quality_flags.append("alias_propagation_applied")

        stage_counts["quality_gate_snippets_selected"] = 0
        stage_counts["quality_gate_review"] = 0
        stage_counts["quality_gate_rejected"] = 0

        if recognizer._needs_quality_review(text, merged):
            requires_manual_review = True
            quality_flags.append("quality_anomaly_detected")

        if entities_filter:
            merged = recognizer._filter_entities(merged, entities_filter)

        project_default_public_results = getattr(recognizer, "_project_default_public_results", None)
        merged = (
            project_default_public_results(merged)
            if callable(project_default_public_results)
            else list(merged)
        )
        final_results = _validate_and_expand(text, merged)
        final_entities = [result.to_dict() for result in final_results]
        contextual_service = ContextualDesensitizationService(
            review_service=recognizer.review_service,
        )
        final_entities = await contextual_service.refine_recognition_entities(
            text=text,
            entities=final_entities,
            use_llm=False,
            llm_model=(
                review_result.model_name
                if review_result and review_result.model_name
                else settings.get_default_review_llm_model()
            ),
            source_metadata=source_metadata,
            source_structure=source_structure,
            progress_callback=None,
        )
        structure_short_org_before_postprocess = _structure_short_org_counts(final_entities)
        final_entities = _postprocess_final_entities(contextual_service, text, final_entities)
        structure_short_org_after_postprocess = _structure_short_org_counts(final_entities)
        contextual_quality_metadata = contextual_service.get_last_quality_metadata()
        contextual_quality_passed = bool(contextual_quality_metadata.get("quality_gate_passed", True))
        contextual_quality_reason = str(contextual_quality_metadata.get("quality_gate_reason") or "").strip()
        if not contextual_quality_passed:
            requires_manual_review = True
            quality_flags.append("contextual_quality_gate_failed")

        if settings.REVIEW_UNLOAD_AFTER_TASK:
            recognizer.review_service.unload()
        stage_counts["final"] = len(final_entities)
        stage_counts["structure_short_org_before_postprocess"] = int(
            structure_short_org_before_postprocess.get("total") or 0
        )
        stage_counts["structure_short_org_ambiguous_before_postprocess"] = int(
            structure_short_org_before_postprocess.get("ambiguous") or 0
        )
        stage_counts["structure_short_org_after_postprocess"] = int(
            structure_short_org_after_postprocess.get("total") or 0
        )
        stage_counts["structure_short_org_ambiguous_after_postprocess"] = int(
            structure_short_org_after_postprocess.get("ambiguous") or 0
        )
        removed_structure_short_org = max(
            0,
            int(structure_short_org_before_postprocess.get("total") or 0)
            - int(structure_short_org_after_postprocess.get("total") or 0),
        )
        stage_counts["structure_short_org_removed_by_publication_filter"] = removed_structure_short_org
        if removed_structure_short_org:
            quality_flags.append("structure_short_org_publication_filter_applied")

        blocking_quality_flags = [
            item
            for item in quality_flags
            if item.startswith(
                (
                    "review_warning:",
                    "review_model_missing",
                    "quality_anomaly",
                )
            )
        ]
        ledger_decision_count = int(stage_counts.get("ledger_conflict_decisions") or 0)
        ledger_snippet_count = int(stage_counts.get("ledger_conflict_snippets") or 0)
        scheduled_ledger_count = int(stage_counts.get("review_scheduled_ledger_snippets") or 0)
        ledger_completion = _ledger_adjudication_completion(
            review_enabled=bool(settings.ENABLE_QWEN_REVIEW),
            review_result=review_result,
            ledger_decision_count=ledger_decision_count,
            ledger_snippet_count=ledger_snippet_count,
            scheduled_ledger_count=scheduled_ledger_count,
            source_review_queue_count=int(
                metadata.get("subject_ledger_review_queue_count")
                or stage_counts.get("subject_ledger_review_queue")
                or 0
            ),
            resolved_review_queue_count=int(
                resolved_subject_ledger_summary.get("review_queue_count") or 0
            ),
        )
        expected_ledger_decisions = int(ledger_completion["expected_decision_count"])
        ledger_adjudication_incomplete = bool(ledger_completion["incomplete"])
        if ledger_adjudication_incomplete:
            requires_manual_review = True
            quality_flags.append("ledger_conflict_adjudication_incomplete")
        metadata.update(
            {
                "workflow_variant": "lowmem_mid_review",
                "review_configured": bool(settings.ENABLE_QWEN_REVIEW),
                "review_dispatched": bool(settings.ENABLE_QWEN_REVIEW and len(snippets) > 0),
                "review_started": bool(
                    review_result and review_result.model_used
                ),
                "review_completed": bool(review_result and review_result.model_used and not ledger_adjudication_incomplete),
                "recognition_profile": settings.get_high_quality_profile_key(),
                "primary_model": "rule/uie/ner",
                "review_model": (
                    review_result.model_name
                    if review_result and review_result.model_name
                    else settings.get_default_review_llm_model()
                ),
                "review_backend": (
                    review_result.review_backend
                    if review_result
                    else None
                ),
                "active_review_stage": (
                    "standard_review"
                    if review_result and review_result.model_used
                    else None
                ),
                "active_review_model": (
                    review_result.model_name
                    if review_result and review_result.model_name
                    else None
                ),
                "active_review_backend": (
                    review_result.review_backend
                    if review_result
                    else None
                ),
                "review_model_configured": settings.get_default_review_llm_model() or settings.REVIEW_MODEL,
                "review_model_used": bool(review_result and review_result.model_used),
                "contextual_refine_llm_used": False,
                "contextual_refine_llm_skipped_reason": "rule_first_contextual_refine",
                "review_model_fallback_used": bool(review_result and review_result.fallback_used),
                "review_model_loaded": bool(recognizer.review_service.loaded),
                "review_error": review_result.error if review_result else None,
                "review_snippet_count": len(review_snippets),
                "review_snippet_scheduled_count": len(snippets),
                "review_skipped_reason": review_skipped_reason,
                "review_quality_mode": "rule_first",
                "high_risk_subject_unreviewed_count": 0,
                "same_surface_multi_type_unresolved_count": 0,
                "subject_multi_replacement_count": 0,
                "replacement_reused_by_multi_subject_count": 0,
                "qwen_trigger_reasons": review_trigger_reasons,
                **qwen_contribution,
                "ledger_conflict_adjudication_enabled": bool(
                    review_result and review_metadata.get("ledger_conflict_adjudication_enabled")
                ),
                "ledger_conflict_decision_count": ledger_decision_count,
                "ledger_conflict_snippet_count": ledger_snippet_count,
                "ledger_conflict_expected_decision_count": expected_ledger_decisions,
                "ledger_conflict_remaining_review_queue_count": int(
                    ledger_completion["remaining_review_queue_count"]
                ),
                "ledger_conflict_decisions": list(
                    review_metadata.get("ledger_conflict_decisions") or []
                ) if review_result else [],
                "review_scheduled_ledger_snippet_count": int(
                    review_metadata.get("review_scheduled_ledger_snippet_count") or 0
                ) if review_result else 0,
                "review_scheduled_standard_snippet_count": int(
                    review_metadata.get("review_scheduled_standard_snippet_count") or 0
                ) if review_result else 0,
                "review_deterministic_rejection_count": int(
                    review_metadata.get("review_deterministic_rejection_count") or 0
                ) if review_result else 0,
                "review_deterministic_entity_decision_count": int(
                    review_metadata.get("review_deterministic_entity_decision_count") or 0
                ) if review_result else 0,
                "review_deterministic_entity_count": int(
                    review_metadata.get("review_deterministic_entity_count") or 0
                ) if review_result else 0,
                "review_entity_decision_rejection_count": int(
                    review_metadata.get("review_entity_decision_rejection_count") or 0
                ) if review_result else 0,
                "review_entity_decision_entity_count": int(
                    review_metadata.get("review_entity_decision_entity_count") or 0
                ) if review_result else 0,
                "ledger_conflict_adjudication_incomplete": ledger_adjudication_incomplete,
                "resolved_subject_ledger_enabled": bool(resolved_subject_ledger_summary),
                "resolved_subject_ledger_summary": resolved_subject_ledger_summary,
                "resolved_subject_ledger_review_queue_count": int(
                    resolved_subject_ledger_summary.get("review_queue_count") or 0
                ),
                "stage_counts": stage_counts,
                "requires_manual_review": bool(requires_manual_review),
                "quality_gate_passed": (
                    not bool(requires_manual_review)
                    and not blocking_quality_flags
                    and contextual_quality_passed
                ),
                "quality_gate_reason": contextual_quality_reason or None,
                "contextual_quality_gate": contextual_quality_metadata,
                "short_org_publication_filter_version": SHORT_ORG_PUBLICATION_FILTER_VERSION,
                "quality_flags": sorted(set(quality_flags)),
                "review_model_installed": bool(review_available),
            }
        )
        return {
            "entities": final_entities,
            "statistics": _entity_statistics(final_entities),
            "analysis_metadata": metadata,
        }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.qwen_review_worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = asyncio.run(_run(payload))
        output_path.write_text(json.dumps({"ok": True, **result}, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as exc:
        error_payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            output_path.write_text(json.dumps(error_payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
