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
    return RecognizerResult(
        entity_type=entity_type,
        start=start,
        end=end,
        score=score,
        text=text,
        source=source,
        metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
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


def _validate_and_expand(text: str, results: List[RecognizerResult]) -> List[RecognizerResult]:
    manager = object.__new__(PipelineManager)
    validated = manager._validate_results(results, text)
    expanded = manager._expand_repeated_mentions(validated, text)
    return manager._validate_results(expanded, text)


def _results_from_entity_dicts(items: Iterable[Dict[str, Any]]) -> List[RecognizerResult]:
    return _results_from_payload(items)


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
        source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
        source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
        entities_filter = payload.get("entities_filter")
        if entities_filter is not None and not isinstance(entities_filter, list):
            entities_filter = None

        primary_entities = _results_from_payload(payload.get("entities") or [])
        review_surface_entities = _results_from_payload(payload.get("review_entities") or [])
        if not review_surface_entities:
            review_surface_entities = list(primary_entities)
        primary_metadata = dict(payload.get("analysis_metadata") or {})
        recognizer = HighQualityLowMemoryRecognizer()

        snippets = recognizer.snippet_scheduler.build_snippets(text, review_surface_entities)
        review_snippets = recognizer._select_review_snippets_requiring_semantic_recovery(
            snippets,
            review_surface_entities,
        )
        review_trigger_reasons = sorted({snippet.risk_reason for snippet in review_snippets})
        metadata = dict(primary_metadata)
        stage_counts = dict(metadata.get("stage_counts") or {})
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

        results = list(review_surface_entities)
        review_available = recognizer.review_service.installed
        if settings.ENABLE_QWEN_REVIEW and review_snippets and review_available:
            review_result = await recognizer.review_service.review(
                text,
                review_snippets,
                existing_entities=review_surface_entities,
            )
            qwen_contribution = recognizer._measure_qwen_contribution(
                review_surface_entities,
                review_result.entities,
                review_result.raw_candidate_count,
            )
            if review_result.rejected_entities:
                before_reject_count = len(results)
                results = recognizer._apply_review_rejections(results, review_result.rejected_entities)
                rejected_count = max(0, before_reject_count - len(results))
                qwen_contribution["qwen_rejected_entities"] = rejected_count
                if rejected_count:
                    qwen_contribution["qwen_value_level"] = "corrective"
                    quality_flags.append("qwen_rejections_applied")
            results.extend(review_result.entities)
            requires_manual_review = requires_manual_review or review_result.requires_manual_review
            if review_result.error:
                quality_flags.append(f"review_warning:{review_result.error}")
            if review_result.arbitration_used:
                quality_flags.append("heavy_arbitration_used")
            if review_result.arbitration_error:
                quality_flags.append(f"arbitration_warning:{review_result.arbitration_error}")
        elif settings.ENABLE_QWEN_REVIEW and not review_available:
            requires_manual_review = True
            quality_flags.append("review_model_missing")
            review_skipped_reason = "review_model_missing"
        elif settings.ENABLE_QWEN_REVIEW and snippets and not review_snippets:
            review_skipped_reason = "primary_pipeline_sufficient"
        elif settings.ENABLE_QWEN_REVIEW:
            review_skipped_reason = "no_risk_snippets"
        else:
            review_skipped_reason = "review_disabled"

        merged = recognizer.merge_service.merge(results)
        stage_counts["primary_review_surface"] = len(review_surface_entities)
        stage_counts["primary_fallback_entities"] = len(primary_entities)
        stage_counts["risk_snippets"] = len(snippets)
        stage_counts["review_snippets_selected"] = len(review_snippets)
        stage_counts["qwen_raw_candidates"] = int(qwen_contribution["qwen_raw_candidates"])
        stage_counts["qwen_review"] = len(review_result.entities) if review_result else 0
        stage_counts["qwen_new_after_merge"] = int(qwen_contribution["qwen_new_entities_after_merge"])
        stage_counts["qwen_rejected"] = int(qwen_contribution.get("qwen_rejected_entities") or 0)
        stage_counts["post_review_merged"] = len(merged)

        if settings.ALIAS_PROPAGATION:
            before_count = len(merged)
            merged = recognizer.merge_service.propagate_aliases(text, merged)
            alias_added = max(0, len(merged) - before_count)
            stage_counts["alias_propagation_added"] = alias_added
            if alias_added:
                quality_flags.append("alias_propagation_applied")

        if settings.ENABLE_QWEN_REVIEW and review_available:
            quality_gate_snippets = recognizer._build_final_quality_gate_snippets(text, merged)
            stage_counts["quality_gate_snippets_selected"] = len(quality_gate_snippets)
            if quality_gate_snippets:
                gate_result = await recognizer.review_service.review(
                    text,
                    quality_gate_snippets,
                    existing_entities=merged,
                    max_snippets=len(quality_gate_snippets),
                )
                stage_counts["quality_gate_review"] = len(gate_result.entities)
                if gate_result.rejected_entities:
                    before_gate_count = len(merged)
                    merged = recognizer._apply_review_rejections(merged, gate_result.rejected_entities)
                    rejected_gate_count = max(0, before_gate_count - len(merged))
                    stage_counts["quality_gate_rejected"] = rejected_gate_count
                    if rejected_gate_count:
                        quality_flags.append("quality_gate_rejections_applied")
                if gate_result.entities:
                    merged = recognizer.merge_service.merge([*merged, *gate_result.entities])
                    quality_flags.append("quality_gate_entities_added")
                requires_manual_review = requires_manual_review or gate_result.requires_manual_review
                if gate_result.error:
                    quality_flags.append(f"quality_gate_warning:{gate_result.error}")

        if recognizer._needs_quality_review(text, merged):
            requires_manual_review = True
            quality_flags.append("quality_anomaly_detected")

        if entities_filter:
            merged = recognizer._filter_entities(merged, entities_filter)

        final_results = _validate_and_expand(text, merged)
        final_entities = [result.to_dict() for result in final_results]
        contextual_service = ContextualDesensitizationService(
            review_service=recognizer.review_service,
        )
        final_entities = await contextual_service.refine_recognition_entities(
            text=text,
            entities=final_entities,
            use_llm=bool(review_available),
            llm_model=(
                review_result.model_name
                if review_result and review_result.model_name
                else settings.get_default_review_llm_model()
            ),
            source_metadata=source_metadata,
            source_structure=source_structure,
            progress_callback=None,
        )
        final_entities = _postprocess_final_entities(contextual_service, text, final_entities)
        contextual_quality_metadata = contextual_service.get_last_quality_metadata()
        contextual_quality_passed = bool(contextual_quality_metadata.get("quality_gate_passed", True))
        contextual_quality_reason = str(contextual_quality_metadata.get("quality_gate_reason") or "").strip()
        if not contextual_quality_passed:
            requires_manual_review = True
            quality_flags.append("contextual_quality_gate_failed")

        final_gate_result = None
        if settings.ENABLE_QWEN_REVIEW and review_available:
            final_gate_input = _results_from_entity_dicts(final_entities)
            final_quality_gate_snippets = recognizer._build_final_quality_gate_snippets(text, final_gate_input)
            stage_counts["final_quality_gate_snippets_selected"] = len(final_quality_gate_snippets)
            if final_quality_gate_snippets:
                final_gate_result = await recognizer.review_service.review(
                    text,
                    final_quality_gate_snippets,
                    existing_entities=final_gate_input,
                    max_snippets=len(final_quality_gate_snippets),
                )
                stage_counts["final_quality_gate_review"] = len(final_gate_result.entities)
                if final_gate_result.rejected_entities:
                    before_final_gate_count = len(final_gate_input)
                    final_gate_input = recognizer._apply_review_rejections(final_gate_input, final_gate_result.rejected_entities)
                    rejected_final_gate_count = max(0, before_final_gate_count - len(final_gate_input))
                    stage_counts["final_quality_gate_rejected"] = rejected_final_gate_count
                    if rejected_final_gate_count:
                        quality_flags.append("final_quality_gate_rejections_applied")
                if final_gate_result.entities:
                    final_gate_input = recognizer.merge_service.merge([*final_gate_input, *final_gate_result.entities])
                    quality_flags.append("final_quality_gate_entities_added")
                if final_gate_result.error:
                    quality_flags.append(f"final_quality_gate_warning:{final_gate_result.error}")
                requires_manual_review = requires_manual_review or final_gate_result.requires_manual_review
                final_entities = [result.to_dict() for result in _validate_and_expand(text, final_gate_input)]
                final_entities = _postprocess_final_entities(contextual_service, text, final_entities)

        if settings.REVIEW_UNLOAD_AFTER_TASK:
            recognizer.review_service.unload()
        stage_counts["final"] = len(final_entities)

        blocking_quality_flags = [
            item
            for item in quality_flags
            if item.startswith(
                (
                    "review_warning:",
                    "quality_gate_warning:",
                    "final_quality_gate_warning:",
                    "arbitration_warning:",
                    "review_model_missing",
                    "quality_anomaly",
                )
            )
        ]
        metadata.update(
            {
                "workflow_variant": (
                    "local_high_quality_mid_review"
                    if settings.is_local_high_quality_mode()
                    else "lowmem_mid_review"
                ),
                "review_configured": bool(settings.ENABLE_QWEN_REVIEW),
                "review_dispatched": bool(settings.ENABLE_QWEN_REVIEW and len(snippets) > 0),
                "review_started": bool(review_result is not None),
                "review_completed": True,
                "recognition_profile": settings.get_high_quality_profile_key(),
                "primary_model": "rule/uie/ner",
                "review_model": (
                    review_result.model_name
                    if review_result and review_result.model_name
                    else (settings.get_default_review_llm_model() or settings.MID_REVIEW_MODEL)
                ),
                "review_backend": review_result.review_backend if review_result else None,
                "review_model_configured": settings.get_default_review_llm_model() or settings.MID_REVIEW_MODEL,
                "fast_review_model_configured": settings.FAST_REVIEW_MODEL,
                "review_model_used": bool(review_result and review_result.model_used),
                "review_model_fallback_used": bool(review_result and review_result.fallback_used),
                "review_model_loaded": bool(recognizer.review_service.loaded),
                "review_error": review_result.error if review_result else None,
                "review_snippet_count": len(review_snippets),
                "review_snippet_scheduled_count": len(snippets),
                "arbitration_model": review_result.arbitration_model if review_result else None,
                "arbitration_model_configured": settings.HEAVY_ARBITRATION_MODEL,
                "arbitration_used": bool(review_result and review_result.arbitration_used),
                "arbitration_snippet_count": int(review_result.arbitration_snippet_count) if review_result else 0,
                "arbitration_error": review_result.arbitration_error if review_result else None,
                "review_skipped_reason": review_skipped_reason,
                "final_quality_gate_model_used": bool(final_gate_result and final_gate_result.model_used),
                "qwen_trigger_reasons": review_trigger_reasons,
                **qwen_contribution,
                "stage_counts": stage_counts,
                "requires_manual_review": bool(requires_manual_review),
                "quality_gate_passed": (
                    not bool(requires_manual_review)
                    and not blocking_quality_flags
                    and contextual_quality_passed
                ),
                "quality_gate_reason": contextual_quality_reason or None,
                "contextual_quality_gate": contextual_quality_metadata,
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
