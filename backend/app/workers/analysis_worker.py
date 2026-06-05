"""One-shot entity analysis worker.

This process exists to load memory-heavy local models for a single analysis task
and then exit, letting macOS reclaim MLX/transformers memory that Python may not
return to the parent API process promptly.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.anonymization_strategy import (
    DEFAULT_ANONYMIZATION_STRATEGY,
    get_anonymization_strategy_profile,
)
from app.core.config import desensitize_mode_context, settings
from app.core.llm_strategy import get_llm_strategy_profile
from app.engine.desensitization_engine import get_engine


def _get_analysis_metadata(engine, llm_model: Optional[str]) -> Dict[str, Any]:
    if settings.is_high_quality_desensitize_mode():
        recognizer = engine.recognizer_registry.get_recognizer("high_quality_lowmem")
        if recognizer is None or not hasattr(recognizer, "get_last_run_metadata"):
            return {}
        return dict(recognizer.get_last_run_metadata(llm_model))

    recognizer = engine.recognizer_registry.get_recognizer("llm")
    if recognizer is None or not hasattr(recognizer, "get_last_run_metadata"):
        return {}
    return dict(recognizer.get_last_run_metadata(llm_model))


def _batch_source_metadata(
    source_metadata: Optional[Dict[str, Any]],
    *,
    batch_page_count: int,
) -> Dict[str, Any]:
    metadata = dict(source_metadata or {})
    metadata["pages"] = max(1, int(batch_page_count or 1))
    metadata["page_count"] = max(1, int(batch_page_count or 1))
    return metadata


def _batch_source_structure(
    source_structure: Optional[Dict[str, Any]],
    *,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    batch_page_count = int(batch.get("page_count", 0) or len(batch.get("page_numbers", []) or []) or 1)
    batch_start = int(batch.get("start", 0))
    batch_end = int(batch.get("end", batch_start))
    batch_text = str(batch.get("text", "") or "")
    page_numbers = {
        int(value)
        for value in (batch.get("page_numbers") or [])
        if isinstance(value, int) and value > 0
    }

    raw_pages = batch.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        if not isinstance(source_structure, dict):
            return {"page_count": max(1, batch_page_count)}
        raw_pages = source_structure.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        return {"page_count": max(1, batch_page_count)}

    local_pages: List[Dict[str, Any]] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        try:
            page_number = int(raw_page.get("page_number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if page_numbers and page_number not in page_numbers:
            continue

        item = dict(raw_page)
        item["page_number"] = page_number
        try:
            start = int(raw_page.get("start", -1))
            end = int(raw_page.get("end", -1))
        except (TypeError, ValueError):
            start = -1
            end = -1

        if start >= 0 and end >= start:
            local_start = max(0, start - batch_start)
            local_end = max(local_start, min(len(batch_text), end - batch_start))
            item["start"] = local_start
            item["end"] = local_end
            item["text"] = batch_text[local_start:local_end]
        else:
            item["start"] = 0
            item["end"] = 0
            item["text"] = str(raw_page.get("text", "") or "")
        local_pages.append(item)

    if not local_pages:
        return {"page_count": max(1, batch_page_count)}

    return {
        "page_count": max(1, batch_page_count),
        "pages": local_pages,
        "start": batch_start,
        "end": batch_end,
    }


async def _run_large_document_primary_review_surface(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    use_llm = bool(payload.get("use_llm"))
    use_custom = bool(payload.get("use_custom"))
    llm_model = payload.get("llm_model")
    anonymization_strategy = payload.get("anonymization_strategy")
    source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
    source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
    entities_filter = payload.get("entities")

    if entities_filter is not None and not isinstance(entities_filter, list):
        entities_filter = None

    engine = get_engine()
    contextual_service = engine.pipeline_manager.contextual_desensitizer
    page_batches = contextual_service._build_large_document_page_batches(
        text=text,
        source_metadata=source_metadata,
        source_structure=source_structure,
        batch_page_count=10,
    )

    if not page_batches:
        entities = await engine.analyze(
            text=text,
            entities=entities_filter,
            use_llm=use_llm,
            use_custom=use_custom,
            llm_model=llm_model,
            source_metadata=source_metadata,
            source_structure=source_structure,
            progress_callback=None,
        )
        prepared_entities = await engine.prepare_entities_for_anonymization(
            text=text,
            entities=entities,
            use_llm=use_llm,
            llm_model=llm_model,
            anonymization_strategy=anonymization_strategy,
            source_metadata=source_metadata,
            source_structure=source_structure,
        )
        analysis_metadata = _get_analysis_metadata(engine, llm_model)
        analysis_metadata.update(engine.get_last_quality_metadata())
        return {
            "entities": prepared_entities,
            "review_entities": list(prepared_entities),
            "statistics": engine.get_entity_statistics(prepared_entities),
            "analysis_metadata": analysis_metadata,
        }

    strategy_key = get_anonymization_strategy_profile(
        anonymization_strategy or DEFAULT_ANONYMIZATION_STRATEGY
    ).key
    strategy_profile = get_llm_strategy_profile(llm_model or settings.get_default_llm_model()) if use_llm else None
    precision_multi_review = bool(strategy_profile and strategy_profile.key == "precision_4b")
    explicit_types: set[str] = set()
    large_document_policy = contextual_service._large_document_policy(
        text=text,
        entities=[],
        use_llm=use_llm,
        precision_multi_review=precision_multi_review,
        source_metadata=source_metadata,
        source_structure=source_structure,
    )

    aggregated_entities: List[Dict[str, Any]] = []
    aggregated_raw_entities: List[Dict[str, Any]] = []
    batch_runs: List[Dict[str, Any]] = []
    for batch in page_batches:
        batch_text = str(batch.get("text", "") or "")
        batch_index = int(batch.get("index", 0))
        batch_page_numbers = list(batch.get("page_numbers", []))
        if not batch_text.strip():
            batch_runs.append(
                {
                    "batch_index": batch_index,
                    "page_numbers": batch_page_numbers,
                    "input_entity_count": 0,
                    "output_entity_count": 0,
                    "quality_gate_passed": True,
                    "quality_gate_reason": "",
                }
            )
            continue
        batch_page_count = int(batch.get("page_count", 0) or len(batch.get("page_numbers", []) or []) or 1)
        batch_source_metadata = _batch_source_metadata(
            source_metadata,
            batch_page_count=batch_page_count,
        )
        batch_source_structure = _batch_source_structure(
            source_structure,
            batch=batch,
        )
        batch_entities = await engine.analyze(
            text=batch_text,
            entities=entities_filter,
            use_llm=use_llm,
            use_custom=use_custom,
            llm_model=llm_model,
            source_metadata=batch_source_metadata,
            source_structure=batch_source_structure,
            progress_callback=None,
        )
        restored_batch_entities = contextual_service._restore_page_batch_entities_to_global(
            entities=batch_entities,
            batch=batch,
            full_text=text,
        )
        aggregated_raw_entities.extend(restored_batch_entities)

        batch_quality_policy = contextual_service._build_standard_quality_policy(
            text=batch_text,
            entities=batch_entities,
            use_llm=use_llm,
            precision_multi_review=precision_multi_review,
        )
        batch_prepared_entities, batch_quality = await contextual_service._prepare_entities_standard_mode(
            text=batch_text,
            prepared_entities=batch_entities,
            explicit_types=explicit_types,
            strategy_key=strategy_key,
            use_llm=use_llm,
            llm_model=llm_model,
            strategy_profile=strategy_profile,
            quality_policy=batch_quality_policy,
            run_llm_refine=True,
            source_metadata=batch_source_metadata,
            source_structure=batch_source_structure,
        )
        restored_prepared_entities = contextual_service._restore_page_batch_entities_to_global(
            entities=batch_prepared_entities,
            batch=batch,
            full_text=text,
        )
        aggregated_entities.extend(
            contextual_service._sanitize_entity_for_global_reconciliation(
                entity=item,
                batch_index=batch_index,
            )
            for item in restored_prepared_entities
        )
        batch_runs.append(
            {
                "batch_index": batch_index,
                "page_numbers": batch_page_numbers,
                "input_entity_count": len(batch_entities),
                "output_entity_count": len(restored_prepared_entities),
                "quality_gate_passed": bool(batch_quality.get("quality_gate_passed")),
                "quality_gate_reason": str(batch_quality.get("quality_gate_reason", "") or ""),
            }
        )

    aggregated_entities = contextual_service._sort_and_deduplicate_entities(aggregated_entities)
    seed_entities = contextual_service._restore_large_document_anchor_entities(
        text=text,
        prepared_entities=aggregated_entities,
        raw_entities=aggregated_raw_entities,
    )
    prepared_entities = await contextual_service._finalize_large_document_page_batched_entities(
        text=text,
        aggregated_entities=aggregated_entities,
        seed_entities=seed_entities,
        raw_entities=aggregated_raw_entities,
        explicit_types=explicit_types,
        strategy_key=strategy_key,
        use_llm=use_llm,
        llm_model=llm_model,
        strategy_profile=strategy_profile,
        large_document_policy=large_document_policy,
        page_chunk_plan=page_batches,
        batch_runs=batch_runs,
    )
    analysis_metadata = _get_analysis_metadata(engine, llm_model)
    analysis_metadata.update(contextual_service.get_last_quality_metadata())
    return {
        "entities": prepared_entities,
        "review_entities": list(prepared_entities),
        "statistics": engine.get_entity_statistics(prepared_entities),
        "analysis_metadata": analysis_metadata,
    }


async def _run_primary_review_surface(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    use_llm = bool(payload.get("use_llm"))
    use_custom = bool(payload.get("use_custom"))
    llm_model = payload.get("llm_model")
    source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
    source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
    entities_filter = payload.get("entities")

    if entities_filter is not None and not isinstance(entities_filter, list):
        entities_filter = None

    engine = get_engine()
    entities = await engine.analyze(
        text=text,
        entities=entities_filter,
        use_llm=use_llm,
        use_custom=use_custom,
        llm_model=llm_model,
        source_metadata=source_metadata,
        source_structure=source_structure,
        progress_callback=None,
    )
    recognizer = engine.recognizer_registry.get_recognizer("high_quality_lowmem")
    if recognizer is None:
        raise RuntimeError("high_quality_lowmem recognizer unavailable")
    artifacts = (
        recognizer.get_last_run_artifacts()
        if hasattr(recognizer, "get_last_run_artifacts")
        else {}
    )
    review_entities = artifacts.get("pre_review_merged")
    if not isinstance(review_entities, list) or not review_entities:
        review_entities = list(entities)

    return {
        "entities": entities,
        "review_entities": review_entities,
        "statistics": engine.get_entity_statistics(entities),
        "analysis_metadata": _get_analysis_metadata(engine, llm_model),
    }


async def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    with desensitize_mode_context(payload.get("desensitize_mode")):
        text = str(payload.get("text") or "")
        use_llm = bool(payload.get("use_llm"))
        use_custom = bool(payload.get("use_custom"))
        llm_model = payload.get("llm_model")
        source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
        source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
        entities_filter = payload.get("entities")
        stage_mode = str(payload.get("stage_mode") or "").strip()

        if entities_filter is not None and not isinstance(entities_filter, list):
            entities_filter = None

        if (
            stage_mode == "primary_review_surface"
            and use_llm
            and settings.is_high_quality_desensitize_mode()
        ):
            if str(payload.get("analysis_workflow_mode") or "").strip() == "large_document_pre_routed":
                return await _run_large_document_primary_review_surface(payload)
            return await _run_primary_review_surface(payload)

        engine = get_engine()
        entities = await engine.analyze(
            text=text,
            entities=entities_filter,
            use_llm=use_llm,
            use_custom=use_custom,
            llm_model=llm_model,
            source_metadata=source_metadata,
            source_structure=source_structure,
            progress_callback=None,
        )
        return {
            "entities": entities,
            "statistics": engine.get_entity_statistics(entities),
            "analysis_metadata": _get_analysis_metadata(engine, llm_model),
        }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.analysis_worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
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
