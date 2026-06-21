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
from typing import Any, Dict, Optional

from app.core.config import desensitize_mode_context, settings
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
    review_only_candidates: list[Dict[str, Any]] = []
    for key in ("rule_first_rejected_entities", "alias_backscan_review_candidates"):
        rows = artifacts.get(key)
        if isinstance(rows, list):
            review_only_candidates.extend(dict(item) for item in rows if isinstance(item, dict))

    return {
        "entities": entities,
        "review_entities": review_entities,
        "review_only_candidates": review_only_candidates,
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
                raise RuntimeError("large_document_pre_routed_must_use_parent_process_grouped_default_line")
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
