"""Candidate conversion and evidence helpers."""

from __future__ import annotations

import re
from typing import Iterable

from app.core.recognizer_base import RecognizerResult
from app.rules.types import Candidate, RuleEvidence
from app.services.lowmem_entity_utils import normalize_entity_text


def source_layer_for_result(result: RecognizerResult) -> str:
    metadata = dict(result.metadata or {})
    explicit = str(metadata.get("source_layer") or "").strip()
    if explicit:
        return explicit
    source = str(result.source or "").lower()
    if source in {"regex", "custom"} or "pattern" in source:
        return "regex"
    if "structure" in source or "contract" in source:
        return "structure"
    if "uie" in source:
        return "uie"
    if "ner" in source:
        return "ner"
    if "qwen" in source or "review" in source or source in {"llm", "ollama"}:
        return "small_model"
    if "alias" in source or "propagat" in source:
        return "alias"
    return "unknown"


def candidate_from_result(result: RecognizerResult, index: int) -> Candidate:
    metadata = dict(result.metadata or {})
    return Candidate(
        candidate_id=str(metadata.get("candidate_id") or f"C{index:05d}"),
        entity_type=str(result.entity_type or "").strip().upper(),
        text=str(result.text or ""),
        start=int(result.start),
        end=int(result.end),
        source=str(result.source or ""),
        source_layer=source_layer_for_result(result),
        confidence_raw=float(result.score or 0.0),
        metadata=metadata,
        unit_id=str(metadata.get("docx_unit_id") or ""),
    )


def candidates_from_results(results: Iterable[RecognizerResult]) -> list[Candidate]:
    return [candidate_from_result(result, index) for index, result in enumerate(results or [], start=1)]


def with_rule_metadata(
    result: RecognizerResult,
    *,
    evidence: RuleEvidence,
    canonical_key: str = "",
    canonical_role: str = "",
    subject_id: str = "",
    extra: dict | None = None,
) -> RecognizerResult:
    metadata = dict(result.metadata or {})
    metadata["rule_first"] = evidence.to_metadata()
    metadata["source_layer"] = source_layer_for_result(result)
    metadata["normalized_text"] = metadata.get("normalized_text") or normalize_entity_text(result.text)
    if canonical_key:
        metadata["canonical_key"] = canonical_key
    if canonical_role:
        metadata["canonical_role"] = canonical_role
    if subject_id:
        metadata["subject_id"] = subject_id
    if extra:
        metadata.update(extra)
    return RecognizerResult(
        entity_type=result.entity_type,
        start=result.start,
        end=result.end,
        score=max(float(result.score or 0.0), float(evidence.confidence or 0.0)),
        text=result.text,
        source=result.source,
        metadata=metadata,
    )


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))
