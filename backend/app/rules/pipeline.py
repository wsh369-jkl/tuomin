"""Rule-first candidate processing for the default desensitization workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.core.recognizer_base import RecognizerResult
from app.rules.boundary_repair import BoundaryRepair
from app.rules.default_subject_policy import canonicalize_default_result
from app.rules.evidence import source_layer_for_result, with_rule_metadata
from app.rules.false_positive_rules import FalsePositiveRules
from app.rules.format_recognizer import FormatRecognizer
from app.rules.directory_quality_gate import DirectoryQualityGate
from app.rules.subject_ledger import SubjectLedgerBuilder
from app.rules.type_recognizers import TypeRuleRecognizers
from app.rules.types import RuleEvidence
from app.services.lowmem_entity_utils import (
    deduplicate_results,
    is_org_like_text,
    is_probable_person,
    normalize_entity_text,
)


@dataclass
class RuleFirstPipelineResult:
    results: list[RecognizerResult]
    metadata: dict[str, Any]
    rejected_results: list[RecognizerResult]


class RuleFirstPipeline:
    """Deterministic rule layer that runs inside the existing default workflow."""

    def __init__(self) -> None:
        self.format_recognizer = FormatRecognizer()
        self.type_recognizers = TypeRuleRecognizers()
        self.boundary_repair = BoundaryRepair()
        self.false_positive_rules = FalsePositiveRules()
        self.subject_ledger = SubjectLedgerBuilder()
        self.directory_quality_gate = DirectoryQualityGate()

    def apply(
        self,
        *,
        text: str,
        results: Iterable[RecognizerResult],
        source_structure: dict[str, Any] | None = None,
    ) -> RuleFirstPipelineResult:
        input_results = list(results or [])
        format_results = self.format_recognizer.recognize(text)
        type_rule_results = self.type_recognizers.recognize(text, source_structure=source_structure)
        repaired_results: list[RecognizerResult] = []
        rejected_results: list[RecognizerResult] = []
        action_counts: dict[str, int] = {"keep": 0, "reject": 0, "repair": 0, "review": 0}
        type_change_count = 0
        boundary_repair_count = 0
        validator_pass_count = 0
        validator_fail_count = 0

        for raw_result in [*input_results, *format_results, *type_rule_results]:
            result = canonicalize_default_result(raw_result)
            if result is None:
                rejected_results.append(raw_result)
                action_counts["reject"] = action_counts.get("reject", 0) + 1
                continue
            normalized_result, type_changed = self._repair_type(result)
            type_change_count += int(type_changed)
            for repaired, repairs in self.boundary_repair.repair(text, normalized_result):
                if repairs:
                    boundary_repair_count += 1
                reject, positive, negative = self.false_positive_rules.assess(repaired)
                validation = self.format_recognizer.validate(repaired.entity_type, repaired.text)
                validators_passed = tuple({*validation.passed, *self._metadata_list(repaired, "validators_passed")})
                validators_failed = tuple({*validation.failed, *self._metadata_list(repaired, "validators_failed")})
                validator_pass_count += len(validators_passed)
                validator_fail_count += len(validators_failed)
                action = "reject" if reject else self._action_for(repaired, negative, validators_failed)
                risk_level = self._risk_level_for(repaired, negative, validators_failed, action)
                confidence = max(float(repaired.score or 0.0), float(validation.confidence or 0.0))
                evidence = RuleEvidence(
                    candidate_id=self._candidate_id(repaired),
                    entity_type=repaired.entity_type,
                    text=repaired.text,
                    positive_signals=tuple(sorted(set(positive))),
                    negative_signals=tuple(sorted(set(negative))),
                    validators_passed=tuple(sorted(validators_passed)),
                    validators_failed=tuple(sorted(validators_failed)),
                    boundary_repairs=tuple(sorted(set(repairs))),
                    confidence=confidence,
                    risk_level=risk_level,
                    action=action,
                )
                action_counts[action] = action_counts.get(action, 0) + 1
                updated = with_rule_metadata(repaired, evidence=evidence)
                if reject:
                    rejected_results.append(self._rejected_result_for_review(updated))
                else:
                    repaired_results.append(updated)

        deduped = deduplicate_results(repaired_results)
        ledger_result = self.subject_ledger.build(deduped)
        with_subjects = ledger_result.results
        directory_quality = self.directory_quality_gate.evaluate(with_subjects)
        metadata = {
            "rule_first_enabled": True,
            "rule_first_input_candidate_count": len(input_results),
            "rule_first_format_candidate_count": len(format_results),
            "rule_first_type_rule_candidate_count": len(type_rule_results),
            "rule_first_output_candidate_count": len(with_subjects),
            "rule_first_rejected_candidate_count": len(rejected_results),
            "rule_first_action_counts": dict(sorted(action_counts.items())),
            "rule_first_type_change_count": type_change_count,
            "rule_first_boundary_repair_count": boundary_repair_count,
            "rule_first_validator_pass_count": validator_pass_count,
            "rule_first_validator_fail_count": validator_fail_count,
            "rule_first_subject_graph": ledger_result.subject_graph_summary,
            "rule_first_subject_ledger": ledger_result.ledger,
            "rule_first_subject_ledger_summary": ledger_result.summary,
            "rule_first_unresolved_subject_count": int(
                ledger_result.summary.get("unresolved_subject_count") or 0
            ),
            "rule_first_review_queue_count": action_counts.get("review", 0)
            + int(ledger_result.summary.get("review_queue_count") or 0),
            **directory_quality,
        }
        return RuleFirstPipelineResult(
            results=with_subjects,
            metadata=metadata,
            rejected_results=rejected_results,
        )

    @staticmethod
    def _candidate_id(result: RecognizerResult) -> str:
        metadata = dict(result.metadata or {})
        return str(metadata.get("candidate_id") or f"{result.source}:{result.entity_type}:{result.start}:{result.end}")

    @staticmethod
    def _rejected_result_for_review(result: RecognizerResult) -> RecognizerResult:
        metadata = dict(result.metadata or {})
        repaired_from = metadata.get("boundary_repaired_from")
        if not isinstance(repaired_from, dict):
            return result
        original_text = str(repaired_from.get("text") or "")
        try:
            original_start = int(repaired_from.get("start"))
            original_end = int(repaired_from.get("end"))
        except (TypeError, ValueError):
            return result
        if not original_text or original_end <= original_start or original_text == result.text:
            return result
        metadata["rejected_after_boundary_repair"] = True
        metadata["rejected_repaired_candidate"] = {
            "start": result.start,
            "end": result.end,
            "text": result.text,
        }
        return RecognizerResult(
            entity_type=result.entity_type,
            start=original_start,
            end=original_end,
            score=result.score,
            text=original_text,
            source=result.source,
            metadata=metadata,
        )

    @staticmethod
    def _metadata_list(result: RecognizerResult, key: str) -> list[str]:
        value = (result.metadata or {}).get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        if isinstance(value, tuple):
            return [str(item) for item in value if str(item)]
        return []

    @staticmethod
    def _repair_type(result: RecognizerResult) -> tuple[RecognizerResult, bool]:
        entity_type = str(result.entity_type or "").upper()
        normalized = normalize_entity_text(result.text)
        metadata = dict(result.metadata or {})
        if (
            entity_type == "ORGANIZATION"
            and metadata.get("trigger") in {"function_action_boundary_short_org", "parallel_action_boundary_short_org"}
        ):
            return result, False
        new_type = entity_type
        if entity_type == "PERSON" and is_org_like_text(normalized):
            new_type = "ORGANIZATION"
        elif entity_type == "ORGANIZATION" and is_probable_person(normalized) and not is_org_like_text(normalized):
            new_type = "PERSON"
        if new_type == entity_type:
            return result, False
        metadata["rule_type_repaired_from"] = entity_type
        return (
            RecognizerResult(
                entity_type=new_type,
                start=result.start,
                end=result.end,
                score=max(float(result.score or 0.0), 0.84),
                text=result.text,
                source=result.source,
                metadata=metadata,
            ),
            True,
        )

    @staticmethod
    def _action_for(result: RecognizerResult, negative: list[str], validators_failed: tuple[str, ...]) -> str:
        entity_type = str(result.entity_type or "").upper()
        if negative or validators_failed:
            return "review"
        if source_layer_for_result(result) in {"regex", "structure"}:
            return "keep"
        if entity_type in {"PERSON", "ORGANIZATION", "GOVERNMENT"} and len(normalize_entity_text(result.text)) <= 3:
            return "review"
        return "keep"

    @staticmethod
    def _risk_level_for(
        result: RecognizerResult,
        negative: list[str],
        validators_failed: tuple[str, ...],
        action: str,
    ) -> str:
        entity_type = str(result.entity_type or "").upper()
        if action == "reject" or validators_failed:
            return "high"
        if negative:
            return "high"
        if entity_type in {"PERSON", "ORGANIZATION", "GOVERNMENT"} and len(normalize_entity_text(result.text)) <= 3:
            return "medium"
        return "low"
