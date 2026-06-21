"""Rule-first candidate processing for the default desensitization workflow."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
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
        rejected_reason_counts: Counter[str] = Counter()
        rejected_source_counts: Counter[str] = Counter()
        rejected_type_counts: Counter[str] = Counter()
        rejected_channel_counts: Counter[str] = Counter()
        rejected_trigger_counts: Counter[str] = Counter()
        rejected_source_layer_counts: Counter[str] = Counter()
        rejected_review_only_reason_counts: Counter[str] = Counter()
        rejected_review_only_candidate_count = 0
        type_change_count = 0
        boundary_repair_count = 0
        validator_pass_count = 0
        validator_fail_count = 0

        for raw_result in [*input_results, *format_results, *type_rule_results]:
            result = canonicalize_default_result(raw_result)
            if result is None:
                rejected = self._annotate_rejected_result(
                    raw_result,
                    stage="canonicalize_default_result",
                    reasons=("unsupported_internal_entity_type",),
                    positive=(),
                    negative=("unsupported_internal_entity_type",),
                    validators_passed=(),
                    validators_failed=(),
                    repairs=(),
                    action="reject",
                    risk_level="high",
                )
                rejected_results.append(rejected)
                self._record_rejected_candidate(
                    rejected,
                    reasons=("unsupported_internal_entity_type",),
                    rejected_reason_counts=rejected_reason_counts,
                    rejected_source_counts=rejected_source_counts,
                    rejected_type_counts=rejected_type_counts,
                    rejected_channel_counts=rejected_channel_counts,
                    rejected_trigger_counts=rejected_trigger_counts,
                    rejected_source_layer_counts=rejected_source_layer_counts,
                    rejected_review_only_reason_counts=rejected_review_only_reason_counts,
                )
                if self._is_review_only_rejected_candidate(rejected):
                    rejected_review_only_candidate_count += 1
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
                    reasons = self._reject_reasons(
                        positive=positive,
                        negative=negative,
                        validators_failed=validators_failed,
                        repairs=repairs,
                    )
                    rejected = self._rejected_result_for_review(updated)
                    rejected = self._annotate_rejected_result(
                        rejected,
                        stage="false_positive_rules",
                        reasons=reasons,
                        positive=positive,
                        negative=negative,
                        validators_passed=validators_passed,
                        validators_failed=validators_failed,
                        repairs=repairs,
                        action=action,
                        risk_level=risk_level,
                    )
                    rejected_results.append(rejected)
                    self._record_rejected_candidate(
                        rejected,
                        reasons=reasons,
                        rejected_reason_counts=rejected_reason_counts,
                        rejected_source_counts=rejected_source_counts,
                        rejected_type_counts=rejected_type_counts,
                        rejected_channel_counts=rejected_channel_counts,
                        rejected_trigger_counts=rejected_trigger_counts,
                        rejected_source_layer_counts=rejected_source_layer_counts,
                        rejected_review_only_reason_counts=rejected_review_only_reason_counts,
                    )
                    if self._is_review_only_rejected_candidate(rejected):
                        rejected_review_only_candidate_count += 1
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
            "rule_first_rejected_review_only_candidate_count": rejected_review_only_candidate_count,
            "rule_first_rejected_reason_counts": dict(sorted(rejected_reason_counts.items())),
            "rule_first_rejected_source_counts": dict(sorted(rejected_source_counts.items())),
            "rule_first_rejected_type_counts": dict(sorted(rejected_type_counts.items())),
            "rule_first_rejected_channel_counts": dict(sorted(rejected_channel_counts.items())),
            "rule_first_rejected_trigger_counts": dict(sorted(rejected_trigger_counts.items())),
            "rule_first_rejected_source_layer_counts": dict(sorted(rejected_source_layer_counts.items())),
            "rule_first_rejected_review_only_reason_counts": dict(
                sorted(rejected_review_only_reason_counts.items())
            ),
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

    @classmethod
    def _annotate_rejected_result(
        cls,
        result: RecognizerResult,
        *,
        stage: str,
        reasons: Iterable[str],
        positive: Iterable[str],
        negative: Iterable[str],
        validators_passed: Iterable[str],
        validators_failed: Iterable[str],
        repairs: Iterable[str],
        action: str,
        risk_level: str,
    ) -> RecognizerResult:
        metadata = dict(result.metadata or {})
        reason_list = cls._clean_strings(reasons) or ("unknown_reject_reason",)
        positive_list = cls._clean_strings(positive)
        negative_list = cls._clean_strings(negative)
        passed_list = cls._clean_strings(validators_passed)
        failed_list = cls._clean_strings(validators_failed)
        repair_list = cls._clean_strings(repairs)
        source_layer = str(metadata.get("source_layer") or source_layer_for_result(result) or "unknown")
        trigger = str(metadata.get("trigger") or "")
        recognition_channel = cls._recognition_channel_for(result)
        review_only, review_reasons = cls._review_only_rejected_candidate_reasons(
            result,
            reasons=reason_list,
            metadata=metadata,
        )
        metadata.update(
            {
                "rule_first_rejected": True,
                "rule_first_reject_stage": stage,
                "rule_first_reject_reasons": list(reason_list),
                "rule_first_positive_signals": list(positive_list),
                "rule_first_negative_signals": list(negative_list),
                "rule_first_validators_passed": list(passed_list),
                "rule_first_validators_failed": list(failed_list),
                "rule_first_boundary_repairs": list(repair_list),
                "rule_first_action": action,
                "rule_first_risk_level": risk_level,
                "rule_first_candidate_id": cls._candidate_id(result),
                "rule_first_source": str(result.source or "unknown"),
                "rule_first_source_layer": source_layer,
                "rule_first_entity_type": str(result.entity_type or "UNKNOWN").upper(),
                "rule_first_recognition_channel": recognition_channel,
                "rule_first_review_only_candidate": bool(review_only),
                "rule_first_review_only_reasons": list(review_reasons),
            }
        )
        if trigger:
            metadata["rule_first_trigger"] = trigger
        return RecognizerResult(
            entity_type=result.entity_type,
            start=result.start,
            end=result.end,
            score=result.score,
            text=result.text,
            source=result.source,
            metadata=metadata,
        )

    @classmethod
    def _record_rejected_candidate(
        cls,
        result: RecognizerResult,
        *,
        reasons: Iterable[str],
        rejected_reason_counts: Counter[str],
        rejected_source_counts: Counter[str],
        rejected_type_counts: Counter[str],
        rejected_channel_counts: Counter[str],
        rejected_trigger_counts: Counter[str],
        rejected_source_layer_counts: Counter[str],
        rejected_review_only_reason_counts: Counter[str],
    ) -> None:
        metadata = dict(result.metadata or {})
        for reason in cls._clean_strings(reasons) or ("unknown_reject_reason",):
            rejected_reason_counts[reason] += 1
        rejected_source_counts[str(result.source or "unknown")] += 1
        rejected_type_counts[str(result.entity_type or "UNKNOWN").upper()] += 1
        rejected_channel_counts[str(metadata.get("rule_first_recognition_channel") or "unknown")] += 1
        rejected_source_layer_counts[str(metadata.get("rule_first_source_layer") or "unknown")] += 1
        trigger = str(metadata.get("rule_first_trigger") or metadata.get("trigger") or "")
        if trigger:
            rejected_trigger_counts[trigger] += 1
        for reason in cls._clean_strings(metadata.get("rule_first_review_only_reasons") or ()):
            rejected_review_only_reason_counts[reason] += 1

    @staticmethod
    def _reject_reasons(
        *,
        positive: Iterable[str],
        negative: Iterable[str],
        validators_failed: Iterable[str],
        repairs: Iterable[str],
    ) -> tuple[str, ...]:
        reasons = [str(item) for item in negative if str(item)]
        reasons.extend(f"validator_failed:{item}" for item in validators_failed if str(item))
        if not reasons and any(str(item) == "boundary_repair_rejected" for item in repairs):
            reasons.append("boundary_repair_rejected")
        if not reasons and positive:
            reasons.append("positive_signals_rejected")
        return tuple(sorted(set(reasons))) or ("unknown_reject_reason",)

    @classmethod
    def _is_review_only_rejected_candidate(cls, result: RecognizerResult) -> bool:
        metadata = dict(result.metadata or {})
        return bool(metadata.get("rule_first_review_only_candidate"))

    @classmethod
    def _review_only_rejected_candidate_reasons(
        cls,
        result: RecognizerResult,
        *,
        reasons: Iterable[str],
        metadata: dict[str, Any],
    ) -> tuple[bool, tuple[str, ...]]:
        reason_set = set(cls._clean_strings(reasons))
        hard_reject_reasons = {
            "empty_text",
            "identity_or_role_term",
            "field_label_only",
            "state_or_generic_sentence",
            "non_subject_action_or_function_term",
            "generic_organization_term",
            "position_title_not_person",
            "position_title_not_organization",
            "organization_shape_not_person",
            "address_label_only",
            "unsupported_internal_entity_type",
        }
        if reason_set & hard_reject_reasons:
            return False, ()
        review_reasons: list[str] = []
        trigger = str(metadata.get("trigger") or "")
        source_layer = str(metadata.get("source_layer") or source_layer_for_result(result) or "")
        docx_container_type = str(metadata.get("docx_container_type") or "")
        if cls._is_structure_reviewable_reject(
            result,
            reasons=reason_set,
            metadata=metadata,
            source_layer=source_layer,
            trigger=trigger,
            docx_container_type=docx_container_type,
        ):
            review_reasons.append("structure_weak_subject_rejected")
        if source_layer == "structure":
            review_reasons.append("structure_source_rejected")
        if trigger in {
            "alias_definition",
            "table_label",
            "docx_table_label_neighbor",
            "docx_table_stacked_label_neighbor",
            "docx_structure_unit_rule_pass",
            "function_action_boundary_short_org",
            "parallel_action_boundary_short_org",
            "right_boundary_short_org",
        }:
            review_reasons.append(f"trigger:{trigger}")
        if docx_container_type == "table_cell":
            review_reasons.append("docx_table_cell_rejected")
        if metadata.get("short_org_candidate"):
            review_reasons.append("short_org_candidate_rejected")
        if metadata.get("rejected_after_boundary_repair") or metadata.get("rejected_repaired_candidate"):
            review_reasons.append("boundary_repaired_candidate_rejected")
        if metadata.get("boundary_repair_rejected") or "boundary_repair_rejected" in set(reasons):
            review_reasons.append("boundary_repair_rejected")
        if metadata.get("official_institution") or metadata.get("official_institution_family"):
            review_reasons.append("official_institution_candidate_rejected")
        return bool(review_reasons), tuple(sorted(set(review_reasons)))

    @classmethod
    def _is_structure_reviewable_reject(
        cls,
        result: RecognizerResult,
        *,
        reasons: set[str],
        metadata: dict[str, Any],
        source_layer: str,
        trigger: str,
        docx_container_type: str,
    ) -> bool:
        if source_layer != "structure":
            return False
        if not cls._has_strong_structure_review_signal(
            metadata=metadata,
            trigger=trigger,
            docx_container_type=docx_container_type,
        ):
            return False
        reviewable_reasons = {
            "weak_organization_shape",
            "weak_person_shape",
            "weak_location_shape",
            "weak_government_shape",
            "person_shape",
            "person_shape_not_organization",
            "ambiguous_person_or_short_org_shape",
            "sentence_fragment",
            "long_narrative_fragment",
            "boundary_repair_rejected",
            "validator_failed:weak_organization_shape",
            "validator_failed:weak_person_shape",
            "validator_failed:weak_location_shape",
            "validator_failed:weak_government_shape",
        }
        if reasons & reviewable_reasons:
            return True
        normalized = normalize_entity_text(result.text)
        return bool(metadata.get("short_org_candidate") and len(normalized) >= 2)

    @staticmethod
    def _has_strong_structure_review_signal(
        *,
        metadata: dict[str, Any],
        trigger: str,
        docx_container_type: str,
    ) -> bool:
        if trigger in {
            "alias_definition",
            "table_label",
            "docx_table_label_neighbor",
            "docx_table_stacked_label_neighbor",
            "docx_structure_unit_rule_pass",
            "function_action_boundary_short_org",
            "parallel_action_boundary_short_org",
            "right_boundary_short_org",
        }:
            return True
        if docx_container_type == "table_cell":
            return True
        if metadata.get("short_org_candidate"):
            return True
        return False

    @classmethod
    def _recognition_channel_for(cls, result: RecognizerResult) -> str:
        metadata = dict(result.metadata or {})
        explicit = str(metadata.get("recognition_channel") or "").strip()
        if explicit:
            return explicit
        trigger = str(metadata.get("trigger") or "").strip()
        source = str(result.source or "").strip()
        if source == "rule_official_institution" or metadata.get("official_institution"):
            return "official_institution"
        if source == "rule_docx_structure":
            container = str(metadata.get("docx_container_type") or "").strip()
            if trigger:
                return f"docx_structure:{trigger}"
            if container:
                return f"docx_structure:{container}"
            return "docx_structure"
        if source == "rule_organization_context":
            return f"organization_context:{trigger}" if trigger else "organization_context"
        if source == "rule_organization":
            if metadata.get("short_org_candidate"):
                return "organization_structure_short"
            if metadata.get("label"):
                return "organization_structure_label"
            return "organization_full_name"
        if source == "rule_alias" or trigger == "alias_definition":
            return "alias_definition"
        if source == "contract_structure_backfill":
            return f"contract_structure:{trigger}" if trigger else "contract_structure"
        if source:
            return source
        return "unknown"

    @staticmethod
    def _clean_strings(values: Iterable[Any]) -> tuple[str, ...]:
        if isinstance(values, str):
            value = values.strip()
            return (value,) if value else ()
        return tuple(str(item).strip() for item in values if str(item).strip())

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
        metadata = dict(result.metadata or {})
        if negative or validators_failed:
            return "review"
        if metadata.get("requires_manual_review") or metadata.get("short_org_candidate"):
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
