"""General evaluation utilities for rule-first legal-practice desensitization.

The evaluator is deliberately separate from recognizers. It measures whether a
default-line run satisfies generic legal-practice quality properties; it does
not feed sample-derived rules back into recognition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from app.core.recognizer_base import RecognizerResult
from app.services.lowmem_entity_utils import normalize_entity_text


@dataclass(frozen=True)
class ExpectedEntity:
    entity_type: str
    text: str
    start: int
    end: int
    subject_key: str = ""
    replacement: str = ""
    unit_id: str = ""
    requires_rewrite: bool = True


@dataclass(frozen=True)
class ActualEntity:
    entity_type: str
    text: str
    start: int
    end: int
    source: str = ""
    score: float = 0.0
    subject_key: str = ""
    replacement: str = ""
    unit_id: str = ""
    rewritable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityError:
    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None
    reason: str = ""


@dataclass
class RuleEvaluationReport:
    expected_count: int
    actual_count: int
    true_positive_count: int
    false_positive_count: int
    missed_count: int
    type_error_count: int
    boundary_error_count: int
    subject_split_count: int
    wrong_merge_count: int
    subject_multi_replacement_count: int
    replacement_reused_by_multi_subject_count: int
    docx_unmapped_count: int
    docx_unrewritable_count: int
    by_type: dict[str, dict[str, float | int]]
    false_positives: list[EntityError] = field(default_factory=list)
    missed_entities: list[EntityError] = field(default_factory=list)
    type_errors: list[EntityError] = field(default_factory=list)
    boundary_errors: list[EntityError] = field(default_factory=list)
    subject_splits: dict[str, list[str]] = field(default_factory=dict)
    wrong_merges: dict[str, list[str]] = field(default_factory=dict)
    replacement_conflicts: dict[str, Any] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        denominator = self.true_positive_count + self.false_positive_count
        return _safe_ratio(self.true_positive_count, denominator)

    @property
    def recall(self) -> float:
        denominator = self.true_positive_count + self.missed_count
        return _safe_ratio(self.true_positive_count, denominator)

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        if precision + recall <= 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @property
    def quality_gate_passed(self) -> bool:
        return all(
            value == 0
            for value in (
                self.false_positive_count,
                self.missed_count,
                self.type_error_count,
                self.boundary_error_count,
                self.subject_split_count,
                self.wrong_merge_count,
                self.subject_multi_replacement_count,
                self.replacement_reused_by_multi_subject_count,
                self.docx_unmapped_count,
                self.docx_unrewritable_count,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_gate_passed": self.quality_gate_passed,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "missed_count": self.missed_count,
            "type_error_count": self.type_error_count,
            "boundary_error_count": self.boundary_error_count,
            "subject_split_count": self.subject_split_count,
            "wrong_merge_count": self.wrong_merge_count,
            "subject_multi_replacement_count": self.subject_multi_replacement_count,
            "replacement_reused_by_multi_subject_count": self.replacement_reused_by_multi_subject_count,
            "docx_unmapped_count": self.docx_unmapped_count,
            "docx_unrewritable_count": self.docx_unrewritable_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "by_type": self.by_type,
            "false_positives": [_error_to_dict(item) for item in self.false_positives],
            "missed_entities": [_error_to_dict(item) for item in self.missed_entities],
            "type_errors": [_error_to_dict(item) for item in self.type_errors],
            "boundary_errors": [_error_to_dict(item) for item in self.boundary_errors],
            "subject_splits": self.subject_splits,
            "wrong_merges": self.wrong_merges,
            "replacement_conflicts": self.replacement_conflicts,
        }


class RuleFirstEvaluator:
    """Evaluate entity, subject, replacement, and DOCX-location invariants."""

    def evaluate(
        self,
        *,
        expected: Iterable[ExpectedEntity | Mapping[str, Any]],
        actual: Iterable[ActualEntity | RecognizerResult | Mapping[str, Any]],
    ) -> RuleEvaluationReport:
        expected_entities = [self._expected_entity(item) for item in expected]
        actual_entities = [self._actual_entity(item) for item in actual]

        exact_matches: list[tuple[int, int]] = []
        used_expected: set[int] = set()
        used_actual: set[int] = set()
        for expected_index, expected_entity in enumerate(expected_entities):
            for actual_index, actual_entity in enumerate(actual_entities):
                if actual_index in used_actual:
                    continue
                if self._is_exact_match(expected_entity, actual_entity):
                    exact_matches.append((expected_index, actual_index))
                    used_expected.add(expected_index)
                    used_actual.add(actual_index)
                    break

        type_errors: list[EntityError] = []
        boundary_errors: list[EntityError] = []
        resolved_expected = set(used_expected)
        resolved_actual = set(used_actual)

        for expected_index, expected_entity in enumerate(expected_entities):
            if expected_index in resolved_expected:
                continue
            candidates = [
                (actual_index, actual_entity)
                for actual_index, actual_entity in enumerate(actual_entities)
                if actual_index not in resolved_actual and self._same_span_or_text(expected_entity, actual_entity)
            ]
            if candidates:
                actual_index, actual_entity = candidates[0]
                type_errors.append(
                    EntityError(
                        expected=_expected_to_dict(expected_entity),
                        actual=_actual_to_dict(actual_entity),
                        reason="same_span_or_text_different_type",
                    )
                )
                resolved_expected.add(expected_index)
                resolved_actual.add(actual_index)
                continue

            candidates = [
                (actual_index, actual_entity)
                for actual_index, actual_entity in enumerate(actual_entities)
                if actual_index not in resolved_actual and self._same_type_overlap(expected_entity, actual_entity)
            ]
            if candidates:
                actual_index, actual_entity = candidates[0]
                boundary_errors.append(
                    EntityError(
                        expected=_expected_to_dict(expected_entity),
                        actual=_actual_to_dict(actual_entity),
                        reason="same_type_overlapping_span",
                    )
                )
                resolved_expected.add(expected_index)
                resolved_actual.add(actual_index)

        missed_entities = [
            EntityError(expected=_expected_to_dict(entity), reason="not_detected")
            for index, entity in enumerate(expected_entities)
            if index not in resolved_expected
        ]
        false_positives = [
            EntityError(actual=_actual_to_dict(entity), reason="unexpected_entity")
            for index, entity in enumerate(actual_entities)
            if index not in resolved_actual
        ]

        by_type = self._type_metrics(
            expected_entities=expected_entities,
            actual_entities=actual_entities,
            exact_matches=exact_matches,
            false_positive_indices={index for index in range(len(actual_entities)) if index not in resolved_actual},
            missed_indices={index for index in range(len(expected_entities)) if index not in resolved_expected},
        )
        subject_report = self._subject_metrics(
            expected_entities=expected_entities,
            actual_entities=actual_entities,
            exact_matches=exact_matches,
        )
        replacement_report = self._replacement_metrics(actual_entities)
        docx_report = self._docx_metrics(actual_entities)

        return RuleEvaluationReport(
            expected_count=len(expected_entities),
            actual_count=len(actual_entities),
            true_positive_count=len(exact_matches),
            false_positive_count=len(false_positives),
            missed_count=len(missed_entities),
            type_error_count=len(type_errors),
            boundary_error_count=len(boundary_errors),
            subject_split_count=len(subject_report["subject_splits"]),
            wrong_merge_count=len(subject_report["wrong_merges"]),
            subject_multi_replacement_count=len(replacement_report["subject_multi_replacement"]),
            replacement_reused_by_multi_subject_count=len(replacement_report["replacement_reused_by_multi_subject"]),
            docx_unmapped_count=docx_report["docx_unmapped_count"],
            docx_unrewritable_count=docx_report["docx_unrewritable_count"],
            by_type=by_type,
            false_positives=false_positives,
            missed_entities=missed_entities,
            type_errors=type_errors,
            boundary_errors=boundary_errors,
            subject_splits=subject_report["subject_splits"],
            wrong_merges=subject_report["wrong_merges"],
            replacement_conflicts=replacement_report,
        )

    @staticmethod
    def _expected_entity(item: ExpectedEntity | Mapping[str, Any]) -> ExpectedEntity:
        if isinstance(item, ExpectedEntity):
            return item
        return ExpectedEntity(
            entity_type=_type_of(item),
            text=str(item.get("text") or ""),
            start=int(item.get("start") or 0),
            end=int(item.get("end") or 0),
            subject_key=str(item.get("subject_key") or item.get("canonical_key") or ""),
            replacement=str(item.get("replacement") or ""),
            unit_id=str(item.get("unit_id") or ""),
            requires_rewrite=bool(item.get("requires_rewrite", True)),
        )

    @staticmethod
    def _actual_entity(item: ActualEntity | RecognizerResult | Mapping[str, Any]) -> ActualEntity:
        if isinstance(item, ActualEntity):
            return item
        if isinstance(item, RecognizerResult):
            metadata = dict(item.metadata or {})
            return ActualEntity(
                entity_type=str(item.entity_type or "").upper(),
                text=str(item.text or ""),
                start=int(item.start),
                end=int(item.end),
                source=str(item.source or ""),
                score=float(item.score or 0.0),
                subject_key=_subject_key_from(metadata),
                replacement=str(metadata.get("replacement") or ""),
                unit_id=_unit_id_from(metadata),
                rewritable=not _metadata_marks_unrewritable(metadata),
                metadata=metadata,
            )
        metadata = dict(item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {})
        return ActualEntity(
            entity_type=_type_of(item),
            text=str(item.get("text") or ""),
            start=int(item.get("start") or 0),
            end=int(item.get("end") or 0),
            source=str(item.get("source") or ""),
            score=float(item.get("score") or 0.0),
            subject_key=str(item.get("subject_key") or item.get("canonical_key") or metadata.get("canonical_key") or ""),
            replacement=str(item.get("replacement") or metadata.get("replacement") or ""),
            unit_id=str(item.get("unit_id") or metadata.get("unit_id") or metadata.get("docx_unit_id") or ""),
            rewritable=not _metadata_marks_unrewritable(metadata)
            and not bool(item.get("docx_review_required") or item.get("unrewritable")),
            metadata=metadata,
        )

    @staticmethod
    def _is_exact_match(expected: ExpectedEntity, actual: ActualEntity) -> bool:
        return (
            _normalize_type(expected.entity_type) == _normalize_type(actual.entity_type)
            and expected.start == actual.start
            and expected.end == actual.end
            and normalize_entity_text(expected.text) == normalize_entity_text(actual.text)
        )

    @staticmethod
    def _same_span_or_text(expected: ExpectedEntity, actual: ActualEntity) -> bool:
        return (
            expected.start == actual.start
            and expected.end == actual.end
            or normalize_entity_text(expected.text) == normalize_entity_text(actual.text)
        )

    @staticmethod
    def _same_type_overlap(expected: ExpectedEntity, actual: ActualEntity) -> bool:
        return (
            _normalize_type(expected.entity_type) == _normalize_type(actual.entity_type)
            and expected.start < actual.end
            and actual.start < expected.end
        )

    @staticmethod
    def _type_metrics(
        *,
        expected_entities: list[ExpectedEntity],
        actual_entities: list[ActualEntity],
        exact_matches: list[tuple[int, int]],
        false_positive_indices: set[int],
        missed_indices: set[int],
    ) -> dict[str, dict[str, float | int]]:
        true_by_type: dict[str, int] = defaultdict(int)
        expected_by_type: dict[str, int] = defaultdict(int)
        actual_by_type: dict[str, int] = defaultdict(int)
        fp_by_type: dict[str, int] = defaultdict(int)
        missed_by_type: dict[str, int] = defaultdict(int)

        for entity in expected_entities:
            expected_by_type[_normalize_type(entity.entity_type)] += 1
        for entity in actual_entities:
            actual_by_type[_normalize_type(entity.entity_type)] += 1
        for expected_index, _ in exact_matches:
            true_by_type[_normalize_type(expected_entities[expected_index].entity_type)] += 1
        for index in false_positive_indices:
            fp_by_type[_normalize_type(actual_entities[index].entity_type)] += 1
        for index in missed_indices:
            missed_by_type[_normalize_type(expected_entities[index].entity_type)] += 1

        output: dict[str, dict[str, float | int]] = {}
        for entity_type in sorted(set(expected_by_type) | set(actual_by_type)):
            true_count = true_by_type[entity_type]
            false_count = fp_by_type[entity_type]
            missed_count = missed_by_type[entity_type]
            precision = _safe_ratio(true_count, true_count + false_count)
            recall = _safe_ratio(true_count, true_count + missed_count)
            output[entity_type] = {
                "expected": expected_by_type[entity_type],
                "actual": actual_by_type[entity_type],
                "true_positive": true_count,
                "false_positive": false_count,
                "missed": missed_count,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        return output

    @staticmethod
    def _subject_metrics(
        *,
        expected_entities: list[ExpectedEntity],
        actual_entities: list[ActualEntity],
        exact_matches: list[tuple[int, int]],
    ) -> dict[str, dict[str, list[str]]]:
        expected_to_actual: dict[str, set[str]] = defaultdict(set)
        actual_to_expected: dict[str, set[str]] = defaultdict(set)
        for expected_index, actual_index in exact_matches:
            expected_key = expected_entities[expected_index].subject_key
            actual_key = actual_entities[actual_index].subject_key
            if expected_key and actual_key:
                expected_to_actual[expected_key].add(actual_key)
                actual_to_expected[actual_key].add(expected_key)

        return {
            "subject_splits": {
                expected_key: sorted(actual_keys)
                for expected_key, actual_keys in expected_to_actual.items()
                if len(actual_keys) > 1
            },
            "wrong_merges": {
                actual_key: sorted(expected_keys)
                for actual_key, expected_keys in actual_to_expected.items()
                if len(expected_keys) > 1
            },
        }

    @staticmethod
    def _replacement_metrics(actual_entities: list[ActualEntity]) -> dict[str, Any]:
        by_subject: dict[str, set[str]] = defaultdict(set)
        by_replacement: dict[str, set[str]] = defaultdict(set)
        for entity in actual_entities:
            if entity.subject_key and entity.replacement:
                by_subject[entity.subject_key].add(entity.replacement)
                by_replacement[entity.replacement].add(entity.subject_key)
        return {
            "subject_multi_replacement": {
                key: sorted(values) for key, values in by_subject.items() if len(values) > 1
            },
            "replacement_reused_by_multi_subject": {
                key: sorted(values) for key, values in by_replacement.items() if len(values) > 1
            },
        }

    @staticmethod
    def _docx_metrics(actual_entities: list[ActualEntity]) -> dict[str, int]:
        unmapped = 0
        unrewritable = 0
        for entity in actual_entities:
            metadata = dict(entity.metadata or {})
            if metadata.get("requires_docx_mapping") and not entity.unit_id:
                unmapped += 1
            if not entity.rewritable:
                unrewritable += 1
        return {
            "docx_unmapped_count": unmapped,
            "docx_unrewritable_count": unrewritable,
        }


def _normalize_type(value: str) -> str:
    normalized = str(value or "").strip().upper()
    aliases = {
        "PERSON_NAME": "PERSON",
        "COMPANY_NAME": "ORGANIZATION",
        "LOCATION": "ADDRESS",
        "ID_CARD": "CN_ID_CARD",
        "PHONE": "CN_PHONE",
        "BANK_CARD": "CN_BANK_CARD",
    }
    return aliases.get(normalized, normalized)


def _type_of(item: Mapping[str, Any]) -> str:
    return _normalize_type(str(item.get("type") or item.get("entity_type") or ""))


def _subject_key_from(metadata: Mapping[str, Any]) -> str:
    return str(
        metadata.get("canonical_key")
        or metadata.get("subject_key")
        or metadata.get("subject_id")
        or ""
    )


def _unit_id_from(metadata: Mapping[str, Any]) -> str:
    return str(
        metadata.get("unit_id")
        or metadata.get("docx_unit_id")
        or metadata.get("coverage_first_unit_id")
        or ""
    )


def _metadata_marks_unrewritable(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get("docx_review_required")
        or metadata.get("docx_unrewritable")
        or metadata.get("rewrite_unavailable")
        or metadata.get("unrewritable")
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(numerator) / float(denominator))), 4)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _expected_to_dict(entity: ExpectedEntity) -> dict[str, Any]:
    return {
        "type": _normalize_type(entity.entity_type),
        "text": entity.text,
        "start": entity.start,
        "end": entity.end,
        "subject_key": entity.subject_key,
        "replacement": entity.replacement,
        "unit_id": entity.unit_id,
        "requires_rewrite": entity.requires_rewrite,
    }


def _actual_to_dict(entity: ActualEntity) -> dict[str, Any]:
    return {
        "type": _normalize_type(entity.entity_type),
        "text": entity.text,
        "start": entity.start,
        "end": entity.end,
        "source": entity.source,
        "score": entity.score,
        "subject_key": entity.subject_key,
        "replacement": entity.replacement,
        "unit_id": entity.unit_id,
        "rewritable": entity.rewritable,
    }


def _error_to_dict(error: EntityError) -> dict[str, Any]:
    return {
        "expected": error.expected,
        "actual": error.actual,
        "reason": error.reason,
    }
