"""Unified subject admission gate for the default desensitization line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.core.recognizer_base import RecognizerResult
from app.rules.evidence import source_layer_for_result
from app.services.lowmem_entity_utils import (
    is_non_subject_action_or_function_term,
    is_non_subject_generic_org_reference,
    is_weak_function_stripped_org,
    looks_like_organization_short_name,
    normalize_entity_text,
    subject_noun_gate,
)


DEFAULT_SUBJECT_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "GOVERNMENT"}


@dataclass(frozen=True)
class SubjectAdmissionDecision:
    action: str
    risk_level: str
    positive_signals: tuple[str, ...]
    negative_signals: tuple[str, ...]
    source_layer: str = ""
    entity_type: str = ""

    @property
    def rejected(self) -> bool:
        return self.action == "reject"

    def to_metadata(self) -> dict[str, object]:
        return {
            "action": self.action,
            "risk_level": self.risk_level,
            "positive_signals": list(self.positive_signals),
            "negative_signals": list(self.negative_signals),
            "source_layer": self.source_layer,
            "entity_type": self.entity_type,
        }


class SubjectAdmissionGate:
    """One observable entrance for post-boundary subject candidates.

    This gate does not replace recall or boundary repair. It decides whether a
    candidate that already reached the post-boundary layer is accepted, sent to
    review, or rejected, while keeping the decision reason visible in metadata.
    """

    def decide(
        self,
        result: RecognizerResult,
        *,
        false_positive_reject: bool,
        positive_signals: Iterable[str],
        negative_signals: Iterable[str],
        validators_failed: Iterable[str],
    ) -> SubjectAdmissionDecision:
        entity_type = str(result.entity_type or "").strip().upper()
        normalized = normalize_entity_text(result.text)
        positives = set(self._clean_strings(positive_signals))
        negatives = set(self._clean_strings(negative_signals))
        validator_failures = set(self._clean_strings(validators_failed))

        if entity_type in DEFAULT_SUBJECT_TYPES:
            allow_short_org = entity_type == "ORGANIZATION" and looks_like_organization_short_name(normalized)
            passed, reason = subject_noun_gate(entity_type, normalized, allow_short_org=allow_short_org)
            if passed:
                positives.add(reason)
            else:
                negatives.add(reason)

        if false_positive_reject:
            action = "reject"
        elif validator_failures:
            action = "review"
        elif negatives:
            action = "review"
        else:
            action = self._default_action(result)

        return SubjectAdmissionDecision(
            action=action,
            risk_level=self._risk_level(result, action=action, negatives=negatives, validators_failed=validator_failures),
            positive_signals=tuple(sorted(positives)),
            negative_signals=tuple(sorted(negatives)),
            source_layer=source_layer_for_result(result),
            entity_type=entity_type,
        )

    @classmethod
    def passes_subject_shape(
        cls,
        entity_type: str,
        text: str,
        *,
        allow_short_org: bool = False,
    ) -> tuple[bool, str]:
        """Shared shape check for recall-time pruning before boundary/gate."""

        normalized_type = str(entity_type or "").strip().upper()
        normalized = normalize_entity_text(text)
        if normalized_type == "ORGANIZATION" and allow_short_org:
            allow_short_org = looks_like_organization_short_name(normalized)
        return subject_noun_gate(normalized_type, normalized, allow_short_org=allow_short_org)

    @staticmethod
    def is_action_or_function_text(text: str) -> bool:
        return is_non_subject_action_or_function_term(normalize_entity_text(text))

    @staticmethod
    def is_generic_org_reference(text: str) -> bool:
        return is_non_subject_generic_org_reference(normalize_entity_text(text))

    @classmethod
    def is_non_subject_expression(cls, text: str) -> bool:
        normalized = normalize_entity_text(text)
        return cls.is_generic_org_reference(normalized) or cls.is_action_or_function_text(normalized)

    @staticmethod
    def is_weak_function_stripped_subject(text: str) -> bool:
        return is_weak_function_stripped_org(text)

    @staticmethod
    def _default_action(result: RecognizerResult) -> str:
        entity_type = str(result.entity_type or "").strip().upper()
        metadata = dict(result.metadata or {})
        if metadata.get("requires_manual_review") or metadata.get("short_org_candidate"):
            return "review"
        if source_layer_for_result(result) in {"regex", "structure"}:
            return "keep"
        if entity_type in {"PERSON", "ORGANIZATION", "GOVERNMENT"} and len(normalize_entity_text(result.text)) <= 3:
            return "review"
        return "keep"

    @staticmethod
    def _risk_level(
        result: RecognizerResult,
        *,
        action: str,
        negatives: set[str],
        validators_failed: set[str],
    ) -> str:
        entity_type = str(result.entity_type or "").strip().upper()
        if action == "reject" or validators_failed:
            return "high"
        if negatives:
            return "high"
        if entity_type in {"PERSON", "ORGANIZATION", "GOVERNMENT"} and len(normalize_entity_text(result.text)) <= 3:
            return "medium"
        return "low"

    @staticmethod
    def _clean_strings(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(str(item).strip() for item in values or () if str(item).strip())
