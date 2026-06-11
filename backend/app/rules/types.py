"""Shared rule-first data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RiskLevel = Literal["low", "medium", "high"]
RuleAction = Literal["keep", "reject", "repair", "review"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    entity_type: str
    text: str
    start: int
    end: int
    source: str
    source_layer: str
    confidence_raw: float
    metadata: dict[str, Any] = field(default_factory=dict)
    unit_id: str = ""


@dataclass(frozen=True)
class RuleEvidence:
    candidate_id: str
    entity_type: str
    text: str
    positive_signals: tuple[str, ...] = ()
    negative_signals: tuple[str, ...] = ()
    validators_passed: tuple[str, ...] = ()
    validators_failed: tuple[str, ...] = ()
    boundary_repairs: tuple[str, ...] = ()
    confidence: float = 0.0
    risk_level: RiskLevel = "medium"
    action: RuleAction = "review"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "entity_type": self.entity_type,
            "positive_signals": list(self.positive_signals),
            "negative_signals": list(self.negative_signals),
            "validators_passed": list(self.validators_passed),
            "validators_failed": list(self.validators_failed),
            "boundary_repairs": list(self.boundary_repairs),
            "confidence": float(self.confidence),
            "risk_level": self.risk_level,
            "action": self.action,
        }


@dataclass
class SubjectCard:
    subject_id: str
    entity_type: str
    canonical_text: str
    surfaces: set[str] = field(default_factory=set)
    occurrence_ids: list[str] = field(default_factory=list)
    source_layers: set[str] = field(default_factory=set)
    risk_flags: set[str] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)
    canonical_key: str = ""
    unresolved: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "entity_type": self.entity_type,
            "canonical_text": self.canonical_text,
            "surfaces": sorted(self.surfaces),
            "source_layers": sorted(self.source_layers),
            "risk_flags": sorted(self.risk_flags),
            "evidence": list(self.evidence),
            "canonical_key": self.canonical_key or self.subject_id,
            "unresolved": bool(self.unresolved),
        }
