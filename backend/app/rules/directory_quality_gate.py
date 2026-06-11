"""Directory-level rule quality checks before replacement/export."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.core.recognizer_base import RecognizerResult
from app.services.lowmem_entity_utils import normalize_entity_text


class DirectoryQualityGate:
    """Cheap deterministic checks that expose directory risks early."""

    def evaluate(self, results: list[RecognizerResult]) -> dict[str, Any]:
        by_surface: dict[str, set[str]] = defaultdict(set)
        by_subject: dict[str, set[str]] = defaultdict(set)
        unreviewed_high_risk = 0
        review_required = 0
        unmapped_docx = 0
        hard_identity_conflict = 0
        for result in results:
            metadata = dict(result.metadata or {})
            normalized = normalize_entity_text(result.text)
            if normalized:
                by_surface[normalized].add(str(result.entity_type or "").upper())
            canonical_key = str(metadata.get("canonical_key") or "").strip()
            if canonical_key:
                by_subject[canonical_key].add(normalized or str(result.text or ""))
            rule_first = metadata.get("rule_first") if isinstance(metadata.get("rule_first"), dict) else {}
            if rule_first.get("risk_level") == "high" and rule_first.get("action") == "review":
                unreviewed_high_risk += 1
            if rule_first.get("action") == "review" or metadata.get("requires_manual_review"):
                review_required += 1
            if metadata.get("docx_review_required"):
                unmapped_docx += 1
            subject_graph = metadata.get("rule_subject_graph")
            if isinstance(subject_graph, dict) and "hard_identity_conflict" in set(subject_graph.get("risk_flags") or []):
                hard_identity_conflict += 1

        same_surface_multi_type = {
            surface: sorted(types)
            for surface, types in by_surface.items()
            if len(types) > 1
        }
        split_subjects = {
            key: sorted(values)
            for key, values in by_subject.items()
            if len(values) > 1
        }
        blocking_issue_count = (
            len(same_surface_multi_type)
            + unreviewed_high_risk
            + unmapped_docx
            + hard_identity_conflict
        )
        return {
            "directory_quality_gate_enabled": True,
            "directory_quality_gate_passed": blocking_issue_count == 0,
            "directory_quality_gate_blocking_issue_count": blocking_issue_count,
            "same_surface_multi_type_unresolved_count": len(same_surface_multi_type),
            "same_surface_multi_type_unresolved": same_surface_multi_type,
            "high_risk_subject_unreviewed_count": unreviewed_high_risk,
            "rule_first_review_required_entity_count": review_required,
            "unmapped_occurrence_count": unmapped_docx,
            "hard_identity_conflict_count": hard_identity_conflict,
            "same_subject_surface_variant_count": len(split_subjects),
            "same_subject_surface_variants": split_subjects,
        }
