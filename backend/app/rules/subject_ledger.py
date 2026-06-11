"""Read-only subject ledger for the default rule-first workflow.

The ledger is deliberately observational in this phase: it records occurrences,
subject cards, identity edges, and review decisions from already accepted rule
results. It must not add, drop, merge, or reorder recognizer results.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

from app.core.recognizer_base import RecognizerResult
from app.rules.default_subject_policy import DEFAULT_SUBJECT_TYPES, canonical_default_entity_type, projected_default_subject_type
from app.rules.evidence import source_layer_for_result
from app.rules.types import SubjectCard
from app.services.lowmem_entity_utils import (
    IDENTITY_REFERENCE_TERMS,
    NON_ENTITY_ROLE_TERMS,
    is_org_like_text,
    is_probable_person,
    looks_like_organization_short_name,
    normalize_entity_text,
)


SUBJECT_ENTITY_TYPES = {
    *DEFAULT_SUBJECT_TYPES,
    "PERSON_NAME",
    "LEGAL_REPRESENTATIVE",
    "CONTACT_PERSON",
    "SIGNATORY",
    "ORGANIZATION",
    "COMPANY_NAME",
    "ACCOUNT_NAME",
    "BANK_NAME",
    "ALIAS",
    "LOCATION",
    "ADDRESS",
    "GOVERNMENT",
    "GOVERNMENT_AGENCY",
    "COURT",
}

ORGANIZATION_FAMILY_TYPES = {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "ALIAS"}
PERSON_FAMILY_TYPES = {"PERSON", "PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}
LOCATION_FAMILY_TYPES = {"LOCATION", "ADDRESS"}
GOVERNMENT_FAMILY_TYPES = {"GOVERNMENT", "GOVERNMENT_AGENCY", "COURT", "BANK_NAME"}
ORG_REGION_PREFIX_PATTERN = re.compile(
    r"^(?:(?:[\u4e00-\u9fa5]{2,16}(?:省|自治区|特别行政区|市|地区|自治州|盟|区|县|旗|自治县|自治旗|镇|乡|街道))+|"
    r"(?:北京|上海|天津|重庆|广州|深圳|杭州|南京|苏州|成都|武汉|西安|长沙|郑州|青岛|宁波|佛山|东莞|"
    r"厦门|福州|济南|合肥|昆明|南宁|贵阳|南昌|海口|太原|沈阳|长春|哈尔滨|石家庄|呼和浩特|"
    r"乌鲁木齐|拉萨|银川|西宁|广东|广西|海南|河北|河南|湖北|湖南|江苏|浙江|安徽|福建|"
    r"江西|山东|山西|陕西|四川|贵州|云南|辽宁|吉林|黑龙江|甘肃|青海|台湾|内蒙古|宁夏|新疆|西藏|香港|澳门))"
)
ORG_LEGAL_SUFFIX_PATTERN = re.compile(
    r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司|集团|分公司|子公司|公司|商行|工作室|合作社|经营部|"
    r"银行|支行|分行|营业部|研究院|研究所|服务中心|技术中心|事务所)$"
)
ORG_BUSINESS_SUFFIX_TERMS = tuple(
    sorted(
        {
            "建筑劳务",
            "工程建设",
            "工程技术",
            "工程设计",
            "建筑工程",
            "新能源",
            "新材料",
            "信息技术",
            "技术服务",
            "商务服务",
            "设计咨询",
            "检测技术",
            "供应链服务",
            "供应链",
            "科技",
            "工程",
            "建设",
            "贸易",
            "实业",
            "发展",
            "咨询",
            "服务",
            "管理",
            "材料",
            "电力",
            "能源",
            "建筑",
            "环保",
            "智能",
            "信息",
            "网络",
            "电子",
            "机械",
            "设备",
            "制造",
            "劳务",
        },
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class LedgerOccurrence:
    occurrence_id: str
    entity_type: str
    text: str
    normalized_text: str
    start: int
    end: int
    source: str
    source_layer: str
    score: float
    subject_id: str
    status: str
    risk_flags: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    docx_unit_id: str = ""
    canonical_key: str = ""
    canonical_role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "entity_type": self.entity_type,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "source_layer": self.source_layer,
            "score": float(self.score),
            "subject_id": self.subject_id,
            "status": self.status,
            "risk_flags": list(self.risk_flags),
            "evidence": list(self.evidence),
            "docx_unit_id": self.docx_unit_id,
            "canonical_key": self.canonical_key,
            "canonical_role": self.canonical_role,
        }


@dataclass
class LedgerSubject:
    subject_id: str
    family: str
    primary_type: str
    canonical_text: str
    canonical_key: str
    canonical_role: str = ""
    occurrence_ids: list[str] = field(default_factory=list)
    surfaces: set[str] = field(default_factory=set)
    source_layers: set[str] = field(default_factory=set)
    evidence: set[str] = field(default_factory=set)
    risk_flags: set[str] = field(default_factory=set)
    status: str = "confirmed_subject"
    replacement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "family": self.family,
            "primary_type": self.primary_type,
            "canonical_text": self.canonical_text,
            "canonical_key": self.canonical_key,
            "canonical_role": self.canonical_role,
            "occurrence_ids": list(self.occurrence_ids),
            "surfaces": sorted(self.surfaces),
            "source_layers": sorted(self.source_layers),
            "evidence": sorted(self.evidence),
            "risk_flags": sorted(self.risk_flags),
            "status": self.status,
            "replacement": self.replacement,
        }


@dataclass(frozen=True)
class LedgerEdge:
    edge_id: str
    source_subject_id: str
    target_subject_id: str
    relation: str
    evidence: tuple[str, ...] = ()
    confidence: float = 0.0
    status: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_subject_id": self.source_subject_id,
            "target_subject_id": self.target_subject_id,
            "relation": self.relation,
            "evidence": list(self.evidence),
            "confidence": float(self.confidence),
            "status": self.status,
        }


@dataclass(frozen=True)
class LedgerDecision:
    decision_id: str
    subject_id: str
    occurrence_id: str
    action: str
    reason: str
    risk_level: str
    evidence: tuple[str, ...] = ()
    decision_scope: str = "occurrence"
    edge_id: str = ""
    source_subject_id: str = ""
    target_subject_id: str = ""
    edge_relation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "subject_id": self.subject_id,
            "occurrence_id": self.occurrence_id,
            "action": self.action,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "evidence": list(self.evidence),
            "decision_scope": self.decision_scope,
            "edge_id": self.edge_id,
            "source_subject_id": self.source_subject_id,
            "target_subject_id": self.target_subject_id,
            "edge_relation": self.edge_relation,
        }


@dataclass(frozen=True)
class SubjectLedgerBuildResult:
    results: list[RecognizerResult]
    ledger: dict[str, Any]
    summary: dict[str, Any]
    subject_graph_summary: dict[str, Any]


@dataclass(frozen=True)
class SubjectLedgerAdjudicationResult:
    ledger: dict[str, Any]
    summary: dict[str, Any]
    subject_id_map: dict[str, str]


class SubjectLedgerAdjudicationResolver:
    """Apply model adjudication decisions to a rule-first subject ledger."""

    EDGE_STATUS_BY_ACTION = {
        "merge_to_canonical": "adjudicated_merge",
        "keep_separate": "adjudicated_keep_separate",
        "confirm": "adjudicated_confirm_source",
        "manual_review": "manual_review_required",
        "reject": "adjudicated_reject_source",
    }

    TERMINAL_EDGE_STATUSES = {
        "adjudicated_merge",
        "adjudicated_keep_separate",
        "adjudicated_confirm_source",
        "adjudicated_reject_source",
    }

    def apply(self, ledger: dict[str, Any], decisions: list[dict[str, Any]]) -> SubjectLedgerAdjudicationResult:
        source_ledger = ledger if isinstance(ledger, dict) else {}
        decision_rows = [dict(item) for item in decisions or [] if isinstance(item, dict)]
        edge_decisions = {
            str(item.get("edge_id") or "").strip(): item
            for item in decision_rows
            if str(item.get("edge_id") or "").strip()
        }
        subject_id_map = self._build_subject_id_map(edge_decisions)
        resolved_edges = self._resolve_edges(source_ledger.get("edges") or [], edge_decisions)
        resolved_occurrences = self._resolve_occurrences(source_ledger.get("occurrences") or [], decision_rows, subject_id_map)
        resolved_subjects = self._resolve_subjects(source_ledger.get("subjects") or [], resolved_occurrences, subject_id_map)
        unresolved_subject_ids = {
            str(item.get("subject_id") or "")
            for item in resolved_occurrences
            if str(item.get("status") or "") in {
                "ambiguous_short_subject",
                "unresolved_alias",
                "weak_reference",
                "weak_identity_edge",
                "hard_conflict",
                "manual_review_required",
            }
        }
        resolved_decisions = self._resolve_decisions(source_ledger.get("decisions") or [], decision_rows, resolved_edges)
        resolved_ledger = {
            **source_ledger,
            "version": "subject_ledger.v1.resolved",
            "mode": "resolved",
            "occurrences": resolved_occurrences,
            "subjects": resolved_subjects,
            "edges": resolved_edges,
            "decisions": resolved_decisions,
            "adjudication_decisions": decision_rows,
            "subject_id_map": subject_id_map,
        }
        summary = self._summary(
            resolved_ledger,
            unresolved_subject_ids=unresolved_subject_ids,
        )
        return SubjectLedgerAdjudicationResult(
            ledger=resolved_ledger,
            summary=summary,
            subject_id_map=subject_id_map,
        )

    @staticmethod
    def _build_subject_id_map(edge_decisions: dict[str, dict[str, Any]]) -> dict[str, str]:
        subject_id_map: dict[str, str] = {}
        for decision in edge_decisions.values():
            action = str(decision.get("action") or "").strip().lower()
            if action != "merge_to_canonical":
                continue
            source_subject_id = str(
                decision.get("source_subject_id")
                or decision.get("subject_id")
                or ""
            ).strip()
            target_subject_id = str(
                decision.get("target_subject_id")
                or decision.get("canonical_subject_id")
                or ""
            ).strip()
            if source_subject_id and target_subject_id and source_subject_id != target_subject_id:
                subject_id_map[source_subject_id] = target_subject_id
        return subject_id_map

    def _resolve_edges(
        self,
        edges: list[Any],
        edge_decisions: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            row = dict(edge)
            edge_id = str(row.get("edge_id") or "").strip()
            decision = edge_decisions.get(edge_id)
            if decision:
                action = str(decision.get("action") or "").strip().lower()
                row["status"] = self.EDGE_STATUS_BY_ACTION.get(action, "manual_review_required")
                row["adjudication_action"] = action
                row["adjudication_confidence"] = self._coerce_float(decision.get("confidence"))
                row["adjudication_reason"] = str(decision.get("reason") or "")
                row["adjudicated_canonical_subject_id"] = str(
                    decision.get("target_subject_id")
                    or decision.get("canonical_subject_id")
                    or ""
                )
                row["requires_manual_review"] = bool(decision.get("requires_manual_review")) or action == "manual_review"
            resolved.append(row)
        return resolved

    def _resolve_occurrences(
        self,
        occurrences: list[Any],
        decisions: list[dict[str, Any]],
        subject_id_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        decisions_by_occurrence = {
            str(item.get("occurrence_id") or "").strip(): item
            for item in decisions
            if str(item.get("occurrence_id") or "").strip()
        }
        resolved: list[dict[str, Any]] = []
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            row = dict(occurrence)
            original_subject_id = str(row.get("subject_id") or "").strip()
            occurrence_id = str(row.get("occurrence_id") or "").strip()
            decision = decisions_by_occurrence.get(occurrence_id)
            action = str((decision or {}).get("action") or "").strip().lower()
            if original_subject_id in subject_id_map:
                row["original_subject_id"] = original_subject_id
                row["subject_id"] = subject_id_map[original_subject_id]
                row["status"] = "confirmed_subject"
                row["adjudication_action"] = "merge_to_canonical"
            elif action in {"confirm", "keep_separate"}:
                row["status"] = "confirmed_subject" if action == "confirm" else "confirmed_separate_subject"
                row["adjudication_action"] = action
            elif action == "reject":
                row["status"] = "rejected_subject"
                row["adjudication_action"] = action
            elif action == "manual_review":
                row["status"] = "manual_review_required"
                row["requires_manual_review"] = True
                row["adjudication_action"] = action
            if decision:
                row["adjudication_confidence"] = self._coerce_float(decision.get("confidence"))
                row["adjudication_reason"] = str(decision.get("reason") or "")
                row["adjudicated_edge_id"] = str(decision.get("edge_id") or "")
            resolved.append(row)
        return resolved

    def _resolve_subjects(
        self,
        subjects: list[Any],
        occurrences: list[dict[str, Any]],
        subject_id_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        source_subjects = [
            dict(subject)
            for subject in subjects
            if isinstance(subject, dict)
        ]
        occurrence_ids_by_subject: dict[str, list[str]] = defaultdict(list)
        surface_by_subject: dict[str, set[str]] = defaultdict(set)
        statuses_by_subject: dict[str, set[str]] = defaultdict(set)
        for occurrence in occurrences:
            subject_id = str(occurrence.get("subject_id") or "").strip()
            if not subject_id:
                continue
            occurrence_id = str(occurrence.get("occurrence_id") or "").strip()
            if occurrence_id:
                occurrence_ids_by_subject[subject_id].append(occurrence_id)
            text = str(occurrence.get("text") or "").strip()
            if text:
                surface_by_subject[subject_id].add(text)
            status = str(occurrence.get("status") or "").strip()
            if status:
                statuses_by_subject[subject_id].add(status)

        merged_from_by_target: dict[str, list[str]] = defaultdict(list)
        for source_subject_id, target_subject_id in subject_id_map.items():
            merged_from_by_target[target_subject_id].append(source_subject_id)

        subject_by_id: dict[str, dict[str, Any]] = {}
        for subject in source_subjects:
            original_subject_id = str(subject.get("subject_id") or "").strip()
            if not original_subject_id:
                continue
            target_subject_id = subject_id_map.get(original_subject_id, original_subject_id)
            if original_subject_id in subject_id_map:
                continue
            row = dict(subject)
            row["subject_id"] = target_subject_id
            if target_subject_id in occurrence_ids_by_subject:
                row["occurrence_ids"] = occurrence_ids_by_subject[target_subject_id]
            surfaces = set(row.get("surfaces") or [])
            surfaces.update(surface_by_subject.get(target_subject_id, set()))
            row["surfaces"] = sorted(str(item) for item in surfaces if str(item))
            row["merged_from_subject_ids"] = sorted(merged_from_by_target.get(target_subject_id, []))
            row["status"] = self._subject_status(statuses_by_subject.get(target_subject_id, set()))
            subject_by_id[target_subject_id] = row

        for subject_id in occurrence_ids_by_subject:
            subject_by_id.setdefault(
                subject_id,
                {
                    "subject_id": subject_id,
                    "family": "",
                    "primary_type": "",
                    "canonical_text": sorted(surface_by_subject.get(subject_id, {subject_id}))[0],
                    "canonical_key": subject_id,
                    "occurrence_ids": occurrence_ids_by_subject[subject_id],
                    "surfaces": sorted(surface_by_subject.get(subject_id, set())),
                    "source_layers": [],
                    "evidence": [],
                    "risk_flags": [],
                    "status": self._subject_status(statuses_by_subject.get(subject_id, set())),
                    "replacement": "",
                    "merged_from_subject_ids": [],
                },
            )
        return sorted(subject_by_id.values(), key=lambda item: str(item.get("subject_id") or ""))

    def _resolve_decisions(
        self,
        ledger_decisions: list[Any],
        adjudication_decisions: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        terminal_edges = {
            str(edge.get("edge_id") or "")
            for edge in edges
            if str(edge.get("status") or "") in self.TERMINAL_EDGE_STATUSES
        }
        resolved: list[dict[str, Any]] = []
        for decision in ledger_decisions:
            if not isinstance(decision, dict):
                continue
            edge_id = str(decision.get("edge_id") or "")
            if edge_id and edge_id in terminal_edges:
                continue
            resolved.append(dict(decision))
        for decision in adjudication_decisions:
            row = dict(decision)
            row["decision_id"] = str(row.get("decision_id") or f"ADJ::{row.get('edge_id') or row.get('occurrence_id') or len(resolved) + 1}")
            row["resolved"] = not bool(row.get("requires_manual_review")) and str(row.get("action") or "") != "manual_review"
            if not row["resolved"]:
                resolved.append(row)
        return resolved

    @staticmethod
    def _subject_status(statuses: set[str]) -> str:
        if not statuses:
            return "unknown"
        if "manual_review_required" in statuses:
            return "manual_review_required"
        if "hard_conflict" in statuses:
            return "hard_conflict"
        if "weak_identity_edge" in statuses:
            return "weak_identity_edge"
        if "ambiguous_short_subject" in statuses:
            return "ambiguous_short_subject"
        if "confirmed_separate_subject" in statuses:
            return "confirmed_separate_subject"
        if statuses <= {"rejected_subject"}:
            return "rejected_subject"
        return "confirmed_subject"

    def _summary(self, ledger: dict[str, Any], *, unresolved_subject_ids: set[str]) -> dict[str, Any]:
        subjects = [item for item in ledger.get("subjects") or [] if isinstance(item, dict)]
        occurrences = [item for item in ledger.get("occurrences") or [] if isinstance(item, dict)]
        edges = [item for item in ledger.get("edges") or [] if isinstance(item, dict)]
        decisions = [item for item in ledger.get("decisions") or [] if isinstance(item, dict)]
        edge_status_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        subject_status_counts: dict[str, int] = {}
        occurrence_status_counts: dict[str, int] = {}
        for edge in edges:
            status = str(edge.get("status") or "")
            edge_status_counts[status] = edge_status_counts.get(status, 0) + 1
            action = str(edge.get("adjudication_action") or "")
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
        for subject in subjects:
            status = str(subject.get("status") or "")
            subject_status_counts[status] = subject_status_counts.get(status, 0) + 1
        for occurrence in occurrences:
            status = str(occurrence.get("status") or "")
            occurrence_status_counts[status] = occurrence_status_counts.get(status, 0) + 1
        review_queue_count = sum(
            1
            for edge in edges
            if str(edge.get("status") or "") in {"needs_adjudication", "manual_review_required"}
        )
        manual_review_edge_ids = {
            str(edge.get("edge_id") or "")
            for edge in edges
            if str(edge.get("status") or "") == "manual_review_required"
        }
        review_queue_count += sum(
            1
            for decision in decisions
            if (
                bool(decision.get("requires_manual_review"))
                or str(decision.get("action") or "") == "manual_review"
            )
            and str(decision.get("edge_id") or "") not in manual_review_edge_ids
        )
        return {
            "subject_ledger_enabled": True,
            "subject_ledger_mode": "resolved",
            "occurrence_count": len(occurrences),
            "subject_count": len(subjects),
            "edge_count": len(edges),
            "decision_count": len(decisions),
            "edge_status_counts": dict(sorted(edge_status_counts.items())),
            "adjudication_action_counts": dict(sorted(action_counts.items())),
            "status_counts": dict(sorted(subject_status_counts.items())),
            "occurrence_status_counts": dict(sorted(occurrence_status_counts.items())),
            "resolved_subject_count": len(subjects),
            "resolved_edge_count": len(edges),
            "resolved_review_queue_count": review_queue_count,
            "unresolved_subject_count": len(unresolved_subject_ids),
            "manual_review_subject_count": subject_status_counts.get("manual_review_required", 0),
            "adjudicated_merge_count": action_counts.get("merge_to_canonical", 0),
            "adjudicated_keep_separate_count": action_counts.get("keep_separate", 0),
            "adjudicated_reject_count": action_counts.get("reject", 0),
            "review_queue_count": review_queue_count,
        }

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class SubjectLedgerBuilder:
    """Build a read-only subject ledger from accepted rule-first results."""

    def build(self, results: list[RecognizerResult]) -> SubjectLedgerBuildResult:
        accepted, subject_graph_summary = self._apply_subject_graph_compatibility(list(results or []))
        occurrence_rows: list[LedgerOccurrence] = []
        subject_rows: dict[str, LedgerSubject] = {}
        occurrence_subject: dict[str, str] = {}
        decisions: list[LedgerDecision] = []

        for index, result in enumerate(accepted, start=1):
            if not self._project_subject_type(result):
                continue
            occurrence = self._build_occurrence(result, index)
            occurrence_rows.append(occurrence)
            occurrence_subject[occurrence.occurrence_id] = occurrence.subject_id
            subject = subject_rows.get(occurrence.subject_id)
            if subject is None:
                subject = self._new_subject(occurrence)
                subject_rows[occurrence.subject_id] = subject
            self._merge_occurrence_into_subject(subject, occurrence)

        edges = self._build_edges(list(subject_rows.values()))
        occurrence_rows = self._apply_edge_review_requirements(
            occurrence_rows=occurrence_rows,
            subject_rows=subject_rows,
            edges=edges,
        )
        decisions = [
            decision
            for occurrence in occurrence_rows
            for decision in [self._build_decision(occurrence)]
            if decision
        ]
        self._finalize_subject_statuses(subject_rows, occurrence_rows)
        updated_results = self._annotate_results(
            accepted,
            occurrence_subject=occurrence_subject,
            occurrence_rows=occurrence_rows,
            subject_rows=subject_rows,
            edges=edges,
        )
        ledger = {
            "version": "subject_ledger.v1",
            "mode": "read_only",
            "occurrences": [item.to_dict() for item in occurrence_rows],
            "subjects": [item.to_dict() for item in sorted(subject_rows.values(), key=self._subject_sort_key)],
            "edges": [item.to_dict() for item in edges],
            "decisions": [item.to_dict() for item in decisions],
        }
        summary = self._summary(
            occurrence_rows=occurrence_rows,
            subjects=list(subject_rows.values()),
            edges=edges,
            decisions=decisions,
        )
        return SubjectLedgerBuildResult(
            results=updated_results,
            ledger=ledger,
            summary=summary,
            subject_graph_summary=subject_graph_summary,
        )

    def _apply_subject_graph_compatibility(
        self,
        results: list[RecognizerResult],
    ) -> tuple[list[RecognizerResult], dict[str, Any]]:
        cards = self._build_initial_subject_cards(results)
        parent = {card.subject_id: card.subject_id for card in cards}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        by_surface_type: dict[tuple[str, str], list[SubjectCard]] = defaultdict(list)
        for card in cards:
            for surface in card.surfaces:
                normalized = normalize_entity_text(surface)
                if normalized:
                    by_surface_type[(card.entity_type, normalized)].append(card)
        for rows in by_surface_type.values():
            if len(rows) > 1:
                anchor = rows[0]
                for card in rows[1:]:
                    union(anchor.subject_id, card.subject_id)

        org_cards = [card for card in cards if card.entity_type in ORGANIZATION_FAMILY_TYPES]
        for left_index, left in enumerate(org_cards):
            for right in org_cards[left_index + 1 :]:
                if self._has_hard_org_conflict(left, right):
                    left.risk_flags.add("hard_identity_conflict")
                    right.risk_flags.add("hard_identity_conflict")
                    left.unresolved = True
                    right.unresolved = True
                elif self._looks_like_weak_org_identity_match(left, right):
                    left.risk_flags.add("weak_identity_edge")
                    right.risk_flags.add("weak_identity_edge")

        merged_cards: dict[str, SubjectCard] = {}
        for card in cards:
            root = find(card.subject_id)
            target = merged_cards.get(root)
            if target is None:
                target = SubjectCard(
                    subject_id=root,
                    entity_type=card.entity_type,
                    canonical_text=card.canonical_text,
                    canonical_key=f"RULE_SUBJECT::{root}",
                )
                merged_cards[root] = target
            target.surfaces.update(card.surfaces)
            target.occurrence_ids.extend(card.occurrence_ids)
            target.source_layers.update(card.source_layers)
            target.risk_flags.update(card.risk_flags)
            target.evidence.extend(card.evidence)
            if len(normalize_entity_text(card.canonical_text)) > len(normalize_entity_text(target.canonical_text)):
                target.canonical_text = card.canonical_text
            if card.unresolved:
                target.unresolved = True

        occurrence_to_card: dict[str, SubjectCard] = {}
        for card in merged_cards.values():
            card.canonical_key = f"RULE_SUBJECT::{card.subject_id}"
            for occurrence_id in card.occurrence_ids:
                occurrence_to_card[occurrence_id] = card

        updated: list[RecognizerResult] = []
        for index, result in enumerate(results, start=1):
            occurrence_id = self._occurrence_id(result, index)
            card = occurrence_to_card.get(occurrence_id)
            metadata = dict(result.metadata or {})
            if card:
                metadata["subject_id"] = card.subject_id
                metadata.setdefault("canonical_key", card.canonical_key)
                metadata["canonical_subject_text"] = card.canonical_text
                metadata["subject_surfaces"] = sorted(card.surfaces)
                metadata["rule_subject_graph"] = card.to_metadata()
            updated.append(
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text=result.text,
                    source=result.source,
                    metadata=metadata,
                )
            )

        return updated, self._subject_graph_summary(cards, list(merged_cards.values()))

    def _build_initial_subject_cards(self, results: list[RecognizerResult]) -> list[SubjectCard]:
        cards: list[SubjectCard] = []
        for index, result in enumerate(results, start=1):
            entity_type = str(result.entity_type or "").upper()
            projected_type = self._project_subject_type(result)
            if not projected_type:
                continue
            normalized = normalize_entity_text(result.text)
            if not normalized:
                continue
            subject_id = f"S{len(cards) + 1:05d}"
            metadata = dict(result.metadata or {})
            surfaces = {str(result.text or ""), normalized}
            if metadata.get("normalized_text"):
                surfaces.add(str(metadata["normalized_text"]))
            if metadata.get("canonical"):
                surfaces.add(str(metadata["canonical"]))
            if metadata.get("definition_full_text"):
                surfaces.add(str(metadata["definition_full_text"]))
            evidence: list[str] = []
            for key in ("credit_code", "unified_social_credit_code", "cn_credit_code", "bank_account", "account_no"):
                if metadata.get(key):
                    evidence.append(f"{key}:{metadata[key]}")
            canonical = max(surfaces, key=lambda item: len(normalize_entity_text(item)))
            risk_flags: set[str] = set()
            if entity_type in ORGANIZATION_FAMILY_TYPES and looks_like_organization_short_name(normalized):
                risk_flags.add("short_org_alias")
            if len(normalized) <= 2 and entity_type in ORGANIZATION_FAMILY_TYPES:
                risk_flags.add("very_short_alias")
            card = SubjectCard(
                subject_id=subject_id,
                entity_type=projected_type,
                canonical_text=canonical,
                surfaces=surfaces,
                occurrence_ids=[self._occurrence_id(result, index)],
                source_layers={str(metadata.get("source_layer") or result.source or "")},
                risk_flags=risk_flags,
                evidence=evidence,
                unresolved=bool(risk_flags),
            )
            cards.append(card)
        return cards

    @staticmethod
    def _pool_type(entity_type: str) -> str:
        if entity_type in GOVERNMENT_FAMILY_TYPES:
            return "GOVERNMENT"
        if entity_type in ORGANIZATION_FAMILY_TYPES:
            return "ORGANIZATION"
        if entity_type in PERSON_FAMILY_TYPES:
            return "PERSON"
        if entity_type in LOCATION_FAMILY_TYPES:
            return "LOCATION"
        if entity_type in GOVERNMENT_FAMILY_TYPES:
            return "GOVERNMENT"
        return entity_type

    def _looks_like_weak_org_identity_match(self, left: SubjectCard, right: SubjectCard) -> bool:
        left_surfaces = {normalize_entity_text(item) for item in left.surfaces if normalize_entity_text(item)}
        right_surfaces = {normalize_entity_text(item) for item in right.surfaces if normalize_entity_text(item)}
        for left_surface in left_surfaces:
            for right_surface in right_surfaces:
                if not left_surface or not right_surface or left_surface == right_surface:
                    continue
                longer, shorter = (
                    (left_surface, right_surface)
                    if len(left_surface) >= len(right_surface)
                    else (right_surface, left_surface)
                )
                if len(shorter) < 2 or len(longer) < 4:
                    continue
                if shorter in longer and (is_org_like_text(longer) or looks_like_organization_short_name(shorter)):
                    return True
                if self._org_core(longer) and self._org_core(longer) == self._org_core(shorter):
                    return True
                if self._looks_like_org_alias_core_match(longer, shorter):
                    return True
        return False

    @staticmethod
    def _has_hard_org_conflict(left: SubjectCard, right: SubjectCard) -> bool:
        left_evidence = set(left.evidence)
        right_evidence = set(right.evidence)
        for prefix in ("credit_code:", "unified_social_credit_code:", "cn_credit_code:", "bank_account:", "account_no:"):
            left_values = {item for item in left_evidence if item.startswith(prefix)}
            right_values = {item for item in right_evidence if item.startswith(prefix)}
            if left_values and right_values and left_values.isdisjoint(right_values):
                return True
        return False

    @staticmethod
    def _org_core(value: str) -> str:
        normalized = normalize_entity_text(value)
        normalized = ORG_REGION_PREFIX_PATTERN.sub("", normalized)
        normalized = ORG_LEGAL_SUFFIX_PATTERN.sub("", normalized)
        return normalized if len(normalized) >= 2 else ""

    @classmethod
    def _org_brand_core(cls, value: str) -> str:
        core = cls._org_core(value)
        if len(core) < 2:
            return ""
        for token in ORG_BUSINESS_SUFFIX_TERMS:
            if core.endswith(token) and len(core) - len(token) >= 2:
                return core[: -len(token)]
        return core

    @classmethod
    def _looks_like_org_alias_core_match(cls, longer: str, shorter: str) -> bool:
        longer_core = cls._org_core(longer)
        shorter_core = cls._org_core(shorter)
        longer_brand = cls._org_brand_core(longer)
        shorter_brand = cls._org_brand_core(shorter)
        longer_candidates = {item for item in (longer_core, longer_brand) if len(item) >= 2}
        shorter_candidates = {item for item in (shorter_core, shorter_brand) if len(item) >= 2}
        if not longer_candidates or not shorter_candidates:
            return False
        for left_core in longer_candidates:
            for right_core in shorter_candidates:
                if left_core == right_core:
                    return True
                inner_longer, inner_shorter = (
                    (left_core, right_core)
                    if len(left_core) >= len(right_core)
                    else (right_core, left_core)
                )
                if len(inner_shorter) >= 2 and inner_shorter in inner_longer:
                    return True
        return False

    @staticmethod
    def _subject_graph_summary(initial_cards: list[SubjectCard], merged_cards: list[SubjectCard]) -> dict[str, Any]:
        unresolved = [card for card in merged_cards if card.unresolved]
        risk_counts: dict[str, int] = {}
        for card in merged_cards:
            for flag in card.risk_flags:
                risk_counts[flag] = risk_counts.get(flag, 0) + 1
        return {
            "initial_subject_count": len(initial_cards),
            "compiled_subject_count": len(merged_cards),
            "merged_subject_count": max(0, len(initial_cards) - len(merged_cards)),
            "unresolved_subject_count": len(unresolved),
            "hard_identity_conflict_count": risk_counts.get("hard_identity_conflict", 0),
            "risk_flag_counts": dict(sorted(risk_counts.items())),
        }

    def _build_occurrence(self, result: RecognizerResult, index: int) -> LedgerOccurrence:
        metadata = dict(result.metadata or {})
        entity_type = self._project_subject_type(result)
        normalized = normalize_entity_text(result.text)
        occurrence_id = self._occurrence_id(result, index)
        subject_id = self._subject_id_for_result(result, index)
        risk_flags = set(self._metadata_list(metadata, "ledger_risk_flags"))
        evidence = set(self._evidence_for_result(result))
        rule_first = metadata.get("rule_first") if isinstance(metadata.get("rule_first"), dict) else {}
        risk_flags.update(self._risk_flags_for_occurrence(result, metadata, rule_first))
        status = self._occurrence_status(result, metadata, risk_flags)
        canonical_key = str(metadata.get("canonical_key") or "").strip()
        canonical_role = str(metadata.get("canonical_role") or "").strip().upper()
        return LedgerOccurrence(
            occurrence_id=occurrence_id,
            entity_type=entity_type,
            text=str(result.text or ""),
            normalized_text=normalized,
            start=int(result.start),
            end=int(result.end),
            source=str(result.source or ""),
            source_layer=source_layer_for_result(result),
            score=float(result.score or 0.0),
            subject_id=subject_id,
            status=status,
            risk_flags=tuple(sorted(risk_flags)),
            evidence=tuple(sorted(evidence)),
            docx_unit_id=str(metadata.get("docx_unit_id") or ""),
            canonical_key=canonical_key,
            canonical_role=canonical_role,
        )

    @staticmethod
    def _project_subject_type(result: RecognizerResult) -> str:
        projected = projected_default_subject_type(result)
        return projected if projected in DEFAULT_SUBJECT_TYPES else ""

    @staticmethod
    def _occurrence_id(result: RecognizerResult, index: int) -> str:
        metadata = dict(result.metadata or {})
        return str(
            metadata.get("occurrence_id")
            or f"O{index:05d}:{int(result.start)}:{int(result.end)}:{str(result.entity_type or '').upper()}"
        )

    def _subject_id_for_result(self, result: RecognizerResult, index: int) -> str:
        metadata = dict(result.metadata or {})
        subject_graph = metadata.get("rule_subject_graph")
        if isinstance(subject_graph, dict):
            subject_id = str(subject_graph.get("subject_id") or "").strip()
            if subject_id:
                return subject_id
        subject_id = str(metadata.get("subject_id") or "").strip()
        if subject_id:
            return subject_id
        canonical_key = str(metadata.get("canonical_key") or "").strip()
        if canonical_key:
            return self._safe_subject_id(f"CANONICAL::{canonical_key}")
        normalized = normalize_entity_text(result.text)
        entity_type = str(result.entity_type or "").upper()
        return self._safe_subject_id(f"OBS::{entity_type}::{normalized or index}")

    def _new_subject(self, occurrence: LedgerOccurrence) -> LedgerSubject:
        canonical_key = occurrence.canonical_key or f"LEDGER::{occurrence.subject_id}"
        return LedgerSubject(
            subject_id=occurrence.subject_id,
            family=self._family_for_type(occurrence.entity_type),
            primary_type=occurrence.entity_type,
            canonical_text=occurrence.text,
            canonical_key=canonical_key,
            canonical_role=occurrence.canonical_role,
            status=self._subject_status_from_occurrences([occurrence]),
        )

    def _merge_occurrence_into_subject(
        self,
        subject: LedgerSubject,
        occurrence: LedgerOccurrence,
    ) -> None:
        subject.occurrence_ids.append(occurrence.occurrence_id)
        if occurrence.text:
            subject.surfaces.add(occurrence.text)
        if occurrence.normalized_text:
            subject.surfaces.add(occurrence.normalized_text)
        if occurrence.source_layer:
            subject.source_layers.add(occurrence.source_layer)
        subject.evidence.update(occurrence.evidence)
        subject.risk_flags.update(occurrence.risk_flags)
        if occurrence.canonical_role and not subject.canonical_role:
            subject.canonical_role = occurrence.canonical_role
        if occurrence.canonical_key and subject.canonical_key.startswith("LEDGER::"):
            subject.canonical_key = occurrence.canonical_key
        if self._prefer_canonical_text(occurrence, subject):
            subject.canonical_text = occurrence.text
            subject.primary_type = occurrence.entity_type

    @staticmethod
    def _prefer_canonical_text(occurrence: LedgerOccurrence, subject: LedgerSubject) -> bool:
        current_norm = normalize_entity_text(subject.canonical_text)
        candidate_norm = occurrence.normalized_text
        if len(candidate_norm) != len(current_norm):
            return len(candidate_norm) > len(current_norm)
        return occurrence.start < 10**12 and occurrence.text < subject.canonical_text

    def _build_edges(self, subjects: list[LedgerSubject]) -> list[LedgerEdge]:
        edges: list[LedgerEdge] = []
        by_surface: dict[tuple[str, str], list[LedgerSubject]] = defaultdict(list)
        by_canonical: dict[str, list[LedgerSubject]] = defaultdict(list)
        for subject in subjects:
            if subject.canonical_key and not subject.canonical_key.startswith("LEDGER::"):
                by_canonical[subject.canonical_key].append(subject)
            for surface in subject.surfaces:
                normalized = normalize_entity_text(surface)
                if normalized:
                    by_surface[(subject.family, normalized)].append(subject)

        edge_index = 0
        for canonical_key, rows in sorted(by_canonical.items()):
            if len(rows) < 2:
                continue
            root = sorted(rows, key=self._subject_sort_key)[0]
            for subject in rows:
                if subject.subject_id == root.subject_id:
                    continue
                edge_index += 1
                edges.append(
                    LedgerEdge(
                        edge_id=f"E{edge_index:05d}",
                        source_subject_id=subject.subject_id,
                        target_subject_id=root.subject_id,
                        relation="same_canonical_key",
                        evidence=(f"canonical_key:{canonical_key}",),
                        confidence=1.0,
                        status="confirmed",
                    )
                )

        for (_family, surface), rows in sorted(by_surface.items()):
            unique_rows = sorted({row.subject_id: row for row in rows}.values(), key=self._subject_sort_key)
            if len(unique_rows) < 2:
                continue
            root = unique_rows[0]
            for subject in unique_rows[1:]:
                edge_index += 1
                edges.append(
                    LedgerEdge(
                        edge_id=f"E{edge_index:05d}",
                        source_subject_id=subject.subject_id,
                        target_subject_id=root.subject_id,
                        relation="same_surface",
                        evidence=(f"surface:{surface}",),
                        confidence=0.92,
                        status="observed",
                    )
                )
        existing_pairs = {
            tuple(sorted((edge.source_subject_id, edge.target_subject_id)))
            for edge in edges
        }
        org_subjects = sorted(
            [subject for subject in subjects if subject.family == "organization"],
            key=self._subject_sort_key,
        )
        for left_index, left in enumerate(org_subjects):
            for right in org_subjects[left_index + 1 :]:
                pair = tuple(sorted((left.subject_id, right.subject_id)))
                if pair in existing_pairs:
                    continue
                relation = self._organization_subject_edge_relation(left, right)
                if not relation:
                    continue
                edge_index += 1
                edges.append(
                    LedgerEdge(
                        edge_id=f"E{edge_index:05d}",
                        source_subject_id=right.subject_id,
                        target_subject_id=left.subject_id,
                        relation=relation["relation"],
                        evidence=tuple(relation["evidence"]),
                        confidence=relation["confidence"],
                        status="needs_adjudication",
                    )
                )
                existing_pairs.add(pair)
        return edges

    def _organization_subject_edge_relation(
        self,
        left: LedgerSubject,
        right: LedgerSubject,
    ) -> dict[str, Any] | None:
        left_surfaces = {normalize_entity_text(item) for item in left.surfaces if normalize_entity_text(item)}
        right_surfaces = {normalize_entity_text(item) for item in right.surfaces if normalize_entity_text(item)}
        for left_surface in left_surfaces:
            for right_surface in right_surfaces:
                if not left_surface or not right_surface or left_surface == right_surface:
                    continue
                longer, shorter = (
                    (left_surface, right_surface)
                    if len(left_surface) >= len(right_surface)
                    else (right_surface, left_surface)
                )
                if len(shorter) >= 2 and len(longer) >= 4 and shorter in longer:
                    return {
                        "relation": "possible_alias",
                        "evidence": (f"contains:{shorter}->{longer}",),
                        "confidence": 0.68,
                    }
                left_core = self._org_core(left_surface)
                right_core = self._org_core(right_surface)
                if left_core and left_core == right_core:
                    return {
                        "relation": "same_org_core",
                        "evidence": (f"org_core:{left_core}",),
                        "confidence": 0.58,
                    }
                left_brand = self._org_brand_core(left_surface)
                right_brand = self._org_brand_core(right_surface)
                if left_brand and right_brand:
                    if left_brand == right_brand:
                        return {
                            "relation": "same_org_brand_core",
                            "evidence": (f"org_brand_core:{left_brand}",),
                            "confidence": 0.62,
                        }
                    longer_brand, shorter_brand = (
                        (left_brand, right_brand)
                        if len(left_brand) >= len(right_brand)
                        else (right_brand, left_brand)
                    )
                    if len(shorter_brand) >= 2 and shorter_brand in longer_brand:
                        return {
                            "relation": "possible_alias_core",
                            "evidence": (f"org_alias_core:{shorter_brand}->{longer_brand}",),
                            "confidence": 0.6,
                        }
        return None

    def _apply_edge_review_requirements(
        self,
        *,
        occurrence_rows: list[LedgerOccurrence],
        subject_rows: dict[str, LedgerSubject],
        edges: list[LedgerEdge],
    ) -> list[LedgerOccurrence]:
        review_edges_by_source: dict[str, list[LedgerEdge]] = defaultdict(list)
        for edge in edges:
            if edge.status == "needs_adjudication":
                review_edges_by_source[edge.source_subject_id].append(edge)
        if not review_edges_by_source:
            return occurrence_rows
        updated: list[LedgerOccurrence] = []
        for occurrence in occurrence_rows:
            review_edges = review_edges_by_source.get(occurrence.subject_id) or []
            if not review_edges:
                updated.append(occurrence)
                continue
            subject = subject_rows.get(occurrence.subject_id)
            if subject is not None:
                subject.risk_flags.add("weak_identity_edge")
            risk_flags = set(occurrence.risk_flags)
            risk_flags.add("weak_identity_edge")
            evidence = list(occurrence.evidence)
            for edge in review_edges:
                edge_evidence = f"edge_review:{edge.edge_id}:{edge.relation}:{edge.target_subject_id}"
                if edge_evidence not in evidence:
                    evidence.append(edge_evidence)
            status = occurrence.status
            if status == "confirmed_subject":
                status = "weak_identity_edge"
            updated.append(
                replace(
                    occurrence,
                    status=status,
                    risk_flags=tuple(sorted(risk_flags)),
                    evidence=tuple(evidence),
                )
            )
        return updated

    @staticmethod
    def _edge_review_by_source(
        edges: list[LedgerEdge],
        subject_rows: dict[str, LedgerSubject],
    ) -> dict[str, dict[str, Any]]:
        edge_by_source: dict[str, dict[str, Any]] = {}
        for edge in edges:
            if edge.status != "needs_adjudication":
                continue
            target = subject_rows.get(edge.target_subject_id)
            edge_by_source.setdefault(
                edge.source_subject_id,
                {
                    "subject_ledger_edge_id": edge.edge_id,
                    "subject_ledger_edge_relation": edge.relation,
                    "subject_ledger_edge_status": edge.status,
                    "subject_ledger_edge_target_subject_id": edge.target_subject_id,
                    "subject_ledger_edge_target_canonical_text": target.canonical_text if target else "",
                    "subject_ledger_edge_target_canonical_key": target.canonical_key if target else "",
                    "subject_ledger_edge_evidence": list(edge.evidence),
                },
            )
        return edge_by_source

    def _finalize_subject_statuses(
        self,
        subjects: dict[str, LedgerSubject],
        occurrences: list[LedgerOccurrence],
    ) -> None:
        by_subject: dict[str, list[LedgerOccurrence]] = defaultdict(list)
        for occurrence in occurrences:
            by_subject[occurrence.subject_id].append(occurrence)
        for subject_id, subject in subjects.items():
            subject.status = self._subject_status_from_occurrences(by_subject.get(subject_id, []))

    def _subject_status_from_occurrences(self, occurrences: list[LedgerOccurrence]) -> str:
        if not occurrences:
            return "unknown"
        statuses = {item.status for item in occurrences}
        if "hard_conflict" in statuses:
            return "hard_conflict"
        if "non_subject_role" in statuses and statuses <= {"non_subject_role"}:
            return "non_subject_role"
        if "weak_identity_edge" in statuses:
            return "weak_identity_edge"
        if any(self._has_confirming_subject_evidence(item) for item in occurrences):
            return "confirmed_subject"
        if "alias_without_anchor" in statuses:
            return "unresolved_alias"
        if "ambiguous_short_subject" in statuses:
            return "ambiguous_short_subject"
        if "weak_reference" in statuses and statuses <= {"weak_reference"}:
            return "weak_reference"
        return "confirmed_subject"

    @staticmethod
    def _has_confirming_subject_evidence(occurrence: LedgerOccurrence) -> bool:
        if occurrence.status == "confirmed_subject":
            if occurrence.entity_type in ORGANIZATION_FAMILY_TYPES:
                return (
                    is_org_like_text(occurrence.normalized_text)
                    and not looks_like_organization_short_name(occurrence.normalized_text)
                )
            return True
        if "definition_anchor" in occurrence.evidence:
            return True
        canonical_key = occurrence.canonical_key
        return bool(
            canonical_key
            and not canonical_key.startswith("ORG_OCC_")
            and not canonical_key.startswith("RULE_SUBJECT::")
        )

    def _annotate_results(
        self,
        results: list[RecognizerResult],
        *,
        occurrence_subject: dict[str, str],
        occurrence_rows: list[LedgerOccurrence],
        subject_rows: dict[str, LedgerSubject],
        edges: list[LedgerEdge],
    ) -> list[RecognizerResult]:
        occurrence_by_span = {
            (item.entity_type, item.start, item.end, item.text): item
            for item in occurrence_rows
        }
        edge_review_by_source = self._edge_review_by_source(edges, subject_rows)
        annotated: list[RecognizerResult] = []
        for result in results:
            key = (str(result.entity_type or "").upper(), int(result.start), int(result.end), str(result.text or ""))
            occurrence = occurrence_by_span.get(key)
            if occurrence is None:
                annotated.append(result)
                continue
            subject = subject_rows.get(occurrence_subject.get(occurrence.occurrence_id, ""))
            metadata = dict(result.metadata or {})
            metadata["subject_ledger_occurrence_id"] = occurrence.occurrence_id
            metadata["subject_ledger_subject_id"] = occurrence.subject_id
            metadata["subject_ledger_status"] = occurrence.status
            if subject is not None:
                metadata["subject_ledger_family"] = subject.family
                metadata["subject_ledger_canonical_text"] = subject.canonical_text
                metadata["subject_ledger_canonical_key"] = subject.canonical_key
                metadata["subject_ledger_subject_status"] = subject.status
            edge_review = edge_review_by_source.get(occurrence.subject_id)
            if edge_review:
                metadata.update(edge_review)
            annotated.append(
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text=result.text,
                    source=result.source,
                    metadata=metadata,
                )
            )
        return annotated

    def _build_decision(self, occurrence: LedgerOccurrence) -> LedgerDecision | None:
        if occurrence.status == "confirmed_subject":
            return None
        edge_decision = self._build_edge_decision(occurrence)
        if edge_decision is not None:
            return edge_decision
        reason_by_status = {
            "ambiguous_short_subject": "short subject requires identity evidence",
            "unresolved_alias": "alias lacks canonical anchor",
            "non_subject_role": "role or identity reference is not a subject",
            "weak_reference": "weak organization reference requires local antecedent",
            "hard_conflict": "subject evidence has hard conflict",
            "weak_identity_edge": "weak subject identity edge requires adjudication",
        }
        return LedgerDecision(
            decision_id=f"D::{occurrence.occurrence_id}",
            subject_id=occurrence.subject_id,
            occurrence_id=occurrence.occurrence_id,
            action="review" if occurrence.status not in {"non_subject_role"} else "reject_candidate",
            reason=reason_by_status.get(occurrence.status, occurrence.status),
            risk_level="high" if occurrence.status == "hard_conflict" else "medium",
            evidence=occurrence.evidence,
        )

    @staticmethod
    def _build_edge_decision(occurrence: LedgerOccurrence) -> LedgerDecision | None:
        for evidence in occurrence.evidence:
            text = str(evidence or "")
            if not text.startswith("edge_review:"):
                continue
            parts = text.split(":", 3)
            if len(parts) != 4:
                continue
            _, edge_id, relation, target_subject_id = parts
            if not edge_id or not target_subject_id:
                continue
            return LedgerDecision(
                decision_id=f"D::{edge_id}::{occurrence.occurrence_id}",
                subject_id=occurrence.subject_id,
                occurrence_id=occurrence.occurrence_id,
                action="review",
                reason=f"weak identity edge requires adjudication: {relation}",
                risk_level="medium",
                evidence=occurrence.evidence,
                decision_scope="edge",
                edge_id=edge_id,
                source_subject_id=occurrence.subject_id,
                target_subject_id=target_subject_id,
                edge_relation=relation,
            )
        return None

    def _occurrence_status(
        self,
        result: RecognizerResult,
        metadata: dict[str, Any],
        risk_flags: set[str],
    ) -> str:
        entity_type = str(result.entity_type or "").upper()
        normalized = normalize_entity_text(result.text)
        if "hard_identity_conflict" in risk_flags:
            return "hard_conflict"
        if normalized in IDENTITY_REFERENCE_TERMS or normalized in NON_ENTITY_ROLE_TERMS:
            return "non_subject_role"
        if (
            entity_type in ORGANIZATION_FAMILY_TYPES
            and looks_like_organization_short_name(normalized)
            and not self._has_stable_identity_anchor(metadata)
        ):
            return "ambiguous_short_subject"
        if bool(metadata.get("weak_reference")):
            return "weak_reference"
        return "confirmed_subject"

    def _risk_flags_for_occurrence(
        self,
        result: RecognizerResult,
        metadata: dict[str, Any],
        rule_first: dict[str, Any],
    ) -> set[str]:
        entity_type = str(result.entity_type or "").upper()
        normalized = normalize_entity_text(result.text)
        flags = set()
        subject_graph = metadata.get("rule_subject_graph")
        if isinstance(subject_graph, dict):
            flags.update(str(item) for item in subject_graph.get("risk_flags") or [] if str(item))
        if rule_first.get("action") == "review":
            flags.add("rule_review_required")
        if rule_first.get("risk_level") == "high":
            flags.add("high_risk")
        if entity_type in ORGANIZATION_FAMILY_TYPES and looks_like_organization_short_name(normalized):
            flags.add("short_org_alias")
        if normalized in IDENTITY_REFERENCE_TERMS or normalized in NON_ENTITY_ROLE_TERMS:
            flags.add("role_or_identity_reference")
        if entity_type in PERSON_FAMILY_TYPES and is_org_like_text(normalized):
            flags.add("person_org_type_conflict")
        if entity_type in ORGANIZATION_FAMILY_TYPES and is_probable_person(normalized) and not is_org_like_text(normalized):
            flags.add("org_person_type_conflict")
        return flags

    @staticmethod
    def _evidence_for_result(result: RecognizerResult) -> set[str]:
        metadata = dict(result.metadata or {})
        evidence = {f"source:{result.source}", f"source_layer:{source_layer_for_result(result)}"}
        if metadata.get("definition_alias") or metadata.get("definition_full_text") or metadata.get("canonical"):
            evidence.add("definition_anchor")
        if metadata.get("canonical_key"):
            evidence.add("canonical_key")
        for key in ("cn_credit_code", "credit_code", "unified_social_credit_code", "bank_account", "account_no"):
            if metadata.get(key):
                evidence.add(f"{key}:{metadata[key]}")
        if metadata.get("docx_unit_id"):
            evidence.add("docx_unit")
        rule_first = metadata.get("rule_first")
        if isinstance(rule_first, dict):
            for signal in rule_first.get("positive_signals") or []:
                evidence.add(f"positive:{signal}")
            for validator in rule_first.get("validators_passed") or []:
                evidence.add(f"validator:{validator}")
        return evidence

    @staticmethod
    def _has_stable_identity_anchor(metadata: dict[str, Any]) -> bool:
        canonical_key = str(metadata.get("canonical_key") or "").strip()
        if canonical_key and not canonical_key.startswith("RULE_SUBJECT::"):
            return True
        return bool(
            metadata.get("definition_alias")
            or metadata.get("definition_full_text")
            or metadata.get("canonical")
            or metadata.get("residual_scan")
        )

    @staticmethod
    def _family_for_type(entity_type: str) -> str:
        if entity_type in GOVERNMENT_FAMILY_TYPES:
            return "government"
        if entity_type in ORGANIZATION_FAMILY_TYPES:
            return "organization"
        if entity_type in PERSON_FAMILY_TYPES:
            return "person"
        if entity_type == "LOCATION":
            return "location"
        if entity_type == "GOVERNMENT":
            return "government"
        return entity_type.lower()

    @staticmethod
    def _metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
        value = metadata.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        if isinstance(value, tuple):
            return [str(item) for item in value if str(item)]
        return []

    @staticmethod
    def _safe_subject_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value or "").strip("_") or "OBS_UNKNOWN"

    @staticmethod
    def _subject_sort_key(subject: LedgerSubject) -> tuple[int, str]:
        starts = []
        for occurrence_id in subject.occurrence_ids:
            match = re.search(r":(\d+):\d+:", occurrence_id)
            if match:
                starts.append(int(match.group(1)))
        return (min(starts) if starts else 10**12, subject.subject_id)

    @staticmethod
    def _summary(
        *,
        occurrence_rows: list[LedgerOccurrence],
        subjects: list[LedgerSubject],
        edges: list[LedgerEdge],
        decisions: list[LedgerDecision],
    ) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        for subject in subjects:
            status_counts[subject.status] = status_counts.get(subject.status, 0) + 1
            family_counts[subject.family] = family_counts.get(subject.family, 0) + 1
            for flag in subject.risk_flags:
                risk_counts[flag] = risk_counts.get(flag, 0) + 1
        occurrence_status_counts: dict[str, int] = {}
        for occurrence in occurrence_rows:
            occurrence_status_counts[occurrence.status] = occurrence_status_counts.get(occurrence.status, 0) + 1
        return {
            "subject_ledger_enabled": True,
            "subject_ledger_mode": "read_only",
            "occurrence_count": len(occurrence_rows),
            "subject_count": len(subjects),
            "edge_count": len(edges),
            "decision_count": len(decisions),
            "status_counts": dict(sorted(status_counts.items())),
            "occurrence_status_counts": dict(sorted(occurrence_status_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "risk_flag_counts": dict(sorted(risk_counts.items())),
            "unresolved_subject_count": sum(
                count
                for status, count in status_counts.items()
                if status
                in {
                    "ambiguous_short_subject",
                    "unresolved_alias",
                    "weak_reference",
                    "weak_identity_edge",
                    "hard_conflict",
                }
            ),
            "ambiguous_short_subject_count": status_counts.get("ambiguous_short_subject", 0),
            "alias_without_anchor_count": status_counts.get("unresolved_alias", 0),
            "role_term_occurrence_count": occurrence_status_counts.get("non_subject_role", 0),
            "hard_identity_conflict_count": status_counts.get("hard_conflict", 0),
            "review_queue_count": len(decisions),
        }
