"""Build coverage obligations before looking at recognized entities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.coverage_first.document_graph import DocumentGraph, DocumentUnit


@dataclass(frozen=True)
class CoverageObligation:
    obligation_id: str
    category: str
    unit_ids: list[str]
    expected_entity_types: list[str]
    required_checks: list[str]
    priority: str
    reason: str
    evidence_text_window: str
    rewrite_required: bool


PARTY_TERMS = ("甲方", "乙方", "丙方", "委托方", "受托方", "发包人", "承包人", "采购人", "供应商")
LEGAL_PARTY_TERMS = ("原告", "被告", "申请人", "被申请人", "上诉人", "被上诉人", "第三人")
CONTACT_TERMS = ("联系人", "联系电话", "手机号", "电话", "通讯地址", "联系地址", "住所", "住址", "地址")
BANK_TERMS = ("开户行", "开户银行", "户名", "账户", "账号", "银行账号", "收款单位", "付款单位")
SIGNATURE_TERMS = ("签章", "盖章", "落款", "签署", "签名", "签字")
DEFINITION_TERMS = ("以下简称", "下称", "简称", "又称")
ORG_SHAPE_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}(?:公司|集团|银行|法院|检察院|仲裁委员会|中心|研究院|事务所|商行|工作室|合作社)"
)
PERSON_WITH_ROLE_RE = re.compile(r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|经理|主任|负责人|联系人|代表)")


def build_coverage_plan(document_graph: DocumentGraph) -> dict[str, Any]:
    obligations: list[CoverageObligation] = []
    if not document_graph.enabled:
        return {
            "enabled": False,
            "obligations": [],
            "summary": {
                "enabled": False,
                "obligation_count": 0,
                "high_priority_obligation_count": 0,
            },
        }

    for unit in document_graph.units:
        obligation = _obligation_for_unit(len(obligations) + 1, unit)
        if obligation is not None:
            obligations.append(obligation)

    category_counts: dict[str, int] = {}
    expected_type_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    rewrite_required_count = 0
    for obligation in obligations:
        category_counts[obligation.category] = category_counts.get(obligation.category, 0) + 1
        priority_counts[obligation.priority] = priority_counts.get(obligation.priority, 0) + 1
        if obligation.rewrite_required:
            rewrite_required_count += 1
        for entity_type in obligation.expected_entity_types:
            expected_type_counts[entity_type] = expected_type_counts.get(entity_type, 0) + 1

    return {
        "enabled": True,
        "obligations": [obligation.__dict__ for obligation in obligations],
        "summary": {
            "enabled": True,
            "obligation_count": len(obligations),
            "high_priority_obligation_count": priority_counts.get("high", 0),
            "medium_priority_obligation_count": priority_counts.get("medium", 0),
            "rewrite_required_obligation_count": rewrite_required_count,
            "category_counts": dict(sorted(category_counts.items())),
            "expected_type_counts": dict(sorted(expected_type_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
        },
    }


def _obligation_for_unit(index: int, unit: DocumentUnit) -> CoverageObligation | None:
    compact = re.sub(r"\s+", "", unit.text or "")
    if not compact:
        return None
    category, reason, expected, checks, priority = _classify_unit(unit, compact)
    if not expected and priority == "low":
        return None
    return CoverageObligation(
        obligation_id=f"CO{index:05d}",
        category=category,
        unit_ids=[unit.unit_id],
        expected_entity_types=sorted(set(expected)),
        required_checks=sorted(set(checks)),
        priority=priority,
        reason=reason,
        evidence_text_window=unit.text[:800],
        rewrite_required=not unit.exactly_rewritable,
    )


def _classify_unit(unit: DocumentUnit, compact: str) -> tuple[str, str, list[str], list[str], str]:
    expected: list[str] = []
    checks: list[str] = []
    category = "ordinary_narrative"
    reason = "ordinary_text_with_sensitive_shape"
    priority = "medium"

    def add(types: list[str], required_checks: list[str]) -> None:
        expected.extend(types)
        checks.extend(required_checks)

    if any(term in compact for term in BANK_TERMS):
        category = "bank_account_block"
        reason = "bank_or_account_label"
        priority = "high"
        add(["ORGANIZATION", "BANK_NAME", "ACCOUNT_NAME", "CN_BANK_CARD"], ["bank_account_coverage", "organization_coverage"])
    elif any(term in compact for term in PARTY_TERMS):
        category = "party_block"
        reason = "party_role_label"
        priority = "high"
        add(["ORGANIZATION", "PERSON", "ALIAS"], ["organization_coverage", "person_coverage", "identity_relation_coverage"])
    elif any(term in compact for term in LEGAL_PARTY_TERMS):
        category = "legal_caption_block"
        reason = "legal_party_label"
        priority = "high"
        add(["ORGANIZATION", "PERSON"], ["organization_coverage", "person_coverage"])
    elif any(term in compact for term in SIGNATURE_TERMS):
        category = "signature_block"
        reason = "signature_or_seal_label"
        priority = "high"
        add(["ORGANIZATION", "PERSON"], ["organization_coverage", "person_coverage"])
    elif any(term in compact for term in CONTACT_TERMS):
        category = "contact_block"
        reason = "contact_or_address_label"
        priority = "high"
        add(["PERSON", "ADDRESS", "LOCATION", "CN_PHONE", "LANDLINE_PHONE"], ["person_coverage", "address_coverage", "phone_or_id_coverage"])
    elif any(term in compact for term in DEFINITION_TERMS):
        category = "definition_block"
        reason = "alias_definition_label"
        priority = "high"
        add(["ORGANIZATION", "PERSON", "ALIAS"], ["alias_definition_coverage", "identity_relation_coverage"])
    elif unit.container_type in {"table_cell", "table_row"}:
        category = "table_sensitive_column"
        reason = "docx_table_unit_with_sensitive_shape"
        priority = "medium"
        add(["ORGANIZATION", "PERSON", "ADDRESS", "ACCOUNT_NAME", "BANK_NAME"], ["organization_coverage", "person_coverage", "address_coverage"])
    elif unit.story_type in {"header", "footer"} or unit.container_type in {"header", "footer"}:
        category = "header_footer_block"
        reason = "header_or_footer_visible_text"
        priority = "medium"
        add(["ORGANIZATION", "PERSON", "CONTRACT_NO", "CASE_NO"], ["organization_coverage", "person_coverage"])
    elif unit.story_type == "textbox" or unit.container_type == "textbox":
        category = "textbox_block"
        reason = "textbox_visible_text"
        priority = "medium"
        add(["ORGANIZATION", "PERSON", "ADDRESS"], ["organization_coverage", "person_coverage", "address_coverage"])
    elif unit.story_type in {"footnote", "endnote", "comment"} or unit.container_type in {"footnote", "endnote", "comment"}:
        category = "footnote_comment_block"
        reason = "note_or_comment_visible_text"
        priority = "medium"
        add(["ORGANIZATION", "PERSON", "ADDRESS"], ["organization_coverage", "person_coverage", "address_coverage"])
    elif ORG_SHAPE_RE.search(compact):
        add(["ORGANIZATION"], ["organization_coverage"])
    elif PERSON_WITH_ROLE_RE.search(compact):
        add(["PERSON"], ["person_coverage"])
    else:
        category = "boilerplate_low_value"
        reason = "no_sensitive_structure_or_shape"
        priority = "low"

    if not unit.exactly_rewritable:
        checks.append("rewrite_addressability")
        if priority == "low":
            priority = "medium"
    return category, reason, expected, checks, priority
