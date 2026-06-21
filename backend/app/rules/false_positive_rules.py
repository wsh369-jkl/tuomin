"""General false-positive filters for Chinese legal and contract documents."""

from __future__ import annotations

import re

from app.core.recognizer_base import RecognizerResult
from app.rules.default_subject_policy import canonical_default_entity_type
from app.rules.subject_admission_gate import SubjectAdmissionGate
from app.services.lowmem_entity_utils import (
    NON_ENTITY_ROLE_TERMS,
    is_generic_organization_term,
    is_identity_reference_term,
    is_org_like_text,
    is_position_title,
    is_probable_person,
    looks_like_organization_short_name,
    normalize_entity_text,
)


FIELD_LABEL_TERMS = {
    "证据内容",
    "是否具备",
    "材料名称",
    "备注",
    "序号",
    "金额",
    "日期",
    "地址",
    "电话",
    "项目地址",
    "联系地址",
    "注册地址",
    "住所",
}
STATE_OR_SENTENCE_TERMS = {
    "需要补充侦查",
    "是否完成",
    "是否通过",
    "通过",
    "通过了",
    "公司提交材料",
    "公司认为",
    "公司名称",
    "客户公司",
    "该公司",
    "贵公司",
}
NARRATIVE_VERBS = (
    "认为",
    "提交",
    "说明",
    "需要",
    "应当",
    "请求",
    "证明",
    "负责",
    "继续",
    "履约",
    "结算",
    "付款",
    "收款",
    "承担",
)
SUBJECT_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "GOVERNMENT"}
FORMAT_TYPES: set[str] = set()


class FalsePositiveRules:
    def assess(self, result: RecognizerResult) -> tuple[bool, list[str], list[str]]:
        entity_type = str(result.entity_type or "").strip().upper()
        text = str(result.text or "")
        normalized = normalize_entity_text(text)
        positive: list[str] = []
        negative: list[str] = []
        if not normalized:
            return True, positive, ["empty_text"]
        if entity_type in FORMAT_TYPES:
            positive.append("format_entity")
            return False, positive, negative
        if normalized in NON_ENTITY_ROLE_TERMS or is_identity_reference_term(normalized):
            return True, positive, ["identity_or_role_term"]
        if normalized in FIELD_LABEL_TERMS:
            return True, positive, ["field_label_only"]
        if normalized in STATE_OR_SENTENCE_TERMS:
            return True, positive, ["state_or_generic_sentence"]
        public_subject_type = canonical_default_entity_type(entity_type, normalized)
        if public_subject_type in SUBJECT_TYPES and SubjectAdmissionGate.is_action_or_function_text(normalized):
            return True, positive, ["non_subject_action_or_function_term"]
        if entity_type == "PERSON":
            if is_position_title(normalized):
                return True, positive, ["position_title_not_person"]
            if is_org_like_text(normalized):
                return True, positive, ["organization_shape_not_person"]
            if not is_probable_person(normalized):
                negative.append("weak_person_shape")
            else:
                positive.append("person_shape")
        if entity_type in {"ORGANIZATION", "GOVERNMENT"}:
            if is_generic_organization_term(normalized):
                return True, positive, ["generic_organization_term"]
            if is_position_title(normalized):
                return True, positive, ["position_title_not_organization"]
            if is_probable_person(normalized) and not is_org_like_text(normalized):
                if looks_like_organization_short_name(normalized):
                    negative.append("ambiguous_person_or_short_org_shape")
                else:
                    return True, positive, ["person_shape_not_organization"]
            if is_org_like_text(normalized):
                positive.append("organization_shape")
            elif looks_like_organization_short_name(normalized):
                positive.append("short_organization_shape")

        if entity_type in SUBJECT_TYPES:
            if self._looks_like_sentence_fragment(normalized):
                return True, positive, ["sentence_fragment"]
            if len(normalized) > 40 and any(token in normalized for token in NARRATIVE_VERBS):
                return True, positive, ["long_narrative_fragment"]
        if entity_type == "LOCATION" and normalized in FIELD_LABEL_TERMS:
            return True, positive, ["address_label_only"]
        return False, positive, negative

    @staticmethod
    def _alias_has_anchor(result: RecognizerResult) -> bool:
        metadata = dict(result.metadata or {})
        return bool(
            metadata.get("canonical")
            or metadata.get("definition_full_text")
            or metadata.get("definition_alias")
            or metadata.get("alias")
        )

    @staticmethod
    def _looks_like_sentence_fragment(normalized: str) -> bool:
        if len(normalized) <= 6:
            return False
        punctuation_count = len(re.findall(r"[，。；：,.!?！？;:]", normalized))
        if punctuation_count >= 2:
            return True
        if len(normalized) >= 18 and any(token in normalized for token in NARRATIVE_VERBS):
            return True
        if any(token in normalized for token in ("是否", "以及", "但是", "因此", "故", "判令")) and len(normalized) >= 10:
            return True
        return False
