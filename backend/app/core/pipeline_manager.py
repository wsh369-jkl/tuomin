"""Pipeline manager for recognition and anonymization."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.core.operator_registry import OperatorRegistry
from app.core.config import settings
from app.core.recognizer_base import RecognizerResult
from app.core.recognizer_registry import RecognizerRegistry
from app.rules.default_subject_policy import DEFAULT_SUBJECT_TYPES, canonicalize_default_result
from app.services.contextual_desensitization_service import ContextualDesensitizationService
from app.services.lowmem_entity_utils import (
    IDENTITY_REFERENCE_TERMS,
    NON_ENTITY_ROLE_TERMS,
    expand_subject_span_to_containing_shape,
    is_identity_reference_term,
    is_official_institution_text,
    is_org_like_text,
    is_position_title,
    looks_like_organization_short_name,
)

logger = logging.getLogger(__name__)


class PipelineManager:
    """Coordinate recognizers, merge their output, and apply operators."""

    PRIORITY_MAP = {
        "regex": 1,
        "rule_format": 1,
        "rule_organization": 2,
        "rule_person": 2,
        "rule_address": 2,
        "rule_bank": 2,
        "rule_alias": 2,
        "custom": 2,
        "contract": 3,
        "contract_structure_backfill": 4,
        "high_quality_lowmem": 5,
        "uie": 6,
        "ner": 7,
        "secondary_ner": 7,
        "qwen_fragment_review": 8,
        "alias_propagation": 9,
        "llm": 10,
        "ollama": 10,
        "propagate": 11,
    }

    ENTITY_SPECIFICITY = {
        "BANK_NAME": 1,
        "ACCOUNT_NAME": 1,
        "CONTRACT_NO": 1,
        "CN_PHONE": 1,
        "LANDLINE_PHONE": 1,
        "CN_BANK_CARD": 1,
        "CN_ID_CARD": 1,
        "CN_CREDIT_CODE": 1,
        "EMAIL_ADDRESS": 1,
        "CASE_NO": 1,
        "AMOUNT": 1,
        "DATE": 1,
        "ADDRESS": 2,
        "COURT": 2,
        "PROJECT": 2,
        "LOCATION": 2,
        "GOVERNMENT": 2,
        "POSITION": 2,
        "PERSON": 2,
        "PERSON_NAME": 2,
        "ORGANIZATION": 3,
        "COMPANY_NAME": 3,
        "ALIAS": 3,
    }

    PROPAGATION_TYPES = {
        "PERSON",
        "PERSON_NAME",
        "ORGANIZATION",
        "COMPANY_NAME",
        "ACCOUNT_NAME",
        "BANK_NAME",
        "PROJECT",
        "LOCATION",
        "ADDRESS",
        "GOVERNMENT",
        "COURT",
        "CONTRACT_NO",
        "CASE_NO",
        "ALIAS",
    }

    SPATIAL_NOISE_TERMS = {
        "岗位",
        "职位",
        "职务",
        "工作",
        "重点",
        "严重",
        "调整",
        "区域",
        "地点",
        "地址",
        "国家",
        "法定",
        "代表",
        "代表人",
        "法定代表人",
        "法定代理人",
        "法人",
        "法人代表",
        "负责",
        "负责人",
        "联系",
        "联系人",
        "南区",
        "北区",
        "东区",
        "西区",
    }
    GENERIC_ORGANIZATION_TERMS = {
        "公司",
        "银行",
        "法院",
        "人民法院",
        "集团",
        "机构",
        "单位",
        "部门",
        "销售部",
        "经销商",
        "劳动行政部门",
        "个人银行",
        "国家",
        "法定",
        "代表",
        "代表人",
        "法定代表人",
        "法定代理人",
        "法人",
        "法人代表",
        "负责",
        "负责人",
        "联系",
        "联系人",
        "人民共和国",
        "中华人民共和国",
        "和国",
        "国务院",
        "中国气象局",
        "我中心",
        "本中心",
        "贵中心",
        "本公司",
        "贵公司",
        "审查机构",
        "图审机构",
        "工图审查机构",
    }
    GENERIC_ORGANIZATION_TERMS.update(IDENTITY_REFERENCE_TERMS)
    LOW_INFORMATION_ORGANIZATION_ALIASES = {
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
        "供应链",
        "供应链服务",
    }
    ORGANIZATION_BUSINESS_SUFFIXES = tuple(sorted(LOW_INFORMATION_ORGANIZATION_ALIASES, key=len, reverse=True))
    PUBLIC_ORG_NOISE_TERMS = {
        "中华人民共和国",
        "人民共和国",
        "和国",
        "国务院",
        "中国气象局",
        "国家市场监督管理总局",
        "国家税务总局",
    }
    NON_SENSITIVE_LABEL_TERMS = {
        "国家",
        "法定",
        "代表",
        "代表人",
        "法定代表人",
        "法人",
        "法人代表",
        "负责",
        "负责人",
        "联系",
        "联系人",
        "地址",
        "住所",
        "住址",
        "法院",
        "人民法院",
        "一审法院",
        "二审法院",
        "原审法院",
        "本院",
        "贵院",
        *NON_ENTITY_ROLE_TERMS,
    }
    NON_SENSITIVE_LABEL_TERMS.update(IDENTITY_REFERENCE_TERMS)
    ADDRESS_LABEL_PATTERN = re.compile(
        r"(?:住址|住所|住所地|注册地址|送达地址|通讯地址|联系地址|工作地址|地址)\s*[:：]?\s*$"
    )
    DATE_LIKE_PATTERN = re.compile(r"\d{4}[./年-]\d{1,2}(?:[./月-]\d{1,2})?日?")
    DATE_VALUE_PATTERN = re.compile(
        r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*(?:\d{1,2}|xx|XX|\*)?\s*[日号]?|"
        r"\d{4}[./-]\d{1,2}(?:[./-](?:\d{1,2}|xx|XX|\*))?)"
    )
    ORGANIZATION_SUFFIX_PATTERN = re.compile(
        r"(?:股份有限公司|有限责任公司|有限公司|集团有限公司|集团|分公司|子公司|"
        r"银行(?:股份有限公司)?(?:[\u4e00-\u9fa5A-Za-z0-9]{0,12}(?:支行|分行))?|"
        r"人民法院|中级人民法院|高级人民法院|仲裁委员会|人民检察院|公安局|派出所|"
        r"律师事务所|会计师事务所|研究院|研究所|服务中心|技术中心|管理委员会|委员会|公司)$"
    )

    AUTHORITATIVE_TYPES = {
        "CN_ID_CARD",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
        "CONTRACT_NO",
        "CASE_NO",
    }

    def __init__(
        self,
        recognizer_registry: RecognizerRegistry,
        operator_registry: OperatorRegistry,
    ) -> None:
        self.recognizer_registry = recognizer_registry
        self.operator_registry = operator_registry
        self.contextual_desensitizer = ContextualDesensitizationService()
        logger.info("Pipeline manager initialized.")

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        recognizer_names: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        source_metadata: Optional[Dict] = None,
        source_structure: Optional[Dict] = None,
        progress_callback=None,
    ) -> List[Dict]:
        logger.info("Analyze pipeline started, text_length=%s", len(text))

        results = await self.recognizer_registry.analyze(
            text,
            entities,
            recognizer_names,
            llm_model=llm_model,
            source_metadata=source_metadata,
            source_structure=source_structure,
            progress_callback=progress_callback,
        )
        merged_results = self._merge_and_deduplicate(results)
        validated_results = self._validate_results(merged_results, text)
        validated_results = self._expand_repeated_mentions(validated_results, text)
        validated_results = self._validate_results(validated_results, text)
        dict_results = [result.to_dict() for result in validated_results]
        dict_results = await self.contextual_desensitizer.refine_recognition_entities(
            text=text,
            entities=dict_results,
            use_llm=bool(
                recognizer_names
                and ("llm" in recognizer_names or "high_quality_lowmem" in recognizer_names)
            ),
            llm_model=llm_model,
            source_metadata=source_metadata,
            source_structure=source_structure,
            progress_callback=progress_callback,
        )

        logger.info("Analyze pipeline finished, entities=%s", len(dict_results))
        return dict_results

    async def anonymize(
        self,
        text: str,
        entities: List[Dict],
        operator_config: Optional[Dict[str, Dict]] = None,
    ) -> str:
        logger.info("Anonymize pipeline started, entities=%s", len(entities))

        config = self._get_default_operator_config()
        if operator_config:
            config.update(operator_config)

        sorted_entities = sorted(entities or [], key=lambda item: item["start"], reverse=True)
        result = text

        for entity in sorted_entities:
            entity_type = entity["type"]
            start = entity["start"]
            end = entity["end"]

            if entity.get("replacement_method") == "preserve":
                continue
            if (
                entity_type in self.contextual_desensitizer.PRESERVE_TYPES
                and not entity.get("replacement")
                and not (operator_config and entity_type in operator_config)
            ):
                continue
            if entity.get("replacement"):
                replacement = str(entity["replacement"])
                result = result[:start] + replacement + result[end:]
                continue

            entity_config = config.get(entity_type, config.get("default", {}))
            operator_name = entity_config.get("operator", "mask")
            params = entity_config.get("params", {})

            operator = self.operator_registry.get_operator(operator_name)
            if operator is None:
                logger.warning(
                    "Operator %s is missing, fallback to mask.",
                    operator_name,
                )
                operator = self.operator_registry.get_operator("mask")

            try:
                replacement = operator.operate(result, start, end, entity_type, **params)
                result = result[:start] + replacement + result[end:]
            except Exception as exc:
                logger.error("Anonymize failed for %s: %s", entity_type, exc)

        logger.info("Anonymize pipeline finished.")
        return result

    async def prepare_entities_for_anonymization(
        self,
        *,
        text: str,
        entities: List[Dict],
        use_llm: bool,
        operator_config: Optional[Dict[str, Dict]] = None,
        llm_model: Optional[str] = None,
        anonymization_strategy: Optional[str] = None,
        source_metadata: Optional[Dict] = None,
        source_structure: Optional[Dict] = None,
    ) -> List[Dict]:
        return await self.contextual_desensitizer.prepare_entities(
            text=text,
            entities=entities,
            use_llm=use_llm,
            operator_config=operator_config,
            llm_model=llm_model,
            anonymization_strategy=anonymization_strategy,
            source_metadata=source_metadata,
            source_structure=source_structure,
        )

    def get_last_quality_metadata(self) -> Dict:
        return self.contextual_desensitizer.get_last_quality_metadata()

    def _merge_and_deduplicate(
        self,
        results: List[RecognizerResult],
    ) -> List[RecognizerResult]:
        sorted_results = sorted(
            results,
            key=lambda item: (
                self.PRIORITY_MAP.get(item.source, 99),
                -float(item.score),
                -(item.end - item.start),
                item.start,
                item.end,
            ),
        )

        merged: List[RecognizerResult] = []
        for result in sorted_results:
            overlapping_index = self._find_overlapping_index(result, merged)
            if overlapping_index is None:
                merged.append(result)
                continue

            existing = merged[overlapping_index]
            if self._should_replace(existing, result):
                merged[overlapping_index] = result

        merged.sort(key=lambda item: item.start)
        logger.debug("Merged entities: %s -> %s", len(results), len(merged))
        return merged

    def _find_overlapping_index(
        self,
        result: RecognizerResult,
        existing_results: List[RecognizerResult],
    ) -> int | None:
        for index, existing in enumerate(existing_results):
            if (
                result.start < existing.end
                and result.end > existing.start
            ):
                return index
        return None

    def _should_replace(
        self,
        existing: RecognizerResult,
        candidate: RecognizerResult,
    ) -> bool:
        same_span = existing.start == candidate.start and existing.end == candidate.end
        if not same_span:
            if (
                candidate.entity_type in self.AUTHORITATIVE_TYPES
                and existing.entity_type not in self.AUTHORITATIVE_TYPES
            ):
                return True
            if (
                existing.entity_type in self.AUTHORITATIVE_TYPES
                and candidate.entity_type not in self.AUTHORITATIVE_TYPES
            ):
                return False
            if existing.entity_type == candidate.entity_type:
                candidate_len = candidate.end - candidate.start
                existing_len = existing.end - existing.start
                if candidate_len > existing_len and self.PRIORITY_MAP.get(candidate.source, 99) <= self.PRIORITY_MAP.get(existing.source, 99) + 2:
                    return True
                return False
            return False

        existing_rank = self.ENTITY_SPECIFICITY.get(existing.entity_type, 99)
        candidate_rank = self.ENTITY_SPECIFICITY.get(candidate.entity_type, 99)
        if candidate_rank != existing_rank:
            return candidate_rank < existing_rank

        return float(candidate.score) > float(existing.score)

    def _validate_results(
        self,
        results: List[RecognizerResult],
        text: str,
    ) -> List[RecognizerResult]:
        validated: List[RecognizerResult] = []

        for raw_result in results:
            result = canonicalize_default_result(raw_result)
            if result is None:
                continue
            if result.start < 0 or result.end > len(text) or result.start >= result.end:
                logger.warning("Invalid entity span skipped: %s", result)
                continue

            actual_text = text[result.start : result.end]
            if actual_text != result.text:
                logger.warning(
                    "Entity text mismatch skipped: expected=%r actual=%r",
                    result.text,
                    actual_text,
                )
                continue

            result = self._expand_result_to_containing_subject_shape(result, text)

            if not self._is_valid_entity_candidate(result, text):
                logger.debug("Invalid entity candidate skipped: %s", result)
                continue

            validated.append(result)

        logger.debug("Validated entities: %s -> %s", len(results), len(validated))
        return validated

    @staticmethod
    def _expand_result_to_containing_subject_shape(
        result: RecognizerResult,
        text: str,
    ) -> RecognizerResult:
        expanded_span = expand_subject_span_to_containing_shape(
            text,
            result.start,
            result.end,
            result.entity_type,
        )
        if expanded_span is None:
            return result
        expanded_start, expanded_end = expanded_span
        if expanded_start == result.start and expanded_end == result.end:
            return result
        expanded_text = text[expanded_start:expanded_end]
        if not expanded_text:
            return result
        metadata = dict(result.metadata or {})
        metadata.setdefault(
            "default_validation_expanded_from",
            {
                "start": result.start,
                "end": result.end,
                "text": result.text,
            },
        )
        return RecognizerResult(
            entity_type=result.entity_type,
            start=expanded_start,
            end=expanded_end,
            score=max(float(result.score or 0.0), 0.86),
            text=expanded_text,
            source=result.source,
            metadata=metadata,
        )

    def _get_default_operator_config(self) -> Dict[str, Dict]:
        return {
            "PERSON": {
                "operator": "replace",
                "params": {"new_value": "[\u59d3\u540d]"},
            },
            "PERSON_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u59d3\u540d]"},
            },
            "ORGANIZATION": {
                "operator": "replace",
                "params": {"new_value": "[\u673a\u6784]"},
            },
            "COMPANY_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u673a\u6784]"},
            },
            "LOCATION": {
                "operator": "replace",
                "params": {"new_value": "[\u5730\u5740]"},
            },
            "ADDRESS": {
                "operator": "replace",
                "params": {"new_value": "[\u5730\u5740]"},
            },
            "GOVERNMENT": {
                "operator": "replace",
                "params": {"new_value": "[\u653f\u5e9c\u673a\u6784]"},
            },
            "COURT": {
                "operator": "replace",
                "params": {"new_value": "[\u653f\u5e9c\u673a\u6784]"},
            },
            "BANK_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u673a\u6784]"},
            },
            "ACCOUNT_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u673a\u6784]"},
            },
            "ALIAS": {
                "operator": "replace",
                "params": {"new_value": "[\u673a\u6784]"},
            },
            "PROJECT": {
                "operator": "replace",
                "params": {"new_value": "[\u673a\u6784]"},
            },
            "CONTRACT_NO": {
                "operator": "mask",
                "params": {"keep_start": 0, "keep_end": 0},
            },
            "CASE_NO": {
                "operator": "mask",
                "params": {"keep_start": 0, "keep_end": 0},
            },
            "default": {
                "operator": "mask",
                "params": {},
            },
        }
    def _expand_repeated_mentions(
        self,
        results: List[RecognizerResult],
        text: str,
    ) -> List[RecognizerResult]:
        expanded = list(results)
        seen = {(item.entity_type, item.start, item.end, item.text) for item in results}
        ledger_added = 0

        for result in sorted(results, key=lambda item: (-(item.end - item.start), item.start)):
            for candidate_text in self._derive_subject_ledger_propagation_texts(result):
                if len(candidate_text.strip()) < 2:
                    continue
                normalized_candidate = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", candidate_text or "")
                if not self._should_propagate_ledger_subject_surface(
                    candidate_text,
                    result.entity_type,
                    normalized_candidate=normalized_candidate,
                ):
                    continue
                anchor_allows_short_org = self._allows_anchored_short_organization_propagation(
                    result,
                    candidate_text,
                )
                for start, end, matched_text in self._find_all_occurrence_spans(text, candidate_text):
                    key = (result.entity_type, start, end, matched_text)
                    if key in seen:
                        continue
                    if self._overlaps_existing(start, end, expanded):
                        continue
                    if (
                        result.entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "PROJECT", "ALIAS"}
                        and len(normalized_candidate) <= 6
                        and not self.ORGANIZATION_SUFFIX_PATTERN.search(normalized_candidate)
                        and not self._has_short_organization_occurrence_context(text, start, end)
                    ):
                        continue
                    metadata = dict(result.metadata or {})
                    metadata.update(
                        {
                            "derived_from": result.text,
                            "candidate_text": candidate_text,
                            "propagated_from_subject_ledger": True,
                            "propagated_from_stable_seed": True,
                        }
                    )
                    if anchor_allows_short_org:
                        metadata["anchored_short_org_propagation"] = True
                    expanded.append(
                        RecognizerResult(
                            entity_type=result.entity_type,
                            start=start,
                            end=end,
                            score=max(0.74, float(result.score) - 0.06),
                            text=matched_text,
                            source="propagate",
                            metadata=metadata,
                        )
                    )
                    seen.add(key)
                    ledger_added += 1
            if result.entity_type not in self.PROPAGATION_TYPES:
                continue
            if not self._stable_seed_for_default_propagation(result):
                continue

            for candidate_text in self._derive_candidate_texts(result.text, result.entity_type):
                if len(candidate_text.strip()) < 2:
                    continue
                anchor_allows_short_org = self._allows_anchored_short_organization_propagation(result, candidate_text)
                if not self._should_propagate_candidate(candidate_text, result.entity_type, anchor_allows_short_org=anchor_allows_short_org):
                    continue

                for start, end, matched_text in self._find_all_occurrence_spans(text, candidate_text):
                    key = (result.entity_type, start, end, matched_text)
                    if key in seen:
                        continue
                    if self._overlaps_existing(start, end, expanded):
                        continue
                    if anchor_allows_short_org and not self._has_short_organization_occurrence_context(text, start, end):
                        continue

                    expanded.append(
                        RecognizerResult(
                            entity_type=result.entity_type,
                            start=start,
                            end=end,
                            score=max(0.72, float(result.score) - 0.08),
                            text=matched_text,
                            source="propagate",
                            metadata={
                                **dict(result.metadata or {}),
                                "derived_from": result.text,
                                "candidate_text": candidate_text,
                                "propagated_from_stable_seed": True,
                            },
                        )
                    )
                    seen.add(key)

        expanded.sort(key=lambda item: (item.start, item.end))
        if ledger_added:
            logger.info("Subject ledger propagation added %s repeated mentions.", ledger_added)
        return expanded

    def _derive_subject_ledger_propagation_texts(self, result: RecognizerResult) -> List[str]:
        if result.entity_type not in self.PROPAGATION_TYPES:
            return []
        metadata = dict(result.metadata or {})
        if str(metadata.get("subject_ledger_status") or "") not in {"", "confirmed_subject"}:
            return []
        if str(metadata.get("subject_ledger_subject_status") or "") not in {"", "confirmed_subject"}:
            return []
        candidates: List[str] = []
        for value in (
            metadata.get("subject_ledger_canonical_text"),
            metadata.get("canonical_subject_text"),
            metadata.get("canonical"),
            metadata.get("definition_full_text"),
            result.text,
        ):
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        for value in metadata.get("subject_surfaces") or []:
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        deduped: List[str] = []
        result_normalized = re.sub(r"\s+", "", result.text or "")
        for candidate in candidates:
            normalized = re.sub(r"\s+", "", candidate or "")
            if not normalized:
                continue
            if normalized in self.NON_SENSITIVE_LABEL_TERMS:
                continue
            if normalized not in deduped:
                deduped.append(normalized if normalized != result_normalized else candidate)
        return sorted(deduped, key=lambda item: (-len(re.sub(r"\s+", "", item or "")), item))

    def _should_propagate_ledger_subject_surface(
        self,
        candidate_text: str,
        entity_type: str,
        *,
        normalized_candidate: str,
    ) -> bool:
        if not normalized_candidate or normalized_candidate in self.NON_SENSITIVE_LABEL_TERMS:
            return False
        if entity_type in {"GOVERNMENT", "COURT", "BANK_NAME"}:
            return is_official_institution_text(normalized_candidate)
        if entity_type in {"LOCATION", "ADDRESS"}:
            return self._is_detailed_location_candidate(normalized_candidate)
        if entity_type in {"PERSON", "PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}:
            return self._is_valid_person_candidate(normalized_candidate, "propagate")
        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "PROJECT", "ALIAS"}:
            if self.ORGANIZATION_SUFFIX_PATTERN.search(normalized_candidate):
                return self._is_valid_organization_candidate(normalized_candidate, "propagate")
            if 2 <= len(normalized_candidate) <= 6:
                return (
                    looks_like_organization_short_name(normalized_candidate)
                    and not self._is_low_information_organization_alias(normalized_candidate)
                    and not self._looks_like_location_alias(normalized_candidate)
                )
            return self._is_valid_organization_candidate(normalized_candidate, "propagate")
        return self._should_propagate_candidate(candidate_text, entity_type)

    @staticmethod
    def _stable_seed_for_default_propagation(result: RecognizerResult) -> bool:
        metadata = dict(result.metadata or {})
        if metadata.get("requires_manual_review"):
            return False
        if metadata.get("candidate_types") or metadata.get("boundary_repaired_from"):
            return False
        if metadata.get("default_subject_policy_type_from"):
            return False
        review_statuses = {
            "ambiguous_short_subject",
            "unresolved_alias",
            "alias_without_anchor",
            "weak_reference",
            "weak_identity_edge",
            "hard_conflict",
        }
        if str(metadata.get("subject_ledger_status") or "") in review_statuses:
            return False
        if str(metadata.get("subject_ledger_subject_status") or "") in review_statuses:
            return False
        rule_first = metadata.get("rule_first")
        if isinstance(rule_first, dict) and (
            str(rule_first.get("action") or "") == "review"
            or str(rule_first.get("risk_level") or "") == "high"
        ):
            return False
        return True

    def _is_valid_entity_candidate(self, result: RecognizerResult, text: str) -> bool:
        normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", result.text or "")
        if not normalized:
            return False
        if normalized in self.NON_SENSITIVE_LABEL_TERMS:
            return False
        if result.entity_type in self.AUTHORITATIVE_TYPES:
            return True
        if result.entity_type in {"DATE", "AMOUNT"}:
            return True
        if result.entity_type == "ALIAS":
            metadata = dict(result.metadata or {})
            return bool(metadata.get("canonical") or metadata.get("definition_full_text") or metadata.get("definition_alias"))
        if result.entity_type in {"PERSON", "PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}:
            if not self._is_valid_person_candidate(normalized, result.source):
                return False
        if result.entity_type in {"LOCATION", "ADDRESS"}:
            if len(normalized) < 2 or normalized in self.SPATIAL_NOISE_TERMS:
                return False
            if self._is_broad_region_list(normalized):
                return False
            if self.DATE_LIKE_PATTERN.fullmatch((result.text or "").strip()):
                return False
            if result.source in {"uie", "ner", "secondary_ner", "propagate"} and self._is_weak_region_reference(normalized):
                return False
            if result.source in {"ner", "secondary_ner", "propagate"} and not self._has_supported_location_context(
                result,
                text,
            ):
                return False
            if (
                result.entity_type == "ADDRESS"
                and result.source in {"uie", "ner", "secondary_ner", "propagate"}
                and not self._has_supported_location_context(result, text)
                and not self._is_detailed_location_candidate(normalized)
            ):
                return False
        if result.entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "BANK_NAME", "PROJECT", "GOVERNMENT", "COURT"}:
            if result.entity_type in {"GOVERNMENT", "COURT", "BANK_NAME"} and not is_official_institution_text(normalized):
                return False
            if result.entity_type == "GOVERNMENT":
                if self._is_procedural_court_reference(result):
                    return False
                if normalized in {"法院", "人民法院"}:
                    return False
                if self._is_public_legal_authority_citation(result, text):
                    return False
            if result.source in {"ner", "secondary_ner", "uie", "propagate"} and not self._is_valid_organization_candidate(
                normalized,
                result.source,
            ):
                if not self._is_valid_anchored_propagated_organization_alias(result, text, normalized):
                    return False
        if result.entity_type == "POSITION":
            if normalized in {
                "经销商",
                "公司",
                "部门",
                "销售部",
                "销售",
                "管理",
                "高级",
                "级管理",
                "销售人员",
                "销售类",
                "销售岗",
                "岗位",
                "人员",
                "技术人",
                "联系人",
                "代理人",
                "委托代理人",
                "职位",
                "职务",
            }:
                return False
            if is_identity_reference_term(normalized):
                return False
            return is_position_title(normalized)
        return True

    @staticmethod
    def _is_valid_person_candidate(normalized: str, source: str) -> bool:
        if not normalized:
            return False
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return False
        role_noise = {
            *NON_ENTITY_ROLE_TERMS,
            "法定代表人",
            "法定代理人",
            "法人代表",
            "委托代理人",
            "委托诉讼代理人",
            "诉讼代理人",
            "代理人",
            "负责人",
            "联系人",
            "签署人",
            "经办人",
        }
        if normalized in role_noise:
            return False
        if any(
            token in normalized
            for token in (
                "委托",
                "代理人",
                "代表人",
                "负责人",
                "联系人",
                "法定代理人",
                "控告人",
                "举报人",
                "申诉人",
                "起诉人",
                "自诉人",
                "申请人",
                "被申请人",
                "上诉人",
                "被上诉人",
                "原告",
                "被告",
                "第三人",
            )
        ):
            return False
        if any(token in normalized for token in ("公司", "银行", "法院", "项目", "地址", "电话", "账号")):
            return False
        if is_org_like_text(normalized):
            return False
        if source in {"uie", "ner", "secondary_ner", "propagate", "qwen_fragment_review"}:
            return re.fullmatch(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", normalized) is not None
        return True

    def _has_supported_location_context(self, result: RecognizerResult, text: str) -> bool:
        normalized = re.sub(r"\s+", "", result.text or "")
        before = text[max(0, result.start - 24) : result.start]
        after = text[result.end : min(len(text), result.end + 16)]

        if self.ADDRESS_LABEL_PATTERN.search(before):
            return True
        if any(token in normalized for token in ("路", "街", "号", "栋", "室", "村", "组", "小区", "大厦", "广场")):
            return True
        if any(token in after for token in ("路", "街", "号", "栋", "室", "村", "组")) and any(
            token in normalized for token in ("省", "市", "区", "县", "镇", "乡")
        ):
            return True
        if re.search(r"(?:高级|销售|经理|岗位|区域|负责|业务|工作|出差)", after):
            return False
        return False

    @staticmethod
    def _is_procedural_court_reference(result: RecognizerResult) -> bool:
        normalized = re.sub(r"\s+", "", result.text or "")
        return normalized in {
            "一审法院",
            "二审法院",
            "原审法院",
            "再审法院",
            "执行法院",
            "上级法院",
            "下级法院",
            "项目所在地人民法院",
            "所在地人民法院",
            "本院",
            "贵院",
        }

    @staticmethod
    def _is_public_legal_authority_citation(result: RecognizerResult, text: str) -> bool:
        normalized = re.sub(r"\s+", "", result.text or "")
        if normalized not in {"最高人民法院", "最高人民检察院", *PipelineManager.PUBLIC_ORG_NOISE_TERMS}:
            return False
        before = text[max(0, result.start - 32) : result.start]
        after = text[result.end : min(len(text), result.end + 40)]
        context = before + after
        return any(token in context for token in ("根据", "依据", "包括但不限于", "令第", "法律", "法规", "规定", "办法", "条例", "规章"))

    def _should_propagate_candidate(
        self,
        candidate_text: str,
        entity_type: str,
        *,
        anchor_allows_short_org: bool = False,
    ) -> bool:
        normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", candidate_text or "")
        if not normalized or normalized in self.SPATIAL_NOISE_TERMS:
            return False
        if entity_type in {"LOCATION", "ADDRESS"}:
            return self._is_detailed_location_candidate(normalized)
        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "BANK_NAME", "GOVERNMENT", "COURT", "PROJECT", "ALIAS"}:
            if entity_type == "ALIAS" and normalized in self.NON_SENSITIVE_LABEL_TERMS:
                return False
            if anchor_allows_short_org and len(normalized) <= 6:
                return True
            return self._is_valid_organization_candidate(normalized, "propagate")
        return True

    def _allows_anchored_short_organization_propagation(self, result: RecognizerResult, candidate_text: str) -> bool:
        entity_type = str(result.entity_type or "").upper()
        if entity_type not in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "PROJECT", "ALIAS"}:
            return False
        normalized_candidate = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", candidate_text or "")
        if not 2 <= len(normalized_candidate) <= 6:
            return False
        if normalized_candidate in self.NON_SENSITIVE_LABEL_TERMS:
            return False
        if self._is_low_information_organization_alias(normalized_candidate):
            return False
        normalized_source = re.sub(r"\s+", "", result.text or "")
        if is_official_institution_text(normalized_source):
            return False
        metadata = dict(result.metadata or {})
        canonical = str(metadata.get("canonical") or metadata.get("definition_full_text") or "")
        if result.entity_type == "ALIAS" and canonical:
            normalized_source = re.sub(r"\s+", "", canonical)
            if is_official_institution_text(normalized_source):
                return False
        if not self.ORGANIZATION_SUFFIX_PATTERN.search(normalized_source):
            return False
        if normalized_candidate not in re.sub(r"\s+", "", normalized_source):
            return False
        source = str(result.source or "")
        if source in {"contract_structure_backfill", "rule_organization", "rule_alias", "rule_first"}:
            return True
        if metadata.get("source_layer") in {"structure", "rule"}:
            return True
        if metadata.get("trigger") in {"party_label", "alias_definition", "signature_subject", "inline_party_role"}:
            return True
        return False

    def _is_valid_anchored_propagated_organization_alias(
        self,
        result: RecognizerResult,
        text: str,
        normalized: str,
    ) -> bool:
        if str(result.source or "") != "propagate":
            return False
        metadata = dict(result.metadata or {})
        if not metadata.get("propagated_from_stable_seed"):
            return False
        if not 2 <= len(normalized) <= 6:
            return False
        if normalized in self.NON_SENSITIVE_LABEL_TERMS:
            return False
        if self._is_low_information_organization_alias(normalized):
            return False
        if self._looks_like_location_alias(normalized):
            return False
        derived_from = re.sub(r"\s+", "", str(metadata.get("derived_from") or ""))
        if not derived_from or not self.ORGANIZATION_SUFFIX_PATTERN.search(derived_from):
            return False
        if normalized not in derived_from:
            return False
        return self._has_short_organization_occurrence_context(text, result.start, result.end)

    @staticmethod
    def _has_short_organization_occurrence_context(text: str, start: int, end: int) -> bool:
        before = text[max(0, start - 18) : start]
        after = text[end : min(len(text), end + 24)]
        sentence_before = re.split(r"[。；;\n\r]", before)[-1]
        sentence_after = re.split(r"[。；;\n\r]", after, maxsplit=1)[0]
        context = sentence_before + sentence_after
        if re.search(r"(?:甲方|乙方|丙方|委托方|受托方|发包人|承包人|供应商|申请人|被申请人|原告|被告|第三人)\s*[:：]?\s*$", sentence_before):
            return True
        if re.search(
            r"(?:负责|继续|履约|履行|结算|付款|收款|交付|供货|签署|签订|盖章|落款|承担|"
            r"确认|通知|函告|起诉|上诉|申请|被申请|委托|授权|指定|收款|付款|签章)",
            context,
        ):
            return True
        if re.search(r"(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?$", sentence_before):
            return True
        if re.search(r"^(?:公司|集团|商行|工作室|合作社|经营部)", sentence_after):
            return True
        return False

    def _is_valid_organization_candidate(self, normalized: str, source: str) -> bool:
        if not normalized or normalized in self.SPATIAL_NOISE_TERMS:
            return False
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return False
        if normalized in self.GENERIC_ORGANIZATION_TERMS:
            return False
        if normalized in self.PUBLIC_ORG_NOISE_TERMS:
            return False
        if (
            re.fullmatch(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", normalized)
            and not is_org_like_text(normalized)
            and not (
                source in {"qwen_fragment_review", "ollama", "llm", "ollama_prose", "ollama_specialized"}
                and looks_like_organization_short_name(normalized)
            )
        ):
            return False
        if source in {"uie", "ner", "secondary_ner", "propagate"} and len(normalized) <= 2 and not self.ORGANIZATION_SUFFIX_PATTERN.search(normalized):
            return False
        if source in {"uie", "ner", "secondary_ner", "propagate"} and normalized in {"气象", "中心", "服务", "和国"}:
            return False
        if source in {"ner", "secondary_ner", "uie", "propagate"} and normalized.endswith(("部", "部门")) and not self.ORGANIZATION_SUFFIX_PATTERN.search(normalized):
            return False
        if len(normalized) > 12 and any(
            token in normalized
            for token in ("该表", "列明", "部分", "直接付款", "个人银行", "转账凭证", "证明", "收取", "款项", "认为", "提交")
        ):
            return False
        if len(normalized) > 16 and any(token in normalized for token in ("的", "是", "至", "了")):
            return False
        if any(token in normalized for token in ("主要负责的是", "负责的是", "通讯地址", "工作地址")):
            return False
        if self._looks_like_address_fragment(normalized) and not self.ORGANIZATION_SUFFIX_PATTERN.search(normalized):
            return False
        if source == "propagate" and len(normalized) <= 3 and self._looks_like_location_alias(normalized):
            return False
        if source in {"ner", "secondary_ner", "propagate"} and len(normalized) <= 2 and not self.ORGANIZATION_SUFFIX_PATTERN.search(normalized):
            return False
        return True

    def _looks_like_address_fragment(self, normalized: str) -> bool:
        address_tokens = ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "号", "栋", "室", "园", "小区")
        has_multiple_address_tokens = sum(1 for token in address_tokens if token in normalized) >= 2
        has_structured_building_marker = bool(
            re.search(r"(?:[A-Za-z]?\d+组团|\d+(?:号楼|栋|幢|单元|室)|\d+-\d+)", normalized, re.I)
        )
        has_place_context = any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "园", "小区"))
        return has_multiple_address_tokens or (has_structured_building_marker and has_place_context)

    def _looks_like_location_alias(self, normalized: str) -> bool:
        location_aliases = {
            "北京",
            "上海",
            "天津",
            "重庆",
            "深圳",
            "广州",
            "杭州",
            "南京",
            "成都",
            "武汉",
            "西安",
            "苏州",
            "佛山",
            "东莞",
        }
        if normalized in location_aliases:
            return True
        return normalized.endswith(("省", "市", "区", "县", "镇", "乡", "村", "路", "街"))

    def _is_weak_region_reference(self, normalized: str) -> bool:
        if normalized in {"国家", "全国", "全省", "全市", "全区", "南区", "北区", "东区", "西区"}:
            return True
        if re.fullmatch(r"\d+个(?:省份|省|城市|地区|区域)", normalized):
            return True
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}(?:省|市|区|县)?", normalized):
            if normalized.endswith(("路", "街", "村", "镇", "乡")):
                return False
            common_regions = {
                "北京", "上海", "天津", "重庆", "广州", "深圳", "广东", "浙江", "福建", "河南",
                "江西", "湖南", "湖北", "广西", "贵州", "云南", "四川", "西藏", "安徽", "江苏",
                "山东", "山西", "河北", "河南", "辽宁", "吉林", "黑龙江", "陕西", "甘肃", "青海",
                "宁夏", "新疆", "内蒙古", "海南", "香港", "澳门", "台湾",
                "北京市", "上海市", "天津市", "重庆市", "广东省", "浙江省", "福建省", "安徽省",
                "贵州省", "云南省", "四川省", "湖北省", "湖南省", "河南省", "江西省", "广西",
            }
            return normalized in common_regions
        return False

    @staticmethod
    def _is_broad_region_list(normalized: str) -> bool:
        if any(token in normalized for token in ("业务覆盖区域", "负责区域", "销售区域", "管辖区域")):
            return True
        separators = normalized.count("、") + normalized.count("+") + normalized.count("/")
        if separators >= 3 and not any(token in normalized for token in ("路", "街", "号", "栋", "室", "村", "组", "小区", "大厦", "广场")):
            return True
        return False

    @staticmethod
    def _is_detailed_location_candidate(normalized: str) -> bool:
        return len(normalized) >= 4 and any(
            token in normalized
            for token in ("路", "街", "号", "栋", "室", "村", "组", "小区", "大厦", "广场")
        )

    def _derive_candidate_texts(self, text: str, entity_type: str) -> List[str]:
        candidates = [text]
        trimmed = text.strip()

        organization_family = {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "PROJECT", "ALIAS"}
        official_family = {"BANK_NAME", "GOVERNMENT", "COURT"}
        if entity_type in {"ORGANIZATION", "LOCATION", "GOVERNMENT", "COMPANY_NAME", "ACCOUNT_NAME", "BANK_NAME", "COURT", "PROJECT"}:
            stripped = re.sub(r"^(?:[\u4e00-\u9fa5]{2,9}(?:省|市|区|县|镇|乡|街道))+", "", trimmed)
            if len(stripped) >= 4 and stripped != trimmed:
                if entity_type != "LOCATION" or self._is_valid_location_candidate(stripped):
                    candidates.append(stripped)

        if entity_type in organization_family:
            match = re.search(
                r"([\u4e00-\u9fa5A-Za-z0-9]{2,}(?:股份有限公司|有限责任公司|有限公司|集团有限公司|服务中心|技术中心|研究院|事务所|银行|支行|分行|研究所))$",
                trimmed,
            )
            if match:
                candidates.append(match.group(1))
        elif entity_type in official_family:
            if is_official_institution_text(trimmed):
                candidates.append(trimmed)

        if entity_type in {"PERSON", "PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}:
            candidates.extend(self._derive_person_aliases(trimmed))

        if entity_type == "ALIAS":
            metadata_text = ""
            # ALIAS seeds carry their full subject in metadata; the caller passes
            # only text here, so keep direct alias propagation conservative.
            metadata_text = trimmed
            if metadata_text:
                candidates.append(metadata_text)

        if entity_type in organization_family and not is_official_institution_text(trimmed):
            candidates.extend(self._derive_organization_aliases(trimmed))

        deduplicated: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduplicated:
                deduplicated.append(candidate)
        if entity_type in organization_family:
            deduplicated = [
                item
                for _index, item in sorted(
                    enumerate(deduplicated),
                    key=lambda row: (-len(re.sub(r"\s+", "", row[1] or "")), row[0]),
                )
            ]
        return deduplicated

    def _derive_organization_aliases(self, text: str) -> List[str]:
        aliases: List[str] = []
        stripped = text.strip()
        if len(stripped) < 4:
            return aliases
        company_like = self._looks_like_company_subject(stripped)

        alias_source = re.sub(r"[（(][^）)]{1,12}[）)]", "", stripped)
        region_stripped = re.sub(
            r"^(?:[\u4e00-\u9fa5]{2,9}(?:省|市|区|县|镇|乡|街道))+",
            "",
            alias_source,
        )
        compact_region = re.sub(
            r"^(?:北京|上海|广州|深圳|天津|重庆|杭州|南京|苏州|成都|武汉|西安|长沙|郑州|青岛|宁波|佛山|东莞|厦门|福州|济南|合肥|昆明|南宁|贵阳|南昌|海口|太原|沈阳|长春|哈尔滨|石家庄|呼和浩特|乌鲁木齐|拉萨|银川|西宁|广东|广西|海南|河北|河南|湖北|湖南|江苏|浙江|安徽|福建|江西|山东|山西|陕西|四川|贵州|云南|辽宁|吉林|黑龙江|甘肃|青海|台湾|内蒙古|宁夏|新疆|西藏|香港|澳门)",
            "",
            region_stripped,
        )
        core = re.sub(
            r"(股份有限公司|有限责任公司|有限公司|集团有限公司|集团|研究院|研究所|服务中心|事务所|分公司|子公司|公司)$",
            "",
            compact_region or region_stripped,
        )
        business_stripped = self._strip_organization_business_suffix(core)
        if len(core) >= 2:
            aliases.append(core)
            if company_like:
                for size in (2, 3, 4, 5, 6):
                    if len(core) > size:
                        aliases.append(core[:size])

        brand = business_stripped or core
        if len(brand) >= 2 and brand != core:
            aliases.append(brand)
            if company_like:
                for size in (2, 3, 4, 5, 6):
                    if len(brand) > size:
                        aliases.append(brand[:size])

        if company_like and len(brand) >= 2:
            aliases.append(f"{brand}公司")
            aliases.append(f"{brand}集团")

        if company_like and len(core) >= 2:
            aliases.append(f"{core}公司")

        if company_like:
            for tail_alias in self._derive_tail_name_aliases(brand or core):
                aliases.append(tail_alias)
                aliases.append(f"{tail_alias}公司")

        deduplicated: List[str] = []
        for alias in aliases:
            normalized_alias = re.sub(r"\s+", "", alias or "")
            if not 2 <= len(normalized_alias) <= len(stripped):
                continue
            if normalized_alias == stripped:
                continue
            if (
                normalized_alias != re.sub(r"\s+", "", stripped)
                and self._is_low_information_organization_alias(normalized_alias)
            ):
                continue
            if alias not in deduplicated:
                deduplicated.append(alias)
        return deduplicated

    @staticmethod
    def _looks_like_company_subject(text: str) -> bool:
        normalized = re.sub(r"\s+", "", text or "")
        return any(
            token in normalized
            for token in (
                "公司",
                "有限公司",
                "有限责任公司",
                "股份有限公司",
                "集团",
                "集团有限公司",
            )
        )

    def _strip_organization_business_suffix(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text or "")
        if len(normalized) < 3:
            return normalized
        for token in self.ORGANIZATION_BUSINESS_SUFFIXES:
            if normalized.endswith(token) and len(normalized) - len(token) >= 2:
                return normalized[: -len(token)]
        return normalized

    def _is_low_information_organization_alias(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text or "")
        if len(normalized) < 2:
            return True
        if normalized in self.LOW_INFORMATION_ORGANIZATION_ALIASES:
            return True
        remainder = normalized
        while remainder:
            matched = False
            for token in self.ORGANIZATION_BUSINESS_SUFFIXES:
                if remainder.startswith(token):
                    remainder = remainder[len(token) :]
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _derive_person_aliases(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", "", text)
        if re.fullmatch(r"[\u4e00-\u9fa5\u00b7]{2,5}", normalized) is None:
            return []
        if len(normalized) <= 1:
            return []
        if len(normalized) == 2:
            return [f"{normalized[0]}某"]
        return [f"{normalized[0]}某{normalized[-1]}"]

    def _derive_tail_name_aliases(self, text: str) -> List[str]:
        normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]", "", text)
        if len(normalized) < 3:
            return []

        aliases: List[str] = []
        for size in (2, 3, 4, 5, 6):
            if len(normalized) <= size:
                continue
            alias = normalized[-size:]
            if len(alias) >= 2:
                aliases.append(alias)
        return aliases

    def _derive_project_aliases(self, text: str) -> List[str]:
        aliases: List[str] = []
        stripped = text.strip()
        if len(stripped) < 4:
            return aliases

        core = re.sub(r"(项目|工程|标段)$", "", stripped)
        if len(core) >= 2 and core != stripped:
            aliases.append(core)

        short_core = re.sub(r"(采购|建设|安装|服务|检测|治理|改造|施工)$", "", core)
        if len(short_core) >= 2 and short_core != core:
            aliases.append(short_core)

        return [alias for alias in aliases if alias != stripped]

    def _is_valid_location_candidate(self, text: str) -> bool:
        address_tokens = ["省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室", "广场"]
        generic_place_names = ["人民法院", "法院", "人民检察院", "检察院", "公安局", "派出所", "委员会"]

        if any(token in text for token in address_tokens):
            return True
        if text in generic_place_names:
            return False
        return False

    def _find_all_occurrences(self, text: str, target: str) -> List[int]:
        positions: List[int] = []
        start = 0
        while target:
            index = text.find(target, start)
            if index == -1:
                break
            positions.append(index)
            start = index + len(target)
        return positions

    def _find_all_occurrence_spans(self, text: str, target: str) -> List[tuple[int, int, str]]:
        matches = [
            (start, start + len(target), text[start : start + len(target)])
            for start in self._find_all_occurrences(text, target)
        ]
        seen = {(start, end) for start, end, _ in matches}

        normalized_target = re.sub(r"\s+", "", target)
        if not normalized_target:
            return matches

        normalized_chars: List[str] = []
        index_map: List[int] = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            normalized_chars.append(char)
            index_map.append(index)

        normalized_text = "".join(normalized_chars)
        search_from = 0

        while True:
            normalized_start = normalized_text.find(normalized_target, search_from)
            if normalized_start == -1:
                break

            normalized_end = normalized_start + len(normalized_target) - 1
            start = index_map[normalized_start]
            end = index_map[normalized_end] + 1
            if (start, end) not in seen:
                matches.append((start, end, text[start:end]))
                seen.add((start, end))
            search_from = normalized_start + len(normalized_target)

        return matches

    def _overlaps_existing(
        self,
        start: int,
        end: int,
        results: List[RecognizerResult],
    ) -> bool:
        for item in results:
            if start < item.end and end > item.start:
                return True
        return False

    def get_statistics(self) -> Dict:
        return {
            "recognizers": self.recognizer_registry.get_statistics(),
            "operators": self.operator_registry.get_statistics(),
        }
