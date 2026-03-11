"""Pipeline manager for recognition and anonymization."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.core.operator_registry import OperatorRegistry
from app.core.recognizer_base import RecognizerResult
from app.core.recognizer_registry import RecognizerRegistry
from app.services.contextual_desensitization_service import ContextualDesensitizationService

logger = logging.getLogger(__name__)


class PipelineManager:
    """Coordinate recognizers, merge their output, and apply operators."""

    PRIORITY_MAP = {
        "llm": 1,
        "ollama": 1,
        "contract": 2,
        "custom": 3,
        "propagate": 4,
        "regex": 5,
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
        "AMOUNT": 1,
        "DATE": 1,
        "PROJECT": 2,
        "LOCATION": 2,
        "POSITION": 2,
        "PERSON": 2,
        "PERSON_NAME": 2,
        "ORGANIZATION": 3,
        "COMPANY_NAME": 3,
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
        "CONTRACT_NO",
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
    ) -> List[Dict]:
        logger.info("Analyze pipeline started, text_length=%s", len(text))

        results = await self.recognizer_registry.analyze(
            text,
            entities,
            recognizer_names,
            llm_model=llm_model,
        )
        merged_results = self._merge_and_deduplicate(results)
        validated_results = self._validate_results(merged_results, text)
        validated_results = self._expand_repeated_mentions(validated_results, text)
        dict_results = [result.to_dict() for result in validated_results]
        dict_results = await self.contextual_desensitizer.refine_recognition_entities(
            text=text,
            entities=dict_results,
            use_llm=bool(recognizer_names and "llm" in recognizer_names),
            llm_model=llm_model,
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

        if not entities:
            return text

        config = self._get_default_operator_config()
        if operator_config:
            config.update(operator_config)

        sorted_entities = sorted(entities, key=lambda item: item["start"], reverse=True)
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
    ) -> List[Dict]:
        return await self.contextual_desensitizer.prepare_entities(
            text=text,
            entities=entities,
            use_llm=use_llm,
            operator_config=operator_config,
            llm_model=llm_model,
            anonymization_strategy=anonymization_strategy,
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

        for result in results:
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

            validated.append(result)

        logger.debug("Validated entities: %s -> %s", len(results), len(validated))
        return validated

    def _get_default_operator_config(self) -> Dict[str, Dict]:
        return {
            "CN_ID_CARD": {
                "operator": "mask",
                "params": {"keep_start": 6, "keep_end": 4},
            },
            "CN_PHONE": {
                "operator": "mask",
                "params": {"keep_start": 3, "keep_end": 4},
            },
            "LANDLINE_PHONE": {
                "operator": "mask",
                "params": {"keep_start": 4, "keep_end": 2},
            },
            "CN_BANK_CARD": {
                "operator": "mask",
                "params": {"keep_start": 4, "keep_end": 4},
            },
            "CN_CREDIT_CODE": {
                "operator": "mask",
                "params": {"keep_start": 8, "keep_end": 2},
            },
            "EMAIL_ADDRESS": {
                "operator": "mask",
                "params": {"mask_email": True},
            },
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
                "params": {"new_value": "[\u516c\u53f8]"},
            },
            "LOCATION": {
                "operator": "replace",
                "params": {"new_value": "[\u5730\u5740]"},
            },
            "POSITION": {
                "operator": "replace",
                "params": {"new_value": "[\u804c\u4f4d]"},
            },
            "PROJECT": {
                "operator": "replace",
                "params": {"new_value": "[\u9879\u76ee\u540d\u79f0]"},
            },
            "CONTRACT_NO": {
                "operator": "mask",
                "params": {"keep_start": 0, "keep_end": 0},
            },
            "BANK_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u5f00\u6237\u884c]"},
            },
            "ACCOUNT_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u6237\u540d]"},
            },
            "PROJECT_CODE": {
                "operator": "replace",
                "params": {"new_value": "[\u9879\u76ee\u4ee3\u53f7]"},
            },
            "PRODUCT_NAME": {
                "operator": "replace",
                "params": {"new_value": "[\u4ea7\u54c1\u540d\u79f0]"},
            },
            "SENSITIVE_TERM": {
                "operator": "replace",
                "params": {"new_value": "[\u654f\u611f\u672f\u8bed]"},
            },
            "AMOUNT": {
                "operator": "replace",
                "params": {"new_value": "[\u91d1\u989d]"},
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

        for result in sorted(results, key=lambda item: (-(item.end - item.start), item.start)):
            if result.entity_type not in self.PROPAGATION_TYPES:
                continue

            for candidate_text in self._derive_candidate_texts(result.text, result.entity_type):
                if len(candidate_text.strip()) < 2:
                    continue

                for start, end, matched_text in self._find_all_occurrence_spans(text, candidate_text):
                    key = (result.entity_type, start, end, matched_text)
                    if key in seen:
                        continue
                    if self._overlaps_existing(start, end, expanded):
                        continue

                    expanded.append(
                        RecognizerResult(
                            entity_type=result.entity_type,
                            start=start,
                            end=end,
                            score=max(0.72, float(result.score) - 0.08),
                            text=matched_text,
                            source="propagate",
                            metadata={"derived_from": result.text, "candidate_text": candidate_text},
                        )
                    )
                    seen.add(key)

        expanded.sort(key=lambda item: (item.start, item.end))
        return expanded

    def _derive_candidate_texts(self, text: str, entity_type: str) -> List[str]:
        candidates = [text]
        trimmed = text.strip()

        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "PROJECT", "LOCATION"}:
            stripped = re.sub(r"^(?:[\u4e00-\u9fa5]{2,9}(?:省|市|区|县|镇|乡|街道))+", "", trimmed)
            if len(stripped) >= 4 and stripped != trimmed:
                if entity_type != "LOCATION" or self._is_valid_location_candidate(stripped):
                    candidates.append(stripped)

        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            match = re.search(
                r"([\u4e00-\u9fa5A-Za-z0-9]{2,}(?:股份有限公司|有限责任公司|有限公司|集团有限公司|服务中心|技术中心|研究院|事务所|银行|支行|分行|研究所))$",
                trimmed,
            )
            if match:
                candidates.append(match.group(1))

        if entity_type == "PROJECT":
            match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9.\-]{2,}(?:项目|工程|标段))$", trimmed)
            if match:
                candidates.append(match.group(1))

        if entity_type == "BANK_NAME":
            match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{2,}银行(?:股份有限公司)?[\u4e00-\u9fa5A-Za-z0-9]{0,10}(?:支行|分行)?)$", trimmed)
            if match:
                candidates.append(match.group(1))

        if entity_type in {"PERSON", "PERSON_NAME"}:
            candidates.extend(self._derive_person_aliases(trimmed))

        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            candidates.extend(self._derive_organization_aliases(trimmed))

        if entity_type == "PROJECT":
            candidates.extend(self._derive_project_aliases(trimmed))

        deduplicated: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduplicated:
                deduplicated.append(candidate)
        return deduplicated

    def _derive_organization_aliases(self, text: str) -> List[str]:
        aliases: List[str] = []
        stripped = text.strip()
        if len(stripped) < 4:
            return aliases

        region_stripped = re.sub(
            r"^(?:[\u4e00-\u9fa5]{2,9}(?:省|市|区|县|镇|乡|街道))+",
            "",
            stripped,
        )
        core = re.sub(
            r"(股份有限公司|有限责任公司|有限公司|集团有限公司|集团|研究院|研究所|服务中心|事务所|分公司|子公司|公司)$",
            "",
            region_stripped,
        )
        if len(core) >= 2:
            aliases.append(core)

        brand = re.sub(
            r"(科技|工程|建设|贸易|实业|发展|咨询|服务|管理|材料|电力|能源|建筑|环保|智能|信息|网络|电子|机械|设备|制造)$",
            "",
            core,
        )
        if len(brand) >= 2 and brand != core:
            aliases.append(brand)

        if len(brand) >= 2:
            aliases.append(f"{brand}公司")
            aliases.append(f"{brand}集团")

        if len(core) >= 2:
            aliases.append(f"{core}公司")

        for tail_alias in self._derive_tail_name_aliases(brand or core):
            aliases.append(tail_alias)
            aliases.append(f"{tail_alias}公司")

        return [alias for alias in aliases if 2 <= len(alias) <= len(stripped) and alias != stripped]

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
        if len(normalized) < 4:
            return []

        aliases: List[str] = []
        for size in (2, 3, 4):
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
