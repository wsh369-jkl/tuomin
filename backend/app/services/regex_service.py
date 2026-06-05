"""Regex-based recognizer for deterministic Chinese sensitive entities."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RegexService:
    """Recognize strongly formatted entities with deterministic regex rules."""

    PATTERNS: Dict[str, Dict[str, object]] = {
        "CN_ID_CARD": {
            "regex": (
                r"(?<!\d)(?:[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
                r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
                r"|[1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3})(?!\d)"
            ),
            "score": 0.95,
            "name": "Chinese ID card",
            "priority": 1,
        },
        "CN_PHONE": {
            "regex": r"(?<!\d)1[3-9]\d(?:[ \t\u00a0\-－—–]?\d{4}){2}(?!\d)",
            "score": 0.90,
            "name": "Chinese mobile phone",
            "priority": 2,
        },
        "CN_CREDIT_CODE": {
            "regex": r"(?=[0-9A-HJ-NPQRTUWXY]{18})(?=.*[A-HJ-NPQRTUWXY])[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}",
            "score": 0.95,
            "name": "Unified social credit code",
            "priority": 1,
        },
        "CN_BANK_CARD": {
            "regex": r"(?<!\d)\d{16,20}(?!\d)",
            "score": 0.85,
            "name": "Bank card",
            "priority": 4,
        },
        "AMOUNT": {
            "regex": r"\d+(?:\.\d+)?[万亿]?元",
            "score": 0.80,
            "name": "Amount",
            "priority": 5,
        },
        "EMAIL_ADDRESS": {
            "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "score": 0.90,
            "name": "Email",
            "priority": 2,
        },
    }

    CREDIT_CODE_LABELS = ["统一社会信用代码", "社会信用代码", "信用代码"]
    ID_CARD_LABELS = ["身份证号码", "身份证号", "公民身份号码", "公民身份证号码"]
    ACCOUNT_NUMBER_LABELS = [
        "银行账号",
        "银行账户",
        "账号",
        "帐号",
        "账户号码",
        "账户号",
        "账户",
        "收款账号",
        "收款账户",
        "付款账号",
        "付款账户",
    ]

    def __init__(self) -> None:
        logger.info("Initializing regex recognizer...")
        self.compiled_patterns = {
            entity_type: re.compile(str(pattern_info["regex"]))
            for entity_type, pattern_info in self.PATTERNS.items()
        }
        logger.info("Regex recognizer initialized with %s patterns", len(self.compiled_patterns))

    async def analyze(self, text: str, language: str = "zh") -> List[Dict]:
        """Return non-overlapping deterministic matches ordered by position."""

        logger.info("Regex analysis started, text_length=%s, language=%s", len(text), language)
        raw_entities: List[Dict] = []

        try:
            for entity_type, compiled_regex in self.compiled_patterns.items():
                pattern_info = self.PATTERNS[entity_type]
                for match in compiled_regex.finditer(text):
                    start, end = match.span()
                    start, end, matched_text = self._expand_contextual_match(
                        text=text,
                        start=start,
                        end=end,
                        entity_type=entity_type,
                    )
                    if not matched_text:
                        continue

                    resolved_type = self._resolve_contextual_entity_type(
                        text=text,
                        start=start,
                        end=end,
                        entity_type=entity_type,
                    )
                    resolved_pattern = self.PATTERNS[resolved_type]
                    raw_entities.append(
                        {
                            "type": resolved_type,
                            "start": start,
                            "end": end,
                            "score": float(resolved_pattern["score"]),
                            "text": matched_text,
                            "source": "regex",
                            "_priority": int(resolved_pattern["priority"]),
                        }
                    )

            resolved_entities = self._resolve_overlaps(raw_entities)
            logger.info("Regex analysis completed, entities=%s", len(resolved_entities))
            return resolved_entities
        except Exception as exc:
            logger.error("Regex analysis failed: %s", exc)
            return []

    async def anonymize(
        self,
        text: str,
        entities: List[Dict],
        operators: Optional[Dict] = None,
    ) -> str:
        """Apply simple masking to regex-detected entities."""

        logger.info("Regex anonymization started, entities=%s", len(entities))

        try:
            sorted_entities = sorted(entities, key=lambda item: int(item["start"]), reverse=True)
            result = text

            for entity in sorted_entities:
                start = int(entity["start"])
                end = int(entity["end"])
                entity_type = str(entity["type"])
                original_text = str(entity["text"])
                masked_text = self._mask_entity(original_text, entity_type)
                result = result[:start] + masked_text + result[end:]

            logger.info("Regex anonymization completed")
            return result
        except Exception as exc:
            logger.error("Regex anonymization failed: %s", exc)
            return text

    def _resolve_overlaps(self, entities: List[Dict]) -> List[Dict]:
        if not entities:
            return []

        ordered = sorted(
            entities,
            key=lambda item: (
                int(item["start"]),
                int(item["_priority"]),
                -(int(item["end"]) - int(item["start"])),
                -float(item["score"]),
            ),
        )
        selected: List[Dict] = []

        for entity in ordered:
            overlap_index = self._find_overlap(selected, entity)
            if overlap_index == -1:
                selected.append(entity)
                continue

            current = selected[overlap_index]
            if self._prefer_candidate(entity, current):
                selected[overlap_index] = entity

        selected.sort(key=lambda item: (int(item["start"]), int(item["end"])))
        for entity in selected:
            entity.pop("_priority", None)
        return selected

    def _find_overlap(self, selected: List[Dict], candidate: Dict) -> int:
        candidate_start = int(candidate["start"])
        candidate_end = int(candidate["end"])

        for index, entity in enumerate(selected):
            entity_start = int(entity["start"])
            entity_end = int(entity["end"])
            if candidate_start < entity_end and candidate_end > entity_start:
                return index
        return -1

    def _prefer_candidate(self, candidate: Dict, current: Dict) -> bool:
        candidate_priority = int(candidate["_priority"])
        current_priority = int(current["_priority"])
        if candidate_priority != current_priority:
            return candidate_priority < current_priority

        candidate_length = int(candidate["end"]) - int(candidate["start"])
        current_length = int(current["end"]) - int(current["start"])
        if candidate_length != current_length:
            return candidate_length > current_length

        return float(candidate["score"]) > float(current["score"])

    def _expand_contextual_match(
        self,
        *,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> tuple[int, int, str]:
        if entity_type == "CN_BANK_CARD" and self._has_context_label(
            text=text,
            start=start,
            labels=self.ID_CARD_LABELS,
        ):
            if end < len(text) and text[end] in {"X", "x"}:
                end += 1

        matched_text = text[start:end]
        if entity_type == "CN_BANK_CARD":
            digit_count = len(re.sub(r"\D", "", matched_text))
            if digit_count == 20 and not self._has_context_label(
                text=text,
                start=start,
                labels=self.ACCOUNT_NUMBER_LABELS,
            ):
                return start, end, ""

        return start, end, text[start:end]

    def _resolve_contextual_entity_type(
        self,
        *,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> str:
        if entity_type != "CN_BANK_CARD":
            return entity_type

        if self._has_context_label(text=text, start=start, labels=self.CREDIT_CODE_LABELS):
            return "CN_CREDIT_CODE"
        if self._has_context_label(text=text, start=start, labels=self.ID_CARD_LABELS):
            return "CN_ID_CARD"
        return entity_type

    def _has_context_label(self, *, text: str, start: int, labels: List[str]) -> bool:
        prefix = re.sub(r"\s+", "", text[max(0, start - 24) : start])
        return any(label in prefix for label in labels)

    def _mask_entity(self, text: str, entity_type: str) -> str:
        if entity_type == "CN_ID_CARD":
            if len(text) >= 10:
                return text[:6] + "*" * (len(text) - 10) + text[-4:]
            return "*" * len(text)

        if entity_type == "CN_PHONE":
            if len(text) == 11:
                return text[:3] + "****" + text[-4:]
            return "*" * len(text)

        if entity_type == "CN_CREDIT_CODE":
            if len(text) >= 10:
                return text[:8] + "*" * (len(text) - 10) + text[-2:]
            return "*" * len(text)

        if entity_type == "CN_BANK_CARD":
            if len(text) >= 8:
                return text[:4] + "*" * (len(text) - 8) + text[-4:]
            return "*" * len(text)

        if entity_type == "AMOUNT":
            return "[金额]"

        if entity_type == "EMAIL_ADDRESS":
            if "@" in text:
                username, domain = text.split("@", 1)
                if len(username) > 1:
                    masked_username = username[0] + "*" * (len(username) - 1)
                else:
                    masked_username = "*"
                return masked_username + "@" + domain
            return "*" * len(text)

        if entity_type == "PERSON":
            return "[姓名]"

        if entity_type == "ORGANIZATION":
            return "[公司名称]"

        if entity_type == "LOCATION":
            return "[地址]"

        if entity_type == "POSITION":
            return "[职位]"

        return "*" * len(text)
