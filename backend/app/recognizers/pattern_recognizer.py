"""Pattern-based recognizers for structured Chinese entities."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.core.identifier_rules import (
    ALL_IDENTIFIER_LABELS,
    CASE_NUMBER_VALUE_PATTERN,
    build_spaced_label_pattern,
    resolve_identifier_kind,
)
from app.core.recognizer_base import BaseRecognizer, RecognizerResult

logger = logging.getLogger(__name__)
IDENTIFIER_LABEL_PATTERN = build_spaced_label_pattern(ALL_IDENTIFIER_LABELS)


class PatternRecognizer(BaseRecognizer):
    """Base recognizer that matches entities with regular expressions."""

    def __init__(
        self,
        name: str,
        patterns: Dict[str, Dict],
        supported_language: str = "zh",
        version: str = "1.0.0",
    ) -> None:
        super().__init__(
            name=name,
            supported_entities=list(patterns.keys()),
            supported_language=supported_language,
            version=version,
        )
        self.patterns = patterns
        self.compiled_patterns: Dict[str, Dict] = {}

        for entity_type, pattern_info in patterns.items():
            flags = pattern_info.get("flags", 0)
            try:
                self.compiled_patterns[entity_type] = {
                    "regex": re.compile(pattern_info["regex"], flags),
                    "score": pattern_info.get("score", 0.9),
                    "group": pattern_info.get("group"),
                }
            except re.error as exc:
                logger.error("Failed to compile regex for %s: %s", entity_type, exc)

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        if not self.enabled:
            return []

        target_entities = entities or self.supported_entities
        results: List[RecognizerResult] = []

        for entity_type in target_entities:
            pattern_info = self.compiled_patterns.get(entity_type)
            if pattern_info is None:
                continue

            regex = pattern_info["regex"]
            score = pattern_info["score"]
            group_name = pattern_info.get("group")

            for match in regex.finditer(text):
                start, end, matched_text = self._extract_match(match, group_name)
                if start >= end or not matched_text:
                    continue

                metadata = {"pattern": regex.pattern}
                if "label" in match.re.groupindex:
                    label_value = (match.group("label") or "").strip()
                    if label_value:
                        metadata["label"] = label_value
                if entity_type == "CONTRACT_NO":
                    metadata["identifier_kind"] = self._resolve_identifier_kind(
                        matched_text,
                        metadata.get("label"),
                    )

                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        score=score,
                        text=matched_text,
                        source=self.name,
                        metadata=metadata,
                    )
                )

        return results

    def _extract_match(
        self,
        match: re.Match[str],
        group_name: Optional[str],
    ) -> tuple[int, int, str]:
        if group_name and group_name in match.re.groupindex and match.group(group_name):
            start, end = match.span(group_name)
            matched_text = match.group(group_name).strip()
            return start, end, matched_text

        if "value" in match.re.groupindex and match.group("value"):
            start, end = match.span("value")
            matched_text = match.group("value").strip()
            return start, end, matched_text

        if "value_inline" in match.re.groupindex and match.group("value_inline"):
            start, end = match.span("value_inline")
            matched_text = match.group("value_inline").strip()
            return start, end, matched_text

        start, end = match.span()
        matched_text = match.group().strip()
        return start, end, matched_text

    @staticmethod
    def _resolve_identifier_kind(value: str, label: Optional[str]) -> str:
        return resolve_identifier_kind(value=value, label=label)


class ChinesePatternRecognizer(PatternRecognizer):
    """Regex recognizer for common structured entities in Chinese contracts."""

    def __init__(self) -> None:
        patterns = {
            "CN_ID_CARD": {
                "regex": r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
                "score": 0.95,
            },
            "CN_PHONE": {
                "regex": r"(?<!\d)1[3-9]\d{9}(?!\d)",
                "score": 0.92,
            },
            "LANDLINE_PHONE": {
                "regex": r"(?<!\d)(?:0\d{2,3}[-－—–]\d{7,8}|0\d{9,11}|400[-－—–]\d{3}[-－—–]\d{4}|400\d{7})(?!\d)",
                "score": 0.88,
            },
            "CN_CREDIT_CODE": {
                "regex": r"(?<![0-9A-Z])(?!\d{18}(?![0-9A-Z]))[0-9A-HJ-NPQRTUWXY]{18}(?![0-9A-Z])",
                "score": 0.95,
            },
            "CN_BANK_CARD": {
                "regex": r"(?<!\d)\d{16,19}(?!\d)",
                "score": 0.9,
            },
            "EMAIL_ADDRESS": {
                "regex": r"(?<![\w.-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])",
                "score": 0.92,
            },
            "AMOUNT": {
                "regex": r"(?:[\u00a5\uffe5]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:\u4ebf\u5143|\u4e07\u5143|\u5143|\u4e07|\u4ebf)|(?:\u4eba\u6c11\u5e01)?[零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟]+(?:\u5143|\u5706)(?:\u6574|\u6b63)?",
                "score": 0.86,
            },
            "DATE": {
                "regex": r"(?:\d{4}[\u5e74/-]\d{1,2}[\u6708/-](?:\d{1,2}|xx|XX)[\u65e5\u53f7]?|\d{4}\u5e74\d{1,2}\u6708(?:\d{1,2}|xx|XX)\u65e5?)",
                "score": 0.88,
            },
            "CONTRACT_NO": {
                "regex": (
                    rf"(?:(?:(?P<label>{IDENTIFIER_LABEL_PATTERN})\s*[:\uff1a]\s*)"
                    r"(?P<value>[A-Za-z0-9\u4e00-\u9fa5\-\[\]\u3014\u3015\uff08\uff09()/. ]{4,}?)(?=$|[\r\n\t,\uff0c;\uff1b]))"
                    rf"|(?P<value_inline>{CASE_NUMBER_VALUE_PATTERN})"
                ),
                "score": 0.9,
                "group": "value",
            },
        }

        super().__init__(name="regex", patterns=patterns, supported_language="zh")
