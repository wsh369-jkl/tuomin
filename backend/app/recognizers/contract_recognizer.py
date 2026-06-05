"""Contract-specific recognizer for common labelled fields."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.core.identifier_rules import (
    CASE_NUMBER_LABELS,
    build_spaced_label_pattern,
    extract_case_number,
    label_matches,
    resolve_identifier_kind,
)
from app.core.recognizer_base import BaseRecognizer, RecognizerResult


class ContractFieldRecognizer(BaseRecognizer):
    """Extract entities from common contract labels."""

    STANDALONE_LABEL_PREFIX = re.compile(
        r"^(?:[\s【\[\(（]*|"
        r"第?[一二三四五六七八九十\d]+[、.)．）]?\s*|"
        r"[（(][一二三四五六七八九十\d]+[）)]\s*|"
        r"[-*•]+\s*)$"
    )

    FIELD_SPECS: List[Dict] = [
        {
            "entity_type": "CONTRACT_NO",
            "labels": [
                "\u5408\u540c\u7f16\u53f7",
                "\u5408\u540c\u53f7",
                "\u9879\u76ee\u7f16\u53f7",
                "\u62db\u6807\u7f16\u53f7",
                "\u534f\u8bae\u7f16\u53f7",
                "\u6848\u53f7",
                "\u6848\u4ef6\u7f16\u53f7",
                "\u53d7\u7406\u6848\u53f7",
                "\u6267\u884c\u6848\u53f7",
                "原审案号",
                "一审案号",
                "二审案号",
                "再审案号",
                "上诉案号",
                "审理案号",
            ],
            "score": 0.95,
        },
        {
            "entity_type": "PROJECT",
            "labels": [
                "\u5de5\u7a0b\u540d\u79f0",
                "\u9879\u76ee\u540d\u79f0",
                "\u9879\u76ee\u540d\u79f0\u53ca\u5730\u70b9",
                "\u9879\u76ee\u540d\u79f0\u53ca\u5730\u5740",
                "\u6807\u6bb5\u540d\u79f0",
            ],
            "score": 0.94,
        },
        {
            "entity_type": "LOCATION",
            "labels": [
                "\u5de5\u7a0b\u5730\u5740",
                "\u9879\u76ee\u5730\u5740",
                "\u5de5\u7a0b\u5730\u70b9",
                "\u5efa\u8bbe\u5730\u70b9",
                "\u8eab\u4efd\u8bc1\u4f4f\u5740",
                "\u4f4f\u5740",
                "\u4f4f\u6240",
                "\u4f4f\u6240\u5730",
                "\u6ce8\u518c\u5730\u5740",
                "\u529e\u516c\u5730\u5740",
                "\u5bb6\u5ead\u5730\u5740",
                "\u8eab\u4efd\u8bc1\u4f4f\u5740",
                "\u7ecf\u5e38\u5c45\u4f4f\u5730",
                "\u6237\u7c4d\u5730\u5740",
                "\u6237\u7c4d\u5730",
                "\u73b0\u4f4f\u5740",
                "\u901a\u8baf\u5730\u5740",
                "\u9001\u8fbe\u5730\u5740",
                "\u5730\u5740",
                "\u8054\u7cfb\u5730\u5740",
            ],
            "score": 0.91,
        },
        {
            "entity_type": "ORGANIZATION",
            "labels": [
                "\u59d4\u6258\u65b9\uff08\u7532\u65b9\uff09",
                "\u59d4\u6258\u65b9(\u7532\u65b9)",
                "\u59d4\u6258\u65b9",
                "\u7532\u65b9",
                "\u53d7\u6258\u65b9\uff08\u4e59\u65b9\uff09",
                "\u53d7\u6258\u65b9(\u4e59\u65b9)",
                "\u53d7\u6258\u65b9",
                "\u4e59\u65b9",
                "\u53d1\u5305\u4eba",
                "\u627f\u5305\u4eba",
                "\u91c7\u8d2d\u4eba",
                "\u4f9b\u5e94\u5546",
                "\u6536\u6b3e\u5355\u4f4d",
            ],
            "score": 0.94,
        },
        {
            "entity_type": "PERSON",
            "labels": [
                "\u6cd5\u5b9a\u4ee3\u8868\u4eba",
                "\u6cd5\u4eba\u4ee3\u8868",
                "\u8054\u7cfb\u4eba",
                "\u59d4\u6258\u4ee3\u7406\u4eba",
                "\u9879\u76ee\u8d1f\u8d23\u4eba",
                "\u8d1f\u8d23\u4eba",
                "\u7ecf\u529e\u4eba",
                "\u62c5\u4fdd\u4eba",
                "\u4fdd\u8bc1\u4eba",
                "\u7b7e\u7f72\u4eba",
                "\u4e19\u65b9\u62c5\u4fdd\u4eba",
                "\u4e59\u65b9\u7ecf\u529e\u4eba",
                "\u7532\u65b9\u7ecf\u529e\u4eba",
            ],
            "score": 0.9,
        },
        {
            "entity_type": "POSITION",
            "labels": [
                "\u804c\u52a1",
                "\u804c\u4f4d",
                "\u5c97\u4f4d",
            ],
            "score": 0.87,
        },
        {
            "entity_type": "BANK_NAME",
            "labels": [
                "\u5f00\u6237\u884c",
                "\u5f00\u6237\u94f6\u884c",
                "\u94f6\u884c\u540d\u79f0",
            ],
            "score": 0.93,
        },
        {
            "entity_type": "ACCOUNT_NAME",
            "labels": [
                "\u8d26\u6237\u540d\u79f0",
                "\u8d26\u6237\u540d",
                "\u6237\u540d",
                "\u8d26\u6237",
            ],
            "score": 0.9,
        },
        {
            "entity_type": "CN_PHONE",
            "labels": [
                "\u8054\u7cfb\u7535\u8bdd",
                "\u7535\u8bdd",
                "\u8054\u7cfb\u4eba\u7535\u8bdd",
                "\u8054\u7cfb\u65b9\u5f0f",
                "\u624b\u673a\u53f7",
                "\u624b\u673a\u53f7\u7801",
                "\u624b\u673a",
            ],
            "score": 0.86,
        },
        {
            "entity_type": "CN_BANK_CARD",
            "labels": [
                "\u94f6\u884c\u8d26\u53f7",
                "\u94f6\u884c\u8d26\u6237",
                "\u8d26\u53f7",
                "\u5e10\u53f7",
                "\u8d26\u6237\u53f7\u7801",
                "\u8d26\u6237\u53f7",
                "\u6536\u6b3e\u8d26\u53f7",
                "\u6536\u6b3e\u8d26\u6237",
                "\u4ed8\u6b3e\u8d26\u53f7",
                "\u4ed8\u6b3e\u8d26\u6237",
            ],
            "score": 0.9,
        },
    ]

    TAIL_MARKERS = re.compile(
        r"(?=(?:"
        r"\u7532\u65b9|\u4e59\u65b9|\u59d4\u6258\u65b9|\u53d7\u6258\u65b9|"
        r"\u6cd5\u5b9a\u4ee3\u8868\u4eba|\u8054\u7cfb\u4eba|\u9879\u76ee\u8d1f\u8d23\u4eba|"
        r"\u7ecf\u529e\u4eba|\u62c5\u4fdd\u4eba|\u8054\u7cfb\u7535\u8bdd|\u624b\u673a\u53f7|"
        r"\u6ce8\u518c\u5730\u5740|\u8054\u7cfb\u5730\u5740|\u4f4f\u5740|\u5730\u5740|"
        r"\u5f00\u6237\u884c|\u8d26\u53f7|\u8d26\u6237|\u6237\u540d|\u8eab\u4efd\u8bc1\u53f7\u7801|\u8eab\u4efd\u8bc1\u53f7"
        r")\s*[:\uff1a])"
    )
    INLINE_PERSON_LABEL_RE = re.compile(
        r"(?P<label>[\u7532\u4e59\u4e19]?\u65b9?(?:\u6cd5\u5b9a\u4ee3\u8868\u4eba|\u6cd5\u4eba\u4ee3\u8868|\u9879\u76ee\u8d1f\u8d23\u4eba|\u7ecf\u529e\u4eba|\u8054\u7cfb\u4eba|\u62c5\u4fdd\u4eba|\u4fdd\u8bc1\u4eba|\u7b7e\u7f72\u4eba|\u8d1f\u8d23\u4eba))"
        r"\s*[:\uff1a]?\s*(?P<value>(?:[\u4e00-\u9fa5]{2,4}|[\u4e00-\u9fa5]{2,8}\u00b7[\u4e00-\u9fa5]{2,8}))"
        r"(?=[\uff0c,;\uff1b\u3002\\s]|$|\u8eab\u4efd\u8bc1\u53f7\u7801|\u8eab\u4efd\u8bc1\u53f7|\u7535\u5b50\u90ae\u7bb1|\u90ae\u7bb1|\u8054\u7cfb\u7535\u8bdd|\u8054\u7cfb\u5730\u5740|\u4f4f\u5740)"
    )
    INLINE_LOCATION_LABEL_RE = re.compile(
        r"(?P<label>\u6ce8\u518c\u5730\u5740|\u8054\u7cfb\u5730\u5740|\u901a\u8baf\u5730\u5740|\u9001\u8fbe\u5730\u5740|\u529e\u516c\u5730\u5740|\u5bb6\u5ead\u5730\u5740|\u8eab\u4efd\u8bc1\u4f4f\u5740|\u7ecf\u5e38\u5c45\u4f4f\u5730|\u6237\u7c4d\u5730\u5740|\u6237\u7c4d\u5730|\u73b0\u4f4f\u5740|\u4f4f\u5740|\u4f4f\u6240\u5730|\u4f4f\u6240|\u4f4f\u6240\u5730|\u5730\u5740)"
        r"\s*[:\uff1a]?\s*(?P<value>[^\n\r\uff1b;\u3002]{4,120})"
    )

    def __init__(self) -> None:
        supported_entities = sorted({item["entity_type"] for item in self.FIELD_SPECS})
        super().__init__(
            name="contract",
            supported_entities=supported_entities,
            supported_language="zh",
            version="1.0.0",
        )
        self._compiled_specs = [
            {
                **item,
                "regex": re.compile(
                    rf"(?P<label>{build_spaced_label_pattern(tuple(item['labels']))})"
                    rf"\s*(?:[:\uff1a]\s*|\s+)(?P<value>.+)"
                ),
            }
            for item in self.FIELD_SPECS
        ]

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        if not self.enabled:
            return []

        results: List[RecognizerResult] = []
        line_offset = 0

        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                line_offset += len(raw_line)
                continue

            segments = [
                segment
                for segment in re.split(r"(?:\s{2,}|\t+|\u3000+)", line)
                if segment.strip()
            ]
            if not segments:
                line_offset += len(raw_line)
                continue

            segment_cursor = 0
            for segment in segments:
                segment_start = line.find(segment, segment_cursor)
                if segment_start == -1:
                    continue
                segment_cursor = segment_start + len(segment)
                results.extend(
                    self._analyze_segment(
                        segment=segment,
                        segment_offset=line_offset + segment_start,
                        entities=entities,
                    )
                )

            line_offset += len(raw_line)

        return results

    def _analyze_segment(
        self,
        *,
        segment: str,
        segment_offset: int,
        entities: Optional[List[str]],
    ) -> List[RecognizerResult]:
        results: List[RecognizerResult] = []

        for spec in self._compiled_specs:
            entity_type = spec["entity_type"]
            if entities and entity_type not in entities:
                continue

            match = spec["regex"].search(segment)
            if match is None:
                continue
            if not self._is_standalone_label_match(segment, match.start("label")):
                continue

            raw_value = self._clean_value(entity_type, match.group("value"))
            if not raw_value or not self._looks_like_entity(entity_type, raw_value):
                continue

            local_start = segment.find(raw_value, match.start("value"))
            if local_start == -1:
                continue

            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=segment_offset + local_start,
                    end=segment_offset + local_start + len(raw_value),
                    score=spec["score"],
                    text=raw_value,
                    source=self.name,
                    metadata={
                        "label": match.group("label"),
                        "identifier_kind": self._resolve_identifier_kind(
                            match.group("label"),
                            raw_value,
                        ),
                    }
                    if entity_type == "CONTRACT_NO"
                    else {"label": match.group("label")},
                )
            )

        results.extend(
            self._extract_inline_labelled_entities(
                segment=segment,
                segment_offset=segment_offset,
                entities=entities,
            )
        )
        return results

    def _extract_inline_labelled_entities(
        self,
        *,
        segment: str,
        segment_offset: int,
        entities: Optional[List[str]],
    ) -> List[RecognizerResult]:
        results: List[RecognizerResult] = []

        if not entities or "PERSON" in entities:
            for match in self.INLINE_PERSON_LABEL_RE.finditer(segment):
                value = match.group("value").strip()
                if not self._looks_like_entity("PERSON", value):
                    continue
                results.append(
                    RecognizerResult(
                        entity_type="PERSON",
                        start=segment_offset + match.start("value"),
                        end=segment_offset + match.end("value"),
                        score=0.89,
                        text=value,
                        source=self.name,
                        metadata={"label": match.group("label"), "inline_label": True},
                    )
                )

        if not entities or "LOCATION" in entities:
            for match in self.INLINE_LOCATION_LABEL_RE.finditer(segment):
                raw_value = self._clean_value("LOCATION", match.group("value"))
                if not raw_value or not self._looks_like_entity("LOCATION", raw_value):
                    continue
                local_start = segment.find(raw_value, match.start("value"))
                if local_start == -1:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type="LOCATION",
                        start=segment_offset + local_start,
                        end=segment_offset + local_start + len(raw_value),
                        score=0.9,
                        text=raw_value,
                        source=self.name,
                        metadata={"label": match.group("label"), "inline_label": True},
                    )
                )

        return results

    def _clean_value(self, entity_type: str, value: str) -> str:
        cleaned = value.strip()
        cleaned = cleaned.split("\u3000")[0].strip()
        cleaned = self.TAIL_MARKERS.split(cleaned, maxsplit=1)[0].strip()
        if entity_type in {"ORGANIZATION", "ACCOUNT_NAME", "PROJECT"}:
            cleaned = re.split(
                r"\s*[（(]\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?[^\n\r）)]{1,24}[”\"'‘’]?\s*[）)]",
                cleaned,
                maxsplit=1,
            )[0].strip()
            cleaned = re.split(
                r"\s*，?\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?[^\n\r，,；;。]{1,24}[”\"'‘’]?",
                cleaned,
                maxsplit=1,
            )[0].strip()
        cleaned = cleaned.strip("\uff1a:;,.\uff0c\uff1b\u3002")
        cleaned = cleaned.replace("\uff08\u76d6\u7ae0\uff09", "").replace("(\u76d6\u7ae0)", "")
        cleaned = cleaned.strip()

        if entity_type == "CONTRACT_NO":
            case_number = extract_case_number(cleaned)
            if case_number:
                return case_number
            cleaned = re.split(r"[\uff0c,\uff1b;\u3002]", cleaned, maxsplit=1)[0].strip()
            cleaned = cleaned.rstrip("】】）)]").strip()
            cleaned = cleaned.strip()

        if entity_type == "CN_PHONE":
            phone_match = re.search(
                r"(1[3-9]\d(?:[ \t\u00a0\-－—–]?\d{4}){2}|(?:0\d{2,3}[-－—–]\d{7,8}|0\d{9,11}|400[-－—–]\d{3}[-－—–]\d{4}|400\d{7}))",
                cleaned,
            )
            return phone_match.group(1) if phone_match else ""

        if entity_type == "CN_BANK_CARD":
            account_match = re.search(r"(?<!\d)\d{16,20}(?!\d)", cleaned)
            return account_match.group(0) if account_match else ""

        if entity_type == "PERSON":
            person_match = re.search(r"[\u4e00-\u9fa5\u00b7]{2,8}", cleaned)
            return person_match.group(0) if person_match else ""

        if entity_type == "ACCOUNT_NAME" and re.fullmatch(r"\d{6,}", cleaned):
            return ""

        return cleaned

    def _looks_like_entity(self, entity_type: str, value: str) -> bool:
        if len(value) < 2:
            return False

        if entity_type == "ORGANIZATION":
            if any(punctuation in value for punctuation in [",", "\uff0c", "\u3002", ";", "\uff1b"]):
                return False
            tokens = [
                "\u516c\u53f8",
                "\u4e2d\u5fc3",
                "\u96c6\u56e2",
                "\u94f6\u884c",
                "\u5b66\u9662",
                "\u5c40",
                "\u9662",
            ]
            return any(token in value for token in tokens)

        if entity_type == "LOCATION":
            tokens = [
                "\u7701",
                "\u5e02",
                "\u533a",
                "\u53bf",
                "\u9547",
                "\u8857",
                "\u8def",
                "\u9053",
                "\u6751",
                "\u53f7",
            ]
            return any(token in value for token in tokens)

        if entity_type == "PROJECT":
            return len(value) >= 4 and not value.isdigit()

        if entity_type == "BANK_NAME":
            return "\u94f6\u884c" in value

        if entity_type == "ACCOUNT_NAME":
            return any("\u4e00" <= char <= "\u9fff" for char in value)

        if entity_type == "CN_BANK_CARD":
            return re.fullmatch(r"\d{16,20}", value) is not None

        if entity_type == "CONTRACT_NO":
            if len(value) < 4 or len(value) > 48:
                return False
            if any(punctuation in value for punctuation in [",", "\uff0c", "\u3002", ";", "\uff1b"]):
                return False
            if any(
                token in value
                for token in [
                    "\u4e0a\u8bc9\u4eba",
                    "\u88ab\u4e0a\u8bc9\u4eba",
                    "\u4eba\u6c11\u6cd5\u9662",
                    "\u6c11\u4e8b\u5224\u51b3\u4e66",
                    "\u7279\u5411\u8d35\u9662",
                    "\u4e70\u5356\u5408\u540c\u7ea0\u7eb7",
                ]
            ):
                return False
            if label_matches(value, CASE_NUMBER_LABELS):
                return False
            if extract_case_number(value):
                return True
            return re.fullmatch(r"[A-Za-z0-9\u4e00-\u9fa5\-_/.()\uff08\uff09\[\]\u3010\u3011]{4,40}", value) is not None

        return True

    def _is_standalone_label_match(self, segment: str, label_start: int) -> bool:
        prefix = segment[:label_start]
        normalized_prefix = prefix.strip()
        if not normalized_prefix:
            return True
        return self.STANDALONE_LABEL_PREFIX.fullmatch(normalized_prefix) is not None

    @staticmethod
    def _resolve_identifier_kind(label: str, value: str) -> str:
        return resolve_identifier_kind(value=value, label=label)
