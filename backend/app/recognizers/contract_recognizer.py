"""Contract-specific recognizer for common labelled fields."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.core.identifier_rules import build_spaced_label_pattern, resolve_identifier_kind
from app.core.recognizer_base import BaseRecognizer, RecognizerResult


class ContractFieldRecognizer(BaseRecognizer):
    """Extract entities from common contract labels."""

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
            ],
            "score": 0.86,
        },
    ]

    TAIL_MARKERS = re.compile(
        r"(?=(?:"
        r"\u7532\u65b9|\u4e59\u65b9|\u59d4\u6258\u65b9|\u53d7\u6258\u65b9|"
        r"\u6cd5\u5b9a\u4ee3\u8868\u4eba|\u8054\u7cfb\u4eba|\u8054\u7cfb\u7535\u8bdd|"
        r"\u5730\u5740|\u5f00\u6237\u884c|\u8d26\u53f7|\u8d26\u6237|\u6237\u540d"
        r")\s*[:\uff1a])"
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

        return results

    def _clean_value(self, entity_type: str, value: str) -> str:
        cleaned = value.strip()
        cleaned = cleaned.split("\u3000")[0].strip()
        cleaned = self.TAIL_MARKERS.split(cleaned, maxsplit=1)[0].strip()
        cleaned = cleaned.strip("\uff1a:;,.\uff0c\uff1b\u3002")
        cleaned = cleaned.replace("\uff08\u76d6\u7ae0\uff09", "").replace("(\u76d6\u7ae0)", "")
        cleaned = cleaned.strip()

        if entity_type == "CN_PHONE":
            phone_match = re.search(
                r"(1[3-9]\d{9}|(?:0\d{2,3}[-－—–]\d{7,8}|0\d{9,11}|400[-－—–]\d{3}[-－—–]\d{4}|400\d{7}))",
                cleaned,
            )
            return phone_match.group(1) if phone_match else ""

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

        if entity_type == "CONTRACT_NO":
            return len(value) >= 4

        return True

    @staticmethod
    def _resolve_identifier_kind(label: str, value: str) -> str:
        return resolve_identifier_kind(value=value, label=label)
