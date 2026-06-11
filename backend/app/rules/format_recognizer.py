"""Deterministic recognizer and validators for format-style entities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.identifier_rules import CASE_NUMBER_VALUE_PATTERN, looks_like_case_number
from app.core.recognizer_base import RecognizerResult


_ID_CARD = re.compile(r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d(?:[ \t\u00a0\-－—–]?\d{4}){2}(?!\d)")
_EMAIL = re.compile(r"(?<![\w.-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_CREDIT_CODE = re.compile(r"(?<![0-9A-Z])(?!\d{18}(?![0-9A-Z]))[0-9A-HJ-NPQRTUWXY]{18}(?![0-9A-Z])")
_BANK_CARD = re.compile(r"(?<!\d)\d(?:[ \t\u00a0\-－—–]?\d){15,19}(?!\d)")
_CASE_NO = re.compile(CASE_NUMBER_VALUE_PATTERN)
_CONTRACT_NO = re.compile(
    r"(?:合同编号|合同号|协议编号|项目编号|招标编号)\s*[:：]\s*"
    r"(?P<value>[A-Za-z0-9\u4e00-\u9fa5\-\[\]【】〔〕（）()/.]{4,80})"
)

_CREDIT_CODE_CHARS = "0123456789ABCDEFGHJKLMNPQRTUWXY"
_CREDIT_CODE_WEIGHTS = (1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28)
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_CODES = "10X98765432"


@dataclass(frozen=True)
class FormatValidation:
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    confidence: float


class FormatRecognizer:
    """Recognize and validate stable structured identifiers without models."""

    def recognize(self, text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        for entity_type, regex, score in (
            ("CN_ID_CARD", _ID_CARD, 0.98),
            ("CN_PHONE", _PHONE, 0.95),
            ("EMAIL_ADDRESS", _EMAIL, 0.96),
            ("CN_CREDIT_CODE", _CREDIT_CODE, 0.97),
            ("CN_BANK_CARD", _BANK_CARD, 0.9),
            ("CASE_NO", _CASE_NO, 0.93),
        ):
            for match in regex.finditer(text or ""):
                value = match.group(0).strip()
                if not value:
                    continue
                validation = self.validate(entity_type, value)
                if validation.failed and entity_type in {"CN_ID_CARD", "CN_CREDIT_CODE"}:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=max(score, validation.confidence),
                        text=value,
                        source="rule_format",
                        metadata={
                            "source_layer": "regex",
                            "rule_recognizer": "format",
                            "validators_passed": list(validation.passed),
                            "validators_failed": list(validation.failed),
                        },
                    )
                )
        for match in _CONTRACT_NO.finditer(text or ""):
            value = (match.group("value") or "").strip()
            start, end = match.span("value")
            if value:
                results.append(
                    RecognizerResult(
                        entity_type="CONTRACT_NO",
                        start=start,
                        end=end,
                        score=0.92,
                        text=value,
                        source="rule_format",
                        metadata={"source_layer": "regex", "rule_recognizer": "format", "label": "合同编号"},
                    )
                )
        return results

    def validate(self, entity_type: str, value: str) -> FormatValidation:
        normalized = self._normalize_identifier(value)
        passed: list[str] = []
        failed: list[str] = []
        confidence = 0.82
        if entity_type == "CN_ID_CARD":
            if self._valid_id_card(normalized):
                passed.append("cn_id_checksum")
                confidence = 0.99
            else:
                failed.append("cn_id_checksum")
                confidence = 0.2
        elif entity_type == "CN_CREDIT_CODE":
            if self._valid_credit_code(normalized):
                passed.append("credit_code_checksum")
                confidence = 0.98
            else:
                failed.append("credit_code_checksum")
                confidence = 0.25
        elif entity_type == "CN_PHONE":
            if re.fullmatch(r"1[3-9]\d{9}", normalized):
                passed.append("cn_mobile_shape")
                confidence = 0.96
            else:
                failed.append("cn_mobile_shape")
        elif entity_type == "CN_BANK_CARD":
            if 16 <= len(normalized) <= 20 and normalized.isdigit() and len(set(normalized)) > 1:
                passed.append("bank_card_shape")
                if self._valid_luhn(normalized):
                    passed.append("luhn")
                    confidence = 0.95
                else:
                    confidence = 0.88
            else:
                failed.append("bank_card_shape")
                confidence = 0.35
        elif entity_type == "EMAIL_ADDRESS":
            if _EMAIL.fullmatch(value.strip()):
                passed.append("email_shape")
                confidence = 0.96
            else:
                failed.append("email_shape")
        elif entity_type == "CASE_NO":
            if looks_like_case_number(value):
                passed.append("case_no_shape")
                confidence = 0.95
            else:
                failed.append("case_no_shape")
        elif entity_type == "CONTRACT_NO":
            if 4 <= len(normalized) <= 80 and re.search(r"[A-Za-z0-9]", normalized):
                passed.append("contract_no_shape")
                confidence = 0.9
            else:
                failed.append("contract_no_shape")
        return FormatValidation(tuple(passed), tuple(failed), confidence)

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return re.sub(r"[\s\-－—–]", "", str(value or "")).upper()

    @staticmethod
    def _valid_id_card(value: str) -> bool:
        if not re.fullmatch(r"[1-9]\d{16}[\dX]", value):
            return False
        total = sum(int(value[index]) * _ID_WEIGHTS[index] for index in range(17))
        return _ID_CHECK_CODES[total % 11] == value[-1]

    @staticmethod
    def _valid_credit_code(value: str) -> bool:
        if not re.fullmatch(r"[0-9A-HJ-NPQRTUWXY]{18}", value):
            return False
        total = 0
        for index, char in enumerate(value[:17]):
            total += _CREDIT_CODE_CHARS.index(char) * _CREDIT_CODE_WEIGHTS[index]
        check_index = (31 - total % 31) % 31
        return _CREDIT_CODE_CHARS[check_index] == value[-1]

    @staticmethod
    def _valid_luhn(value: str) -> bool:
        total = 0
        reverse_digits = [int(char) for char in reversed(value)]
        for index, digit in enumerate(reverse_digits):
            if index % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0
