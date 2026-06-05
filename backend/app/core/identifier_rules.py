"""Shared identifier rules for contract and case numbers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


CONTRACT_IDENTIFIER_LABELS: tuple[str, ...] = (
    "合同编号",
    "合同号",
    "协议编号",
    "项目编号",
    "招标编号",
)

CASE_NUMBER_LABELS: tuple[str, ...] = (
    "案号",
    "案件编号",
    "受理案号",
    "执行案号",
    "原审案号",
    "一审案号",
    "二审案号",
    "再审案号",
    "上诉案号",
    "审理案号",
    "申请执行案号",
    "民事案号",
    "刑事案号",
    "行政案号",
    "仲裁案号",
)

ALL_IDENTIFIER_LABELS: tuple[str, ...] = CONTRACT_IDENTIFIER_LABELS + CASE_NUMBER_LABELS

_BRACKETED_YEAR = r"(?:[\uff08(\u3014\[\u3010\u3016]\s*\d{4}\s*[\uff09)\u3015\]\u3011\u3017])"
_PLAIN_YEAR = r"(?:20\d{2})"
_LEGAL_CASE_TOKEN = (
    r"(?:"
    r"民(?:初|终|再|申|辖终|辖初|保|特)"
    r"|刑(?:初|终|申)"
    r"|行(?:初|终|审|申)"
    r"|执(?:恢|保|异|监|行|前调)?"
    r"|知(?:民初|民终|行终|行初|民辖终)"
    r"|破(?:申|初)?"
    r"|清(?:申|算)?"
    r"|仲"
    r"|财保"
    r"|审监"
    r")"
)

CASE_NUMBER_VALUE_PATTERNS: tuple[str, ...] = (
    rf"{_BRACKETED_YEAR}\s*[A-Za-z\u4e00-\u9fa5]{{1,12}}\s*\d{{0,8}}\s*{_LEGAL_CASE_TOKEN}\s*\d{{1,8}}(?:\s*\u4e4b[\u4e00-\u9fa5\d]+)?\s*\u53f7?",
    rf"{_PLAIN_YEAR}\s*[A-Za-z\u4e00-\u9fa5]{{1,12}}\s*\d{{0,8}}\s*{_LEGAL_CASE_TOKEN}\s*\d{{1,8}}(?:\s*\u4e4b[\u4e00-\u9fa5\d]+)?\s*\u53f7?",
)

CASE_NUMBER_VALUE_PATTERN = "|".join(f"(?:{pattern})" for pattern in CASE_NUMBER_VALUE_PATTERNS)
_CASE_NUMBER_REGEXES = [re.compile(rf"^(?:{pattern})$") for pattern in CASE_NUMBER_VALUE_PATTERNS]
_CASE_NUMBER_SEARCH_REGEXES = [re.compile(pattern) for pattern in CASE_NUMBER_VALUE_PATTERNS]


def compact_text(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or ""))


@lru_cache(maxsize=32)
def _build_spaced_label_pattern_cached(labels: tuple[str, ...]) -> str:
    parts: list[str] = []
    for label in sorted({item for item in labels if item}, key=len, reverse=True):
        parts.append(r"\s*".join(re.escape(char) for char in label))
    return "|".join(f"(?:{part})" for part in parts)


def build_spaced_label_pattern(labels: Iterable[str]) -> str:
    normalized = tuple(str(label).strip() for label in labels if str(label).strip())
    return _build_spaced_label_pattern_cached(normalized)


def label_matches(label_text: str | None, labels: Iterable[str]) -> bool:
    normalized_label = compact_text(label_text)
    if not normalized_label:
        return False
    return any(compact_text(item) in normalized_label for item in labels if item)


def looks_like_case_number(value: str | None) -> bool:
    normalized = compact_text(value)
    if not normalized:
        return False
    return any(regex.fullmatch(normalized) for regex in _CASE_NUMBER_REGEXES)


def extract_case_number(value: str | None) -> str:
    candidate = str(value or "")
    if not candidate.strip():
        return ""

    compact_candidate = compact_text(candidate)
    for regex in _CASE_NUMBER_SEARCH_REGEXES:
        compact_match = regex.search(compact_candidate)
        if compact_match is None:
            continue
        matched_compact = compact_match.group(0)
        normalized = compact_text(matched_compact)
        if not normalized:
            continue

        start = 0
        compact_progress = 0
        for index, char in enumerate(candidate):
            if char.isspace():
                continue
            if compact_progress >= len(compact_candidate):
                break
            if compact_progress == compact_match.start():
                start = index
                break
            compact_progress += 1

        end = len(candidate)
        compact_progress = 0
        start_recorded = False
        for index, char in enumerate(candidate):
            if char.isspace():
                continue
            if compact_progress == compact_match.start() and not start_recorded:
                start = index
                start_recorded = True
            compact_progress += 1
            if compact_progress == compact_match.end():
                end = index + 1
                break

        return candidate[start:end].strip()

    return ""


def mask_case_number_digits_only(value: str | None) -> str:
    return "".join("*" if char.isdigit() else char for char in str(value or ""))


def resolve_identifier_kind(value: str | None, label: str | None = None) -> str:
    if label_matches(label, CASE_NUMBER_LABELS):
        return "case_no"
    if looks_like_case_number(value):
        return "case_no"
    return "contract_no"
