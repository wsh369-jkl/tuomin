"""Recall-first entity merge and alias propagation."""

from __future__ import annotations

import re
from typing import Iterable, List

from app.core.recognizer_base import RecognizerResult
from app.rules.default_subject_policy import DEFAULT_SUBJECT_TYPES, canonicalize_default_result
from app.services.lowmem_entity_utils import (
    ORG_PATTERN,
    P0_ENTITY_TYPES,
    iter_exact_matches,
    is_org_like_text,
    looks_like_organization_short_name,
)


class RecallFirstEntityMergeService:
    """Merge local recognizer candidates without dropping low-confidence findings."""

    SOURCE_PRIORITY = {
        "regex": 100,
        "rule_format": 100,
        "rule_official_institution": 99,
        "rule_organization": 92,
        "rule_organization_context": 94,
        "rule_person": 92,
        "rule_address": 92,
        "rule_alias": 92,
        "rule_docx_structure": 94,
        "custom": 95,
        "contract_structure_backfill": 90,
        "contract": 85,
        "docx_structure_uie": 96,
        "docx_structure_ner": 94,
        "alias_backscan_review": 97,
        "propagate": 96,
        "review_deterministic_decision": 96,
        "qwen_entity_decision": 95,
        "qwen_fragment_review": 78,
        "uie": 70,
        "ner": 65,
        "secondary_ner": 62,
        "alias_propagation": 58,
    }

    HIGH_RISK_TYPES = {
        "CN_ID_CARD",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
        "PERSON",
        "PERSON_NAME",
        "ORGANIZATION",
        "COMPANY_NAME",
        "ACCOUNT_NAME",
        "BANK_NAME",
        "ADDRESS",
        "LOCATION",
        "GOVERNMENT",
        "COURT",
        "CONTRACT_NO",
        "CASE_NO",
        "ALIAS",
    }

    def merge(self, results: Iterable[RecognizerResult]) -> List[RecognizerResult]:
        merged: list[RecognizerResult] = []
        canonical_results = [
            updated
            for result in results
            for updated in [canonicalize_default_result(result)]
            if updated is not None
        ]
        for result in sorted(canonical_results, key=self._sort_key):
            overlap_index = self._find_overlap(result, merged)
            if overlap_index is None:
                merged.append(result)
                continue
            existing = merged[overlap_index]
            replacement = self._choose(existing, result)
            if replacement is not existing:
                merged[overlap_index] = replacement
            elif self._same_span(existing, result) and existing.entity_type != result.entity_type:
                metadata = dict(existing.metadata or {})
                candidates = list(metadata.get("candidate_types") or [])
                if result.entity_type not in candidates:
                    candidates.append(result.entity_type)
                metadata["candidate_types"] = candidates
                metadata["requires_manual_review"] = True
                merged[overlap_index] = RecognizerResult(
                    entity_type=existing.entity_type,
                    start=existing.start,
                    end=existing.end,
                    score=existing.score,
                    text=existing.text,
                    source=existing.source,
                    metadata=metadata,
                )
        merged.sort(key=lambda item: (item.start, item.end))
        return merged

    def propagate_aliases(self, text: str, results: Iterable[RecognizerResult]) -> List[RecognizerResult]:
        expanded = list(results)
        seen = {(item.entity_type, item.start, item.end, item.text) for item in expanded}
        aliases = [
            item
            for item in expanded
            if item.entity_type in {"ALIAS", "ORGANIZATION", "COMPANY_NAME"}
            and len(item.text.strip()) >= 2
            and (
                item.entity_type == "ALIAS"
                or (item.metadata or {}).get("definition_alias")
                or (item.metadata or {}).get("alias_surface")
            )
        ]
        for alias in aliases:
            alias_text = str(
                (alias.metadata or {}).get("alias_surface")
                or ((alias.metadata or {}).get("definition_alias") if isinstance((alias.metadata or {}).get("definition_alias"), str) else "")
                or alias.text
            ).strip()
            canonical = str(
                (alias.metadata or {}).get("canonical")
                or (alias.metadata or {}).get("definition_full_text")
                or ""
            ).strip()
            if alias_text:
                output_type = "GOVERNMENT" if "法院" in canonical else "ORGANIZATION"
                for start, end, matched_text in iter_exact_matches(text, alias_text):
                    key = (output_type, start, end, matched_text)
                    if key in seen:
                        continue
                    if self._span_overlaps(start, end, expanded):
                        continue
                    expanded.append(
                        RecognizerResult(
                            entity_type=output_type,
                            start=start,
                            end=end,
                            score=max(0.78, float(alias.score) - 0.06),
                            text=matched_text,
                            source="alias_propagation",
                            metadata={
                                "source": "alias_propagation",
                                "canonical": canonical or None,
                                "definition_full_text": canonical or None,
                                "definition_alias": alias_text,
                                "alias_surface": alias_text,
                                "requires_manual_review": not bool(canonical),
                            },
                        )
                    )
                    seen.add(key)
            if canonical:
                for start, end, matched_text in iter_exact_matches(text, canonical):
                    output_type = "GOVERNMENT" if "法院" in matched_text else "ORGANIZATION"
                    key = (output_type, start, end, matched_text)
                    if key in seen or self._span_overlaps(start, end, expanded):
                        continue
                    expanded.append(
                        RecognizerResult(
                            entity_type=output_type,
                            start=start,
                            end=end,
                            score=0.88,
                            text=matched_text,
                            source="alias_propagation",
                            metadata={
                                "source": "alias_propagation",
                                "alias": alias_text or alias.text,
                                "canonical": canonical,
                                "definition_full_text": canonical,
                                "definition_alias": alias_text or None,
                            },
                        )
                    )
                    seen.add(key)
        return self.merge(expanded)

    def _sort_key(self, item: RecognizerResult) -> tuple[int, int, float, int, int]:
        return (
            -self.SOURCE_PRIORITY.get(item.source, 1),
            0 if item.entity_type in P0_ENTITY_TYPES else 1,
            -float(item.score),
            -(item.end - item.start),
            item.start,
        )

    @staticmethod
    def _same_span(left: RecognizerResult, right: RecognizerResult) -> bool:
        return left.start == right.start and left.end == right.end

    def _choose(self, existing: RecognizerResult, candidate: RecognizerResult) -> RecognizerResult:
        if existing.entity_type in P0_ENTITY_TYPES and candidate.entity_type not in P0_ENTITY_TYPES:
            return existing
        if candidate.entity_type in P0_ENTITY_TYPES and existing.entity_type not in P0_ENTITY_TYPES:
            return candidate

        same_type = existing.entity_type == candidate.entity_type
        existing_priority = self.SOURCE_PRIORITY.get(existing.source, 1)
        candidate_priority = self.SOURCE_PRIORITY.get(candidate.source, 1)
        if same_type:
            preferred = self._choose_same_type_overlap(
                existing,
                candidate,
                existing_priority=existing_priority,
                candidate_priority=candidate_priority,
            )
            if preferred is not None:
                return preferred

        if self._same_span(existing, candidate) and not same_type:
            if candidate_priority > existing_priority:
                return candidate
            return existing

        subject_location_preference = self._prefer_complete_org_over_embedded_location(existing, candidate)
        if subject_location_preference is not None:
            return subject_location_preference

        existing_len = existing.end - existing.start
        candidate_len = candidate.end - candidate.start
        if (
            candidate.entity_type in self.HIGH_RISK_TYPES
            and candidate_len >= existing_len + 3
        ):
            if self._text_noise_score(candidate) <= self._text_noise_score(existing) and candidate_priority > existing_priority + 10:
                return candidate
            return existing

        if candidate_priority > existing_priority + 10:
            return candidate

        if self._same_span(existing, candidate) and float(candidate.score) > float(existing.score):
            return candidate

        return existing

    @staticmethod
    def _prefer_complete_org_over_embedded_location(
        existing: RecognizerResult,
        candidate: RecognizerResult,
    ) -> RecognizerResult | None:
        pairs = ((existing, candidate), (candidate, existing))
        for org, location in pairs:
            if org.entity_type not in {"ORGANIZATION", "GOVERNMENT"} or location.entity_type != "LOCATION":
                continue
            if org.start <= location.start and org.end >= location.end and is_org_like_text(org.text):
                metadata = dict(org.metadata or {})
                metadata["embedded_location_absorbed"] = location.text
                return RecognizerResult(
                    entity_type=org.entity_type,
                    start=org.start,
                    end=org.end,
                    score=max(float(org.score or 0.0), float(location.score or 0.0), 0.86),
                    text=org.text,
                    source=org.source,
                    metadata=metadata,
                )
        return None

    def _choose_same_type_overlap(
        self,
        existing: RecognizerResult,
        candidate: RecognizerResult,
        *,
        existing_priority: int,
        candidate_priority: int,
    ) -> RecognizerResult | None:
        existing_noise = self._text_noise_score(existing)
        candidate_noise = self._text_noise_score(candidate)
        if existing_noise != candidate_noise:
            if candidate_noise < existing_noise and candidate_priority >= existing_priority - 5:
                return candidate
            return existing

        existing_strength = self._text_strength_score(existing)
        candidate_strength = self._text_strength_score(candidate)
        if existing_strength != candidate_strength:
            if candidate_strength > existing_strength and candidate_priority >= existing_priority - 5:
                return candidate
            return existing

        if candidate_priority != existing_priority:
            if candidate_priority > existing_priority:
                return candidate
            return existing

        existing_len = existing.end - existing.start
        candidate_len = candidate.end - candidate.start
        if candidate_len != existing_len:
            if candidate_len > existing_len and not self._looks_like_suspicious_extension(existing.text, candidate.text):
                return candidate
            return existing

        if float(candidate.score) > float(existing.score):
            return candidate
        return existing

    @staticmethod
    def _looks_like_suspicious_extension(existing_text: str, candidate_text: str) -> bool:
        if candidate_text.startswith(existing_text):
            suffix = candidate_text[len(existing_text) :]
        elif candidate_text.endswith(existing_text):
            suffix = candidate_text[: len(candidate_text) - len(existing_text)]
        else:
            return False
        normalized = re.sub(r"\s+", "", suffix or "")
        if not normalized:
            return False
        if len(normalized) <= 3 and any(token in normalized for token in ("系", "为", "与", "和", "及", "向", "对", "由", "的", "并")):
            return True
        return any(
            token in normalized
            for token in (
                "统一社会信用代码",
                "法定代表人",
                "法定代理人",
                "负责人",
                "联系人",
                "开户行",
                "账号",
                "账户",
                "供货方",
                "采购方",
                "履约",
                "结算",
                "付款",
                "收款",
                "交付",
                "签订",
                "签署",
                "承担",
                "继续",
                "本合同",
                "本协议",
            )
        )

    @staticmethod
    def _text_noise_score(result: RecognizerResult) -> int:
        text = str(result.text or "")
        normalized = re.sub(r"\s+", "", text)
        score = 0
        if ":" in text or "：" in text:
            score += 3
        for token in (
            "统一社会信用代码",
            "社会信用代码",
            "信用代码",
            "法定代表人",
            "法定代理人",
            "负责人",
            "联系人",
            "开户行",
            "账号",
            "账户",
            "供货方",
            "采购方",
            "履约",
            "结算",
            "付款",
            "收款",
            "交付",
            "签订",
            "签署",
            "承担",
            "继续",
            "本合同",
            "本协议",
        ):
            if token in normalized:
                score += 2
        if len(normalized) <= 3 and normalized.endswith(("系", "为")):
            score += 2
        if re.search(r"\d{6,}", normalized):
            score += 2
        return score

    @staticmethod
    def _text_strength_score(result: RecognizerResult) -> int:
        text = str(result.text or "")
        normalized = re.sub(r"\s+", "", text)
        score = 0
        if ORG_PATTERN.fullmatch(normalized):
            score += 3
        if looks_like_organization_short_name(normalized):
            score += 1
        if result.source == "contract_structure_backfill":
            score += 2
        if result.source == "qwen_fragment_review":
            score += 1
        return score

    @staticmethod
    def _find_overlap(result: RecognizerResult, existing_results: list[RecognizerResult]) -> int | None:
        for index, existing in enumerate(existing_results):
            if result.start < existing.end and result.end > existing.start:
                return index
        return None

    @staticmethod
    def _span_overlaps(start: int, end: int, existing_results: list[RecognizerResult]) -> bool:
        return any(start < item.end and end > item.start for item in existing_results)
