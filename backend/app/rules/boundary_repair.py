"""Boundary repair for rule-first candidates."""

from __future__ import annotations

import re

from app.core.recognizer_base import RecognizerResult
from app.rules.subject_admission_gate import SubjectAdmissionGate
from app.services.contract_structure_backfill_service import ContractStructureBackfillService
from app.services.lowmem_entity_utils import (
    OFFICIAL_INSTITUTION_PATTERN,
    ORG_PATTERN,
    expand_org_span_with_left_region_prefix,
    expand_subject_span_to_containing_shape,
    find_short_org_prefix_before_non_subject_boundary,
    is_generic_organization_term,
    is_official_institution_text,
    is_probable_person,
    is_org_like_text,
    looks_like_organization_short_name,
    looks_like_natural_person_name,
    normalize_entity_text,
    strip_leading_subject_function_words,
)


_LEADING_LABEL = re.compile(
    r"^(?:甲方|乙方|丙方|委托方|受托方|发包人|承包人|采购人|供应商|申请人|被申请人|原告|被告|第三人|"
    r"法定代表人|负责人|联系人|户名|账户名称|开户名|开户行|地址|住所|项目地址|工程地址)\s*[:：]?\s*"
)
_LEADING_NOISE = re.compile(
    r"^(?:由|并由|后续由|随后由|另由|另行由|同时由|共同由|分别由|"
    r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令|"
    r"向|对|与|和|及|以及|被|在|经|为|由其|由该|由本)\s*"
)
_SUBJECT_PREFIX_SPLIT = re.compile(
    r"(?:^|[，,；;。、\n\r]|(?:由|并由|后续由|随后由|另由|另行由|同时由|共同由|分别由|"
    r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令|向|对|与|和|及|以及))\s*"
    r"(?P<subject>[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,48}?"
    r"(?:(?:股份有限公司|有限责任公司|集团有限公司|有限公司)"
    r"(?:[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{1,16}分公司)?|"
    r"集团|分公司|子公司|"
    r"银行(?:股份有限公司)?(?:[\u4e00-\u9fa5A-Za-z0-9]{0,12}(?:支行|分行))?|"
    r"人民法院|中级人民法院|高级人民法院|仲裁委员会|人民检察院|公安局|派出所|"
    r"律师事务所|会计师事务所|研究院|研究所|服务中心|技术中心|管理委员会|委员会|公司))"
)
_COMPANY_SUFFIX_SHAPE = re.compile(
    r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司)"
    r"(?:[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{1,16}分公司)?|"
    r"集团|分公司|子公司|公司"
)
_ORG_SUFFIX_ANCHOR_SHAPE = re.compile(
    r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司)"
    r"(?:[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{1,16}分公司)?|"
    r"集团|分公司|子公司|公司|"
    r"人民法院|中级人民法院|高级人民法院|仲裁委员会|人民检察院|公安局|派出所|"
    r"律师事务所|会计师事务所|研究院|研究所|服务中心|技术中心|管理委员会|委员会|"
    r"银行(?:股份有限公司)?(?:[\u4e00-\u9fa5A-Za-z0-9]{0,12}(?:支行|分行|营业部|分理处))?"
)
_ORG_LEFT_BOUNDARY_CUE = re.compile(
    r"(?:"
    r"甲方|乙方|丙方|委托方|受托方|发包人|承包人|采购人|供应商|申请人|被申请人|"
    r"原告|被告|第三人|上诉人|被上诉人|"
    r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令|"
    r"由|并由|后续由|随后由|另由|另行由|同时由|共同由|分别由|"
    r"向|对|与|和|及|以及|同|"
    r"约定|确定|指定|委托|授权|委派|变更为|名称为|主体为|账户名为|户名为|"
    r"系|为|是|即|包括|涉及|不服"
    r")\s*[:：]?\s*$"
)
_ORG_LEFT_BOUNDARY_CUE_ANYWHERE = re.compile(
    r"(?:"
    r"甲方|乙方|丙方|委托方|受托方|发包人|承包人|采购人|供应商|申请人|被申请人|"
    r"原告|被告|第三人|上诉人|被上诉人|"
    r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令|"
    r"由|并由|后续由|随后由|另由|另行由|同时由|共同由|分别由|"
    r"向|对|与|和|及|以及|同|"
    r"合同约定|协议约定|约定|确定|指定|委托|授权|委派|变更为|名称为|主体为|账户名为|户名为|"
    r"主体|名称|户名|系|为|是|即|包括|涉及|不服"
    r")\s*[:：]?"
)
_ORG_LEFT_POLLUTION_TERMS = (
    "甲方",
    "乙方",
    "丙方",
    "委托方",
    "受托方",
    "发包人",
    "承包人",
    "采购人",
    "供应商",
    "申请人",
    "被申请人",
    "原告",
    "被告",
    "第三人",
    "上诉人",
    "被上诉人",
    "合同",
    "协议",
    "主体",
    "名称",
    "户名",
    "条款",
    "约定",
    "确定",
    "指定",
    "委托",
    "授权",
    "委派",
    "通知",
    "要求",
    "说明",
    "确认",
    "请求",
    "判令",
    "通过",
    "根据",
    "依据",
    "按照",
    "经由",
    "经过",
    "负责",
    "继续",
    "履约",
    "履行",
    "结算",
    "付款",
    "收款",
    "交付",
    "承担",
)
_COMMON_REGION_BODY = (
    r"中华人民共和国|国家|中国|全国|中央|最高|"
    r"北京|上海|天津|重庆|广州|深圳|杭州|南京|苏州|成都|武汉|西安|长沙|郑州|青岛|宁波|佛山|东莞|厦门|福州|济南|合肥|"
    r"[\u4e00-\u9fa5]{2,16}(?:省|自治区|特别行政区|市|地区|自治州|盟|区|县|旗|自治县|自治旗)"
)
_COMMON_REGION_PREFIX = re.compile(rf"^(?:{_COMMON_REGION_BODY})")
_COMMON_REGION_ANYWHERE = re.compile(rf"(?:{_COMMON_REGION_BODY})")
_TRAILING_PUNCT = " \t\r\n:：,，;；。.!！?？、（）()《》【】"


def _field_label_tail_pattern() -> re.Pattern[str]:
    labels = sorted(
        {str(label or "").strip() for label in ContractStructureBackfillService.FOLLOWING_FIELD_LABELS if str(label or "").strip()},
        key=len,
        reverse=True,
    )
    if not labels:
        return re.compile(r"a^")
    label_pattern = "|".join(re.escape(label) for label in labels)
    return re.compile(rf"(?:\s|^)(?:{label_pattern})\s*[:：]?.*$")


_FIELD_LABEL_TAIL = _field_label_tail_pattern()
_PERSON_ORG_BRIDGE_VERB_PATTERN = re.compile(
    r"(?:担任|任职|就职|供职|出任|为|是|系|兼任|兼为|任|在|加入|服务于)"
)


class BoundaryRepair:
    def repair(self, text: str, result: RecognizerResult) -> list[tuple[RecognizerResult, list[str]]]:
        narrative_split_items = self._split_narrative_subjects(text, result)
        if narrative_split_items:
            repaired_items: list[tuple[RecognizerResult, list[str]]] = []
            for split_result, split_repairs in narrative_split_items:
                repaired, repairs = self._trim_single(text, split_result)
                repaired_items.append((repaired, [*split_repairs, *repairs]))
            return repaired_items

        base = self._trim_single(text, result)
        if self._needs_split(base):
            split_items = self._split_parallel(text, base)
            if split_items:
                return split_items
        return [base]

    def _trim_single(self, text: str, result: RecognizerResult) -> tuple[RecognizerResult, list[str]]:
        start = int(result.start)
        end = int(result.end)
        expanded_span = expand_subject_span_to_containing_shape(text, start, end, result.entity_type)
        if expanded_span is None and str(result.entity_type or "").upper() in {"ORGANIZATION", "COMPANY_NAME", "GOVERNMENT", "COURT"}:
            expanded_span = expand_org_span_with_left_region_prefix(text, start, end)
        expanded_from = None
        if expanded_span is not None:
            expanded_start, expanded_end = expanded_span
            if expanded_start != start or expanded_end != end:
                expanded_from = {"start": start, "end": end, "text": result.text}
                start, end = expanded_start, expanded_end
        value = text[start:end] if 0 <= start < end <= len(text) else str(result.text or "")
        repairs: list[str] = []
        if expanded_from is not None:
            repairs.append("expand_to_full_subject_shape")
        local_start = 0
        local_end = len(value)

        leading = _LEADING_LABEL.match(value)
        if leading:
            local_start = leading.end()
            repairs.append("strip_field_label")

        while local_start < local_end and value[local_start] in _TRAILING_PUNCT:
            local_start += 1
            repairs.append("strip_leading_punctuation")
        while local_end > local_start and value[local_end - 1] in _TRAILING_PUNCT:
            local_end -= 1
            repairs.append("strip_trailing_punctuation")

        inner = value[local_start:local_end]
        leading_noise = _LEADING_NOISE.match(inner)
        if leading_noise:
            local_start += leading_noise.end()
            repairs.append("strip_leading_noise")
            inner = value[local_start:local_end]
            leading = _LEADING_LABEL.match(inner)
            if leading:
                local_start += leading.end()
                repairs.append("strip_field_label")
                inner = value[local_start:local_end]

        if result.entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "ALIAS", "GOVERNMENT", "COURT", "GOVERNMENT_AGENCY", "BANK_NAME"}:
            cleaned_inner, consumed = strip_leading_subject_function_words(inner)
            if consumed > 0:
                cleaned_normalized = normalize_entity_text(cleaned_inner)
                gate_type = (
                    "GOVERNMENT"
                    if is_official_institution_text(cleaned_normalized)
                    else "ORGANIZATION"
                )
                has_boundary_anchor = (
                    OFFICIAL_INSTITUTION_PATTERN.search(cleaned_normalized)
                    if gate_type == "GOVERNMENT"
                    else (
                        ORG_PATTERN.search(cleaned_normalized)
                        or is_org_like_text(cleaned_inner)
                        or looks_like_organization_short_name(cleaned_normalized)
                    )
                )
                if cleaned_normalized and has_boundary_anchor:
                    local_start += consumed
                    inner = cleaned_inner
                    repairs.append("strip_leading_function_words")
                else:
                    repairs.append("leading_function_word_not_boundary_safe")
        if inner:
            org_match = self._best_organization_span(inner)
        else:
            org_match = None
        if org_match and (org_match[0] > 0 or org_match[1] < len(inner)):
            local_start += org_match[0]
            local_end = local_start + (org_match[1] - org_match[0])
            repairs.append("trim_to_organization_shape")
        else:
            org_start, org_end, org_repairs = self._trim_short_org_like_inner(inner, text)
            if org_repairs:
                local_start += org_start
                local_end = local_start + (org_end - org_start)
                repairs.extend(org_repairs)

        if result.entity_type in {"PERSON", "PERSON_NAME"}:
            person_start, person_end, person_repairs = self._trim_person_inner(value[local_start:local_end])
            if person_repairs:
                local_start += person_start
                local_end = local_start + (person_end - person_start)
                repairs.extend(person_repairs)

        new_start = start + local_start
        new_end = start + local_end
        if new_start < 0 or new_end <= new_start or new_end > len(text):
            return result, repairs
        new_text = text[new_start:new_end]
        normalized_new_text = normalize_entity_text(new_text)
        if (
            result.entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME", "ALIAS", "GOVERNMENT", "COURT", "GOVERNMENT_AGENCY", "BANK_NAME"}
            and any(repair in repairs for repair in ("strip_leading_noise", "strip_leading_function_words"))
            and ORG_PATTERN.search(normalized_new_text)
            and SubjectAdmissionGate.is_weak_function_stripped_subject(new_text)
        ):
            repairs.append("weak_function_stripped_org_boundary_risk")
        if new_text == result.text and new_start == result.start and new_end == result.end:
            return result, repairs
        metadata = dict(result.metadata or {})
        metadata["boundary_repaired_from"] = {"start": result.start, "end": result.end, "text": result.text}
        metadata["normalized_text"] = normalized_new_text
        if expanded_from is not None:
            metadata["boundary_expanded_from"] = expanded_from
        if result.entity_type in {"ORGANIZATION", "COMPANY_NAME"} and looks_like_organization_short_name(normalized_new_text):
            metadata["short_org_candidate"] = True
            metadata["requires_manual_review"] = True
            metadata["identity_surface"] = normalized_new_text
        return (
            RecognizerResult(
                entity_type=result.entity_type,
                start=new_start,
                end=new_end,
                score=max(float(result.score or 0.0), 0.82),
                text=new_text,
                source=result.source,
                metadata=metadata,
            ),
            repairs,
        )

    @staticmethod
    def _best_organization_span(value: str) -> tuple[int, int] | None:
        spans: list[tuple[int, int]] = []
        suffix_span = BoundaryRepair._best_org_suffix_anchored_span(value)
        if suffix_span:
            spans.append(suffix_span)
        for match in _SUBJECT_PREFIX_SPLIT.finditer(value or ""):
            spans.append(match.span("subject"))
        if not spans:
            spans.extend(match.span() for match in OFFICIAL_INSTITUTION_PATTERN.finditer(value or ""))
        if not spans:
            spans.extend(match.span() for match in ORG_PATTERN.finditer(value or ""))
        if not spans:
            return None

        def score(span: tuple[int, int]) -> tuple[int, int, int]:
            start, end = span
            candidate = value[start:end]
            normalized = normalize_entity_text(candidate)
            _, consumed = strip_leading_subject_function_words(candidate)
            prefix_penalty = 3 if consumed > 0 else 0
            role_penalty = 1 if re.match(r"^(?:甲方|乙方|丙方|原告|被告|申请人|被申请人)", normalized) else 0
            anchored_bonus = 2 if _ORG_SUFFIX_ANCHOR_SHAPE.search(normalized) else 0
            region_bonus = 1 if BoundaryRepair._starts_with_region_prefix(normalized) else 0
            pollution_penalty = BoundaryRepair._left_pollution_penalty(normalized)
            return (
                anchored_bonus + region_bonus - prefix_penalty - role_penalty - pollution_penalty,
                -len(normalized),
                -start,
            )

        return max(spans, key=score)

    @staticmethod
    def _best_org_suffix_anchored_span(value: str) -> tuple[int, int] | None:
        if not value:
            return None
        candidates: list[tuple[tuple[int, int, int, int, int, int], tuple[int, int]]] = []
        for suffix_match in _ORG_SUFFIX_ANCHOR_SHAPE.finditer(value):
            end = suffix_match.end()
            min_start = max(0, end - 48)
            left_window = value[min_start:end]
            starts = {min_start}
            for index, char in enumerate(left_window, start=min_start):
                if char in " \t\r\n:：,，;；。.!！?？、（）()《》【】":
                    starts.add(index + 1)
            for cue_match in _ORG_LEFT_BOUNDARY_CUE_ANYWHERE.finditer(left_window):
                cue_end = min_start + cue_match.end()
                if cue_end <= end - 2:
                    starts.add(cue_end)
            for region_start in BoundaryRepair._region_starts(left_window):
                region_start = min_start + region_start
                if region_start <= end - 2:
                    starts.add(region_start)
            for start in starts:
                if start < min_start or start >= end - 1:
                    continue
                candidate = value[start:end].strip(_TRAILING_PUNCT)
                if not candidate:
                    continue
                adjusted_start = start + (len(value[start:end]) - len(value[start:end].lstrip(_TRAILING_PUNCT)))
                normalized = normalize_entity_text(candidate)
                if not normalized:
                    continue
                if SubjectAdmissionGate.is_action_or_function_text(normalized):
                    continue
                if not (
                    ORG_PATTERN.fullmatch(normalized)
                    or ORG_PATTERN.search(normalized)
                    or OFFICIAL_INSTITUTION_PATTERN.fullmatch(normalized)
                    or OFFICIAL_INSTITUTION_PATTERN.search(normalized)
                    or is_org_like_text(candidate)
                ):
                    continue
                gate_type = "GOVERNMENT" if is_official_institution_text(normalized) else "ORGANIZATION"
                passes_shape, _ = SubjectAdmissionGate.passes_subject_shape(
                    gate_type,
                    normalized,
                    allow_short_org=looks_like_organization_short_name(normalized),
                )
                left_context = value[max(0, adjusted_start - 12) : adjusted_start]
                boundary_bonus = 2 if adjusted_start == 0 or not left_context or _ORG_LEFT_BOUNDARY_CUE.search(left_context) else 0
                region_bonus = 3 if BoundaryRepair._starts_with_region_prefix(normalized) else 0
                pollution_penalty = BoundaryRepair._left_pollution_penalty(normalized)
                shape_penalty = 0 if passes_shape else 2
                weak_trim_penalty = 1 if SubjectAdmissionGate.is_weak_function_stripped_subject(candidate) else 0
                suffix_bonus = 2 if _ORG_SUFFIX_ANCHOR_SHAPE.search(normalized) else 1
                score = (
                    region_bonus + boundary_bonus + suffix_bonus - pollution_penalty - shape_penalty - weak_trim_penalty,
                    1 if pollution_penalty == 0 else 0,
                    1 if passes_shape else 0,
                    len(normalized),
                    -adjusted_start,
                    end,
                )
                candidates.append((score, (adjusted_start, end)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _starts_with_region_prefix(normalized: str) -> bool:
        return bool(_COMMON_REGION_PREFIX.match(normalized or ""))

    @staticmethod
    def _region_starts(value: str) -> list[int]:
        starts: list[int] = []
        for index in range(len(value or "")):
            if _COMMON_REGION_PREFIX.match(value[index:]):
                starts.append(index)
        return starts

    @staticmethod
    def _left_pollution_penalty(normalized: str) -> int:
        if not normalized:
            return 4
        penalty = 0
        if normalized.startswith(tuple(_ORG_LEFT_POLLUTION_TERMS)):
            penalty += 4
        if not BoundaryRepair._starts_with_region_prefix(normalized):
            first_region = _COMMON_REGION_PREFIX.search(normalized)
            if first_region and first_region.start() > 0:
                penalty += 3
        for token in _ORG_LEFT_POLLUTION_TERMS:
            index = normalized.find(token)
            if 0 <= index <= 8:
                penalty += 1
                break
        return penalty

    def _trim_short_org_like_inner(self, inner: str, source_text: str = "") -> tuple[int, int, list[str]]:
        repairs: list[str] = []
        start = 0
        end = len(inner)
        working = inner
        normalized_inner = normalize_entity_text(inner)
        complete_org_shape = bool(
            normalized_inner
            and (
                OFFICIAL_INSTITUTION_PATTERN.fullmatch(normalized_inner)
                or ORG_PATTERN.fullmatch(normalized_inner)
                or is_org_like_text(inner)
            )
            and not SubjectAdmissionGate.is_action_or_function_text(normalized_inner)
            and not SubjectAdmissionGate.is_weak_function_stripped_subject(inner)
        )
        inline_spaced_short_org_shape = bool(
            normalized_inner
            and looks_like_organization_short_name(normalized_inner)
            and re.search(r"\s", inner or "")
        )

        field_tail = _FIELD_LABEL_TAIL.search(working)
        if field_tail and field_tail.start() >= 2:
            end = min(end, field_tail.start())
            repairs.append("strip_following_field_tail")
            working = inner[start:end]

        short_org_boundary = None if complete_org_shape else find_short_org_prefix_before_non_subject_boundary(working)
        if short_org_boundary is not None and self._has_full_org_anchor_for_short_surface(working, source_text):
            short_org_boundary = None
        if short_org_boundary is not None:
            org_start, org_end, boundary_kind = short_org_boundary
            if org_end > org_start and (org_start > 0 or org_end < len(working)):
                start += org_start
                end = min(end, start + (org_end - org_start))
                repairs.append(f"strip_non_subject_right_boundary:{boundary_kind}")
                working = inner[start:end]

        delimiter_match = None if complete_org_shape or inline_spaced_short_org_shape else re.search(r"[\s，,；;。:：、（(《【]", working)
        if delimiter_match and delimiter_match.start() >= 2:
            end = min(end, delimiter_match.start())
            repairs.append("strip_delimited_tail")
            working = inner[start:end]

        while start < end and inner[start] in _TRAILING_PUNCT:
            start += 1
            repairs.append("strip_leading_punctuation")
        while end > start and inner[end - 1] in _TRAILING_PUNCT:
            end -= 1
            repairs.append("strip_trailing_punctuation")

        candidate = inner[start:end]
        normalized = normalize_entity_text(candidate)
        if SubjectAdmissionGate.is_generic_org_reference(normalized):
            return 0, len(inner), ["generic_org_reference_not_boundary_safe"]
        if repairs and not (looks_like_organization_short_name(normalized) or ORG_PATTERN.search(normalized)):
            return 0, len(inner), []
        return start, end, repairs

    @staticmethod
    def _has_full_org_anchor_for_short_surface(value: str, source_text: str) -> bool:
        normalized = normalize_entity_text(value)
        if not 4 <= len(normalized) <= 6:
            return False
        if not looks_like_organization_short_name(normalized):
            return False
        for match in ORG_PATTERN.finditer(source_text or ""):
            full = normalize_entity_text(match.group(0))
            if full and normalized in full:
                return True
        return False

    def _split_narrative_subjects(self, text: str, result: RecognizerResult) -> list[tuple[RecognizerResult, list[str]]]:
        entity_type = str(result.entity_type or "").upper()
        if entity_type not in {"PERSON", "PERSON_NAME", "ORGANIZATION", "COMPANY_NAME"}:
            return []
        start = int(result.start)
        end = int(result.end)
        value = text[start:end] if 0 <= start < end <= len(text) else str(result.text or "")
        if entity_type in {"ORGANIZATION", "COMPANY_NAME"}:
            bridge_split = self._split_person_org_bridge_subject(text, result)
            if bridge_split:
                return bridge_split
        if not value or not re.search(
            r"(?:，|,|；|;|、|与|和|及|以及|并由|另由|另行由|同时由|共同|分别|"
            r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令)",
            value,
        ):
            return []

        subject_spans = self._find_subject_spans_in_narrative(value, entity_type)
        if len(subject_spans) <= 1:
            return []
        split_results: list[tuple[RecognizerResult, list[str]]] = []
        seen: set[tuple[int, int, str]] = set()
        for local_start, local_end in subject_spans:
            global_start = start + local_start
            global_end = start + local_end
            subject_text = text[global_start:global_end]
            key = (global_start, global_end, subject_text)
            if key in seen:
                continue
            seen.add(key)
            split_results.append(
                (
                    RecognizerResult(
                        entity_type=result.entity_type,
                        start=global_start,
                        end=global_end,
                        score=max(float(result.score or 0.0), 0.82),
                        text=subject_text,
                        source=result.source,
                        metadata={**dict(result.metadata or {}), "boundary_split_from": result.text},
                    ),
                    ["split_narrative_subjects"],
                )
            )
        return split_results if len(split_results) >= 2 else []

    def _split_person_org_bridge_subject(
        self,
        text: str,
        result: RecognizerResult,
    ) -> list[tuple[RecognizerResult, list[str]]]:
        value = text[int(result.start) : int(result.end)] if 0 <= int(result.start) < int(result.end) <= len(text) else str(result.text or "")
        normalized = normalize_entity_text(value)
        if not normalized or not re.search(r"(?:担任|任职|就职|供职|出任|为|是|系|兼任|兼为|加入|服务于)", normalized):
            return []
        if not (ORG_PATTERN.search(normalized) or OFFICIAL_INSTITUTION_PATTERN.search(normalized)):
            return []
        bridge_candidates = sorted(
            ["担任", "任职", "就职", "供职", "出任", "兼任", "兼为", "服务于", "加入", "为", "是", "系", "在"],
            key=len,
            reverse=True,
        )
        for bridge in bridge_candidates:
            bridge_index = value.find(bridge)
            if bridge_index <= 0:
                continue
            left = value[:bridge_index].rstrip(" \t\r\n:：,，;；。.!！?？、")
            person_match = None
            for candidate in re.finditer(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", left):
                if candidate.end() == len(left) and looks_like_natural_person_name(candidate.group(0)):
                    person_match = candidate
            if person_match is None:
                continue
            person_text = person_match.group(0)
            if not is_probable_person(person_text):
                continue
            subject_tail = value[bridge_index + len(bridge) :].strip(" \t\r\n:：,，;；。.!！?？、")
            if not subject_tail:
                continue
            subject_match = None
            for pattern in (OFFICIAL_INSTITUTION_PATTERN, ORG_PATTERN):
                match = pattern.search(subject_tail)
                if match:
                    subject_match = match
                    break
            if subject_match is None:
                continue
            candidate_subject = subject_tail[subject_match.start() : subject_match.end()]
            candidate_subject = candidate_subject.strip(" \t\r\n:：,，;；。.!！?？、")
            if not candidate_subject:
                continue
            subject_entity_type = "GOVERNMENT" if is_official_institution_text(candidate_subject) else "ORGANIZATION"
            if not SubjectAdmissionGate.passes_subject_shape(
                subject_entity_type,
                candidate_subject,
                allow_short_org=looks_like_organization_short_name(candidate_subject),
            )[0]:
                continue
            person_start = value.rfind(person_text, 0, bridge_index)
            if person_start < 0:
                continue
            person_end = person_start + len(person_text)
            subject_start = bridge_index + len(bridge) + subject_match.start()
            subject_end = bridge_index + len(bridge) + subject_match.end()
            return [
                (
                    RecognizerResult(
                        entity_type="PERSON",
                        start=int(result.start) + person_start,
                        end=int(result.start) + person_end,
                        score=max(float(result.score or 0.0), 0.84),
                        text=person_text,
                        source=result.source,
                        metadata={**dict(result.metadata or {}), "boundary_split_from": result.text, "bridge_split": True},
                    ),
                    ["split_person_org_bridge_subject"],
                ),
                (
                    RecognizerResult(
                        entity_type=subject_entity_type,
                        start=int(result.start) + subject_start,
                        end=int(result.start) + subject_end,
                        score=max(float(result.score or 0.0), 0.84),
                        text=candidate_subject,
                        source=result.source,
                        metadata={**dict(result.metadata or {}), "boundary_split_from": result.text, "bridge_split": True},
                    ),
                    ["split_person_org_bridge_subject"],
                ),
            ]
        return []

    def _find_subject_spans_in_narrative(self, value: str, entity_type: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if entity_type in {"ORGANIZATION", "COMPANY_NAME"}:
            spans.extend(self._clean_organization_span(value, match.start(), match.end()) for match in ORG_PATTERN.finditer(value))
            spans.extend(self._find_short_org_subject_spans(value))
        elif entity_type in {"PERSON", "PERSON_NAME"}:
            spans.extend(self._find_person_subject_spans(value))
        spans = [span for span in spans if span[1] > span[0]]
        spans = self._deduplicate_nested_spans(spans)
        return spans

    @staticmethod
    def _clean_organization_span(value: str, start: int, end: int) -> tuple[int, int]:
        while start < end and value[start] in "与和及、,，;； ":
            start += 1
        while end > start and value[end - 1] in _TRAILING_PUNCT:
            end -= 1
        return start, end

    @staticmethod
    def _find_short_org_subject_spans(value: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pure_parallel_pattern = re.compile(
            r"^(?P<left>[\u4e00-\u9fa5A-Za-z0-9]{2,6})(?:和|与|及|以及|、)(?P<right>[\u4e00-\u9fa5A-Za-z0-9]{2,6})$"
        )
        pure_parallel = pure_parallel_pattern.fullmatch(normalize_entity_text(value))
        if pure_parallel:
            cursor = 0
            for part in (pure_parallel.group("left"), pure_parallel.group("right")):
                index = value.find(part, cursor)
                if index >= 0 and BoundaryRepair._short_org_parallel_item_is_usable(part):
                    spans.append((index, index + len(part)))
                    cursor = index + len(part)
        action_pattern = re.compile(
            r"(?:^|[，,；;。、]|(?:后续|随后|另行)?由|并由|另由|同时由|与|和|及|以及|共同|"
            r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令)"
            r"\s*(?P<subject>[\u4e00-\u9fa5A-Za-z0-9]{2,6})"
            r"(?=\s*(?:账户|银行账户|材料|资料|文件|负责|继续|履约|履行|结算|付款|收款|交付|供货|"
            r"签署|签订|盖章|落款|承担|确认|对账|配合|共同|分别))"
        )
        for match in action_pattern.finditer(value):
            subject = match.group("subject")
            if looks_like_organization_short_name(subject) or BoundaryRepair._short_org_has_parallel_action_context(value, *match.span("subject")):
                spans.append(match.span("subject"))

        parallel_pattern = re.compile(
            r"(?:^|[，,；;。、]|和|与|及|以及)\s*"
            r"(?P<subject>[\u4e00-\u9fa5A-Za-z0-9]{2,8})"
            r"(?=\s*(?:和|与|及|以及|、|共同|分别|均|负责|继续|履约|履行|结算|付款|收款|交付|供货|签署|签订|盖章|落款|承担|确认|对账|配合|。|$))"
        )
        for match in parallel_pattern.finditer(value):
            subject = match.group("subject")
            if looks_like_organization_short_name(subject) or BoundaryRepair._short_org_has_parallel_action_context(value, *match.span("subject")):
                spans.append(match.span("subject"))
        return spans

    @staticmethod
    def _short_org_parallel_item_is_usable(subject: str) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,6}", subject or ""):
            return False
        if subject in {"双方", "各方", "一方", "三方", "材料", "资料", "事项", "情况"}:
            return False
        return not SubjectAdmissionGate.is_action_or_function_text(subject) and not is_generic_organization_term(subject)

    @staticmethod
    def _short_org_has_parallel_action_context(value: str, start: int, end: int) -> bool:
        subject = value[start:end]
        if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,6}", subject or ""):
            return False
        if subject in {"双方", "各方", "一方", "三方", "材料", "资料", "事项", "情况"}:
            return False
        left = value[max(0, start - 12) : start]
        right = value[end : min(len(value), end + 12)]
        has_parallel = any(token in left for token in ("和", "与", "及", "以及", "、", "由", "并由", "另由")) or any(
            token in right for token in ("和", "与", "及", "以及", "、", "均", "共同", "分别")
        )
        has_action = any(
            token in right
            for token in (
                "账户",
                "银行账户",
                "材料",
                "资料",
                "文件",
                "负责",
                "继续",
                "履约",
                "履行",
                "结算",
                "付款",
                "收款",
                "交付",
                "签署",
                "签订",
                "承担",
            )
        )
        has_function_prefix = any(
            token in left
            for token in ("通过", "经过", "经由", "根据", "依据", "按照", "要求", "通知", "说明", "确认", "请求", "判令")
        )
        return (has_parallel or has_function_prefix) and has_action

    @staticmethod
    def _find_person_subject_spans(value: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        person_pattern = re.compile(
            r"(?:^|[，,；;。、]|(?:后续|随后|另行)?由|并由|另由|同时由|与|和|及|以及|共同)"
            r"\s*(?P<subject>[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?)"
            r"(?=\s*(?:签署|签订|确认|负责|承担|付款|收款|交付|履约|履行|共同|分别|、|，|,|；|;|。|$))"
        )
        for match in person_pattern.finditer(value):
            subject = match.group("subject")
            if is_probable_person(subject):
                spans.append(match.span("subject"))
        linked_person_pattern = re.compile(
            r"(?:^|[与和及、，,；;])\s*"
            r"(?P<subject>[\u4e00-\u9fa5]{2,4}?(?:·[\u4e00-\u9fa5]{2,8})?)"
            r"(?=\s*(?:与|和|及|、|共同|分别|签署|签订|确认|负责|承担|付款|收款|交付|履约|履行))"
        )
        for match in linked_person_pattern.finditer(value):
            subject = match.group("subject")
            if is_probable_person(subject):
                spans.append(match.span("subject"))
        return spans

    @staticmethod
    def _deduplicate_nested_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        ordered = sorted(set(spans), key=lambda item: (item[0], -(item[1] - item[0])))
        deduped: list[tuple[int, int]] = []
        for start, end in ordered:
            if any(existing_start <= start and end <= existing_end for existing_start, existing_end in deduped):
                continue
            deduped.append((start, end))
        return deduped

    def _trim_person_inner(self, inner: str) -> tuple[int, int, list[str]]:
        repairs: list[str] = []
        start = 0
        end = len(inner)
        working = inner

        field_tail = _FIELD_LABEL_TAIL.search(working)
        if field_tail and field_tail.start() >= 2:
            end = min(end, field_tail.start())
            repairs.append("strip_following_field_tail")
            working = inner[start:end]

        person_match = re.search(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", working)
        if not person_match:
            return 0, len(inner), []
        if person_match.start() > 0 or person_match.end() < len(working):
            candidate = working[person_match.start() : person_match.end()]
            if is_probable_person(candidate):
                start += person_match.start()
                end = start + len(candidate)
                repairs.append("trim_to_person_shape")

        return start, end, repairs

    @staticmethod
    def _needs_split(item: tuple[RecognizerResult, list[str]]) -> bool:
        result, _ = item
        if result.entity_type not in {"PERSON", "PERSON_NAME", "ORGANIZATION", "COMPANY_NAME"}:
            return False
        normalized = normalize_entity_text(result.text)
        if result.entity_type in {"ORGANIZATION", "COMPANY_NAME"} and re.search(r"(?:和|与|及|以及)", normalized):
            pieces = [normalize_entity_text(piece) for piece in re.split(r"(?:和|与|及|以及)", str(result.text or ""))]
            usable = [piece for piece in pieces if BoundaryRepair._is_valid_parallel_org_piece(piece)]
            return len(usable) >= 2
        if (
            result.entity_type in {"ORGANIZATION", "COMPANY_NAME"}
            and (
                ORG_PATTERN.fullmatch(normalized)
                or OFFICIAL_INSTITUTION_PATTERN.fullmatch(normalized)
            )
        ):
            return False
        if re.search(r"[、/]", normalized):
            return True
        if result.entity_type in {"PERSON", "PERSON_NAME"}:
            return bool(re.search(r"(?:和|与|及)", normalized))
        return False

    @staticmethod
    def _split_parallel(text: str, item: tuple[RecognizerResult, list[str]]) -> list[tuple[RecognizerResult, list[str]]]:
        result, repairs = item
        pieces = [piece for piece in re.split(r"[、/]|(?:和|与|及|以及)", result.text) if piece.strip()]
        if len(pieces) <= 1:
            return []
        split_results: list[tuple[RecognizerResult, list[str]]] = []
        cursor = result.start
        for piece in pieces:
            local = text.find(piece, cursor, result.end)
            if local < 0:
                continue
            end = local + len(piece)
            normalized = normalize_entity_text(piece)
            if result.entity_type in {"PERSON", "PERSON_NAME"} and not is_probable_person(normalized):
                continue
            if result.entity_type in {"ORGANIZATION", "COMPANY_NAME"} and not BoundaryRepair._is_valid_parallel_org_piece(normalized):
                continue
            metadata = {**dict(result.metadata or {}), "boundary_split_from": result.text}
            if result.entity_type in {"ORGANIZATION", "COMPANY_NAME"}:
                metadata.update({"short_org_candidate": True, "identity_surface": normalized})
            split_results.append(
                (
                    RecognizerResult(
                        entity_type=result.entity_type,
                        start=local,
                        end=end,
                        score=max(float(result.score or 0.0), 0.82),
                        text=text[local:end],
                        source=result.source,
                        metadata=metadata,
                    ),
                    [*repairs, "split_parallel_subject"],
                )
            )
            cursor = end
        return split_results if len(split_results) >= 2 else []

    @staticmethod
    def _is_valid_parallel_org_piece(normalized: str) -> bool:
        if not normalized or len(normalized) < 2:
            return False
        if SubjectAdmissionGate.is_action_or_function_text(normalized):
            return False
        return bool(is_org_like_text(normalized) or looks_like_organization_short_name(normalized))
