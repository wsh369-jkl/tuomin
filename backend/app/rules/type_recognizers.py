"""Type-level deterministic recognizers for rule-first candidate recall."""

from __future__ import annotations

import re
from typing import Any

from app.core.recognizer_base import RecognizerResult
from app.rules.default_subject_policy import canonical_default_entity_type
from app.services.contract_structure_backfill_service import ContractStructureBackfillService
from app.services.lowmem_entity_utils import (
    FINANCIAL_INSTITUTION_PATTERN,
    GOVERNMENT_INSTITUTION_PATTERN,
    OFFICIAL_INSTITUTION_PATTERN,
    ORG_PATTERN,
    deduplicate_results,
    find_value_span,
    infer_semantic_type,
    is_generic_organization_term,
    is_government_institution_text,
    is_official_institution_text,
    is_non_subject_action_or_function_term,
    is_non_subject_generic_org_reference,
    is_probable_person,
    is_weak_function_stripped_org,
    find_short_org_prefix_before_non_subject_boundary,
    looks_like_organization_short_name,
    normalize_entity_text,
    remap_sanitized_span,
    sanitize_recognition_text,
    short_org_boundary_has_strong_surface,
    strip_leading_subject_function_words,
    subject_noun_gate,
)


_VALUE_TAIL = r"(?=$|[\r\n\t，,；;。])"
_BACKFILL_LABEL_SPECS = tuple(ContractStructureBackfillService.LABEL_SPECS)
_BACKFILL_LABELS_BY_TYPE = {
    "ORGANIZATION": tuple(
        str(spec["label"])
        for spec in _BACKFILL_LABEL_SPECS
        if str(spec.get("type") or "") in {"ORGANIZATION", "ROLE_SUBJECT"}
    ),
    "PERSON": tuple(
        str(spec["label"])
        for spec in _BACKFILL_LABEL_SPECS
        if str(spec.get("type") or "") == "PERSON"
    ),
    "LOCATION": tuple(
        str(spec["label"])
        for spec in _BACKFILL_LABEL_SPECS
        if str(spec.get("type") or "") == "LOCATION"
    ),
}


def _label_pattern(labels: tuple[str, ...]) -> str:
    if not labels:
        return r"a^"
    return "(?:" + "|".join(re.escape(label) for label in sorted(set(labels), key=len, reverse=True)) + ")"


_ORG_LABELS = _label_pattern(_BACKFILL_LABELS_BY_TYPE["ORGANIZATION"])
_PERSON_LABELS = _label_pattern(_BACKFILL_LABELS_BY_TYPE["PERSON"])
_LOCATION_LABELS = _label_pattern(_BACKFILL_LABELS_BY_TYPE["LOCATION"])
_BACKFILL_CLEANER = ContractStructureBackfillService()
_SHORT_ORG_LEFT_CUES = (
    "是否通过",
    "已经通过",
    "已通过",
    "并通过",
    "通过",
    "经过",
    "经由",
    "根据",
    "依据",
    "按照",
    "要求",
    "通知",
    "说明",
    "确认",
    "请求",
    "判令",
    "后续由",
    "随后由",
    "另行由",
    "同时由",
    "共同由",
    "分别由",
    "并由",
    "另由",
    "由",
    "向",
    "对",
)
_SHORT_ORG_RIGHT_CUES = (
    "银行账户付款",
    "银行账户收款",
    "银行账户转账",
    "银行账户结算",
    "账户付款",
    "账户收款",
    "账户转账",
    "账户结算",
    "银行账户",
    "账户",
    "负责履约",
    "负责履行",
    "负责结算",
    "负责付款",
    "负责收款",
    "负责交付",
    "负责供货",
    "负责施工",
    "负责对账",
    "负责配合",
    "负责协助",
    "负责办理",
    "继续履约",
    "继续履行",
    "继续结算",
    "继续付款",
    "继续收款",
    "继续交付",
    "继续供货",
    "继续施工",
    "承担责任",
    "承担付款",
    "承担结算",
    "承担供货",
    "承担施工",
    "承担交付",
    "履约",
    "履行",
    "结算",
    "付款",
    "收款",
    "交付",
    "供货",
    "施工",
    "对账",
    "配合",
    "协助",
    "办理",
    "签署",
    "签订",
    "盖章",
    "落款",
)
_SHORT_ORG_LEFT_CUE_PATTERN = "|".join(re.escape(item) for item in sorted(_SHORT_ORG_LEFT_CUES, key=len, reverse=True))
_SHORT_ORG_RIGHT_CUE_PATTERN = "|".join(re.escape(item) for item in sorted(_SHORT_ORG_RIGHT_CUES, key=len, reverse=True))
_FUNCTION_CONTEXT_SHORT_ORG_PATTERN = re.compile(
    rf"(?P<left>{_SHORT_ORG_LEFT_CUE_PATTERN})\s*"
    rf"(?P<subject>[\u4e00-\u9fa5A-Za-z0-9]{{2,6}}?)"
    rf"(?P<right>\s*(?:{_SHORT_ORG_RIGHT_CUE_PATTERN}))"
)
_PARALLEL_CONTEXT_SHORT_ORG_PATTERN = re.compile(
    rf"(?P<left>[\u4e00-\u9fa5A-Za-z0-9]{{2,6}})"
    rf"(?P<sep>和|与|及|以及|、)"
    rf"(?P<right>[\u4e00-\u9fa5A-Za-z0-9]{{2,6}}?)"
    rf"(?=\s*(?:均|共同|分别|各自|继续|负责|履约|履行|结算|付款|收款|交付|供货|施工|对账|配合|协助|办理|签署|签订|盖章|落款))"
)
_STRONG_RULE_ORG_SUFFIX_PATTERN = re.compile(
    r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司"
    r"|分公司|子公司|集团|公司"
    r"|商行|合作社|联合社|联社|工作室|经营部|门市部|营业部|办事处"
    r"|基金会|联合会|俱乐部|实验室|门诊部|诊所"
    r"|律师事务所|会计师事务所|研究院|研究所|服务中心|技术中心)$"
)
_WEAK_RULE_ORG_SUFFIX_PATTERN = re.compile(r"(?:学校|大学|学院|医院|协会|学会)$")
_WEAK_RULE_ORG_NARRATIVE_SIGNAL_PATTERN = re.compile(
    r"(?:另有|还有|其中|以及|但是|因此|相关|前述|上述|涉案|本案|该|其|"
    r"通过|经过|根据|依据|按照|通知|说明|提交|确认|要求|请求|判令|"
    r"承担|负责|继续|履行|履约|结算|付款|收款|交付|配合|联系|协商|办理|"
    r"签订|签署|签约|盖章|落款|对账|施工|供货|利用|涉及|处理|完成|"
    r"材料|资料|文件|信息|事项|情况|环节|绑定|账户|款项|手续)"
)
_SHORT_CONTEXT_NON_SUBJECT_PREFIXES = (
    "相关",
    "前述",
    "上述",
    "涉案",
    "本案",
    "本合同",
    "本协议",
    "该",
    "其",
)
_SHORT_CONTEXT_NON_SUBJECT_TAILS = (
    "材料",
    "资料",
    "文件",
    "信息",
    "事项",
    "情况",
    "环节",
    "内容",
    "手续",
    "账户",
    "业务",
    "已经",
    "已",
    "均",
)
_SHORT_CONTEXT_NON_SUBJECT_INFIXES = (
    "材料已经",
    "资料已经",
    "文件已经",
    "信息已经",
    "情况说明",
    "事项说明",
    "账户处理",
    "业务办理",
)


class TypeRuleRecognizers:
    """Recall common entity types from generic legal/contract structure."""

    ORG_PREFIX_NOISE = re.compile(
        r"^(?:由|并由|后续由|随后由|另由|另行由|同时由|共同由|分别由|"
        r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令|"
        r"向|对|与|和|及|以及|将|把|被|为|"
        r"甲方|乙方|丙方|原告|被告|申请人|被申请人|第三人)\s*[:：]?\s*"
    )
    ORG_PREFIX_NOISE = re.compile(
        r"^(?:由|并由|后续由|随后由|另由|另行由|同时由|共同由|分别由|"
        r"通过|经过|经由|根据|依据|按照|要求|通知|说明|确认|请求|判令|"
        r"向|对|与|和|及|以及|将|把|被|为|"
        r"甲方|乙方|丙方|原告|被告|申请人|被申请人|第三人)\s*[:：]?\s*"
    )
    STRUCTURE_UNIT_CONTAINERS = {
        "table_cell",
        "textbox",
        "header",
        "footer",
        "footnote",
        "endnote",
        "paragraph",
    }

    def recognize(
        self,
        text: str,
        *,
        source_structure: dict[str, Any] | None = None,
    ) -> list[RecognizerResult]:
        results = self._recognize_text_subjects(text)
        sanitized_text, index_map = sanitize_recognition_text(text)
        if sanitized_text and sanitized_text != text:
            results.extend(
                self._remap_sanitized_text_results(
                    source_text=text,
                    index_map=index_map,
                    results=self._recognize_text_subjects(sanitized_text),
                )
            )
        results.extend(self._recognize_docx_structure_units(text, source_structure))
        return deduplicate_results(results)

    def _recognize_text_subjects(self, text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        results.extend(self._recognize_organizations(text))
        results.extend(self._recognize_function_context_short_orgs(text))
        results.extend(self._recognize_people(text))
        results.extend(self._recognize_addresses(text))
        results.extend(self._recognize_aliases(text))
        return results

    @staticmethod
    def _remap_sanitized_text_results(
        *,
        source_text: str,
        index_map: list[int],
        results: list[RecognizerResult],
    ) -> list[RecognizerResult]:
        remapped: list[RecognizerResult] = []
        for result in results:
            span = remap_sanitized_span(index_map, result.start, result.end)
            if span is None:
                continue
            start, end = span
            if start < 0 or end <= start or end > len(source_text):
                continue
            metadata = dict(result.metadata or {})
            normalized = normalize_entity_text(result.text)
            metadata["sanitized_rule_match"] = True
            metadata.setdefault("normalized_text", normalized)
            if result.entity_type in {"ORGANIZATION", "COMPANY_NAME", "GOVERNMENT", "GOVERNMENT_AGENCY", "COURT", "BANK_NAME", "ALIAS"}:
                metadata.setdefault("identity_surface", normalized)
            remapped.append(
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=start,
                    end=end,
                    score=result.score,
                    text=source_text[start:end],
                    source=result.source,
                    metadata=metadata,
                )
            )
        return remapped

    def _recognize_organizations(self, text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        seen_spans: set[tuple[int, int]] = set()
        for raw_start, raw_end, matched_text, sanitized_match in self._iter_sanitized_pattern_matches(text, OFFICIAL_INSTITUTION_PATTERN):
            official_spans = self._official_spans_from_match(text, raw_start, raw_end)
            for start, end in official_spans:
                if (start, end) in seen_spans:
                    continue
                value = text[start:end]
                if not is_official_institution_text(value):
                    continue
                entity_type = "GOVERNMENT"
                metadata = {"official_institution": True}
                if sanitized_match:
                    metadata["sanitized_rule_match"] = True
                    metadata["normalized_text"] = normalize_entity_text(value)
                if is_government_institution_text(value):
                    metadata["official_institution_family"] = "government"
                else:
                    metadata["official_institution_family"] = "financial"
                if (start, end) != (raw_start, raw_end):
                    metadata["boundary_repaired_from"] = matched_text
                    metadata["official_institution_pollution_repaired"] = True
                seen_spans.add((start, end))
                results.append(self._make(start, end, entity_type, text, "rule_official_institution", 0.91, metadata=metadata))
        for raw_start, raw_end, matched_text, sanitized_match in self._iter_sanitized_pattern_matches(text, ORG_PATTERN):
            span = self._clean_organization_match(text, raw_start, raw_end)
            if not span:
                continue
            start, end = span
            if (start, end) in seen_spans:
                continue
            value = text[start:end]
            entity_type = canonical_default_entity_type("COURT" if "法院" in value else "ORGANIZATION", value) or "ORGANIZATION"
            if not self._is_strong_rule_organization_value(value, entity_type=entity_type):
                continue
            metadata = {}
            if sanitized_match:
                metadata["sanitized_rule_match"] = True
                metadata["normalized_text"] = normalize_entity_text(value)
            if entity_type == "GOVERNMENT":
                if not is_official_institution_text(value):
                    continue
                metadata["official_institution"] = True
                metadata["official_institution_family"] = (
                    "government" if is_government_institution_text(value) else "financial"
                )
            if (start, end) != (raw_start, raw_end):
                metadata["boundary_repaired_from"] = matched_text
            seen_spans.add((start, end))
            results.append(self._make(start, end, entity_type, text, "rule_organization", 0.88, metadata=metadata))
        label_pattern = re.compile(rf"(?P<label>{_ORG_LABELS})\s*[:：]\s*(?P<value>[^\r\n，,；;。]{{2,80}}){_VALUE_TAIL}")
        for match in label_pattern.finditer(text or ""):
            label = match.group("label")
            value = _BACKFILL_CLEANER._clean_label_value(match.group("value") or "", "ORGANIZATION", label)
            span = find_value_span(text, value, search_start=match.start("value"), search_end=match.end("value"))
            if not span:
                continue
            entity_type = infer_semantic_type(value, label)
            if not entity_type and looks_like_organization_short_name(value):
                entity_type = "ORGANIZATION"
            entity_type = canonical_default_entity_type(entity_type, value)
            if entity_type not in {"ORGANIZATION", "GOVERNMENT"}:
                continue
            if not subject_noun_gate(
                entity_type,
                value,
                allow_short_org=entity_type == "ORGANIZATION" and looks_like_organization_short_name(value),
            )[0]:
                continue
            start, end = span
            metadata = {"label": label, "source_layer": "structure"}
            if entity_type == "ORGANIZATION" and looks_like_organization_short_name(value):
                metadata["short_org_candidate"] = True
                metadata["requires_manual_review"] = True
            results.append(
                self._make(
                    start,
                    end,
                    entity_type,
                    text,
                    "rule_organization",
                    0.9,
                    metadata=metadata,
                )
            )
        return results

    def _recognize_function_context_short_orgs(self, text: str) -> list[RecognizerResult]:
        """Recall short organization subjects only when context supplies both boundaries."""

        results = self._recognize_function_context_short_orgs_direct(text)
        sanitized_text, index_map = sanitize_recognition_text(text)
        if not sanitized_text or sanitized_text == text:
            return results

        seen_spans = {(item.start, item.end, item.source) for item in results}
        for result in self._recognize_function_context_short_orgs_direct(sanitized_text):
            span = remap_sanitized_span(index_map, result.start, result.end)
            if span is None:
                continue
            start, end = span
            key = (start, end, result.source)
            if key in seen_spans:
                continue
            metadata = dict(result.metadata or {})
            metadata["sanitized_rule_match"] = True
            metadata["normalized_text"] = normalize_entity_text(result.text)
            metadata["identity_surface"] = normalize_entity_text(result.text)
            seen_spans.add(key)
            results.append(
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=start,
                    end=end,
                    score=result.score,
                    text=text[start:end],
                    source=result.source,
                    metadata=metadata,
                )
            )
        results.sort(key=lambda item: (item.start, item.end, item.text))
        return results

    def _recognize_function_context_short_orgs_direct(self, text: str) -> list[RecognizerResult]:
        """Recall short organization subjects on a single already-normalized text view."""

        results: list[RecognizerResult] = []
        seen_spans: set[tuple[int, int]] = set()
        for match in _FUNCTION_CONTEXT_SHORT_ORG_PATTERN.finditer(text or ""):
            start, end = match.span("subject")
            if (start, end) in seen_spans:
                continue
            subject = text[start:end]
            right_cue = normalize_entity_text(match.group("right"))
            if not self._is_usable_context_short_org(subject, right_cue=right_cue):
                continue
            seen_spans.add((start, end))
            results.append(
                self._make(
                    start,
                    end,
                    "ORGANIZATION",
                    text,
                    "rule_organization_context",
                    0.86,
                    metadata={
                        "trigger": "function_action_boundary_short_org",
                        "source_layer": "structure",
                        "short_org_candidate": True,
                        "requires_manual_review": True,
                        "identity_surface": normalize_entity_text(subject),
                        "left_boundary_cue": normalize_entity_text(match.group("left")),
                        "right_boundary_cue": right_cue,
                    },
                )
            )
        for match in _PARALLEL_CONTEXT_SHORT_ORG_PATTERN.finditer(text or ""):
            for group_name in ("left", "right"):
                start, end = match.span(group_name)
                if (start, end) in seen_spans:
                    continue
                subject = text[start:end]
                if not self._is_usable_context_short_org(subject, right_cue="parallel_action"):
                    continue
                seen_spans.add((start, end))
                results.append(
                    self._make(
                        start,
                        end,
                        "ORGANIZATION",
                        text,
                        "rule_organization_context",
                        0.84,
                        metadata={
                            "trigger": "parallel_action_boundary_short_org",
                            "source_layer": "structure",
                            "short_org_candidate": True,
                            "requires_manual_review": True,
                            "identity_surface": normalize_entity_text(subject),
                            "parallel_separator": match.group("sep"),
                        },
                    )
                )
        for start, end, boundary_kind, left_cue in self._iter_right_boundary_short_org_spans(text or ""):
            if (start, end) in seen_spans:
                continue
            subject = text[start:end]
            if not self._is_usable_context_short_org(subject, right_cue=boundary_kind):
                continue
            if not left_cue and not short_org_boundary_has_strong_surface(subject):
                continue
            seen_spans.add((start, end))
            results.append(
                self._make(
                    start,
                    end,
                    "ORGANIZATION",
                    text,
                    "rule_organization_context",
                    0.85,
                    metadata={
                        "trigger": "right_boundary_short_org",
                        "source_layer": "structure",
                        "short_org_candidate": True,
                        "requires_manual_review": True,
                        "identity_surface": normalize_entity_text(subject),
                        "right_boundary_cue": boundary_kind,
                        "left_boundary_cue": left_cue,
                    },
                )
            )
        return results

    @staticmethod
    def _iter_right_boundary_short_org_spans(text: str) -> list[tuple[int, int, str, str]]:
        spans: list[tuple[int, int, str, str]] = []
        if not text:
            return spans
        for index, char in enumerate(text):
            if not re.match(r"[\u4e00-\u9fa5A-Za-z0-9]", char):
                continue
            left_cue = TypeRuleRecognizers._short_org_left_boundary_cue(text, index)
            if left_cue is None:
                continue
            found = find_short_org_prefix_before_non_subject_boundary(text[index : min(len(text), index + 32)])
            if not found:
                continue
            local_start, local_end, boundary_kind = found
            spans.append((index + local_start, index + local_end, boundary_kind, left_cue))
        return spans

    @staticmethod
    def _short_org_left_boundary_cue(text: str, index: int) -> str | None:
        if index <= 0:
            return ""
        previous = text[index - 1]
        if not re.match(r"[\u4e00-\u9fa5A-Za-z0-9]", previous):
            return ""
        left = text[max(0, index - 12) : index]
        for cue in sorted(_SHORT_ORG_LEFT_CUES, key=len, reverse=True):
            if left.endswith(cue):
                return normalize_entity_text(cue)
        return None

    @staticmethod
    def _is_usable_context_short_org(value: str, *, right_cue: str = "") -> bool:
        normalized = normalize_entity_text(value)
        if not normalized or not looks_like_organization_short_name(normalized):
            return False
        if TypeRuleRecognizers._looks_like_non_subject_short_context(normalized):
            return False
        org_context_right_cue = any(
            token in normalize_entity_text(right_cue)
            for token in ("账户", "银行账户", "材料", "资料", "文件", "结算", "付款", "收款", "对账")
        )
        if is_probable_person(normalized) and not org_context_right_cue:
            return False
        if is_non_subject_action_or_function_term(normalized):
            return False
        if is_non_subject_generic_org_reference(normalized) or is_generic_organization_term(normalized):
            return False
        return subject_noun_gate("ORGANIZATION", normalized, allow_short_org=True)[0]

    @staticmethod
    def _is_strong_rule_organization_value(value: str, *, entity_type: str) -> bool:
        normalized = normalize_entity_text(value)
        if not normalized:
            return False
        if entity_type == "GOVERNMENT":
            return is_official_institution_text(normalized)
        if is_official_institution_text(normalized):
            return True
        if _STRONG_RULE_ORG_SUFFIX_PATTERN.search(normalized):
            return True
        weak_suffix = _WEAK_RULE_ORG_SUFFIX_PATTERN.search(normalized)
        if not weak_suffix:
            return False
        stem = normalized[: weak_suffix.start()]
        if len(stem) < 2:
            return False
        if _WEAK_RULE_ORG_NARRATIVE_SIGNAL_PATTERN.search(stem):
            return False
        if re.match(r"^(?:国家|中国|全国|中央|最高|北京|上海|天津|重庆|广州|深圳|南京|杭州|成都|武汉|西安)", stem):
            return True
        if re.search(r"(?:省|自治区|特别行政区|市|地区|自治州|盟|区|县|旗|自治县|自治旗)", stem):
            return True
        return len(stem) >= 4

    @staticmethod
    def _looks_like_non_subject_short_context(value: str) -> bool:
        normalized = normalize_entity_text(value)
        if not normalized:
            return False
        if normalized in _SHORT_CONTEXT_NON_SUBJECT_TAILS:
            return True
        if any(token in normalized for token in _SHORT_CONTEXT_NON_SUBJECT_INFIXES):
            return True
        if any(normalized.startswith(prefix) for prefix in _SHORT_CONTEXT_NON_SUBJECT_PREFIXES) and any(
            token in normalized for token in _SHORT_CONTEXT_NON_SUBJECT_TAILS
        ):
            return True
        return False

    def _recognize_docx_structure_units(
        self,
        text: str,
        source_structure: dict[str, Any] | None,
    ) -> list[RecognizerResult]:
        if not text or not isinstance(source_structure, dict):
            return []
        results: list[RecognizerResult] = []
        seen: set[tuple[str, int, int]] = set()
        for unit in self._iter_docx_units(source_structure):
            unit_text = str(unit.get("text") or "")
            if not unit_text.strip():
                continue
            container_type = str(unit.get("container_type") or unit.get("unit_type") or "").strip()
            if container_type not in self.STRUCTURE_UNIT_CONTAINERS:
                continue
            unit_start = self._coerce_int(unit.get("start"), -1)
            unit_end = self._coerce_int(unit.get("end"), -1)
            if unit_start < 0 or unit_end <= unit_start or text[unit_start:unit_end] != unit_text:
                found = text.find(unit_text)
                if found < 0:
                    continue
                unit_start = found
                unit_end = found + len(unit_text)
            for local in self._recognize_unit_subject_shapes(unit_text, container_type):
                start = unit_start + int(local.start)
                end = unit_start + int(local.end)
                if start < unit_start or end > unit_end or text[start:end] != local.text:
                    continue
                key = (local.entity_type, start, end)
                if key in seen:
                    continue
                seen.add(key)
                metadata = dict(local.metadata or {})
                metadata.update(self._unit_metadata(unit, container_type))
                results.append(
                    RecognizerResult(
                        entity_type=local.entity_type,
                        start=start,
                        end=end,
                        score=max(float(local.score or 0.0), 0.89 if container_type == "table_cell" else 0.87),
                        text=text[start:end],
                        source="rule_docx_structure",
                        metadata=metadata,
                    )
                )
        return results

    def _recognize_unit_subject_shapes(self, unit_text: str, container_type: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        seen_spans: set[tuple[int, int]] = set()
        for raw_start, raw_end, matched_text, sanitized_match in self._iter_sanitized_pattern_matches(unit_text, OFFICIAL_INSTITUTION_PATTERN):
            span = self._clean_organization_match(unit_text, raw_start, raw_end)
            if not span or span in seen_spans:
                continue
            start, end = span
            value = unit_text[start:end]
            if not is_official_institution_text(value):
                continue
            seen_spans.add(span)
            metadata = {
                "trigger": "docx_structure_unit_rule_pass",
                "source_layer": "structure",
                "official_institution": True,
                "official_institution_family": "government" if is_government_institution_text(value) else "financial",
                "normalized_text": normalize_entity_text(value),
            }
            if sanitized_match:
                metadata["sanitized_rule_match"] = True
            if span != (raw_start, raw_end):
                metadata["boundary_repaired_from"] = matched_text
            results.append(self._make(start, end, "GOVERNMENT", unit_text, "rule_docx_structure", 0.89, metadata=metadata))
        for raw_start, raw_end, matched_text, sanitized_match in self._iter_sanitized_pattern_matches(unit_text, ORG_PATTERN):
            span = self._clean_organization_match(unit_text, raw_start, raw_end)
            if not span or span in seen_spans:
                continue
            start, end = span
            value = unit_text[start:end]
            entity_type = canonical_default_entity_type("COURT" if "法院" in value else "ORGANIZATION", value)
            if entity_type not in {"ORGANIZATION", "GOVERNMENT"}:
                continue
            if not subject_noun_gate(entity_type, value)[0]:
                continue
            seen_spans.add(span)
            metadata = {
                "trigger": "docx_structure_unit_rule_pass",
                "source_layer": "structure",
                "normalized_text": normalize_entity_text(value),
            }
            if entity_type == "GOVERNMENT":
                metadata["official_institution"] = True
                metadata["official_institution_family"] = "government" if is_government_institution_text(value) else "financial"
            if sanitized_match:
                metadata["sanitized_rule_match"] = True
            if span != (raw_start, raw_end):
                metadata["boundary_repaired_from"] = matched_text
            results.append(self._make(start, end, entity_type, unit_text, "rule_docx_structure", 0.88, metadata=metadata))
        if container_type in {"table_cell", "textbox", "header", "footer", "footnote", "endnote"}:
            results.extend(self._recognize_unit_person_labels(unit_text))
        return results

    def _clean_organization_match(self, text: str, start: int, end: int) -> tuple[int, int] | None:
        value = text[start:end]
        if not value:
            return None
        local_start = 0
        local_end = len(value)
        prefix = self.ORG_PREFIX_NOISE.match(value)
        prefix_consumed = False
        if prefix and prefix.end() < local_end:
            local_start = prefix.end()
            prefix_consumed = True
        candidate = value[local_start:local_end].strip(" \t\r\n，,；;。:：、")
        if not candidate:
            return None
        local_start += value[local_start:local_end].find(candidate)
        local_end = local_start + len(candidate)
        normalized = normalize_entity_text(candidate)
        if not normalized or is_generic_organization_term(normalized):
            return None
        if prefix_consumed and is_weak_function_stripped_org(candidate):
            return None
        if is_non_subject_generic_org_reference(normalized):
            return None
        if is_non_subject_action_or_function_term(normalized):
            return None
        if re.search(r"(?:是否|以及|但是|因此|判令|请求|事实与理由)", normalized) and len(normalized) >= 12:
            return None
        entity_type = "GOVERNMENT" if is_official_institution_text(normalized) else "ORGANIZATION"
        if not subject_noun_gate(entity_type, normalized)[0]:
            embedded = self._embedded_organization_span_after_boundary(value)
            if embedded is None:
                return None
            embedded_start, embedded_end = embedded
            local_start = embedded_start
            local_end = embedded_end
            candidate = value[local_start:local_end]
            normalized = normalize_entity_text(candidate)
            entity_type = "GOVERNMENT" if is_official_institution_text(normalized) else "ORGANIZATION"
            if not subject_noun_gate(entity_type, normalized)[0]:
                return None
        return start + local_start, start + local_end

    @staticmethod
    def _embedded_organization_span_after_boundary(value: str) -> tuple[int, int] | None:
        """Recover a real subject after a polluted official/action prefix."""

        if not value:
            return None
        boundary_pattern = re.compile(
            r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司|分公司|子公司|公司|集团|"
            r"[ \t\r\n,，;；。:：、]|与|和|及|以及|向|对|由|通过|经过|经由|根据|依据|按照|"
            r"要求|通知|说明|确认|请求|判令|将|把|被|为)"
        )
        candidates: list[tuple[int, int]] = []
        for boundary in boundary_pattern.finditer(value):
            offset = boundary.end()
            if offset >= len(value):
                continue
            suffix = value[offset:].lstrip(" \t\r\n,，;；。:：、")
            if not suffix:
                continue
            delta = value[offset:].find(suffix)
            local_start = offset + max(delta, 0)
            for pattern in (OFFICIAL_INSTITUTION_PATTERN, ORG_PATTERN):
                match = pattern.match(suffix)
                if not match:
                    continue
                local_end = local_start + match.end()
                candidate = value[local_start:local_end]
                if is_weak_function_stripped_org(candidate):
                    continue
                normalized = normalize_entity_text(candidate)
                entity_type = "GOVERNMENT" if is_official_institution_text(normalized) else "ORGANIZATION"
                if subject_noun_gate(entity_type, normalized)[0]:
                    candidates.append((local_start, local_end))
        if not candidates:
            return None
        return max(candidates, key=lambda span: (span[1] - span[0], span[0]))

    @staticmethod
    def _official_spans_from_match(text: str, start: int, end: int) -> list[tuple[int, int]]:
        value = text[start:end]
        if not value:
            return []
        if is_official_institution_text(value):
            return [(start, end)]

        spans: list[tuple[int, int]] = []
        embedded = TypeRuleRecognizers._embedded_organization_span_after_boundary(value)
        if embedded is not None:
            embedded_start, embedded_end = embedded
            embedded_value = value[embedded_start:embedded_end]
            if is_official_institution_text(embedded_value):
                spans.append((start + embedded_start, start + embedded_end))
        # If a broad financial/government pattern swallowed the preceding
        # company or narrative text, only restart after a real boundary. This
        # prevents "前一公司 + 银行" from being treated as one official subject
        # while still recovering the real bank/court/agency name.
        restart_offsets = {0}
        for match in re.finditer(
            r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司|分公司|子公司|公司|集团|"
            r"[ \t\r\n,，;；。:：、]|与|和|及|以及|向|对|由|通过|经过|经由|根据|依据|按照|"
            r"要求|通知|说明|确认|请求|判令|将|把|被|为)",
            value,
        ):
            restart_offsets.add(match.end())
        for offset in sorted(item for item in restart_offsets if item < len(value)):
            candidate = value[offset:].strip(" \t\r\n,，;；。:：、")
            if not candidate:
                continue
            delta = value[offset:].find(candidate)
            candidate_start = start + offset + max(delta, 0)
            for pattern in (GOVERNMENT_INSTITUTION_PATTERN, FINANCIAL_INSTITUTION_PATTERN):
                sub_match = pattern.match(candidate)
                if not sub_match:
                    continue
                sub_start = candidate_start + sub_match.start()
                sub_end = candidate_start + sub_match.end()
                sub_value = text[sub_start:sub_end]
                if is_official_institution_text(sub_value):
                    spans.append((sub_start, sub_end))
        spans = sorted(set(spans), key=lambda span: (span[0], -(span[1] - span[0])))
        deduped: list[tuple[int, int]] = []
        for span in spans:
            if any(existing[0] <= span[0] and existing[1] >= span[1] for existing in deduped):
                continue
            deduped.append(span)
        return deduped

    @staticmethod
    def _iter_sanitized_pattern_matches(
        text: str,
        pattern: re.Pattern[str],
    ) -> list[tuple[int, int, str, bool]]:
        if not text:
            return []
        matches: list[tuple[int, int, str, bool]] = []
        seen: set[tuple[int, int, str]] = set()
        for match in pattern.finditer(text):
            start, end = match.span()
            key = (start, end, match.group())
            if key in seen:
                continue
            seen.add(key)
            matches.append((start, end, match.group(), False))

        sanitized_text, index_map = sanitize_recognition_text(text)
        if sanitized_text == text:
            return matches
        for match in pattern.finditer(sanitized_text):
            span = remap_sanitized_span(index_map, match.start(), match.end())
            if span is None:
                continue
            start, end = span
            matched_text = text[start:end]
            key = (start, end, matched_text)
            if key in seen:
                continue
            seen.add(key)
            matches.append((start, end, matched_text, True))
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return matches

    def _recognize_unit_person_labels(self, unit_text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        pattern = re.compile(rf"(?P<label>{_PERSON_LABELS})\s*[:：]?\s*(?P<value>[\u4e00-\u9fa5·\s]{{2,18}}){_VALUE_TAIL}")
        for match in pattern.finditer(unit_text or ""):
            label = match.group("label")
            value = _BACKFILL_CLEANER._clean_label_value(match.group("value") or "", "PERSON", label)
            compact = re.sub(r"\s+", "", value)
            if not is_probable_person(compact):
                continue
            value_start = match.start("value")
            value_end = match.end("value")
            span = self._find_compact_span(unit_text, compact, value_start, value_end)
            if not span:
                continue
            start, end = span
            results.append(
                self._make(
                    start,
                    end,
                    "PERSON",
                    unit_text,
                    "rule_docx_structure",
                    0.92,
                    metadata={
                        "label": label,
                        "trigger": "docx_structure_unit_rule_pass",
                        "source_layer": "structure",
                        "normalized_text": compact,
                    },
                )
            )
        return results

    @staticmethod
    def _find_compact_span(text: str, compact_value: str, search_start: int, search_end: int) -> tuple[int, int] | None:
        bounded = text[search_start:search_end]
        compact_chars: list[str] = []
        index_map: list[int] = []
        for index, char in enumerate(bounded, start=search_start):
            if char.isspace():
                continue
            compact_chars.append(char)
            index_map.append(index)
        compact_text = "".join(compact_chars)
        local = compact_text.find(compact_value)
        if local < 0:
            return None
        return index_map[local], index_map[local + len(compact_value) - 1] + 1

    def _iter_docx_units(self, source_structure: dict[str, Any]):
        seen_unit_ids: set[str] = set()
        raw_units = source_structure.get("docx_text_units")
        if isinstance(raw_units, list):
            for unit in raw_units:
                if not isinstance(unit, dict):
                    continue
                unit_id = str(unit.get("unit_id") or "")
                if unit_id:
                    seen_unit_ids.add(unit_id)
                yield unit
        pages = source_structure.get("pages")
        if not isinstance(pages, list):
            return
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("units"), list):
                continue
            for unit in page["units"]:
                if not isinstance(unit, dict):
                    continue
                unit_id = str(unit.get("unit_id") or "")
                if unit_id and unit_id in seen_unit_ids:
                    continue
                if unit_id:
                    seen_unit_ids.add(unit_id)
                yield unit

    @staticmethod
    def _unit_metadata(unit: dict[str, Any], container_type: str) -> dict[str, Any]:
        rewrite_policy = str(unit.get("rewrite_policy") or "exact")
        return {
            "source": "rule_docx_structure",
            "base_source": "rule_docx_structure",
            "source_layer": "structure",
            "trigger": "docx_structure_unit_rule_pass",
            "docx_unit_id": unit.get("unit_id"),
            "docx_part_name": unit.get("part_name"),
            "docx_container_type": container_type,
            "docx_unit_type": unit.get("unit_type"),
            "docx_table_index": unit.get("table_index"),
            "docx_row_index": unit.get("row_index"),
            "docx_col_index": unit.get("col_index"),
            "docx_rewrite_policy": rewrite_policy,
            "docx_review_required": rewrite_policy != "exact",
        }

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _recognize_people(self, text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        pattern = re.compile(rf"(?P<label>{_PERSON_LABELS})\s*[:：]\s*(?P<value>[\u4e00-\u9fa5·\s]{{2,12}}){_VALUE_TAIL}")
        for match in pattern.finditer(text or ""):
            label = match.group("label")
            value = _BACKFILL_CLEANER._clean_label_value(match.group("value") or "", "PERSON", label)
            compact = re.sub(r"\s+", "", value)
            if not is_probable_person(compact):
                continue
            span = find_value_span(text, value, search_start=match.start("value"), search_end=match.end("value"))
            if not span:
                continue
            start, end = span
            results.append(
                self._make(
                    start,
                    end,
                    "PERSON",
                    text,
                    "rule_person",
                    0.9,
                    metadata={"label": label, "source_layer": "structure", "normalized_text": compact},
                )
            )
        return results

    def _recognize_addresses(self, text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        pattern = re.compile(rf"(?P<label>{_LOCATION_LABELS})\s*[:：]\s*(?P<value>[^\r\n；;。]{{4,120}}){_VALUE_TAIL}")
        for match in pattern.finditer(text or ""):
            label = match.group("label")
            value = _BACKFILL_CLEANER._clean_label_value(match.group("value") or "", "LOCATION", label)
            if not self._looks_like_address(value):
                continue
            span = find_value_span(text, value, search_start=match.start("value"), search_end=match.end("value"))
            if not span:
                continue
            start, end = span
            results.append(
                self._make(
                    start,
                    end,
                    "LOCATION",
                    text,
                    "rule_address",
                    0.88,
                    metadata={"label": label, "source_layer": "structure"},
                )
            )
        return results

    def _recognize_aliases(self, text: str) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        pattern = re.compile(
            r"(?P<full>[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,60}?"
            r"(?:公司|集团|银行|法院|检察院|商行|工作室|合作社|经营部|事务所))"
            r"\s*[（(]\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?"
            r"(?P<alias>[\u4e00-\u9fa5A-Za-z0-9\-]{2,20})[”\"'‘’]?\s*[）)]"
        )
        for match in pattern.finditer(text or ""):
            full = match.group("full").strip()
            alias = match.group("alias").strip()
            full_span = match.span("full")
            alias_span = match.span("alias")
            if not full or not alias:
                continue
            entity_type = canonical_default_entity_type("COURT" if "法院" in full else "ORGANIZATION", full) or "ORGANIZATION"
            results.append(
                self._make(
                    full_span[0],
                    full_span[1],
                    entity_type,
                    text,
                    "rule_alias",
                    0.92,
                    metadata={
                        "definition_alias": alias,
                        "alias_surface": alias,
                        "definition_full_text": full,
                        "canonical": full,
                        "source_layer": "structure",
                    },
                )
            )
            results.append(
                self._make(
                    alias_span[0],
                    alias_span[1],
                    "ALIAS",
                    text,
                    "rule_alias",
                    0.9,
                    metadata={
                        "definition_alias": alias,
                        "alias_surface": alias,
                        "definition_full_text": full,
                        "canonical": full,
                        "source_layer": "structure",
                    },
                )
            )
        return results

    @staticmethod
    def _looks_like_address(value: str) -> bool:
        compact = re.sub(r"\s+", "", value or "")
        if compact in {"地址", "住所", "联系地址", "注册地址", "项目地址"}:
            return False
        return any(token in compact for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室"))

    @staticmethod
    def _make(
        start: int,
        end: int,
        entity_type: str,
        text: str,
        source: str,
        score: float,
        metadata: dict | None = None,
    ) -> RecognizerResult:
        return RecognizerResult(
            entity_type=entity_type,
            start=start,
            end=end,
            score=score,
            text=text[start:end],
            source=source,
            metadata={"source_layer": "structure", **dict(metadata or {})},
        )
