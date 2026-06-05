"""Deterministic contract structure backfill."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.core.recognizer_base import RecognizerResult
from app.services.lowmem_entity_utils import (
    ORG_PATTERN,
    ORG_SUFFIX_PATTERN,
    clean_candidate_text,
    deduplicate_results,
    find_value_span,
    infer_semantic_type,
    is_generic_organization_term,
    is_probable_person,
    looks_like_organization_short_name,
    make_entity,
    remap_results_to_source,
    sanitize_recognition_text,
    strip_identity_reference_prefix,
)


class ContractStructureBackfillService:
    """Extract labelled contract fields without calling a generative model."""

    INLINE_ALIAS_SEGMENT = re.compile(
        r"\s*(?:[（(]\s*(?:以下简称|下称|简称|又称)[^）)]*[）)]|"
        r"[，,]?\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?[\u4e00-\u9fa5A-Za-z0-9\-]{2,32}[”\"'‘’]?)"
    )
    INLINE_ALIAS_TAIL = re.compile(
        r"\s*(?:[（(]\s*(?:以下简称|下称|简称|又称)[^）)]*[）)]|"
        r"[，,]?\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?[\u4e00-\u9fa5A-Za-z0-9\-]{2,32}[”\"'‘’]?)\s*$"
    )

    LABEL_VARIANT_SUFFIX = r"(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?"

    LABEL_SPECS: List[Dict[str, object]] = [
        {"label": "甲方", "type": "ORGANIZATION", "role": "甲方"},
        {"label": "乙方", "type": "ORGANIZATION", "role": "乙方"},
        {"label": "丙方", "type": "ORGANIZATION", "role": "丙方"},
        {"label": "上诉人", "type": "ROLE_SUBJECT", "role": "上诉人"},
        {"label": "被上诉人", "type": "ROLE_SUBJECT", "role": "被上诉人"},
        {"label": "原审原告", "type": "ROLE_SUBJECT", "role": "原审原告"},
        {"label": "原审被告", "type": "ROLE_SUBJECT", "role": "原审被告"},
        {"label": "一审原告", "type": "ROLE_SUBJECT", "role": "一审原告"},
        {"label": "一审被告", "type": "ROLE_SUBJECT", "role": "一审被告"},
        {"label": "申请人", "type": "ROLE_SUBJECT", "role": "申请人"},
        {"label": "被申请人", "type": "ROLE_SUBJECT", "role": "被申请人"},
        {"label": "原告", "type": "ROLE_SUBJECT", "role": "原告"},
        {"label": "被告", "type": "ROLE_SUBJECT", "role": "被告"},
        {"label": "第三人", "type": "ROLE_SUBJECT", "role": "第三人"},
        {"label": "控告人", "type": "ROLE_SUBJECT", "role": "控告人"},
        {"label": "被控告人", "type": "ROLE_SUBJECT", "role": "被控告人"},
        {"label": "举报人", "type": "ROLE_SUBJECT", "role": "举报人"},
        {"label": "被举报人", "type": "ROLE_SUBJECT", "role": "被举报人"},
        {"label": "申诉人", "type": "ROLE_SUBJECT", "role": "申诉人"},
        {"label": "被申诉人", "type": "ROLE_SUBJECT", "role": "被申诉人"},
        {"label": "起诉人", "type": "ROLE_SUBJECT", "role": "起诉人"},
        {"label": "自诉人", "type": "ROLE_SUBJECT", "role": "自诉人"},
        {"label": "法定代表人", "type": "PERSON", "role": "法定代表人"},
        {"label": "法定代理人", "type": "PERSON", "role": "法定代理人"},
        {"label": "联系人", "type": "PERSON", "role": "联系人"},
        {"label": "委托诉讼代理人", "type": "PERSON", "role": "委托诉讼代理人"},
        {"label": "诉讼代理人", "type": "PERSON", "role": "诉讼代理人"},
        {"label": "委托代理人", "type": "PERSON", "role": "委托代理人"},
        {"label": "代理人", "type": "PERSON", "role": "代理人"},
        {"label": "负责人", "type": "PERSON", "role": "负责人"},
        {"label": "开户行", "type": "BANK_NAME", "role": "开户行"},
        {"label": "户名", "type": "ACCOUNT_NAME", "role": "户名"},
        {"label": "账户名称", "type": "ACCOUNT_NAME", "role": "账户名称"},
        {"label": "收款单位", "type": "ORGANIZATION", "role": "收款单位"},
        {"label": "付款单位", "type": "ORGANIZATION", "role": "付款单位"},
        {"label": "项目名称", "type": "PROJECT", "role": "项目名称"},
        {"label": "工程名称", "type": "PROJECT", "role": "工程名称"},
        {"label": "项目地址", "type": "ADDRESS", "role": "项目地址"},
        {"label": "工程地址", "type": "ADDRESS", "role": "工程地址"},
        {"label": "住所", "type": "ADDRESS", "role": "住所"},
        {"label": "住所地", "type": "ADDRESS", "role": "住所地"},
        {"label": "地址", "type": "ADDRESS", "role": "地址"},
        {"label": "住址", "type": "ADDRESS", "role": "住址"},
        {"label": "现住址", "type": "ADDRESS", "role": "现住址"},
        {"label": "户籍地", "type": "ADDRESS", "role": "户籍地"},
        {"label": "户籍地址", "type": "ADDRESS", "role": "户籍地址"},
        {"label": "身份证住址", "type": "ADDRESS", "role": "身份证住址"},
        {"label": "经常居住地", "type": "ADDRESS", "role": "经常居住地"},
        {"label": "通讯地址", "type": "ADDRESS", "role": "通讯地址"},
        {"label": "工作地址", "type": "ADDRESS", "role": "工作地址"},
        {"label": "送达地址", "type": "ADDRESS", "role": "送达地址"},
        {"label": "注册地址", "type": "ADDRESS", "role": "注册地址"},
    ]

    LABEL_TAIL = re.compile(
        r"(?=\s*(?:甲方|乙方|丙方|申请人|被申请人|原告|被告|第三人|"
        r"控告人|被控告人|举报人|被举报人|申诉人|被申诉人|起诉人|自诉人|"
        r"上诉人|被上诉人|原审原告|原审被告|一审原告|一审被告|"
        r"法定代表人|法定代理人|负责人|联系人|委托诉讼代理人|诉讼代理人|委托代理人|代理人|"
        r"开户行|户名|账户名称|收款单位|付款单位|项目名称(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
        r"工程名称(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
        r"项目地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
        r"工程地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
        r"住所地|住所|现住址|住址|户籍地|户籍地址|身份证住址|经常居住地|地址|电话|账号)\s*[:：])"
    )
    NON_VALUE_TERMS = {
        "岗位",
        "职位",
        "职务",
        "工作地址",
        "通讯地址",
        "住所",
        "住所地",
        "地址",
        "签约时间",
        "合同期限",
        "签约各方",
        "甲方",
        "乙方",
        "丙方",
        "上诉人",
        "被上诉人",
        "原审原告",
        "原审被告",
        "一审原告",
        "一审被告",
        "原告",
        "被告",
        "申请人",
        "被申请人",
        "第三人",
        "控告人",
        "被控告人",
        "举报人",
        "被举报人",
        "申诉人",
        "被申诉人",
        "起诉人",
        "自诉人",
        "法定代表人",
        "法定代理人",
        "负责人",
        "联系人",
    }
    ADDRESS_TOKENS = (
        "省",
        "市",
        "区",
        "县",
        "镇",
        "乡",
        "村",
        "路",
        "街",
        "道",
        "号",
        "栋",
        "室",
        "自治区",
        "自治州",
        "自治县",
        "旗",
        "盟",
    )
    DATE_LIKE = re.compile(r"\d{4}[./年-]\d{1,2}(?:[./月-]\d{1,2})?日?")
    FOLLOWING_FIELD_LABELS = tuple(
        sorted(
            {
                *(str(spec["label"]) for spec in LABEL_SPECS),
                "统一社会信用代码",
                "社会信用代码",
                "信用代码",
                "营业执照号码",
                "营业执照号",
                "营业执照",
                "组织机构代码",
                "纳税人识别号",
                "纳税识别号",
                "税号",
                "身份证号码",
                "身份证号",
                "公民身份号码",
                "联系电话",
                "联系地址",
                "电话",
                "手机",
                "邮箱",
                "电子邮箱",
                "邮编",
                "传真",
                "网址",
                "账号",
                "银行账号",
                "账户",
                "开户账号",
                "开户地址",
                "开户银行",
                "开户名称",
            },
            key=len,
            reverse=True,
        )
    )
    PARTY_LIKE_LABELS = {
        "甲方",
        "乙方",
        "丙方",
        "上诉人",
        "被上诉人",
        "原审原告",
        "原审被告",
        "一审原告",
        "一审被告",
        "申请人",
        "被申请人",
        "原告",
        "被告",
        "第三人",
        "控告人",
        "被控告人",
        "举报人",
        "被举报人",
        "申诉人",
        "被申诉人",
        "起诉人",
        "自诉人",
        "收款单位",
        "付款单位",
        "户名",
        "账户名称",
    }
    PARTY_NARRATIVE_TAIL = re.compile(
        r"(?:"
        r"(?:系|为|作为|是)本(?:合同|协议|项目)"
        r"|(?:系|为|作为|是)(?:签约方|供货方|采购方|承包方|发包方|卖方|买方|出租方|承租方|服务方|"
        r"履约方|付款方|收款方|合同主体|履约主体|付款主体|收款主体|合作方)"
        r"|(?:负责|承担|继续|签署|签订|履行|付款|收款|供货|交付|结算|对账|盖章|落款)"
        r")"
    )
    ORG_NOISE_PREFIX_SEPARATORS = (
        "同时设立",
        "共同设立",
        "设立",
        "成立",
        "对接代理商",
        "代理商",
        "后续由",
        "另由",
        "另行由",
        "由",
        "向",
        "对",
        "与",
        "和",
        "及",
        "为",
    )

    def extract(self, text: str) -> List[RecognizerResult]:
        sanitized_text, index_map = sanitize_recognition_text(text)
        working_text = sanitized_text or text
        results: list[RecognizerResult] = []
        results.extend(self._extract_labelled_values(working_text))
        results.extend(self._extract_table_like_label_values(working_text))
        results.extend(self._extract_inline_address_contexts(working_text))
        results.extend(self._extract_inline_residence_addresses(working_text))
        results.extend(self._extract_aliases(working_text))
        results.extend(self._extract_legal_caption_subjects(working_text))
        results.extend(self._extract_inline_party_role_subjects(working_text))
        results.extend(self._extract_court_mentions(working_text))
        results.extend(self._extract_signature_subjects(working_text))
        remapped = remap_results_to_source((item for item in results if item is not None), text, index_map)
        return deduplicate_results(remapped)

    def _extract_labelled_values(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for spec in self.LABEL_SPECS:
            label = str(spec["label"])
            entity_type = str(spec["type"])
            pattern = re.compile(
                rf"(?<![\u4e00-\u9fa5A-Za-z0-9])(?P<label>{re.escape(label)}{self.LABEL_VARIANT_SUFFIX})"
                r"(?:[（(][^）)]{1,30}[）)])?\s*(?:[:：]\s*|\n\s*|\s{2,})"
                r"(?P<value>[^\n\r；;。]{2,140})"
            )
            for match in pattern.finditer(text):
                value = self._clean_label_value(match.group("value"), entity_type, label)
                if not value:
                    continue
                span = find_value_span(text, value, search_start=match.start("value"), search_end=match.end("value"))
                if span is None:
                    continue
                resolved_type = entity_type
                if entity_type == "ROLE_SUBJECT":
                    resolved_type = infer_semantic_type(value, label)
                elif entity_type == "ORGANIZATION" and is_probable_person(value):
                    resolved_type = "PERSON"
                elif entity_type == "ORGANIZATION":
                    resolved_type = infer_semantic_type(value, label)
                result = make_entity(
                    text=text,
                    start=span[0],
                    end=span[1],
                    entity_type=resolved_type,
                    source="contract_structure_backfill",
                    score=0.92,
                    metadata={
                        "source": "contract_structure_backfill",
                        "trigger": "party_label",
                        "label": label,
                        "role": spec.get("role"),
                    },
                )
                if result:
                    results.append(result)
        return results

    def _extract_aliases(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        patterns = [
            re.compile(
                rf"(?P<full>{ORG_PATTERN.pattern})"
                r"\s*[（(]\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?"
                r"(?P<alias>[\u4e00-\u9fa5A-Za-z0-9]{2,20})[”\"'‘’]?\s*[）)]"
            ),
            re.compile(
                rf"(?P<full>{ORG_PATTERN.pattern})"
                r"\s*，?\s*(?:简称|下称|又称)\s*[“\"'‘’]?"
                r"(?P<alias>[\u4e00-\u9fa5A-Za-z0-9]{2,20})[”\"'‘’]?"
            ),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                full_text = self._normalize_alias_definition_full_text(match.group("full"))
                if not full_text:
                    continue
                full_span = find_value_span(
                    text,
                    full_text,
                    search_start=match.start("full"),
                    search_end=match.end("alias"),
                )
                if full_span is None:
                    continue
                alias_span = match.span("alias")
                full = make_entity(
                    text=text,
                    start=full_span[0],
                    end=full_span[1],
                    entity_type="ORGANIZATION",
                    source="contract_structure_backfill",
                    score=0.94,
                    metadata={
                        "source": "contract_structure_backfill",
                        "trigger": "alias_definition",
                        "alias": match.group("alias"),
                    },
                )
                alias = make_entity(
                    text=text,
                    start=alias_span[0],
                    end=alias_span[1],
                    entity_type="ALIAS",
                    source="contract_structure_backfill",
                    score=0.94,
                    metadata={
                        "source": "contract_structure_backfill",
                        "trigger": "alias_definition",
                        "canonical": full_text,
                    },
                )
                if full:
                    results.append(full)
                if alias:
                    results.append(alias)
        return results

    def _extract_table_like_label_values(self, text: str) -> List[RecognizerResult]:
        rows: list[tuple[str, int]] = []
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            line_start = cursor
            cursor += len(raw_line)
            line = raw_line.strip()
            if line:
                rows.append((line, line_start + raw_line.find(line)))

        address_labels = {"通讯地址", "工作地址", "住址", "住所", "住所地", "地址", "项目地址", "工程地址"}
        label_terms = {str(spec["label"]) for spec in self.LABEL_SPECS} | self.NON_VALUE_TERMS
        results: list[RecognizerResult] = []

        for index, (line, _line_start) in enumerate(rows):
            normalized_line = re.sub(r"[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]+$", "", line)
            if normalized_line not in address_labels:
                continue
            for candidate, candidate_start in rows[index + 1 : index + 14]:
                normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", candidate)
                if not normalized or normalized in label_terms:
                    continue
                if self.DATE_LIKE.fullmatch(normalized):
                    continue
                if not self._looks_like_address_value(normalized):
                    continue
                result = make_entity(
                    text=text,
                    start=candidate_start,
                    end=candidate_start + len(candidate),
                    entity_type="ADDRESS",
                    source="contract_structure_backfill",
                    score=0.93,
                    metadata={
                        "source": "contract_structure_backfill",
                        "trigger": "table_label",
                        "label": line,
                        "role": line,
                    },
                )
                if result:
                    results.append(result)
                break
        return results

    def _extract_inline_address_contexts(self, text: str) -> List[RecognizerResult]:
        pattern = re.compile(
            r"(?<![\u4e00-\u9fa5A-Za-z0-9])"
            r"(?P<label>项目地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"工程地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"通讯地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"工作地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"家庭地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"身份证住址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"经常居住地(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"户籍地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"户籍地(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"现住址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"住址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"住所地(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"住所(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?|"
            r"地址(?:[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3})?)"
            r"(?:显示|记载|登记|实际为|实际是|为|是|[:：])?"
            r"\s*[“\"']?"
            r"(?P<value>[^，,。；;\n\r”\"']{4,100})"
            r"[”\"']?"
        )
        results: list[RecognizerResult] = []
        for match in pattern.finditer(text):
            value = clean_candidate_text(match.group("value"))
            value = re.split(r"(?:一致|相同|，|。|；|;|\n)", value, maxsplit=1)[0].strip()
            normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", value)
            if not normalized or self.DATE_LIKE.fullmatch(normalized):
                continue
            if not self._looks_like_address_value(normalized):
                continue
            span = find_value_span(text, value, search_start=match.start("value"), search_end=match.end("value"))
            if span is None:
                continue
            span = self._extend_address_tail(text, span)
            result = make_entity(
                text=text,
                start=span[0],
                end=span[1],
                entity_type="ADDRESS",
                source="contract_structure_backfill",
                score=0.94,
                metadata={
                    "source": "contract_structure_backfill",
                    "trigger": "inline_address_context",
                    "label": match.group("label"),
                    "role": match.group("label"),
                },
            )
            if result:
                results.append(result)
        return results

    def _extract_inline_residence_addresses(self, text: str) -> List[RecognizerResult]:
        pattern = re.compile(
            r"(?:^|[，,；;。\n\r])(?P<label>现住|居住于|居住在)\s*"
            r"(?P<value>[^，,。；;\n\r]{4,120})"
        )
        results: list[RecognizerResult] = []
        for match in pattern.finditer(text):
            value = clean_candidate_text(match.group("value"))
            normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", value)
            if not normalized or self.DATE_LIKE.fullmatch(normalized):
                continue
            if not self._looks_like_address_value(normalized):
                continue
            span = find_value_span(text, value, search_start=match.start("value"), search_end=match.end("value"))
            if span is None:
                continue
            span = self._extend_address_tail(text, span)
            result = make_entity(
                text=text,
                start=span[0],
                end=span[1],
                entity_type="ADDRESS",
                source="contract_structure_backfill",
                score=0.93,
                metadata={
                    "source": "contract_structure_backfill",
                    "trigger": "inline_residence_address",
                    "label": match.group("label"),
                    "role": match.group("label"),
                },
            )
            if result:
                results.append(result)
        return results

    def _extract_legal_caption_subjects(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        caption_pattern = re.compile(
            rf"(?P<left>{ORG_PATTERN.pattern})诉(?P<right>[^\n\r。；;]{{2,160}}?)(?:纠纷|一案)"
        )
        for match in caption_pattern.finditer(text):
            left = self._make_backfill_entity(
                text,
                match.start("left"),
                match.end("left"),
                "ORGANIZATION",
                trigger="legal_caption",
                label="诉讼主体",
            )
            if left:
                results.append(left)

            right_start = match.start("right")
            right_text = match.group("right")
            for org_match in ORG_PATTERN.finditer(right_text):
                org = self._make_backfill_entity(
                    text,
                    right_start + org_match.start(),
                    right_start + org_match.end(),
                    "COURT" if "法院" in org_match.group() else "ORGANIZATION",
                    trigger="legal_caption",
                    label="诉讼主体",
                )
                if org:
                    results.append(org)

            for piece_match in re.finditer(r"[\u4e00-\u9fa5·]{2,20}", right_text):
                value = re.split(r"(?:买卖|合同|纠纷|一案|案由|请求|事实)", piece_match.group(), maxsplit=1)[0]
                if not is_probable_person(value):
                    continue
                local_start = right_start + piece_match.start()
                person = self._make_backfill_entity(
                    text,
                    local_start,
                    local_start + len(value),
                    "PERSON",
                    trigger="legal_caption",
                    label="诉讼主体",
                )
                if person:
                    results.append(person)

        appeal_pattern = re.compile(
            rf"(?P<party>{ORG_PATTERN.pattern})不服(?P<court>[\u4e00-\u9fa5]{{2,40}}人民法院)"
        )
        for match in appeal_pattern.finditer(text):
            party = self._make_backfill_entity(
                text,
                match.start("party"),
                match.end("party"),
                "ORGANIZATION",
                trigger="appeal_caption",
                label="上诉主体",
            )
            court = self._make_backfill_entity(
                text,
                match.start("court"),
                match.end("court"),
                "COURT",
                trigger="appeal_caption",
                label="原审法院",
            )
            if party:
                results.append(party)
            if court:
                results.append(court)
        return results

    def _extract_inline_party_role_subjects(self, text: str) -> List[RecognizerResult]:
        role_pattern = (
            r"上诉人|被上诉人|原审原告|原审被告|一审原告|一审被告|"
            r"申请人|被申请人|原告|被告|第三人|"
            r"控告人|被控告人|举报人|被举报人|申诉人|被申诉人|起诉人|自诉人"
        )
        person_pattern = r"(?:[\u4e00-\u9fa5]{2,4}|[\u4e00-\u9fa5]{2,8}·[\u4e00-\u9fa5]{2,8})"
        pattern = re.compile(
            rf"(?:(?<![\u4e00-\u9fa5A-Za-z0-9])|欠付|应付|支付给|付款给|对|向|由|判令|即)(?P<label>{role_pattern})"
            rf"(?:[（(][^）)]{{1,30}}[）)])?\s*(?P<value>{ORG_PATTERN.pattern}|{person_pattern})"
            r"(?=，|,|。|；|;|、|因|与|和|及|向|诉|不服|对|就|应|为|系|的|$)"
        )
        results: list[RecognizerResult] = []
        for match in pattern.finditer(text):
            if not self._is_valid_inline_party_context(text, match):
                continue
            value = match.group("value")
            entity_type = "COURT" if "法院" in value else "ORGANIZATION"
            if is_probable_person(value):
                entity_type = "PERSON"
                if self._looks_like_action_phrase_person(value):
                    continue
            elif not ORG_PATTERN.search(value):
                continue
            start, end = match.start("value"), match.end("value")
            if entity_type in {"ORGANIZATION", "COURT"}:
                prefixed_remainder = strip_identity_reference_prefix(value)
                if prefixed_remainder:
                    if len(prefixed_remainder) < 2 or is_generic_organization_term(prefixed_remainder):
                        continue
                leading_person = self._leading_person_before_nested_party(value)
                if leading_person:
                    person_start = match.start("value") + value.index(leading_person)
                    person = make_entity(
                        text=text,
                        start=person_start,
                        end=person_start + len(leading_person),
                        entity_type="PERSON",
                        source="contract_structure_backfill",
                        score=0.89,
                        metadata={
                            "source": "contract_structure_backfill",
                            "trigger": "inline_party_role",
                            "label": match.group("label"),
                            "role": match.group("label"),
                        },
                    )
                    if person:
                        results.append(person)
                start, end = self._trim_org_noise_prefix_span(text, start, end)
            result = make_entity(
                text=text,
                start=start,
                end=end,
                entity_type=entity_type,
                source="contract_structure_backfill",
                score=0.89,
                metadata={
                    "source": "contract_structure_backfill",
                    "trigger": "inline_party_role",
                    "label": match.group("label"),
                    "role": match.group("label"),
                },
            )
            if result:
                results.append(result)
        return results

    @staticmethod
    def _is_valid_inline_party_context(text: str, match: re.Match) -> bool:
        label = match.group("label")
        value = match.group("value")
        before = text[max(0, match.start("label") - 8) : match.start("label")]
        between = text[match.end("label") : match.start("value")]
        after = text[match.end("value") : min(len(text), match.end("value") + 12)]

        if re.search(r"[:：]\s*$", between):
            return True
        if before.endswith(("欠付", "应付", "支付给", "付款给", "对", "向", "由", "判令", "即")):
            return True
        if ORG_PATTERN.search(value) and any(token in value for token in ("对被告", "对原告", "向被告", "向原告")):
            return True
        if ORG_PATTERN.search(value) and any(token in after for token in ("因", "不服", "诉", "纠纷", "一案", "与")):
            return True
        if is_probable_person(value) and any(token in after for token in ("，", ",", "、", "欠付", "承担", "对", "向")):
            return True
        if label in {"原告", "被告", "第三人"} and before.endswith(("、", "，", ",", "；", ";")):
            return True
        return False

    @staticmethod
    def _leading_person_before_nested_party(value: str) -> str:
        match = re.match(
            r"(?P<person>[\u4e00-\u9fa5]{2,4})(?:对|向)(?:被告|原告|上诉人|被上诉人|申请人|被申请人|第三人)",
            value or "",
        )
        if not match:
            return ""
        person = match.group("person")
        return person if is_probable_person(person) else ""

    def _extract_court_mentions(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in re.finditer(r"[\u4e00-\u9fa5]{2,40}(?:人民法院|仲裁委员会|人民检察院)", text):
            start = match.start()
            value = match.group()
            for separator in ("根据", "不服", "此致", "向", "至"):
                index = value.rfind(separator)
                if index >= 0:
                    start += index + len(separator)
                    value = value[index + len(separator) :]
                    break
            result = self._make_backfill_entity(
                text,
                start,
                start + len(value),
                "COURT",
                trigger="court_mention",
                label="法院/仲裁机构",
            )
            if result:
                results.append(result)
        return results

    @staticmethod
    def _make_backfill_entity(
        text: str,
        start: int,
        end: int,
        entity_type: str,
        *,
        trigger: str,
        label: str,
    ) -> Optional[RecognizerResult]:
        if entity_type in {"ORGANIZATION", "COURT"}:
            start, end = ContractStructureBackfillService._trim_role_prefix_span(text, start, end)
            start, end = ContractStructureBackfillService._trim_org_noise_prefix_span(text, start, end)
        return make_entity(
            text=text,
            start=start,
            end=end,
            entity_type=entity_type,
            source="contract_structure_backfill",
            score=0.9,
            metadata={
                "source": "contract_structure_backfill",
                "trigger": trigger,
                "label": label,
            },
        )

    @staticmethod
    def _trim_role_prefix_span(text: str, start: int, end: int) -> tuple[int, int]:
        value = text[start:end]
        match = re.match(
            r"(?:上诉人|被上诉人|原审原告|原审被告|一审原告|一审被告|"
            r"申请人|被申请人|原告|被告|第三人|"
            r"控告人|被控告人|举报人|被举报人|申诉人|被申诉人|起诉人|自诉人)"
            r"(?:（[^）]*）|\([^)]*\))?\s*[:：，,、]?",
            value,
        )
        if not match:
            return start, end
        candidate = value[match.end() :].lstrip(" ：:，,、")
        if len(candidate) < 2:
            return start, end
        return end - len(candidate), end

    @staticmethod
    def _trim_org_noise_prefix_span(text: str, start: int, end: int) -> tuple[int, int]:
        value = text[start:end]
        separators = (
            "被告",
            "原告",
            "上诉人",
            "被上诉人",
            "申请人",
            "被申请人",
            "第三人",
            "控告人",
            "被控告人",
            "举报人",
            "被举报人",
            "申诉人",
            "被申诉人",
            "起诉人",
            "自诉人",
            "收款单位",
            "付款单位",
            "对",
            "向",
            "由",
            "与",
            "和",
            "及",
        )
        for separator in separators:
            index = value.rfind(separator)
            if index < 0:
                continue
            candidate = value[index + len(separator) :].lstrip(" ：:，,、")
            if len(candidate) < 2:
                continue
            if ORG_PATTERN.fullmatch(candidate) or re.search(r"(?:人民法院|仲裁委员会|人民检察院)$", candidate):
                return end - len(candidate), end
        return start, end

    @staticmethod
    def _extend_address_tail(text: str, span: tuple[int, int]) -> tuple[int, int]:
        tail = text[span[1] : span[1] + 12]
        match = re.match(r"(?:[-—]\d+[A-Za-z]?)+", tail)
        if not match:
            return span
        return span[0], span[1] + match.end()

    def _extract_signature_subjects(self, text: str) -> List[RecognizerResult]:
        if not text:
            return []
        start = max(0, len(text) - 1800)
        tail = text[start:]
        if not any(token in tail for token in ("签字", "签章", "盖章", "落款", "日期")):
            return []

        results: list[RecognizerResult] = []
        cursor = start
        for raw_line in tail.splitlines(keepends=True):
            line_start = cursor
            cursor += len(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            if not self._looks_like_signature_line(line):
                continue
            for match in ORG_PATTERN.finditer(line):
                result = make_entity(
                    text=text,
                    start=line_start + raw_line.find(line) + match.start(),
                    end=line_start + raw_line.find(line) + match.end(),
                    entity_type="COURT" if "法院" in match.group() else "ORGANIZATION",
                    source="contract_structure_backfill",
                    score=0.88,
                    metadata={
                        "source": "contract_structure_backfill",
                        "trigger": "signature_block",
                        "label": "签字/签章/盖章落款主体",
                    },
                )
                if result:
                    results.append(result)
        return results

    @staticmethod
    def _looks_like_signature_line(line: str) -> bool:
        normalized = re.sub(r"\s+", "", line)
        if not normalized:
            return False
        if any(token in normalized for token in ("签字", "签章", "盖章", "落款")):
            return True
        if re.match(r"^(?:甲方|乙方|丙方|上诉人|申请人|原告|被告|收款单位|付款单位)[:：]", normalized):
            return True
        return False

    def _clean_label_value(self, value: str, entity_type: str, label: str) -> str:
        cleaned = clean_candidate_text(value)
        cleaned = self.LABEL_TAIL.split(cleaned, maxsplit=1)[0].strip()
        if entity_type in {"ORGANIZATION", "ACCOUNT_NAME", "PROJECT"}:
            cleaned = self._strip_inline_alias_segments(cleaned)
            cleaned = self._truncate_at_following_field_label(cleaned, entity_type, label)
            cleaned = self._truncate_party_like_narrative_tail(cleaned, entity_type, label)
            while True:
                stripped = self.INLINE_ALIAS_TAIL.sub("", cleaned).strip("，,；;。 ")
                if stripped == cleaned:
                    break
                cleaned = stripped
        normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", cleaned)
        if not normalized or normalized in self.NON_VALUE_TERMS:
            return ""
        if self.DATE_LIKE.fullmatch(normalized):
            return ""
        if entity_type == "PERSON":
            person_match = re.search(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", cleaned)
            if not person_match:
                return ""
            person = person_match.group()
            return "" if self._looks_like_action_phrase_person(person) else person
        if entity_type == "ADDRESS" and not self._looks_like_address_value(normalized):
            return ""
        return cleaned

    def _normalize_alias_definition_full_text(self, value: str) -> str:
        cleaned = clean_candidate_text(value)
        cleaned = self._strip_inline_alias_segments(cleaned)
        cleaned = self._truncate_at_following_field_label(cleaned, "ORGANIZATION", "简称")
        cleaned = self._truncate_unlabeled_org_narrative_tail(cleaned)
        cleaned = self._trim_org_noise_prefix_text(cleaned)
        cleaned = self._truncate_unlabeled_org_narrative_tail(cleaned)
        org_match = ORG_PATTERN.search(cleaned)
        if org_match:
            cleaned = org_match.group().strip()
        return cleaned.strip(" ：:，,、;；")

    def _strip_inline_alias_segments(self, value: str) -> str:
        cleaned = value
        while True:
            stripped = self.INLINE_ALIAS_SEGMENT.sub("", cleaned).strip("，,；;。 ")
            if stripped == cleaned:
                return stripped
            cleaned = stripped

    def _truncate_at_following_field_label(self, value: str, entity_type: str, label: str) -> str:
        if entity_type not in {"ORGANIZATION", "ACCOUNT_NAME", "PROJECT"}:
            return value
        best = len(value)
        for token in self.FOLLOWING_FIELD_LABELS:
            if token == label:
                continue
            index = value.find(token, 1)
            if index <= 0:
                continue
            best = min(best, index)
        truncated = value[:best].rstrip(" ：:，,、;；")
        return truncated or value

    def _truncate_party_like_narrative_tail(self, value: str, entity_type: str, label: str) -> str:
        if entity_type not in {"ORGANIZATION", "ACCOUNT_NAME"}:
            return value
        if label not in self.PARTY_LIKE_LABELS:
            return value
        org_match = ORG_PATTERN.search(value)
        if org_match:
            return org_match.group().strip()
        narrative_match = self.PARTY_NARRATIVE_TAIL.search(value)
        if narrative_match and narrative_match.start() >= 2:
            candidate = value[: narrative_match.start()].rstrip(" ：:，,、;；")
            if candidate:
                return candidate
        compact = re.sub(r"\s+", "", value)
        if looks_like_organization_short_name(compact) or is_probable_person(compact):
            return compact
        return value

    def _truncate_unlabeled_org_narrative_tail(self, value: str) -> str:
        org_match = ORG_PATTERN.search(value)
        if org_match:
            candidate = org_match.group().strip()
            if candidate:
                return candidate
        narrative_match = self.PARTY_NARRATIVE_TAIL.search(value)
        if narrative_match and narrative_match.start() >= 2:
            candidate = value[: narrative_match.start()].rstrip(" ：:，,、;；")
            if candidate:
                return candidate
        return value

    def _trim_org_noise_prefix_text(self, value: str) -> str:
        cleaned = value.strip(" ：:，,、;；")
        best = cleaned
        for separator in self.ORG_NOISE_PREFIX_SEPARATORS:
            index = cleaned.rfind(separator)
            if index < 0:
                continue
            candidate = cleaned[index + len(separator) :].lstrip(" ：:，,、;；")
            if len(candidate) < 2:
                continue
            if ORG_PATTERN.fullmatch(candidate) or re.search(rf"{ORG_SUFFIX_PATTERN}$", candidate):
                if len(candidate) < len(best):
                    best = candidate
        return best

    @staticmethod
    def _looks_like_action_phrase_person(value: str) -> bool:
        normalized = re.sub(r"\s+", "", value or "")
        return any(
            token in normalized
            for token in (
                "上诉",
                "起诉",
                "提起",
                "贵院",
                "特向",
                "就针",
                "针对",
                "判令",
                "请求",
                "认为",
                "审理",
                "判决",
                "裁定",
            )
        )

    def _looks_like_address_value(self, value: str) -> bool:
        normalized = re.sub(r"\s+", "", value or "")
        if any(token in normalized for token in ("业务覆盖区域", "负责区域", "销售区域", "管辖区域")):
            return False
        detailed_tokens = ("路", "街", "道", "号", "栋", "室", "村", "组", "小区", "大厦", "广场", "园")
        if normalized.count("、") >= 3 and not any(token in normalized for token in detailed_tokens):
            return False
        return any(token in normalized for token in self.ADDRESS_TOKENS)
