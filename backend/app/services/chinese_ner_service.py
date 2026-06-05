"""Chinese NER service for low-memory recall."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from app.core.recognizer_base import RecognizerResult
from app.services.lowmem_entity_utils import (
    DATE_PATTERN,
    GENERIC_ORGANIZATION_TERMS,
    NON_ENTITY_ROLE_TERMS,
    ORG_PATTERN,
    ORG_SUFFIX_PATTERN,
    deduplicate_results,
    has_identity_reference_prefix,
    is_identity_reference_term,
    is_generic_organization_term,
    is_org_like_text,
    is_position_title,
    is_probable_person,
    looks_like_organization_short_name,
    make_entity,
    remap_results_to_source,
    sanitize_recognition_text,
    strip_identity_reference_prefix,
)
from app.services.lowmem_memory import release_runtime_memory
from app.services.lowmem_model_assets import build_model_asset
from app.services.hf_token_classifier import HFTokenClassificationPipeline


class ChineseNERService:
    """Token-classification style NER with optional runtime and regex fallback."""

    LABEL_MAP = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "NAME": "PERSON",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "COMPANY": "ORGANIZATION",
        "GOVERNMENT": "ORGANIZATION",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "ADDRESS": "ADDRESS",
        "TIME": "DATE",
        "DATE": "DATE",
    }

    PERSON_CONTEXT = re.compile(
        r"(?:法定代表人|法人代表|负责人|联系人|委托诉讼代理人|诉讼代理人|委托代理人|代理人|"
        r"签署人|经办人|申请人|被申请人|上诉人|被上诉人|原审原告|原审被告|原告|被告)"
        r"\s*(?:[:：]\s*|\s+)(?P<person>[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?)"
    )
    LOCATION_CONTEXT = re.compile(
        r"(?:身份证住址|经常居住地|户籍地址|户籍地|现住址|住所地|住所|住址|注册地址|送达地址|通讯地址|联系地址|地址)"
        r"\s*(?:[:：]\s*|\s+)(?P<location>[^\n\r；;。]{4,120})"
    )
    PROCEDURAL_PREFIX = re.compile(
        r"^(?:上诉人|被上诉人|原审原告|原审被告|一审原告|一审被告|"
        r"申请人|被申请人|原告|被告|第三人|不服|此致|根据)"
        r"(?:（[^）]*）|\([^)]*\))?\s*[:：，,、]?"
    )
    TRAILING_PUNCTUATION = " \t\r\n，,。；;：:、"
    INLINE_ALIAS_SEGMENT = re.compile(
        r"\s*(?:[（(]\s*(?:以下简称|下称|简称|又称)[^）)]*[）)]|"
        r"[，,]?\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?[\u4e00-\u9fa5A-Za-z0-9\-]{2,32}[”\"'‘’]?)"
    )
    INLINE_ALIAS_TAIL = re.compile(
        r"\s*(?:[（(]\s*(?:以下简称|下称|简称|又称)[^）)]*[）)]|"
        r"[，,]?\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?[\u4e00-\u9fa5A-Za-z0-9\-]{2,32}[”\"'‘’]?)\s*$"
    )
    FOLLOWING_FIELD_LABELS = tuple(
        sorted(
            {
                "甲方",
                "乙方",
                "丙方",
                "申请人",
                "被申请人",
                "原告",
                "被告",
                "第三人",
                "法定代表人",
                "负责人",
                "联系人",
                "委托诉讼代理人",
                "诉讼代理人",
                "委托代理人",
                "代理人",
                "签署人",
                "身份证住址",
                "经常居住地",
                "户籍地址",
                "户籍地",
                "现住址",
                "住址",
                "住所",
                "住所地",
                "注册地址",
                "送达地址",
                "通讯地址",
                "联系地址",
                "地址",
                "开户行",
                "开户银行",
                "银行名称",
                "户名",
                "账户名称",
                "收款单位",
                "付款单位",
                "项目名称",
                "工程名称",
                "合同编号",
                "合同号",
                "案号",
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
                "开户名称",
            },
            key=len,
            reverse=True,
        )
    )
    PARTY_NARRATIVE_TAIL = re.compile(
        r"(?:"
        r"(?:系|为|作为|是)本(?:合同|协议|项目)"
        r"|(?:系|为|作为|是)(?:签约方|供货方|采购方|承包方|发包方|卖方|买方|出租方|承租方|服务方|"
        r"履约方|付款方|收款方|合同主体|履约主体|付款主体|收款主体|合作方)"
        r"|(?:负责|承担|继续|签署|签订|履行|付款|收款|供货|交付|结算|对账|盖章|落款)"
        r"|(?<![\u4e00-\u9fa5A-Za-z0-9])系(?![\u4e00-\u9fa5A-Za-z0-9])"
        r")"
    )
    LOCATION_NOISE_TERMS = {
        "重",
        "岗位",
        "工作",
        "重点",
        "严重",
        "调整",
        "区域",
        "地点",
        "地址",
        "国家",
    }
    PERSON_NOISE_TERMS = {
        "上诉",
        "诉人",
        "原告",
        "被告",
        "申请",
        "法院",
        "公司",
        "法定",
        "代表",
        "代表人",
        "人代表",
        "表人",
        "法人",
        "负责",
        "负责人",
        "联系",
        "联系人",
    }
    PERSON_NOISE_TERMS.update(NON_ENTITY_ROLE_TERMS)
    ORGANIZATION_NOISE_TERMS = {
        "公司",
        "银行",
        "法院",
        "人民法院",
        "集团",
        "机构",
        "单位",
        "个人银行",
        "国家",
        "法定代表人",
        "法人代表",
        "负责人",
        "联系人",
    }
    ORGANIZATION_NOISE_TERMS.update(NON_ENTITY_ROLE_TERMS)
    ORGANIZATION_PROSE_NOISE = (
        "该表",
        "列明",
        "部分",
        "直接付款",
        "个人银行",
        "转账凭证",
        "证明",
        "收取",
        "款项",
        "认为",
        "提交",
    )

    def __init__(self, model_id: str, backend: str = "transformers", source_name: str = "ner") -> None:
        self.model_id = model_id
        self.backend = backend
        self.source_name = source_name
        self.asset = build_model_asset(model_id, role=source_name, backend=backend)
        self.backend_available = False
        self.backend_error: Optional[str] = None
        self._pipeline = None

    @property
    def ready(self) -> bool:
        return self.asset.installed

    def extract(self, text: str) -> List[RecognizerResult]:
        sanitized_text, index_map = sanitize_recognition_text(text)
        working_text = sanitized_text or text
        model_results: list[RecognizerResult] = []
        if self.backend.lower() == "deterministic":
            self.backend_available = True
            self.backend_error = None
        if self.asset.installed and self.backend.lower() == "transformers":
            try:
                model_results = self._extract_with_transformers(working_text)
                self.backend_available = True
                self.backend_error = None
            except Exception as exc:
                self.backend_available = False
                self.backend_error = f"{type(exc).__name__}: {exc}"

        fallback_results = self._fallback_extract(working_text)
        remapped = remap_results_to_source([*model_results, *fallback_results], text, index_map)
        return deduplicate_results(remapped)

    def _extract_with_transformers(self, text: str) -> List[RecognizerResult]:
        if self._pipeline is None:
            if self.asset.path is None:
                return []
            try:
                self._pipeline = HFTokenClassificationPipeline(str(self.asset.path))
            except Exception as exc:
                release_runtime_memory()
                raise RuntimeError(f"transformers_not_available:{type(exc).__name__}: {exc}") from exc

        results: list[RecognizerResult] = []
        for segment, offset in self._iter_segments(text):
            predictions = self._pipeline(segment)
            for item in predictions:
                entity_type = self._map_prediction_type(item)
                if not entity_type:
                    continue
                start = offset + int(item.get("start") or 0)
                end = offset + int(item.get("end") or 0)
                if start >= end:
                    continue
                normalized_span = self._normalize_model_span(text, start, end, entity_type)
                if normalized_span is None:
                    continue
                start, end, value = normalized_span
                if entity_type == "ORGANIZATION" and any(token in value for token in ("法院", "检察院", "仲裁委员会")):
                    entity_type = "COURT"
                result = make_entity(
                    text=text,
                    start=start,
                    end=end,
                    entity_type=entity_type,
                    source=self.source_name,
                    score=float(item.get("score") or 0.76),
                    metadata={
                        "source": "chinese_ner_service",
                        "model": self.model_id,
                        "backend": self.backend,
                        "fallback": False,
                        "ner_label": item.get("entity_group") or item.get("entity"),
                    },
                )
                if result:
                    results.append(result)
        return results

    def _normalize_model_span(
        self,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> Optional[tuple[int, int, str]]:
        value = text[start:end]
        leading_trimmed = len(value) - len(value.lstrip())
        if leading_trimmed:
            start += leading_trimmed
            value = value.lstrip()

        value = value.strip(" \t\r\n，,。；;：:、…“”\"'`")
        end = start + len(value)

        if entity_type in {"ORGANIZATION", "COURT"}:
            org_trimmed = self._trim_to_semantic_org_value(value)
            if org_trimmed != value:
                delta = len(value) - len(org_trimmed)
                start += delta
                value = org_trimmed
                end = start + len(value)

            cleaned_org = self._normalize_org_candidate_text(value)
            if cleaned_org != value:
                value = cleaned_org
                end = start + len(value)

            while True:
                match = self.PROCEDURAL_PREFIX.match(value)
                if not match:
                    break
                prefix_len = match.end()
                remainder = value[prefix_len:].lstrip(self.TRAILING_PUNCTUATION)
                if len(remainder) < 2:
                    break
                start = end - len(value) + prefix_len + (len(value[prefix_len:]) - len(remainder))
                value = remainder.rstrip(self.TRAILING_PUNCTUATION)
                end = start + len(value)

        if not self._is_valid_model_candidate(entity_type, value):
            return None
        return start, end, value

    def _normalize_org_candidate_text(self, value: str) -> str:
        cleaned = value.strip(self.TRAILING_PUNCTUATION)
        cleaned = self._strip_inline_alias_segments(cleaned)
        cleaned = self._truncate_at_following_field_label(cleaned)
        while True:
            stripped = self.INLINE_ALIAS_TAIL.sub("", cleaned).strip(self.TRAILING_PUNCTUATION)
            if stripped == cleaned:
                break
            cleaned = stripped
        org_match = ORG_PATTERN.search(cleaned)
        if org_match:
            return org_match.group().strip()
        narrative_match = self.PARTY_NARRATIVE_TAIL.search(cleaned)
        if narrative_match and narrative_match.start() >= 2:
            candidate = cleaned[: narrative_match.start()].rstrip(self.TRAILING_PUNCTUATION)
            compact = re.sub(r"\s+", "", candidate)
            if candidate and looks_like_organization_short_name(compact):
                return candidate
        if cleaned.endswith(("系", "为")):
            candidate = cleaned[:-1].rstrip(self.TRAILING_PUNCTUATION)
            compact = re.sub(r"\s+", "", candidate)
            if candidate and looks_like_organization_short_name(compact):
                return candidate
        return cleaned

    def _strip_inline_alias_segments(self, value: str) -> str:
        cleaned = value
        while True:
            stripped = self.INLINE_ALIAS_SEGMENT.sub("", cleaned).strip(self.TRAILING_PUNCTUATION)
            if stripped == cleaned:
                return stripped
            cleaned = stripped

    def _truncate_at_following_field_label(self, value: str) -> str:
        best = len(value)
        for token in self.FOLLOWING_FIELD_LABELS:
            index = value.find(token, 1)
            if index <= 0:
                continue
            best = min(best, index)
        truncated = value[:best].rstrip(self.TRAILING_PUNCTUATION)
        return truncated or value

    def _trim_to_semantic_org_value(self, value: str) -> str:
        cleaned = value.strip(self.TRAILING_PUNCTUATION)
        separators = (
            "的是",
            "负责的是",
            "包括",
            "根据",
            "不服",
            "欠付原告",
            "欠付被告",
            "原告",
            "被告",
            "上诉人",
            "被上诉人",
            "申请人",
            "被申请人",
            "第三人",
            "在",
            "于",
            "向",
            "与",
            "和",
            "为",
            "诉",
        )
        for separator in separators:
            index = cleaned.rfind(separator)
            if index < 0:
                continue
            candidate = cleaned[index + len(separator) :].strip(self.TRAILING_PUNCTUATION)
            if separator == "诉" and candidate.startswith("讼"):
                continue
            if len(candidate) >= 2 and re.search(ORG_SUFFIX_PATTERN, candidate):
                return candidate
        return cleaned

    def _is_valid_model_candidate(self, entity_type: str, value: str) -> bool:
        normalized = re.sub(r"[\s，,。；;：:、]+", "", value or "")
        if len(normalized) < 2:
            return False

        if entity_type == "LOCATION":
            if normalized in self.LOCATION_NOISE_TERMS:
                return False
            return True

        if entity_type == "ADDRESS":
            if normalized in self.LOCATION_NOISE_TERMS:
                return False
            address_tokens = ["省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室"]
            return len(normalized) >= 6 or any(token in normalized for token in address_tokens)

        if entity_type == "PERSON":
            if normalized in self.PERSON_NOISE_TERMS or is_identity_reference_term(normalized):
                return False
            if is_org_like_text(normalized) or is_position_title(normalized):
                return False
            if any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "号", "室")):
                return False
            return re.fullmatch(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", normalized) is not None

        if entity_type in {"ORGANIZATION", "COURT"}:
            prefixed_remainder = strip_identity_reference_prefix(normalized)
            if prefixed_remainder:
                normalized = prefixed_remainder
            if normalized in {
                "上诉人",
                "被上诉人",
                "原告",
                "被告",
                "申请人",
                "被申请人",
                "不服",
                *self.ORGANIZATION_NOISE_TERMS,
            }:
                return False
            if is_identity_reference_term(normalized) or is_position_title(normalized):
                return False
            if has_identity_reference_prefix(normalized):
                return False
            if is_generic_organization_term(normalized) or normalized in GENERIC_ORGANIZATION_TERMS:
                return False
            if is_probable_person(normalized) and not is_org_like_text(normalized):
                return False
            if len(normalized) > 12 and any(token in normalized for token in self.ORGANIZATION_PROSE_NOISE):
                return False
            if len(normalized) > 16 and any(token in normalized for token in ("的", "是", "至", "了")):
                return False
            return True

        return True

    def _iter_segments(self, text: str, *, max_chars: int = 450) -> Iterable[tuple[str, int]]:
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            line_start = cursor
            cursor += len(raw_line)
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            if len(line) <= max_chars:
                yield line, line_start
                continue
            start = 0
            while start < len(line):
                end = min(len(line), start + max_chars)
                if end < len(line):
                    split_at = max(
                        line.rfind("。", start, end),
                        line.rfind("；", start, end),
                        line.rfind("，", start, end),
                        line.rfind(" ", start, end),
                    )
                    if split_at > start + 40:
                        end = split_at + 1
                yield line[start:end], line_start + start
                start = end

    def _map_prediction_type(self, item: dict) -> Optional[str]:
        raw_label = str(item.get("entity_group") or item.get("entity") or "").strip()
        if not raw_label:
            return None
        label = raw_label.split("-", maxsplit=1)[-1].upper()
        return self.LABEL_MAP.get(label)

    def _fallback_extract(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        results.extend(self._extract_orgs(text))
        results.extend(self._extract_person_contexts(text))
        results.extend(self._extract_locations(text))
        results.extend(self._extract_dates(text))
        return deduplicate_results(item for item in results if item is not None)

    def _extract_orgs(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in ORG_PATTERN.finditer(text):
            entity_type = "COURT" if "法院" in match.group() else "ORGANIZATION"
            normalized_span = self._normalize_model_span(text, match.start(), match.end(), entity_type)
            if normalized_span is None:
                continue
            start, end, _ = normalized_span
            result = make_entity(
                text=text,
                start=start,
                end=end,
                entity_type=entity_type,
                source=self.source_name,
                score=0.8,
                metadata={
                    "source": "chinese_ner_service",
                    "model": self.model_id,
                    "backend": self.backend,
                    "fallback": True,
                    "ner_label": "ORG",
                },
            )
            if result:
                results.append(result)
        return results

    def _extract_person_contexts(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in self.PERSON_CONTEXT.finditer(text):
            person = match.group("person")
            if not is_probable_person(person):
                continue
            start, end = match.span("person")
            result = make_entity(
                text=text,
                start=start,
                end=end,
                entity_type="PERSON",
                source=self.source_name,
                score=0.78,
                metadata={
                    "source": "chinese_ner_service",
                    "model": self.model_id,
                    "backend": self.backend,
                    "fallback": True,
                    "ner_label": "PER",
                },
            )
            if result:
                results.append(result)
        return results

    def _extract_locations(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in self.LOCATION_CONTEXT.finditer(text):
            value = match.group("location").strip(" ：:，,；;。")
            if not value or len(value) < 4:
                continue
            value = re.split(r"(?=\s*(?:法定代表人|联系人|电话|开户行|账号|甲方|乙方)\s*[:：])", value, maxsplit=1)[0].strip()
            local = text.find(value, match.start("location"), match.end("location"))
            if local < 0:
                continue
            result = make_entity(
                text=text,
                start=local,
                end=local + len(value),
                entity_type="ADDRESS",
                source=self.source_name,
                score=0.74,
                metadata={
                    "source": "chinese_ner_service",
                    "model": self.model_id,
                    "backend": self.backend,
                    "fallback": True,
                    "ner_label": "LOC",
                },
            )
            if result:
                results.append(result)
        return results

    def _extract_dates(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in DATE_PATTERN.finditer(text):
            result = make_entity(
                text=text,
                start=match.start(),
                end=match.end(),
                entity_type="DATE",
                source=self.source_name,
                score=0.72,
                metadata={
                    "source": "chinese_ner_service",
                    "model": self.model_id,
                    "backend": self.backend,
                    "fallback": True,
                    "ner_label": "TIME",
                },
            )
            if result:
                results.append(result)
        return results

    def unload(self) -> None:
        self._pipeline = None
        release_runtime_memory()
