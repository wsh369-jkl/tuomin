"""Chinese UIE-style extraction service for high-quality low-memory mode."""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, List, Optional

from app.core.config import settings
from app.core.recognizer_base import RecognizerResult
from app.rules.subject_admission_gate import SubjectAdmissionGate
from app.services.lowmem_entity_utils import (
    GENERIC_ORGANIZATION_TERMS,
    NON_ENTITY_ROLE_TERMS,
    ORG_PATTERN,
    ORG_SUFFIX_PATTERN,
    clean_candidate_text,
    deduplicate_results,
    find_value_span,
    has_identity_reference_prefix,
    infer_semantic_type,
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
from app.services.legal_text_segmenter import build_legal_text_segment_metadata
from app.services.legal_text_segmenter import iter_legal_text_segments


class ChineseUIEService:
    """Schema-guided Chinese information extraction with deterministic fallback.

    The actual UIE backend is optional. The fallback keeps the workflow stable when
    heavyweight inference libraries are not installed, while preserving the same
    output contract and local model readiness metadata.
    """

    SCHEMA = [
        "自然人姓名",
        "公司名称",
        "机构名称",
        "法院名称",
        "仲裁机构",
        "地址",
        "项目名称",
        "工程名称",
        "开户行",
        "账户名称",
        "收款单位",
        "付款单位",
        "法定代表人",
        "负责人",
        "联系人",
        "委托代理人",
        "签署人",
        "合同编号",
        "案号",
        "简称",
    ]

    LABELS = {
        "ORGANIZATION": [
            "甲方",
            "乙方",
            "丙方",
            "申请人",
            "被申请人",
            "原告",
            "被告",
            "第三人",
            "收款单位",
            "付款单位",
            "账户名称",
            "户名",
        ],
        "PERSON": ["自然人姓名", "法定代表人", "负责人", "联系人", "委托诉讼代理人", "诉讼代理人", "委托代理人", "代理人", "签署人"],
        "ADDRESS": ["身份证住址", "经常居住地", "户籍地址", "户籍地", "现住址", "住址", "住所", "住所地", "注册地址", "送达地址", "通讯地址", "联系地址", "地址"],
        "PROJECT": ["项目名称", "工程名称", "标段名称"],
        "BANK_NAME": ["开户行", "开户银行", "银行名称"],
        "CONTRACT_NO": ["合同编号", "合同号", "协议编号"],
        "CASE_NO": ["案号", "受理案号", "执行案号", "一审案号", "二审案号", "原审案号"],
    }
    PROCEDURAL_PREFIX = re.compile(
        r"^(?:上诉人|被上诉人|原审原告|原审被告|一审原告|一审被告|"
        r"申请人|被申请人|原告|被告|第三人|不服|此致|根据)"
        r"(?:（[^）]*）|\([^)]*\))?\s*[:：，,、]?"
    )
    TRAILING_PUNCTUATION = " \t\r\n，,。；;：:、"
    ADDRESS_TOKENS = ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室", "自治区", "自治州", "自治县", "旗", "盟")
    DATE_LIKE = re.compile(r"\d{4}[./年-]\d{1,2}(?:[./月-]\d{1,2})?日?")
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
                *SCHEMA,
                *(label for labels in LABELS.values() for label in labels),
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
        "申请人",
        "被申请人",
        "原告",
        "被告",
        "第三人",
        "收款单位",
        "付款单位",
        "账户名称",
        "户名",
    }
    PARTY_NARRATIVE_TAIL = re.compile(
        r"(?:"
        r"(?:系|为|作为|是)本(?:合同|协议|项目)"
        r"|(?:系|为|作为|是)(?:签约方|供货方|采购方|承包方|发包方|卖方|买方|出租方|承租方|服务方|"
        r"履约方|付款方|收款方|合同主体|履约主体|付款主体|收款主体|合作方)"
        r"|(?:负责|承担|继续|签署|签订|履行|付款|收款|供货|交付|结算|对账|盖章|落款)"
        r")"
    )
    HF_LABEL_MAP = {
        "NAME": "PERSON",
        "PER": "PERSON",
        "PERSON": "PERSON",
        "COMPANY": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "GOVERNMENT": "ORGANIZATION",
        "ADDRESS": "ADDRESS",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "SCENE": "LOCATION",
        "POSITION": "POSITION",
    }
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
        "公司",
        "银行",
        "法院",
        "人民法院",
        "集团",
        "机构",
        "单位",
        "个人银行",
        "国家",
        "法人代表",
    }
    NON_VALUE_TERMS.update(NON_ENTITY_ROLE_TERMS)
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

    def __init__(self, model_id: str, backend: str = "modelscope") -> None:
        self.model_id = model_id
        self.backend = backend
        self.asset = build_model_asset(model_id, role="primary_ie", backend=backend)
        self.backend_available = False
        self.backend_error: Optional[str] = None
        self._pipeline = None
        self.last_extract_metadata: dict[str, object] = {}

    @property
    def ready(self) -> bool:
        return self.asset.installed

    def extract(self, text: str) -> List[RecognizerResult]:
        sanitized_text, index_map = sanitize_recognition_text(text)
        working_text = sanitized_text or text
        self.last_extract_metadata = {
            "uie_input_length": len(text or ""),
            "uie_working_text_length": len(working_text or ""),
            **{
                f"uie_{key}": value
                for key, value in build_legal_text_segment_metadata(
                    working_text,
                    max_chars=420,
                    overlap_chars=80,
                    min_split_chars=60,
                ).items()
            },
        }
        model_results: list[RecognizerResult] = []
        if self.backend.lower() == "deterministic":
            self.backend_available = True
            self.backend_error = None
        if self.asset.installed and self.backend.lower() in {
            "transformers",
            "transformers_token_classification",
            "hf_token_classification",
            "token_classification",
        }:
            try:
                model_results = self._extract_with_transformers_token_classification(working_text)
                self.backend_available = True
                self.backend_error = None
            except Exception as exc:
                self.backend_available = False
                self.backend_error = f"{type(exc).__name__}: {exc}"
        if self.asset.installed and self.backend.lower() in {"modelscope", "modelscope_siamese_uie", "siamese_uie"}:
            try:
                model_results = self._extract_with_modelscope(working_text)
                self.backend_available = True
                self.backend_error = None
            except Exception as exc:
                self.backend_available = False
                self.backend_error = f"{type(exc).__name__}: {exc}"

        fallback_results = self._fallback_extract(working_text)
        remapped = remap_results_to_source([*model_results, *fallback_results], text, index_map)
        self.last_extract_metadata.update(
            {
                "uie_model_result_count": len(model_results),
                "uie_fallback_result_count": len(fallback_results),
                "uie_remapped_result_count": len(remapped),
            }
        )
        return deduplicate_results(remapped)

    def _extract_with_modelscope(self, text: str) -> List[RecognizerResult]:
        if self._pipeline is None:
            self._guard_modelscope_runtime_requirements()
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
            except Exception as exc:
                raise RuntimeError("modelscope_uie_runtime_not_installed") from exc

            if self.asset.path is None:
                return []
            self._pipeline = pipeline(Tasks.siamese_uie, str(self.asset.path), model_revision="master")

        results: list[RecognizerResult] = []
        for segment, offset in self._iter_segments(text):
            payload = self._pipeline(input=segment, schema=self._modelscope_schema())
            for label, value, score in self._iter_modelscope_candidates(payload):
                cleaned = clean_candidate_text(value)
                if not cleaned:
                    continue
                cleaned = self._normalize_label_candidate(cleaned, self._expected_label_entity_type(label), label)
                if not cleaned:
                    continue
                span = find_value_span(text, cleaned, search_start=offset, search_end=offset + len(segment))
                if span is None:
                    span = find_value_span(text, cleaned)
                if span is None:
                    continue
                entity_type = self._map_schema_type(label, cleaned)
                result = make_entity(
                    text=text,
                    start=span[0],
                    end=span[1],
                    entity_type=entity_type,
                    source="uie",
                    score=score,
                    metadata={
                        "source": "chinese_uie_service",
                        "model": self.model_id,
                        "schema": label,
                        "backend": self.backend,
                        "fallback": False,
                    },
                )
                if result:
                    results.append(result)
        return results

    def _guard_modelscope_runtime_requirements(self) -> None:
        """Avoid ModelScope remote requirements downgrading the Qwen MLX runtime."""
        if settings.ALLOW_UNSAFE_MODELSCOPE_UIE_RUNTIME:
            return
        if self.asset.path is None:
            return
        requirements_path = self.asset.path / "requirements.txt"
        if not requirements_path.exists():
            return
        requirements = requirements_path.read_text(encoding="utf-8", errors="ignore").lower()
        if "transformers==4." in requirements or "transformers<5" in requirements:
            raise RuntimeError(
                "modelscope_uie_runtime_blocked: remote requirements downgrade transformers "
                "and conflict with mlx-lm review runtime"
            )

    def _extract_with_transformers_token_classification(self, text: str) -> List[RecognizerResult]:
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
                entity_type = self._map_hf_prediction_type(item)
                if not entity_type:
                    continue
                start = offset + int(item.get("start") or 0)
                end = offset + int(item.get("end") or 0)
                if start >= end:
                    continue
                normalized_span = self._normalize_hf_span(text, start, end, entity_type)
                if normalized_span is None:
                    continue
                start, end, _value, resolved_type = normalized_span
                result = make_entity(
                    text=text,
                    start=start,
                    end=end,
                    entity_type=resolved_type,
                    source="uie",
                    score=float(item.get("score") or 0.8),
                    metadata={
                        "source": "chinese_uie_service",
                        "model": self.model_id,
                        "schema": item.get("entity_group") or item.get("entity"),
                        "backend": self.backend,
                        "fallback": False,
                    },
                )
                if result:
                    results.append(result)
        return results

    def _iter_segments(self, text: str, *, max_chars: int = 420) -> Iterable[tuple[str, int]]:
        yield from iter_legal_text_segments(text, max_chars=max_chars, overlap_chars=80, min_split_chars=60)

    def _modelscope_schema(self) -> dict[str, None]:
        return {label: None for label in self.SCHEMA}

    def _iter_modelscope_candidates(self, payload: Any, label: str = "") -> Iterator[tuple[str, str, float]]:
        if isinstance(payload, dict):
            candidate_text = self._candidate_text_from_payload(payload)
            if candidate_text:
                yield label or str(payload.get("type") or payload.get("label") or "实体"), candidate_text, self._score_from_payload(payload)
            for key, value in payload.items():
                next_label = str(key) if key in self.SCHEMA or not label else label
                yield from self._iter_modelscope_candidates(value, next_label)
            return
        if isinstance(payload, list):
            for item in payload:
                yield from self._iter_modelscope_candidates(item, label)

    @staticmethod
    def _candidate_text_from_payload(payload: dict) -> str:
        for key in ("text", "span", "entity", "word", "value"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _score_from_payload(payload: dict) -> float:
        for key in ("probability", "prob", "score", "confidence"):
            try:
                return float(payload.get(key))
            except (TypeError, ValueError):
                continue
        return 0.82

    def _map_schema_type(self, label: str, value: str) -> str:
        normalized_label = label or ""
        for entity_type, labels in self.LABELS.items():
            if normalized_label in labels or any(item in normalized_label for item in labels):
                if entity_type in {"CONTRACT_NO", "CASE_NO", "PROJECT", "BANK_NAME", "ADDRESS"}:
                    return entity_type
                return infer_semantic_type(value, normalized_label)
        if normalized_label in {"公司名称", "机构名称", "法院名称", "仲裁机构"}:
            return "COURT" if "法院" in value else "ORGANIZATION"
        if normalized_label == "自然人姓名":
            return "PERSON"
        if normalized_label == "简称":
            return "ALIAS"
        return infer_semantic_type(value, normalized_label)

    def _map_hf_prediction_type(self, item: dict) -> str:
        raw_label = str(item.get("entity_group") or item.get("entity") or "").strip()
        if not raw_label:
            return ""
        label = raw_label.split("-", maxsplit=1)[-1].upper()
        return self.HF_LABEL_MAP.get(label, "")

    def _normalize_hf_span(
        self,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> Optional[tuple[int, int, str, str]]:
        value = text[start:end]
        leading_trimmed = len(value) - len(value.lstrip())
        if leading_trimmed:
            start += leading_trimmed
            value = value.lstrip()

        value = value.strip(" \t\r\n，,。；;：:、…“”\"'`")
        end = start + len(value)

        if entity_type in {"ORGANIZATION", "COURT"}:
            normalized_span = self._normalize_org_span(text, start, end, entity_type)
            if normalized_span is None:
                return None
            start, end = normalized_span
            value = text[start:end]
            entity_type = "COURT" if "法院" in value else "ORGANIZATION"

        normalized = re.sub(r"[\s，,。；;：:、]+", "", value)
        if len(normalized) < 2:
            return None
        if not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", normalized):
            return None
        if normalized in self.NON_VALUE_TERMS:
            return None
        if entity_type == "PERSON":
            if not SubjectAdmissionGate.passes_subject_shape("PERSON", normalized)[0]:
                return None
        if entity_type in {"ADDRESS", "LOCATION"}:
            if self.DATE_LIKE.fullmatch(normalized) or normalized in self.NON_VALUE_TERMS:
                return None
            if entity_type == "ADDRESS" and not any(token in normalized for token in self.ADDRESS_TOKENS):
                return None
            if not SubjectAdmissionGate.passes_subject_shape("LOCATION", normalized)[0]:
                return None
        if entity_type in {"ORGANIZATION", "COURT"}:
            gate_type = "GOVERNMENT" if entity_type == "COURT" else "ORGANIZATION"
            if not SubjectAdmissionGate.passes_subject_shape(gate_type, normalized)[0]:
                return None
        if entity_type == "POSITION" and normalized in self.NON_VALUE_TERMS:
            return None
        if entity_type == "POSITION" and is_identity_reference_term(normalized):
            return None
        return start, end, value, entity_type

    def _fallback_extract(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        results.extend(self._extract_label_values(text))
        results.extend(self._extract_orgs(text))
        results.extend(self._extract_definition_aliases(text))
        return deduplicate_results(item for item in results if item is not None)

    def _extract_label_values(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for entity_type, labels in self.LABELS.items():
            label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
            pattern = re.compile(
                rf"(?P<label>{label_pattern})\s*(?:[:：]\s*|\n\s*|\s{{2,}})"
                rf"(?P<value>[^\n\r；;。]{{2,120}})"
            )
            for match in pattern.finditer(text):
                raw_value = clean_candidate_text(match.group("value"))
                if not raw_value:
                    continue
                raw_value = self._normalize_label_candidate(raw_value, entity_type, match.group("label"))
                if not raw_value:
                    continue
                raw_value = self._trim_tail_label(raw_value)
                if not raw_value:
                    continue
                if not self._is_valid_fallback_label_value(raw_value, entity_type):
                    continue
                span = find_value_span(text, raw_value, search_start=match.start("value"), search_end=match.end("value"))
                if span is None:
                    continue
                resolved_type = infer_semantic_type(raw_value, match.group("label"))
                if entity_type in {"CONTRACT_NO", "CASE_NO", "PROJECT", "BANK_NAME", "ADDRESS"}:
                    resolved_type = entity_type
                result = make_entity(
                    text=text,
                    start=span[0],
                    end=span[1],
                    entity_type=resolved_type,
                    source="uie",
                    score=0.86,
                    metadata={
                        "source": "chinese_uie_service",
                        "model": self.model_id,
                        "schema": match.group("label"),
                        "backend": self.backend,
                        "fallback": True,
                    },
                )
                if result:
                    results.append(result)
        return results

    def _extract_orgs(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in ORG_PATTERN.finditer(text):
            entity_type = "COURT" if "法院" in match.group() else "ORGANIZATION"
            normalized_span = self._normalize_org_span(text, match.start(), match.end(), entity_type)
            if normalized_span is None:
                continue
            start, end = normalized_span
            result = make_entity(
                text=text,
                start=start,
                end=end,
                entity_type=entity_type,
                source="uie",
                score=0.78,
                metadata={
                    "source": "chinese_uie_service",
                    "model": self.model_id,
                    "schema": "机构名称",
                    "backend": self.backend,
                    "fallback": True,
                },
            )
            if result:
                results.append(result)
        return results

    def _is_valid_fallback_label_value(self, value: str, entity_type: str) -> bool:
        normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", value or "")
        if not normalized or normalized in self.NON_VALUE_TERMS:
            return False
        if self.DATE_LIKE.fullmatch(normalized):
            return False
        if entity_type == "PERSON":
            return SubjectAdmissionGate.passes_subject_shape("PERSON", normalized)[0]
        if entity_type in {"ORGANIZATION", "COURT"}:
            gate_type = "GOVERNMENT" if entity_type == "COURT" else "ORGANIZATION"
            return SubjectAdmissionGate.passes_subject_shape(gate_type, normalized)[0]
        if entity_type == "ADDRESS":
            return any(token in normalized for token in self.ADDRESS_TOKENS) and SubjectAdmissionGate.passes_subject_shape("LOCATION", normalized)[0]
        return True

    def _expected_label_entity_type(self, label: str) -> str:
        normalized_label = str(label or "").strip()
        for entity_type, labels in self.LABELS.items():
            if normalized_label in labels or any(item in normalized_label for item in labels):
                return entity_type
        if normalized_label in {"公司名称", "机构名称"}:
            return "ORGANIZATION"
        if normalized_label in {"法院名称", "仲裁机构"}:
            return "COURT"
        if normalized_label == "自然人姓名":
            return "PERSON"
        if normalized_label == "简称":
            return "ALIAS"
        return infer_semantic_type("", normalized_label)

    def _normalize_label_candidate(self, value: str, entity_type: str, label: str) -> str:
        cleaned = value.strip(self.TRAILING_PUNCTUATION)
        if entity_type in {"ORGANIZATION", "COURT", "PROJECT"}:
            cleaned = self._strip_inline_alias_segments(cleaned)
            cleaned = self._truncate_at_following_field_label(cleaned, entity_type, label)
            cleaned = self._truncate_party_like_narrative_tail(cleaned, entity_type, label)
            while True:
                stripped = self.INLINE_ALIAS_TAIL.sub("", cleaned).strip(self.TRAILING_PUNCTUATION)
                if stripped == cleaned:
                    break
                cleaned = stripped
        return cleaned.strip(self.TRAILING_PUNCTUATION)

    def _strip_inline_alias_segments(self, value: str) -> str:
        cleaned = value
        while True:
            stripped = self.INLINE_ALIAS_SEGMENT.sub("", cleaned).strip(self.TRAILING_PUNCTUATION)
            if stripped == cleaned:
                return stripped
            cleaned = stripped

    def _truncate_at_following_field_label(self, value: str, entity_type: str, label: str) -> str:
        if entity_type not in {"ORGANIZATION", "COURT", "PROJECT"}:
            return value
        best = len(value)
        for token in self.FOLLOWING_FIELD_LABELS:
            if token == label:
                continue
            index = value.find(token, 1)
            if index <= 0:
                continue
            best = min(best, index)
        truncated = value[:best].rstrip(self.TRAILING_PUNCTUATION)
        return truncated or value

    def _truncate_party_like_narrative_tail(self, value: str, entity_type: str, label: str) -> str:
        if entity_type not in {"ORGANIZATION", "COURT"}:
            return value
        if label not in self.PARTY_LIKE_LABELS:
            return value
        org_match = ORG_PATTERN.search(value)
        if org_match:
            return org_match.group().strip()
        narrative_match = self.PARTY_NARRATIVE_TAIL.search(value)
        if narrative_match and narrative_match.start() >= 2:
            candidate = value[: narrative_match.start()].rstrip(self.TRAILING_PUNCTUATION)
            if candidate:
                return candidate
        compact = re.sub(r"\s+", "", value)
        if looks_like_organization_short_name(compact):
            return compact
        return value

    def _normalize_org_span(
        self,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> Optional[tuple[int, int]]:
        value = text[start:end]
        leading_trimmed = len(value) - len(value.lstrip())
        if leading_trimmed:
            start += leading_trimmed
            value = value.lstrip()
        value = value.strip(" \t\r\n，,。；;：:、…“”\"'`")
        end = start + len(value)

        if entity_type in {"ORGANIZATION", "COURT"}:
            if SubjectAdmissionGate.is_non_subject_expression(value):
                return None
            org_trimmed = self._trim_to_semantic_org_value(value)
            if org_trimmed != value:
                delta = len(value) - len(org_trimmed)
                start += delta
                value = org_trimmed
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

            prefixed_remainder = strip_identity_reference_prefix(value)
            if prefixed_remainder:
                return None

        normalized = re.sub(r"[\s，,。；;：:、]+", "", value)
        if not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", normalized):
            return None
        if normalized in {
            "上诉人",
            "被上诉人",
            "原告",
            "被告",
            "申请人",
            "被申请人",
            "不服",
            *self.NON_VALUE_TERMS,
        }:
            return None
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return None
        if has_identity_reference_prefix(normalized):
            return None
        if normalized in GENERIC_ORGANIZATION_TERMS:
            return None
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", normalized) and not is_org_like_text(normalized):
            return None
        if len(normalized) > 12 and any(token in normalized for token in self.ORGANIZATION_PROSE_NOISE):
            return None
        if len(normalized) > 16 and any(token in normalized for token in ("的", "是", "至", "了")):
            return None
        gate_type = "GOVERNMENT" if entity_type == "COURT" else "ORGANIZATION"
        return (start, end) if len(normalized) >= 2 and SubjectAdmissionGate.passes_subject_shape(gate_type, normalized)[0] else None

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

    def _extract_definition_aliases(self, text: str) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        pattern = re.compile(
            rf"(?P<full>{ORG_PATTERN.pattern})"
            r"\s*[（(]?\s*(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?"
            r"(?P<alias>[\u4e00-\u9fa5A-Za-z0-9]{2,20})[”\"'‘’]?\s*[）)]?"
        )
        for match in pattern.finditer(text):
            full_span = match.span("full")
            alias_span = match.span("alias")
            full = make_entity(
                text=text,
                start=full_span[0],
                end=full_span[1],
                entity_type="ORGANIZATION",
                source="uie",
                score=0.9,
                metadata={
                    "source": "chinese_uie_service",
                    "schema": "简称",
                    "alias": match.group("alias"),
                    "fallback": True,
                },
            )
            alias = make_entity(
                text=text,
                start=alias_span[0],
                end=alias_span[1],
                entity_type="ALIAS",
                source="uie",
                score=0.9,
                metadata={
                    "source": "chinese_uie_service",
                    "schema": "简称",
                    "canonical": match.group("full"),
                    "fallback": True,
                },
            )
            if full:
                results.append(full)
            if alias:
                results.append(alias)
        return results

    @staticmethod
    def _trim_tail_label(value: str) -> str:
        return re.split(
            r"(?=\s*(?:甲方|乙方|丙方|上诉人|被上诉人|原审原告|原审被告|"
            r"法定代表人|负责人|联系人|委托诉讼代理人|诉讼代理人|委托代理人|代理人|"
            r"身份证住址|经常居住地|户籍地址|户籍地|现住址|住址|住所|地址|开户行|户名|账号|账户)\s*[:：])",
            value,
            maxsplit=1,
        )[0].strip()

    def unload(self) -> None:
        self._pipeline = None
        release_runtime_memory()
