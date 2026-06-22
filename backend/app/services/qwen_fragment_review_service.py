"""Fragment-level Qwen review service for high-quality low-memory mode."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import site
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

from app.core.config import settings
from app.core.recognizer_base import RecognizerResult
from app.rules.default_subject_policy import canonical_default_entity_type
from app.services.lowmem_entity_utils import (
    NON_ENTITY_ROLE_TERMS,
    ORG_PATTERN,
    ORG_SUFFIX_TERMS,
    OFFICIAL_INSTITUTION_PATTERN,
    clean_candidate_text,
    expand_org_span_with_left_region_prefix,
    expand_subject_span_to_containing_shape,
    find_short_org_prefix_before_non_subject_boundary,
    find_value_span,
    infer_semantic_type,
    is_generic_organization_term,
    is_government_institution_text,
    is_identity_reference_term,
    is_non_subject_action_or_function_term,
    is_non_subject_generic_org_reference,
    is_official_institution_text,
    is_org_like_text,
    is_position_title,
    is_probable_person,
    is_weak_function_stripped_org,
    looks_like_organization_short_name,
    normalize_entity_text,
    make_entity,
    strip_identity_reference_prefix,
    strip_leading_subject_function_words,
    subject_noun_gate,
)
from app.services.lowmem_memory import release_runtime_memory
from app.services.lowmem_model_assets import build_model_asset
from app.services.risk_snippet_scheduler import RiskSnippet

logger = logging.getLogger(__name__)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(numerator) / float(denominator))), 4)


REVIEW_CONTEXT_CHAR_RATIO = 2.2
REVIEW_PROMPT_SAFETY_CHARS = 1200
REVIEW_DECISION_SOURCE_NAMES = {
    "qwen_fragment_review",
    "qwen_entity_decision",
    "review_deterministic_decision",
}
RULE_ORG_REVIEW_NON_SUBJECT_PREFIXES = (
    "相关",
    "前述",
    "上述",
    "涉案",
    "本案",
    "本合同",
    "本协议",
    "该",
    "其",
    "另有",
    "还有",
    "其中",
)
RULE_ORG_REVIEW_NON_SUBJECT_TOKENS = (
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
    "处理",
    "办理",
    "提交",
    "说明",
    "确认",
    "完成",
    "涉及",
    "利用",
    "绑定",
)
RULE_ORG_REVIEW_WEAK_SUFFIXES = ("学校", "大学", "学院", "医院", "协会", "学会")
RULE_ORG_REVIEW_STRONG_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "集团有限公司",
    "有限公司",
    "分公司",
    "子公司",
    "集团",
    "公司",
    "商行",
    "合作社",
    "联合社",
    "联社",
    "工作室",
    "经营部",
    "门市部",
    "营业部",
    "办事处",
    "基金会",
    "联合会",
    "俱乐部",
    "实验室",
    "门诊部",
    "诊所",
    "律师事务所",
    "会计师事务所",
    "研究院",
    "研究所",
    "服务中心",
    "技术中心",
)
GENERIC_GOVERNMENT_REVIEW_REFERENCES = {
    "法院",
    "人民法院",
    "一审法院",
    "二审法院",
    "原审法院",
    "审法院",
    "本院",
    "贵院",
    "该院",
    "上级法院",
    "下级法院",
    "执行法院",
}


def _review_prompt_char_budget(*, quality_gate: bool = False, allow_thinking: bool = False) -> int:
    num_ctx = int(getattr(settings, "REVIEW_NUM_CTX", 4096) or 4096)
    max_tokens = int(
        getattr(
            settings,
            "REVIEW_THINKING_MAX_TOKENS" if allow_thinking else "REVIEW_MAX_TOKENS",
            384,
        )
        or 384
    )
    available_tokens = max(512, num_ctx - max_tokens - 256)
    estimated = int(available_tokens * REVIEW_CONTEXT_CHAR_RATIO) - REVIEW_PROMPT_SAFETY_CHARS
    return max(1200, min(3000, estimated))


def _compact_prompt_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    if limit <= 240:
        return text[:limit]
    head = max(80, int(limit * 0.68))
    tail = max(60, limit - head - 36)
    return f"{text[:head]}\n...[已按4096上下文预算截断]...\n{text[-tail:]}"

@dataclass(frozen=True)
class ReviewRuntime:
    backend: str
    model_id: str
    asset: object | None = None
    fallback: bool = False
    tier: str = "local"


@dataclass
class ReviewResult:
    entities: List[RecognizerResult] = field(default_factory=list)
    rejected_entities: List[Dict] = field(default_factory=list)
    model_used: bool = False
    fallback_used: bool = False
    model_name: Optional[str] = None
    review_backend: Optional[str] = None
    requires_manual_review: bool = False
    error: Optional[str] = None
    raw_candidate_count: int = 0
    parsed_snippet_count: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


def _first_surface_text(card: dict) -> str:
    for surface in card.get("surfaces") or []:
        if isinstance(surface, dict):
            text = str(surface.get("text") or "").strip()
        else:
            text = str(surface or "").strip()
        if text:
            return text
    for occurrence in card.get("occurrences") or []:
        if isinstance(occurrence, dict):
            text = str(occurrence.get("text") or "").strip()
            if text:
                return text
    return ""


class QwenFragmentReviewService:
    """Run small Qwen only on risky snippets.

    Backends are optional. Missing runtime packages never interrupt the main
    desensitization flow; the recognizer records manual-review metadata instead.
    """

    ALLOWED_TYPES = {
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "GOVERNMENT",
    }
    TYPE_ALIASES = {
        "ACCOUNT_NAME": "ORGANIZATION",
        "BANK_NAME": "GOVERNMENT",
        "COMPANY": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "ADDRESS": "LOCATION",
        "COURT": "GOVERNMENT",
        "GOVERNMENT_AGENCY": "GOVERNMENT",
        "PERSON_NAME": "PERSON",
        "LEGAL_REPRESENTATIVE": "PERSON",
        "CONTACT_PERSON": "PERSON",
        "SIGNATORY": "PERSON",
    }
    NON_ENTITY_TERMS = {
        "国家",
        "法定",
        "代表",
        "代表人",
        "法定代表人",
        "法人",
        "法人代表",
        "负责",
        "负责人",
        "联系",
        "联系人",
        "地址",
        "住所",
        "住址",
        "通过",
        "通过了",
        "经过",
        "根据",
        "依据",
        "提交",
        "确认",
        "要求",
        "请求",
        "判令",
        "承担",
        "负责",
        "继续",
        "履行",
        "履约",
        "结算",
        "付款",
        "收款",
        "交付",
        "配合",
        "联系",
        "协商",
        "双方",
        "各方",
        "一方",
        "另一方",
        "该公司",
        "法院",
        "人民法院",
        "一审法院",
        "二审法院",
        "原审法院",
        "本院",
        "贵院",
        *NON_ENTITY_ROLE_TERMS,
    }

    PROMPT_TEMPLATE = """你是中文合同/法律文书脱敏复核模型。只输出 JSON，不要解释。
目标：看片段原文，完成两件事：
A. entities：输出片段中应脱敏的真实实体；即使“已识别实体”已经识别正确，也要再次输出，表示确认。
B. rejects：输出“已识别实体”里明显不是敏感实体的候选。
C. entity_decisions：逐项裁决“已识别实体”。如果候选以“通过/经过/根据/依据/要求/通知/由/向/对/与/和”等功能词、动作词、连接词开头，必须排除前面的功能词；如果后半段是完整主体，用 trim_to 输出去掉功能词后的真实主体；如果后半段不是完整主体或只是泛称，使用 reject；如果动作词只出现在主体名称中间，且整体是完整公司/机构/政府/地名，保留该主体；如果一个候选吞了多个并列主体，必须用 split_into 拆开；如果候选只是普通词/动词/代词，使用 reject。
允许类型只能是：PERSON, ORGANIZATION, LOCATION, GOVERNMENT。
重点补漏：复杂中文人名、少数民族/带中点姓名、上诉人/被上诉人/原审原告/原审被告/申请人/被申请人对应的真实主体、委托诉讼代理人、股东/实际控制人、住址/现住址/户籍地/身份证住址/经常居住地、法院/仲裁机构、公司简称关系。法院、检察院、公安、政府、仲裁委员会等统一输出 GOVERNMENT；地址统一输出 LOCATION。开户行、户名、账号、案号、合同编号、项目名称、金额、电话、证号、银行卡号、信用代码都不是主体分类，不要输出为实体。
额外强调：即使全文里从头到尾只出现简称、没有出现完整公司名，只要该 2-6 字简称在片段里承担签约、履约、结算、付款、收款、供货、交付、施工、对账、盖章、落款、对接等组织主体动作，或与另一组织主体并列承担合同义务，也应优先识别为 ORGANIZATION，不要因为缺少“公司/集团/中心”等后缀而漏掉。
当 PERSON 与 ORGANIZATION 难以区分时，如果该词在片段里更像合同/诉讼/交易主体而不是自然人，例如出现在“由X继续履约”“X负责结算”“X向乙方付款”“X与Y签订协议”“X继续供货”“X承担责任”这类语境中，优先输出 ORGANIZATION。
不要把这些普通词或泛称当实体：国家、中华人民共和国、人民共和国、法定代表人、法定代理人、法人代表、负责人、联系人、委托代理人、委托诉讼代理人、诉讼代理人、代理人、控告人、被控告人、举报人、申诉人、起诉人、自诉人、技术人员、总经理、经理、董事长、董事、监事、岗位、职位、职务、我中心、本院、贵院、法院、人民法院、一审法院、二审法院、项目所在地人民法院、地址、合同期限、三个、五个。注意：“签订日期”这个标签不是实体，但具体日期值可以保留，不要放入 rejects。
硬性要求：
1. text 必须是片段原文里的连续字符串，一个字都不能编。
2. entities 放片段里的全部敏感实体，包括已经正确识别的实体、漏掉的实体、需要纠正类型的实体；不要因为已经识别过就输出空。
2.1. 对只有简称的组织主体，不要因为没有全称锚点就省略；只要片段语义足够支持，就直接输出该简称本身。
3. rejects 只放明显错误的已识别实体，例如：普通词“国家/法定代表人/法定代理人/控告人/地址/法院”、普通岗位词、章节标题、泛称“一审法院/二审法院/本院/法院/地址”、吞掉整句的案号/合同编号。
3.1. 对开头就是功能词/动作词的候选必须先杀掉前缀功能词，例如“通过北京某某有限公司”“依据某某委员会文件”“由某某分公司负责”这类候选，如果后半段是完整主体，使用 trim_to 输出“北京某某有限公司”“某某委员会”“某某分公司”；如果后半段不是完整主体或只是“公司/机构/材料”等泛称，使用 reject。
4. 地址要尽量输出完整地址；人名不要拆字；公司/政府机构要输出完整正式名称。
5. 如果片段包含甲乙方、诉讼主体、政府机构、地址、落款主体，entities 通常不应为空；只有账号/案号/合同编号/开户行/户名时可以为空。
6. 最多输出 24 个 entities、16 个 rejects；确实没有敏感实体或错误时才输出空数组。
片段类型：{snippet_type}
已识别实体（可能有错，不要盲信）：
{existing_entities}
片段开始：
{snippet_text}
片段结束。
JSON 格式：{{"entities":[{{"type":"ORGANIZATION","text":"某某公司","role":"甲方"}}],"rejects":[{{"type":"ORGANIZATION","text":"国家","reason":"普通词"}}],"entity_decisions":[{{"text":"通过某某公司","type":"ORGANIZATION","action":"trim_to","trim_to":"某某公司","reason":"去掉开头功能词"}}]}}
"""

    SHORT_ORG_ACTION_PATTERNS = (
        re.compile(
            r"(?:由|与|同|和|向|对)?(?P<subject>[\u4e00-\u9fa5A-Za-z0-9·]{2,8})"
            r"(?:继续履约|负责(?:履约|结算|交付|供货|施工|收款|付款|签约|执行)|"
            r"签订(?:补充)?(?:协议|合同)|提供(?:技术)?服务|承担(?:付款|结算|供货|施工|交付)?责任|"
            r"继续(?:结算|供货|施工)|办理结算|履行(?:付款|交付|供货|施工)?义务|"
            r"配合(?:交付|结算|付款|收款|供货|施工|对账|履约)|"
            r"协助(?:交付|结算|付款|收款|供货|施工|对账|履约)|"
            r"对接(?:付款|收款|交付|供货|施工|结算)|"
            r"承接(?:项目|施工|供货|交付|结算|履约)|"
            r"负责对账|盖章|签章|落款)"
        ),
        re.compile(
            r"(?P<subject>[\u4e00-\u9fa5A-Za-z0-9·]{2,8})"
            r"(?:向(?:甲方|乙方|丙方|对方|另一方)?付款|收款|付款|供货|交付|施工|签约|结算|对账|盖章|签章|落款|履约)"
        ),
    )
    SHORT_ORG_CONTEXT_PATTERNS = (
        re.compile(
            r"(?P<subject>[\u4e00-\u9fa5A-Za-z0-9·]{2,8})"
            r"(?:系|为|属(?:于)?|作为|视为|认定为)(?:涉案|合同|签约)?(?:公司|企业|集团|机构|单位|主体)"
        ),
        re.compile(
            r"(?P<subject>[\u4e00-\u9fa5A-Za-z0-9·]{2,8})"
            r"(?:公司|企业|集团|机构|单位|主体)"
        ),
    )
    SHORT_ORG_LIST_PATTERNS = (
        re.compile(
            r"(?P<label>"
            r"(?:本协议|本合同|本案|涉案|相关|关联|合作|甲方|乙方|丙方|前述|上述)?"
            r"[\u4e00-\u9fa5A-Za-z0-9]{0,10}"
            r"(?:公司|企业|集团|机构|单位|主体|合作方|签约方|合同主体|交易主体|项目主体)"
            r")"
            r"(?:分别为|包括|如下|名单为|为|:|：)"
            r"(?P<items>[^。\n\r；;]{4,80})"
        ),
        re.compile(
            r"(?P<label>"
            r"(?:涉及|包括|相关|合作|稳定合作的|拟签约|已签约|目标|关联)?"
            r"(?:客户|企业|主体|合作方|签约方|用电企业|商户|商行|单位)"
            r")"
            r"(?:分别为|包括|如下|名单为|为|有|:|：)"
            r"(?P<items>[^。\n\r；;]{4,120})"
        ),
    )
    SHORT_ORG_LIST_INVALID_ITEM_TOKENS = {
        "绑定",
        "环节",
        "利用",
        "办理",
        "进行",
        "开展",
        "负责",
        "通过",
        "以及",
        "或者",
        "如果",
        "其中",
        "账户",
        "账号",
        "开户",
        "收款",
        "付款",
        "结算",
        "履约",
        "交付",
        "供货",
        "施工",
        "对账",
        "签约",
        "协议",
        "合同",
        "资料",
        "流程",
        "内容",
        "情况",
        "事项",
        "信息",
        "记录",
    }
    SHORT_ORG_LIST_SPLIT_PATTERN = re.compile(r"(?:、|,|，|/|以及|及|和|与)")
    SHORT_ORG_LIST_SUFFIX_NOISE_PATTERN = re.compile(
        r"(?:"
        r"等(?:(?:多|数|若干)?家)?(?:公司|企业|集团|机构|单位|主体|合作方|签约方|客户|商户|用电企业)?"
        r"|一方|双方|三方|各方"
        r")+$"
    )
    NARRATIVE_PREFIX_PATTERN = re.compile(
        r"^(?:后续|另由|另行由|并由|将由|需由|应由|再由|由|对|向|与|同|和|经核实|经查明|经审查|另由|另行|后由)+"
    )
    SHORT_ORG_EXACT_NOISE_TERMS = {
        "合作",
        "涉案",
        "本案",
        "本协议",
        "本合同",
        "协议",
        "合同",
        "相关",
        "关联",
        "主体",
    }
    SHORT_ORG_PREFIX_TOKENS = (
        "经审查",
        "经查明",
        "经核实",
        "经确认",
        "经认定",
        "经调查",
        "后续由",
        "另行由",
        "另由",
        "并由",
        "将由",
        "需由",
        "应由",
        "再由",
        "仍由",
        "后续",
        "另行",
        "后由",
        "由",
        "对",
        "向",
        "与",
        "同",
        "和",
        "并",
        "仍",
    )
    SHORT_ORG_RELATION_MARKERS = (
        "认定为",
        "视为",
        "作为",
        "属于",
        "属",
        "为",
        "系",
    )
    SHORT_ORG_CONTEXT_SPLIT_TAIL_TOKENS = (
        "涉案",
        "案涉",
        "本案",
        "相关",
        "合同",
        "签约",
        "交易",
        "合作",
        "主体",
        "单位",
        "机构",
        "企业",
        "公司",
        "集团",
    )
    SHORT_ORG_TRAILING_CONTEXT_TOKENS = (
        "涉案",
        "案涉",
        "本案",
        "相关",
    )
    NARRATIVE_ACTION_TOKENS = (
        "继续履约",
        "负责履约",
        "负责结算",
        "负责交付",
        "负责供货",
        "负责施工",
        "负责收款",
        "负责付款",
        "负责签约",
        "负责执行",
        "签订补充协议",
        "签订协议",
        "签订合同",
        "提供服务",
        "承担责任",
        "继续结算",
        "继续供货",
        "继续施工",
        "办理结算",
        "履行义务",
        "配合交付",
        "配合结算",
        "协助交付",
        "协助结算",
        "对接付款",
        "对接收款",
        "承接项目",
        "承接施工",
        "盖章",
        "签章",
        "落款",
    )
    SHORT_ORG_NOISE_TOKENS = (
        "一方",
        "双方",
        "三方",
        "各方",
        "继续",
        "负责",
        "配合",
        "协助",
        "对接",
        "承接",
        "履约",
        "结算",
        "付款",
        "收款",
        "供货",
        "交付",
        "施工",
        "对账",
        "承担",
        "办理",
        "签订",
        "签署",
        "签约",
        "盖章",
        "签章",
        "落款",
        "设立",
        "成立",
        "随后",
    )
    SHORT_ORG_FALSE_TAIL_TOKENS = (
        "不是",
        "并非",
        "非",
        "并不是",
        "不属于",
        "不构成",
        "并不属于",
        "只是",
        "仅是",
    )

    def __init__(self) -> None:
        self.review_asset = build_model_asset(
            settings.REVIEW_MODEL,
            role="review",
            backend=settings.REVIEW_BACKEND,
            memory_tier="review_1_7b_4bit",
        )
        self.fallback_asset = build_model_asset(
            settings.REVIEW_MODEL_FALLBACK,
            role="review_fallback",
            backend=settings.REVIEW_MODEL_FALLBACK_BACKEND,
            memory_tier="review_0_8b_q4",
        )
        self.loaded = False
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_model_path: Optional[str] = None
        self._last_materialize_span_miss_count = 0
        self._last_materialize_gate_reject_count = 0

    @property
    def installed(self) -> bool:
        if self.review_asset.installed:
            return True
        if self.fallback_asset.installed and (
            not settings.is_high_quality_lowmem_mode()
            or settings.LOWMEM_ENABLE_LOCAL_REVIEW_FALLBACK
        ):
            return True
        return False

    @property
    def quality_gate_installed(self) -> bool:
        return False

    async def review(
        self,
        full_text: str,
        snippets: Iterable[RiskSnippet],
        *,
        existing_entities: Optional[Iterable[RecognizerResult]] = None,
        max_snippets: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> ReviewResult:
        if not settings.ENABLE_QWEN_REVIEW:
            return ReviewResult(requires_manual_review=False, error="review_disabled")

        fallback_used = False
        all_results: list[RecognizerResult] = []
        all_rejections: list[Dict] = []
        ledger_decisions: list[Dict[str, Any]] = []
        entity_decision_entity_count = 0
        any_model_used = False
        model_attempted = False
        last_error: Optional[str] = None
        raw_candidate_count = 0
        qwen_discovery_raw_candidate_count = 0
        qwen_discovery_materialized_entity_count = 0
        qwen_discovery_rejected_by_gate_count = 0
        qwen_discovery_span_miss_count = 0
        missing_candidate_materialized_entity_count = 0
        parsed_snippet_count = 0
        existing_list = list(existing_entities or [])
        global_deterministic_decisions = self._deterministic_review_entity_decisions(
            [self._entity_dict_for_review(item) for item in existing_list]
        )
        all_rejections.extend(global_deterministic_decisions["rejections"])
        review_limit = max_snippets or max(1, int(settings.MID_REVIEW_MAX_SNIPPETS or settings.REVIEW_MAX_SNIPPETS))
        scheduled_snippets = self._schedule_review_snippets(list(snippets), review_limit=review_limit)
        scheduled_ledger_count = sum(1 for snippet in scheduled_snippets if self._is_ledger_adjudication_snippet(snippet))
        scheduled_missing_candidate_count = sum(
            1 for snippet in scheduled_snippets if self._is_missing_candidate_review_snippet(snippet)
        )
        scheduled_standard_count = len(scheduled_snippets) - scheduled_ledger_count
        standard_runtime = self._select_review_runtime()
        if not scheduled_snippets:
            if global_deterministic_decisions["entities"]:
                all_results.extend(
                    self._materialize_candidates(
                        full_text,
                        {"entities": global_deterministic_decisions["entities"]},
                        RiskSnippet(
                            "final_subject_adjudication",
                            "final_subject:deterministic_global",
                            0,
                            len(full_text),
                            full_text,
                        ),
                        source_name="review_deterministic_decision",
                    )
                )
            return ReviewResult(
                entities=self._dedupe_results(all_results),
                rejected_entities=self._dedupe_rejections(all_rejections),
                requires_manual_review=False,
                error="no_standard_review_snippets",
                metadata={
                    "review_deterministic_rejection_count": sum(
                        1 for item in self._dedupe_rejections(all_rejections) if item.get("deterministic_review")
                    ),
                    "review_deterministic_entity_decision_count": sum(
                        1 for item in self._dedupe_rejections(all_rejections) if item.get("deterministic_entity_decision")
                    ),
                    "review_deterministic_entity_count": sum(
                        1 for item in self._dedupe_results(all_results)
                        if (item.metadata or {}).get("source") == "review_deterministic_decision"
                    ),
                },
            )
        if standard_runtime is None:
            if global_deterministic_decisions["entities"]:
                all_results.extend(
                    self._materialize_candidates(
                        full_text,
                        {"entities": global_deterministic_decisions["entities"]},
                        RiskSnippet(
                            "final_subject_adjudication",
                            "final_subject:deterministic_global",
                            0,
                            len(full_text),
                            full_text,
                        ),
                        source_name="review_deterministic_decision",
                    )
                )
            return ReviewResult(
                entities=self._dedupe_results(all_results),
                rejected_entities=self._dedupe_rejections(all_rejections),
                requires_manual_review=True,
                error="review_model_not_installed",
                metadata={
                    "review_deterministic_rejection_count": sum(
                        1 for item in self._dedupe_rejections(all_rejections) if item.get("deterministic_review")
                    ),
                    "review_deterministic_entity_decision_count": sum(
                        1 for item in self._dedupe_rejections(all_rejections) if item.get("deterministic_entity_decision")
                    ),
                    "review_deterministic_entity_count": sum(
                        1 for item in self._dedupe_results(all_results)
                        if (item.metadata or {}).get("source") == "review_deterministic_decision"
                    ),
                },
            )
        review_model_name = standard_runtime.model_id
        review_backend = standard_runtime.backend

        for snippet in scheduled_snippets:
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "stage": "review",
                            "current": parsed_snippet_count + 1,
                            "total": len(scheduled_snippets),
                            "message": "正在执行模型审查...",
                        }
                    )
                except Exception:
                    logger.debug("Review progress callback failed", exc_info=True)
            snippet_existing = self._entities_for_snippet(existing_list, snippet, mode="standard")
            deterministic_decisions = self._deterministic_review_entity_decisions(snippet_existing)
            deterministic_rejections = deterministic_decisions["rejections"]
            raw_response: Optional[str] = None
            active_runtime = standard_runtime
            if active_runtime is None:
                last_error = "review_model_not_installed"
                all_rejections.extend(deterministic_rejections)
                continue
            try:
                allow_thinking = self._review_thinking_enabled(active_runtime)
                prompt = self._build_review_prompt_for_snippet(
                    snippet,
                    existing_entities=snippet_existing,
                    allow_thinking=allow_thinking,
                )
                model_attempted = True
                raw_response = await self._review_with_runtime(
                    prompt,
                    active_runtime,
                    allow_thinking=allow_thinking,
                )
                review_model_name = active_runtime.model_id
                review_backend = active_runtime.backend
                any_model_used = True
                fallback_used = fallback_used or active_runtime.fallback
            except Exception as exc:
                logger.warning("Fragment review failed: %s", exc)
                last_error = f"{type(exc).__name__}: {exc}"
                fallback_runtime = self._select_fallback_runtime(active_runtime)
                if fallback_runtime is not None:
                    try:
                        active_runtime = fallback_runtime
                        allow_thinking = self._review_thinking_enabled(active_runtime)
                        prompt = self._build_review_prompt_for_snippet(
                            snippet,
                            existing_entities=snippet_existing,
                            allow_thinking=allow_thinking,
                        )
                        model_attempted = True
                        raw_response = await self._review_with_runtime(
                            prompt,
                            active_runtime,
                            allow_thinking=allow_thinking,
                        )
                        review_model_name = active_runtime.model_id
                        review_backend = active_runtime.backend
                        fallback_used = True
                        any_model_used = True
                    except Exception as fallback_exc:
                        last_error = f"{type(fallback_exc).__name__}: {fallback_exc}"
            if raw_response is None:
                if deterministic_decisions["entities"] and not self._is_ledger_adjudication_snippet(snippet):
                    all_results.extend(
                        self._materialize_candidates(
                            full_text,
                            {"entities": deterministic_decisions["entities"]},
                            snippet,
                            source_name="review_deterministic_decision",
                        )
                    )
                all_rejections.extend(deterministic_rejections)
                if self._is_ledger_adjudication_snippet(snippet):
                    ledger_decisions.append(
                        self._fallback_ledger_decision(
                            snippet,
                            snippet_existing,
                            reason=last_error or "ledger_review_model_response_missing",
                        )
                    )
                if "not_installed" in str(last_error):
                    break
                continue

            parsed = self._parse_json_response(raw_response or "")
            if parsed is None:
                allow_thinking = self._review_thinking_enabled(active_runtime)
                retry_prompt = self._build_review_prompt_for_snippet(
                    snippet,
                    existing_entities=snippet_existing
                    if getattr(snippet, "target_entity", None)
                    else self._entities_for_snippet(
                        existing_list,
                        snippet,
                        limit=12,
                        mode="standard",
                    ),
                    short=True,
                    allow_thinking=allow_thinking,
                )
                try:
                    model_attempted = True
                    retry_response = await self._review_with_runtime(
                        retry_prompt,
                        active_runtime,
                        allow_thinking=allow_thinking,
                    )
                    parsed = self._parse_json_response(retry_response or "")
                except Exception as exc:
                    last_error = f"json_retry_failed:{type(exc).__name__}"
            if parsed is None:
                last_error = last_error or "review_json_parse_failed"
                if deterministic_decisions["entities"] and not self._is_ledger_adjudication_snippet(snippet):
                    all_results.extend(
                        self._materialize_candidates(
                            full_text,
                            {"entities": deterministic_decisions["entities"]},
                            snippet,
                            source_name="review_deterministic_decision",
                        )
                    )
                all_rejections.extend(deterministic_rejections)
                if self._is_ledger_adjudication_snippet(snippet):
                    ledger_decisions.append(
                        self._fallback_ledger_decision(
                            snippet,
                            snippet_existing,
                            reason=last_error,
                        )
                    )
                continue
            raw_candidate_count += len(parsed.get("entities") or [])
            parsed_snippet_count += 1
            if self._is_ledger_adjudication_snippet(snippet):
                decisions = self._materialize_ledger_decisions(parsed, snippet, snippet_existing)
                if not decisions:
                    decisions = [
                        self._fallback_ledger_decision(
                            snippet,
                            snippet_existing,
                            reason="ledger_review_returned_no_decision",
                        )
                    ]
                ledger_decisions.extend(decisions)
                all_rejections.extend(self._ledger_decision_rejections(decisions, snippet_existing))
                all_rejections.extend(deterministic_rejections)
            else:
                if self._is_qwen_coverage_discovery_snippet(snippet):
                    qwen_discovery_raw_candidate_count += len(parsed.get("entities") or [])
                    discovery_results = self._materialize_candidates(full_text, parsed, snippet)
                    qwen_discovery_materialized_entity_count += len(discovery_results)
                    qwen_discovery_rejected_by_gate_count += int(self._last_materialize_gate_reject_count or 0)
                    qwen_discovery_span_miss_count += int(self._last_materialize_span_miss_count or 0)
                    all_results.extend(discovery_results)
                elif self._is_missing_candidate_review_snippet(snippet):
                    missing_payload = self._filter_missing_candidate_payload(full_text, parsed, snippet)
                    if missing_payload.get("entities"):
                        missing_results = self._materialize_candidates(
                            full_text,
                            missing_payload,
                            snippet,
                            source_name="qwen_entity_decision",
                        )
                        missing_candidate_materialized_entity_count += len(missing_results)
                        all_results.extend(missing_results)
                all_rejections.extend(self._materialize_rejections(parsed, snippet_existing))
                entity_decisions = self._materialize_entity_decisions(parsed, snippet_existing)
                all_rejections.extend(entity_decisions["rejections"])
                if entity_decisions["entities"]:
                    entity_decision_entity_count += len(entity_decisions["entities"])
                    all_results.extend(
                        self._materialize_candidates(
                            full_text,
                            {"entities": entity_decisions["entities"]},
                            snippet,
                            source_name="qwen_entity_decision",
                        )
                    )
                if deterministic_decisions["entities"]:
                    all_results.extend(
                        self._materialize_candidates(
                            full_text,
                            {"entities": deterministic_decisions["entities"]},
                            snippet,
                            source_name="review_deterministic_decision",
                        )
                    )
                all_rejections.extend(deterministic_rejections)
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "stage": "review",
                            "current": parsed_snippet_count,
                            "total": len(scheduled_snippets),
                            "message": "模型审查进行中...",
                        }
                    )
                except Exception:
                    logger.debug("Review progress callback failed", exc_info=True)

        ledger_requires_manual = any(bool(item.get("requires_manual_review")) for item in ledger_decisions)
        requires_manual_review = (
            bool(last_error)
            or not any_model_used
            or ledger_requires_manual
        )
        return ReviewResult(
            entities=self._dedupe_results(all_results),
            rejected_entities=self._dedupe_rejections(all_rejections),
            model_used=model_attempted,
            fallback_used=fallback_used,
            model_name=review_model_name,
            review_backend=review_backend,
            requires_manual_review=requires_manual_review,
            error=last_error,
            raw_candidate_count=raw_candidate_count,
            parsed_snippet_count=parsed_snippet_count,
            metadata={
                "ledger_conflict_adjudication_enabled": bool(ledger_decisions),
                "ledger_conflict_decisions": ledger_decisions,
                "ledger_conflict_decision_count": len(ledger_decisions),
                "ledger_conflict_manual_review_required": ledger_requires_manual,
                "ledger_conflict_snippet_count": sum(
                    1 for snippet in scheduled_snippets if self._is_ledger_adjudication_snippet(snippet)
                ),
                "review_scheduled_snippet_count": len(scheduled_snippets),
                "review_scheduled_ledger_snippet_count": scheduled_ledger_count,
                "review_scheduled_standard_snippet_count": scheduled_standard_count,
                "review_scheduled_missing_candidate_snippet_count": scheduled_missing_candidate_count,
                "qwen_discovery_snippet_selected_count": sum(
                    1 for snippet in scheduled_snippets if self._is_qwen_coverage_discovery_snippet(snippet)
                ),
                "qwen_discovery_raw_candidate_count": qwen_discovery_raw_candidate_count,
                "qwen_discovery_materialized_entity_count": qwen_discovery_materialized_entity_count,
                "qwen_discovery_rejected_by_gate_count": qwen_discovery_rejected_by_gate_count,
                "qwen_discovery_span_miss_count": qwen_discovery_span_miss_count,
                "missing_candidate_materialized_entity_count": missing_candidate_materialized_entity_count,
                "review_deterministic_rejection_count": sum(
                    1 for item in self._dedupe_rejections(all_rejections) if item.get("deterministic_review")
                ),
                "review_deterministic_entity_decision_count": sum(
                    1 for item in self._dedupe_rejections(all_rejections) if item.get("deterministic_entity_decision")
                ),
                "review_deterministic_entity_count": sum(
                    1 for item in self._dedupe_results(all_results)
                    if (item.metadata or {}).get("source") == "review_deterministic_decision"
                ),
                "review_entity_decision_rejection_count": sum(
                    1
                    for item in self._dedupe_rejections(all_rejections)
                    if item.get("entity_decision") and not item.get("deterministic_entity_decision")
                ),
                "review_entity_decision_entity_count": entity_decision_entity_count,
            },
        )

    @staticmethod
    def _entity_dict_for_review(entity: RecognizerResult) -> dict:
        metadata = dict(entity.metadata or {})
        return {
            "type": entity.entity_type,
            "text": entity.text,
            "start": int(entity.start),
            "end": int(entity.end),
            "source": entity.source,
            "metadata": metadata,
            "role": metadata.get("role") or metadata.get("label"),
        }

    @staticmethod
    def _schedule_review_snippets(snippets: list[RiskSnippet], *, review_limit: int) -> list[RiskSnippet]:
        """Never drop ledger conflicts, discovery coverage, or DOCX structure behind the generic cap."""
        if not snippets:
            return []
        ledger_snippets = [
            snippet
            for snippet in snippets
            if QwenFragmentReviewService._is_ledger_adjudication_snippet(snippet)
        ]
        final_subject_snippets = [
            snippet
            for snippet in snippets
            if not QwenFragmentReviewService._is_ledger_adjudication_snippet(snippet)
            and QwenFragmentReviewService._is_final_subject_adjudication_snippet(snippet)
        ]
        structure_snippets = [
            snippet
            for snippet in snippets
            if not QwenFragmentReviewService._is_ledger_adjudication_snippet(snippet)
            and not QwenFragmentReviewService._is_final_subject_adjudication_snippet(snippet)
            and str(snippet.risk_reason or "").startswith("docx_structure:")
        ]
        coverage_discovery_snippets = [
            snippet
            for snippet in snippets
            if not QwenFragmentReviewService._is_ledger_adjudication_snippet(snippet)
            and not QwenFragmentReviewService._is_final_subject_adjudication_snippet(snippet)
            and not str(snippet.risk_reason or "").startswith("docx_structure:")
            and QwenFragmentReviewService._is_qwen_coverage_discovery_snippet(snippet)
        ]
        non_ledger = [
            snippet
            for snippet in snippets
            if not QwenFragmentReviewService._is_ledger_adjudication_snippet(snippet)
            and not QwenFragmentReviewService._is_final_subject_adjudication_snippet(snippet)
            and not str(snippet.risk_reason or "").startswith("docx_structure:")
            and not QwenFragmentReviewService._is_qwen_coverage_discovery_snippet(snippet)
        ]
        ordinary_limit = max(1, int(review_limit or 1))
        structure_limit = max(2, min(len(structure_snippets), max(ordinary_limit // 2, 6)))
        priority_count = min(structure_limit, len(structure_snippets)) + len(coverage_discovery_snippets)
        remaining_ordinary_limit = max(1, ordinary_limit - priority_count)
        return [
            *ledger_snippets,
            *final_subject_snippets,
            *structure_snippets[:structure_limit],
            *coverage_discovery_snippets,
            *non_ledger[:remaining_ordinary_limit],
        ]

    @staticmethod
    def _is_final_subject_adjudication_snippet(snippet: RiskSnippet) -> bool:
        return (
            snippet.snippet_type == "rule_first_review_block"
            or str(snippet.risk_reason or "").startswith(("rule_first:", "final_subject:"))
        )

    @staticmethod
    def _is_qwen_coverage_discovery_snippet(snippet: RiskSnippet) -> bool:
        return (
            str(snippet.snippet_type or "") == "qwen_coverage_discovery"
            or str(snippet.risk_reason or "").startswith("qwen_discovery:")
        )

    @staticmethod
    def _is_missing_candidate_review_snippet(snippet: RiskSnippet) -> bool:
        return str(snippet.snippet_type or "") == "missing_candidate_review"

    def _build_coverage_discovery_prompt(
        self,
        snippet: RiskSnippet,
        *,
        allow_thinking: bool = False,
    ) -> str:
        thinking_prefix = "" if allow_thinking else "/no_think\n"
        prompt_budget = _review_prompt_char_budget(
            quality_gate=False,
            allow_thinking=allow_thinking,
        )
        snippet_limit = int(settings.REVIEW_MAX_CHARS_PER_SNIPPET or 1200)
        snippet_text = _compact_prompt_text(
            str(snippet.text or ""),
            min(snippet_limit, max(420, prompt_budget // 2)),
        )
        target = getattr(snippet, "target_entity", None)
        target_metadata = (
            target.get("metadata")
            if isinstance(target, dict) and isinstance(target.get("metadata"), dict)
            else {}
        )
        target_json = json.dumps(
            {
                key: target_metadata.get(key)
                for key in (
                    "docx_unit_id",
                    "docx_container_type",
                    "docx_part_name",
                    "docx_table_index",
                    "docx_row_index",
                    "docx_col_index",
                    "span_resolution",
                )
                if target_metadata.get(key) is not None
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = (
            thinking_prefix
            + "你是中文合同脱敏的最终查漏模型。只输出 JSON，不要解释。\n"
            + "任务：只检查这个结构单元中仍可能漏掉的脱敏主体，并输出 entities。"
            + "这是查漏任务，不是全文自由抽取；text 必须是片段里的连续原文，不能改写、概括或补字。\n"
            + "只允许四类主体：PERSON、ORGANIZATION、GOVERNMENT、LOCATION。"
            + "公司/私人单位按 ORGANIZATION；法院、银行、金融机构、政府机关、仲裁/检察/公安/监管机构等官方或准官方机构按 GOVERNMENT；地名按 LOCATION；人名按 PERSON。\n"
            + "重点查漏：公司全称、带国家/省/市/区县前缀的公司全称、括号内主体、表格单元格主体、文本框/页眉/页脚主体、甲乙方/原被告/申请人等后面的主体、并列主体中的每一项。\n"
            + "必须排除：通过/根据/要求/通知/提交/确认/负责等动作词，甲方/乙方/原告/被告等角色标签本身，该公司/本公司/相关公司等泛称，账户/开户行/金额/日期/编号/地址标签/合同期限。\n"
            + "如果一个短语左侧有动作词或角色标签，只输出后面的真实主体；如果主体在括号中，只输出括号内主体，不带括号。"
            + "如果没有真实主体，输出空 entities。\n"
            + f"结构信息：{target_json}\n"
            + f"片段开始：\n{snippet_text}\n片段结束。\n"
            + "JSON 格式：{\"entities\":[{\"type\":\"ORGANIZATION\",\"text\":\"北京某某有限公司\"}],\"rejects\":[],\"entity_decisions\":[]}\n"
        )
        return self._fit_prompt_to_review_budget(
            prompt,
            quality_gate=False,
            allow_thinking=allow_thinking,
        )

    async def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        quality_gate: bool = False,
    ) -> Dict[str, object]:
        runtime = self._select_review_runtime()
        if runtime is None:
            raise RuntimeError("review_model_not_installed")

        active_runtime = runtime
        try:
            raw_response = await self._review_with_runtime(
                prompt,
                active_runtime,
                allow_thinking=False,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("Low-memory resolution review failed: %s", exc)
            fallback_runtime = self._select_fallback_runtime(active_runtime)
            if fallback_runtime is None:
                raise
            active_runtime = fallback_runtime
            raw_response = await self._review_with_runtime(
                prompt,
                active_runtime,
                allow_thinking=False,
                max_tokens=max_tokens,
            )

        return {
            "response": raw_response,
            "model": active_runtime.model_id,
            "backend": active_runtime.backend,
            "fallback_used": bool(active_runtime.fallback),
        }

    def _select_review_runtime(self) -> Optional[ReviewRuntime]:
        if self.review_asset.installed:
            return ReviewRuntime(
                backend=settings.REVIEW_BACKEND.lower().strip(),
                model_id=self.review_asset.model_id,
                asset=self.review_asset,
                tier="local_review",
            )
        if self.fallback_asset.installed and (
            not settings.is_high_quality_lowmem_mode()
            or settings.LOWMEM_ENABLE_LOCAL_REVIEW_FALLBACK
        ):
            return ReviewRuntime(
                backend=settings.REVIEW_MODEL_FALLBACK_BACKEND.lower().strip(),
                model_id=self.fallback_asset.model_id,
                asset=self.fallback_asset,
                fallback=True,
                tier="local_review_fallback",
            )
        return None

    def _select_fallback_runtime(self, active_runtime: ReviewRuntime) -> Optional[ReviewRuntime]:
        if settings.is_high_quality_lowmem_mode() and not settings.LOWMEM_ENABLE_LOCAL_REVIEW_FALLBACK:
            return None
        if active_runtime.asset == self.fallback_asset:
            return None
        if self.fallback_asset.installed:
            return ReviewRuntime(
                backend=settings.REVIEW_MODEL_FALLBACK_BACKEND.lower().strip(),
                model_id=self.fallback_asset.model_id,
                asset=self.fallback_asset,
                fallback=True,
                tier="local_review_fallback",
            )
        return None

    def _select_runtime_for_snippet(self, snippet: RiskSnippet) -> Optional[ReviewRuntime]:
        return self._select_review_runtime()

    @staticmethod
    def _dedupe_results(results: list[RecognizerResult]) -> list[RecognizerResult]:
        deduped: list[RecognizerResult] = []
        index_by_key: dict[tuple[str, int, int, str], int] = {}
        for item in sorted(results, key=lambda result: (result.start, result.end, result.entity_type, result.text)):
            key = (item.entity_type, int(item.start), int(item.end), item.text)
            if key in index_by_key:
                existing_index = index_by_key[key]
                existing = deduped[existing_index]
                existing_metadata = dict(existing.metadata or {})
                item_metadata = dict(item.metadata or {})
                if item_metadata.get("entity_decision") and not existing_metadata.get("entity_decision"):
                    deduped[existing_index] = item
                continue
            index_by_key[key] = len(deduped)
            deduped.append(item)
        return deduped

    @staticmethod
    def _dedupe_rejections(rejections: list[Dict]) -> list[Dict]:
        deduped: list[Dict] = []
        index_by_key: dict[tuple[str, str, int, int], int] = {}
        for item in rejections:
            key = (
                str(item.get("type") or "").upper(),
                str(item.get("text") or ""),
                int(item.get("start") or 0),
                int(item.get("end") or 0),
            )
            if key in index_by_key:
                existing_index = index_by_key[key]
                existing = deduped[existing_index]
                existing_rank = (
                    int(bool(existing.get("entity_decision"))) * 2
                    + int(bool(existing.get("deterministic_entity_decision")))
                    + int(bool(existing.get("deterministic_review")))
                )
                item_rank = (
                    int(bool(item.get("entity_decision"))) * 2
                    + int(bool(item.get("deterministic_entity_decision")))
                    + int(bool(item.get("deterministic_review")))
                )
                if item_rank > existing_rank:
                    deduped[existing_index] = item
                continue
            index_by_key[key] = len(deduped)
            deduped.append(item)
        return deduped

    def _build_prompt(
        self,
        snippet: RiskSnippet,
        *,
        existing_entities: Optional[list[dict]] = None,
        short: bool = False,
        allow_thinking: bool = False,
    ) -> str:
        quality_gate = False
        prompt_budget = _review_prompt_char_budget(
            quality_gate=quality_gate,
            allow_thinking=allow_thinking,
        )
        snippet_limit = int(settings.REVIEW_MAX_CHARS_PER_SNIPPET or 1200)
        snippet_text = _compact_prompt_text(str(snippet.text or ""), min(snippet_limit, max(320, prompt_budget // 3)))
        existing_payload = self._compact_existing_entities_for_prompt(
            existing_entities or [],
            max_items=16,
            char_budget=max(520, prompt_budget // 3),
        )
        existing_json = json.dumps(existing_payload, ensure_ascii=False, separators=(",", ":"))
        short_org_hints = self._format_short_org_action_hints(snippet_text)
        thinking_prefix = "" if allow_thinking else "/no_think\n"
        if short:
            prompt = (
                f"{thinking_prefix}只输出 JSON。重新检查片段：entities 输出真实敏感实体，rejects 输出已识别里的明显错误。"
                "即使已识别实体正确，也要在 entities 再输出确认。text 必须来自原文。"
                "对已识别实体必须用 entity_decisions 裁决：keep/reject/trim_to/split_into。"
                "候选以“通过/经过/根据/依据/由/向/对/与/和”等前置功能词开头时，必须排除前缀功能词；"
                "后半段是完整主体则 trim_to 后半段，后半段不是完整主体或只是泛称则 reject。"
                "动作词在主体名称中间时才 keep。"
                "不要抽取国家/中华人民共和国/人民共和国/法定代表人/委托代理人/技术人员/我中心/"
                "一审法院/二审法院/本院/法院/地址/合同期限/三个/五个；具体日期值不要放入 rejects。\n"
                f"{short_org_hints}"
                f"已识别（可能有错）：{existing_json}\n"
                f"片段：\n{snippet_text}\n"
                "格式：{\"entities\":[{\"type\":\"ORGANIZATION\",\"text\":\"某某公司\"}],"
                "\"rejects\":[{\"type\":\"ORGANIZATION\",\"text\":\"国家\",\"reason\":\"普通词\"}],"
                "\"entity_decisions\":[{\"text\":\"通过某某公司\",\"type\":\"ORGANIZATION\","
                "\"action\":\"trim_to\",\"trim_to\":\"某某公司\",\"reason\":\"去掉开头功能词\"}]}\n"
            )
            return self._fit_prompt_to_review_budget(
                prompt,
                quality_gate=quality_gate,
                allow_thinking=allow_thinking,
            )
        compact_prompt = (
            thinking_prefix
            + "你是中文合同/法律文书脱敏复核模型。只输出 JSON。\n"
            + "任务：根据片段原文输出真实敏感 entities，并把已识别实体中的明显错误放入 rejects。"
            + "同时用 entity_decisions 逐项裁决已识别实体，action 只能是 keep/reject/trim_to/split_into。"
            + "候选以“通过/经过/根据/依据/由/向/对/与/和”等前置功能词开头时，必须排除前缀功能词；后半段是完整主体则 trim_to 后半段，后半段不是完整主体或只是泛称则 reject；动作词在主体名称中间时才 keep。"
            + "text 必须来自片段连续原文；身份词/职务词/标签词/普通词不要当实体；金额和日期默认不脱敏。"
            + "2-6字简称在履约/结算/付款/签章语境中按 ORGANIZATION 重点审查。\n"
            + f"片段类型：{snippet.snippet_type}\n"
            + f"已识别实体：{existing_json}\n"
            + f"片段开始：\n{snippet_text}\n片段结束。\n"
            + "JSON 格式：{\"entities\":[{\"type\":\"ORGANIZATION\",\"text\":\"某某公司\"}],"
            + "\"rejects\":[{\"type\":\"ORGANIZATION\",\"text\":\"国家\",\"reason\":\"普通词\"}],"
            + "\"entity_decisions\":[{\"text\":\"通过某某公司\",\"type\":\"ORGANIZATION\","
            + "\"action\":\"trim_to\",\"trim_to\":\"某某公司\",\"reason\":\"去掉开头功能词\"}]}\n"
            + short_org_hints
        )
        long_prompt = (
            thinking_prefix
            + self.PROMPT_TEMPLATE.format(
                snippet_type=snippet.snippet_type,
                snippet_text=snippet_text,
                existing_entities=existing_json,
            )
            + short_org_hints
        )
        prompt = compact_prompt if len(long_prompt) > prompt_budget else long_prompt
        return self._fit_prompt_to_review_budget(
            prompt,
            quality_gate=quality_gate,
            allow_thinking=allow_thinking,
        )

    @staticmethod
    def _compact_existing_entities_for_prompt(
        entities: Iterable[dict],
        *,
        max_items: int,
        char_budget: int,
    ) -> list[dict]:
        compacted: list[dict] = []
        used_chars = 2
        ordered_entities = sorted(
            [item for item in entities or [] if isinstance(item, dict)],
            key=lambda item: (
                -QwenFragmentReviewService._existing_entity_suspicion_score(item),
                int(item.get("start") or 0),
                int(item.get("end") or 0),
            ),
        )
        for item in ordered_entities:
            if not isinstance(item, dict):
                continue
            payload = {
                "type": str(item.get("type") or item.get("entity_type") or "").upper(),
                "text": str(item.get("text") or "")[:80],
                "start": int(item.get("start") or 0),
                "end": int(item.get("end") or 0),
            }
            role = str(item.get("role") or "").strip()
            if role:
                payload["role"] = role[:20]
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if compacted and (len(compacted) >= max_items or used_chars + len(encoded) > char_budget):
                break
            compacted.append(payload)
            used_chars += len(encoded) + 1
        return compacted

    @staticmethod
    def _fit_prompt_to_review_budget(
        prompt: str,
        *,
        quality_gate: bool,
        allow_thinking: bool,
    ) -> str:
        budget = _review_prompt_char_budget(quality_gate=quality_gate, allow_thinking=allow_thinking)
        return _compact_prompt_text(prompt, budget)

    @classmethod
    def _collect_short_org_action_candidates(cls, snippet_text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in cls.SHORT_ORG_ACTION_PATTERNS:
            for match in pattern.finditer(snippet_text or ""):
                subject = cls._trim_short_org_candidate_core(str(match.group("subject") or "").strip())
                normalized = normalize_entity_text(subject)
                if normalized in seen or not cls._short_org_candidate_is_usable(normalized):
                    continue
                seen.add(normalized)
                candidates.append(subject)
        return candidates[:8]

    @classmethod
    def _format_short_org_action_hints(cls, snippet_text: str) -> str:
        candidates = cls._collect_contextual_short_org_candidates(snippet_text)
        if not candidates:
            return ""
        hint_json = json.dumps(candidates, ensure_ascii=False)
        list_groups = cls._collect_short_org_list_groups(snippet_text)
        list_completion_hint = (
            "如果片段中存在并列主体列表，必须逐项输出列表中的每一个主体，不能漏掉首项、中间项或末项。\n"
            if any(len(group["items"]) >= 2 for group in list_groups)
            else ""
        )
        return (
            "组织主体动作候选（仅提示你重点复核，不代表已确认）："
            f"{hint_json}\n"
            "如果这些候选在片段中更像合同、诉讼或交易主体，而不是自然人，请优先按 ORGANIZATION 审查。\n"
            f"{list_completion_hint}"
        )

    @classmethod
    def _short_org_candidate_is_usable(cls, normalized: str) -> bool:
        if not normalized:
            return False
        if is_non_subject_action_or_function_term(normalized):
            return False
        if not 2 <= len(normalized) <= 8:
            return False
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return False
        if normalized in cls.NON_ENTITY_TERMS:
            return False
        if re.fullmatch(r"[\dA-Za-z]+", normalized):
            return False
        if any(token in normalized for token in ("甲方", "乙方", "丙方", "对方", "一方", "双方", "各方")):
            return False
        return True

    @classmethod
    def _looks_like_short_org_list_item(cls, text: str) -> bool:
        normalized = normalize_entity_text(text)
        if not normalized or not cls._short_org_candidate_is_usable(normalized):
            return False
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return False
        if any(token in normalized for token in cls.SHORT_ORG_LIST_INVALID_ITEM_TOKENS):
            return False
        if len(normalized) >= 5 and cls._looks_like_narrative_wrapped_candidate(normalized):
            return False
        if len(normalized) >= 5 and sum(1 for ch in normalized if "\u4e00" <= ch <= "\u9fff") >= 5:
            org_suffix_hits = sum(1 for token in ORG_SUFFIX_TERMS if token and token in normalized)
            if (
                org_suffix_hits == 0
                and not looks_like_organization_short_name(normalized)
                and not any(
                    hint in normalized
                    for hint in ("科技", "材料", "环保", "工艺", "鞋厂", "塑料", "洁净", "商行", "工作室", "合作社")
                )
            ):
                return False
        return True

    @classmethod
    def _collect_contextual_short_org_candidates(cls, snippet_text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _append(value: str, *, matched_text: str = "", suffix_text: str = "") -> None:
            cleaned_value = cls._trim_short_org_candidate_core(value)
            normalized = normalize_entity_text(cleaned_value)
            if cls._is_noisy_short_org_candidate(
                normalized,
                matched_text=matched_text,
                suffix_text=suffix_text,
            ):
                return
            if normalized in seen or not cls._short_org_candidate_is_usable(normalized):
                return
            seen.add(normalized)
            candidates.append(cleaned_value.strip())

        for value in cls._collect_short_org_list_candidates(snippet_text):
            _append(value)

        for value in cls._collect_short_org_action_candidates(snippet_text):
            _append(value)

        for pattern in cls.SHORT_ORG_CONTEXT_PATTERNS:
            for match in pattern.finditer(snippet_text or ""):
                subject = str(match.group("subject") or "")
                matched_text = str(match.group(0) or "")
                suffix_text = matched_text[len(subject) :] if len(matched_text) >= len(subject) else ""
                _append(subject, matched_text=matched_text, suffix_text=suffix_text)

        return candidates[:12]

    @classmethod
    def _collect_short_org_list_groups(cls, snippet_text: str) -> list[Dict[str, object]]:
        groups: list[Dict[str, object]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()

        def _append_group(label: str, items_text: str) -> None:
            cleaned_label = clean_candidate_text(label)
            cleaned_items = clean_candidate_text(items_text)
            if not cleaned_items:
                return
            usable_items: list[str] = []
            local_seen: set[str] = set()
            for raw_item in cls.SHORT_ORG_LIST_SPLIT_PATTERN.split(cleaned_items):
                candidate = cls._trim_short_org_list_item(raw_item)
                normalized = normalize_entity_text(candidate)
                if (
                    not normalized
                    or normalized in local_seen
                    or not cls._looks_like_short_org_list_item(candidate)
                ):
                    continue
                local_seen.add(normalized)
                usable_items.append(candidate)
            if len(usable_items) < 2:
                return
            signature = (
                normalize_entity_text(cleaned_label),
                tuple(normalize_entity_text(item) for item in usable_items),
            )
            if signature in seen:
                return
            seen.add(signature)
            groups.append({"label": cleaned_label, "items": usable_items})

        for pattern in cls.SHORT_ORG_LIST_PATTERNS:
            for match in pattern.finditer(snippet_text or ""):
                _append_group(str(match.group("label") or ""), str(match.group("items") or ""))

        # Generic fallback for enumerations like "...客户包括A、B、C..." where fixed labels vary a lot.
        normalized_text = clean_candidate_text(snippet_text or "")
        if normalized_text and any(sep in normalized_text for sep in ("、", "，", ",")):
            trigger_match = re.search(
                r"(?P<label>[\u4e00-\u9fa5A-Za-z0-9]{0,16}"
                r"(?:客户|企业|公司|集团|机构|单位|主体|合作方|签约方|商户|用电企业))"
                r"(?:分别为|包括|如下|名单为|为|有|:|：)",
                normalized_text,
            )
            if trigger_match:
                label = str(trigger_match.group("label") or "")
                items_text = normalized_text[trigger_match.end() :]
                items_text = re.split(r"[。；;\n\r]", items_text, maxsplit=1)[0]
                _append_group(label, items_text)

        return groups[:6]

    @classmethod
    def _collect_short_org_list_candidates(cls, snippet_text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for group in cls._collect_short_org_list_groups(snippet_text):
            usable_items = list(group.get("items") or [])
            if len(usable_items) < 2:
                continue
            for candidate in usable_items:
                normalized = normalize_entity_text(candidate)
                if normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(candidate)
        return candidates[:12]

    @classmethod
    def _trim_short_org_list_item(cls, value: str) -> str:
        candidate = clean_candidate_text(value)
        candidate = cls.SHORT_ORG_LIST_SUFFIX_NOISE_PATTERN.sub("", candidate).strip(" \t\r\n:：,，;；。()（）")
        return candidate

    @classmethod
    def _is_strong_short_org_list_label(cls, label: str) -> bool:
        normalized = normalize_entity_text(label)
        if not normalized:
            return False
        return any(
            token in normalized
            for token in (
                "公司",
                "企业",
                "集团",
                "机构",
                "单位",
                "合作方",
                "签约方",
                "合同主体",
                "交易主体",
                "项目主体",
                "关联企业",
                "涉案公司",
            )
        )

    @classmethod
    def _is_noisy_short_org_candidate(
        cls,
        normalized: str,
        *,
        matched_text: str = "",
        suffix_text: str = "",
    ) -> bool:
        if not normalized:
            return True
        if is_non_subject_action_or_function_term(normalized):
            return True
        if normalized in cls.SHORT_ORG_EXACT_NOISE_TERMS:
            return True
        trimmed = normalize_entity_text(cls._trim_short_org_candidate_core(normalized))
        if trimmed and trimmed != normalized:
            return True
        if any(token in normalized for token in cls.SHORT_ORG_NOISE_TOKENS):
            return True
        if normalized[0] in "由与向对和及同并将把给就再另续仍经":
            return True
        if normalized[-1] in "由与向对和及同并将把给就再另续仍经":
            return True
        if any(normalized.startswith(prefix) and len(normalized) > len(prefix) for prefix in cls.SHORT_ORG_PREFIX_TOKENS):
            return True
        for marker in cls.SHORT_ORG_RELATION_MARKERS:
            index = normalized.find(marker)
            if index <= 0:
                continue
            tail = normalized[index + len(marker) :]
            if not tail or any(token in tail for token in cls.SHORT_ORG_CONTEXT_SPLIT_TAIL_TOKENS):
                return True

        matched_normalized = normalize_entity_text(matched_text)
        suffix_normalized = normalize_entity_text(suffix_text)
        if not matched_normalized or not suffix_normalized:
            return False

        explicit_suffix = cls._longest_explicit_org_suffix(matched_normalized)
        if not explicit_suffix or len(explicit_suffix) <= len(suffix_normalized):
            return False
        leftover_suffix = explicit_suffix[: len(explicit_suffix) - len(suffix_normalized)]
        return bool(leftover_suffix) and normalized.endswith(leftover_suffix)

    @staticmethod
    def _longest_explicit_org_suffix(value: str) -> str:
        normalized = normalize_entity_text(value)
        matches = [suffix for suffix in ORG_SUFFIX_TERMS if normalized.endswith(suffix)]
        if not matches:
            return ""
        return max(matches, key=len)

    @classmethod
    def _strip_narrative_prefixes(cls, value: str) -> str:
        cleaned = clean_candidate_text(value)
        previous = None
        while cleaned and cleaned != previous:
            previous = cleaned
            cleaned = cls.NARRATIVE_PREFIX_PATTERN.sub("", cleaned).strip(" \t\r\n:：,，;；。")
        return cleaned

    @classmethod
    def _trim_short_org_candidate_core(cls, value: str) -> str:
        normalized = normalize_entity_text(cls._strip_narrative_prefixes(value))
        if not normalized:
            return ""
        bounded = find_short_org_prefix_before_non_subject_boundary(normalized)
        if bounded is not None and bounded[0] == 0 and bounded[1] > bounded[0]:
            return normalized[bounded[0] : bounded[1]]

        previous = None
        while normalized and normalized != previous:
            previous = normalized
            for prefix in sorted(cls.SHORT_ORG_PREFIX_TOKENS, key=len, reverse=True):
                if normalized.startswith(prefix) and len(normalized) - len(prefix) >= 2:
                    normalized = normalized[len(prefix) :]
                    break

            marker_trimmed = False
            for marker in sorted(cls.SHORT_ORG_RELATION_MARKERS, key=len, reverse=True):
                index = normalized.find(marker)
                if index < 2:
                    continue
                head = normalized[:index]
                tail = normalized[index + len(marker) :]
                if not head:
                    continue
                if not tail or any(token in tail for token in cls.SHORT_ORG_CONTEXT_SPLIT_TAIL_TOKENS):
                    normalized = head
                    marker_trimmed = True
                    break
            if marker_trimmed:
                continue

            trailing_trimmed = False
            for token in sorted(
                {*(cls.SHORT_ORG_NOISE_TOKENS), *(cls.SHORT_ORG_TRAILING_CONTEXT_TOKENS)},
                key=len,
                reverse=True,
            ):
                if normalized.endswith(token) and len(normalized) - len(token) >= 2:
                    normalized = normalized[: -len(token)]
                    trailing_trimmed = True
                    break
            if not trailing_trimmed:
                break

        return normalized.strip()

    def _build_review_prompt_for_snippet(
        self,
        snippet: RiskSnippet,
        *,
        existing_entities: Optional[list[dict]] = None,
        short: bool = False,
        allow_thinking: bool = False,
    ) -> str:
        if self._is_ledger_adjudication_snippet(snippet):
            return self._build_ledger_adjudication_prompt(
                snippet,
                existing_entities=existing_entities,
                allow_thinking=allow_thinking,
            )
        if self._is_qwen_coverage_discovery_snippet(snippet):
            return self._build_coverage_discovery_prompt(
                snippet,
                allow_thinking=allow_thinking,
            )
        return self._build_prompt(
            snippet,
            existing_entities=existing_entities,
            short=short,
            allow_thinking=allow_thinking,
        )

    @staticmethod
    def _is_ledger_adjudication_snippet(snippet: RiskSnippet) -> bool:
        return (
            str(snippet.snippet_type or "") == "ledger_conflict_adjudication"
            or str(snippet.risk_reason or "").startswith("subject_ledger:")
        )

    def _build_ledger_adjudication_prompt(
        self,
        snippet: RiskSnippet,
        *,
        existing_entities: Optional[list[dict]] = None,
        allow_thinking: bool = False,
    ) -> str:
        thinking = "/think" if allow_thinking else "/no_think"
        prompt_budget = _review_prompt_char_budget(quality_gate=False, allow_thinking=allow_thinking)
        snippet_text = _compact_prompt_text(
            str(snippet.text or ""),
            min(int(settings.REVIEW_MAX_CHARS_PER_SNIPPET or 900), max(320, prompt_budget // 3)),
        )
        target_entity = getattr(snippet, "target_entity", None)
        target_payload = target_entity if isinstance(target_entity, dict) else {}
        target_metadata = (
            target_payload.get("metadata")
            if isinstance(target_payload.get("metadata"), dict)
            else {}
        )
        edge_id = str(target_metadata.get("subject_ledger_edge_id") or "").strip()
        source_subject_id = str(target_metadata.get("subject_ledger_subject_id") or "").strip()
        target_subject_id = str(target_metadata.get("subject_ledger_edge_target_subject_id") or "").strip()
        target_canonical_text = str(target_metadata.get("subject_ledger_edge_target_canonical_text") or "").strip()
        edge_relation = str(target_metadata.get("subject_ledger_edge_relation") or "").strip()
        target_json = json.dumps(target_payload, ensure_ascii=False, separators=(",", ":"))
        existing_payload = self._compact_existing_entities_for_prompt(
            existing_entities or [],
            max_items=8,
            char_budget=max(280, prompt_budget // 4),
        )
        existing_json = json.dumps(existing_payload, ensure_ascii=False, separators=(",", ":"))
        if edge_id and target_subject_id:
            task_line = (
                "任务：只裁决这一条 ledger identity edge，不做全文补漏，不发明主体，不改无关实体。"
                "判断 source_subject 是否与 target_subject 是同一主体。\n"
                f"edge：edge_id={edge_id}，relation={edge_relation}，"
                f"source_subject_id={source_subject_id}，target_subject_id={target_subject_id}，"
                f"target_canonical_text={target_canonical_text}\n"
            )
            schema = (
                "JSON 格式：{\"decisions\":[{\"decision_scope\":\"edge\","
                "\"edge_id\":\"E00001\",\"occurrence_id\":\"O00001:0:2:ORGANIZATION\","
                "\"source_subject_id\":\"S00002\",\"target_subject_id\":\"S00001\","
                "\"subject_id\":\"S00002\",\"action\":\"merge_to_canonical\","
                "\"canonical_subject_id\":\"S00001\",\"confidence\":0.86,"
                "\"reason\":\"上下文证明 source 是 target 简称\"}],"
                "\"requires_manual_review\":false}\n"
            )
        else:
            task_line = "任务：只裁决 target_entity 这一条 ledger 候选，不做全文补漏，不发明主体，不改无关实体。\n"
            schema = (
                "JSON 格式：{\"decisions\":[{\"occurrence_id\":\"O00001:0:2:ORGANIZATION\","
                "\"subject_id\":\"S00001\",\"action\":\"confirm\",\"canonical_subject_id\":\"S00001\","
                "\"confidence\":0.86,\"reason\":\"片段中承担付款主体动作\"}],\"requires_manual_review\":false}\n"
            )
        prompt = (
            f"{thinking}\n"
            "你是中文合同脱敏主体台账冲突裁决模型。只输出 JSON，不要解释。\n"
            f"{task_line}"
            "可选 action 只能是 confirm、merge_to_canonical、keep_separate、reject、manual_review。\n"
            "confirm：该候选本身是应脱敏主体，但不改变现有主体归属；"
            "merge_to_canonical：source_subject 与 target_subject 是同一主体，合并到 canonical_subject_id/target_subject_id；"
            "keep_separate：source_subject 是应脱敏主体但与 target_subject 不是同一主体；"
            "reject：该候选不是敏感实体；manual_review：证据不足。\n"
            "当存在 edge_id 时，必须返回 decision_scope=edge、edge_id、source_subject_id、target_subject_id；"
            "只有能证明同一主体时才使用 merge_to_canonical，否则使用 keep_separate 或 manual_review。\n"
            "短组织简称只有在承担签约、履约、付款、收款、结算、交付、施工、对账、盖章、落款等主体动作，"
            "或与 canonical_text 明显共享商业核心时，才能 confirm 或 merge_to_canonical。\n"
            "不要输出 entities；如需拒绝，使用 decisions action=reject，不要使用 rejects。\n"
            f"风险原因：{snippet.risk_reason}\n"
            f"target_entity：{target_json}\n"
            f"附近已识别实体：{existing_json}\n"
            f"片段：\n{snippet_text}\n"
            f"{schema}"
        )
        return self._fit_prompt_to_review_budget(
            prompt,
            quality_gate=False,
            allow_thinking=allow_thinking,
        )

    @staticmethod
    def _materialize_ledger_decisions(
        payload: Dict[str, Any],
        snippet: RiskSnippet,
        existing_entities: list[dict],
    ) -> list[Dict[str, Any]]:
        target_entity = getattr(snippet, "target_entity", None)
        target = target_entity if isinstance(target_entity, dict) else {}
        fallback_existing = existing_entities[0] if existing_entities else {}
        target_metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
        if not target_metadata and isinstance(fallback_existing.get("metadata"), dict):
            target_metadata = fallback_existing.get("metadata") or {}
        allowed_actions = {"confirm", "merge_to_canonical", "keep_separate", "reject", "manual_review"}
        rows = payload.get("decisions")
        if not isinstance(rows, list):
            rows = []
        if not rows and payload.get("action"):
            rows = [payload]
        decisions: list[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip().lower()
            if action not in allowed_actions:
                action = "manual_review"
            edge_id = str(
                row.get("edge_id")
                or target_metadata.get("subject_ledger_edge_id")
                or ""
            ).strip()
            source_subject_id = str(
                row.get("source_subject_id")
                or target_metadata.get("subject_ledger_subject_id")
                or ""
            ).strip()
            target_subject_id = str(
                row.get("target_subject_id")
                or target_metadata.get("subject_ledger_edge_target_subject_id")
                or ""
            ).strip()
            edge_relation = str(
                row.get("edge_relation")
                or row.get("relation")
                or target_metadata.get("subject_ledger_edge_relation")
                or ""
            ).strip()
            occurrence_id = str(
                row.get("occurrence_id")
                or target_metadata.get("subject_ledger_occurrence_id")
                or ""
            ).strip()
            subject_id = str(
                row.get("subject_id")
                or source_subject_id
                or target_metadata.get("subject_ledger_subject_id")
                or ""
            ).strip()
            canonical_subject_id = str(
                row.get("canonical_subject_id")
                or (target_subject_id if action == "merge_to_canonical" else "")
                or source_subject_id
                or target_metadata.get("subject_ledger_subject_id")
                or ""
            ).strip()
            if edge_id and action == "confirm" and source_subject_id:
                canonical_subject_id = source_subject_id
            decisions.append(
                {
                    "decision_scope": "edge" if edge_id else str(row.get("decision_scope") or "occurrence"),
                    "edge_id": edge_id,
                    "source_subject_id": source_subject_id,
                    "target_subject_id": target_subject_id,
                    "edge_relation": edge_relation,
                    "occurrence_id": occurrence_id,
                    "subject_id": subject_id,
                    "action": action,
                    "canonical_subject_id": canonical_subject_id,
                    "confidence": QwenFragmentReviewService._coerce_confidence(row.get("confidence")),
                    "reason": str(row.get("reason") or "").strip(),
                    "requires_manual_review": bool(row.get("requires_manual_review")) or action == "manual_review",
                    "text": str(target.get("text") or fallback_existing.get("text") or ""),
                    "type": str(target.get("type") or fallback_existing.get("type") or "").upper(),
                    "start": int(target.get("start") or fallback_existing.get("start") or 0),
                    "end": int(target.get("end") or fallback_existing.get("end") or 0),
                    "source": target.get("source") or fallback_existing.get("source"),
                }
            )
        if not decisions and payload.get("requires_manual_review"):
            decisions.append(
                {
                    "decision_scope": "edge" if target_metadata.get("subject_ledger_edge_id") else "occurrence",
                    "edge_id": str(target_metadata.get("subject_ledger_edge_id") or ""),
                    "source_subject_id": str(target_metadata.get("subject_ledger_subject_id") or ""),
                    "target_subject_id": str(target_metadata.get("subject_ledger_edge_target_subject_id") or ""),
                    "edge_relation": str(target_metadata.get("subject_ledger_edge_relation") or ""),
                    "occurrence_id": str(target_metadata.get("subject_ledger_occurrence_id") or ""),
                    "subject_id": str(target_metadata.get("subject_ledger_subject_id") or ""),
                    "action": "manual_review",
                    "canonical_subject_id": str(
                        target_metadata.get("subject_ledger_edge_target_subject_id")
                        or target_metadata.get("subject_ledger_subject_id")
                        or ""
                    ),
                    "confidence": 0.0,
                    "reason": "model_requires_manual_review",
                    "requires_manual_review": True,
                    "text": str(target.get("text") or fallback_existing.get("text") or ""),
                    "type": str(target.get("type") or fallback_existing.get("type") or "").upper(),
                    "start": int(target.get("start") or fallback_existing.get("start") or 0),
                    "end": int(target.get("end") or fallback_existing.get("end") or 0),
                    "source": target.get("source") or fallback_existing.get("source"),
                }
            )
        return decisions

    @staticmethod
    def _fallback_ledger_decision(
        snippet: RiskSnippet,
        existing_entities: list[dict],
        *,
        reason: str,
    ) -> Dict[str, Any]:
        target_entity = getattr(snippet, "target_entity", None)
        target = target_entity if isinstance(target_entity, dict) else {}
        fallback_existing = existing_entities[0] if existing_entities else {}
        target_metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
        if not target_metadata and isinstance(fallback_existing.get("metadata"), dict):
            target_metadata = fallback_existing.get("metadata") or {}
        edge_id = str(target_metadata.get("subject_ledger_edge_id") or "").strip()
        source_subject_id = str(target_metadata.get("subject_ledger_subject_id") or "").strip()
        target_subject_id = str(target_metadata.get("subject_ledger_edge_target_subject_id") or "").strip()
        occurrence_id = str(target_metadata.get("subject_ledger_occurrence_id") or "").strip()
        subject_id = source_subject_id or str(target_metadata.get("subject_ledger_subject_id") or "").strip()
        return {
            "decision_scope": "edge" if edge_id else "occurrence",
            "edge_id": edge_id,
            "source_subject_id": source_subject_id,
            "target_subject_id": target_subject_id,
            "edge_relation": str(target_metadata.get("subject_ledger_edge_relation") or ""),
            "occurrence_id": occurrence_id,
            "subject_id": subject_id,
            "action": "manual_review",
            "canonical_subject_id": target_subject_id or subject_id,
            "confidence": 0.0,
            "reason": str(reason or "ledger_review_requires_manual_review"),
            "requires_manual_review": True,
            "text": str(target.get("text") or fallback_existing.get("text") or ""),
            "type": str(target.get("type") or fallback_existing.get("type") or "").upper(),
            "start": int(target.get("start") or fallback_existing.get("start") or 0),
            "end": int(target.get("end") or fallback_existing.get("end") or 0),
            "source": target.get("source") or fallback_existing.get("source"),
        }

    @staticmethod
    def _ledger_decision_rejections(
        decisions: list[Dict[str, Any]],
        existing_entities: list[dict],
    ) -> list[Dict[str, Any]]:
        rejected: list[Dict[str, Any]] = []
        for decision in decisions:
            if decision.get("action") != "reject":
                continue
            target = None
            occurrence_id = str(decision.get("occurrence_id") or "")
            for existing in existing_entities:
                metadata = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
                if occurrence_id and metadata.get("subject_ledger_occurrence_id") == occurrence_id:
                    target = existing
                    break
            target = target or decision
            rejected.append(
                {
                    "text": str(target.get("text") or ""),
                    "type": str(target.get("type") or "").upper(),
                    "start": int(target.get("start") or 0),
                    "end": int(target.get("end") or 0),
                    "source": target.get("source"),
                    "reason": str(decision.get("reason") or "ledger_adjudication_reject"),
                    "ledger_decision": dict(decision),
                }
            )
        return rejected

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _review_thinking_enabled(runtime: ReviewRuntime | None) -> bool:
        return False

    @staticmethod
    def _entities_for_snippet(
        entities: Iterable[RecognizerResult],
        snippet: RiskSnippet,
        *,
        limit: int = 24,
        mode: str = "standard",
    ) -> list[dict]:
        target_entity = getattr(snippet, "target_entity", None)
        if isinstance(target_entity, dict) and target_entity:
            return [
                {
                    "type": str(target_entity.get("type") or target_entity.get("entity_type") or "").upper(),
                    "text": str(target_entity.get("text") or ""),
                    "start": int(target_entity.get("start") or 0),
                    "end": int(target_entity.get("end") or 0),
                    "source": target_entity.get("source"),
                    "metadata": dict(target_entity.get("metadata") or {}),
                    "role": (
                        (target_entity.get("metadata") or {}).get("role")
                        if isinstance(target_entity.get("metadata"), dict)
                        else None
                    )
                    or (
                        (target_entity.get("metadata") or {}).get("label")
                        if isinstance(target_entity.get("metadata"), dict)
                        else None
                    ),
                }
            ]
        items: list[dict] = []
        for entity in entities:
            overlap = entity.start < snippet.end and entity.end > snippet.start
            nearby = mode == "quality_gate" and entity.start < snippet.end + 80 and entity.end > snippet.start - 80
            if overlap or nearby:
                items.append(
                    {
                        "type": entity.entity_type,
                        "text": entity.text,
                        "start": entity.start,
                        "end": entity.end,
                        "source": entity.source,
                        "metadata": dict(entity.metadata or {}),
                        "role": (entity.metadata or {}).get("role") or (entity.metadata or {}).get("label"),
                    }
                )
        if mode == "quality_gate":
            items.sort(
                key=lambda item: (
                    -QwenFragmentReviewService._existing_entity_suspicion_score(item),
                    int(item.get("start") or 0),
                    int(item.get("end") or 0),
                )
            )
        else:
            items.sort(key=lambda item: (int(item.get("start") or 0), int(item.get("end") or 0)))
        return items[:limit]

    @staticmethod
    def _existing_entity_suspicion_score(item: dict) -> int:
        text = str(item.get("text") or "")
        entity_type = str(item.get("type") or "").upper()
        normalized = normalize_entity_text(text)
        if not normalized:
            return 0
        score = 0
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            score += 100
        if normalized in QwenFragmentReviewService.NON_ENTITY_TERMS:
            score += 100
        if entity_type == "PERSON" and is_org_like_text(normalized):
            score += 85
        if entity_type in {"ORGANIZATION", "GOVERNMENT"} and is_probable_person(normalized):
            score += 80
        if entity_type == "GOVERNMENT" and normalized in {"法院", "人民法院", "一审法院", "二审法院", "本院", "贵院"}:
            score += 90
        if entity_type in {"ORGANIZATION", "GOVERNMENT"} and len(normalized) > 18 and any(
            token in normalized for token in ("不服", "请求", "认为", "提交", "证明", "承担责任", "通过", "经过", "根据", "依据")
        ):
            score += 70
        if entity_type in {"ORGANIZATION", "GOVERNMENT"} and any(
            normalized.startswith(prefix)
            for prefix in ("通过", "经过", "根据", "依据", "按照", "由", "向", "对", "与", "和")
        ):
            score += 90
        if entity_type in {"ORGANIZATION", "GOVERNMENT"} and looks_like_organization_short_name(normalized):
            score += 25
        return score

    async def _review_with_runtime(
        self,
        prompt: str,
        runtime: ReviewRuntime,
        *,
        allow_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        if runtime.asset is None:
            raise RuntimeError("review_model_not_installed")
        return await self._review_with_backend(
            prompt,
            runtime.asset,
            runtime.backend,
            max_tokens=max_tokens,
        )

    async def _review_with_backend(
        self,
        prompt: str,
        asset,
        backend: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        normalized_backend = backend.lower().strip()
        if normalized_backend == "lmstudio":
            return await self._invoke_with_optional_max_tokens(
                self._review_with_lmstudio,
                prompt,
                model_id=asset.model_id,
                max_tokens=max_tokens,
            )
        if asset.path is None:
            raise RuntimeError("review_model_not_installed")
        if normalized_backend == "mlx":
            if not asset.path.is_dir():
                raise RuntimeError("mlx_backend_requires_model_directory")
            return await self._invoke_with_optional_max_tokens(
                self._review_with_mlx,
                prompt,
                str(asset.path),
                max_tokens=max_tokens,
            )
        if normalized_backend in {"llama_cpp", "llamacpp", "gguf"}:
            return await self._invoke_with_optional_max_tokens(
                self._review_with_llama_cpp,
                prompt,
                str(asset.path),
                max_tokens=max_tokens,
            )
        raise RuntimeError(f"unsupported_review_backend:{normalized_backend}")

    @staticmethod
    async def _invoke_with_optional_max_tokens(func, *args, max_tokens: Optional[int] = None, **kwargs):
        if max_tokens is None:
            return await func(*args, **kwargs)
        try:
            return await func(*args, max_tokens=max_tokens, **kwargs)
        except TypeError as exc:
            if "max_tokens" not in str(exc):
                raise
            return await func(*args, **kwargs)

    async def _review_with_lmstudio(
        self,
        prompt: str,
        *,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": model_id or settings.REVIEW_MODEL,
            "messages": [
                {"role": "system", "content": "关闭思考。你只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.REVIEW_TEMPERATURE,
            "max_tokens": int(max_tokens or settings.REVIEW_MAX_TOKENS),
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{settings.LMSTUDIO_BASE_URL.rstrip('/')}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _review_with_mlx(
        self,
        prompt: str,
        model_path: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except Exception as first_exc:
            try:
                user_site = site.getusersitepackages()
            except Exception:
                user_site = ""
            if user_site and user_site not in sys.path:
                sys.path.append(user_site)
                importlib.invalidate_caches()
            try:
                from mlx_lm import generate, load
                from mlx_lm.sample_utils import make_sampler
            except Exception as exc:
                raise RuntimeError("mlx_lm_not_installed") from first_exc or exc

        def _run() -> str:
            if self._mlx_model is None or self._mlx_tokenizer is None or self._mlx_model_path != model_path:
                self._mlx_model, self._mlx_tokenizer = load(model_path)
                self._mlx_model_path = model_path
                self.loaded = True

            messages = [
                {"role": "system", "content": "关闭思考。你只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = prompt
            apply_chat_template = getattr(self._mlx_tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                try:
                    formatted_prompt = apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except TypeError:
                    formatted_prompt = apply_chat_template(messages, tokenize=False)

            return generate(
                self._mlx_model,
                self._mlx_tokenizer,
                formatted_prompt,
                max_tokens=int(max_tokens or settings.REVIEW_MAX_TOKENS),
                sampler=make_sampler(temp=settings.REVIEW_TEMPERATURE),
                verbose=False,
            )

        return await asyncio.to_thread(_run)

    async def _review_with_llama_cpp(
        self,
        prompt: str,
        model_path: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("llama_cpp_not_installed") from exc

        def _run() -> str:
            llm = Llama(model_path=model_path, n_ctx=settings.REVIEW_NUM_CTX, verbose=False)
            output = llm(
                prompt,
                max_tokens=int(max_tokens or settings.REVIEW_MAX_TOKENS),
                temperature=settings.REVIEW_TEMPERATURE,
                stop=["\n\n\n"],
            )
            return output["choices"][0]["text"]

        return await asyncio.to_thread(_run)

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict]:
        if not raw:
            return None
        decoder = json.JSONDecoder()
        candidate_starts = [index for index, char in enumerate(raw) if char == "{"]
        for start in candidate_starts:
            fragment = raw[start:].strip()
            try:
                payload, _ = decoder.raw_decode(fragment)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and (
                    isinstance(payload.get("entities"), list)
                    or isinstance(payload.get("rejects"), list)
                    or isinstance(payload.get("entity_decisions"), list)
                    or isinstance(payload.get("decisions"), list)
                    or isinstance(payload.get("missing_entities"), list)
                    or isinstance(payload.get("add_missing_entities"), list)
                    or isinstance(payload.get("action"), str)
                    or isinstance(payload.get("requires_manual_review"), bool)
                )
            ):
                payload.setdefault("entities", [])
                payload.setdefault("rejects", [])
                return payload

        # Some local models wrap JSON in fenced blocks or append prose between
        # objects. Try smaller brace-delimited spans before giving up.
        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, flags=re.DOTALL):
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and (
                    isinstance(payload.get("entities"), list)
                    or isinstance(payload.get("rejects"), list)
                    or isinstance(payload.get("entity_decisions"), list)
                    or isinstance(payload.get("decisions"), list)
                    or isinstance(payload.get("missing_entities"), list)
                    or isinstance(payload.get("add_missing_entities"), list)
                    or isinstance(payload.get("action"), str)
                    or isinstance(payload.get("requires_manual_review"), bool)
                )
            ):
                payload.setdefault("entities", [])
                payload.setdefault("rejects", [])
                return payload

        salvaged_entities: list[Dict] = []
        for match in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
            try:
                item = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            if (
                isinstance(item, dict)
                and isinstance(item.get("type"), str)
                and isinstance(item.get("text"), str)
            ):
                salvaged_entities.append(item)
        if salvaged_entities:
            return {"entities": salvaged_entities}
        return None

    def _materialize_candidates(
        self,
        full_text: str,
        payload: Dict,
        snippet: RiskSnippet,
        *,
        source_name: str = "qwen_fragment_review",
    ) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        self._last_materialize_span_miss_count = 0
        self._last_materialize_gate_reject_count = 0
        is_discovery = self._is_qwen_coverage_discovery_snippet(snippet)
        for item in payload.get("entities", []):
            if isinstance(item, str):
                raw_text = item.strip()
                entity_type = self._infer_string_candidate_type(raw_text)
                role = None
                canonical = None
                source_decision = ""
                entity_decision_original_text = ""
                entity_decision_reason = ""
            elif isinstance(item, dict):
                entity_type = str(item.get("type") or "").strip().upper()
                raw_text = str(item.get("text") or "").strip()
                role = item.get("role")
                canonical = item.get("canonical")
                subject_id = str(item.get("subject_id") or "").strip()
                canonical_subject_id = str(item.get("canonical_subject_id") or "").strip()
                source_decision = str(item.get("source_decision") or "").strip()
                entity_decision_original_text = str(item.get("entity_decision_original_text") or "").strip()
                entity_decision_reason = str(item.get("entity_decision_reason") or "").strip()
            else:
                continue
            if isinstance(item, str):
                subject_id = ""
                canonical_subject_id = ""
            if not entity_type or not raw_text:
                continue
            entity_type = self._normalize_entity_type(entity_type, raw_text, role=role, snippet=snippet)
            if not entity_type:
                self._last_materialize_gate_reject_count += 1
                continue
            materialized_text, span = self._materialize_candidate_span(
                full_text,
                raw_text,
                snippet,
                entity_type,
            )
            if span is None or not materialized_text:
                self._last_materialize_span_miss_count += 1
                continue
            if not self._looks_like_review_candidate(entity_type, materialized_text):
                self._last_materialize_gate_reject_count += 1
                continue
            gate_passed, _ = subject_noun_gate(entity_type, materialized_text, allow_short_org=True)
            review_short_org = (
                entity_type == "ORGANIZATION"
                and looks_like_organization_short_name(materialized_text)
                and self._snippet_supports_short_org_resolution(snippet, materialized_text)
            )
            if not gate_passed and not review_short_org:
                self._last_materialize_gate_reject_count += 1
                continue
            metadata = {
                "source": source_name,
                "snippet_type": snippet.snippet_type,
                "risk_reason": snippet.risk_reason,
                "role": role,
                "canonical": canonical,
                "review_subject_id": subject_id or None,
                "review_canonical_subject_id": canonical_subject_id or subject_id or None,
                "review": source_name in REVIEW_DECISION_SOURCE_NAMES,
                "source_layer": (
                    "llm_review"
                    if source_name in REVIEW_DECISION_SOURCE_NAMES
                    else ""
                ),
            }
            if source_decision:
                metadata["entity_decision"] = True
                metadata["source_decision"] = source_decision
                metadata["entity_decision_action"] = source_decision
            if entity_decision_original_text:
                metadata["entity_decision_original_text"] = entity_decision_original_text
            if entity_decision_reason:
                metadata["entity_decision_reason"] = entity_decision_reason
            if materialized_text != raw_text:
                metadata["materialized_from"] = raw_text
            if is_discovery:
                metadata["qwen_coverage_discovery"] = True
                target = getattr(snippet, "target_entity", None)
                target_metadata = (
                    target.get("metadata")
                    if isinstance(target, dict) and isinstance(target.get("metadata"), dict)
                    else {}
                )
                target_snapshot = {
                    key: target_metadata.get(key)
                    for key in (
                        "docx_unit_id",
                        "docx_container_type",
                        "docx_part_name",
                        "docx_table_index",
                        "docx_row_index",
                        "docx_cell_index",
                        "span_resolution",
                    )
                    if target_metadata.get(key) is not None
                }
                if target_snapshot:
                    metadata["qwen_discovery_target"] = target_snapshot
            result = make_entity(
                text=full_text,
                start=span[0],
                end=span[1],
                entity_type=entity_type,
                source=source_name,
                score=0.87,
                metadata=metadata,
            )
            if result:
                results.append(result)
        results.extend(
            self._supplement_missing_short_org_list_entities(
                full_text=full_text,
                payload=payload,
                snippet=snippet,
                existing_results=results,
                source_name=source_name,
            )
        )
        return results

    def _filter_missing_candidate_payload(
        self,
        full_text: str,
        payload: Dict,
        snippet: RiskSnippet,
    ) -> Dict[str, list]:
        target = getattr(snippet, "target_entity", None)
        if not isinstance(target, dict) or not target:
            return {"entities": []}
        target_text = str(target.get("text") or "").strip()
        target_type = self._normalize_entity_type(
            str(target.get("type") or target.get("entity_type") or "").upper(),
            target_text,
            role=None,
            snippet=snippet,
        )
        try:
            target_start = int(target.get("start") or 0)
            target_end = int(target.get("end") or 0)
        except (TypeError, ValueError):
            target_start = 0
            target_end = 0
        if not target_text or not target_type or target_end <= target_start:
            return {"entities": []}
        target_norm = normalize_entity_text(target_text)
        allowed: list[dict] = []
        for item in payload.get("entities") or []:
            if isinstance(item, str):
                candidate_text = item.strip()
                candidate_type = target_type
                role = None
            elif isinstance(item, dict):
                candidate_text = str(item.get("text") or "").strip()
                candidate_type = self._normalize_entity_type(
                    str(item.get("type") or target_type or "").upper(),
                    candidate_text,
                    role=item.get("role"),
                    snippet=snippet,
                )
                role = item.get("role")
            else:
                continue
            if not candidate_text or candidate_type != target_type:
                continue
            candidate_norm = normalize_entity_text(candidate_text)
            if not candidate_norm:
                continue
            if not (
                candidate_norm == target_norm
                or candidate_norm in target_norm
                or target_norm in candidate_norm
            ):
                continue
            span = find_value_span(
                full_text,
                candidate_text,
                search_start=max(0, snippet.start),
                search_end=min(len(full_text), snippet.end),
            )
            if span is None:
                continue
            if not (span[0] < target_end and span[1] > target_start):
                continue
            allowed.append(
                {
                    "type": candidate_type,
                    "text": candidate_text,
                    "role": role,
                    "source_decision": "missing_candidate_confirm",
                    "entity_decision_reason": "missing_candidate_review_target_matched",
                }
            )
        return {"entities": allowed}

    def _supplement_missing_short_org_list_entities(
        self,
        *,
        full_text: str,
        payload: Dict,
        snippet: RiskSnippet,
        existing_results: List[RecognizerResult],
        source_name: str,
    ) -> List[RecognizerResult]:
        list_groups = self._collect_short_org_list_groups(snippet.text)
        if not list_groups:
            return []

        present_norms = {
            normalize_entity_text(item.text)
            for item in existing_results
            if item.entity_type in {"ORGANIZATION", "GOVERNMENT"}
        }
        any_present_norms = {
            normalize_entity_text(item.text)
            for item in existing_results
            if normalize_entity_text(item.text)
        }
        reject_norms: set[str] = set()
        for item in payload.get("rejects") or []:
            if isinstance(item, str):
                normalized = normalize_entity_text(item)
            elif isinstance(item, dict):
                normalized = normalize_entity_text(str(item.get("text") or ""))
            else:
                normalized = ""
            if normalized:
                reject_norms.add(normalized)

        supplements: list[RecognizerResult] = []
        for group in list_groups:
            label = str(group.get("label") or "")
            items = [str(item).strip() for item in (group.get("items") or []) if str(item).strip()]
            if len(items) < 2:
                continue
            normalized_items = [normalize_entity_text(item) for item in items if normalize_entity_text(item)]
            if not normalized_items:
                continue
            group_has_surface_signal = any(norm in any_present_norms for norm in normalized_items)
            strong_label = self._is_strong_short_org_list_label(label)
            if not strong_label and not group_has_surface_signal:
                continue
            for item in items:
                normalized = normalize_entity_text(item)
                if not normalized or normalized in present_norms or normalized in reject_norms:
                    continue
                materialized_text, span = self._materialize_candidate_span(
                    full_text,
                    item,
                    snippet,
                    "ORGANIZATION",
                )
                if span is None or not materialized_text:
                    continue
                if not self._looks_like_review_candidate("ORGANIZATION", materialized_text):
                    continue
                metadata = {
                    "source": source_name,
                    "snippet_type": snippet.snippet_type,
                    "risk_reason": snippet.risk_reason,
                    "review": source_name in REVIEW_DECISION_SOURCE_NAMES,
                    "source_layer": (
                        "llm_review"
                        if source_name in REVIEW_DECISION_SOURCE_NAMES
                        else ""
                    ),
                    "materialized_from": item,
                    "list_completion": True,
                    "list_label": label,
                    "short_org_candidate": True,
                    "identity_surface": normalize_entity_text(materialized_text),
                }
                if self._is_qwen_coverage_discovery_snippet(snippet):
                    metadata["qwen_coverage_discovery"] = True
                result = make_entity(
                    text=full_text,
                    start=span[0],
                    end=span[1],
                    entity_type="ORGANIZATION",
                    source=source_name,
                    score=0.87,
                    metadata=metadata,
                )
                if result:
                    supplements.append(result)
                    present_norms.add(normalized)
                    any_present_norms.add(normalized)
        return supplements

    def _materialize_candidate_span(
        self,
        full_text: str,
        raw_text: str,
        snippet: RiskSnippet,
        entity_type: str,
    ) -> tuple[str, Optional[tuple[int, int]]]:
        for candidate_text in self._build_materialization_candidates(raw_text, entity_type, snippet):
            span = find_value_span(
                full_text,
                candidate_text,
                search_start=snippet.start,
                search_end=snippet.end,
            )
            if span is None:
                continue
            exact_raw_span = find_value_span(
                full_text,
                raw_text,
                search_start=snippet.start,
                search_end=snippet.end,
            )
            expanded_span = expand_subject_span_to_containing_shape(
                full_text,
                span[0],
                span[1],
                entity_type,
            )
            if expanded_span is not None:
                expanded_text = full_text[expanded_span[0] : expanded_span[1]]
                if exact_raw_span is not None and self._expanded_span_adds_context_noise(
                    expanded_text,
                    raw_text,
                    entity_type,
                ):
                    span = exact_raw_span
                else:
                    span = expanded_span
            return full_text[span[0] : span[1]], span
        return "", None

    @staticmethod
    def _expanded_span_adds_context_noise(expanded_text: str, raw_text: str, entity_type: str) -> bool:
        if str(entity_type or "").upper() not in {"ORGANIZATION", "COMPANY_NAME", "GOVERNMENT", "COURT"}:
            return False
        expanded_norm = normalize_entity_text(expanded_text)
        raw_norm = normalize_entity_text(raw_text)
        if not expanded_norm or not raw_norm or expanded_norm == raw_norm:
            return False
        raw_index = expanded_norm.find(raw_norm)
        if raw_index <= 0:
            return False
        left = expanded_norm[:raw_index]
        return bool(
            left
            and (
                left[-1] in "（）()《》【】"
                or left in {
                    "甲方",
                    "乙方",
                    "丙方",
                    "丁方",
                    "委托方",
                    "受托方",
                    "发包人",
                    "承包人",
                    "采购人",
                    "供应商",
                    "上诉人",
                    "被上诉人",
                    "原告",
                    "被告",
                    "第三人",
                    "申请人",
                    "被申请人",
                }
                or is_non_subject_action_or_function_term(left)
                or re.search(
                    r"(?:通过|经过|根据|依据|按照|经由|并由|由|向|对|与|和|及|以及|合同由|协议由|材料由)$",
                    left,
                )
            )
        )

    def _build_materialization_candidates(
        self,
        raw_text: str,
        entity_type: str,
        snippet: RiskSnippet,
    ) -> List[str]:
        cleaned = clean_candidate_text(raw_text)
        if not cleaned:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def _append(value: str) -> None:
            candidate = clean_candidate_text(value)
            if entity_type == "ORGANIZATION" and not ORG_PATTERN.search(normalize_entity_text(candidate)):
                trimmed_candidate = self._trim_short_org_candidate_core(candidate)
                if trimmed_candidate:
                    candidate = trimmed_candidate
            normalized = normalize_entity_text(candidate)
            if not candidate or not normalized or normalized in seen:
                return
            if entity_type == "ORGANIZATION" and is_non_subject_generic_org_reference(normalized):
                return
            if entity_type == "ORGANIZATION" and is_non_subject_action_or_function_term(normalized):
                return
            if entity_type == "ORGANIZATION" and not ORG_PATTERN.search(normalized) and self._is_noisy_short_org_candidate(normalized):
                return
            seen.add(normalized)
            candidates.append(candidate)

        if entity_type == "ORGANIZATION":
            prefixed_remainder = strip_identity_reference_prefix(cleaned)
            if prefixed_remainder:
                if is_generic_organization_term(prefixed_remainder):
                    return []
                _append(prefixed_remainder)
            function_stripped, consumed = strip_leading_subject_function_words(cleaned)
            if consumed > 0:
                function_stripped_normalized = normalize_entity_text(function_stripped)
                if function_stripped_normalized and not is_non_subject_generic_org_reference(function_stripped_normalized):
                    gate_type = "GOVERNMENT" if is_official_institution_text(function_stripped_normalized) else "ORGANIZATION"
                    if (
                        not is_weak_function_stripped_org(function_stripped)
                        and subject_noun_gate(gate_type, function_stripped, allow_short_org=False)[0]
                    ):
                        _append(function_stripped)
                return candidates

            if is_org_like_text(cleaned) and not self._looks_like_wrapped_org_candidate(cleaned):
                _append(cleaned)

            contextual_short_candidates = self._collect_contextual_short_org_candidates(cleaned)
            for candidate in contextual_short_candidates:
                _append(candidate)
            for match in ORG_PATTERN.finditer(cleaned):
                _append(match.group(0))
            cleaned_normalized = normalize_entity_text(cleaned)
            for candidate in self._collect_contextual_short_org_candidates(snippet.text):
                candidate_normalized = normalize_entity_text(candidate)
                if not candidate_normalized:
                    continue
                if (
                    candidate_normalized == cleaned_normalized
                    or candidate_normalized in cleaned_normalized
                    or cleaned_normalized in candidate_normalized
                ):
                    _append(candidate)
            if not prefixed_remainder and not is_org_like_text(cleaned) and not self._looks_like_wrapped_org_candidate(cleaned, contextual_short_candidates):
                _append(cleaned)
            return candidates

        if not self._looks_like_narrative_wrapped_candidate(cleaned):
            _append(cleaned)
        _append(cleaned)
        return candidates

    @classmethod
    def _looks_like_narrative_wrapped_candidate(cls, text: str) -> bool:
        normalized = normalize_entity_text(text)
        if not normalized:
            return False
        if cls.NARRATIVE_PREFIX_PATTERN.match(normalized):
            return True
        return any(token in normalized for token in cls.NARRATIVE_ACTION_TOKENS)

    @classmethod
    def _looks_like_wrapped_org_candidate(
        cls,
        text: str,
        contextual_short_candidates: Optional[List[str]] = None,
    ) -> bool:
        normalized = normalize_entity_text(text)
        if not normalized:
            return False
        if cls._looks_like_narrative_wrapped_candidate(text):
            return True
        prefixed_remainder = strip_identity_reference_prefix(normalized)
        if prefixed_remainder:
            return True
        for candidate in contextual_short_candidates or []:
            candidate_normalized = normalize_entity_text(candidate)
            if candidate_normalized and candidate_normalized != normalized and candidate_normalized in normalized:
                return True
        return False

    @staticmethod
    def _materialize_rejections(payload: Dict, existing_entities: list[dict]) -> List[Dict]:
        rejected: list[Dict] = []
        raw_rejects = payload.get("rejects") or payload.get("rejections") or payload.get("invalid_entities") or []
        if not isinstance(raw_rejects, list):
            return rejected

        for item in raw_rejects[:16]:
            if isinstance(item, str):
                reject_text = item.strip()
                reject_type = ""
                reason = ""
            elif isinstance(item, dict):
                reject_text = str(item.get("text") or "").strip()
                reject_type = str(item.get("type") or "").strip().upper()
                reason = str(item.get("reason") or "").strip()
            else:
                continue

            if not reject_text:
                continue

            expected_type = QwenFragmentReviewService._expected_corrected_type_from_reason(reason)
            if expected_type and not QwenFragmentReviewService._payload_has_corrected_entity(
                payload,
                reject_text,
                expected_type,
            ):
                continue

            matched_existing = None
            normalized_reject_text = normalize_entity_text(reject_text)
            for existing in existing_entities:
                existing_text = str(existing.get("text") or "")
                existing_type = str(existing.get("type") or "").upper()
                if reject_type and existing_type != reject_type:
                    continue
                if reject_text != existing_text:
                    continue
                matched_existing = existing
                break
            if matched_existing is None and QwenFragmentReviewService._allow_type_mismatch_rejection(
                reject_text,
                reject_type,
                reason,
            ):
                for existing in existing_entities:
                    if reject_text == str(existing.get("text") or ""):
                        matched_existing = existing
                        break
            if matched_existing is None and normalized_reject_text:
                for existing in existing_entities:
                    existing_text = str(existing.get("text") or "")
                    existing_type = str(existing.get("type") or "").upper()
                    if reject_type and existing_type != reject_type and not QwenFragmentReviewService._allow_type_mismatch_rejection(
                        reject_text,
                        reject_type,
                        reason,
                    ):
                        continue
                    if normalize_entity_text(existing_text) != normalized_reject_text:
                        continue
                    matched_existing = existing
                    break
            if matched_existing is None:
                continue
            rejected.append(
                {
                    "text": str(matched_existing.get("text") or ""),
                    "type": str(matched_existing.get("type") or "").upper(),
                    "start": int(matched_existing.get("start") or 0),
                    "end": int(matched_existing.get("end") or 0),
                    "source": matched_existing.get("source"),
                    "reason": reason,
                }
            )
        return rejected

    @staticmethod
    def _materialize_entity_decisions(payload: Dict, existing_entities: list[dict]) -> Dict[str, list]:
        """Turn model review decisions for existing entities into executable edits."""
        rows = payload.get("entity_decisions") or payload.get("entity_decision") or []
        if not isinstance(rows, list):
            return {"rejections": [], "entities": []}

        rejections: list[Dict] = []
        entities: list[Dict] = []
        for row in rows[:24]:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip().lower()
            if action not in {"reject", "trim_to", "split_into", "keep"}:
                continue
            matched = QwenFragmentReviewService._match_existing_entity_decision(row, existing_entities)
            if matched is None:
                continue
            if action == "keep":
                continue
            matched_text = str(matched.get("text") or "")
            reason = str(row.get("reason") or f"entity_decision_{action}").strip()
            rejections.append(
                {
                    "text": matched_text,
                    "type": str(matched.get("type") or "").upper(),
                    "start": int(matched.get("start") or 0),
                    "end": int(matched.get("end") or 0),
                    "source": matched.get("source"),
                    "reason": reason,
                    "entity_decision": True,
                    "entity_decision_action": action,
                }
            )
            if action == "reject":
                continue
            corrected_items: list[object]
            if action == "trim_to":
                corrected_items = [row.get("trim_to") or row.get("text_after") or row.get("correct_text")]
            else:
                corrected_items = row.get("split_into") or row.get("entities") or row.get("items") or []
                if not isinstance(corrected_items, list):
                    corrected_items = []
            for corrected in corrected_items:
                if isinstance(corrected, str):
                    corrected_text = corrected.strip()
                    corrected_type = str(row.get("correct_type") or row.get("type") or matched.get("type") or "").upper()
                    role = row.get("role")
                elif isinstance(corrected, dict):
                    corrected_text = str(corrected.get("text") or corrected.get("trim_to") or "").strip()
                    corrected_type = str(corrected.get("type") or row.get("correct_type") or row.get("type") or matched.get("type") or "").upper()
                    role = corrected.get("role") or row.get("role")
                else:
                    continue
                if not corrected_text:
                    continue
                corrected_type = canonical_default_entity_type(
                    QwenFragmentReviewService.TYPE_ALIASES.get(corrected_type, corrected_type),
                    corrected_text,
                )
                if corrected_type not in QwenFragmentReviewService.ALLOWED_TYPES:
                    corrected_type = QwenFragmentReviewService._infer_static_string_candidate_type(corrected_text)
                if corrected_type not in QwenFragmentReviewService.ALLOWED_TYPES:
                    continue
                if not subject_noun_gate(corrected_type, corrected_text, allow_short_org=False)[0]:
                    continue
                if corrected_type == "ORGANIZATION" and is_weak_function_stripped_org(corrected_text):
                    original_text = str(row.get("text") or row.get("original_text") or matched_text or "")
                    _, consumed = strip_leading_subject_function_words(original_text)
                    if consumed > 0:
                        continue
                entities.append(
                    {
                        "type": corrected_type,
                        "text": corrected_text,
                        "role": role,
                        "source_decision": action,
                        "entity_decision_action": action,
                        "entity_decision_original_text": matched_text,
                        "entity_decision_reason": reason,
                    }
                )
        return {"rejections": rejections, "entities": entities}

    @staticmethod
    def _match_existing_entity_decision(row: dict, existing_entities: list[dict]) -> Optional[dict]:
        row_text = str(row.get("text") or row.get("original_text") or "").strip()
        row_type = str(row.get("type") or "").strip().upper()
        row_start = row.get("start")
        row_end = row.get("end")
        try:
            row_start_int = int(row_start)
            row_end_int = int(row_end)
        except (TypeError, ValueError):
            row_start_int = -1
            row_end_int = -1
        normalized_text = normalize_entity_text(row_text)
        for existing in existing_entities or []:
            if not isinstance(existing, dict):
                continue
            existing_type = str(existing.get("type") or existing.get("entity_type") or "").upper()
            if row_type and existing_type != row_type:
                continue
            if row_start_int >= 0 and row_end_int >= 0:
                if int(existing.get("start") or 0) == row_start_int and int(existing.get("end") or 0) == row_end_int:
                    return existing
            existing_text = str(existing.get("text") or "")
            if row_text and existing_text == row_text:
                return existing
            if normalized_text and normalize_entity_text(existing_text) == normalized_text:
                return existing
        return None

    @staticmethod
    def _deterministic_review_rejections(existing_entities: list[dict]) -> List[Dict]:
        rejected: list[Dict] = []
        for existing in existing_entities or []:
            if not isinstance(existing, dict):
                continue
            reason = QwenFragmentReviewService._deterministic_rejection_reason(existing)
            if not reason:
                continue
            rejected.append(
                {
                    "text": str(existing.get("text") or ""),
                    "type": str(existing.get("type") or existing.get("entity_type") or "").upper(),
                    "start": int(existing.get("start") or 0),
                    "end": int(existing.get("end") or 0),
                    "source": existing.get("source"),
                    "reason": reason,
                    "deterministic_review": True,
                }
            )
        return rejected

    @staticmethod
    def _deterministic_review_entity_decisions(existing_entities: list[dict]) -> Dict[str, list]:
        """Materialize obvious review edits even when the local model misses them."""
        rejections: list[Dict] = []
        entities: list[Dict] = []
        seen_rejections: set[tuple[str, str, int, int, str]] = set()
        seen_entities: set[tuple[str, str, str, str]] = set()

        def _append_rejection(existing: dict, reason: str, action: str) -> None:
            text = str(existing.get("text") or "")
            entity_type = QwenFragmentReviewService._review_subject_type(existing)
            key = (
                entity_type,
                text,
                int(existing.get("start") or 0),
                int(existing.get("end") or 0),
                action,
            )
            if key in seen_rejections:
                return
            seen_rejections.add(key)
            rejections.append(
                {
                    "text": text,
                    "type": entity_type,
                    "start": int(existing.get("start") or 0),
                    "end": int(existing.get("end") or 0),
                    "source": existing.get("source"),
                    "reason": reason,
                    "deterministic_review": True,
                    "deterministic_entity_decision": True,
                    "entity_decision": True,
                    "entity_decision_action": action,
                }
            )

        def _append_entity(
            entity_type: str,
            text: str,
            *,
            original_text: str,
            action: str,
            reason: str,
        ) -> None:
            normalized_text = normalize_entity_text(text)
            normalized_original = normalize_entity_text(original_text)
            key = (entity_type, normalized_text, action, normalized_original)
            if not normalized_text or key in seen_entities:
                return
            seen_entities.add(key)
            entities.append(
                {
                    "type": entity_type,
                    "text": text,
                    "source_decision": action,
                    "entity_decision_action": action,
                    "entity_decision_original_text": original_text,
                    "entity_decision_reason": reason,
                }
            )

        for existing in existing_entities or []:
            if not isinstance(existing, dict):
                continue
            text = str(existing.get("text") or "").strip()
            entity_type = QwenFragmentReviewService._review_subject_type(existing)
            if not text or entity_type not in QwenFragmentReviewService.ALLOWED_TYPES:
                continue
            normalized = normalize_entity_text(text)

            reason = QwenFragmentReviewService._deterministic_final_subject_rejection_reason(existing)
            if reason:
                _append_rejection(existing, reason, "reject")
                continue

            if entity_type in {"ORGANIZATION", "GOVERNMENT"}:
                reason = QwenFragmentReviewService._deterministic_rule_organization_rejection_reason(existing)
                if reason:
                    _append_rejection(existing, reason, "reject")
                    continue

                stripped, consumed = strip_leading_subject_function_words(text)
                stripped_normalized = normalize_entity_text(stripped)
                if consumed > 0:
                    gate_type = (
                        "GOVERNMENT"
                        if is_official_institution_text(stripped_normalized)
                        else "ORGANIZATION"
                    )
                    reason = "deterministic_trim_leading_function_word"
                    if (
                        stripped_normalized
                        and not is_non_subject_generic_org_reference(stripped_normalized)
                        and not is_generic_organization_term(stripped_normalized)
                        and not (
                            gate_type == "ORGANIZATION"
                            and is_weak_function_stripped_org(stripped)
                        )
                        and subject_noun_gate(gate_type, stripped, allow_short_org=False)[0]
                    ):
                        _append_rejection(existing, reason, "trim_to")
                        _append_entity(
                            gate_type,
                            stripped,
                            original_text=text,
                            action="trim_to",
                            reason=reason,
                        )
                    else:
                        _append_rejection(existing, "deterministic_reject_leading_function_word", "reject")
                    continue

                split_items = QwenFragmentReviewService._deterministic_split_parallel_subjects(
                    text,
                    entity_type,
                )
                if split_items:
                    reason = "deterministic_split_parallel_subjects"
                    _append_rejection(existing, reason, "split_into")
                    for split_type, split_text in split_items:
                        _append_entity(
                            split_type,
                            split_text,
                            original_text=text,
                            action="split_into",
                            reason=reason,
                        )
                    continue

                trimmed_subject = QwenFragmentReviewService._deterministic_trim_dirty_subject_text(text, entity_type)
                if trimmed_subject and normalize_entity_text(trimmed_subject) != normalized:
                    trimmed_type = (
                        "GOVERNMENT"
                        if is_official_institution_text(trimmed_subject)
                        else canonical_default_entity_type(entity_type, trimmed_subject)
                    )
                    if trimmed_type in QwenFragmentReviewService.ALLOWED_TYPES and subject_noun_gate(
                        trimmed_type,
                        trimmed_subject,
                        allow_short_org=trimmed_type == "ORGANIZATION",
                    )[0]:
                        reason = "deterministic_trim_non_subject_boundary"
                        _append_rejection(existing, reason, "trim_to")
                        _append_entity(
                            trimmed_type,
                            trimmed_subject,
                            original_text=text,
                            action="trim_to",
                            reason=reason,
                        )
                        continue

            reason = QwenFragmentReviewService._deterministic_rejection_reason(existing)
            if reason:
                _append_rejection(existing, reason, "reject")
                continue

        return {"rejections": rejections, "entities": entities}

    @staticmethod
    def _deterministic_final_subject_rejection_reason(existing: dict) -> str:
        text = str(existing.get("text") or "").strip()
        normalized = normalize_entity_text(text)
        entity_type = QwenFragmentReviewService._review_subject_type(existing)
        if not normalized or entity_type not in QwenFragmentReviewService.ALLOWED_TYPES:
            return ""
        if entity_type == "GOVERNMENT" and normalized in GENERIC_GOVERNMENT_REVIEW_REFERENCES:
            return "deterministic_final_subject_generic_government_reference"
        if is_identity_reference_term(normalized):
            return "deterministic_final_subject_identity_reference"
        if is_position_title(normalized):
            return "deterministic_final_subject_position_title"
        if normalized in QwenFragmentReviewService.NON_ENTITY_TERMS:
            return "deterministic_final_subject_non_entity_term"
        if entity_type in {"PERSON", "ORGANIZATION", "GOVERNMENT"} and is_non_subject_action_or_function_term(normalized):
            return "deterministic_final_subject_action_or_function"
        if entity_type == "PERSON":
            if is_org_like_text(normalized) or is_official_institution_text(normalized) or subject_noun_gate("LOCATION", normalized)[0]:
                return "deterministic_final_subject_person_type_mismatch"
            if len(normalized) > 8 and any(
                token in normalized
                for token in ("请求", "认为", "提交", "证明", "负责", "联系", "地址", "法院", "公司", "材料", "情况")
            ):
                return "deterministic_final_subject_person_narrative_fragment"
        if entity_type == "LOCATION":
            if is_org_like_text(normalized) or is_official_institution_text(normalized):
                return "deterministic_final_subject_location_type_mismatch"
            if is_probable_person(normalized):
                return "deterministic_final_subject_location_person_mismatch"
            if len(normalized) > 30 and any(token in normalized for token in ("请求", "认为", "提交", "证明", "负责", "联系")):
                return "deterministic_final_subject_location_narrative_fragment"
        if entity_type == "GOVERNMENT":
            if not is_official_institution_text(normalized) and not subject_noun_gate("GOVERNMENT", normalized)[0]:
                return "deterministic_final_subject_weak_government_shape"
        if entity_type == "ORGANIZATION":
            if is_generic_organization_term(normalized):
                return "deterministic_final_subject_generic_organization_term"
            if is_probable_person(normalized) and not is_org_like_text(normalized):
                return "deterministic_final_subject_organization_person_mismatch"
        return ""

    @staticmethod
    def _review_subject_type(existing: dict) -> str:
        text = str(existing.get("text") or "").strip()
        raw_type = str(existing.get("type") or existing.get("entity_type") or "").strip().upper()
        entity_type = canonical_default_entity_type(raw_type, text)
        if entity_type in QwenFragmentReviewService.ALLOWED_TYPES:
            return entity_type
        normalized = normalize_entity_text(text)
        if raw_type in {"GOVERNMENT", "GOVERNMENT_AGENCY", "COURT", "BANK_NAME"}:
            return "GOVERNMENT"
        if raw_type in {"PERSON", "PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}:
            return "PERSON"
        if raw_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            return "GOVERNMENT" if is_official_institution_text(normalized) else "ORGANIZATION"
        if raw_type in {"LOCATION", "ADDRESS"}:
            return "LOCATION"
        return ""

    @staticmethod
    def _deterministic_rule_organization_rejection_reason(existing: dict) -> str:
        text = str(existing.get("text") or "").strip()
        entity_type = canonical_default_entity_type(
            str(existing.get("type") or existing.get("entity_type") or "").upper(),
            text,
        )
        if entity_type not in {"ORGANIZATION", "GOVERNMENT"}:
            return ""
        source = str(existing.get("source") or "").strip()
        metadata = dict(existing.get("metadata") or {})
        rule_first = metadata.get("rule_first")
        candidate_id = str(rule_first.get("candidate_id") or "") if isinstance(rule_first, dict) else ""
        if source not in {"rule_organization", "rule_organization_context"} and not candidate_id.startswith("rule_organization"):
            return ""

        normalized = normalize_entity_text(text)
        if not normalized:
            return "deterministic_rule_org_empty"
        if is_official_institution_text(normalized):
            return ""
        if is_non_subject_generic_org_reference(normalized):
            return "deterministic_rule_org_generic_reference"
        if is_non_subject_action_or_function_term(normalized):
            return "deterministic_rule_org_action_or_function"
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return "deterministic_rule_org_role_or_title"
        if is_generic_organization_term(normalized):
            return "deterministic_rule_org_generic_term"

        strong_suffix = any(normalized.endswith(suffix) for suffix in RULE_ORG_REVIEW_STRONG_SUFFIXES)
        if source == "rule_organization_context" and not strong_suffix:
            if QwenFragmentReviewService._looks_like_non_subject_rule_org_phrase(normalized):
                return "deterministic_rule_org_context_non_subject_phrase"
            return ""

        weak_suffix = next((suffix for suffix in RULE_ORG_REVIEW_WEAK_SUFFIXES if normalized.endswith(suffix)), "")
        if weak_suffix:
            stem = normalized[: -len(weak_suffix)]
            if not stem or QwenFragmentReviewService._looks_like_non_subject_rule_org_phrase(stem):
                return "deterministic_rule_org_weak_suffix_narrative"
            if len(stem) < 4 and not re.search(r"(?:省|自治区|特别行政区|市|地区|自治州|盟|区|县|旗)", stem):
                return "deterministic_rule_org_weak_suffix_low_evidence"

        if not strong_suffix and QwenFragmentReviewService._looks_like_non_subject_rule_org_phrase(normalized):
            return "deterministic_rule_org_non_subject_phrase"
        return ""

    @staticmethod
    def _looks_like_non_subject_rule_org_phrase(normalized: str) -> bool:
        if not normalized:
            return False
        if any(normalized.startswith(prefix) for prefix in RULE_ORG_REVIEW_NON_SUBJECT_PREFIXES):
            return True
        if any(token in normalized for token in RULE_ORG_REVIEW_NON_SUBJECT_TOKENS) and not any(
            normalized.endswith(suffix) for suffix in RULE_ORG_REVIEW_STRONG_SUFFIXES
        ):
            return True
        return False

    @staticmethod
    def _deterministic_trim_dirty_subject_text(text: str, entity_type: str) -> str:
        """Trim obvious non-subject tail from an organization/government candidate."""

        value = str(text or "").strip()
        normalized = normalize_entity_text(value)
        if not normalized or entity_type not in {"ORGANIZATION", "GOVERNMENT"}:
            return ""
        if ORG_PATTERN.fullmatch(normalized) or is_official_institution_text(normalized):
            return ""
        complete_matches: list[tuple[int, int, str]] = []
        for pattern in (OFFICIAL_INSTITUTION_PATTERN, ORG_PATTERN):
            for match in pattern.finditer(value):
                candidate = value[match.start() : match.end()]
                candidate_type = "GOVERNMENT" if is_official_institution_text(candidate) else "ORGANIZATION"
                if subject_noun_gate(candidate_type, candidate, allow_short_org=False)[0]:
                    complete_matches.append((match.start(), match.end(), candidate))
        if complete_matches:
            start, end, candidate = max(
                complete_matches,
                key=lambda item: (
                    item[0] == 0,
                    item[1] - item[0],
                    -item[0],
                ),
            )
            if (start, end) != (0, len(value)):
                return candidate

        short_boundary = find_short_org_prefix_before_non_subject_boundary(value)
        if short_boundary is None:
            return ""
        start, end, _boundary_kind = short_boundary
        if start != 0 or end >= len(value):
            return ""
        candidate = value[start:end]
        if not looks_like_organization_short_name(candidate):
            return ""
        if not subject_noun_gate("ORGANIZATION", candidate, allow_short_org=True)[0]:
            return ""
        return candidate

    @staticmethod
    def _deterministic_split_parallel_subjects(text: str, entity_type: str) -> list[tuple[str, str]]:
        normalized = normalize_entity_text(text)
        if not normalized or len(normalized) > 120:
            return []
        if not re.search(r"[、，,及与和]", normalized):
            return []

        direct_parts = [part for part in re.split(r"[、，,及与和]+", normalized) if part]
        if len(direct_parts) >= 2:
            direct_items: list[tuple[str, str]] = []
            for part in direct_parts[:8]:
                part_type = canonical_default_entity_type(entity_type, part)
                if part_type not in QwenFragmentReviewService.ALLOWED_TYPES:
                    part_type = "GOVERNMENT" if canonical_default_entity_type("ORGANIZATION", part) == "GOVERNMENT" else "ORGANIZATION"
                if not subject_noun_gate(part_type, part, allow_short_org=False)[0]:
                    direct_items = []
                    break
                direct_items.append((part_type, part))
            if len({normalize_entity_text(item[1]) for item in direct_items}) >= 2:
                return direct_items

        matches = [(match.group(0), match.start(), match.end()) for match in ORG_PATTERN.finditer(normalized)]
        if len(matches) < 2:
            return []
        split_items: list[tuple[str, str]] = []
        previous_end = -1
        for matched_text, start, end in matches[:8]:
            stripped_text, consumed = strip_leading_subject_function_words(matched_text)
            if consumed > 0:
                matched_text = stripped_text
            if previous_end >= 0:
                between = normalized[previous_end:start]
                if not between or not re.fullmatch(r"[、，,及与和]+", between):
                    return []
            previous_end = end
            split_type = canonical_default_entity_type(entity_type, matched_text)
            if split_type not in QwenFragmentReviewService.ALLOWED_TYPES:
                split_type = "GOVERNMENT" if canonical_default_entity_type("ORGANIZATION", matched_text) == "GOVERNMENT" else "ORGANIZATION"
            if not subject_noun_gate(split_type, matched_text, allow_short_org=False)[0]:
                return []
            split_items.append((split_type, matched_text))
        if len({normalize_entity_text(item[1]) for item in split_items}) < 2:
            return []
        return split_items

    @staticmethod
    def _deterministic_rejection_reason(existing: dict) -> str:
        text = str(existing.get("text") or "")
        normalized = normalize_entity_text(text)
        entity_type = canonical_default_entity_type(
            str(existing.get("type") or existing.get("entity_type") or "").upper(),
            text,
        )
        if not normalized or entity_type not in {"PERSON", "ORGANIZATION", "LOCATION", "GOVERNMENT"}:
            return ""
        if entity_type in {"PERSON", "ORGANIZATION"} and is_non_subject_action_or_function_term(normalized):
            return "deterministic_non_subject_action_or_function"
        if is_identity_reference_term(normalized):
            return "deterministic_identity_reference_term"
        if is_position_title(normalized):
            return "deterministic_position_title"
        if normalized in QwenFragmentReviewService.NON_ENTITY_TERMS:
            return "deterministic_non_entity_term"
        if entity_type in {"ORGANIZATION", "GOVERNMENT"} and is_generic_organization_term(normalized):
            return "deterministic_generic_organization_term"
        if entity_type == "GOVERNMENT" and normalized in {"法院", "人民法院", "一审法院", "二审法院", "原审法院", "本院", "贵院"}:
            return "deterministic_generic_government_reference"
        if entity_type in {"ORGANIZATION", "GOVERNMENT"} and len(normalized) > 18 and any(
            token in normalized for token in ("不服", "请求", "认为", "提交", "证明", "承担责任", "是否", "通过")
        ):
            return "deterministic_narrative_fragment"
        return ""

    @staticmethod
    def _allow_type_mismatch_rejection(reject_text: str, reject_type: str, reason: str) -> bool:
        normalized = normalize_entity_text(reject_text)
        if not normalized:
            return False
        if is_identity_reference_term(normalized) or is_position_title(normalized):
            return True
        if normalized in QwenFragmentReviewService.NON_ENTITY_TERMS:
            return True
        reason_text = str(reason or "")
        if any(token in reason_text for token in ("普通词", "字段标签", "泛称", "身份", "职务", "角色", "标签", "非实体")):
            return True
        return False

    @classmethod
    def _expected_corrected_type_from_reason(cls, reason: str) -> str:
        reason_text = str(reason or "").upper()
        if "应为" not in reason_text and "TYPE" not in reason_text and "类型错误" not in reason:
            return ""
        candidates = sorted({*cls.ALLOWED_TYPES, *cls.TYPE_ALIASES.keys()}, key=len, reverse=True)
        for candidate in candidates:
            if candidate in reason_text:
                normalized = cls.TYPE_ALIASES.get(candidate, candidate)
                if normalized in cls.ALLOWED_TYPES:
                    return normalized
        return ""

    @classmethod
    def _payload_has_corrected_entity(cls, payload: Dict, reject_text: str, expected_type: str) -> bool:
        normalized_reject_text = normalize_entity_text(reject_text)
        if not normalized_reject_text or expected_type not in cls.ALLOWED_TYPES:
            return False
        for item in payload.get("entities", []):
            if isinstance(item, str):
                entity_text = str(item).strip()
                entity_type = cls._infer_static_string_candidate_type(entity_text)
            elif isinstance(item, dict):
                entity_text = str(item.get("text") or "").strip()
                entity_type = canonical_default_entity_type(
                    cls.TYPE_ALIASES.get(
                        str(item.get("type") or "").strip().upper(),
                        str(item.get("type") or "").strip().upper(),
                    ),
                    entity_text,
                )
            else:
                continue
            if entity_type != expected_type:
                continue
            if normalize_entity_text(entity_text) == normalized_reject_text:
                return True
        return False

    @staticmethod
    def _infer_static_string_candidate_type(text: str) -> str:
        normalized = re.sub(r"[\s，,。；;：:、]+", "", text or "")
        if not normalized:
            return ""
        if is_official_institution_text(normalized):
            return "GOVERNMENT"
        if ORG_PATTERN.search(normalized) or any(token in normalized for token in ("有限公司", "集团", "公司", "事务所", "委员会", "中心")):
            return canonical_default_entity_type("ORGANIZATION", normalized)
        if any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室", "自治区", "自治州", "自治县")) and len(normalized) >= 6:
            return "LOCATION"
        if is_probable_person(normalized):
            return "PERSON"
        return canonical_default_entity_type(infer_semantic_type(normalized, ""), normalized)

    def _infer_string_candidate_type(self, text: str) -> str:
        return self._infer_static_string_candidate_type(text)

    def _normalize_entity_type(
        self,
        entity_type: str,
        text: str,
        role: object = None,
        snippet: Optional[RiskSnippet] = None,
    ) -> str:
        normalized_type = canonical_default_entity_type(self.TYPE_ALIASES.get(entity_type, entity_type), text)
        inferred_type = self._infer_string_candidate_type(text)
        role_text = str(role or "").strip()
        normalized_text = normalize_entity_text(text)
        if normalized_text in self.NON_ENTITY_TERMS:
            return ""
        if normalized_type in {"PERSON", "ORGANIZATION"} and is_non_subject_generic_org_reference(text):
            return ""
        if normalized_type in {"PERSON", "ORGANIZATION"} and is_non_subject_action_or_function_term(text):
            if normalized_type == "ORGANIZATION":
                function_stripped, consumed = strip_leading_subject_function_words(text)
                function_stripped_normalized = normalize_entity_text(function_stripped)
                if (
                    consumed > 0
                    and function_stripped_normalized
                    and not is_non_subject_generic_org_reference(function_stripped_normalized)
                    and not is_weak_function_stripped_org(function_stripped)
                    and subject_noun_gate("ORGANIZATION", function_stripped, allow_short_org=False)[0]
                ):
                    return "ORGANIZATION"
            return ""
        if is_identity_reference_term(text) or is_position_title(text):
            return ""
        prefixed_remainder = strip_identity_reference_prefix(text)
        if prefixed_remainder and is_generic_organization_term(prefixed_remainder):
            return ""
        if any(token in role_text for token in ("住址", "住所", "地址", "开户地址", "通讯地址", "送达地址")):
            return "LOCATION"
        if inferred_type in {"LOCATION", "GOVERNMENT"} and normalized_type == "ORGANIZATION":
            return inferred_type
        if normalized_type == "PERSON" and is_org_like_text(text):
            return canonical_default_entity_type("ORGANIZATION", text)
        if any(token in normalized_text for token in ("法院", "检察院", "公安局", "派出所", "仲裁委员会", "政府", "委员会", "银行", "支行", "分行")):
            return "GOVERNMENT" if is_official_institution_text(normalized_text) else ""
        if normalized_type in {"PERSON", "ORGANIZATION"} and self._snippet_supports_short_org_resolution(snippet, text):
            return "ORGANIZATION"
        if any(
            token in text
            for token in (
                "有限公司",
                "集团",
                "公司",
                "银行",
                "委员会",
                "事务所",
                "商行",
                "合作社",
                "工作室",
                "经营部",
                "门市部",
                "营业部",
                "办事处",
                "基金会",
                "联合会",
                "研究院",
                "研究所",
                "服务中心",
                "技术中心",
                "协会",
                "医院",
                "学校",
            )
        ):
            return "ORGANIZATION" if normalized_type == "PERSON" else normalized_type
        if normalized_type == "ORGANIZATION" and looks_like_organization_short_name(text):
            return "ORGANIZATION"
        if normalized_type in self.ALLOWED_TYPES:
            return normalized_type if subject_noun_gate(normalized_type, text, allow_short_org=True)[0] else ""
        return ""

    @classmethod
    def _snippet_supports_short_org_resolution(
        cls,
        snippet: Optional[RiskSnippet],
        text: str,
    ) -> bool:
        if snippet is None:
            return False
        normalized = normalize_entity_text(text)
        snippet_text = str(snippet.text or "")
        allowed_snippet_type = snippet.snippet_type in {
            "narrative_hotspot",
            "header_party_block",
            "legal_party_block",
        }
        allowed_risk_reason = snippet.risk_reason in {
            "organization_action_cue",
            "document_header",
            "keyword:付款",
            "keyword:收款",
            "keyword:结算",
            "keyword:履约",
            "keyword:交付",
            "keyword:施工",
        }
        contextual_candidates = cls._collect_contextual_short_org_candidates(snippet_text)
        action_cue_present = any(token in normalize_entity_text(snippet_text) for token in cls.NARRATIVE_ACTION_TOKENS)
        company_context_present = any(
            token in normalize_entity_text(snippet_text)
            for token in ("公司", "企业", "集团", "机构", "单位", "主体")
        )
        if not allowed_snippet_type and not allowed_risk_reason and not action_cue_present and not company_context_present:
            return False
        for candidate in contextual_candidates:
            candidate_normalized = normalize_entity_text(candidate)
            if not candidate_normalized:
                continue
            if (
                (cls._short_org_candidate_is_usable(normalized) and (
                    normalized == candidate_normalized
                    or normalized in candidate_normalized
                    or candidate_normalized in normalized
                ))
                or candidate_normalized in normalized
            ):
                return True
        return False

    @staticmethod
    def _looks_like_review_candidate(entity_type: str, text: str) -> bool:
        normalized = re.sub(r"[\s，,。；;：:、]+", "", text or "")
        if len(normalized) < 2:
            return False
        if normalized in QwenFragmentReviewService.NON_ENTITY_TERMS:
            return False
        if entity_type in {"PERSON", "ORGANIZATION"} and is_non_subject_action_or_function_term(normalized):
            return False
        if entity_type == "LOCATION":
            return subject_noun_gate("LOCATION", normalized)[0]
        if entity_type == "GOVERNMENT":
            if normalized in {"一审法院", "二审法院", "原审法院", "本院", "法院"}:
                return False
            return subject_noun_gate("GOVERNMENT", normalized)[0]
        if entity_type == "PERSON":
            return subject_noun_gate("PERSON", normalized)[0]
        if entity_type == "ORGANIZATION":
            if is_non_subject_generic_org_reference(normalized):
                return False
            _, consumed = strip_leading_subject_function_words(normalized)
            if consumed > 0:
                return False
            prefixed_remainder = strip_identity_reference_prefix(normalized)
            if prefixed_remainder and is_generic_organization_term(prefixed_remainder):
                return False
            if prefixed_remainder and any(
                prefixed_remainder.endswith(token)
                for token in QwenFragmentReviewService.SHORT_ORG_FALSE_TAIL_TOKENS
            ):
                return False
            if any(normalized.endswith(token) for token in QwenFragmentReviewService.SHORT_ORG_FALSE_TAIL_TOKENS):
                return False
            if any(token in normalized for token in ("甲方", "乙方", "丙方", "一方", "双方", "三方", "各方")):
                return False
            if any(token in normalized for token in ("随后", "另行", "继续", "负责", "配合", "协助")) and not is_org_like_text(normalized):
                return False
            passed, _ = subject_noun_gate("ORGANIZATION", normalized, allow_short_org=True)
            return passed or looks_like_organization_short_name(normalized)
        return True

    def unload(self) -> None:
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_model_path = None
        self.loaded = False
        release_runtime_memory()
