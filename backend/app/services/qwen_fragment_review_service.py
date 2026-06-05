"""Fragment-level Qwen review service for high-quality low-memory mode."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import site
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import httpx

from app.core.config import settings
from app.core.recognizer_base import RecognizerResult
from app.services.lowmem_entity_utils import (
    NON_ENTITY_ROLE_TERMS,
    ORG_PATTERN,
    ORG_SUFFIX_TERMS,
    clean_candidate_text,
    find_value_span,
    infer_semantic_type,
    is_generic_organization_term,
    is_identity_reference_term,
    is_org_like_text,
    is_position_title,
    is_probable_person,
    looks_like_organization_short_name,
    normalize_entity_text,
    make_entity,
    strip_identity_reference_prefix,
)
from app.services.lowmem_memory import release_runtime_memory
from app.services.lowmem_model_assets import build_model_asset
from app.services.risk_snippet_scheduler import RiskSnippet

logger = logging.getLogger(__name__)


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
    arbitration_model: Optional[str] = None
    arbitration_used: bool = False
    arbitration_snippet_count: int = 0
    arbitration_error: Optional[str] = None


class QwenFragmentReviewService:
    """Run small Qwen only on risky snippets.

    Backends are optional. Missing runtime packages never interrupt the main
    desensitization flow; the recognizer records manual-review metadata instead.
    """

    ALLOWED_TYPES = {
        "PERSON",
        "ORGANIZATION",
        "ADDRESS",
        "BANK_NAME",
        "ACCOUNT_NAME",
        "PROJECT",
        "CONTRACT_NO",
        "CASE_NO",
        "COURT",
        "ALIAS",
        "CN_ID_CARD",
        "CN_CREDIT_CODE",
        "CN_BANK_CARD",
        "CN_PHONE",
    }
    TYPE_ALIASES = {
        "ID_CARD": "CN_ID_CARD",
        "IDCARD": "CN_ID_CARD",
        "CREDIT_CODE": "CN_CREDIT_CODE",
        "BANK_CARD": "CN_BANK_CARD",
        "PHONE": "CN_PHONE",
        "COMPANY": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "LOCATION": "ADDRESS",
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
允许类型：PERSON, ORGANIZATION, ADDRESS, BANK_NAME, ACCOUNT_NAME, PROJECT, CONTRACT_NO, CASE_NO, COURT, ALIAS, CN_ID_CARD, CN_CREDIT_CODE, CN_BANK_CARD, CN_PHONE。
重点补漏：复杂中文人名、少数民族/带中点姓名、上诉人/被上诉人/原审原告/原审被告/申请人/被申请人对应的真实主体、委托诉讼代理人、股东/实际控制人、住址/现住址/户籍地/身份证住址/经常居住地、法院/仲裁机构、账户名/开户行、公司简称关系。对于没有明示“以下简称”的公司简称、前半段简称、去掉“公司”二字后的简称，可以结合全文高概率推理，归并到同一真实主体。对以人名命名的公司简称、门店、商行、工作室、合作社、中心、研究院、事务所、协会、医院、学校、经营部、门市部等组织主体，不要因为字面像自然人姓名就误判为 PERSON。
额外强调：即使全文里从头到尾只出现简称、没有出现完整公司名，只要该 2-6 字简称在片段里承担签约、履约、结算、付款、收款、供货、交付、施工、对账、盖章、落款、对接等组织主体动作，或与另一组织主体并列承担合同义务，也应优先识别为 ORGANIZATION，不要因为缺少“公司/集团/中心”等后缀而漏掉。
当 PERSON 与 ORGANIZATION 难以区分时，如果该词在片段里更像合同/诉讼/交易主体而不是自然人，例如出现在“由X继续履约”“X负责结算”“X向乙方付款”“X与Y签订协议”“X继续供货”“X承担责任”这类语境中，优先输出 ORGANIZATION。
不要把这些普通词或泛称当实体：国家、中华人民共和国、人民共和国、法定代表人、法定代理人、法人代表、负责人、联系人、委托代理人、委托诉讼代理人、诉讼代理人、代理人、控告人、被控告人、举报人、申诉人、起诉人、自诉人、技术人员、总经理、经理、董事长、董事、监事、岗位、职位、职务、我中心、本院、贵院、法院、人民法院、一审法院、二审法院、项目所在地人民法院、地址、合同期限、三个、五个。注意：“签订日期”这个标签不是实体，但具体日期值可以保留，不要放入 rejects。
硬性要求：
1. text 必须是片段原文里的连续字符串，一个字都不能编。
2. entities 放片段里的全部敏感实体，包括已经正确识别的实体、漏掉的实体、需要纠正类型的实体；不要因为已经识别过就输出空。
2.1. 对只有简称的组织主体，不要因为没有全称锚点就省略；只要片段语义足够支持，就直接输出该简称本身。
3. rejects 只放明显错误的已识别实体，例如：普通词“国家/法定代表人/法定代理人/控告人/地址/法院”、普通岗位词、章节标题、泛称“一审法院/二审法院/本院/法院/地址”、吞掉整句的案号/合同编号。
4. 地址要尽量输出完整地址；人名不要拆字；公司/法院要输出完整正式名称。
5. 如果片段包含合同编号、工程名称、工程地址、甲乙方、开户行、户名、账号、落款主体，entities 通常不应为空。
6. 最多输出 24 个 entities、16 个 rejects；确实没有敏感实体或错误时才输出空数组。
片段类型：{snippet_type}
已识别实体（可能有错，不要盲信）：
{existing_entities}
片段开始：
{snippet_text}
片段结束。
JSON 格式：{{"entities":[{{"type":"ORGANIZATION","text":"某某公司","role":"甲方"}}],"rejects":[{{"type":"ORGANIZATION","text":"国家","reason":"普通词"}}]}}
"""

    QUALITY_GATE_PROMPT_TEMPLATE = """你是中文合同/法律文书脱敏最终高精度审查模型。只输出 JSON，不要解释。
任务：对当前片段做最终纠错审查，尽可能发现前序识别的错误与遗漏。
你必须同时完成两件事：
A. entities：输出片段中应脱敏的真实敏感实体。对“已识别实体”里本来就正确的项，也要再次输出，表示确认。
B. rejects：输出“已识别实体”里明显错误、明显错型、明显吞句、明显普通词/身份词/职务词/标签词的项。
允许类型：PERSON, ORGANIZATION, ADDRESS, BANK_NAME, ACCOUNT_NAME, PROJECT, CONTRACT_NO, CASE_NO, COURT, ALIAS, CN_ID_CARD, CN_CREDIT_CODE, CN_BANK_CARD, CN_PHONE。
最终审查硬规则：
1. 必须逐一复核“已识别实体”中的每一项。对明显错误项，不能沉默，必须放入 rejects。
2. 如果某个已识别 text 本身是真实敏感实体，但前序 type 明显错了，要在 entities 里用正确 type 重新输出，并在 rejects 里打掉旧错项。
3. 重点找前序常见错误：把控告人、法定代理人、联系人、负责人、委托代理人、诉讼代理人、职务、岗位等身份或职务词误当实体；把公司错当人名；把人名错当机构；把整句吞成案号或合同号；漏掉只以简称出现的公司或机构主体。
4. 对没有全称锚点、只单独出现的 2-6 字公司或机构简称，只要它在片段里承担签约、履约、结算、付款、收款、供货、交付、施工、对账、盖章、落款、对接、承担责任等组织主体动作，或与另一主体并列承担合同、诉讼、交易义务，就应优先识别为 ORGANIZATION，不得因缺少“公司”等后缀而省略。
5. 对以人名命名的公司、工作室、商行、合作社、经营部、门市部、营业部、办事处、中心、研究院、事务所、学校、医院、协会、银行等组织主体，不要因为字面像姓名就误判为 PERSON。
6. 不要把这些普通词或泛称当实体：国家、中华人民共和国、人民共和国、法定代表人、法定代理人、法人代表、负责人、联系人、委托代理人、委托诉讼代理人、诉讼代理人、代理人、控告人、被控告人、举报人、申诉人、起诉人、自诉人、技术人员、总经理、经理、董事长、董事、监事、岗位、职位、职务、我中心、本院、贵院、法院、人民法院、一审法院、二审法院、项目所在地人民法院、地址、合同期限、三个、五个。
7. text 必须是片段原文里的连续字符串，一个字都不能编。
8. 金额和日期默认不脱敏；具体日期值也不要误放进 rejects。
9. 输出时优先纠错和补漏，不要保守放过明显问题。
片段类型：{snippet_type}
已识别实体（可能有错，必须逐一复核）：
{existing_entities}
片段开始：
{snippet_text}
片段结束。
JSON 格式：{{"entities":[{{"type":"ORGANIZATION","text":"某某公司","role":"甲方"}}],"rejects":[{{"type":"ORGANIZATION","text":"控告人","reason":"身份指代词"}},{{"type":"PERSON","text":"广州某新能源有限公司","reason":"类型错误，应为ORGANIZATION"}}]}}
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
        self._ollama_models_touched: set[str] = set()

    @property
    def installed(self) -> bool:
        if self.review_asset.installed or self.fallback_asset.installed:
            return True
        return self._select_ollama_review_model(self._installed_ollama_models()) is not None

    async def review(
        self,
        full_text: str,
        snippets: Iterable[RiskSnippet],
        *,
        existing_entities: Optional[Iterable[RecognizerResult]] = None,
        max_snippets: Optional[int] = None,
    ) -> ReviewResult:
        if not settings.ENABLE_QWEN_REVIEW:
            return ReviewResult(requires_manual_review=False, error="review_disabled")
        runtime = self._select_review_runtime()
        if runtime is None:
            return ReviewResult(requires_manual_review=True, error="review_model_not_installed")

        fallback_used = False
        all_results: list[RecognizerResult] = []
        all_rejections: list[Dict] = []
        any_model_used = False
        last_error: Optional[str] = None
        raw_candidate_count = 0
        parsed_snippet_count = 0
        existing_list = list(existing_entities or [])
        review_model_name = runtime.model_id
        review_backend = runtime.backend
        review_limit = max_snippets or max(1, int(settings.MID_REVIEW_MAX_SNIPPETS or settings.REVIEW_MAX_SNIPPETS))
        scheduled_snippets = list(snippets)[:review_limit]

        for snippet in scheduled_snippets:
            snippet_mode = "quality_gate" if snippet.snippet_type == "quality_gate_block" else "standard"
            snippet_existing = self._entities_for_snippet(existing_list, snippet, mode=snippet_mode)
            raw_response: Optional[str] = None
            active_runtime = self._select_runtime_for_snippet(snippet) or runtime
            try:
                allow_thinking = self._review_thinking_enabled(active_runtime)
                prompt = self._build_prompt(
                    snippet,
                    existing_entities=snippet_existing,
                    allow_thinking=allow_thinking,
                )
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
                        prompt = self._build_prompt(
                            snippet,
                            existing_entities=snippet_existing,
                            allow_thinking=allow_thinking,
                        )
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
                    if "not_installed" in str(last_error):
                        break
                    continue

            parsed = self._parse_json_response(raw_response or "")
            if parsed is None:
                allow_thinking = self._review_thinking_enabled(active_runtime)
                retry_prompt = self._build_prompt(
                    snippet,
                    existing_entities=self._entities_for_snippet(
                        existing_list,
                        snippet,
                        limit=24 if snippet.snippet_type == "quality_gate_block" else 12,
                        mode=snippet_mode,
                    ),
                    short=True,
                    allow_thinking=allow_thinking,
                )
                try:
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
                continue
            raw_candidate_count += len(parsed.get("entities") or [])
            parsed_snippet_count += 1
            all_results.extend(self._materialize_candidates(full_text, parsed, snippet))
            all_rejections.extend(self._materialize_rejections(parsed, snippet_existing))

        arbitration_model: Optional[str] = None
        arbitration_used = False
        arbitration_snippet_count = 0
        arbitration_error: Optional[str] = None
        arbitration_requires_manual = False
        arbitration_snippets = self._select_arbitration_snippets(
            scheduled_snippets,
            parsed_snippet_count=parsed_snippet_count,
            review_error=last_error,
        )
        heavy_arbitration_enabled = settings.ENABLE_HEAVY_ARBITRATION and (
            not settings.is_high_quality_lowmem_mode() or settings.LOWMEM_ENABLE_HEAVY_ARBITRATION
        )
        if heavy_arbitration_enabled and arbitration_snippets:
            arbitration_model = self._select_ollama_arbitration_model()
            if arbitration_model:
                arbitration_runtime = ReviewRuntime(
                    backend="ollama",
                    model_id=arbitration_model,
                    tier="heavy_arbitration",
                )
                for snippet in arbitration_snippets[: max(1, settings.HEAVY_ARBITRATION_MAX_SNIPPETS)]:
                    snippet_existing = self._entities_for_snippet(existing_list, snippet, limit=18)
                    prompt = self._build_arbitration_prompt(snippet, existing_entities=snippet_existing)
                    try:
                        raw_response = await self._review_with_runtime(
                            prompt,
                            arbitration_runtime,
                            allow_thinking=self._arbitration_thinking_enabled(),
                        )
                        arbitration_used = True
                        any_model_used = True
                        parsed = self._parse_json_response(raw_response or "")
                        if parsed is None:
                            arbitration_error = "arbitration_json_parse_failed"
                            continue
                        arbitration_snippet_count += 1
                        arbitration_requires_manual = arbitration_requires_manual or bool(
                            parsed.get("requires_manual_review")
                        )
                        raw_candidate_count += len(parsed.get("entities") or [])
                        all_results.extend(
                            self._materialize_candidates(
                                full_text,
                                parsed,
                                snippet,
                                source_name="qwen_heavy_arbitration",
                            )
                        )
                        all_rejections.extend(self._materialize_rejections(parsed, snippet_existing))
                    except Exception as exc:
                        arbitration_error = f"{type(exc).__name__}: {exc}"
                        logger.warning("Heavy arbitration failed: %s", exc)
                        break
            else:
                arbitration_error = "arbitration_model_not_installed"

        requires_manual_review = bool(last_error) or not any_model_used or arbitration_requires_manual
        if arbitration_error:
            requires_manual_review = True
        return ReviewResult(
            entities=all_results,
            rejected_entities=all_rejections,
            model_used=any_model_used,
            fallback_used=fallback_used,
            model_name=review_model_name,
            review_backend=review_backend,
            requires_manual_review=requires_manual_review,
            error=last_error,
            raw_candidate_count=raw_candidate_count,
            parsed_snippet_count=parsed_snippet_count,
            arbitration_model=arbitration_model,
            arbitration_used=arbitration_used,
            arbitration_snippet_count=arbitration_snippet_count,
            arbitration_error=arbitration_error,
        )

    async def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
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
        ollama_models = self._installed_ollama_models()
        if settings.is_high_quality_lowmem_mode():
            if self.review_asset.installed:
                return ReviewRuntime(
                    backend=settings.REVIEW_BACKEND.lower().strip(),
                    model_id=self.review_asset.model_id,
                    asset=self.review_asset,
                    tier="local_review",
                )
            if settings.LOWMEM_ENABLE_LOCAL_REVIEW_FALLBACK and self.fallback_asset.installed:
                return ReviewRuntime(
                    backend=settings.REVIEW_MODEL_FALLBACK_BACKEND.lower().strip(),
                    model_id=self.fallback_asset.model_id,
                    asset=self.fallback_asset,
                    fallback=True,
                    tier="local_review_fallback",
                )
            ollama_model = self._select_ollama_review_model(ollama_models)
            if ollama_model:
                return ReviewRuntime(backend="ollama", model_id=ollama_model, tier="fast_primary")
            return None

        ollama_model = self._select_ollama_review_model(ollama_models)
        if ollama_model:
            tier = "mid_review" if settings.is_mid_review_ollama_model(ollama_model) else "fast_primary"
            return ReviewRuntime(backend="ollama", model_id=ollama_model, tier=tier)
        if self.review_asset.installed:
            return ReviewRuntime(
                backend=settings.REVIEW_BACKEND.lower().strip(),
                model_id=self.review_asset.model_id,
                asset=self.review_asset,
                tier="local_review",
            )
        if self.fallback_asset.installed:
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
        if snippet.snippet_type != "quality_gate_block":
            return self._select_review_runtime()
        return self._select_quality_gate_runtime()

    def _select_quality_gate_runtime(self) -> Optional[ReviewRuntime]:
        base_runtime = self._select_review_runtime()
        if settings.is_high_quality_lowmem_mode():
            if base_runtime is None:
                return None
            return ReviewRuntime(
                backend=base_runtime.backend,
                model_id=base_runtime.model_id,
                asset=base_runtime.asset,
                fallback=base_runtime.fallback,
                tier="quality_gate_high_precision",
            )
        installed_models = self._installed_ollama_models()

        for candidate in (
            settings.HEAVY_ARBITRATION_MODEL,
            settings.MID_REVIEW_MODEL,
            settings.MID_REVIEW_FALLBACK_MODEL,
        ):
            normalized = str(candidate or "").strip()
            if normalized and normalized in installed_models:
                return ReviewRuntime(
                    backend="ollama",
                    model_id=normalized,
                    tier="quality_gate_high_precision",
                )

        for model_name in settings.get_ollama_model_options(available_models=list(installed_models)):
            if model_name in installed_models and settings.is_review_capable_ollama_model(model_name):
                return ReviewRuntime(
                    backend="ollama",
                    model_id=model_name,
                    tier="quality_gate_high_precision",
                )
        return base_runtime

    def _installed_ollama_models(self) -> set[str]:
        try:
            with httpx.Client(timeout=2.5) as client:
                response = client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
                response.raise_for_status()
                data = response.json()
        except Exception:
            return set()
        models = data.get("models") if isinstance(data, dict) else []
        installed: set[str] = set()
        for item in models or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                installed.add(name)
        return installed

    def _select_ollama_review_model(self, installed_models: set[str]) -> Optional[str]:
        if not installed_models:
            return None
        if settings.is_high_quality_lowmem_mode() and not settings.LOWMEM_ALLOW_MID_REVIEW_MODEL:
            for candidate in settings.get_lowmem_ollama_review_candidates(
                available_models=list(installed_models)
            ):
                if candidate in installed_models:
                    return candidate
            return None
        exact_priority = [
            settings.MID_REVIEW_MODEL,
            settings.MID_REVIEW_FALLBACK_MODEL,
            settings.FAST_REVIEW_MODEL,
            settings.get_effective_ollama_model(),
        ]
        for candidate in exact_priority:
            if candidate in installed_models:
                return candidate
        for model_name in settings.get_ollama_model_options(available_models=list(installed_models)):
            if model_name not in installed_models:
                continue
            if settings.is_mid_review_ollama_model(model_name) or settings.is_fast_primary_ollama_model(model_name):
                return model_name
        return None

    def _select_ollama_arbitration_model(self) -> Optional[str]:
        if settings.is_high_quality_lowmem_mode() and not settings.LOWMEM_ENABLE_HEAVY_ARBITRATION:
            return None
        installed = self._installed_ollama_models()
        if settings.HEAVY_ARBITRATION_MODEL in installed:
            return settings.HEAVY_ARBITRATION_MODEL
        for model_name in settings.get_ollama_model_options(available_models=list(installed)):
            if model_name in installed and settings.is_heavy_arbitration_ollama_model(model_name):
                return model_name
        return None

    def _build_prompt(
        self,
        snippet: RiskSnippet,
        *,
        existing_entities: Optional[list[dict]] = None,
        short: bool = False,
        allow_thinking: bool = False,
    ) -> str:
        snippet_text = snippet.text[: min(settings.REVIEW_MAX_CHARS_PER_SNIPPET, 900)]
        existing_json = json.dumps(existing_entities or [], ensure_ascii=False)
        short_org_hints = self._format_short_org_action_hints(snippet_text)
        thinking_prefix = "" if allow_thinking else "/no_think\n"
        if snippet.snippet_type == "quality_gate_block":
            if short:
                return (
                    f"{thinking_prefix}只输出 JSON。你在做最终高精度脱敏审查，必须逐一复核已识别实体，"
                    "对明显错误项必须输出到 rejects，不能沉默；对真实敏感实体必须输出到 entities。"
                    "重点发现：身份指代词/职务词误识别、公司与人名错分、吞句案号、只出现简称的公司漏识别。"
                    "对只出现简称但承担履约、结算、付款、收款、供货、交付、承担责任等组织动作的 2-6 字主体，"
                    "优先输出 ORGANIZATION。金额和日期默认不脱敏。\n"
                    f"{short_org_hints}"
                    f"已识别（必须逐一复核）：{existing_json}\n"
                    f"片段：\n{snippet_text}\n"
                    "格式：{\"entities\":[{\"type\":\"ORGANIZATION\",\"text\":\"某某公司\"}],"
                    "\"rejects\":[{\"type\":\"ORGANIZATION\",\"text\":\"控告人\",\"reason\":\"身份指代词\"}]}\n"
                )
            return (
                thinking_prefix
                + self.QUALITY_GATE_PROMPT_TEMPLATE.format(
                    snippet_type=snippet.snippet_type,
                    snippet_text=snippet_text,
                    existing_entities=existing_json,
                )
                + short_org_hints
            )
        if short:
            return (
                f"{thinking_prefix}只输出 JSON。重新检查片段：entities 输出真实敏感实体，rejects 输出已识别里的明显错误。"
                "即使已识别实体正确，也要在 entities 再输出确认。text 必须来自原文。"
                "不要抽取国家/中华人民共和国/人民共和国/法定代表人/委托代理人/技术人员/我中心/"
                "一审法院/二审法院/本院/法院/地址/合同期限/三个/五个；具体日期值不要放入 rejects。\n"
                f"{short_org_hints}"
                f"已识别（可能有错）：{existing_json}\n"
                f"片段：\n{snippet_text}\n"
                "格式：{\"entities\":[{\"type\":\"ORGANIZATION\",\"text\":\"某某公司\"}],"
                "\"rejects\":[{\"type\":\"ORGANIZATION\",\"text\":\"国家\",\"reason\":\"普通词\"}]}\n"
            )
        return (
            thinking_prefix
            + self.PROMPT_TEMPLATE.format(
                snippet_type=snippet.snippet_type,
                snippet_text=snippet_text,
                existing_entities=existing_json,
            )
            + short_org_hints
        )

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
                    or cls._is_noisy_short_org_candidate(normalized)
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
        candidate = cls._trim_short_org_candidate_core(value)
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

    def _build_arbitration_prompt(
        self,
        snippet: RiskSnippet,
        *,
        existing_entities: Optional[list[dict]] = None,
    ) -> str:
        thinking = "/think" if self._arbitration_thinking_enabled() else "/no_think"
        snippet_text = snippet.text[: min(settings.REVIEW_MAX_CHARS_PER_SNIPPET, 900)]
        existing_json = json.dumps(existing_entities or [], ensure_ascii=False)
        return (
            f"{thinking}\n"
            "你是中文合同/法律文书脱敏疑难仲裁模型。只输出 JSON，不要解释。\n"
            "任务：仅根据片段原文和已识别实体，处理质量闸失败证据。"
            "只允许三类结论：补充应脱敏实体到 entities，确认误识别到 rejects，"
            "或在无法判断时设置 requires_manual_review=true。\n"
            "不得全文扩写，不得编造原文不存在的 text。金额和日期默认不脱敏。\n"
            "片段类型："
            f"{snippet.snippet_type}\n"
            "风险原因："
            f"{snippet.risk_reason}\n"
            f"已识别实体：{existing_json}\n"
            f"片段：\n{snippet_text}\n"
            "JSON 格式：{\"entities\":[{\"type\":\"ORGANIZATION\",\"text\":\"某某公司\",\"role\":\"甲方\"}],"
            "\"rejects\":[{\"type\":\"ORGANIZATION\",\"text\":\"地址\",\"reason\":\"普通词\"}],"
            "\"requires_manual_review\":false}\n"
        )

    @staticmethod
    def _review_thinking_enabled(runtime: ReviewRuntime | None) -> bool:
        if runtime is None or runtime.backend.lower().strip() != "ollama":
            return False
        if runtime.tier == "quality_gate_high_precision":
            return True
        mode = str(settings.REVIEW_THINKING_MODE or "").strip().lower()
        if mode in {"", "off", "false", "0", "none", "disabled"}:
            return False
        if mode in {"on", "true", "1", "all", "review", "mid_review", "middle"}:
            return runtime.tier == "mid_review"
        return False

    @staticmethod
    def _arbitration_thinking_enabled() -> bool:
        return str(settings.REVIEW_THINKING_MODE or "").strip().lower() in {
            "on",
            "true",
            "1",
            "all",
            "arbitration",
            "heavy",
        }

    @staticmethod
    def _select_arbitration_snippets(
        snippets: Iterable[RiskSnippet],
        *,
        parsed_snippet_count: int,
        review_error: Optional[str],
    ) -> list[RiskSnippet]:
        snippet_list = list(snippets)
        if not snippet_list:
            return []
        trigger_types = {"conflict_block", "ocr_anomaly_block"}
        trigger_reasons = {"uie_ner_overlap_conflict", "long_document_low_entity_density"}
        selected = [
            snippet
            for snippet in snippet_list
            if snippet.snippet_type in trigger_types
            or snippet.risk_reason in trigger_reasons
            or "conflict" in snippet.risk_reason
            or "low_entity_density" in snippet.risk_reason
        ]
        if review_error or parsed_snippet_count == 0:
            selected = selected or snippet_list[: max(1, settings.HEAVY_ARBITRATION_MAX_SNIPPETS)]
        return selected[: max(1, settings.HEAVY_ARBITRATION_MAX_SNIPPETS)]

    @staticmethod
    def _entities_for_snippet(
        entities: Iterable[RecognizerResult],
        snippet: RiskSnippet,
        *,
        limit: int = 24,
        mode: str = "standard",
    ) -> list[dict]:
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
        if entity_type in {"CASE_NO", "CONTRACT_NO"} and (
            len(normalized) > (48 if entity_type == "CASE_NO" else 80)
            or any(token in normalized for token in ("上诉人", "被上诉人", "不服", "提起上诉", "事实与理由", "以下简称"))
        ):
            score += 90
        if entity_type in {"PERSON", "PERSON_NAME"} and is_org_like_text(normalized):
            score += 85
        if entity_type in {"ORGANIZATION", "COMPANY_NAME"} and is_probable_person(normalized):
            score += 80
        if entity_type == "COURT" and normalized in {"法院", "人民法院", "一审法院", "二审法院", "本院", "贵院"}:
            score += 90
        if entity_type in {"ORGANIZATION", "COMPANY_NAME"} and len(normalized) > 18 and any(
            token in normalized for token in ("不服", "请求", "认为", "提交", "证明", "承担责任")
        ):
            score += 70
        if entity_type in {"ORGANIZATION", "COMPANY_NAME"} and looks_like_organization_short_name(normalized):
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
        if runtime.backend.lower().strip() == "ollama":
            return await self._invoke_with_optional_max_tokens(
                self._review_with_ollama,
                prompt,
                model_id=runtime.model_id,
                allow_thinking=allow_thinking,
                max_tokens=max_tokens,
                tier=runtime.tier,
            )
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

    async def _review_with_ollama(
        self,
        prompt: str,
        *,
        model_id: str,
        allow_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tier: str = "local",
    ) -> str:
        token_limit = int(max_tokens or (settings.REVIEW_THINKING_MAX_TOKENS if allow_thinking else settings.REVIEW_MAX_TOKENS))
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "1m",
            "think": bool(allow_thinking),
            "options": {
                "num_ctx": settings.REVIEW_NUM_CTX,
                "num_predict": token_limit,
                "temperature": settings.REVIEW_TEMPERATURE,
            },
        }
        if tier == "quality_gate_high_precision":
            timeout_seconds = max(90, int(settings.REVIEW_OLLAMA_TIMEOUT))
            payload["options"]["num_predict"] = max(token_limit, 640)
        elif settings.is_high_quality_lowmem_mode():
            timeout_seconds = max(10, int(settings.LOWMEM_REVIEW_OLLAMA_TIMEOUT))
        else:
            timeout_seconds = max(30, int(settings.REVIEW_OLLAMA_TIMEOUT))
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate", json=payload)
            if response.status_code >= 400 and "think" in response.text.lower():
                # Older Ollama builds may not understand the top-level `think`
                # switch; non-thinking calls still keep prompt-level `/no_think`.
                payload.pop("think", None)
                response = await client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        self._ollama_models_touched.add(model_id)
        return str(data.get("response") or "")

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
                and (isinstance(payload.get("entities"), list) or isinstance(payload.get("rejects"), list))
            ):
                payload.setdefault("entities", [])
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
                and (isinstance(payload.get("entities"), list) or isinstance(payload.get("rejects"), list))
            ):
                payload.setdefault("entities", [])
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
        for item in payload.get("entities", []):
            if isinstance(item, str):
                raw_text = item.strip()
                entity_type = self._infer_string_candidate_type(raw_text)
                role = None
                canonical = None
            elif isinstance(item, dict):
                entity_type = str(item.get("type") or "").strip().upper()
                raw_text = str(item.get("text") or "").strip()
                role = item.get("role")
                canonical = item.get("canonical")
            else:
                continue
            if not entity_type or not raw_text:
                continue
            entity_type = self._normalize_entity_type(entity_type, raw_text, role=role, snippet=snippet)
            if not entity_type:
                continue
            materialized_text, span = self._materialize_candidate_span(
                full_text,
                raw_text,
                snippet,
                entity_type,
            )
            if span is None or not materialized_text:
                continue
            if not self._looks_like_review_candidate(entity_type, materialized_text):
                continue
            metadata = {
                "source": source_name,
                "snippet_type": snippet.snippet_type,
                "risk_reason": snippet.risk_reason,
                "role": role,
                "canonical": canonical,
                "review": source_name in {"qwen_fragment_review", "qwen_heavy_arbitration"},
                "source_layer": "llm_review" if source_name in {"qwen_fragment_review", "qwen_heavy_arbitration"} else "",
            }
            if materialized_text != raw_text:
                metadata["materialized_from"] = raw_text
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
            if item.entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}
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
                    "review": source_name in {"qwen_fragment_review", "qwen_heavy_arbitration"},
                    "source_layer": "llm_review" if source_name in {"qwen_fragment_review", "qwen_heavy_arbitration"} else "",
                    "materialized_from": item,
                    "list_completion": True,
                    "list_label": label,
                    "short_org_candidate": True,
                    "identity_surface": normalize_entity_text(materialized_text),
                }
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
            return full_text[span[0] : span[1]], span
        return "", None

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
            if entity_type == "ORGANIZATION":
                trimmed_candidate = self._trim_short_org_candidate_core(candidate)
                if trimmed_candidate:
                    candidate = trimmed_candidate
            normalized = normalize_entity_text(candidate)
            if not candidate or not normalized or normalized in seen:
                return
            if entity_type == "ORGANIZATION" and self._is_noisy_short_org_candidate(normalized):
                return
            seen.add(normalized)
            candidates.append(candidate)

        if entity_type == "ORGANIZATION":
            prefixed_remainder = strip_identity_reference_prefix(cleaned)
            if prefixed_remainder:
                if is_generic_organization_term(prefixed_remainder):
                    return []
                _append(prefixed_remainder)

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
                entity_type = cls.TYPE_ALIASES.get(str(item.get("type") or "").strip().upper(), str(item.get("type") or "").strip().upper())
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
        if "法院" in normalized or "仲裁委员会" in normalized or "检察院" in normalized:
            return "COURT"
        if "银行" in normalized and any(token in normalized for token in ("支行", "分行", "开户行")):
            return "BANK_NAME"
        if ORG_PATTERN.search(normalized) or any(token in normalized for token in ("有限公司", "集团", "公司", "事务所", "委员会", "中心")):
            return "ORGANIZATION"
        if any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室", "自治区", "自治州", "自治县")) and len(normalized) >= 6:
            return "ADDRESS"
        if is_probable_person(normalized):
            return "PERSON"
        return infer_semantic_type(normalized, "")

    def _infer_string_candidate_type(self, text: str) -> str:
        return self._infer_static_string_candidate_type(text)

    def _normalize_entity_type(
        self,
        entity_type: str,
        text: str,
        role: object = None,
        snippet: Optional[RiskSnippet] = None,
    ) -> str:
        normalized_type = self.TYPE_ALIASES.get(entity_type, entity_type)
        inferred_type = self._infer_string_candidate_type(text)
        role_text = str(role or "").strip()
        if is_identity_reference_term(text) or is_position_title(text):
            return ""
        prefixed_remainder = strip_identity_reference_prefix(text)
        if prefixed_remainder and is_generic_organization_term(prefixed_remainder):
            return ""
        if any(token in role_text for token in ("住址", "住所", "地址", "开户地址", "通讯地址", "送达地址")):
            return "ADDRESS"
        if inferred_type in {"ADDRESS", "COURT", "BANK_NAME"} and normalized_type == "ORGANIZATION":
            return inferred_type
        if normalized_type == "PERSON" and is_org_like_text(text):
            return "ORGANIZATION"
        if "法院" in text:
            return "COURT"
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
        return normalized_type if normalized_type in self.ALLOWED_TYPES else ""

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
            "quality_gate_block",
            "narrative_hotspot",
            "header_party_block",
            "legal_party_block",
        }
        allowed_risk_reason = snippet.risk_reason in {
            "organization_action_cue",
            "final_entity_quality_gate",
            "final_entity_quality_gate_gap",
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
        if entity_type in {"CASE_NO", "CONTRACT_NO"}:
            return bool(re.search(r"[（(][^）)]{2,}[）)]|号|第.+号|[A-Za-z].*\d|\d.*[A-Za-z]", normalized))
        if entity_type == "ADDRESS":
            return len(normalized) >= 6 or any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室", "自治区", "自治州", "自治县", "旗", "盟"))
        if entity_type == "COURT":
            if normalized in {"一审法院", "二审法院", "原审法院", "本院", "法院"}:
                return False
            return any(token in normalized for token in ("人民法院", "中级人民法院", "高级人民法院", "仲裁委员会", "人民检察院"))
        if entity_type == "PERSON":
            if any(token in normalized for token in ("省", "市", "区", "县", "镇", "乡", "村", "路", "街", "号", "室")):
                return False
            return re.fullmatch(r"[\u4e00-\u9fa5]{2,8}(?:·[\u4e00-\u9fa5]{2,8})?", normalized) is not None
        if entity_type == "ORGANIZATION":
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
            return (
                any(
                    token in normalized
                    for token in (
                        "公司",
                        "集团",
                        "银行",
                        "委员会",
                        "事务所",
                        "中心",
                        "法院",
                        "检察院",
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
                        "协会",
                        "医院",
                        "学校",
                    )
                )
                or looks_like_organization_short_name(normalized)
                or len(normalized) >= 4
            )
        return True

    def unload(self) -> None:
        for model_id in list(self._ollama_models_touched):
            try:
                with httpx.Client(timeout=3.0) as client:
                    client.post(
                        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                        json={
                            "model": model_id,
                            "prompt": "",
                            "stream": False,
                            "keep_alive": 0,
                        },
                    )
            except Exception:
                logger.debug("Failed to unload Ollama review model %s", model_id, exc_info=True)
        self._ollama_models_touched.clear()
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_model_path = None
        self.loaded = False
        release_runtime_memory()
