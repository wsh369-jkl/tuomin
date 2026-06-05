"""Ollama-backed service for Chinese legal/business entity recognition."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx
import requests

from app.core.identifier_rules import ALL_IDENTIFIER_LABELS, compact_text, label_matches, looks_like_case_number
from app.core.llm_strategy import get_llm_strategy_profile, get_runtime_llm_strategy_profile
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaLLMService:
    """Use a local Ollama model to extract semantic entities from Chinese documents."""

    SUPPORTED_ENTITY_TYPES = {
        "PERSON": "Natural person names, signatories, contacts, and legal representatives.",
        "ORGANIZATION": "Party names, companies, institutions, and project owners.",
        "LOCATION": "Project addresses, residence addresses, registered addresses, correspondence addresses, court names, counties, towns, roads, and sites.",
        "POSITION": "Job titles such as manager, legal representative, and project lead.",
        "PROJECT": "Project names, engineering names, and procurement names.",
        "CONTRACT_NO": "Contract numbers, project numbers, and bid numbers.",
        "BANK_NAME": "Bank and branch names.",
        "ACCOUNT_NAME": "Account holder names, payee names, and collection units.",
    }

    DOCUMENT_TYPE_LABELS = {
        "contract": "合同/协议类",
        "legal_application": "申请/诉状类",
        "enforcement_paper": "执行/保全类",
        "financial_account_material": "账户/结算材料类",
        "other": "其他正式文书",
    }

    NON_ENTITY_HEADING_TERMS = {
        "民事上诉状",
        "民事起诉状",
        "民事答辩状",
        "上诉请求",
        "诉讼请求",
        "请求事项",
        "申请事项",
        "事实与理由",
        "事实和理由",
        "理由如下",
        "申请理由",
        "事实与依据",
        "判令",
        "综上",
        "此致",
        "敬礼",
        "特此申请",
        "特此上诉",
    }

    DENSE_NARRATIVE_TOKENS = [
        "股东",
        "账户",
        "居间费",
        "货款",
        "退货款",
        "退款",
        "转给",
        "支付",
        "收取",
        "证据",
        "备注",
        "混同",
        "连带",
        "承担",
    ]

    DOCUMENT_TYPE_HINTS = {
        "contract": [
            ("合同", 4),
            ("协议", 4),
            ("甲方", 3),
            ("乙方", 3),
            ("委托方", 3),
            ("受托方", 3),
            ("合同编号", 3),
            ("签订日期", 2),
            ("工程名称", 2),
            ("项目名称", 2),
        ],
        "legal_application": [
            ("申请书", 4),
            ("起诉状", 4),
            ("答辩状", 4),
            ("申请人", 3),
            ("被申请人", 3),
            ("原告", 3),
            ("被告", 3),
            ("第三人", 2),
            ("请求事项", 3),
            ("事实和理由", 3),
            ("人民法院", 2),
        ],
        "enforcement_paper": [
            ("执行", 4),
            ("续行冻结", 5),
            ("保全", 4),
            ("申请执行人", 4),
            ("被执行人", 4),
            ("冻结", 3),
            ("人民法院", 3),
            ("执行裁定", 3),
            ("民事调解书", 2),
            ("民事判决书", 2),
        ],
        "financial_account_material": [
            ("开户行", 4),
            ("户名", 4),
            ("账号", 4),
            ("账户", 3),
            ("收款", 3),
            ("付款", 3),
            ("银行", 2),
            ("支行", 2),
            ("分行", 2),
            ("结算", 2),
        ],
    }

    DOCUMENT_TYPE_PROMPT_HINTS = {
        "contract": (
            "Focus on full party names, contract numbers, project names, project addresses, "
            "legal representatives, contacts, and signature/footer blocks."
        ),
        "legal_application": (
            "Focus on applicants, respondents, plaintiffs, defendants, agents, court or institution "
            "names, residence or registered addresses, and non-standard header/footer blocks."
        ),
        "enforcement_paper": (
            "Focus on enforcement applicants/respondents, preserved or frozen account sections, full "
            "bank branch names, residence addresses, and court names."
        ),
        "financial_account_material": (
            "Focus on bank branches, account holders, settlement units, payer/payee subjects, and "
            "account-related heading or table blocks."
        ),
        "other": (
            "Focus on full official names, long addresses, institution lines, contact blocks, and "
            "footer/signature sections."
        ),
    }

    FOCUS_LIBRARY = {
        "party_header": {
            "instruction": (
                "Inspect title/header and party blocks for full subject names, role-aligned entities, "
                "legal representatives, contacts, project names, and contract/case identifiers."
            ),
            "tokens": [
                "甲方",
                "乙方",
                "丙方",
                "委托方",
                "受托方",
                "申请人",
                "被申请人",
                "申请执行人",
                "被执行人",
                "原告",
                "被告",
                "第三人",
                "法定代表人",
                "法人代表",
                "联系人",
                "合同编号",
                "案号",
            ],
            "head_lines": 14,
            "radius": 1,
            "max_matches": 10,
        },
        "project_section": {
            "instruction": (
                "Inspect project and contract metadata lines for full project names, engineering names, "
                "contract numbers, project owners, and project addresses."
            ),
            "tokens": [
                "工程名称",
                "项目名称",
                "项目编号",
                "工程地址",
                "项目地址",
                "合同编号",
                "技术服务合同",
                "采购",
                "招标",
            ],
            "head_lines": 12,
            "radius": 1,
            "max_matches": 8,
        },
        "address_section": {
            "instruction": (
                "Inspect full addresses and location-heavy lines, including residence, address, court, "
                "delivery, and registered-address sections. Extract full official names and full addresses."
            ),
            "tokens": [
                "住址",
                "地址",
                "身份证住址",
                "住所",
                "住所地",
                "通讯地址",
                "送达地址",
                "注册地址",
                "办公地址",
                "工程地址",
                "项目地址",
                "人民法院",
                "检察院",
                "公安局",
                "街道",
                "大道",
                "广场",
            ],
            "radius": 1,
            "max_matches": 12,
        },
        "account_section": {
            "instruction": (
                "Inspect bank and settlement sections. Extract BANK_NAME and ACCOUNT_NAME when they appear "
                "verbatim, but do not return account numbers or amounts."
            ),
            "tokens": [
                "开户行",
                "户名",
                "账号",
                "账户",
                "收款",
                "付款",
                "银行",
                "支行",
                "分行",
                "财付通",
                "支付宝",
            ],
            "radius": 1,
            "max_matches": 12,
        },
        "narrative_subjects": {
            "instruction": (
                "Inspect dense narrative or factual paragraphs for short repeated company mentions, "
                "relationship-triggered person names, abbreviated organizations, and account/payment subjects."
            ),
            "tokens": [
                "公司",
                "集团",
                "股东",
                "法定代表人",
                "法人代表",
                "实际控制人",
                "付款至",
                "支付给",
                "转给",
                "汇至",
                "收款至",
                "个人银行账户",
                "银行账户",
                "简称",
                "以下简称",
                "又称",
                "签署",
                "证据",
            ],
            "radius": 1,
            "max_matches": 14,
        },
        "institution_section": {
            "instruction": (
                "Inspect institution and case lines for full official organization names, court names, "
                "service centers, agencies, and other formal institutional subjects."
            ),
            "tokens": [
                "人民法院",
                "仲裁委员会",
                "检察院",
                "公安局",
                "气象服务中心",
                "服务中心",
                "研究院",
                "事务所",
                "有限公司",
                "有限责任公司",
            ],
            "radius": 1,
            "max_matches": 10,
        },
        "footer_signature": {
            "instruction": (
                "Inspect footer and signature blocks for repeated parties, institutions, contacts, positions, "
                "signatories, dates, and overlooked subject mentions."
            ),
            "tokens": [
                "此致",
                "申请人",
                "具状人",
                "法定代表人",
                "委托代理人",
                "联系人",
                "联系电话",
                "盖章",
                "签字",
                "日期",
            ],
            "tail_lines": 14,
            "radius": 2,
            "max_matches": 10,
        },
    }

    DOC_TYPE_FOCUS_PLAN = {
        "contract": ["party_header", "project_section", "narrative_subjects", "address_section", "footer_signature", "account_section"],
        "legal_application": ["party_header", "narrative_subjects", "address_section", "institution_section", "footer_signature"],
        "enforcement_paper": [
            "party_header",
            "narrative_subjects",
            "address_section",
            "account_section",
            "institution_section",
            "footer_signature",
        ],
        "financial_account_material": [
            "account_section",
            "narrative_subjects",
            "party_header",
            "address_section",
            "footer_signature",
        ],
        "other": ["party_header", "narrative_subjects", "address_section", "footer_signature", "institution_section"],
    }

    DEFINITION_TRIGGER_TOKENS = [
        "以下简称",
        "下称",
        "简称",
        "又称",
        "项目公司",
        "申请人",
        "被申请人",
        "被申请人公司",
        "甲公司",
        "乙公司",
        "丙公司",
    ]

    SPECIALIZED_PASS_LIBRARY = {
        "person": {
            "instruction": (
                "Focus only on natural persons. Recover full names, masked forms such as 某某/某, "
                "signatories, legal representatives, contacts, agents, shareholders, controllers, and "
                "cross-page repeated person mentions."
            ),
            "categories": {"person", "relation", "footer", "definition"},
        },
        "organization": {
            "instruction": (
                "Focus only on organizations. Recover full companies, institutions, project companies, "
                "party subjects, signature blocks, shortened company mentions, and later repeated short forms."
            ),
            "categories": {"organization", "relation", "footer", "definition"},
        },
        "location": {
            "instruction": (
                "Focus only on locations and full addresses. Recover residence, registered, service, project, "
                "delivery, court, and branch location lines, including split multi-line addresses."
            ),
            "categories": {"address", "footer", "definition"},
        },
        "account_identifier": {
            "instruction": (
                "Focus on bank names, account holders, contract numbers, case numbers, project numbers, and "
                "other labeled identifiers. Extract subjects and names, but do not return amounts."
            ),
            "categories": {"account", "identifier", "definition"},
        },
        "relation_alias": {
            "instruction": (
                "Focus on relationship lines and abbreviation definitions. Recover pairs such as full name "
                "and short name, applicant/respondent role lines, and project-company definitions."
            ),
            "categories": {"definition", "relation", "organization", "person"},
        },
        "repetition_hotspot": {
            "instruction": (
                "Focus on dense middle-body paragraphs where the same sensitive subjects appear repeatedly. "
                "Exhaustively recover every repeated organization, person, and closely related short-form "
                "mention in this hotspot block, even when later sentences omit labels."
            ),
            "categories": {"repetition", "organization", "person", "relation", "definition"},
        },
        "residual": {
            "instruction": (
                "Do a final residual sweep. Recover any still-missed verbatim person, organization, location, "
                "bank, account, project, or contract-number mentions in this high-risk block."
            ),
            "categories": {"definition", "relation", "address", "organization", "person", "account", "identifier", "footer", "repetition"},
        },
    }

    HIGH_RISK_CATEGORY_TOKENS = {
        "definition": ["以下简称", "下称", "简称", "又称", "项目公司", "甲公司", "乙公司", "丙公司"],
        "relation": ["股东", "法定代表人", "法人代表", "实际控制人", "联系人", "委托代理人", "代理人", "支付给", "汇至", "收款", "付款"],
        "address": ["地址", "住址", "住所", "通讯地址", "送达地址", "注册地址", "办公地址", "项目地址", "人民法院", "检察院"],
        "account": ["开户行", "户名", "账户", "账号", "收款单位", "付款单位", "银行"],
        "identifier": ["合同编号", "合同号", "案号", "项目编号", "统一社会信用代码"],
        "footer": ["签字", "签章", "盖章", "落款", "法定代表人", "法人代表", "日期"],
        "organization": ["公司", "集团", "中心", "研究院", "事务所", "银行", "支行", "分行"],
        "person": ["联系人", "法定代表人", "法人代表", "代理人", "股东", "委托人", "收款人", "付款人"],
        "repetition": [
            "再次",
            "多次",
            "反复",
            "仍",
            "继续",
            "后续",
            "另行",
            "一并",
            "分别",
            "再次向",
            "上诉请求",
            "事实与理由",
            "综上",
            "居间费",
            "退货款",
            "退款",
            "证据",
        ],
    }

    ENTITY_TYPE_PRIORITY = {
        "BANK_NAME": 1,
        "ACCOUNT_NAME": 2,
        "ORGANIZATION": 3,
        "PROJECT": 4,
        "CONTRACT_NO": 5,
        "LOCATION": 6,
        "POSITION": 7,
        "PERSON": 8,
    }

    BANK_TOKENS = ["银行", "支行", "分行", "营业部", "分理处", "信用社", "联社"]
    INSTITUTION_TOKENS = [
        "法院",
        "检察院",
        "公安局",
        "仲裁委员会",
        "事务所",
        "研究院",
        "服务中心",
        "管理局",
        "委员会",
        "中心",
        "学院",
        "医院",
    ]
    ORGANIZATION_TOKENS = ["公司", "集团", "中心", "研究院", "事务所", "院", "局", "所"] + BANK_TOKENS
    LOCATION_TOKENS = ["省", "市", "区", "县", "镇", "乡", "村", "路", "街", "号", "栋", "室", "广场", "大道"]
    ACCOUNT_LABEL_TOKENS = ["户名", "账户名称", "账户名", "收款单位", "收款方", "付款单位", "付款方", "结算单位"]
    BANK_LABEL_TOKENS = ["开户行", "收款银行", "付款银行"]
    PROJECT_LABEL_TOKENS = ["工程名称", "项目名称", "采购项目", "标段名称", "项目"]
    LEGAL_PARTY_LABEL_TOKENS = [
        "上诉人",
        "被上诉人",
        "原审原告",
        "原审被告",
        "原审原告一",
        "原审原告二",
        "原审原告三",
        "原审被告一",
        "原审被告二",
        "原审被告三",
        "申请人",
        "被申请人",
        "申请执行人",
        "被执行人",
        "原告",
        "原告一",
        "原告二",
        "原告三",
        "被告",
        "被告一",
        "被告二",
        "被告三",
        "第三人",
    ]
    PERSON_LABEL_TOKENS = ["法定代表人", "法人代表", "联系人", "委托代理人", "代理人", "法代"]
    ORGANIZATION_LABEL_TOKENS = [
        "甲方",
        "乙方",
        "丙方",
        "委托方",
        "受托方",
        "申请单位",
        "被申请单位",
    ] + LEGAL_PARTY_LABEL_TOKENS
    ADDRESS_LABEL_TOKENS = [
        "地址",
        "联系地址",
        "住址",
        "身份证住址",
        "住所",
        "住所地",
        "通讯地址",
        "送达地址",
        "注册地址",
        "办公地址",
        "工程地址",
        "项目地址",
    ]
    ORGANIZATION_GENERIC_TERMS = {
        "公司",
        "集团",
        "机构",
        "单位",
        "企业",
        "法院",
        "人民法院",
        "中级人民法院",
        "高级人民法院",
        "检察院",
        "公安局",
        "事务所",
        "研究院",
        "服务中心",
        "中心",
        "银行",
        "支行",
        "分行",
        "判决",
        "原审判决",
        "一审判决",
        "二审判决",
    }
    ORGANIZATION_STOP_PREFIXES = (
        "在",
        "详见",
        "见",
        "向",
        "将",
        "把",
        "与",
        "及",
        "和",
        "由",
        "被",
        "对",
        "按",
        "经",
        "于",
        "从",
        "就",
        "为",
        "并",
        "且",
        "后",
        "前",
        "原审",
        "一审",
        "二审",
        "已经",
        "再次",
        "另",
    )
    ORGANIZATION_NOISE_FRAGMENTS = {
        "收取",
        "取了",
        "提交",
        "证明",
        "备注",
        "内容",
        "形成",
        "属于",
        "支付",
        "付款",
        "转账",
        "凭证",
        "证据",
        "股东与",
        "指示",
        "要求",
        "表示",
        "主张",
        "签署",
        "签字",
        "确认",
        "办理",
    }
    IDENTIFIER_LABEL_TOKENS = (
        "统一社会信用代码",
        "信用代码",
        "身份证号",
        "身份证号码",
        "公民身份号码",
        "银行卡",
        "银行账号",
        "账号",
        "账户",
        "户名",
        "开户行",
    )

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 300,
        num_ctx: int | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.timeout = timeout
        self.num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        self.last_extract_metadata: Dict[str, Any] = {}
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.available = self._check_connection()

        logger.info("Initialized Ollama service: %s (%s)", self.base_url, self.model)
        if not self.available:
            logger.warning("Ollama is not reachable. Run `ollama serve` first.")

    def _check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_last_extract_metadata(self) -> Dict[str, Any]:
        return dict(self.last_extract_metadata)

    def set_progress_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self.progress_callback = callback

    def _emit_progress(
        self,
        *,
        stage: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
    ) -> None:
        if self.progress_callback is None:
            return

        try:
            self.progress_callback(
                {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": message,
                }
            )
        except Exception:
            logger.debug("Failed to emit LLM progress update", exc_info=True)

    def _run_async(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError("Synchronous Ollama helper cannot run inside an active event loop.")

    def extract_entities(self, text: str) -> List[Dict]:
        return self._run_async(self.extract_entities_async(text))

    async def extract_entities_async(self, text: str) -> List[Dict]:
        self.last_extract_metadata = {}
        logger.info("Ollama recognition started, text_length=%s", len(text))

        if not text.strip():
            return []

        if not await self._check_connection_async():
            self.available = False
            logger.warning("Ollama is unavailable, skip LLM recognition.")
            return []

        indexed_lines = self._build_indexed_lines(text)
        profile = self._get_model_profile(text_length=len(text), line_count=len(indexed_lines))
        self._emit_progress(stage="prepare", current=1, total=1, message="正在准备识别策略...")
        document_context = await self._build_document_context_async(text, profile)
        definition_hints = self._extract_definition_hints(text, indexed_lines)
        if definition_hints:
            document_context["definition_hints"] = definition_hints
        chunks = self._chunk_text(
            text,
            target_size=profile["chunk_target_size"],
            overlap=profile["chunk_overlap"],
        )
        entities: List[Dict] = []
        high_risk_blocks: List[Dict[str, Any]] = []

        for chunk_index, chunk in enumerate(chunks, start=1):
            prompt = self._build_prompt(
                chunk["text"],
                document_context=document_context,
                pass_name="base",
            )
            try:
                payload = await self._call_ollama_async(
                    prompt,
                    num_predict=profile["extract_num_predict"],
                )
                chunk_entities = self._parse_response(
                    payload=payload,
                    chunk_text=chunk["text"],
                    chunk_start=chunk["start"],
                )
                entities.extend(chunk_entities)
                logger.info(
                    "Ollama chunk %s/%s finished, entities=%s",
                    chunk_index,
                    len(chunks),
                    len(chunk_entities),
                )
                self._emit_progress(
                    stage="base",
                    current=chunk_index,
                    total=len(chunks),
                    message=f"正在识别正文第 {chunk_index}/{len(chunks)} 段...",
                )
            except Exception as exc:
                logger.error("Ollama chunk %s failed: %s", chunk_index, exc)

        review_27b_workflow = self._is_review_27b_workflow(profile)
        recall_passes_used: List[str] = []
        specialized_passes_used: List[str] = []
        repetition_hotspots: List[Dict[str, Any]] = []
        preliminary_entities = self._deduplicate_entities(
            self._refine_entities(
                text,
                entities,
                document_context=document_context,
            )
        )
        if profile["targeted_recall_passes"] > 0 and not review_27b_workflow:
            recall_snippets = self._build_targeted_recall_snippets(
                text,
                document_context=document_context,
                max_snippets=profile["targeted_recall_passes"],
                max_chars=profile["recall_snippet_size"],
            )
            for recall_index, snippet in enumerate(recall_snippets, start=1):
                prompt = self._build_prompt(
                    snippet["text"],
                    document_context=document_context,
                    pass_name=snippet["name"],
                    focus_instruction=snippet["instruction"],
                )
                try:
                    payload = await self._call_ollama_async(
                        prompt,
                        num_predict=profile["recall_num_predict"],
                    )
                    snippet_entities = self._parse_response(
                        payload=payload,
                        chunk_text=snippet["text"],
                        chunk_start=int(snippet["start"]),
                    )
                    entities.extend(snippet_entities)
                    recall_passes_used.append(snippet["name"])
                    logger.info(
                        "Ollama recall pass %s/%s (%s) finished, entities=%s",
                        recall_index,
                        len(recall_snippets),
                        snippet["name"],
                        len(snippet_entities),
                    )
                    self._emit_progress(
                        stage="recall",
                        current=recall_index,
                        total=len(recall_snippets),
                        message=f"正在执行重点补查 {recall_index}/{len(recall_snippets)}...",
                    )
                except Exception as exc:
                    logger.error(
                        "Ollama recall pass %s (%s) failed: %s",
                        recall_index,
                        snippet["name"],
                        exc,
                    )

        if profile["specialized_passes"] and profile["high_risk_block_limit"] > 0:
            if review_27b_workflow:
                specialized_snippets = self._build_review_27b_snippets(
                    text=text,
                    indexed_lines=indexed_lines,
                    document_context=document_context,
                    preliminary_entities=preliminary_entities,
                    definition_hints=definition_hints,
                    profile=profile,
                )
            else:
                high_risk_blocks = self._schedule_high_risk_blocks(
                    text,
                    indexed_lines,
                    max_chars=profile["recall_snippet_size"],
                    max_blocks=profile["high_risk_block_limit"],
                )
                repetition_hotspots = self._build_repetition_hotspot_blocks(
                    text,
                    indexed_lines,
                    preliminary_entities,
                    max_chars=profile["recall_snippet_size"],
                    max_blocks=max(2, min(6, profile["high_risk_block_limit"])),
                )
                if repetition_hotspots:
                    high_risk_blocks = self._merge_ranked_blocks(
                        high_risk_blocks,
                        repetition_hotspots,
                        max_blocks=profile["high_risk_block_limit"],
                    )
                specialized_snippets = self._build_specialized_recall_snippets(
                    high_risk_blocks,
                    definitions=definition_hints,
                    max_extra_passes=profile["complexity_extra_passes"],
                    max_total_passes=profile["specialized_pass_limit"],
                )
            stage_name = "deep_review" if review_27b_workflow else "specialized"
            stage_label = "定向精审" if review_27b_workflow else "专项复查"
            source_name = "ollama_review_27b" if review_27b_workflow else "ollama_specialized"
            for snippet_index, snippet in enumerate(specialized_snippets, start=1):
                prompt = self._build_prompt(
                    snippet["text"],
                    document_context=document_context,
                    pass_name=snippet["name"],
                    focus_instruction=snippet["instruction"],
                )
                try:
                    payload = await self._call_ollama_async(
                        prompt,
                        num_predict=profile["recall_num_predict"],
                    )
                    snippet_entities = self._parse_response(
                        payload=payload,
                        chunk_text=snippet["text"],
                        chunk_start=int(snippet["start"]),
                    )
                    for entity in snippet_entities:
                        entity["source"] = source_name
                        entity.setdefault("metadata", {})
                        entity["metadata"]["pass_name"] = snippet["name"]
                        entity["metadata"]["risk_categories"] = list(snippet.get("categories", []))
                        if review_27b_workflow:
                            entity["metadata"]["workflow"] = "review_27b"
                    entities.extend(snippet_entities)
                    specialized_passes_used.append(snippet["name"])
                    logger.info(
                        "Ollama %s pass %s/%s (%s) finished, entities=%s",
                        "review_27b" if review_27b_workflow else "specialized",
                        snippet_index,
                        len(specialized_snippets),
                        snippet["name"],
                        len(snippet_entities),
                    )
                    self._emit_progress(
                        stage=stage_name,
                        current=snippet_index,
                        total=len(specialized_snippets),
                        message=f"正在执行{stage_label} {snippet_index}/{len(specialized_snippets)}...",
                    )
                except Exception as exc:
                    logger.error(
                        "Ollama %s pass %s (%s) failed: %s",
                        "review_27b" if review_27b_workflow else "specialized",
                        snippet_index,
                        snippet["name"],
                        exc,
                    )

        self._emit_progress(stage="finalize", current=1, total=1, message="正在整理识别结果...")
        refined_entities = self._refine_entities(
            text,
            entities,
            document_context=document_context,
        )
        deduplicated = self._deduplicate_entities(refined_entities)
        if profile["prefer_llm_first"]:
            structured_entities = self._recover_labeled_entities(
                text,
                deduplicated,
                document_context=document_context,
            )
            if structured_entities:
                deduplicated = self._deduplicate_entities(deduplicated + structured_entities)
            prose_entities = self._recover_prose_entities(
                text,
                deduplicated,
                document_context=document_context,
            )
            if prose_entities:
                deduplicated = self._deduplicate_entities(deduplicated + prose_entities)
        if profile["enable_definition_recall"] and definition_hints:
            definition_entities = self._recover_definition_entities(
                text,
                deduplicated,
                definitions=definition_hints,
            )
            if definition_entities:
                deduplicated = self._deduplicate_entities(deduplicated + definition_entities)
        deduplicated = self._apply_definition_hints(deduplicated, definition_hints)

        high_risk_payload = [
            {
                "name": block["name"],
                "risk_score": block["risk_score"],
                "complexity_score": block["complexity_score"],
                "categories": list(block.get("categories", [])),
            }
            for block in high_risk_blocks
        ]
        self.last_extract_metadata = {
            "engine_strategy": profile["engine_strategy"],
            "document_type": document_context["document_type"],
            "document_type_label": document_context["label"],
            "document_type_reason": document_context.get("reason", ""),
            "document_type_confidence": document_context.get("confidence", ""),
            "document_type_source": document_context.get("source", ""),
            "focus_plan": list(document_context.get("focuses", [])),
            "recall_passes": recall_passes_used,
            "specialized_passes": specialized_passes_used,
            "definition_hints": [
                {
                    "alias": item["alias"],
                    "full_text": item["full_text"],
                    "entity_type": item["entity_type"],
                    "canonical_key": item["canonical_key"],
                }
                for item in definition_hints[:20]
            ],
            "high_risk_blocks": high_risk_payload,
            "repetition_hotspots": [
                {
                    "name": block["name"],
                    "risk_score": block["risk_score"],
                    "complexity_score": block["complexity_score"],
                    "categories": list(block.get("categories", [])),
                    "matched_tokens": list(block.get("matched_tokens", []))[:6],
                }
                for block in repetition_hotspots[:8]
            ],
            "chunk_count": len(chunks),
            "entity_count": len(deduplicated),
            "structured_backfill": bool(profile["prefer_llm_first"]),
            "prose_backfill": bool(profile["prefer_llm_first"]),
            "quality_mode": "review_targeted_precision" if review_27b_workflow else "stable_high_recall",
            "workflow_variant": "review_27b" if review_27b_workflow else "precision_4b",
            "runtime_budget_tier": profile["runtime_budget_tier"],
            "runtime_budget_label": profile["runtime_budget_label"],
        }
        logger.info(
            "Ollama recognition finished, strategy=%s, document_type=%s, entities=%s, recall_passes=%s, specialized_passes=%s",
            profile["engine_strategy"],
            document_context["document_type"],
            len(deduplicated),
            recall_passes_used,
            specialized_passes_used,
        )
        return deduplicated

    async def _build_document_context_async(self, text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        indexed_lines = self._build_indexed_lines(text)
        heuristic = self._infer_document_type_from_rules(text, indexed_lines)
        llm_choice = {}

        if profile["prefer_llm_first"]:
            llm_choice = await self._classify_document_type_async(
                indexed_lines=indexed_lines,
                heuristic=heuristic,
                num_predict=profile["classify_num_predict"],
            )

        selected = self._merge_document_type_candidates(heuristic, llm_choice)
        selected["label"] = self.DOCUMENT_TYPE_LABELS.get(
            selected["document_type"],
            self.DOCUMENT_TYPE_LABELS["other"],
        )
        selected["focuses"] = self._build_focus_plan(
            selected["document_type"],
            llm_choice.get("focuses", []),
        )
        selected["prompt_hint"] = self.DOCUMENT_TYPE_PROMPT_HINTS.get(
            selected["document_type"],
            self.DOCUMENT_TYPE_PROMPT_HINTS["other"],
        )
        return selected

    def _build_document_context(self, text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_async(self._build_document_context_async(text, profile))

    def _infer_document_type_from_rules(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        head_lines = [item["text"] for item in indexed_lines[:18]]
        tail_lines = [item["text"] for item in indexed_lines[-10:]]
        sample_text = "\n".join(head_lines + tail_lines)
        sample_text = f"{sample_text}\n{text[:2200]}"
        title_block = "\n".join(head_lines[:4])

        scores: Dict[str, int] = {key: 0 for key in self.DOCUMENT_TYPE_LABELS if key != "other"}
        reasons: Dict[str, List[str]] = {key: [] for key in scores}

        for document_type, hints in self.DOCUMENT_TYPE_HINTS.items():
            for token, weight in hints:
                if token in sample_text:
                    scores[document_type] += weight
                    reasons[document_type].append(token)

        if "合同" in title_block or "协议" in title_block:
            scores["contract"] += 3
        if "申请书" in title_block or "起诉状" in title_block or "答辩状" in title_block:
            scores["legal_application"] += 3
        if re.search(r"\(\d{4}\).{0,20}号", title_block):
            scores["legal_application"] += 2
            scores["enforcement_paper"] += 2
        if "续行冻结" in title_block or ("执行" in title_block and "申请书" in title_block):
            scores["enforcement_paper"] += 4
        if "开户行" in sample_text and ("账户" in sample_text or "账号" in sample_text):
            scores["financial_account_material"] += 3

        best_type = max(scores, key=scores.get) if scores else "other"
        best_score = scores.get(best_type, 0)
        if best_score < 4:
            return {
                "document_type": "other",
                "confidence": "low",
                "reason": "No strong rule-based title or section signals were found.",
                "source": "heuristic",
                "score": best_score,
            }

        matched_tokens = reasons.get(best_type, [])[:6]
        reason_text = ", ".join(matched_tokens) if matched_tokens else "matched title and section cues"
        return {
            "document_type": best_type,
            "confidence": self._score_to_confidence(best_score),
            "reason": reason_text,
            "source": "heuristic",
            "score": best_score,
        }

    def _classify_document_type(
        self,
        *,
        indexed_lines: List[Dict[str, Any]],
        heuristic: Dict[str, Any],
        num_predict: int,
    ) -> Dict[str, Any]:
        return self._run_async(
            self._classify_document_type_async(
                indexed_lines=indexed_lines,
                heuristic=heuristic,
                num_predict=num_predict,
            )
        )

    async def _classify_document_type_async(
        self,
        *,
        indexed_lines: List[Dict[str, Any]],
        heuristic: Dict[str, Any],
        num_predict: int,
    ) -> Dict[str, Any]:
        prompt = self._build_document_classification_prompt(indexed_lines, heuristic)
        try:
            payload = await self.generate_json_async(prompt, num_predict=num_predict)
        except Exception as exc:
            logger.warning("Document type classification failed: %s", exc)
            return {}

        parsed = self._load_json_object(payload, required_keys={"document_type"})
        if not isinstance(parsed, dict):
            return {}

        document_type = str(parsed.get("document_type", "")).strip().lower()
        if document_type not in self.DOCUMENT_TYPE_LABELS:
            return {}

        confidence = str(parsed.get("confidence", "")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        focuses: List[str] = []
        raw_focuses = parsed.get("focuses")
        if isinstance(raw_focuses, list):
            for item in raw_focuses:
                focus_key = str(item).strip()
                if focus_key in self.FOCUS_LIBRARY and focus_key not in focuses:
                    focuses.append(focus_key)

        return {
            "document_type": document_type,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "")).strip(),
            "source": "llm",
            "focuses": focuses,
        }

    def _merge_document_type_candidates(
        self,
        heuristic: Dict[str, Any],
        llm_choice: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not llm_choice:
            return dict(heuristic)

        heuristic_type = heuristic.get("document_type", "other")
        llm_type = llm_choice.get("document_type", "other")
        heuristic_rank = self._confidence_rank(str(heuristic.get("confidence", "low")))
        llm_rank = self._confidence_rank(str(llm_choice.get("confidence", "low")))

        if llm_type == heuristic_type:
            return {
                "document_type": llm_type,
                "confidence": llm_choice.get("confidence", heuristic.get("confidence", "medium")),
                "reason": llm_choice.get("reason") or heuristic.get("reason", ""),
                "source": "heuristic+llm",
            }

        heuristic_score = int(heuristic.get("score", 0) or 0)
        if heuristic_type != "other" and heuristic_score >= 10 and heuristic_rank >= llm_rank:
            return dict(heuristic)
        if heuristic_type == "other" and llm_type != "other":
            return dict(llm_choice)
        if llm_rank >= heuristic_rank:
            return dict(llm_choice)
        return dict(heuristic)

    def _build_focus_plan(self, document_type: str, llm_focuses: List[str]) -> List[str]:
        default_plan = self.DOC_TYPE_FOCUS_PLAN.get(document_type, self.DOC_TYPE_FOCUS_PLAN["other"])
        combined: List[str] = []
        for focus_name in list(default_plan) + list(llm_focuses):
            if focus_name in self.FOCUS_LIBRARY and focus_name not in combined:
                combined.append(focus_name)
        return combined

    def _build_document_classification_prompt(
        self,
        indexed_lines: List[Dict[str, Any]],
        heuristic: Dict[str, Any],
    ) -> str:
        head_text = "\n".join(item["text"] for item in indexed_lines[:14])
        tail_text = "\n".join(item["text"] for item in indexed_lines[-10:])
        allowed_focuses = json.dumps(sorted(self.FOCUS_LIBRARY.keys()), ensure_ascii=False)
        return f"""
You are classifying a Chinese legal/business document for downstream entity extraction.

Choose exactly one document_type from:
["contract","legal_application","enforcement_paper","financial_account_material","other"]

Also choose 2 to 5 relevant focus keys from:
{allowed_focuses}

Return strict JSON only in this exact shape:
{{
  "document_type": "contract",
  "confidence": "high",
  "reason": "title contains 技术服务合同 and the header contains 甲方/乙方",
  "focuses": ["party_header", "project_section", "footer_signature"]
}}

Heuristic guess:
{json.dumps(heuristic, ensure_ascii=False, indent=2)}

Header excerpt:
\"\"\"
{head_text}
\"\"\"

Footer excerpt:
\"\"\"
{tail_text}
\"\"\"
""".strip()

    def _build_targeted_recall_snippets(
        self,
        text: str,
        *,
        document_context: Dict[str, Any],
        max_snippets: int,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        if max_snippets <= 0 or max_chars <= 0:
            return []

        indexed_lines = self._build_indexed_lines(text)
        if not indexed_lines:
            return []

        snippets: List[Dict[str, Any]] = []
        seen_texts = set()

        focus_snippet_groups: List[List[Dict[str, Any]]] = []
        for focus_name in document_context.get("focuses", []):
            focus_config = self.FOCUS_LIBRARY.get(focus_name)
            if not focus_config:
                continue

            group: List[Dict[str, Any]] = []
            block_count = 0
            for block in self._build_focus_blocks(text, indexed_lines, focus_config, max_chars):
                normalized = re.sub(r"\s+", "", block["text"])
                if not normalized:
                    continue
                block_count += 1
                snippet_name = focus_name if block_count == 1 else f"{focus_name}_{block_count}"
                group.append(
                    {
                        "name": snippet_name,
                        "instruction": str(focus_config["instruction"]),
                        "text": block["text"],
                        "start": block["start"],
                        "normalized": normalized,
                    }
                )

            if group:
                focus_snippet_groups.append(group)

        group_index = 0
        while len(snippets) < max_snippets and focus_snippet_groups:
            added_in_round = False
            for group in focus_snippet_groups:
                if len(snippets) >= max_snippets:
                    break
                if group_index >= len(group):
                    continue

                snippet = group[group_index]
                normalized = str(snippet["normalized"])
                if normalized in seen_texts:
                    continue

                seen_texts.add(normalized)
                snippets.append(
                    {
                        "name": str(snippet["name"]),
                        "instruction": str(snippet["instruction"]),
                        "text": str(snippet["text"]),
                        "start": int(snippet["start"]),
                    }
                )
                added_in_round = True

            if not added_in_round:
                break
            group_index += 1

        if len(snippets) < max_snippets:
            for snippet in self._build_high_risk_recall_snippets(text, indexed_lines, max_chars):
                if len(snippets) >= max_snippets:
                    break

                normalized = re.sub(r"\s+", "", snippet["text"])
                if not normalized or normalized in seen_texts:
                    continue

                seen_texts.add(normalized)
                snippets.append(snippet)

        return snippets

    def _build_high_risk_recall_snippets(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        if not indexed_lines:
            return []

        label_tokens = (
            self.BANK_LABEL_TOKENS
            + self.ACCOUNT_LABEL_TOKENS
            + self.PROJECT_LABEL_TOKENS
            + self.ORGANIZATION_LABEL_TOKENS
            + self.ADDRESS_LABEL_TOKENS
            + self.PERSON_LABEL_TOKENS
            + ["合同编号", "合同号", "案号", "盖章", "签字"]
        )
        narrative_tokens = [
            "公司",
            "集团",
            "股东",
            "法定代表人",
            "实际控制人",
            "付款至",
            "支付给",
            "转给",
            "汇给",
            "汇至",
            "收款至",
            "个人银行账户",
            "个人账户",
            "银行账户",
            "简称",
            "以下简称",
            "又称",
            "签署",
        ]

        selected_indices = set()
        total_lines = len(indexed_lines)

        for item in indexed_lines:
            line_text = str(item["text"])
            risk_score = 0

            if any(token in line_text for token in label_tokens):
                risk_score += 2
            if any(token in line_text for token in narrative_tokens):
                risk_score += 1
            if ("：" in line_text or ":" in line_text) and len(line_text) <= 140:
                risk_score += 1
            if sum(token in line_text for token in self.LOCATION_TOKENS) >= 2:
                risk_score += 1
            if any(token in line_text for token in ["盖章", "签字", "落款"]):
                risk_score += 1
            if "公司" in line_text and any(token in line_text for token in ["股东", "支付", "付款", "转给", "汇", "账户"]):
                risk_score += 1

            if risk_score < 2:
                continue

            start = max(0, int(item["index"]) - 1)
            end = min(total_lines, int(item["index"]) + 2)
            for offset in range(start, end):
                selected_indices.add(offset)

        if not selected_indices:
            return []

        instruction = (
            "Inspect high-risk label/value lines, split rows, footer blocks, and address-heavy lines. "
            "Recover full organizations, persons, projects, contract numbers, bank names, account names, "
            "and full addresses that may have been missed in the primary pass."
        )
        snippets: List[Dict[str, Any]] = []
        current_indices: List[int] = []
        previous_index: int | None = None

        for index in sorted(selected_indices):
            if previous_index is not None and index > previous_index + 1:
                if current_indices:
                    block = self._materialize_focus_block(text, indexed_lines, current_indices, max_chars)
                    if block.get("text"):
                        snippets.append(
                            {
                                "name": f"high_risk_{len(snippets) + 1}",
                                "instruction": instruction,
                                "text": block["text"],
                                "start": block["start"],
                            }
                        )
                current_indices = [index]
            else:
                current_indices.append(index)
            previous_index = index

        if current_indices:
            block = self._materialize_focus_block(text, indexed_lines, current_indices, max_chars)
            if block.get("text"):
                snippets.append(
                    {
                        "name": f"high_risk_{len(snippets) + 1}",
                        "instruction": instruction,
                        "text": block["text"],
                        "start": block["start"],
                    }
                )

        return snippets

    def _schedule_high_risk_blocks(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        *,
        max_chars: int,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        if not indexed_lines or max_blocks <= 0:
            return []

        candidates: List[Dict[str, Any]] = []
        total_lines = len(indexed_lines)

        for item in indexed_lines:
            line_text = str(item["text"]).strip()
            if not line_text:
                continue

            categories: set[str] = set()
            matched_tokens: List[str] = []
            risk_score = 0
            complexity_score = 0

            for category, tokens in self.HIGH_RISK_CATEGORY_TOKENS.items():
                hits = [token for token in tokens if token in line_text]
                if not hits:
                    continue
                categories.add(category)
                matched_tokens.extend(hits[:3])
                risk_score += 2 if category in {"definition", "account", "identifier", "footer"} else 1

            if ("：" in line_text or ":" in line_text) and len(line_text) <= 140:
                complexity_score += 1
            if sum(token in line_text for token in self.LOCATION_TOKENS) >= 2:
                categories.add("address")
                risk_score += 1
                complexity_score += 1
            if len(line_text) >= 48:
                complexity_score += 1
            if len(categories) >= 3:
                complexity_score += 1
            if any(token in line_text for token in ["申请人", "被申请人", "以下简称", "简称", "又称", "项目公司"]):
                complexity_score += 1
            if any(token in line_text for token in ["签字", "签章", "盖章", "落款"]):
                categories.add("footer")
                complexity_score += 1
            if risk_score < 2:
                continue

            center_index = int(item["index"])
            start_index = max(0, center_index - 1)
            end_index = min(total_lines, center_index + 2)
            block_indices = list(range(start_index, end_index))
            block = self._materialize_focus_block(text, indexed_lines, block_indices, max_chars)
            normalized = re.sub(r"\s+", "", str(block.get("text", "")))
            if not normalized:
                continue

            candidates.append(
                {
                    "name": f"risk_{len(candidates) + 1}",
                    "text": block["text"],
                    "start": int(block["start"]),
                    "risk_score": risk_score,
                    "complexity_score": complexity_score,
                    "categories": sorted(categories),
                    "matched_tokens": sorted(set(matched_tokens))[:8],
                    "normalized": normalized,
                }
            )

        deduplicated: Dict[str, Dict[str, Any]] = {}
        for item in candidates:
            existing = deduplicated.get(item["normalized"])
            if existing is None or (
                item["complexity_score"],
                item["risk_score"],
                -item["start"],
            ) > (
                existing["complexity_score"],
                existing["risk_score"],
                -existing["start"],
            ):
                deduplicated[item["normalized"]] = item

        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (
                -int(item["complexity_score"]),
                -int(item["risk_score"]),
                item["start"],
            ),
        )
        return ordered[:max_blocks]

    def _build_repetition_hotspot_blocks(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        *,
        max_chars: int,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        if not indexed_lines or max_blocks <= 0:
            return []

        candidate_occurrences: Dict[tuple[str, str], Dict[str, Any]] = {}
        repeatable_types = {"ORGANIZATION", "PERSON", "PROJECT", "BANK_NAME", "ACCOUNT_NAME"}

        def record_candidate(entity_type: str, candidate_text: str, start: int, end: int) -> None:
            normalized = re.sub(r"\s+", "", str(candidate_text or ""))
            if len(normalized) < 2:
                return
            if entity_type == "ORGANIZATION" and not self._is_valid_repetition_organization_candidate(normalized):
                return
            if entity_type == "PERSON" and not self._looks_like_person_name(normalized):
                return
            key = (entity_type, normalized)
            payload = candidate_occurrences.setdefault(
                key,
                {
                    "type": entity_type,
                    "text": normalized,
                    "occurrences": [],
                },
            )
            payload["occurrences"].append((int(start), int(end)))

        for entity in entities:
            entity_type = str(entity.get("type", "")).upper()
            if entity_type not in repeatable_types:
                continue
            entity_text = str(entity.get("text", ""))
            for start, end, matched_text in self._find_entity_spans(text, entity_text):
                record_candidate(entity_type, matched_text, start, end)

        for start, end, matched_text in self._extract_prose_organization_matches(text):
            record_candidate("ORGANIZATION", matched_text, start, end)
        for start, end, matched_text in self._extract_prose_person_matches(text):
            record_candidate("PERSON", matched_text, start, end)

        total_lines = len(indexed_lines)
        middle_start = max(1, int(total_lines * 0.18))
        middle_end = max(middle_start + 1, int(total_lines * 0.88))
        candidates: List[Dict[str, Any]] = []

        for payload in candidate_occurrences.values():
            occurrences = sorted(set(payload["occurrences"]))
            if len(occurrences) < 2:
                continue

            line_hits = [
                self._line_index_for_offset(indexed_lines, start)
                for start, _end in occurrences
            ]
            line_hits = [index for index in line_hits if index >= 0]
            if len(line_hits) < 2:
                continue

            line_hit_counter = Counter(line_hits)
            dominant_line = line_hit_counter.most_common(1)[0][0]
            same_line_dense = line_hit_counter[dominant_line] >= 3

            middle_hits = [index for index in line_hits if middle_start <= index <= middle_end]
            if len(middle_hits) < 2 and len(line_hits) < 3 and not same_line_dense:
                continue

            cluster_indices = self._select_dense_line_cluster(middle_hits or line_hits)
            if len(cluster_indices) < 2:
                if not same_line_dense:
                    continue
                cluster_indices = [dominant_line]

            if same_line_dense and dominant_line not in cluster_indices:
                cluster_indices = sorted(set(cluster_indices + [dominant_line]))
            if len(cluster_indices) < 1:
                continue

            block_start = max(0, cluster_indices[0] - 1)
            block_end = min(total_lines, cluster_indices[-1] + 2)
            block = self._materialize_focus_block(
                text,
                indexed_lines,
                list(range(block_start, block_end)),
                max_chars,
            )
            normalized_block = re.sub(r"\s+", "", str(block.get("text", "")))
            if not normalized_block:
                continue

            candidate_text = str(payload["text"])
            candidate_categories = {"repetition"}
            if payload["type"] == "ORGANIZATION":
                candidate_categories.add("organization")
            elif payload["type"] == "PERSON":
                candidate_categories.add("person")
            else:
                candidate_categories.add("relation")

            risk_score = min(8, 2 + len(occurrences))
            complexity_score = min(7, 2 + len(cluster_indices))
            if middle_hits:
                complexity_score += 1
            if same_line_dense:
                complexity_score += 2

            candidates.append(
                {
                    "name": f"repetition_{len(candidates) + 1}",
                    "text": str(block["text"]),
                    "start": int(block["start"]),
                    "risk_score": risk_score,
                    "complexity_score": complexity_score,
                    "categories": sorted(candidate_categories),
                    "matched_tokens": [candidate_text],
                    "normalized": normalized_block,
                }
            )

        candidates.extend(
            self._build_dense_narrative_hotspot_blocks(
                text,
                indexed_lines,
                max_chars=max_chars,
                max_blocks=max_blocks,
            )
        )

        return self._merge_ranked_blocks([], candidates, max_blocks=max_blocks)

    def _build_dense_narrative_hotspot_blocks(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        *,
        max_chars: int,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        if not indexed_lines or max_blocks <= 0:
            return []

        total_lines = len(indexed_lines)
        middle_start = max(1, int(total_lines * 0.18))
        middle_end = max(middle_start + 1, int(total_lines * 0.88))
        candidates: List[Dict[str, Any]] = []

        for line in indexed_lines:
            line_index = int(line["index"])
            if line_index < middle_start or line_index > middle_end:
                continue

            line_text = str(line.get("text", "")).strip()
            normalized_line = re.sub(r"\s+", "", line_text)
            if len(normalized_line) < 30:
                continue

            cue_hits = [token for token in self.DENSE_NARRATIVE_TOKENS if token in line_text]
            if not cue_hits:
                continue

            organization_hits = {
                match[2]
                for match in self._extract_prose_organization_matches(line_text)
                if self._is_valid_repetition_organization_candidate(match[2])
            }
            person_hits = {
                match[2]
                for match in self._extract_prose_person_matches(line_text)
                if self._looks_like_person_name(match[2])
            }
            short_company_hits = [
                item
                for item in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,8}公司", line_text)
                if self._is_valid_repetition_organization_candidate(item)
            ]
            short_company_counter = Counter(short_company_hits)

            subject_count = len(organization_hits) + len(person_hits) + len(short_company_counter)
            repeated_short_mentions = max(short_company_counter.values(), default=0)
            if subject_count < 3 and repeated_short_mentions < 2:
                continue

            block = self._materialize_focus_block(
                text,
                indexed_lines,
                list(range(max(0, line_index - 1), min(total_lines, line_index + 2))),
                max_chars,
            )
            normalized_block = re.sub(r"\s+", "", str(block.get("text", "")))
            if not normalized_block:
                continue

            matched_tokens = sorted(organization_hits | person_hits)[:4]
            matched_tokens.extend(cue_hits[:3])
            if short_company_counter:
                matched_tokens.extend(
                    [item for item, _count in short_company_counter.most_common(2)]
                )

            categories = {"repetition", "relation"}
            if organization_hits or short_company_counter:
                categories.add("organization")
            if person_hits:
                categories.add("person")

            candidates.append(
                {
                    "name": f"dense_narrative_{len(candidates) + 1}",
                    "text": str(block["text"]),
                    "start": int(block["start"]),
                    "risk_score": min(9, 3 + len(cue_hits) + repeated_short_mentions),
                    "complexity_score": min(9, 3 + subject_count),
                    "categories": sorted(categories),
                    "matched_tokens": matched_tokens[:8],
                    "normalized": normalized_block,
                }
            )

        return self._merge_ranked_blocks([], candidates, max_blocks=max_blocks)

    def _merge_ranked_blocks(
        self,
        base_blocks: List[Dict[str, Any]],
        extra_blocks: List[Dict[str, Any]],
        *,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for item in list(base_blocks) + list(extra_blocks):
            normalized = str(item.get("normalized") or re.sub(r"\s+", "", str(item.get("text", ""))))
            if not normalized:
                continue
            candidate = dict(item)
            candidate["normalized"] = normalized
            existing = merged.get(normalized)
            if existing is None:
                merged[normalized] = candidate
                continue

            merged_categories = sorted(set(existing.get("categories", [])) | set(candidate.get("categories", [])))
            merged_tokens = sorted(set(existing.get("matched_tokens", [])) | set(candidate.get("matched_tokens", [])))
            if (
                int(candidate.get("complexity_score", 0)),
                int(candidate.get("risk_score", 0)),
                -int(candidate.get("start", 0)),
            ) > (
                int(existing.get("complexity_score", 0)),
                int(existing.get("risk_score", 0)),
                -int(existing.get("start", 0)),
            ):
                winner = candidate
            else:
                winner = existing
            winner["categories"] = merged_categories
            winner["matched_tokens"] = merged_tokens[:8]
            merged[normalized] = winner

        ordered = sorted(
            merged.values(),
            key=lambda item: (
                -int(item.get("complexity_score", 0)),
                -int(item.get("risk_score", 0)),
                int(item.get("start", 0)),
            ),
        )
        return ordered[:max_blocks]

    def _line_index_for_offset(self, indexed_lines: List[Dict[str, Any]], offset: int) -> int:
        for item in indexed_lines:
            if int(item["start"]) <= offset < int(item["end"]):
                return int(item["index"])
        return -1

    def _select_dense_line_cluster(self, line_hits: List[int]) -> List[int]:
        if not line_hits:
            return []

        sorted_hits = sorted(line_hits)
        best_cluster: List[int] = []
        current_cluster: List[int] = []
        previous: Optional[int] = None

        for index in sorted_hits:
            if previous is None or index <= previous + 3:
                current_cluster.append(index)
            else:
                if len(current_cluster) > len(best_cluster):
                    best_cluster = current_cluster
                current_cluster = [index]
            previous = index

        if len(current_cluster) > len(best_cluster):
            best_cluster = current_cluster

        return sorted(set(best_cluster))

    def _is_valid_repetition_organization_candidate(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if len(normalized) < 2:
            return False
        if normalized in self.ORGANIZATION_GENERIC_TERMS:
            return False
        if normalized in {"建筑劳务", "劳务", "工程建设", "工程技术", "工程设计", "建筑工程"}:
            return False
        if re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,8}", normalized):
            return not any(fragment in normalized for fragment in self.ORGANIZATION_NOISE_FRAGMENTS)
        return self._is_valid_organization_candidate_text(normalized)

    def _build_specialized_recall_snippets(
        self,
        blocks: List[Dict[str, Any]],
        *,
        definitions: List[Dict[str, Any]],
        max_extra_passes: int,
        max_total_passes: int,
    ) -> List[Dict[str, Any]]:
        if not blocks:
            return []

        snippets: List[Dict[str, Any]] = []
        seen = set()
        definition_categories = {"definition"} if definitions else set()

        for pass_name, config in self.SPECIALIZED_PASS_LIBRARY.items():
            if max_total_passes > 0 and len(snippets) >= max_total_passes:
                break

            categories = set(config["categories"]) | definition_categories
            matching_blocks = [
                block
                for block in blocks
                if categories.intersection(set(block.get("categories", [])))
            ]
            if not matching_blocks:
                continue

            budget = 1
            if any(int(block.get("complexity_score", 0)) >= 3 for block in matching_blocks):
                budget += min(max_extra_passes, 2)

            for block in matching_blocks[:budget]:
                if max_total_passes > 0 and len(snippets) >= max_total_passes:
                    break
                key = (pass_name, block["normalized"])
                if key in seen:
                    continue
                seen.add(key)
                snippets.append(
                    {
                        "name": f"{pass_name}_{len(snippets) + 1}",
                        "instruction": str(config["instruction"]),
                        "text": str(block["text"]),
                        "start": int(block["start"]),
                        "categories": sorted(set(block.get("categories", []))),
                    }
                )

        return snippets


    def _extract_definition_hints(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        patterns = [
            re.compile(
                r'(?P<full>[\u4e00-\u9fa5A-Za-z0-9\uFF08\uFF09()\u00B7\s]{2,80}?)'
                r'(?:\uFF08|\()(?:(?:\u4EE5\u4E0B)?\u7B80\u79F0|\u4E0B\u79F0|\u53C8\u79F0)\s*["\'“”‘’]?'
                r'(?P<alias>[\u4e00-\u9fa5A-Za-z0-9]{2,20})["\'“”‘’]?(?:\uFF09|\))'
            ),
            re.compile(
                r'(?P<full>[\u4e00-\u9fa5A-Za-z0-9\u00B7\s]{2,80}?)[\uFF0C,]\s*'
                r'(?:(?:\u4EE5\u4E0B)?\u7B80\u79F0|\u4E0B\u79F0|\u53C8\u79F0)\s*["\'“”‘’]?'
                r'(?P<alias>[\u4e00-\u9fa5A-Za-z0-9]{2,20})["\'“”‘’]?'
            ),
        ]

        role_anchor_patterns = [
            re.compile(
                r'^(?P<alias>甲方|乙方|丙方|甲公司|乙公司|丙公司|项目公司|申请人|被申请人(?:公司)?|申请执行人|被执行人|原告|被告|第三人)\s*[:：]\s*(?P<full>.+)$'
            ),
            re.compile(
                r'^(?P<alias>项目公司|申请人|被申请人(?:公司)?|申请执行人|被执行人|原告|被告|第三人)\s*(?:系|为|指)\s*(?P<full>.+)$'
            ),
        ]

        hints: List[Dict[str, Any]] = []
        seen = set()

        for pattern in patterns:
            for match in pattern.finditer(text):
                full_text = self._clean_definition_text(match.group("full"))
                alias = self._clean_definition_text(match.group("alias"))
                if not full_text or not alias or full_text == alias:
                    continue

                entity_type = self._infer_definition_entity_type(full_text, alias)
                if not entity_type:
                    continue

                key = (entity_type, re.sub(r"\s+", "", full_text), re.sub(r"\s+", "", alias))
                if key in seen:
                    continue
                seen.add(key)

                hints.append(
                    {
                        "full_text": full_text,
                        "alias": alias,
                        "entity_type": entity_type,
                        "canonical_key": f"DEF_{entity_type}_{len(hints) + 1}",
                        "role": self._infer_definition_role(alias),
                        "evidence": self._locate_definition_evidence(indexed_lines, match.start(), match.end()),
                    }
                )

        for item in indexed_lines:
            line_text = str(item.get("text", "")).strip()
            if not line_text or len(line_text) > 160:
                continue
            for pattern in role_anchor_patterns:
                match = pattern.search(line_text)
                if not match:
                    continue

                full_text = self._clean_definition_text(match.group("full"))
                alias = self._clean_definition_text(match.group("alias"))
                if not full_text or not alias or full_text == alias:
                    continue

                entity_type = self._infer_definition_entity_type(full_text, alias)
                if not entity_type:
                    continue

                key = (entity_type, re.sub(r"\s+", "", full_text), re.sub(r"\s+", "", alias))
                if key in seen:
                    continue
                seen.add(key)

                hints.append(
                    {
                        "full_text": full_text,
                        "alias": alias,
                        "entity_type": entity_type,
                        "canonical_key": f"DEF_{entity_type}_{len(hints) + 1}",
                        "role": self._infer_definition_role(alias),
                        "evidence": line_text,
                    }
                )
                break

        return hints

    def _extract_definition_hints_safe(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._extract_definition_hints(text, indexed_lines)

    def _extract_definition_hints_v2(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._extract_definition_hints(text, indexed_lines)

    def _clean_definition_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", "", str(text or ""))
        cleaned = cleaned.strip(" \t\r\n,;.:，；。：()（）[]{}【】\"'“”‘’")
        return re.sub(r"(以下简称|下称|又称)$", "", cleaned)

    def _infer_definition_entity_type(self, full_text: str, alias: str) -> str:
        combined = f"{full_text}{alias}"
        if any(
            token in combined
            for token in [
                "\u516C\u53F8",
                "\u96C6\u56E2",
                "\u4E2D\u5FC3",
                "\u7814\u7A76\u9662",
                "\u4E8B\u52A1\u6240",
                "\u94F6\u884C",
                "\u652F\u884C",
                "\u5206\u884C",
                "\u6CD5\u9662",
                "\u68C0\u5BDF\u9662",
            ]
        ):
            return "ORGANIZATION"
        if any(token in combined for token in ["\u9879\u76EE", "\u5DE5\u7A0B", "\u6807\u6BB5"]):
            return "PROJECT"
        if any(
            token in combined
            for token in ["\u5730\u5740", "\u4F4F\u5740", "\u4F4F\u6240", "\u8DEF", "\u8857", "\u53F7"]
        ):
            return "LOCATION"
        if self._looks_like_person_name(alias) or self._looks_like_person_name(full_text):
            return "PERSON"
        return ""

    def _infer_definition_role(self, alias: str) -> str:
        if alias in {"\u7532\u516C\u53F8", "\u7532\u65B9"}:
            return "PARTY_A"
        if alias in {"\u4E59\u516C\u53F8", "\u4E59\u65B9"}:
            return "PARTY_B"
        if alias in {"\u4E19\u516C\u53F8", "\u4E19\u65B9"}:
            return "PARTY_C"
        return ""

    def _locate_definition_evidence(
        self,
        indexed_lines: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> str:
        if not indexed_lines:
            return ""
        for item in indexed_lines:
            line_start = int(item["start"])
            line_end = int(item["end"])
            if start < line_end and end > line_start:
                return str(item["text"]).strip()
        return ""

    def _recover_definition_entities(
        self,
        text: str,
        existing_entities: List[Dict],
        *,
        definitions: List[Dict[str, Any]],
    ) -> List[Dict]:
        if not definitions:
            return []

        recovered: List[Dict] = []
        occupied = {
            (int(entity["start"]), int(entity["end"]), str(entity["text"]).strip())
            for entity in existing_entities
        }
        occupied_ranges = [
            (int(entity["start"]), int(entity["end"]))
            for entity in existing_entities
        ]

        for definition in definitions:
            metadata = {
                "canonical_key": definition["canonical_key"],
                "definition_alias": definition["alias"],
                "definition_full_text": definition["full_text"],
                "definition_role": definition.get("role", ""),
                "definition_evidence": definition.get("evidence", ""),
            }
            for value in [definition["full_text"], definition["alias"]]:
                for start, end, matched_text in self._find_entity_spans(text, value):
                    self._record_recovered_entity(
                        recovered,
                        occupied,
                        occupied_ranges,
                        entity_type=str(definition["entity_type"]),
                        start=start,
                        end=end,
                        matched_text=matched_text,
                        score=0.91,
                        source="ollama_definition",
                        metadata=metadata,
                    )

        return recovered

    def _apply_definition_hints(
        self,
        entities: List[Dict],
        definitions: List[Dict[str, Any]],
    ) -> List[Dict]:
        if not definitions:
            return entities

        definition_map: Dict[str, Dict[str, Any]] = {}
        for definition in definitions:
            for value in [definition["full_text"], definition["alias"]]:
                normalized = re.sub(r"\s+", "", value)
                if normalized:
                    definition_map[normalized] = definition

        enriched: List[Dict] = []
        for entity in entities:
            item = dict(entity)
            normalized = re.sub(r"\s+", "", str(item.get("text", "")))
            definition = definition_map.get(normalized)
            if definition:
                item["canonical_key"] = definition["canonical_key"]
                item.setdefault("metadata", {})
                item["metadata"]["canonical_key"] = definition["canonical_key"]
                item["metadata"]["definition_alias"] = definition["alias"]
                item["metadata"]["definition_full_text"] = definition["full_text"]
                if definition.get("role"):
                    item["canonical_role"] = definition["role"]
                    item["metadata"]["canonical_role"] = definition["role"]
            enriched.append(item)
        return enriched

    def _build_focus_blocks(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        focus_config: Dict[str, Any],
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        selected_indices = set()
        total_lines = len(indexed_lines)

        head_lines = int(focus_config.get("head_lines", 0) or 0)
        tail_lines = int(focus_config.get("tail_lines", 0) or 0)
        radius = int(focus_config.get("radius", 0) or 0)
        max_matches = int(focus_config.get("max_matches", 8) or 8)
        tokens = [str(token) for token in focus_config.get("tokens", [])]

        for index in range(min(head_lines, total_lines)):
            selected_indices.add(index)

        for index in range(max(0, total_lines - tail_lines), total_lines):
            selected_indices.add(index)

        matched_indices: List[int] = []
        for item in indexed_lines:
            line_text = item["text"]
            if any(token in line_text for token in tokens):
                matched_indices.append(item["index"])
            if len(matched_indices) >= max_matches:
                break

        for index in matched_indices:
            start = max(0, index - radius)
            end = min(total_lines, index + radius + 1)
            for offset in range(start, end):
                selected_indices.add(offset)

        if not selected_indices:
            return []

        sorted_indices = sorted(selected_indices)
        blocks: List[Dict[str, Any]] = []
        current_indices: List[int] = []
        previous_index: int | None = None

        for index in sorted_indices:
            if previous_index is not None and index > previous_index + 1:
                if current_indices:
                    blocks.append(self._materialize_focus_block(text, indexed_lines, current_indices, max_chars))
                current_indices = [index]
            else:
                current_indices.append(index)
            previous_index = index

        if current_indices:
            blocks.append(self._materialize_focus_block(text, indexed_lines, current_indices, max_chars))

        return [block for block in blocks if block.get("text")]

    def _build_indexed_lines(self, text: str) -> List[Dict[str, Any]]:
        indexed_lines: List[Dict[str, Any]] = []
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            normalized = raw_line.strip()
            line_start = cursor
            line_end = cursor + len(raw_line)
            cursor = line_end
            if not normalized:
                continue
            indexed_lines.append(
                {
                    "index": len(indexed_lines),
                    "text": normalized,
                    "start": line_start,
                    "end": line_end,
                }
            )
        return indexed_lines

    def _materialize_focus_block(
        self,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        indices: List[int],
        max_chars: int,
    ) -> Dict[str, Any]:
        if not indices:
            return {"text": "", "start": 0}

        start = int(indexed_lines[indices[0]]["start"])
        end = int(indexed_lines[indices[-1]]["end"])
        snippet_text = text[start:end].strip()
        if len(snippet_text) > max_chars:
            snippet_text = snippet_text[:max_chars]
        return {"text": snippet_text, "start": start}

    def _score_to_confidence(self, score: int) -> str:
        if score >= 10:
            return "high"
        if score >= 6:
            return "medium"
        return "low"

    def _confidence_rank(self, confidence: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(confidence.lower(), 1)

    def _chunk_text(
        self,
        text: str,
        target_size: int = 1800,
        overlap: int = 180,
    ) -> List[Dict[str, int | str]]:
        if len(text) <= target_size:
            return [{"text": text, "start": 0}]

        chunks: List[Dict[str, int | str]] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + target_size, text_length)
            if end < text_length:
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + 400:
                    end = newline_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                actual_start = text.find(chunk_text, start, end)
                if actual_start == -1:
                    actual_start = start
                chunks.append({"text": chunk_text, "start": actual_start})

            if end >= text_length:
                break

            start = max(end - overlap, start + 1)

        return chunks

    def _is_review_27b_workflow(self, profile: Dict[str, Any]) -> bool:
        return str(profile.get("engine_strategy", "")).strip() == "review_27b"

    def _build_extraction_strategy_note(self, strategy_key: str) -> str:
        if strategy_key == "review_27b":
            return (
                "Because this is the dedicated 27B review workflow, prefer one accurate full-pass extraction "
                "followed by narrow gap checks around high-risk blocks, repeated mentions, abbreviation anchors, "
                "and role-bound subjects. Avoid broad duplicate recall and keep every recovery grounded in visible text."
            )
        return (
            "Because this is the local 4B stability strategy, prioritize verbatim recall within the provided text "
            "block, abbreviation-linked mentions, split addresses, cross-page repeated subjects, and residual "
            "entities hidden in dense narrative paragraphs. Keep every recovery grounded in visible evidence."
        )

    def _should_run_compact_review_27b_follow_up(
        self,
        *,
        preliminary_entities: List[Dict[str, Any]],
        definition_hints: List[Dict[str, Any]],
    ) -> bool:
        if not preliminary_entities:
            return True
        if not definition_hints:
            return False

        normalize = lambda value: re.sub(r"\s+", "", str(value or "")).strip()
        normalized_entities = {
            normalize(entity.get("text", ""))
            for entity in preliminary_entities
            if str(entity.get("text", "")).strip()
        }
        lightweight_aliases = {
            normalize(token)
            for token in (self.ORGANIZATION_LABEL_TOKENS + self.LEGAL_PARTY_LABEL_TOKENS)
        }

        for item in definition_hints:
            alias = normalize(item.get("alias", ""))
            full_text = normalize(item.get("full_text", ""))
            if full_text and full_text not in normalized_entities:
                return True
            if alias and alias not in lightweight_aliases:
                return True
        return False

    def _build_review_27b_snippets(
        self,
        *,
        text: str,
        indexed_lines: List[Dict[str, Any]],
        document_context: Dict[str, Any],
        preliminary_entities: List[Dict[str, Any]],
        definition_hints: List[Dict[str, Any]],
        profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        budget_tier = str(profile.get("runtime_budget_tier") or "").strip().lower()
        max_chars = max(0, int(profile.get("recall_snippet_size") or 0))
        max_blocks = max(0, int(profile.get("high_risk_block_limit") or 0))
        max_total_passes = max(0, int(profile.get("specialized_pass_limit") or 0))
        if max_chars <= 0 or max_total_passes <= 0:
            return []

        if budget_tier == "review_compact":
            if not self._should_run_compact_review_27b_follow_up(
                preliminary_entities=preliminary_entities,
                definition_hints=definition_hints,
            ):
                return []
            compact_text = text if len(text) <= max_chars else text[:max_chars]
            if not compact_text.strip():
                return []
            return [
                {
                    "name": "precision_gap_review",
                    "instruction": (
                        "Review this short document once for any missed verbatim organizations, persons, "
                        "locations, role-bound names, abbreviation-linked mentions, and footer/signature subjects. "
                        "Recover only genuinely missed items."
                    ),
                    "text": compact_text,
                    "start": 0,
                    "categories": ["definition", "relation", "organization", "person", "address", "footer"],
                }
            ]

        high_risk_blocks = self._schedule_high_risk_blocks(
            text,
            indexed_lines,
            max_chars=max_chars,
            max_blocks=max_blocks,
        )
        repetition_hotspots = self._build_repetition_hotspot_blocks(
            text,
            indexed_lines,
            preliminary_entities,
            max_chars=max_chars,
            max_blocks=max(1, min(4, max_blocks)),
        )
        if repetition_hotspots:
            high_risk_blocks = self._merge_ranked_blocks(
                high_risk_blocks,
                repetition_hotspots,
                max_blocks=max_blocks,
            )

        snippets = self._build_specialized_recall_snippets(
            high_risk_blocks,
            definitions=definition_hints,
            max_extra_passes=max(0, int(profile.get("complexity_extra_passes") or 0)),
            max_total_passes=max_total_passes,
        )
        if snippets:
            return snippets

        fallback_snippets = self._build_targeted_recall_snippets(
            text,
            document_context=document_context,
            max_snippets=1,
            max_chars=max_chars,
        )
        return fallback_snippets[:1]

    def _build_prompt(
        self,
        text: str,
        *,
        document_context: Dict[str, Any] | None = None,
        pass_name: str = "base",
        focus_instruction: str | None = None,
    ) -> str:
        strategy = get_llm_strategy_profile(self.model)
        document_context = document_context or {
            "document_type": "other",
            "label": self.DOCUMENT_TYPE_LABELS["other"],
            "prompt_hint": self.DOCUMENT_TYPE_PROMPT_HINTS["other"],
            "reason": "",
        }
        entity_lines = "\n".join(
            f"- {entity_type}: {description}"
            for entity_type, description in self.SUPPORTED_ENTITY_TYPES.items()
        )
        label_hints = ", ".join(
            [
                "甲方",
                "乙方",
                "丙方",
                "委托方",
                "受托方",
                "申请人",
                "被申请人",
                "申请执行人",
                "被执行人",
                "原告",
                "被告",
                "第三人",
                "合同编号",
                "工程名称",
                "项目名称",
                "工程地址",
                "项目地址",
                "住址",
                "身份证住址",
                "住所",
                "住所地",
                "通讯地址",
                "送达地址",
                "注册地址",
                "办公地址",
                "开户行",
                "户名",
                "法定代表人",
                "联系人",
            ]
        )
        strategy_note = self._build_extraction_strategy_note(strategy.key)
        pass_note = (
            "This is the primary full-text extraction pass."
            if pass_name == "base"
            else f"This is a targeted recall pass named {pass_name}. Use the focus instruction to recover easy-to-miss entities."
        )
        focus_note = focus_instruction or "Use general extraction judgment."
        document_note = (
            f"Inferred document type: {document_context.get('label', self.DOCUMENT_TYPE_LABELS['other'])}. "
            f"Primary focus: {document_context.get('prompt_hint', self.DOCUMENT_TYPE_PROMPT_HINTS['other'])}"
        )
        reason_note = str(document_context.get("reason", "")).strip()
        definition_hints = document_context.get("definition_hints") or []
        definition_note = "No explicit abbreviation definitions were pre-detected."
        if definition_hints:
            preview = [
                f"{item['full_text']} -> {item['alias']}"
                for item in definition_hints[:6]
                if item.get("full_text") and item.get("alias")
            ]
            if preview:
                definition_note = "Known abbreviation anchors in this document: " + "; ".join(preview)

        return f"""
You are an expert Chinese legal document entity recognizer.
The document may be a contract, application, pleading, enforcement paper, statement, account material, or other formal Chinese business/legal text.
Extract only entities that should usually be anonymized.

Entity types:
{entity_lines}

Document context:
- {document_note}
- Classification evidence: {reason_note or "No extra classification evidence was available."}
- Definition evidence: {definition_note}
- {pass_note}
- Focus instruction: {focus_note}

Rules:
1. Return only entities that appear verbatim in the provided text.
2. Prefer complete values that appear after labels such as {label_hints}.
3. For LOCATION, pay special attention to full addresses after labels such as 住址, 身份证住址, 住所, 住所地, 通讯地址, 送达地址, 注册地址, 办公地址, 工程地址, 项目地址.
4. If a line contains a person or organization and then an address label, extract both the subject and the full address value.
5. Also inspect signature areas, footer blocks, table-like rows, account sections, header blocks, and repeated mentions outside standard labels.
6. In dense narrative prose, explicitly recover short repeated organization aliases and person names bound to relationships or actions, especially near cues such as 股东, 付款至, 支付给, 转账, and 个人银行账户.
7. If one line contains multiple different subjects, return each subject separately instead of merging them into one long span.
8. If the same subject appears in both a full name and a shorter repeated form, prefer the more complete form when it appears verbatim, but still return the shorter form if it also appears verbatim and should be anonymized separately.
9. Do not return phone numbers, bank card numbers, ID cards, emails, or amounts here. Those are handled by rule-based recognizers.
10. Preserve full official court, bank branch, and institution names when they appear verbatim.
11. When a block contains terms such as 以下简称, 简称, 下称, 又称, 项目公司, 甲公司, or 乙公司, treat full names and shortened mentions as equally important if both appear verbatim.
12. Do not explain your answer.
13. If your model has a thinking mode, keep the reasoning hidden and still output only the final JSON object.
14. Output strict JSON in this exact shape:
{{"entities":[{{"type":"PROJECT","text":"原文中逐字出现的项目名称"}}]}}
15. {strategy_note}

Text:
\"\"\"
{text}
\"\"\"
""".strip()

    def generate_json(
        self,
        prompt: str,
        num_predict: int = 768,
        images: Optional[List[bytes]] = None,
    ) -> Dict[str, Any]:
        payload = self._build_generate_payload(
            prompt,
            num_predict=num_predict,
            images=images,
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned HTTP {response.status_code}")

        return response.json()

    def _build_generate_payload(
        self,
        prompt: str,
        *,
        num_predict: int,
        images: Optional[List[bytes]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "top_p": 0.8,
                "top_k": 30,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        if settings.is_review_capable_ollama_model(self.model):
            payload["think"] = False
        if images:
            payload["images"] = [
                base64.b64encode(image_bytes).decode("ascii")
                for image_bytes in images
                if image_bytes
            ]
        return payload

    async def generate_json_async(
        self,
        prompt: str,
        num_predict: int = 768,
        images: Optional[List[bytes]] = None,
    ) -> Dict[str, Any]:
        payload = self._build_generate_payload(
            prompt,
            num_predict=num_predict,
            images=images,
        )
        timeout = httpx.Timeout(self.timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API returned HTTP {response.status_code}")
        return response.json()

    def _call_ollama(self, prompt: str, num_predict: int = 768) -> Dict[str, Any]:
        return self.generate_json(prompt, num_predict)

    async def _call_ollama_async(self, prompt: str, num_predict: int = 768) -> Dict[str, Any]:
        return await self.generate_json_async(prompt, num_predict)

    def extract_document_text_from_image(
        self,
        image_bytes: bytes,
        *,
        page_number: int | None = None,
        total_pages: int | None = None,
    ) -> Dict[str, Any]:
        return self._run_async(
            self.extract_document_text_from_image_async(
                image_bytes,
                page_number=page_number,
                total_pages=total_pages,
            )
        )

    async def extract_document_text_from_image_async(
        self,
        image_bytes: bytes,
        *,
        page_number: int | None = None,
        total_pages: int | None = None,
    ) -> Dict[str, Any]:
        if not image_bytes:
            return {"text": "", "quality": "low", "warnings": ["empty_image_input"]}

        if not await self._check_connection_async():
            self.available = False
            raise RuntimeError("Ollama is unavailable for OCR.")

        review_ocr = settings.is_review_capable_ollama_model(self.model)
        prompt = self._build_document_ocr_prompt(
            page_number=page_number,
            total_pages=total_pages,
            review_ocr=review_ocr,
        )
        payload = await self.generate_json_async(
            prompt,
            num_predict=1200 if review_ocr else 2200,
            images=[image_bytes],
        )
        parsed = self._load_json_object(payload, required_keys={"text"})
        if not isinstance(parsed, dict):
            logger.warning("No valid OCR JSON payload found in Ollama response.")
            return {"text": "", "quality": "low", "warnings": ["invalid_ocr_payload"]}

        text = self._normalize_document_page_text(parsed.get("text", ""))
        blocks = self._normalize_document_blocks(parsed.get("blocks"))
        lines = self._normalize_document_lines(parsed.get("lines"), fallback_blocks=blocks)
        if not text and blocks:
            text = self._blocks_to_text(blocks)
        if not text and lines:
            text = self._lines_to_text(lines)
        warnings = parsed.get("warnings")
        if not isinstance(warnings, list):
            warnings = []

        return {
            "text": text,
            "quality": str(parsed.get("quality", "unknown")).strip().lower() or "unknown",
            "layout": str(parsed.get("layout", "plain_text")).strip().lower() or "plain_text",
            "blocks": blocks,
            "lines": lines,
            "warnings": [str(item).strip() for item in warnings if str(item).strip()],
        }

    async def _check_connection_async(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def _build_document_ocr_prompt(
        self,
        *,
        page_number: int | None,
        total_pages: int | None,
        review_ocr: bool = False,
    ) -> str:
        page_label = ""
        if page_number is not None and total_pages is not None:
            page_label = f" This is page {page_number} of {total_pages}."
        elif page_number is not None:
            page_label = f" This is page {page_number}."

        if review_ocr:
            return (
                "You are reconstructing a Chinese contract or legal document page from a PDF image with a "
                "dedicated 27B OCR workflow."
                f"{page_label}\n"
                "Return only strict JSON in this exact shape:\n"
                '{"text":"...", "quality":"high|medium|low", "layout":"plain_text|mixed", '
                '"lines":[{"text":"...", "bbox":[0,0,1000,1000]}], "warnings":["..."]}\n'
                "Rules:\n"
                "1. Transcribe visible text faithfully in reading order. Do not summarize, rewrite, or translate.\n"
                "2. Preserve names, addresses, dates, punctuation, numbers, and line breaks when readable.\n"
                "3. In lines, emit one entry per visible text line. Use tight page-relative integer bboxes from 0 to 1000.\n"
                "4. If the page is partly unreadable, keep readable text and use [UNREADABLE] only where necessary.\n"
                "5. Do not emit blocks, tables, or extra structure beyond text, quality, layout, lines, and warnings.\n"
                "6. Output JSON only."
            )

        return (
            "You are reconstructing a Chinese contract or legal document page from a PDF image."
            f"{page_label}\n"
            "Read the page carefully and return only strict JSON in this exact shape:\n"
            '{"text":"...", "quality":"high|medium|low", "layout":"plain_text|table_like|mixed", '
            '"blocks":[{"type":"title|paragraph|line|table|spacer","text":"...",'
            '"align":"left|center|right","indent":0,"rows":[["..."]],"blank_before":0}],'
            '"lines":[{"text":"...", "bbox":[0,0,1000,1000]}],"warnings":["..."]}\n'
            "Rules:\n"
            "1. Transcribe visible text faithfully. Do not summarize, rewrite, or translate.\n"
            "2. Preserve reading order from top to bottom, left to right.\n"
            "3. Keep clause numbers, punctuation, dates, amounts, names, contract numbers, and addresses exactly when readable.\n"
            "4. Preserve paragraph boundaries with newline characters.\n"
            "5. Use blocks to preserve layout: short centered headings should be title blocks; normal lines should be line or paragraph blocks; tables should be table blocks with rows.\n"
            "6. In lines, list every visible text line or table row in reading order. Use tab separators between cells when a row is table-like.\n"
            "7. Every line bbox must be a tight box around that line, using page-relative integers from 0 to 1000: [left, top, right, bottom].\n"
            "8. For blank lines or large visual gaps, you may emit spacer blocks with blank_before or type=spacer.\n"
            "9. If the page contains tables, keep each row and each cell separate in rows.\n"
            "10. If some characters are unreadable, use [UNREADABLE] in that position instead of guessing.\n"
            "11. Do not add content that is not visible in the image.\n"
            "12. Output JSON only."
        )

    def _normalize_document_page_text(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u3000", " ")
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _normalize_document_blocks(self, blocks: Any) -> List[Dict[str, Any]]:
        if not isinstance(blocks, list):
            return []

        normalized_blocks: List[Dict[str, Any]] = []
        for item in blocks:
            if not isinstance(item, dict):
                continue

            block_type = str(item.get("type", "line")).strip().lower()
            if block_type not in {"title", "paragraph", "line", "table", "spacer"}:
                block_type = "line"

            if block_type == "table":
                rows = item.get("rows")
                if not isinstance(rows, list):
                    continue
                normalized_rows: List[List[str]] = []
                for row in rows:
                    if not isinstance(row, list):
                        continue
                    normalized_row = [
                        self._normalize_document_page_text(str(cell))
                        for cell in row
                        if str(cell).strip()
                    ]
                    if normalized_row:
                        normalized_rows.append(normalized_row)
                if not normalized_rows:
                    continue
                normalized_blocks.append(
                    {
                        "type": "table",
                        "rows": normalized_rows,
                        "align": "left",
                    }
                )
                continue

            if block_type == "spacer":
                blank_before = int(item.get("blank_before", item.get("count", 1)) or 1)
                normalized_blocks.append(
                    {
                        "type": "spacer",
                        "blank_before": max(1, min(blank_before, 3)),
                    }
                )
                continue

            text = self._normalize_document_page_text(item.get("text", ""))
            if not text:
                continue

            align = str(item.get("align", "left")).strip().lower()
            if align not in {"left", "center", "right"}:
                align = "left"

            indent = item.get("indent", 0)
            try:
                indent_value = max(0, min(int(indent), 4))
            except (TypeError, ValueError):
                indent_value = 0

            blank_before = item.get("blank_before", 0)
            try:
                blank_before_value = max(0, min(int(blank_before), 3))
            except (TypeError, ValueError):
                blank_before_value = 0

            normalized_blocks.append(
                {
                    "type": block_type,
                    "text": text,
                    "align": align,
                    "indent": indent_value,
                    "blank_before": blank_before_value,
                    "bbox": self._normalize_relative_bbox(item.get("bbox")),
                }
            )

        return normalized_blocks

    def _normalize_document_lines(
        self,
        lines: Any,
        *,
        fallback_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_lines: List[Dict[str, Any]] = []
        if isinstance(lines, list):
            for item in lines:
                if not isinstance(item, dict):
                    continue
                text = self._normalize_document_page_text(item.get("text", ""))
                bbox = self._normalize_relative_bbox(item.get("bbox"))
                if not text or bbox is None:
                    continue
                normalized_lines.append({"text": text, "bbox": bbox})

        if normalized_lines:
            return normalized_lines

        fallback_lines: List[Dict[str, Any]] = []
        for block in fallback_blocks or []:
            if not isinstance(block, dict):
                continue
            text = self._normalize_document_page_text(block.get("text", ""))
            bbox = self._normalize_relative_bbox(block.get("bbox"))
            if not text or bbox is None:
                continue
            fallback_lines.append({"text": text, "bbox": bbox})
        return fallback_lines

    def _normalize_relative_bbox(self, bbox: Any) -> Optional[List[float]]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        try:
            numeric = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None

        max_value = max(abs(value) for value in numeric)
        if max_value > 1.5:
            scale = 1000.0 if max_value > 10 else 100.0
            numeric = [value / scale for value in numeric]

        left, top, right, bottom = [min(max(value, 0.0), 1.0) for value in numeric]
        if right <= left or bottom <= top:
            return None
        return [round(left, 5), round(top, 5), round(right, 5), round(bottom, 5)]

    def _blocks_to_text(self, blocks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for block in blocks:
            block_type = str(block.get("type", "line")).strip().lower()
            if block_type == "spacer":
                blank_before = max(1, int(block.get("blank_before", 1) or 1))
                parts.extend([""] * blank_before)
                continue
            if block_type == "table":
                rows = block.get("rows") or []
                for row in rows:
                    if isinstance(row, list):
                        parts.append("\t".join(str(cell).strip() for cell in row if str(cell).strip()))
                continue

            text = self._normalize_document_page_text(block.get("text", ""))
            if text:
                blank_before = max(0, int(block.get("blank_before", 0) or 0))
                if blank_before:
                    parts.extend([""] * blank_before)
                parts.append(text)

        return "\n".join(parts).strip()

    def _lines_to_text(self, lines: List[Dict[str, Any]]) -> str:
        return "\n".join(
            self._normalize_document_page_text(item.get("text", ""))
            for item in lines
            if self._normalize_document_page_text(item.get("text", ""))
        ).strip()

    def _get_model_profile(
        self,
        *,
        text_length: int = 0,
        line_count: int = 0,
    ) -> Dict[str, Any]:
        runtime = get_runtime_llm_strategy_profile(
            self.model,
            text_length=text_length,
            line_count=line_count,
        )
        strategy = runtime.strategy
        return {
            "engine_strategy": strategy.key,
            "chunk_target_size": strategy.chunk_target_size,
            "chunk_overlap": strategy.chunk_overlap,
            "extract_num_predict": strategy.extract_num_predict,
            "recall_num_predict": strategy.recall_num_predict,
            "classify_num_predict": strategy.classify_num_predict,
            "targeted_recall_passes": strategy.targeted_recall_passes,
            "recall_snippet_size": strategy.recall_snippet_size,
            "prefer_llm_first": strategy.prefer_llm_first,
            "high_risk_block_limit": strategy.high_risk_block_limit,
            "specialized_passes": strategy.specialized_passes,
            "specialized_pass_limit": strategy.specialized_pass_limit,
            "complexity_extra_passes": strategy.complexity_extra_passes,
            "enable_definition_recall": strategy.enable_definition_recall,
            "enable_residual_scan": strategy.enable_residual_scan,
            "runtime_budget_tier": runtime.budget_tier,
            "runtime_budget_label": runtime.budget_label,
        }

    def _parse_response(
        self,
        *,
        payload: Dict[str, Any],
        chunk_text: str,
        chunk_start: int,
    ) -> List[Dict]:
        parsed_entities = self._load_entities(payload)
        results: List[Dict] = []

        for item in parsed_entities:
            entity_type = str(item.get("type", "")).strip().upper()
            entity_text = self._sanitize_llm_entity_text(
                entity_type,
                str(item.get("text", "")).strip(),
            )

            if entity_type not in self.SUPPORTED_ENTITY_TYPES or not entity_text:
                continue
            if not self._is_valid_entity(entity_type, entity_text):
                continue

            for start, end, matched_text in self._find_entity_spans(chunk_text, entity_text):
                results.append(
                    {
                        "type": entity_type,
                        "text": matched_text,
                        "start": chunk_start + start,
                        "end": chunk_start + end,
                        "score": 0.9,
                        "source": "ollama",
                    }
                )

        return results

    def _is_valid_entity(self, entity_type: str, entity_text: str) -> bool:
        normalized_text = re.sub(r"\s+", "", entity_text)
        if self._is_non_entity_heading_text(normalized_text):
            return False

        if entity_type == "PROJECT":
            if normalized_text.endswith("合同"):
                return False
            if any(token in entity_text for token in ["项目", "工程", "标段"]):
                return True
            if re.search(r"\d+(?:\.\d+)?MW", entity_text, re.I):
                return True
            if "风电场" in entity_text:
                return True
            return False

        if entity_type == "BANK_NAME":
            return "银行" in entity_text

        if entity_type == "ACCOUNT_NAME":
            return not re.fullmatch(r"\d{6,}", normalized_text)

        if entity_type == "ORGANIZATION":
            if any(punctuation in entity_text for punctuation in [",", "，", "。", ";", "；"]):
                return False
            return self._is_valid_organization_candidate_text(entity_text)

        if entity_type == "CONTRACT_NO":
            return len(normalized_text) >= 4

        return True

    def _load_entities(self, payload: Dict[str, Any]) -> List[Dict]:
        parsed = self._load_json_object(payload, required_keys={"entities"}, allow_list=True)
        if isinstance(parsed, dict):
            entities = parsed.get("entities")
            if isinstance(entities, list):
                return [item for item in entities if isinstance(item, dict)]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

        logger.warning("No valid JSON payload found in Ollama response.")
        return []

    def _sanitize_llm_entity_text(self, entity_type: str, entity_text: str) -> str:
        cleaned = str(entity_text or "").strip().strip("“”\"'`")
        if not cleaned:
            return ""

        previous = ""
        while cleaned and cleaned != previous:
            previous = cleaned
            match = re.match(r"^(?P<label>[^:：\n]{1,40})\s*[:：]\s*(?P<value>.+)$", cleaned)
            if not match:
                break

            label = match.group("label").strip()
            if not self._looks_like_entity_label_prefix(label, entity_type):
                break

            cleaned_value = self._clean_labeled_value(match.group("value"), label, entity_type)
            cleaned = cleaned_value or match.group("value").strip()

        if entity_type == "PERSON":
            person_match = re.match(r"(?P<value>[\u4e00-\u9fa5·某]{2,8})", cleaned)
            if person_match:
                cleaned = person_match.group("value")
        elif entity_type == "LOCATION":
            cleaned = self._clean_labeled_value(cleaned, "", entity_type) or cleaned

        return cleaned.strip(" ，,。；;：:、“”‘’（）()[]【】")

    def _looks_like_entity_label_prefix(self, label: str, entity_type: str) -> bool:
        normalized = re.sub(r"\s+", "", str(label or ""))
        if not normalized:
            return False

        if any(token in normalized for token in self.LEGAL_PARTY_LABEL_TOKENS):
            return entity_type in {"PERSON", "ORGANIZATION"}

        if entity_type == "PERSON":
            return any(token in normalized for token in self.PERSON_LABEL_TOKENS)

        if entity_type == "ORGANIZATION":
            return any(token in normalized for token in self.ORGANIZATION_LABEL_TOKENS)

        if entity_type == "LOCATION":
            if any(token in normalized for token in self.ADDRESS_LABEL_TOKENS):
                return True
            return any(token in normalized for token in ["地址", "住址", "住所", "住所地"])

        if entity_type == "PROJECT":
            return any(token in normalized for token in self.PROJECT_LABEL_TOKENS)

        if entity_type == "BANK_NAME":
            return any(token in normalized for token in self.BANK_LABEL_TOKENS)

        if entity_type == "ACCOUNT_NAME":
            return any(token in normalized for token in self.ACCOUNT_LABEL_TOKENS)

        return False

    def _load_json_object(
        self,
        payload: Any,
        *,
        required_keys: set[str] | None = None,
        allow_list: bool = False,
    ) -> Any:
        for candidate in self._extract_candidates(payload):
            parsed = self._coerce_candidate(candidate)
            if isinstance(parsed, dict):
                if required_keys and not required_keys.issubset(parsed.keys()):
                    continue
                return parsed
            if allow_list and isinstance(parsed, list):
                return parsed
        return None

    def _extract_candidates(self, payload: Any) -> Iterable[Any]:
        yield payload

        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, dict):
                yield message
                for key in ("content", "thinking"):
                    value = message.get(key)
                    if value:
                        yield value

            for key in ("response", "thinking", "output", "text"):
                value = payload.get(key)
                if value:
                    yield value

    def _coerce_candidate(self, candidate: Any) -> Any:
        if isinstance(candidate, (dict, list)):
            return candidate

        if not isinstance(candidate, str):
            return None

        for snippet in self._extract_json_snippets(candidate):
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

        return None

    def _extract_json_snippets(self, text: str) -> Iterable[str]:
        cleaned = text.strip()
        if cleaned:
            yield cleaned

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fence_match:
            yield fence_match.group(1).strip()

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            yield text[first_brace : last_brace + 1]

        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            yield text[first_bracket : last_bracket + 1]

    def _find_entity_spans(self, text: str, entity_text: str) -> List[tuple[int, int, str]]:
        spans = self._find_exact_spans(text, entity_text)
        if spans:
            return spans

        normalized_entity = re.sub(r"\s+", "", entity_text)
        if not normalized_entity:
            return []

        normalized_chars: List[str] = []
        index_map: List[int] = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            normalized_chars.append(char)
            index_map.append(index)

        normalized_text = "".join(normalized_chars)
        results: List[tuple[int, int, str]] = []
        search_from = 0

        while True:
            normalized_start = normalized_text.find(normalized_entity, search_from)
            if normalized_start == -1:
                break

            normalized_end = normalized_start + len(normalized_entity) - 1
            original_start = index_map[normalized_start]
            original_end = index_map[normalized_end] + 1
            results.append((original_start, original_end, text[original_start:original_end]))
            search_from = normalized_start + len(normalized_entity)

        return results

    def _find_exact_spans(self, text: str, entity_text: str) -> List[tuple[int, int, str]]:
        results: List[tuple[int, int, str]] = []
        search_from = 0

        while entity_text:
            start = text.find(entity_text, search_from)
            if start == -1:
                break
            end = start + len(entity_text)
            results.append((start, end, text[start:end]))
            search_from = end

        return results

    def _refine_entities(
        self,
        text: str,
        entities: List[Dict],
        *,
        document_context: Dict[str, Any],
    ) -> List[Dict]:
        refined_entities: List[Dict] = []
        for entity in entities:
            item = dict(entity)
            refined_type = self._infer_refined_entity_type(
                text,
                item,
                document_context=document_context,
            )
            if refined_type != item["type"] and self._is_valid_entity(refined_type, item["text"]):
                item["type"] = refined_type
                item["score"] = max(float(item.get("score", 0.9)), 0.92)
                item["source"] = f"{item.get('source', 'ollama')}_refined"
            refined_entities.append(item)
        return refined_entities

    def _infer_refined_entity_type(
        self,
        text: str,
        entity: Dict[str, Any],
        *,
        document_context: Dict[str, Any],
    ) -> str:
        entity_type = str(entity.get("type", "")).strip().upper()
        entity_text = str(entity.get("text", "")).strip()
        if not entity_type or not entity_text:
            return entity_type

        line_context = self._build_line_context(text, int(entity["start"]), int(entity["end"]))
        label_hint = self._infer_label_hint(line_context["prefix"])

        if label_hint == "BANK_NAME" and self._looks_like_bank_name(entity_text):
            return "BANK_NAME"
        if label_hint == "ACCOUNT_NAME":
            return "ACCOUNT_NAME"
        if label_hint == "PROJECT" and self._looks_like_project_name(entity_text):
            return "PROJECT"
        if label_hint == "LOCATION" and self._looks_like_location_text(entity_text):
            return "LOCATION"
        if label_hint == "ORGANIZATION" and self._looks_like_organization_text(entity_text):
            return "ORGANIZATION"

        if self._looks_like_bank_name(entity_text):
            return "BANK_NAME"
        if self._looks_like_institution_name(entity_text) or self._has_strong_organization_token(entity_text):
            return "ORGANIZATION"
        if entity_type == "ORGANIZATION" and self._looks_like_location_text(entity_text) and not self._has_strong_organization_token(entity_text):
            return "LOCATION"
        if entity_type == "LOCATION" and self._looks_like_organization_text(entity_text):
            if document_context.get("document_type") in {"legal_application", "enforcement_paper"}:
                return "ORGANIZATION"
        if entity_type in {"PERSON", "ORGANIZATION"} and label_hint == "ACCOUNT_NAME":
            return "ACCOUNT_NAME"
        if entity_type == "ORGANIZATION" and self._looks_like_project_name(entity_text):
            return "PROJECT"
        return entity_type

    def _recover_labeled_entities(
        self,
        text: str,
        existing_entities: List[Dict],
        *,
        document_context: Dict[str, Any],
    ) -> List[Dict]:
        recovered: List[Dict] = []
        occupied = {
            (int(entity["start"]), int(entity["end"]), str(entity["text"]).strip())
            for entity in existing_entities
        }
        occupied_ranges = [
            (int(entity["start"]), int(entity["end"]))
            for entity in existing_entities
        ]
        indexed_lines = self._build_indexed_lines(text)
        label_specs = [
            ("BANK_NAME", self.BANK_LABEL_TOKENS),
            ("ACCOUNT_NAME", self.ACCOUNT_LABEL_TOKENS),
            ("PROJECT", self.PROJECT_LABEL_TOKENS),
            ("LOCATION", self.ADDRESS_LABEL_TOKENS),
            ("ORGANIZATION", self.ORGANIZATION_LABEL_TOKENS),
            ("CONTRACT_NO", list(ALL_IDENTIFIER_LABELS)),
            ("PERSON", self.PERSON_LABEL_TOKENS + self.LEGAL_PARTY_LABEL_TOKENS),
        ]

        for line_index, line in enumerate(indexed_lines):
            line_text = str(line["text"])

            for entity_type, labels in label_specs:
                label = self._select_line_label(line_text, labels)
                if not label:
                    continue

                candidates = self._extract_labeled_candidates(indexed_lines, line_index, label, entity_type)
                search_end_index = min(
                    len(indexed_lines) - 1,
                    line_index + (3 if entity_type == "LOCATION" else 2),
                )

                for value in candidates:
                    if not self._is_valid_entity(entity_type, value):
                        continue
                    if entity_type == "PERSON" and not self._looks_like_person_name(value):
                        continue
                    if entity_type == "ORGANIZATION" and not self._looks_like_organization_text(value):
                        continue
                    if entity_type == "LOCATION" and not self._looks_like_location_text(value):
                        continue
                    if entity_type == "PROJECT" and not self._looks_like_project_name(value):
                        continue
                    if entity_type == "CONTRACT_NO" and not self._looks_like_contract_no(value):
                        continue

                    spans = self._find_entity_spans(text, value)
                    for start, end, matched_text in spans:
                        if start < int(line["start"]):
                            continue
                        if end > int(indexed_lines[search_end_index]["end"]):
                            continue
                        if self._record_recovered_entity(
                            recovered,
                            occupied,
                            occupied_ranges,
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            matched_text=matched_text,
                            score=0.93,
                            source="ollama_structured",
                        ):
                            break

        return self._refine_entities(
            text,
            recovered,
            document_context=document_context,
        )

    def _select_line_label(self, line_text: str, labels: List[str]) -> str:
        stripped = line_text.strip()
        if "：" in stripped or ":" in stripped:
            prefix = re.split(r"[:：]", stripped, maxsplit=1)[0].strip()
            if label_matches(prefix, labels):
                return prefix

        normalized = compact_text(stripped)
        return next(
            (
                item
                for item in sorted(labels, key=len, reverse=True)
                if normalized.startswith(compact_text(item))
            ),
            "",
        )

    def _recover_prose_entities(
        self,
        text: str,
        existing_entities: List[Dict],
        *,
        document_context: Dict[str, Any],
    ) -> List[Dict]:
        recovered: List[Dict] = []
        occupied = {
            (int(entity["start"]), int(entity["end"]), str(entity["text"]).strip())
            for entity in existing_entities
        }
        occupied_ranges = [
            (int(entity["start"]), int(entity["end"]))
            for entity in existing_entities
        ]

        organization_matches = sorted(
            self._extract_prose_organization_matches(text),
            key=lambda item: (-(item[1] - item[0]), item[0]),
        )
        for start, end, matched_text in organization_matches:
            self._record_recovered_entity(
                recovered,
                occupied,
                occupied_ranges,
                entity_type="ORGANIZATION",
                start=start,
                end=end,
                matched_text=matched_text,
                score=0.87,
                source="ollama_prose",
            )

        person_matches = sorted(
            self._extract_prose_person_matches(text),
            key=lambda item: (item[0], item[1]),
        )
        for start, end, matched_text in person_matches:
            self._record_recovered_entity(
                recovered,
                occupied,
                occupied_ranges,
                entity_type="PERSON",
                start=start,
                end=end,
                matched_text=matched_text,
                score=0.86,
                source="ollama_prose",
            )

        return self._refine_entities(
            text,
            recovered,
            document_context=document_context,
        )

    def _record_recovered_entity(
        self,
        recovered: List[Dict[str, Any]],
        occupied: set[tuple[int, int, str]],
        occupied_ranges: List[tuple[int, int]],
        *,
        entity_type: str,
        start: int,
        end: int,
        matched_text: str,
        score: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        key = (start, end, matched_text)
        if key in occupied:
            return False
        if any(start < existing_end and end > existing_start for existing_start, existing_end in occupied_ranges):
            return False

        recovered.append(
            {
                "type": entity_type,
                "text": matched_text,
                "start": start,
                "end": end,
                "score": score,
                "source": source,
            }
        )
        if metadata:
            recovered[-1]["metadata"] = dict(metadata)
        occupied.add(key)
        occupied_ranges.append((start, end))
        return True

    def _extract_prose_organization_matches(self, text: str) -> List[tuple[int, int, str]]:
        pattern = re.compile(
            r"[\u4e00-\u9fa5A-Za-z0-9]{2,40}?"
            r"(?:股份有限公司|有限责任公司|有限公司|集团有限公司|集团|公司|研究院|研究所|事务所|服务中心|中心|银行|支行|分行)"
        )
        matches: List[tuple[int, int, str]] = []
        seen = set()

        for raw_match in pattern.finditer(text):
            candidate = self._clean_prose_organization_candidate(raw_match.group(0))
            if not candidate or not self._is_valid_prose_organization_candidate(candidate):
                continue

            for start, end, matched_text in self._find_entity_spans(text, candidate):
                key = (start, end, matched_text)
                if key in seen:
                    continue
                matches.append(key)
                seen.add(key)

        return matches

    def _extract_prose_person_matches(self, text: str) -> List[tuple[int, int, str]]:
        patterns = [
            re.compile(
                r"(?:股东|法定代表人|法人代表|实际控制人|联系人|经办人|委托代理人|代理人|收款人|付款人|签署人|法代)"
                r"\s*[：:为系是]?\s*(?P<value>[\u4e00-\u9fa5·某]{2,5}(?:[、和及与][\u4e00-\u9fa5·某]{2,5})+)"
                r"(?=[，,；;。]|$|\s|的|将|并|向|与|于|负责|签署|签字|收款|办理|确认|提供|主张|再次|指示|要求|表示)"
            ),
            re.compile(
                r"(?:股东|法定代表人|法人代表|实际控制人|联系人|经办人|委托代理人|代理人|收款人|付款人|签署人|法代)"
                r"\s*[：:为系是]?\s*(?P<value>[\u4e00-\u9fa5·某]{2,5})"
                r"(?=[，,；;。]|$|\s|的|将|并|向|与|于|负责|签署|签字|收款|办理|确认|提供|主张|再次|指示|要求|表示)"
            ),
            re.compile(
                r"(?:付款至|支付给|转给|汇给|汇至|转至|打入|收款至|交由)"
                r"\s*(?P<value>[\u4e00-\u9fa5·某]{2,5}(?:[、和及与][\u4e00-\u9fa5·某]{2,5})+)"
                r"(?=(?:个人)?(?:银行账户|个人账户|本人账户|账户)|[，,；;。]|$|\s|的|将|并|向|与|于|负责|签署|签字|收款|办理|确认|提供|主张|再次|指示|要求|表示)"
            ),
            re.compile(
                r"(?:付款至|支付给|转给|汇给|汇至|转至|打入|收款至|交由)"
                r"\s*(?P<value>[\u4e00-\u9fa5·某]{2,5})"
                r"(?=(?:个人)?(?:银行账户|个人账户|本人账户|账户)|[，,；;。]|$|\s|的|将|并|向|与|于|负责|签署|签字|收款|办理|确认|提供|主张|再次|指示|要求|表示)"
            ),
            re.compile(r"(?P<value>[\u4e00-\u9fa5·某]{2,5})(?:的)?(?:个人银行账户|个人账户|本人账户|银行账户)"),
            re.compile(
                r"(?:由|系由|并由)\s*(?P<value>[\u4e00-\u9fa5·某]{2,5})"
                r"(?=签署|签字|收取|收款|办理|经手|确认)"
            ),
        ]
        matches: List[tuple[int, int, str]] = []
        seen = set()

        for pattern in patterns:
            for match in pattern.finditer(text):
                raw_value = match.group("value")
                for candidate in self._split_person_candidate_group(raw_value):
                    candidate = self._clean_prose_person_candidate(candidate)
                    if not self._is_valid_prose_person_candidate(candidate):
                        continue
                    for start, end, matched_text in self._find_entity_spans(text, candidate):
                        key = (start, end, matched_text)
                        if key in seen:
                            continue
                        matches.append(key)
                        seen.add(key)

        return matches

    def _split_person_candidate_group(self, value: str) -> List[str]:
        if not re.search(r"[、和及与]", value):
            return [value]
        return [item for item in re.split(r"[、和及与]", value) if item]

    def _clean_prose_person_candidate(self, text: str) -> str:
        cleaned = text.strip(" ，,。；;：:、“”‘’（）()[]【】")
        cleaned = re.sub(r"^(?:支付给|付款至|转给|汇给|汇至|转至|打入|收款至|交由|付给|给|向|由|与|及|和)+", "", cleaned)
        stop_tokens = [
            "共同",
            "一同",
            "一起",
            "分别",
            "再次",
            "先后",
            "后续",
            "指示",
            "要求",
            "表示",
            "负责",
            "签署",
            "签字",
            "收款",
            "办理",
            "确认",
            "提供",
            "主张",
        ]
        cut_index = len(cleaned)
        for token in stop_tokens:
            position = cleaned.find(token)
            if position >= 2:
                cut_index = min(cut_index, position)
        cleaned = cleaned[:cut_index]
        cleaned = re.sub(r"(?:个人|本人)$", "", cleaned)
        cleaned = re.sub(r"(?:指示|要求|表示|负责|签署|签字|收款|办理|确认|提供|主张|再次)$", "", cleaned)
        return cleaned.strip(" ，,。；;：:、“”‘’（）()[]【】")

    def _clean_prose_organization_candidate(self, text: str) -> str:
        cleaned = text.strip(" ，,。；;：:、“”‘’（）()[]【】")
        previous = ""
        while cleaned and cleaned != previous:
            previous = cleaned
            cleaned = re.sub(
                r"^(?:后|再|又|并|仍|另|其|向其|向|将|把|与|由|被|为|同|对|按|经由|案涉|其中|以下简称|简称|即|系由|系|属|并由|后由|后又向|再次向|在|详见|见|已经|再次)+",
                "",
                cleaned,
            )
            cleaned = re.sub(
                r"^(?:上诉人|被上诉人|原审原告|原审被告(?:[一二三四五六七八九十]|\d+)?|原告|被告|申请人|被申请人|申请执行人|被执行人|第三人|委托方|受托方|发包人|承包人|供货方|采购方|收款单位|付款单位|股东|法定代表人|法人代表|实际控制人)+",
                "",
                cleaned,
            )

        tail_match = re.search(
            r"([\u4e00-\u9fa5A-Za-z0-9]{2,30}"
            r"(?:股份有限公司|有限责任公司|有限公司|集团有限公司|集团|公司|研究院|研究所|事务所|服务中心|中心|银行|支行|分行))$",
            cleaned,
        )
        if tail_match:
            cleaned = tail_match.group(1)
        return cleaned.strip(" ，,。；;：:、“”‘’（）()[]【】")

    def _strip_organization_suffix(self, text: str) -> str:
        return re.sub(
            r"(股份有限公司|有限责任公司|有限公司|集团有限公司|集团|研究院|研究所|事务所|服务中心|中心|银行|支行|分行|子公司|分公司|公司)$",
            "",
            text,
        )

    def _looks_like_identifier_text(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if not normalized:
            return False
        if any(token in normalized for token in self.IDENTIFIER_LABEL_TOKENS):
            return True
        if re.fullmatch(r"[0-9A-Z]{2,}", normalized):
            return True
        if re.fullmatch(r"\d{15,19}[0-9Xx]?", normalized):
            return True
        if re.fullmatch(r"[0-9A-HJ-NPQRTUWXY]{18}", normalized):
            return True
        if re.search(r"[0-9A-Z]{4,}", normalized) and re.search(r"\d", normalized) and not re.search(r"[\u4e00-\u9fa5]", normalized):
            return True
        return False

    def _is_valid_organization_candidate_text(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if len(normalized) < 4:
            return False
        if self._looks_like_identifier_text(normalized):
            return False
        if normalized in self.ORGANIZATION_GENERIC_TERMS:
            return False
        if any(normalized.endswith(token) for token in ("法院", "检察院", "公安局", "仲裁委员会")) and len(normalized) < 4:
            return False
        if any(normalized.startswith(prefix) and len(normalized) > len(prefix) + 1 for prefix in self.ORGANIZATION_STOP_PREFIXES):
            return False
        if any(fragment in normalized for fragment in self.ORGANIZATION_NOISE_FRAGMENTS):
            return False

        if self._looks_like_institution_name(normalized):
            return len(normalized) >= 4 and normalized not in self.ORGANIZATION_GENERIC_TERMS

        if not self._has_strong_organization_token(normalized):
            return False

        core = self._strip_organization_suffix(normalized)
        if len(core) < 2:
            return False
        if core in {"本", "该", "我", "贵", "相关", "上述", "涉案", "某"}:
            return False
        if any(fragment in core for fragment in self.ORGANIZATION_NOISE_FRAGMENTS):
            return False
        if any(core.startswith(prefix) and len(core) > len(prefix) for prefix in self.ORGANIZATION_STOP_PREFIXES):
            return False
        return True

    def _is_valid_prose_organization_candidate(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if normalized in {"本公司", "该公司", "我公司", "贵公司", "相关公司", "上述公司", "涉案公司", "某公司"}:
            return False
        if any(token in normalized for token in ["个人银行", "个人账户", "本人账户", "收款", "付款", "账户"]):
            return False
        return self._is_valid_organization_candidate_text(normalized)

    def _is_valid_prose_person_candidate(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if re.fullmatch(r"[\u4e00-\u9fa5·某]{2,5}", normalized) is None:
            return False
        if any(token in normalized for token in ["公司", "集团", "银行", "法院", "检察院", "公安局", "中心", "项目", "工程", "账户"]):
            return False
        return True

    def _extract_labeled_candidates(
        self,
        indexed_lines: List[Dict[str, Any]],
        line_index: int,
        label: str,
        entity_type: str,
    ) -> List[str]:
        line_text = str(indexed_lines[line_index]["text"])
        candidates: List[str] = []

        inline_value = self._extract_labeled_value(line_text, label, entity_type)
        if inline_value:
            candidates.append(inline_value)

        if entity_type == "LOCATION" and inline_value:
            extended_value = self._extend_location_candidate(indexed_lines, line_index, inline_value)
            if extended_value and extended_value not in candidates:
                candidates.append(extended_value)

        if self._looks_like_label_only_line(line_text, label):
            following_value = self._extract_following_labeled_value(indexed_lines, line_index, entity_type)
            if following_value and following_value not in candidates:
                candidates.append(following_value)

        return candidates

    def _build_line_context(self, text: str, start: int, end: int) -> Dict[str, str]:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)

        line_text = text[line_start:line_end]
        prefix = line_text[: max(0, start - line_start)]
        suffix = line_text[max(0, end - line_start) :]
        return {
            "line": line_text,
            "prefix": prefix,
            "suffix": suffix,
        }

    def _infer_label_hint(self, prefix: str) -> str:
        normalized_prefix = prefix[-40:]
        if any(token in normalized_prefix for token in self.BANK_LABEL_TOKENS):
            return "BANK_NAME"
        if any(token in normalized_prefix for token in self.ACCOUNT_LABEL_TOKENS):
            return "ACCOUNT_NAME"
        if any(token in normalized_prefix for token in self.ADDRESS_LABEL_TOKENS):
            return "LOCATION"
        if any(token in normalized_prefix for token in self.PROJECT_LABEL_TOKENS):
            return "PROJECT"
        if any(token in normalized_prefix for token in self.ORGANIZATION_LABEL_TOKENS):
            return "ORGANIZATION"
        return ""

    def _extract_labeled_value(self, line_text: str, label: str, entity_type: str) -> str:
        pattern = rf"{re.escape(label)}(?:[（(][^）)]{{0,20}}[）)])?\s*[：:]\s*(?P<value>.+)"
        match = re.search(pattern, line_text)
        if not match:
            return ""

        value = match.group("value").strip()
        stop_tokens = ["；", ";", "。", "\n"]
        if entity_type in {"PERSON", "ORGANIZATION", "ACCOUNT_NAME", "BANK_NAME", "PROJECT", "CONTRACT_NO"}:
            stop_tokens.extend(["，", ",", "（", "("])
        if entity_type == "LOCATION":
            stop_tokens.extend(["联系电话", "统一社会信用代码", "公民身份证", "身份证号码", "公民身份号码"])
        if entity_type in {"ORGANIZATION", "PERSON", "ACCOUNT_NAME"}:
            stop_tokens.extend(
                [token for token in self.ORGANIZATION_LABEL_TOKENS if token != label]
                + self.BANK_LABEL_TOKENS
                + self.ACCOUNT_LABEL_TOKENS
                + self.PERSON_LABEL_TOKENS
                + ["联系电话"]
            )

        cut_index = len(value)
        for token in stop_tokens:
            position = value.find(token)
            if position != -1:
                cut_index = min(cut_index, position)

        cleaned = value[:cut_index].strip(" ：:，,。；;")
        if "：" in cleaned or ":" in cleaned:
            cleaned = re.split(r"[：:]", cleaned, maxsplit=1)[0].strip()
        if not cleaned:
            return ""
        if cleaned == label:
            return ""
        if entity_type == "PERSON" and any(token in cleaned for token in self.PERSON_LABEL_TOKENS + ["联系电话"]):
            return ""
        return cleaned

    def _extract_labeled_value(self, line_text: str, label: str, entity_type: str) -> str:
        patterns = [
            rf"{re.escape(label)}(?:[（(][^）)]{{0,20}}[）)])?\s*[:：]\s*(?P<value>.+)",
            rf"{re.escape(label)}(?:[（(][^）)]{{0,20}}[）)])?\s{{2,}}(?P<value>.+)",
            rf"^{re.escape(label)}(?:[（(][^）)]{{0,20}}[）)])?\s+(?P<value>.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, line_text)
            if not match:
                continue
            cleaned = self._clean_labeled_value(match.group("value"), label, entity_type)
            if cleaned:
                return cleaned
        return ""

    def _clean_labeled_value(self, value: str, label: str, entity_type: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            return ""

        stop_tokens = ["；", ";", "。", "\n"]
        if entity_type in {"PERSON", "ORGANIZATION", "ACCOUNT_NAME", "BANK_NAME", "PROJECT", "CONTRACT_NO"}:
            stop_tokens.extend(["，", ",", "（", "("])
        if entity_type == "LOCATION":
            stop_tokens.extend(["联系电话", "统一社会信用代码", "公民身份号码", "身份证号码", "身份证号"])
        if entity_type in {"ORGANIZATION", "PERSON", "ACCOUNT_NAME"}:
            stop_tokens.extend(
                [token for token in self.ORGANIZATION_LABEL_TOKENS if token != label]
                + self.BANK_LABEL_TOKENS
                + self.ACCOUNT_LABEL_TOKENS
                + self.PERSON_LABEL_TOKENS
                + ["联系电话"]
            )

        cut_index = len(cleaned_value)
        for token in stop_tokens:
            position = cleaned_value.find(token)
            if position != -1:
                cut_index = min(cut_index, position)

        cleaned = cleaned_value[:cut_index].strip(" ：:，,。；;()（）")
        if not cleaned or cleaned == label:
            return ""
        if self._is_non_entity_heading_text(cleaned):
            return ""
        if entity_type == "PERSON" and any(token in cleaned for token in self.PERSON_LABEL_TOKENS + ["联系电话"]):
            return ""
        return cleaned

    def _looks_like_label_only_line(self, line_text: str, label: str) -> bool:
        remainder = line_text.replace(label, "", 1)
        remainder = re.sub(r"[（(][^）)]{0,20}[）)]", "", remainder, count=1)
        normalized = re.sub(r"[\s:：\-—_()（）]", "", remainder)
        return not normalized

    def _extract_following_labeled_value(
        self,
        indexed_lines: List[Dict[str, Any]],
        line_index: int,
        entity_type: str,
    ) -> str:
        parts: List[str] = []
        max_follow_lines = 3 if entity_type == "LOCATION" else 1

        for offset in range(1, max_follow_lines + 1):
            next_index = line_index + offset
            if next_index >= len(indexed_lines):
                break

            next_line = str(indexed_lines[next_index]["text"]).strip()
            if not next_line:
                continue
            if self._is_probable_new_label_line(next_line):
                break

            parts.append(next_line)
            if entity_type != "LOCATION":
                break
            if not self._should_continue_location(next_line):
                break

        if not parts:
            return ""

        candidate = "".join(parts)
        return self._clean_labeled_value(candidate, "", entity_type)

    def _extend_location_candidate(
        self,
        indexed_lines: List[Dict[str, Any]],
        line_index: int,
        current_value: str,
    ) -> str:
        parts = [current_value]

        for offset in range(1, 3):
            next_index = line_index + offset
            if next_index >= len(indexed_lines):
                break

            next_line = str(indexed_lines[next_index]["text"]).strip()
            if not next_line or self._is_probable_new_label_line(next_line):
                break
            if not self._should_continue_location(next_line):
                break

            parts.append(next_line)

        candidate = "".join(parts)
        return self._clean_labeled_value(candidate, "", "LOCATION") or current_value

    def _is_probable_new_label_line(self, line_text: str) -> bool:
        stripped = line_text.strip().strip("：:")
        if self._is_non_entity_heading_text(stripped):
            return True
        label_tokens = (
            self.BANK_LABEL_TOKENS
            + self.ACCOUNT_LABEL_TOKENS
            + self.PROJECT_LABEL_TOKENS
            + self.ORGANIZATION_LABEL_TOKENS
            + self.ADDRESS_LABEL_TOKENS
            + self.PERSON_LABEL_TOKENS
            + ["合同编号", "合同号", "案号"]
        )
        return any(token in line_text for token in label_tokens) and ("：" in line_text or ":" in line_text)

    def _should_continue_location(self, line_text: str) -> bool:
        if not line_text or len(line_text) > 120:
            return False
        if "：" in line_text or ":" in line_text:
            return False
        if self._looks_like_location_text(line_text):
            return True
        return sum(token in line_text for token in self.LOCATION_TOKENS) >= 1

    def _looks_like_bank_name(self, text: str) -> bool:
        return any(token in text for token in self.BANK_TOKENS)

    def _looks_like_institution_name(self, text: str) -> bool:
        return any(token in text for token in self.INSTITUTION_TOKENS)

    def _looks_like_organization_text(self, text: str) -> bool:
        return self._is_valid_organization_candidate_text(text)

    def _has_strong_organization_token(self, text: str) -> bool:
        return any(token in text for token in self.ORGANIZATION_TOKENS)

    def _looks_like_location_text(self, text: str) -> bool:
        if self._looks_like_bank_name(text):
            return False
        if any(token in text for token in ["法院", "检察院", "公安局", "仲裁委员会", "仲裁院"]):
            return True
        token_hits = sum(1 for token in self.LOCATION_TOKENS if token in text)
        if token_hits >= 2:
            return True
        return re.search(r"\d+号", text) is not None

    def _looks_like_project_name(self, text: str) -> bool:
        return any(token in text for token in ["项目", "工程", "标段", "采购"]) or re.search(r"\d+(?:\.\d+)?MW", text, re.I) is not None

    def _looks_like_contract_no(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        if len(normalized) < 4:
            return False
        if looks_like_case_number(normalized):
            return True
        return bool(re.search(r"[\[\]（）()A-Za-z0-9\-]+", normalized))

    def _looks_like_person_name(self, text: str) -> bool:
        if self._is_non_entity_heading_text(text):
            return False
        return re.fullmatch(r"[\u4e00-\u9fa5·某]{2,8}", text) is not None

    def _is_non_entity_heading_text(self, text: str) -> bool:
        normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", str(text or ""))
        if not normalized:
            return False
        return normalized in self.NON_ENTITY_HEADING_TERMS

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        unique = {}
        for entity in entities:
            key = (entity["start"], entity["end"], entity["text"])
            existing = unique.get(key)
            if existing is None or self._should_prefer_entity(existing, entity):
                unique[key] = entity
        return sorted(unique.values(), key=lambda item: (item["start"], item["end"]))

    def _should_prefer_entity(self, existing: Dict, candidate: Dict) -> bool:
        existing_rank = self.ENTITY_TYPE_PRIORITY.get(str(existing.get("type", "")).upper(), 99)
        candidate_rank = self.ENTITY_TYPE_PRIORITY.get(str(candidate.get("type", "")).upper(), 99)
        if candidate_rank != existing_rank:
            return candidate_rank < existing_rank

        existing_score = float(existing.get("score", 0.0))
        candidate_score = float(candidate.get("score", 0.0))
        if candidate_score != existing_score:
            return candidate_score > existing_score

        existing_source = str(existing.get("source", ""))
        candidate_source = str(candidate.get("source", ""))
        return candidate_source.count("structured") + candidate_source.count("refined") > existing_source.count("structured") + existing_source.count("refined")

    def test_connection(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                model_names = self.list_models()
                return {
                    "status": "connected",
                    "available_models": model_names,
                    "current_model": self.model,
                    "model_exists": self.model in model_names,
                }
            return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"status": "disconnected", "message": str(exc)}

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return []
            models = response.json().get("models", [])
            return [
                str(model.get("name")).strip()
                for model in models
                if isinstance(model, dict) and str(model.get("name", "")).strip()
            ]
        except Exception:
            return []
