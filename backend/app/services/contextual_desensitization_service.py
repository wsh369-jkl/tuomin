"""Context-aware replacement generation for anonymization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from app.core.anonymization_strategy import (
    DEFAULT_ANONYMIZATION_STRATEGY,
    get_anonymization_strategy_profile,
)
from app.core.config import settings
from app.core.identifier_rules import (
    label_matches,
    looks_like_case_number,
    mask_case_number_digits_only,
)
from app.core.llm_strategy import (
    get_llm_strategy_profile,
    get_runtime_llm_strategy_profile,
    parse_model_size_in_b,
)
from app.services.ollama_service import OllamaLLMService

logger = logging.getLogger(__name__)


class ContextualDesensitizationService:
    """Generate readable, non-sensitive replacements for detected entities."""

    GROUPABLE_TYPES = {
        "PERSON",
        "PERSON_NAME",
        "ORGANIZATION",
        "COMPANY_NAME",
        "LOCATION",
        "POSITION",
        "PROJECT",
        "CONTRACT_NO",
        "BANK_NAME",
        "ACCOUNT_NAME",
        "PROJECT_CODE",
        "PRODUCT_NAME",
        "SENSITIVE_TERM",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_ID_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
    }

    # Replacements are now deterministic aliases. The LLM still participates in
    # recognition, but replacement wording should stay stable and easy to read.
    LLM_TYPES: set[str] = set()

    PRESERVE_TYPES = {"AMOUNT", "DATE"}

    GROUP_FAMILIES = {
        "PERSON": "person",
        "PERSON_NAME": "person",
        "ORGANIZATION": "organization",
        "COMPANY_NAME": "organization",
        "ACCOUNT_NAME": "organization",
        "BANK_NAME": "bank",
        "PROJECT": "project",
        "LOCATION": "location",
        "POSITION": "position",
        "CONTRACT_NO": "contract_no",
        "PROJECT_CODE": "project_code",
        "PRODUCT_NAME": "product",
        "SENSITIVE_TERM": "term",
        "CN_PHONE": "phone",
        "LANDLINE_PHONE": "phone",
        "CN_BANK_CARD": "bank_card",
        "CN_ID_CARD": "id_card",
        "CN_CREDIT_CODE": "credit_code",
        "EMAIL_ADDRESS": "email",
    }

    TYPE_PRIORITY = {
        "ORGANIZATION": 1,
        "COMPANY_NAME": 1,
        "BANK_NAME": 2,
        "ACCOUNT_NAME": 3,
        "PERSON": 4,
        "PERSON_NAME": 4,
        "PROJECT": 5,
        "LOCATION": 6,
        "POSITION": 7,
        "CONTRACT_NO": 8,
        "PROJECT_CODE": 9,
        "PRODUCT_NAME": 10,
        "SENSITIVE_TERM": 11,
        "CN_PHONE": 12,
        "LANDLINE_PHONE": 12,
        "CN_BANK_CARD": 12,
        "CN_ID_CARD": 12,
        "CN_CREDIT_CODE": 12,
        "EMAIL_ADDRESS": 12,
    }

    SOURCE_LAYER_PRIORITY = {
        "deterministic_rule": 1,
        "structured_label": 2,
        "definition_anchor": 3,
        "llm_semantic": 4,
        "llm_review": 5,
        "propagated": 6,
        "residual": 7,
        "unknown": 9,
    }

    HIGH_CONFIDENCE_IDENTIFIER_TYPES = {
        "CONTRACT_NO",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_ID_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
    }

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
        "仍",
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
        "判决",
    }

    LOW_INFORMATION_ORGANIZATION_ALIASES = {
        "科技",
        "工程",
        "建设",
        "贸易",
        "实业",
        "发展",
        "咨询",
        "服务",
        "管理",
        "材料",
        "电力",
        "能源",
        "建筑",
        "环保",
        "智能",
        "信息",
        "网络",
        "电子",
        "机械",
        "设备",
        "制造",
        "劳务",
        "建筑劳务",
        "工程建设",
        "工程技术",
        "工程设计",
        "建筑工程",
    }

    NON_ENTITY_FIXED_TERMS = {
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

    ORGANIZATION_IDENTIFIER_LABELS = (
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

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CN_SERIALS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]

    RESOLUTION_ROLES = {
        "PARTY_A",
        "PARTY_B",
        "PARTY_C",
        "PROJECT",
        "LOCATION",
        "BANK",
        "ACCOUNT",
        "CONTACT",
        "LEGAL_REPRESENTATIVE",
        "PHONE",
        "CONTRACT_NO",
        "POSITION",
        "ORGANIZATION",
        "PERSON",
        "OTHER",
    }

    REVIEW_ENTITY_TYPES = {
        "PERSON",
        "PERSON_NAME",
        "ORGANIZATION",
        "COMPANY_NAME",
        "LOCATION",
        "POSITION",
        "PROJECT",
        "CONTRACT_NO",
        "BANK_NAME",
        "ACCOUNT_NAME",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_ID_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
    }

    ROLE_LABELS = [
        "甲方",
        "乙方",
        "丙方",
        "委托方",
        "受托方",
        "发包人",
        "承包人",
        "采购人",
        "供应商",
        "收款单位",
        "法定代表人",
        "联系人",
        "项目负责人",
        "经办人",
        "开户行",
        "户名",
        "账户",
        "帐户",
        "账号",
        "帐号",
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
        "职务",
        "职位",
        "岗位",
    ]

    COMPANY_SUFFIXES = [
        "集团股份有限公司",
        "控股股份有限公司",
        "股份有限公司",
        "有限责任公司",
        "集团有限公司",
        "控股有限公司",
        "有限公司",
        "有限合伙",
        "普通合伙",
        "分公司",
        "子公司",
        "集团",
        "公司",
    ]

    LOCATION_SEGMENT_SUFFIXES = [
        "经济技术开发区",
        "高新技术产业开发区",
        "高新技术开发区",
        "高新区",
        "开发区",
        "工业园区",
        "科技园区",
        "新区",
        "街道",
        "自治区",
        "自治州",
        "地区",
        "园区",
        "省",
        "市",
        "区",
        "县",
        "旗",
        "乡",
        "镇",
    ]

    UNSUFFIXED_REGION_PREFIXES = [
        "北京",
        "上海",
        "天津",
        "重庆",
        "广东",
        "广西",
        "海南",
        "河北",
        "河南",
        "湖北",
        "湖南",
        "江苏",
        "浙江",
        "安徽",
        "福建",
        "江西",
        "山东",
        "山西",
        "陕西",
        "四川",
        "贵州",
        "云南",
        "辽宁",
        "吉林",
        "黑龙江",
        "甘肃",
        "青海",
        "台湾",
        "内蒙古",
        "宁夏",
        "新疆",
        "西藏",
        "香港",
        "澳门",
        "广州",
        "深圳",
        "珠海",
        "汕头",
        "佛山",
        "韶关",
        "河源",
        "梅州",
        "汕尾",
        "东莞",
        "中山",
        "江门",
        "阳江",
        "湛江",
        "茂名",
        "肇庆",
        "清远",
        "潮州",
        "揭阳",
        "云浮",
        "惠州",
        "杭州",
        "宁波",
        "温州",
        "嘉兴",
        "湖州",
        "绍兴",
        "金华",
        "台州",
        "丽水",
        "舟山",
        "南京",
        "苏州",
        "无锡",
        "常州",
        "南通",
        "扬州",
        "徐州",
        "镇江",
        "泰州",
        "盐城",
        "淮安",
        "宿迁",
        "连云港",
        "成都",
        "武汉",
        "西安",
        "长沙",
        "郑州",
        "青岛",
        "厦门",
        "福州",
        "济南",
        "合肥",
        "昆明",
        "南宁",
        "贵阳",
        "南昌",
        "海口",
        "太原",
        "沈阳",
        "长春",
        "哈尔滨",
        "石家庄",
        "呼和浩特",
        "乌鲁木齐",
        "拉萨",
        "银川",
        "西宁",
    ]

    PERSON_POOL = [
        "赵明远",
        "钱安宁",
        "孙景和",
        "李清越",
        "周扬飞",
        "吴扬信",
        "郑承宇",
        "王家玥",
        "冯知行",
        "陈佑安",
    ]

    COMPANY_PREFIX = [
        "华宏",
        "远衡",
        "云泽",
        "景岳",
        "启安",
        "星瀚",
        "瑞禾",
        "明川",
        "永海",
        "安策",
    ]

    PROJECT_PREFIX = [
        "华澜",
        "云岳",
        "明辰",
        "启远",
        "瑞风",
        "德海",
        "佳安",
        "远峰",
    ]

    BRANCH_NAMES = [
        "华南支行",
        "新城支行",
        "东湖支行",
        "科创支行",
        "经开支行",
        "滨江支行",
    ]

    PROVINCES = ["广东省", "江苏省", "浙江省", "湖南省", "四川省", "福建省"]
    CITIES = ["宁州市", "云州市", "昌江市", "康州市", "惠安市", "景川市"]
    DISTRICTS = ["安和区", "云山区", "琴湾区", "新城区", "明湖区", "景秀区"]
    COUNTIES = ["永宁县", "昌安县", "景和县", "明清县", "安宜县"]
    TOWNS = ["瑞景镇", "安和镇", "星泽镇", "远川镇", "新湖镇"]
    ROADS = ["创新路", "科创大道", "景明路", "新兴街", "晨光路"]

    INDUSTRY_KEYWORDS = {
        "新能源": ["风电", "光伏", "新能源", "发电", "储能", "电力"],
        "气象": ["气象", "雷电", "防雷", "防御雷电"],
        "检测": ["检测", "检验", "监测", "复检", "验收"],
        "工程": ["工程", "建设", "施工", "安装", "改造"],
        "采购": ["采购", "供应", "招标", "投标", "设备"],
        "设计": ["设计", "图纸", "咨询", "规划"],
        "信息化": ["系统", "平台", "数据", "软件", "信息化"],
    }

    COMPANY_THEME_MAP = {
        "新能源": {
            "company": "新能源开发",
            "center": "新能源服务",
            "institute": "新能源研究",
        },
        "气象": {
            "company": "气象科技",
            "center": "气象技术",
            "institute": "气象研究",
        },
        "检测": {
            "company": "检测技术",
            "center": "检测服务",
            "institute": "检测研究",
        },
        "工程": {
            "company": "工程建设",
            "center": "工程技术",
            "institute": "工程设计",
        },
        "采购": {
            "company": "供应链服务",
            "center": "采购服务",
            "institute": "采购咨询",
        },
        "设计": {
            "company": "设计咨询",
            "center": "设计服务",
            "institute": "设计研究",
        },
        "信息化": {
            "company": "数据科技",
            "center": "信息服务",
            "institute": "数据研究",
        },
        "default": {
            "company": "商务服务",
            "center": "综合服务",
            "institute": "业务研究",
        },
    }

    ORG_SUFFIX_PATTERNS = [
        "股份有限公司",
        "有限责任公司",
        "有限公司",
        "集团有限公司",
        "服务中心",
        "技术中心",
        "研究院",
        "事务所",
        "支行",
        "分行",
        "银行",
        "局",
        "院",
        "所",
    ]

    def __init__(self) -> None:
        self._ollama_services: Dict[str, OllamaLLMService] = {}
        self.last_quality_metadata: Dict[str, Any] = {}

    def get_last_quality_metadata(self) -> Dict[str, Any]:
        return dict(self.last_quality_metadata)

    async def refine_recognition_entities(
        self,
        *,
        text: str,
        entities: List[Dict],
        use_llm: bool,
        llm_model: Optional[str] = None,
    ) -> List[Dict]:
        refined_entities = [dict(entity) for entity in entities]
        refined_entities = self._apply_metadata_identity_hints(refined_entities)
        if not refined_entities or not use_llm:
            return refined_entities

        ollama = self._get_ollama_service(llm_model=llm_model)
        if ollama is None or not ollama.available:
            return refined_entities

        contexts = [self._build_context(text, entity) for entity in refined_entities]
        llm_result = await self._build_llm_resolution_result(
            text=text,
            entities=refined_entities,
            contexts=contexts,
            llm_model=llm_model,
        )
        if not llm_result:
            return refined_entities

        updates_by_id = {
            item["id"]: item
            for item in llm_result.get("entity_updates", [])
            if isinstance(item, dict) and item.get("id")
        }

        for index, entity in enumerate(refined_entities, start=1):
            update = updates_by_id.get(f"E{index}")
            if not update:
                continue

            canonical_key = self._sanitize_canonical_key(update.get("canonical_key"))
            if canonical_key:
                entity["canonical_key"] = canonical_key

            canonical_role = str(update.get("canonical_role", "")).strip().upper()
            if canonical_role in self.RESOLUTION_ROLES:
                entity["canonical_role"] = canonical_role

        extra_entities = self._materialize_llm_extra_entities(
            text=text,
            existing_entities=refined_entities,
            extra_items=llm_result.get("extra_entities", []),
        )
        if extra_entities:
            refined_entities.extend(extra_entities)
            refined_entities.sort(key=lambda item: (item["start"], item["end"]))

        return refined_entities

    async def prepare_entities(
        self,
        *,
        text: str,
        entities: List[Dict],
        use_llm: bool,
        operator_config: Optional[Dict[str, Dict]] = None,
        llm_model: Optional[str] = None,
        anonymization_strategy: Optional[str] = None,
    ) -> List[Dict]:
        prepared_entities = self._apply_metadata_identity_hints([dict(entity) for entity in entities])
        prepared_entities = self._prune_invalid_entities(prepared_entities)
        explicit_types = set((operator_config or {}).keys()) - {"default"}
        strategy_key = get_anonymization_strategy_profile(anonymization_strategy).key
        strategy_profile = get_llm_strategy_profile(llm_model or settings.get_default_llm_model()) if use_llm else None
        precision_multi_review = bool(strategy_profile and strategy_profile.key == "precision_4b")
        self.last_quality_metadata = {}
        residual_rounds: List[Dict[str, Any]] = []
        text_groups: Dict[str, Dict] = {}
        group_keys: List[str] = []
        contexts: List[Dict[str, Any]] = []

        max_rounds = 3 if precision_multi_review else (2 if use_llm else 1)
        for round_index in range(max_rounds):
            contexts = [self._build_context(text, entity) for entity in prepared_entities]
            text_groups, group_keys = self._group_entities_by_identity(
                prepared_entities,
                contexts,
                explicit_types,
            )
            entity_memory = self._build_entity_memory(text_groups)
            residual_entities = self._materialize_residual_entities(
                text=text,
                entities=prepared_entities,
                entity_memory=entity_memory,
                explicit_types=explicit_types,
            )
            residual_rounds.append(
                {
                    "round": round_index + 1,
                    "memory_groups": len(entity_memory),
                    "residual_entities_added": len(residual_entities),
                }
            )
            if not residual_entities:
                break
            prepared_entities.extend(residual_entities)
            prepared_entities = self._sort_and_deduplicate_entities(prepared_entities)
            prepared_entities = self._apply_metadata_identity_hints(prepared_entities)
            prepared_entities = self._prune_invalid_entities(prepared_entities)

        contexts = [self._build_context(text, entity) for entity in prepared_entities]
        text_groups, group_keys = self._group_entities_by_identity(
            prepared_entities,
            contexts,
            explicit_types,
        )
        entity_memory = self._build_entity_memory(text_groups)
        replacement_bundle = self._build_deterministic_group_replacement_bundle(
            text_groups,
            strategy_key=strategy_key,
        )
        replacements = replacement_bundle["replacements"]
        base_replacements = replacement_bundle["base_replacements"]
        collision_suffixes = replacement_bundle["collision_suffixes"]
        llm_updates: Dict[str, str] = {}

        if use_llm:
            llm_updates = await self._build_llm_group_replacements(
                text_groups,
                replacements,
                llm_model=llm_model,
            )
            replacements.update(llm_updates)

        amount_cluster_values = self._build_amount_cluster_values(prepared_entities, contexts)

        for index, entity in enumerate(prepared_entities):
            entity_type = entity["type"]
            entity_context = contexts[index]
            entity["context_label"] = entity_context.get("label") or None
            entity["context_role"] = entity_context.get("role") or None
            if entity_type in explicit_types:
                continue

            if entity_type in self.GROUPABLE_TYPES:
                group_key = group_keys[index]
                replacement = replacements.get(group_key)
                if replacement:
                    metadata = dict(entity.get("metadata") or {})
                    if group_key not in llm_updates:
                        rendered_replacement, replacement_family_key = self._render_group_replacement(
                            entity=entity,
                            context=entity_context,
                            group=text_groups[group_key],
                            group_key=group_key,
                            default_replacement=replacement,
                            base_replacement=base_replacements.get(group_key, replacement),
                            collision_suffix=collision_suffixes.get(group_key, ""),
                            strategy_key=strategy_key,
                        )
                        replacement = rendered_replacement
                        if replacement_family_key:
                            metadata["replacement_family_key"] = replacement_family_key
                    if self._should_preserve_surface_replacement(entity_type, entity.get("text", ""), replacement):
                        entity.pop("replacement", None)
                        if metadata:
                            entity["metadata"] = metadata
                        entity["replacement_method"] = "preserve"
                        continue
                    entity["replacement"] = replacement
                    if metadata:
                        entity["metadata"] = metadata
                    entity["replacement_method"] = (
                        "llm_contextual" if group_key in llm_updates else "contextual"
                    )
                    continue

            structured_replacement = self._build_structured_replacement(
                entity=entity,
                context=entity_context,
                amount_cluster_values=amount_cluster_values,
            )
            if structured_replacement:
                entity["replacement"] = structured_replacement
                entity["replacement_method"] = "structured"
            elif entity_type in self.PRESERVE_TYPES:
                entity.pop("replacement", None)
                entity["replacement_method"] = "preserve"

        prepared_entities = self._harmonize_duplicate_text_replacements(prepared_entities)
        repair_rounds: List[Dict[str, Any]] = []
        quality_gate_passed = False
        quality_gate_reason = ""

        max_repair_rounds = 4 if precision_multi_review else 3
        for repair_index in range(max_repair_rounds):
            contexts = [self._build_context(text, entity) for entity in prepared_entities]
            text_groups, group_keys = self._group_entities_by_identity(
                prepared_entities,
                contexts,
                explicit_types,
            )
            entity_memory = self._build_entity_memory(text_groups)
            prepared_entities = self._annotate_entities_with_evidence(
                entities=prepared_entities,
                contexts=contexts,
                group_keys=group_keys,
                entity_memory=entity_memory,
            )
            consistency_issues = self._collect_consistency_issues(
                entities=prepared_entities,
                group_keys=group_keys,
            )
            residual_hits = self._collect_residual_hits(
                text=text,
                entities=prepared_entities,
                entity_memory=entity_memory,
                explicit_types=explicit_types,
            )
            quality_gate_passed = not consistency_issues and not residual_hits
            repair_rounds.append(
                {
                    "round": repair_index + 1,
                    "consistency_issue_count": len(consistency_issues),
                    "residual_hit_count": len(residual_hits),
                    "passed": quality_gate_passed,
                }
            )
            if quality_gate_passed:
                break

            prepared_entities, repaired = self._repair_quality_issues(
                text=text,
                entities=prepared_entities,
                explicit_types=explicit_types,
                strategy_key=strategy_key,
                group_keys=group_keys,
                groups=text_groups,
                entity_memory=entity_memory,
                consistency_issues=consistency_issues,
                residual_hits=residual_hits,
            )
            if not repaired:
                quality_gate_reason = self._summarize_quality_gate_failure(consistency_issues, residual_hits)
                break

        prepared_entities = self._prune_invalid_entities(prepared_entities)
        contexts = [self._build_context(text, entity) for entity in prepared_entities]
        text_groups, group_keys = self._group_entities_by_identity(
            prepared_entities,
            contexts,
            explicit_types,
        )
        entity_memory = self._build_entity_memory(text_groups)
        prepared_entities = self._annotate_entities_with_evidence(
            entities=prepared_entities,
            contexts=contexts,
            group_keys=group_keys,
            entity_memory=entity_memory,
        )
        consistency_issues = self._collect_consistency_issues(
            entities=prepared_entities,
            group_keys=group_keys,
        )
        residual_hits = self._collect_residual_hits(
            text=text,
            entities=prepared_entities,
            entity_memory=entity_memory,
            explicit_types=explicit_types,
        )
        quality_gate_passed = not consistency_issues and not residual_hits
        if not quality_gate_reason and not quality_gate_passed:
            quality_gate_reason = self._summarize_quality_gate_failure(consistency_issues, residual_hits)
        self.last_quality_metadata = {
            "engine_strategy": strategy_profile.key if strategy_profile else "rules_only",
            "recall_passes": residual_rounds,
            "repair_rounds": repair_rounds,
            "entity_memory_groups": len(entity_memory),
            "consistency_issues": consistency_issues,
            "residual_hits": residual_hits,
            "quality_gate_passed": quality_gate_passed,
            "quality_gate_reason": quality_gate_reason,
            "arbitration_conflicts": sum(1 for item in entity_memory.values() if item.get("conflict")),
            "evidence_summary": [
                {
                    "canonical_key": key,
                    "primary_text": value.get("primary_text", ""),
                    "source_layer": value.get("source_layer", ""),
                    "conflict": bool(value.get("conflict")),
                    "conflict_reasons": list(value.get("conflict_reasons", [])),
                    "evidence_chain": list(value.get("evidence_chain", []))[:3],
                }
                for key, value in list(entity_memory.items())[:20]
            ],
        }
        return prepared_entities

    def _apply_metadata_identity_hints(self, entities: List[Dict]) -> List[Dict]:
        enriched: List[Dict] = []
        for entity in entities:
            item = dict(entity)
            metadata = item.get("metadata") or {}
            if not item.get("canonical_key") and metadata.get("canonical_key"):
                item["canonical_key"] = metadata["canonical_key"]
            if not item.get("canonical_role") and metadata.get("canonical_role"):
                item["canonical_role"] = metadata["canonical_role"]
            enriched.append(item)
        return enriched

    def _sort_and_deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        deduplicated: Dict[tuple[str, int, int, str], Dict[str, Any]] = {}
        for entity in entities:
            key = (
                str(entity.get("type", "")),
                int(entity.get("start", 0)),
                int(entity.get("end", 0)),
                str(entity.get("text", "")),
            )
            existing = deduplicated.get(key)
            if existing is None or float(entity.get("score", 0.0)) > float(existing.get("score", 0.0)):
                deduplicated[key] = entity
        return sorted(deduplicated.values(), key=lambda item: (int(item["start"]), int(item["end"])))

    def _build_entity_memory(self, groups: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        memory: Dict[str, Dict[str, Any]] = {}
        for group_key, group in groups.items():
            arbitration = self._arbitrate_group_memory(group)
            variants = self._derive_group_variants(group_key, group)
            canonical_role = arbitration["canonical_role"]
            primary_entity = arbitration["primary_entity"]
            primary_type = str(primary_entity["type"])
            sources = sorted({str(entity.get("source", "")) for entity in group["entities"] if entity.get("source")})
            memory[group_key] = {
                "canonical_key": group_key,
                "primary_type": primary_type,
                "primary_text": primary_entity["text"],
                "canonical_role": canonical_role,
                "variants": sorted(variants, key=lambda item: (-len(re.sub(r"\s+", "", item)), item)),
                "source_texts": sorted(group.get("source_texts", set())),
                "sources": sources,
                "first_start": int(group["first_start"]),
                "last_start": max(int(entity["start"]) for entity in group["entities"]),
                "confidence": max(float(entity.get("score", 0.0)) for entity in group["entities"]),
                "confirmed": bool(self._canonical_key_from_entity(primary_entity) or arbitration["source_layer"] in {"deterministic_rule", "definition_anchor"}),
                "conflict": arbitration["conflict"],
                "conflict_reasons": arbitration["conflict_reasons"],
                "source_layer": arbitration["source_layer"],
                "source_priority": arbitration["source_priority"],
                "evidence_chain": arbitration["evidence_chain"],
                "primary_start": int(primary_entity["start"]),
            }
        return memory

    def _arbitrate_group_memory(self, group: Dict[str, Any]) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        explicit_keys = {
            self._canonical_key_from_entity(entity)
            for entity in group["entities"]
            if self._canonical_key_from_entity(entity)
        }
        party_roles = {
            str(entity.get("canonical_role", "")).strip().upper()
            for entity in group["entities"]
            if str(entity.get("canonical_role", "")).strip().upper() in {"PARTY_A", "PARTY_B", "PARTY_C"}
        }

        for entity, context in zip(group["entities"], group["contexts"]):
            source_layer = self._source_layer(entity)
            source_priority = self._source_priority(entity)
            candidates.append(
                {
                    "entity": entity,
                    "context": context,
                    "source_layer": source_layer,
                    "source_priority": source_priority,
                    "context_specificity": self._context_specificity(context),
                    "score": float(entity.get("score", 0.0)),
                    "canonical_key": self._canonical_key_from_entity(entity),
                }
            )

        candidates.sort(
            key=lambda item: (
                item["source_priority"],
                self._entity_priority(item["entity"]),
                -item["context_specificity"],
                -item["score"],
                int(item["entity"]["start"]),
            )
        )
        winner = candidates[0]
        conflict_reasons: List[str] = []
        if len(explicit_keys) > 1:
            conflict_reasons.append("multiple_canonical_keys")
        if len(party_roles) > 1:
            conflict_reasons.append("multiple_party_roles")
        if len(candidates) > 1:
            runner_up = candidates[1]
            if (
                winner["source_priority"] == runner_up["source_priority"]
                and self._normalize_group_text(str(winner["entity"]["text"])) != self._normalize_group_text(str(runner_up["entity"]["text"]))
            ):
                conflict_reasons.append("multiple_top_candidates_same_priority")

        evidence_chain = [
            {
                "text": str(item["entity"]["text"]),
                "type": str(item["entity"]["type"]),
                "source": str(item["entity"].get("source", "")),
                "source_layer": item["source_layer"],
                "source_priority": item["source_priority"],
                "score": item["score"],
                "label": item["context"].get("label", ""),
                "role": item["context"].get("role", ""),
                "line": item["context"].get("line", ""),
            }
            for item in candidates[:10]
        ]

        return {
            "primary_entity": winner["entity"],
            "primary_context": winner["context"],
            "canonical_role": self._pick_group_role(group),
            "source_layer": winner["source_layer"],
            "source_priority": winner["source_priority"],
            "conflict": bool(conflict_reasons),
            "conflict_reasons": conflict_reasons,
            "evidence_chain": evidence_chain,
        }

    def _derive_group_variants(self, group_key: str, group: Dict[str, Any]) -> set[str]:
        variants = {str(group_key), str(group["primary_entity"]["text"])}
        primary_type = str(group["primary_entity"]["type"])
        primary_text = str(group["primary_entity"]["text"])
        metadata_candidates = []
        for entity in group["entities"]:
            metadata = entity.get("metadata") or {}
            metadata_candidates.extend(
                [
                    metadata.get("definition_alias"),
                    metadata.get("definition_full_text"),
                    metadata.get("residual_variant"),
                ]
            )
            variants.add(str(entity.get("text", "")))

        if primary_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            variants.update(self._derive_organization_identity_aliases(group["primary_entity"]["text"]))
        elif primary_type in {"PERSON", "PERSON_NAME"}:
            variants.update(self._derive_person_identity_aliases(group["primary_entity"]["text"]))
        elif primary_type == "PROJECT":
            variants.update(self._derive_project_identity_aliases(group["primary_entity"]["text"]))

        for candidate in metadata_candidates:
            if candidate:
                variants.add(str(candidate))

        if primary_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            metadata_variant_keys = {
                self._normalize_group_text(str(candidate))
                for candidate in metadata_candidates
                if candidate
            }
            return {
                variant
                for variant in variants
                if self._is_valid_organization_variant(
                    variant,
                    primary_text=primary_text,
                    metadata=(
                        {"definition_alias": variant}
                        if self._normalize_group_text(str(variant)) in metadata_variant_keys
                        else None
                    ),
                )
            }
        return {variant for variant in variants if len(re.sub(r"\s+", "", variant)) >= 2}

    def _derive_person_identity_aliases(self, text: str) -> set[str]:
        normalized = self._normalize_person_source_text(text)
        if len(normalized) < 2:
            return set()
        aliases = {normalized}
        aliases.add(self._build_person_alias(normalized))
        return {alias for alias in aliases if alias}

    def _is_low_information_organization_alias(self, text: str) -> bool:
        normalized = self._normalize_group_text(text)
        if len(normalized) < 2:
            return True
        if normalized in self.LOW_INFORMATION_ORGANIZATION_ALIASES:
            return True

        remainder = normalized
        while remainder:
            matched = False
            for token in sorted(self.LOW_INFORMATION_ORGANIZATION_ALIASES, key=len, reverse=True):
                if remainder.startswith(token):
                    remainder = remainder[len(token) :]
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _derive_project_identity_aliases(self, text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text)
        aliases = {normalized}
        stripped = re.sub(r"(项目|工程|标段)$", "", normalized)
        if len(stripped) >= 2:
            aliases.add(stripped)
        return {alias for alias in aliases if alias}

    def _materialize_residual_entities(
        self,
        *,
        text: str,
        entities: List[Dict],
        entity_memory: Dict[str, Dict[str, Any]],
        explicit_types: set[str],
    ) -> List[Dict]:
        occupied = {(int(item["start"]), int(item["end"]), str(item["type"])) for item in entities}
        occupied_ranges = [(int(item["start"]), int(item["end"])) for item in entities]
        residual_entities: List[Dict] = []

        for group_key, memory in entity_memory.items():
            entity_type = str(memory["primary_type"])
            if entity_type in explicit_types:
                continue
            if entity_type not in self.GROUPABLE_TYPES:
                continue

            for variant in memory["variants"]:
                variant_norm = re.sub(r"\s+", "", variant)
                if len(variant_norm) < 2:
                    continue
                if (
                    entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}
                    and variant_norm != self._normalize_group_text(str(memory.get("primary_text", "")))
                    and self._is_low_information_organization_alias(variant_norm)
                ):
                    continue
                for start, end in self._find_text_spans(text, variant):
                    key = (start, end, entity_type)
                    if key in occupied:
                        continue
                    if any(start < existing_end and end > existing_start for existing_start, existing_end in occupied_ranges):
                        continue
                    residual_entities.append(
                        {
                            "type": entity_type,
                            "text": text[start:end],
                            "start": start,
                            "end": end,
                            "score": 0.82,
                            "source": "memory_residual",
                            "canonical_key": group_key,
                            "canonical_role": memory.get("canonical_role") or None,
                            "metadata": {
                                "canonical_key": group_key,
                                "canonical_role": memory.get("canonical_role") or "",
                                "residual_variant": variant,
                                "residual_scan": True,
                            },
                        }
                    )
                    occupied.add(key)
                    occupied_ranges.append((start, end))
                    break

        return residual_entities

    def _collect_residual_hits(
        self,
        *,
        text: str,
        entities: List[Dict],
        entity_memory: Dict[str, Dict[str, Any]],
        explicit_types: set[str],
    ) -> List[Dict[str, Any]]:
        covered = {
            (int(entity["start"]), int(entity["end"]), str(entity["type"]))
            for entity in entities
        }
        covered_ranges = [(int(entity["start"]), int(entity["end"])) for entity in entities]
        hits: List[Dict[str, Any]] = []

        for group_key, memory in entity_memory.items():
            entity_type = str(memory["primary_type"])
            if entity_type in explicit_types:
                continue
            for variant in memory["variants"]:
                if (
                    entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}
                    and self._normalize_group_text(str(variant)) != self._normalize_group_text(str(memory.get("primary_text", "")))
                    and self._is_low_information_organization_alias(str(variant))
                ):
                    continue
                for start, end in self._find_text_spans(text, variant):
                    if (start, end, entity_type) in covered:
                        continue
                    if any(start < existing_end and end > existing_start for existing_start, existing_end in covered_ranges):
                        continue
                    context = self._build_context(
                        text,
                        {
                            "type": entity_type,
                            "text": text[start:end],
                            "start": start,
                            "end": end,
                        },
                    )
                    hits.append(
                        {
                            "canonical_key": group_key,
                            "type": entity_type,
                            "variant": variant,
                            "start": start,
                            "end": end,
                            "line": context.get("line", ""),
                            "window_text": context.get("window_text", ""),
                        }
                    )
                    if len(hits) >= 20:
                        return hits
                    break
        return hits

    def _normalize_replacement_text(self, value: Any) -> str:
        if value is None:
            return ""
        replacement = str(value).strip()
        if replacement.lower() == "none":
            return ""
        return replacement

    def _prune_invalid_entities(self, entities: List[Dict]) -> List[Dict]:
        filtered: List[Dict] = []
        removed = 0
        for entity in entities:
            entity_type = str(entity.get("type", "")).upper()
            if entity_type in {"PERSON", "PERSON_NAME", "ORGANIZATION", "COMPANY_NAME", "PROJECT"} and self._is_non_entity_heading_candidate(
                str(entity.get("text", ""))
            ):
                removed += 1
                continue
            if entity_type in {"ORGANIZATION", "COMPANY_NAME"} and not self._is_valid_organization_variant(
                str(entity.get("text", "")),
                primary_text=str(entity.get("text", "")),
                source=str(entity.get("source", "")),
                metadata=entity.get("metadata") or {},
            ):
                removed += 1
                continue
            if entity_type in self.PRESERVE_TYPES:
                entity = dict(entity)
                entity.pop("replacement", None)
            filtered.append(entity)

        if removed:
            logger.info("Pruned %s invalid organization entities before replacement", removed)
        return filtered

    def _is_non_entity_heading_candidate(self, text: str) -> bool:
        normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", str(text or ""))
        if not normalized:
            return False
        return normalized in self.NON_ENTITY_FIXED_TERMS

    def _strip_organization_suffix(self, text: str) -> str:
        return re.sub(
            r"(股份有限公司|有限责任公司|有限公司|集团有限公司|集团|研究院|研究所|事务所|服务中心|中心|银行|支行|分行|子公司|分公司|公司)$",
            "",
            text,
        )

    def _looks_like_identifier_like_text(self, text: str) -> bool:
        normalized = self._normalize_group_text(text)
        if not normalized:
            return False
        if any(label in text for label in self.ORGANIZATION_IDENTIFIER_LABELS):
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

    def _is_valid_organization_variant(
        self,
        text: str,
        *,
        primary_text: str = "",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        normalized = self._normalize_group_text(text)
        if len(normalized) < 2:
            return False
        if self._looks_like_identifier_like_text(normalized):
            return False
        if normalized in self.ORGANIZATION_GENERIC_TERMS:
            return False
        if any(normalized.endswith(token) for token in ("法院", "检察院", "公安局", "仲裁委员会")) and len(normalized) < 4:
            return False
        if any(token in normalized for token in self.ORGANIZATION_IDENTIFIER_LABELS):
            return False
        if any(normalized.startswith(prefix) and len(normalized) > len(prefix) + 1 for prefix in self.ORGANIZATION_STOP_PREFIXES):
            return False
        if any(fragment in normalized for fragment in self.ORGANIZATION_NOISE_FRAGMENTS):
            return False

        core = self._strip_organization_suffix(normalized)
        if self._looks_like_organization_name(normalized):
            metadata = metadata or {}
            if len(core) < 2:
                anchor_short_name = (
                    bool(metadata.get("definition_alias") or metadata.get("definition_full_text"))
                    or bool(metadata.get("canonical_key"))
                    or bool(metadata.get("residual_scan"))
                    or source in {"memory_residual", "residual_repair", "propagate", "ollama_definition"}
                )
                if anchor_short_name and normalized.endswith(("公司", "集团", "银行", "支行", "分行")):
                    return core not in {"本", "该", "我", "贵", "相关", "上述", "涉案", "某"}
                return False
            if core in {"本", "该", "我", "贵", "相关", "上述", "涉案", "某"}:
                return False
            if any(fragment in core for fragment in self.ORGANIZATION_NOISE_FRAGMENTS):
                return False
            if any(core.startswith(prefix) and len(core) > len(prefix) for prefix in self.ORGANIZATION_STOP_PREFIXES):
                return False
            return True

        if re.fullmatch(r"[\u4e00-\u9fa5]{2,6}", normalized) is None:
            return False
        if source.startswith("ollama") and source not in {"memory_residual", "residual_repair", "propagate"}:
            metadata = metadata or {}
            if not (
                metadata.get("canonical_key")
                or metadata.get("definition_alias")
                or metadata.get("definition_full_text")
            ):
                return False

        primary_normalized = self._normalize_group_text(primary_text)
        if primary_normalized and normalized not in primary_normalized and primary_normalized not in normalized:
            metadata = metadata or {}
            if not (metadata.get("definition_alias") or metadata.get("definition_full_text")):
                return False
        return True

    def _collect_consistency_issues(
        self,
        *,
        entities: List[Dict],
        group_keys: List[str],
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        replacement_sets: Dict[str, set[str]] = {}
        seen_texts: Dict[str, set[str]] = {}

        for entity, group_key in zip(entities, group_keys):
            replacement = self._normalize_replacement_text(entity.get("replacement"))
            if not replacement:
                continue
            metadata = entity.get("metadata") or {}
            replacement_family_key = str(metadata.get("replacement_family_key", "")).strip()
            replacement_sets.setdefault(group_key, set()).add(replacement_family_key or replacement)
            normalized_text = self._normalize_group_text(str(entity.get("text", "")))
            seen_texts.setdefault(normalized_text, set()).add(replacement)

        for group_key, values in replacement_sets.items():
            if len(values) > 1:
                issues.append(
                    {
                        "type": "canonical_key_multi_replacement",
                        "canonical_key": group_key,
                        "replacements": sorted(values),
                    }
                )

        for normalized_text, values in seen_texts.items():
            if normalized_text and len(values) > 1:
                issues.append(
                    {
                        "type": "same_text_multi_replacement",
                        "text": normalized_text,
                        "replacements": sorted(values),
                    }
                )

        return issues

    def _harmonize_duplicate_text_replacements(self, entities: List[Dict]) -> List[Dict]:
        preferred_replacements: Dict[str, str] = {}
        ordered_entities = sorted(entities, key=lambda item: (int(item["start"]), int(item["end"])))
        for entity in ordered_entities:
            replacement = self._normalize_replacement_text(entity.get("replacement"))
            if not replacement:
                continue
            normalized = self._normalize_group_text(str(entity.get("text", "")))
            preferred_replacements.setdefault(normalized, replacement)

        harmonized: List[Dict] = []
        for entity in ordered_entities:
            item = dict(entity)
            normalized = self._normalize_group_text(str(item.get("text", "")))
            replacement = preferred_replacements.get(normalized)
            if replacement and item.get("replacement") != replacement:
                item["replacement"] = replacement
                item["replacement_method"] = "contextual_consistent"
            harmonized.append(item)
        return harmonized

    def _annotate_entities_with_evidence(
        self,
        *,
        entities: List[Dict],
        contexts: List[Dict[str, Any]],
        group_keys: List[str],
        entity_memory: Dict[str, Dict[str, Any]],
    ) -> List[Dict]:
        annotated: List[Dict] = []
        for entity, context, group_key in zip(entities, contexts, group_keys):
            item = dict(entity)
            metadata = dict(item.get("metadata") or {})
            memory = entity_memory.get(group_key, {})
            source_layer = self._source_layer(item)
            source_priority = self._source_priority(item)
            metadata.update(
                {
                    "source_layer": source_layer,
                    "source_priority": source_priority,
                    "trigger_line": context.get("line", ""),
                    "trigger_label": context.get("label", ""),
                    "trigger_role": context.get("role", ""),
                    "evidence_window": context.get("window_text", ""),
                    "resolved_group_key": group_key,
                    "memory_primary_text": memory.get("primary_text", ""),
                    "memory_primary_type": memory.get("primary_type", ""),
                    "memory_conflict": bool(memory.get("conflict")),
                    "memory_conflict_reasons": list(memory.get("conflict_reasons", [])),
                    "arbitration_result": self._entity_arbitration_result(item, memory),
                }
            )
            if memory.get("canonical_key") and not item.get("canonical_key") and self._sanitize_canonical_key(memory["canonical_key"]):
                item["canonical_key"] = memory["canonical_key"]
            if memory.get("canonical_role") and not item.get("canonical_role"):
                item["canonical_role"] = memory["canonical_role"]
            item["metadata"] = metadata
            annotated.append(item)
        return annotated

    def _entity_arbitration_result(self, entity: Dict[str, Any], memory: Dict[str, Any]) -> str:
        primary_text = str(memory.get("primary_text", ""))
        if not primary_text:
            return "unknown"
        same_text = self._normalize_group_text(str(entity.get("text", ""))) == self._normalize_group_text(primary_text)
        if same_text and int(entity.get("start", 0)) == int(memory.get("first_start", 0)):
            return "winner"
        if same_text:
            return "aligned"
        return "supporting"

    def _repair_quality_issues(
        self,
        *,
        text: str,
        entities: List[Dict],
        explicit_types: set[str],
        strategy_key: str,
        group_keys: List[str],
        groups: Dict[str, Dict[str, Any]],
        entity_memory: Dict[str, Dict[str, Any]],
        consistency_issues: List[Dict[str, Any]],
        residual_hits: List[Dict[str, Any]],
    ) -> tuple[List[Dict], bool]:
        repaired_entities = [dict(entity) for entity in entities]
        changed = False

        if residual_hits:
            extra_entities = self._materialize_residual_hit_entities(
                text=text,
                residual_hits=residual_hits,
                entity_memory=entity_memory,
                existing_entities=repaired_entities,
            )
            if extra_entities:
                repaired_entities.extend(extra_entities)
                repaired_entities = self._sort_and_deduplicate_entities(repaired_entities)
                changed = True

        if consistency_issues or changed:
            contexts = [self._build_context(text, entity) for entity in repaired_entities]
            groups, group_keys = self._group_entities_by_identity(
                repaired_entities,
                contexts,
                explicit_types,
            )
            replacement_bundle = self._build_deterministic_group_replacement_bundle(
                groups=groups,
                strategy_key=strategy_key,
            )
            replacement_map = replacement_bundle["replacements"]
            base_replacements = replacement_bundle["base_replacements"]
            collision_suffixes = replacement_bundle["collision_suffixes"]
            for entity, group_key, context in zip(repaired_entities, group_keys, contexts):
                if entity["type"] in explicit_types or entity["type"] not in self.GROUPABLE_TYPES:
                    continue
                replacement = replacement_map.get(group_key)
                metadata = dict(entity.get("metadata") or {})
                if replacement:
                    replacement, replacement_family_key = self._render_group_replacement(
                        entity=entity,
                        context=context,
                        group=groups[group_key],
                        group_key=group_key,
                        default_replacement=replacement,
                        base_replacement=base_replacements.get(group_key, replacement),
                        collision_suffix=collision_suffixes.get(group_key, ""),
                        strategy_key=strategy_key,
                    )
                    if replacement_family_key:
                        metadata["replacement_family_key"] = replacement_family_key
                    elif "replacement_family_key" in metadata:
                        metadata.pop("replacement_family_key", None)
                if self._should_preserve_surface_replacement(entity["type"], entity.get("text", ""), replacement):
                    entity.pop("replacement", None)
                    if metadata:
                        entity["metadata"] = metadata
                    entity["replacement_method"] = "preserve"
                    changed = True
                    continue
                if replacement and entity.get("replacement") != replacement:
                    entity["replacement"] = replacement
                    if metadata:
                        entity["metadata"] = metadata
                    entity["replacement_method"] = "contextual_repaired"
                    changed = True

            repaired_entities = self._harmonize_duplicate_text_replacements(repaired_entities)

        return repaired_entities, changed

    def _materialize_residual_hit_entities(
        self,
        *,
        text: str,
        residual_hits: List[Dict[str, Any]],
        entity_memory: Dict[str, Dict[str, Any]],
        existing_entities: List[Dict],
    ) -> List[Dict]:
        occupied = {(int(item["start"]), int(item["end"]), str(item["type"])) for item in existing_entities}
        occupied_ranges = [(int(item["start"]), int(item["end"])) for item in existing_entities]
        additions: List[Dict] = []
        for hit in residual_hits[:24]:
            entity_type = str(hit.get("type", ""))
            start = int(hit.get("start", 0))
            end = int(hit.get("end", 0))
            canonical_key = str(hit.get("canonical_key", ""))
            if (start, end, entity_type) in occupied or start >= end:
                continue
            if any(start < existing_end and end > existing_start for existing_start, existing_end in occupied_ranges):
                continue
            memory = entity_memory.get(canonical_key, {})
            additions.append(
                {
                    "type": entity_type,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "score": 0.84,
                    "source": "residual_repair",
                    "canonical_key": canonical_key or None,
                    "canonical_role": memory.get("canonical_role") or None,
                    "metadata": {
                        "canonical_key": canonical_key,
                        "canonical_role": memory.get("canonical_role") or "",
                        "residual_variant": hit.get("variant", ""),
                        "residual_scan": True,
                        "repair_round": True,
                    },
                }
            )
            occupied.add((start, end, entity_type))
            occupied_ranges.append((start, end))
        return additions

    def _rebuild_group_replacements(
        self,
        *,
        groups: Dict[str, Dict[str, Any]],
        entities: List[Dict],
        group_keys: List[str],
        strategy_key: str,
    ) -> Dict[str, str]:
        rebuilt = self._build_deterministic_group_replacements(groups, strategy_key=strategy_key)
        group_existing: Dict[str, Dict[str, int]] = {}
        for entity, group_key in zip(entities, group_keys):
            replacement = self._normalize_replacement_text(entity.get("replacement"))
            if not replacement:
                continue
            bucket = group_existing.setdefault(group_key, {})
            bucket[replacement] = bucket.get(replacement, 0) + 1

        for group_key, bucket in group_existing.items():
            if not bucket:
                continue
            preferred = sorted(bucket.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0][0]
            rebuilt[group_key] = preferred
        return rebuilt

    def _summarize_quality_gate_failure(
        self,
        consistency_issues: List[Dict[str, Any]],
        residual_hits: List[Dict[str, Any]],
    ) -> str:
        reasons: List[str] = []
        if consistency_issues:
            reasons.append(f"consistency_issues={len(consistency_issues)}")
        if residual_hits:
            reasons.append(f"residual_hits={len(residual_hits)}")
        return ", ".join(reasons)

    def _group_entities_by_identity(
        self,
        entities: List[Dict],
        contexts: List[Dict],
        explicit_types: set[str],
    ) -> tuple[Dict[str, Dict], List[str]]:
        groups: Dict[str, Dict] = {}
        group_keys: List[str] = []

        for index, (entity, context) in enumerate(zip(entities, contexts)):
            entity_type = entity["type"]
            if entity_type not in self.GROUPABLE_TYPES:
                group_keys.append(entity["text"])
                continue
            if entity_type in explicit_types:
                group_keys.append(entity["text"])
                continue

            group_key = self._resolve_group_key(index, entities, contexts)
            group_keys.append(group_key)

            group = groups.setdefault(
                group_key,
                {
                    "text": group_key,
                    "entities": [],
                    "contexts": [],
                    "primary_entity": entity,
                    "primary_context": context,
                    "first_start": entity["start"],
                    "source_texts": set(),
                    "entity_types": set(),
                },
            )
            group["entities"].append(entity)
            group["contexts"].append(context)
            group["source_texts"].add(entity["text"])
            group["entity_types"].add(entity_type)
            group["first_start"] = min(group["first_start"], entity["start"])

            current_primary = group["primary_entity"]
            if self._should_promote_group_primary(
                candidate_entity=entity,
                candidate_context=context,
                current_entity=current_primary,
                current_context=group["primary_context"],
            ):
                group["primary_entity"] = entity
                group["primary_context"] = context

        return groups, group_keys

    def _should_promote_group_primary(
        self,
        *,
        candidate_entity: Dict,
        candidate_context: Dict,
        current_entity: Dict,
        current_context: Dict,
    ) -> bool:
        candidate_priority = self._entity_priority(candidate_entity)
        current_priority = self._entity_priority(current_entity)
        if candidate_priority != current_priority:
            return candidate_priority < current_priority

        candidate_source = self._source_priority(candidate_entity)
        current_source = self._source_priority(current_entity)
        if candidate_source != current_source:
            return candidate_source < current_source

        candidate_specificity = self._context_specificity(candidate_context)
        current_specificity = self._context_specificity(current_context)
        if candidate_specificity != current_specificity:
            return candidate_specificity > current_specificity

        return int(candidate_entity["start"]) < int(current_entity["start"])

    def _build_deterministic_group_replacement_bundle(
        self,
        groups: Dict[str, Dict],
        *,
        strategy_key: str = DEFAULT_ANONYMIZATION_STRATEGY,
    ) -> Dict[str, Dict[str, str]]:
        base_replacements: Dict[str, str] = {}
        alias_state = {
            "counters": {},
            "base_usage": {},
        }

        ordered_groups = sorted(groups.items(), key=lambda item: item[1]["first_start"])
        for group_key, group in ordered_groups:
            base_replacements[group_key] = self._generate_group_replacement(
                group=group,
                alias_state=alias_state,
                strategy_key=strategy_key,
            )
        collision_suffixes = {group_key: "" for group_key in base_replacements}
        replacements = dict(base_replacements)
        collision_groups: Dict[str, List[str]] = {}
        for group_key, replacement in base_replacements.items():
            collision_groups.setdefault(replacement, []).append(group_key)

        for replacement, group_keys in collision_groups.items():
            if len(group_keys) <= 1:
                continue
            ordered_group_keys = sorted(group_keys, key=lambda key: groups[key]["first_start"])
            for index, group_key in enumerate(ordered_group_keys):
                suffix = self._to_alpha(index)
                collision_suffixes[group_key] = suffix
                replacements[group_key] = f"{replacement}{suffix}"

        return {
            "base_replacements": base_replacements,
            "replacements": replacements,
            "collision_suffixes": collision_suffixes,
        }

    def _build_deterministic_group_replacements(
        self,
        groups: Dict[str, Dict],
        *,
        strategy_key: str = DEFAULT_ANONYMIZATION_STRATEGY,
    ) -> Dict[str, str]:
        bundle = self._build_deterministic_group_replacement_bundle(
            groups,
            strategy_key=strategy_key,
        )
        return bundle["replacements"]

    def _disambiguate_colliding_replacements(
        self,
        groups: Dict[str, Dict],
        replacements: Dict[str, str],
    ) -> Dict[str, str]:
        collision_groups: Dict[str, List[str]] = {}
        for group_key, replacement in replacements.items():
            collision_groups.setdefault(replacement, []).append(group_key)

        resolved = dict(replacements)
        for replacement, group_keys in collision_groups.items():
            if len(group_keys) <= 1:
                continue

            ordered_group_keys = sorted(group_keys, key=lambda key: groups[key]["first_start"])
            if all(
                str(groups[key]["primary_entity"].get("type", "")).upper() in self.HIGH_CONFIDENCE_IDENTIFIER_TYPES
                for key in ordered_group_keys
            ):
                continue
            for index, group_key in enumerate(ordered_group_keys):
                suffix = self._to_alpha(index)
                resolved[group_key] = f"{replacement}{suffix}"

        return resolved

    def _render_group_replacement(
        self,
        *,
        entity: Dict[str, Any],
        context: Dict[str, Any],
        group: Dict[str, Any],
        group_key: str,
        default_replacement: str,
        base_replacement: str,
        collision_suffix: str,
        strategy_key: str,
    ) -> tuple[str, Optional[str]]:
        strategy_profile = get_anonymization_strategy_profile(strategy_key)
        if strategy_profile.key != "official":
            return default_replacement, None

        family = self.GROUP_FAMILIES.get(str(entity.get("type", "")), "")
        source_text = str(entity.get("text", ""))
        role = str(context.get("role", "") or group.get("canonical_role", "") or "")
        label = str(context.get("label", "") or "")

        rendered = ""
        if family == "organization":
            rendered = self._build_surface_aware_organization_alias(
                source_text=source_text,
                role=role,
                label=label,
            )
        elif family == "bank":
            rendered = self._build_official_bank_alias(source_text)

        if not rendered:
            return default_replacement, None

        if collision_suffix:
            rendered = f"{rendered}{collision_suffix}"
        return rendered, f"surface::{group_key}"

    async def _build_llm_group_replacements(
        self,
        groups: Dict[str, Dict],
        fallback_replacements: Dict[str, str],
        llm_model: Optional[str] = None,
    ) -> Dict[str, str]:
        llm_candidates = [
            group
            for group in groups.values()
            if group["primary_entity"]["type"] in self.LLM_TYPES
        ]
        if not llm_candidates:
            return {}

        ollama = self._get_ollama_service(llm_model=llm_model)
        if ollama is None or not ollama.available:
            return {}

        prompt_items = []
        source_map: Dict[str, str] = {}

        for index, group in enumerate(llm_candidates, start=1):
            entity = group["primary_entity"]
            context = group["primary_context"]
            item_id = f"E{index}"
            source_map[item_id] = group["text"]
            prompt_items.append(
                {
                    "id": item_id,
                    "type": entity["type"],
                    "text": group["text"],
                    "label": context.get("label") or "",
                    "role": context.get("role") or "",
                    "line": context.get("line") or "",
                    "fallback": fallback_replacements.get(group["text"], ""),
                }
            )

        prompt = self._build_llm_prompt(prompt_items)

        try:
            payload = await asyncio.to_thread(ollama.generate_json, prompt, 1200)
            parsed = self._parse_llm_replacements(payload)
        except Exception as exc:
            logger.warning("Context-aware LLM anonymization failed: %s", exc)
            return {}

        updates: Dict[str, str] = {}
        used_replacements = set(fallback_replacements.values())

        for item in parsed:
            item_id = str(item.get("id", "")).strip()
            replacement = self._normalize_replacement_text(item.get("replacement"))
            source_text = source_map.get(item_id)
            if not source_text or not replacement:
                continue

            primary_entity = groups[source_text]["primary_entity"]
            if not self._is_valid_llm_replacement(
                entity_type=primary_entity["type"],
                source_text=source_text,
                replacement=replacement,
            ):
                continue

            if replacement in used_replacements and replacement != fallback_replacements.get(source_text):
                continue

            updates[source_text] = replacement
            used_replacements.add(replacement)

        return updates

    async def _build_llm_resolution_result(
        self,
        *,
        text: str,
        entities: List[Dict],
        contexts: List[Dict],
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt_items = []
        for index, (entity, context) in enumerate(zip(entities, contexts), start=1):
            prompt_items.append(
                {
                    "id": f"E{index}",
                    "type": entity["type"],
                    "text": entity["text"],
                    "label": context.get("label") or "",
                    "role_hint": context.get("role") or "",
                    "line": context.get("line") or "",
                }
            )

        focus_lines = self._collect_review_focus_lines(
            text=text,
            contexts=contexts,
            max_lines=self._get_review_focus_line_limit(llm_model, text_length=len(text)),
        )
        prompt = self._build_llm_resolution_prompt(
            text=text,
            prompt_items=prompt_items,
            focus_lines=focus_lines,
            llm_model=llm_model,
        )
        ollama = self._get_ollama_service(llm_model=llm_model)
        if ollama is None:
            return {}

        try:
            payload = await asyncio.to_thread(
                ollama.generate_json,
                prompt,
                self._get_resolution_num_predict(llm_model, text_length=len(text)),
            )
        except Exception as exc:
            logger.warning("LLM entity resolution failed: %s", exc)
            return {}

        parsed = self._extract_resolution_json(payload)
        return parsed if isinstance(parsed, dict) else {}

    def _build_llm_resolution_prompt(
        self,
        *,
        text: str,
        prompt_items: List[Dict[str, str]],
        focus_lines: List[str],
        llm_model: Optional[str] = None,
    ) -> str:
        runtime = get_runtime_llm_strategy_profile(
            llm_model or settings.get_default_llm_model(),
            text_length=len(text),
        )
        strategy = runtime.strategy
        items_json = json.dumps(prompt_items, ensure_ascii=False, indent=2)
        focus_json = json.dumps(focus_lines, ensure_ascii=False, indent=2)
        review_limit = self._get_review_text_limit(llm_model, text_length=len(text))
        review_text = text if len(text) <= review_limit else text[:review_limit]
        strategy_note = (
            "Strategy note: this document uses the local 4B stability strategy. Recover missed entities, "
            "abbreviation-linked mentions, footer/signature subjects, and full addresses, but keep every "
            "judgment tied to verbatim text and the supplied focus lines."
        )
        return f"""
You are reviewing entity recognition for a Chinese legal/business document.
The document may be a contract, application, enforcement paper, pleading, statement, or other formal filing.

Goals:
1. Decide which detected entities refer to the same real-world subject.
2. Assign a stable canonical_key so the same subject gets the same replacement everywhere.
3. Assign a canonical_role from this set only:
["PARTY_A","PARTY_B","PARTY_C","PROJECT","LOCATION","BANK","ACCOUNT","CONTACT","LEGAL_REPRESENTATIVE","PHONE","CONTRACT_NO","POSITION","ORGANIZATION","PERSON","OTHER"]
4. Find additional entity mentions that appear verbatim in the document text but are missing from the detected entity list, especially in dense narrative/factual paragraphs.
5. Prefer full official names and full addresses over shortened mentions when both appear verbatim.

Rules:
1. Use the same canonical_key for the same subject across full names, account names, repeated mentions, and footer/signature mentions.
2. If an entity clearly refers to 甲方, use canonical_key PARTY_A. If it refers to 乙方, use PARTY_B. If it refers to 丙方, use PARTY_C.
3. Recover missed entities from non-standard sections such as 住址/住所/住所地/身份证住址/通讯地址/送达地址, header blocks, footer blocks, signature blocks, account sections, court/institution lines, and dense narrative sections.
4. In narrative text, pay special attention to short repeated company mentions like “铁鑫公司/荔富公司”, and person names triggered by cues such as “股东”“付款至”“支付给”“个人银行账户”.
5. Extra entities must appear verbatim in the provided text.
6. Do not add AMOUNT or DATE as extra entities.
7. Do not invent abbreviations, aliases, or replacements.
8. Return strict JSON only in this exact format:
{{
  "entity_updates":[
    {{"id":"E1","canonical_key":"PARTY_A","canonical_role":"PARTY_A"}}
  ],
  "extra_entities":[
    {{"type":"LANDLINE_PHONE","text":"0763—3910858","canonical_key":"PHONE_1","canonical_role":"PHONE"}}
  ]
}}

Detected entities:
{items_json}

High-value focus lines:
{focus_json}

{strategy_note}

Contract text:
\"\"\"
{review_text}
\"\"\"
""".strip()

    def _collect_review_focus_lines(
        self,
        *,
        text: str,
        contexts: List[Dict],
        max_lines: int = 18,
    ) -> List[str]:
        focus_tokens = [
            "甲方",
            "乙方",
            "丙方",
            "委托方",
            "受托方",
            "发包人",
            "承包人",
            "收款单位",
            "法定代表人",
            "联系人",
            "开户行",
            "户名",
            "账户",
            "帐号",
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
            "人民法院",
            "检察院",
            "公安局",
        ]
        narrative_tokens = [
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
            "签字",
            "证据",
        ]
        organization_tokens = ["公司", "集团", "中心", "研究院", "事务所", "银行", "支行", "分行"]
        address_tokens = ["省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "号", "栋", "室", "广场"]

        ordered_lines: List[str] = []
        seen = set()

        def add_line(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            ordered_lines.append(normalized)

        for context in contexts:
            line = str(context.get("line") or "").strip()
            if line:
                add_line(line)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(line) < 4:
                continue
            if any(token in line for token in focus_tokens):
                add_line(line)
                continue
            if any(token in line for token in narrative_tokens):
                add_line(line)
                continue
            if any(token in line for token in organization_tokens) and any(token in line for token in ["股东", "付款", "支付", "转给", "汇", "账户"]):
                add_line(line)
                continue
            if re.search(r"[\u4e00-\u9fa5·某]{2,5}(?:个人银行账户|个人账户|银行账户)", line):
                add_line(line)
                continue
            if sum(token in line for token in address_tokens) >= 2 and len(line) <= 120:
                add_line(line)

        return ordered_lines[:max_lines]

    def _get_review_text_limit(self, llm_model: Optional[str], text_length: int = 0) -> int:
        runtime = get_runtime_llm_strategy_profile(
            llm_model or settings.get_default_llm_model(),
            text_length=text_length,
        )
        return runtime.strategy.review_text_limit

    def _get_resolution_num_predict(self, llm_model: Optional[str], text_length: int = 0) -> int:
        runtime = get_runtime_llm_strategy_profile(
            llm_model or settings.get_default_llm_model(),
            text_length=text_length,
        )
        return runtime.strategy.resolution_num_predict

    def _get_review_focus_line_limit(self, llm_model: Optional[str], text_length: int = 0) -> int:
        runtime = get_runtime_llm_strategy_profile(
            llm_model or settings.get_default_llm_model(),
            text_length=text_length,
        )
        return runtime.strategy.focus_line_limit

    def _parse_model_size(self, llm_model: Optional[str]) -> float:
        return parse_model_size_in_b(llm_model or settings.get_default_llm_model())

    def _extract_resolution_json(self, payload: Dict[str, Any]) -> Any:
        candidates: List[Any] = [payload]
        if isinstance(payload, dict):
            for key in ("response", "thinking", "output", "text"):
                value = payload.get(key)
                if value:
                    candidates.append(value)
            message = payload.get("message")
            if isinstance(message, dict):
                candidates.append(message)
                for key in ("content", "thinking"):
                    value = message.get(key)
                    if value:
                        candidates.append(value)

        for candidate in candidates:
            parsed = self._coerce_json(candidate)
            if isinstance(parsed, dict) and (
                isinstance(parsed.get("entity_updates"), list) or isinstance(parsed.get("extra_entities"), list)
            ):
                return parsed
        return {}

    def _sanitize_canonical_key(self, value: Any) -> str:
        if value is None:
            return ""
        cleaned = re.sub(r"[^A-Z0-9_]", "", str(value).strip().upper())
        if len(cleaned) < 3:
            return ""
        return cleaned[:48]

    def _materialize_llm_extra_entities(
        self,
        *,
        text: str,
        existing_entities: List[Dict],
        extra_items: Any,
    ) -> List[Dict]:
        if not isinstance(extra_items, list):
            return []

        seen = {
            (entity["type"], entity["start"], entity["end"], entity["text"])
            for entity in existing_entities
        }
        occupied = [(entity["start"], entity["end"]) for entity in existing_entities]
        extra_entities: List[Dict] = []

        for item in extra_items:
            if not isinstance(item, dict):
                continue

            entity_type = str(item.get("type", "")).strip().upper()
            entity_text = str(item.get("text", "")).strip()
            if entity_type not in self.REVIEW_ENTITY_TYPES or not entity_text:
                continue
            if not self._looks_like_valid_review_entity(entity_type, entity_text):
                continue

            canonical_key = self._sanitize_canonical_key(item.get("canonical_key"))
            canonical_role = str(item.get("canonical_role", "")).strip().upper()

            for start, end in self._find_text_spans(text, entity_text):
                key = (entity_type, start, end, entity_text)
                if key in seen:
                    continue
                if any(start < existing_end and end > existing_start for existing_start, existing_end in occupied):
                    continue

                entity: Dict[str, Any] = {
                    "type": entity_type,
                    "text": entity_text,
                    "start": start,
                    "end": end,
                    "score": 0.78,
                    "source": "llm_review",
                    "metadata": {"review": True},
                }
                if canonical_key:
                    entity["canonical_key"] = canonical_key
                if canonical_role in self.RESOLUTION_ROLES:
                    entity["canonical_role"] = canonical_role

                extra_entities.append(entity)
                seen.add(key)
                occupied.append((start, end))

        return extra_entities

    def _looks_like_valid_review_entity(self, entity_type: str, entity_text: str) -> bool:
        normalized = entity_text.strip()
        if len(normalized) < 2:
            return False
        if entity_type in {"PERSON", "PERSON_NAME"}:
            return self._looks_like_person_name(normalized)
        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            return any("\u4e00" <= char <= "\u9fff" for char in normalized)
        if entity_type == "BANK_NAME":
            return "银行" in normalized
        if entity_type == "LOCATION":
            return any(
                token in normalized
                for token in ["省", "市", "区", "县", "镇", "路", "街", "村", "号", "栋", "室", "广场", "法院", "检察院"]
            )
        if entity_type == "PROJECT":
            return any(token in normalized for token in ["项目", "工程", "标段"])
        if entity_type == "CONTRACT_NO":
            return len(normalized) >= 4
        if entity_type in {"CN_PHONE", "LANDLINE_PHONE"}:
            return re.search(r"\d", normalized) is not None
        if entity_type in {"CN_BANK_CARD", "CN_ID_CARD", "CN_CREDIT_CODE"}:
            return re.search(r"[A-Z0-9]", normalized, re.I) is not None
        if entity_type == "EMAIL_ADDRESS":
            return "@" in normalized
        return True

    def _find_exact_spans(self, text: str, target: str) -> List[int]:
        positions: List[int] = []
        start = 0
        while target:
            index = text.find(target, start)
            if index == -1:
                break
            positions.append(index)
            start = index + len(target)
        return positions

    def _find_text_spans(self, text: str, target: str) -> List[tuple[int, int]]:
        exact_positions = [(start, start + len(target)) for start in self._find_exact_spans(text, target)]
        if exact_positions:
            return exact_positions

        normalized_target = re.sub(r"\s+", "", target)
        if not normalized_target:
            return []

        normalized_chars: List[str] = []
        index_map: List[int] = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            normalized_chars.append(char)
            index_map.append(index)

        normalized_text = "".join(normalized_chars)
        positions: List[tuple[int, int]] = []
        search_from = 0

        while True:
            normalized_start = normalized_text.find(normalized_target, search_from)
            if normalized_start == -1:
                break

            normalized_end = normalized_start + len(normalized_target) - 1
            start = index_map[normalized_start]
            end = index_map[normalized_end] + 1
            positions.append((start, end))
            search_from = normalized_start + len(normalized_target)

        return positions

    def _build_llm_prompt(self, prompt_items: List[Dict[str, str]]) -> str:
        items_json = json.dumps(prompt_items, ensure_ascii=False, indent=2)
        return f"""
You are anonymizing a Chinese business contract.
For each entity below, generate a readable fictional replacement that preserves the business meaning and role in context.

Rules:
1. Use realistic but fictional Chinese names, institutions, project names, addresses, or bank names.
2. Never output placeholders like [机构], [金额], [项目名称], [姓名].
3. Do not reuse the original text.
4. Keep the replacement concise and suitable for contracts.
5. Keep organization-like entities organization-like, person-like entities person-like, and location-like entities location-like.
6. Return strict JSON only in this format:
{{"replacements":[{{"id":"E1","replacement":"华宸新能源有限公司"}}]}}

Entities:
{items_json}
""".strip()

    def _parse_llm_replacements(self, payload: Dict[str, Any]) -> List[Dict]:
        candidates: List[Any] = [payload]
        if isinstance(payload, dict):
            for key in ("response", "thinking", "output", "text"):
                value = payload.get(key)
                if value:
                    candidates.append(value)
            message = payload.get("message")
            if isinstance(message, dict):
                candidates.append(message)
                for key in ("content", "thinking"):
                    value = message.get(key)
                    if value:
                        candidates.append(value)

        for candidate in candidates:
            parsed = self._coerce_json(candidate)
            if isinstance(parsed, dict):
                replacements = parsed.get("replacements")
                if isinstance(replacements, list):
                    return [item for item in replacements if isinstance(item, dict)]
        return []

    def _coerce_json(self, candidate: Any) -> Any:
        if isinstance(candidate, (dict, list)):
            return candidate
        if not isinstance(candidate, str):
            return None

        snippets = [candidate.strip()]
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
        if fence_match:
            snippets.append(fence_match.group(1).strip())
        first_brace = candidate.find("{")
        last_brace = candidate.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            snippets.append(candidate[first_brace : last_brace + 1])

        for snippet in snippets:
            if not snippet:
                continue
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
        return None

    def _is_valid_llm_replacement(
        self,
        *,
        entity_type: str,
        source_text: str,
        replacement: str,
    ) -> bool:
        if replacement == source_text:
            return False
        if self._has_meaningful_overlap(source_text, replacement):
            return False
        if any(token in replacement for token in ["[", "]", "占位", "机构", "项目名称", "金额"]):
            return False
        if entity_type in {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}:
            return any(token in replacement for token in ["公司", "中心", "集团", "研究院", "所", "院"])
        if entity_type == "BANK_NAME":
            return "银行" in replacement
        if entity_type in {"PERSON", "PERSON_NAME"}:
            return re.fullmatch(r"[\u4e00-\u9fa5\u00b7]{2,8}", replacement) is not None
        if entity_type == "LOCATION":
            return any(token in replacement for token in ["省", "市", "区", "县", "镇", "路"])
        return True

    def _has_meaningful_overlap(self, source_text: str, replacement: str) -> bool:
        source = re.sub(r"\s+", "", source_text)
        target = re.sub(r"\s+", "", replacement)
        if len(source) < 3 or len(target) < 3:
            return False

        ignore_tokens = {
            "有限公司",
            "股份有限公司",
            "有限责任公司",
            "服务中心",
            "技术中心",
            "研究院",
            "银行",
            "支行",
            "项目",
            "工程",
            "服务",
            "技术",
            "检测",
            "风电",
            "公司",
            "中心",
        }

        for size in range(min(6, len(source)), 2, -1):
            for index in range(0, len(source) - size + 1):
                token = source[index : index + size]
                if token in ignore_tokens:
                    continue
                if token in target:
                    return True
        return False

    def _build_context(self, text: str, entity: Dict) -> Dict:
        start = entity["start"]
        end = entity["end"]
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)

        line = text[line_start:line_end]
        previous_line = self._neighbor_line(text, line_start - 1, reverse=True)
        next_line = self._neighbor_line(text, line_end + 1, reverse=False)
        local_start = start - line_start
        local_end = end - line_start
        before = line[:local_start].strip()
        after = line[local_end:].strip()
        label = self._extract_label(before, previous_line)
        role = self._infer_role(entity["type"], before, after, label, previous_line, next_line)
        window_text = " ".join(part for part in [previous_line, line.strip(), next_line] if part)

        return {
            "line_start": line_start,
            "line_end": line_end,
            "line": line.strip(),
            "before": before[-40:],
            "after": after[:40],
            "previous_line": previous_line,
            "next_line": next_line,
            "window_text": window_text,
            "label": label,
            "role": role,
            "canonical_role": entity.get("canonical_role", ""),
        }

    def _neighbor_line(self, text: str, anchor: int, reverse: bool) -> str:
        if not text:
            return ""

        if reverse:
            if anchor < 0:
                return ""
            end = anchor + 1
            start = text.rfind("\n", 0, max(0, anchor)) + 1
            return text[start:end].strip()

        if anchor >= len(text):
            return ""
        end = text.find("\n", anchor)
        if end == -1:
            end = len(text)
        return text[anchor:end].strip()

    def _extract_label(self, *candidates: str) -> str:
        for candidate in candidates:
            normalized = candidate.replace("\u3000", " ").strip()
            if not normalized:
                continue

            best_label = ""
            best_index = -1
            for label in self.ROLE_LABELS:
                index = normalized.rfind(label)
                if index > best_index:
                    best_label = label
                    best_index = index
            if best_label:
                return best_label

            match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9（）()]+)\s*[:：]\s*$", normalized)
            if match:
                return match.group(1)

            if normalized in self.ROLE_LABELS:
                return normalized
        return ""

    def _infer_role(
        self,
        entity_type: str,
        before: str,
        after: str,
        label: str,
        previous_line: str,
        next_line: str,
    ) -> str:
        core_text = f"{before} {label} {after}"
        window_text = f"{previous_line} {core_text} {next_line}"
        nearest_role = self._infer_party_role_from_before(before)

        if label in {"工程地址", "项目地址"}:
            return "项目地址"
        if label in {"住址", "身份证住址", "住所", "住所地", "通讯地址", "送达地址"}:
            return label
        if entity_type == "LOCATION":
            return "地址"
        if label == "开户行" or entity_type == "BANK_NAME":
            return "开户行"
        if label in {"户名", "账户", "帐户", "账号", "帐号"} or entity_type == "ACCOUNT_NAME":
            return "户名"
        if entity_type == "PROJECT":
            if "检测" in window_text:
                return "检测项目"
            if "风电" in window_text:
                return "风电项目"

        if nearest_role:
            return nearest_role
        if label in {"甲方", "乙方", "丙方"}:
            return label
        if any(token in core_text for token in ["甲方", "委托方", "发包人", "采购人"]):
            return "甲方"
        if any(token in core_text for token in ["乙方", "受托方", "承包人", "供应商"]):
            return "乙方"
        if "收款单位" in core_text:
            return "收款"
        if entity_type in {"PERSON", "PERSON_NAME"}:
            if "法定代表人" in core_text:
                return "法代"
            if "联系人" in core_text:
                return "联系人"
            if "项目负责人" in core_text:
                return "项目负责人"
        return ""

    def _infer_party_role_from_before(self, before: str) -> str:
        mapping = {
            "甲方": "甲方",
            "乙方": "乙方",
            "丙方": "丙方",
            "委托方": "甲方",
            "发包人": "甲方",
            "采购人": "甲方",
            "受托方": "乙方",
            "承包人": "乙方",
            "供应商": "乙方",
        }
        best_role = ""
        best_index = -1

        for token, role in mapping.items():
            pattern = re.compile(rf"{re.escape(token)}(?:（[^）]*）|\([^)]*\))?\s*[:：]?\s*$")
            match = pattern.search(before)
            if match and match.start() >= best_index:
                best_role = role
                best_index = match.start()

        return best_role

    def _entity_priority(self, entity: Dict) -> int:
        return self.TYPE_PRIORITY.get(entity["type"], 99)

    def _source_layer(self, entity: Dict) -> str:
        source = str(entity.get("source", "")).lower()
        metadata = entity.get("metadata") or {}
        if any(token in source for token in ["regex", "contract", "custom"]):
            return "deterministic_rule"
        if "structured" in source:
            return "structured_label"
        if "definition" in source or metadata.get("definition_alias") or metadata.get("definition_full_text"):
            return "definition_anchor"
        if source == "llm_review" or metadata.get("review"):
            return "llm_review"
        if "ollama" in source or source == "llm":
            return "llm_semantic"
        if "prose" in source or "propagate" in source:
            return "propagated"
        if "residual" in source:
            return "residual"
        return "unknown"

    def _source_priority(self, entity: Dict) -> int:
        layer = self._source_layer(entity)
        priority = self.SOURCE_LAYER_PRIORITY.get(layer, 9)
        if entity["type"] in self.HIGH_CONFIDENCE_IDENTIFIER_TYPES and layer == "deterministic_rule":
            return 0
        if entity["type"] == "CONTRACT_NO" and layer in {"deterministic_rule", "structured_label"}:
            return 0
        return priority

    def _context_specificity(self, context: Dict) -> int:
        score = 0
        if context.get("label"):
            score += 2
        if context.get("role"):
            score += 2
        if context.get("previous_line") or context.get("next_line"):
            score += 1
        return score

    def _resolve_group_key(
        self,
        index: int,
        entities: List[Dict],
        contexts: List[Dict],
    ) -> str:
        entity = entities[index]
        context = contexts[index]
        canonical_key = self._canonical_key_from_entity(entity)
        if canonical_key:
            return canonical_key

        best_key = entity["text"]
        best_norm = self._normalize_group_text(entity["text"])
        best_start = entity["start"]

        for other_index, (other_entity, other_context) in enumerate(zip(entities, contexts)):
            if other_index == index:
                continue
            if self._should_share_identity(entity, context, other_entity, other_context):
                other_canonical_key = self._canonical_key_from_entity(other_entity)
                if other_canonical_key:
                    return other_canonical_key
                other_norm = self._normalize_group_text(other_entity["text"])
                if len(other_norm) > len(best_norm) or (
                    len(other_norm) == len(best_norm) and other_entity["start"] < best_start
                ):
                    best_key = other_entity["text"]
                    best_norm = other_norm
                    best_start = other_entity["start"]

        return best_key

    def _should_share_identity(
        self,
        entity: Dict,
        context: Dict,
        other_entity: Dict,
        other_context: Dict,
    ) -> bool:
        canonical_key = self._canonical_key_from_entity(entity)
        other_canonical_key = self._canonical_key_from_entity(other_entity)
        if canonical_key and other_canonical_key:
            return canonical_key == other_canonical_key

        family = self.GROUP_FAMILIES.get(entity["type"])
        other_family = self.GROUP_FAMILIES.get(other_entity["type"])
        if family != other_family:
            if {
                self.GROUP_FAMILIES.get(entity["type"]),
                self.GROUP_FAMILIES.get(other_entity["type"]),
            } != {"organization"}:
                return False

        text = self._normalize_group_text(entity["text"])
        other_text = self._normalize_group_text(other_entity["text"])
        if not text or not other_text:
            return False
        if text == other_text:
            return True

        role = self._party_role_from_context(context)
        other_role = self._party_role_from_context(other_context)
        if role and other_role and role != other_role:
            return False

        if family == "person":
            return self._share_person_identity(text, other_text)

        if family == "organization":
            if not self._is_valid_organization_variant(entity["text"], primary_text=other_entity["text"]):
                return False
            if not self._is_valid_organization_variant(other_entity["text"], primary_text=entity["text"]):
                return False

        if family == "organization" and self._share_organization_identity(entity["text"], other_entity["text"]):
            return True

        shorter, longer = (text, other_text) if len(text) <= len(other_text) else (other_text, text)
        if len(shorter) < 4 or shorter not in longer:
            return False

        if family == "organization":
            return self._looks_like_organization_name(entity["text"]) and self._looks_like_organization_name(other_entity["text"])
        if family == "bank":
            return "银行" in entity["text"] and "银行" in other_entity["text"]
        if family == "project":
            return any(token in longer for token in ["项目", "工程", "标段"])
        return False

    def _canonical_key_from_entity(self, entity: Dict) -> str:
        canonical_key = self._sanitize_canonical_key(entity.get("canonical_key"))
        if canonical_key:
            return canonical_key
        metadata = entity.get("metadata") or {}
        return self._sanitize_canonical_key(metadata.get("canonical_key"))

    def _share_person_identity(self, left: str, right: str) -> bool:
        if len(left) != len(right):
            return False

        for left_char, right_char in zip(left, right):
            if left_char == right_char:
                continue
            if "某" in {left_char, right_char}:
                continue
            return False
        return True

    def _share_organization_identity(self, left: str, right: str) -> bool:
        left_norm = self._normalize_group_text(left)
        right_norm = self._normalize_group_text(right)
        if not left_norm or not right_norm:
            return False

        left_aliases = self._derive_organization_identity_aliases(left)
        right_aliases = self._derive_organization_identity_aliases(right)
        return right_norm in left_aliases or left_norm in right_aliases

    def _derive_organization_identity_aliases(self, text: str) -> set[str]:
        normalized = self._normalize_group_text(text)
        if len(normalized) < 4:
            return (
                {normalized}
                if normalized and self._is_valid_organization_variant(normalized, primary_text=normalized)
                else set()
            )

        aliases = {normalized}
        region_stripped = re.sub(
            r"^(?:[\u4e00-\u9fa5]{2,9}(?:省|市|区|县|镇|乡|街道))+",
            "",
            normalized,
        )
        core = self._strip_organization_suffix(region_stripped)
        brand = re.sub(
            r"(科技|工程|建设|贸易|实业|发展|咨询|服务|管理|材料|电力|能源|建筑|环保|智能|信息|网络|电子|机械|设备|制造)$",
            "",
            core,
        )
        compact_brand = re.sub(
            r"^(?:北京|上海|广州|深圳|天津|重庆|杭州|南京|苏州|成都|武汉|西安|长沙|郑州|青岛|宁波|佛山|东莞|厦门|福州|济南|合肥|昆明|南宁|贵阳|南昌|海口|太原|沈阳|长春|哈尔滨|石家庄|呼和浩特|乌鲁木齐|拉萨|银川|西宁)",
            "",
            brand,
        )

        for candidate in [region_stripped]:
            candidate = self._normalize_group_text(candidate)
            if len(candidate) >= 2:
                aliases.add(candidate)
        for candidate in [core, brand, compact_brand]:
            candidate = self._normalize_group_text(candidate)
            if 2 <= len(candidate) <= 12:
                aliases.add(candidate)
                aliases.add(self._normalize_group_text(f"{candidate}公司"))

        if self._looks_like_company_subject(normalized):
            tail_source = self._normalize_group_text(compact_brand or brand or core)
            for size in (2, 3, 4):
                if len(tail_source) <= size:
                    continue
                candidate = tail_source[-size:]
                if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", candidate):
                    aliases.add(candidate)
                    aliases.add(self._normalize_group_text(f"{candidate}公司"))

        return {
            alias
            for alias in aliases
            if self._is_valid_organization_variant(alias, primary_text=normalized)
            and not (
                self._normalize_group_text(alias) != normalized
                and self._is_low_information_organization_alias(alias)
            )
        }

    def _normalize_group_text(self, text: str) -> str:
        return re.sub(r"[\s（）()【】\[\]<>《》\-－—–]", "", text)

    def _party_role_from_context(self, context: Dict) -> str:
        canonical_role = str(context.get("canonical_role", "")).strip().upper()
        if canonical_role == "PARTY_A":
            return "甲方"
        if canonical_role == "PARTY_B":
            return "乙方"
        if canonical_role == "PARTY_C":
            return "丙方"

        role = context.get("role", "")
        if role in {"甲方", "乙方", "丙方"}:
            return role
        label = context.get("label", "")
        if label in {"甲方", "乙方", "丙方"}:
            return label
        window_text = context.get("window_text", "")
        for token in ["甲方", "乙方", "丙方"]:
            if token in window_text:
                return token
        return ""

    def _pick_group_role(self, group: Dict) -> str:
        canonical_roles = [
            str(entity.get("canonical_role", "")).strip().upper()
            for entity in group["entities"]
            if entity.get("canonical_role")
        ]
        for canonical_role in canonical_roles:
            if canonical_role == "PARTY_A":
                return "甲方"
            if canonical_role == "PARTY_B":
                return "乙方"
            if canonical_role == "PARTY_C":
                return "丙方"
            if canonical_role == "PROJECT":
                return "项目"
            if canonical_role == "LOCATION":
                return "项目地址"
            if canonical_role == "BANK":
                return "开户行"
            if canonical_role == "ACCOUNT":
                return "户名"
            if canonical_role == "CONTACT":
                return "联系人"
            if canonical_role == "LEGAL_REPRESENTATIVE":
                return "法代"
            if canonical_role == "PHONE":
                return "联系电话"

        preferred_roles = [
            "甲方",
            "乙方",
            "丙方",
            "收款",
            "法代",
            "联系人",
            "项目负责人",
            "开户行",
            "户名",
            "项目地址",
            "住址",
            "身份证住址",
            "住所",
            "住所地",
            "通讯地址",
            "送达地址",
            "检测项目",
            "风电项目",
        ]
        roles = [context.get("role", "") for context in group["contexts"] if context.get("role")]
        for preferred in preferred_roles:
            if preferred in roles:
                return preferred
        for context in group["contexts"]:
            party_role = self._party_role_from_context(context)
            if party_role:
                return party_role
        return roles[0] if roles else ""

    def _pick_group_label(self, group: Dict) -> str:
        preferred_labels = [
            "甲方",
            "乙方",
            "丙方",
            "收款单位",
            "法定代表人",
            "联系人",
            "项目负责人",
            "开户行",
            "户名",
            "账户",
            "帐户",
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
            "合同编号",
        ]
        labels = [context.get("label", "") for context in group["contexts"] if context.get("label")]
        for preferred in preferred_labels:
            if preferred in labels:
                return preferred
        return labels[0] if labels else ""

    def _generate_group_replacement(
        self,
        *,
        group: Dict,
        alias_state: Dict[str, Dict[str, int]],
        strategy_key: str = DEFAULT_ANONYMIZATION_STRATEGY,
    ) -> str:
        entity_type = group["primary_entity"]["type"]
        source_text = group["text"]
        group_role = self._pick_group_role(group)
        group_label = self._pick_group_label(group)
        family = self.GROUP_FAMILIES.get(entity_type, entity_type.lower())
        strategy_profile = get_anonymization_strategy_profile(strategy_key)

        if family == "person":
            if strategy_profile.key == "serial_roles":
                return self._next_scoped_alias("人员", alias_state, style="serial", always_append=True)
            return self._build_person_alias(source_text)
        if family == "organization":
            return self._build_organization_alias(
                group_role,
                group_label,
                source_text,
                alias_state,
                strategy_key=strategy_profile.key,
            )
        if family == "bank":
            if strategy_profile.key == "official":
                return self._build_official_bank_alias(source_text)
            return self._build_role_alias(
                group_role,
                "开户行",
                alias_state,
                always_append=group_role not in {"甲方", "乙方", "丙方"},
            )
        if family == "project":
            return self._next_letter_alias("项目", alias_state)
        if family == "location":
            if strategy_profile.key == "official" and self._should_preserve_location_verbatim(source_text):
                return source_text
            return self._next_letter_alias("地区", alias_state)
        if family == "position":
            return source_text
        if family == "contract_no":
            return self._mask_identifier(
                source_text,
                metadata=group["primary_entity"].get("metadata") or {},
            )
        if family == "project_code":
            return self._next_letter_alias("项目代号", alias_state)
        if family == "product":
            return self._next_letter_alias("产品", alias_state)
        if family == "term":
            return self._next_letter_alias("术语", alias_state)
        if family == "phone":
            return self._mask_identifier(source_text)
        if family == "bank_card":
            return self._mask_identifier(source_text)
        if family == "id_card":
            return self._mask_identifier(source_text)
        if family == "credit_code":
            return self._mask_identifier(source_text)
        if family == "email":
            return self._next_letter_alias("邮箱", alias_state)
        return source_text

    def _build_person_alias(
        self,
        source_text: str,
    ) -> str:
        normalized = self._normalize_person_source_text(source_text)
        if len(normalized) <= 1:
            return "某"
        if len(normalized) == 2:
            return f"{normalized[0]}某"
        return f"{normalized[0]}某{normalized[-1]}"

    def _build_organization_alias(
        self,
        role: str,
        label: str,
        source_text: str,
        alias_state: Dict[str, Dict[str, int]],
        *,
        strategy_key: str = DEFAULT_ANONYMIZATION_STRATEGY,
    ) -> str:
        strategy_profile = get_anonymization_strategy_profile(strategy_key)
        if self._looks_like_person_name(source_text):
            if strategy_profile.key == "serial_roles":
                return self._next_scoped_alias("人员", alias_state, style="serial", always_append=True)
            return self._build_person_alias(source_text)

        if strategy_profile.key == "serial_roles":
            party_role = role if role in {"甲方", "乙方", "丙方"} else ""
            if party_role:
                return party_role
            if role == "收款" or label == "收款单位":
                return self._next_scoped_alias("收款单位", alias_state, style="serial", always_append=True)
            return self._next_scoped_alias("单位", alias_state, style="serial", always_append=True)

        readable_alias = self._build_surface_aware_organization_alias(
            source_text=source_text,
            role=role,
            label=label,
        )
        if readable_alias:
            return readable_alias

        party_role = role if role in {"甲方", "乙方", "丙方"} else ""
        if party_role:
            return self._next_scoped_alias(f"{party_role}单位", alias_state, style="letter", always_append=False)
        if role == "收款" or label == "收款单位":
            return self._next_letter_alias("收款单位", alias_state)
        return self._next_letter_alias("单位", alias_state)

    def _build_role_alias(
        self,
        role: str,
        noun: str,
        alias_state: Dict[str, Dict[str, int]],
        *,
        always_append: bool,
    ) -> str:
        party_role = role if role in {"甲方", "乙方", "丙方"} else ""
        prefix = f"{party_role}{noun}" if party_role else noun
        return self._next_scoped_alias(prefix, alias_state, style="letter", always_append=always_append)

    def _looks_like_organization_name(self, text: str) -> bool:
        return any(token in text for token in ["公司", "中心", "集团", "银行", "研究院", "事务所", "支行", "分行", "院", "局", "所"])

    def _looks_like_person_name(self, text: str) -> bool:
        return re.fullmatch(r"[\u4e00-\u9fa5\u00b7]{2,8}", text) is not None

    def _normalize_person_source_text(self, text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fa5\u00b7]", "", text).replace("·", "")

    def _looks_like_company_subject(self, text: str) -> bool:
        return any(
            token in text
            for token in [
                "公司",
                "有限公司",
                "有限责任公司",
                "股份有限公司",
                "集团",
                "集团有限公司",
            ]
        )

    def _build_official_company_alias(self, source_text: str) -> str:
        prefix, core, suffix = self._split_company_name_parts(source_text)
        masked_core = self._mask_company_core(core)
        if masked_core:
            return f"{prefix}{masked_core}{suffix}"
        leading = self._extract_leading_cjk_char(core or source_text)
        if suffix:
            return f"{prefix}{leading or '某'}某{suffix}"
        return f"{prefix}{leading or '某'}某"

    def _build_surface_aware_organization_alias(
        self,
        *,
        source_text: str,
        role: str,
        label: str,
    ) -> str:
        normalized = re.sub(r"\s+", "", source_text)
        if not normalized:
            return ""
        if self._looks_like_company_subject(normalized):
            return self._build_official_company_alias(normalized)
        if self._looks_like_bank_subject(normalized):
            return self._build_official_bank_alias(normalized)
        if "人民法院" in normalized or normalized.endswith("法院"):
            return self._build_hierarchical_judicial_alias(
                normalized,
                ("高级人民法院", "中级人民法院", "基层人民法院", "人民法院", "法院"),
            )
        if "人民检察院" in normalized or normalized.endswith("检察院"):
            return self._build_hierarchical_judicial_alias(
                normalized,
                ("人民检察院", "检察院"),
            )
        if normalized.endswith("公安局"):
            return self._build_region_terminal_alias(normalized, ("公安局",), "某公安局")
        if normalized.endswith("仲裁委员会"):
            return self._build_region_terminal_alias(normalized, ("仲裁委员会",), "某仲裁委")

        generic_alias = self._build_generic_institution_alias(normalized)
        if generic_alias:
            return generic_alias

        if role in {"甲方", "乙方", "丙方"}:
            return f"{role}单位"
        if role == "收款" or label == "收款单位":
            return "某收款单位"
        return ""

    def _looks_like_bank_subject(self, text: str) -> bool:
        return any(token in text for token in ["银行", "支行", "分行", "营业部", "分理处", "信用社", "联社"])

    def _build_official_bank_alias(self, source_text: str) -> str:
        normalized = re.sub(r"\s+", "", source_text)
        if not normalized:
            return ""
        if "银行" in normalized:
            index = normalized.find("银行")
            core = normalized[:index]
            suffix = normalized[index:]
            masked_core = self._mask_company_core(core) if core else "某"
            return f"{masked_core}{suffix}"

        for suffix in ["支行", "分行", "营业部", "分理处", "信用社", "联社"]:
            if not normalized.endswith(suffix):
                continue
            base = normalized[: -len(suffix)]
            prefix = self._extract_region_prefix(base)
            core = base[len(prefix) :] or base
            masked_core = self._mask_company_core(core) if core else "某"
            return f"{prefix}{masked_core}{suffix}"

        masked = self._mask_company_core(normalized)
        return masked or "某银行"

    def _build_region_terminal_alias(
        self,
        text: str,
        tokens: tuple[str, ...],
        alias_tail: str,
    ) -> str:
        for token in tokens:
            index = text.rfind(token)
            if index == -1:
                continue
            region = self._extract_region_prefix(text[:index])
            return f"{region}{alias_tail}" if region else alias_tail
        return alias_tail

    def _build_hierarchical_judicial_alias(
        self,
        text: str,
        tokens: tuple[str, ...],
    ) -> str:
        matched_token = ""
        matched_index = -1
        for token in tokens:
            index = text.rfind(token)
            if index == -1:
                continue
            if index > matched_index or (index == matched_index and len(token) > len(matched_token)):
                matched_token = token
                matched_index = index
        if matched_index == -1:
            return text

        prefix = text[:matched_index]
        suffix = text[matched_index:]
        segments = self._split_location_segments(prefix)
        if len(segments) <= 1:
            return text

        target_index = -1
        city_seen = False
        for index, segment in enumerate(segments):
            level = self._classify_location_segment(segment)
            if level in {"province", "city"}:
                city_seen = True
                continue
            if city_seen and level == "local":
                target_index = index

        if target_index == -1:
            return text

        masked_segments = list(segments)
        masked_segments[target_index] = self._mask_administrative_segment(masked_segments[target_index])
        rebuilt_prefix = "".join(masked_segments) + prefix[len("".join(segments)) :]
        return f"{rebuilt_prefix}{suffix}"

    def _build_generic_institution_alias(self, source_text: str) -> str:
        for suffix in [
            "事务所",
            "研究院",
            "研究所",
            "服务中心",
            "中心",
            "委员会",
            "管理局",
            "学院",
            "医院",
            "单位",
            "局",
            "院",
            "所",
        ]:
            if not source_text.endswith(suffix):
                continue
            base = source_text[: -len(suffix)]
            prefix = self._extract_region_prefix(base)
            core = base[len(prefix) :] or base
            if not core:
                return f"{prefix}某{suffix}" if prefix else f"某{suffix}"
            masked_core = self._mask_company_core(core)
            if not masked_core:
                masked_core = "某"
            return f"{prefix}{masked_core}{suffix}"
        return ""

    def _split_location_segments(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", "", text)
        segments: List[str] = []
        remaining = normalized
        while remaining:
            matched = False
            for suffix_token in self.LOCATION_SEGMENT_SUFFIXES:
                pattern = rf"^(?P<segment>[\u4e00-\u9fa5]{{1,12}}{re.escape(suffix_token)})"
                match = re.match(pattern, remaining)
                if not match:
                    continue
                segment = match.group("segment")
                segments.append(segment)
                remaining = remaining[len(segment) :]
                matched = True
                break
            if not matched:
                break
        return segments

    def _classify_location_segment(self, segment: str) -> str:
        if segment.endswith(("省", "自治区")):
            return "province"
        if segment.endswith(("市", "自治州", "地区")):
            return "city"
        if segment.endswith(("区", "县", "旗", "乡", "镇", "街道", "新区", "开发区", "园区", "高新区")):
            return "local"
        return "other"

    def _mask_administrative_segment(self, segment: str) -> str:
        for suffix_token in self.LOCATION_SEGMENT_SUFFIXES:
            if not segment.endswith(suffix_token):
                continue
            base = segment[: -len(suffix_token)]
            if len(base) >= 2:
                return f"{base[0]}*{base[2:]}{suffix_token}"
            if len(base) == 1:
                return f"*{suffix_token}"
            return segment
        return segment

    def _extract_region_prefix(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        prefix = ""
        remaining = normalized
        while remaining:
            matched = False
            for suffix_token in self.LOCATION_SEGMENT_SUFFIXES:
                pattern = rf"^(?P<segment>[\u4e00-\u9fa5]{{2,12}}{re.escape(suffix_token)})"
                match = re.match(pattern, remaining)
                if not match:
                    continue
                segment = match.group("segment")
                prefix += segment
                remaining = remaining[len(segment) :]
                matched = True
                break
            if not matched:
                break
        return prefix

    def _extract_unsuffixed_region_prefix(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text)
        if len(normalized) < 4:
            return ""
        for candidate in sorted(self.UNSUFFIXED_REGION_PREFIXES, key=len, reverse=True):
            if normalized.startswith(candidate) and len(normalized) > len(candidate) + 1:
                return candidate
        return ""

    def _split_company_name_parts(self, source_text: str) -> tuple[str, str, str]:
        normalized = re.sub(r"\s+", "", source_text)
        suffix = ""
        for candidate in self.COMPANY_SUFFIXES:
            if normalized.endswith(candidate):
                suffix = candidate
                normalized = normalized[: -len(candidate)]
                break

        prefix = self._extract_region_prefix(normalized)
        if not prefix:
            prefix = self._extract_unsuffixed_region_prefix(normalized)
        remaining = normalized[len(prefix) :]
        core = remaining or normalized
        return prefix, core, suffix

    def _should_preserve_location_verbatim(self, source_text: str) -> bool:
        normalized = re.sub(r"\s+", "", source_text)
        if not normalized:
            return False
        trimmed = normalized.rstrip("等")
        if self._is_preservable_region_token(trimmed):
            return True

        if any(separator in trimmed for separator in ["、", "，", ",", "+", "＋", "/", "／"]):
            parts = [
                part
                for part in re.split(r"[、，,+＋/／]", trimmed)
                if part
            ]
            if len(parts) >= 2 and all(self._is_preservable_region_token(part) for part in parts):
                return True

        return False

    def _is_preservable_region_token(self, token: str) -> bool:
        normalized = re.sub(r"\s+", "", token)
        if not normalized:
            return False
        if normalized in self.UNSUFFIXED_REGION_PREFIXES:
            return True

        segments = self._split_location_segments(normalized)
        if not segments:
            return False
        if "".join(segments) != normalized:
            return False
        return len(segments) == 1 and self._classify_location_segment(segments[0]) in {"province", "city"}

    def _should_preserve_surface_replacement(self, entity_type: str, source_text: Any, replacement: Any) -> bool:
        family = self.GROUP_FAMILIES.get(str(entity_type), "")
        source_value = str(source_text or "")
        replacement_value = str(replacement or "")
        if family == "position":
            return bool(source_value) and replacement_value == source_value
        if family != "location":
            return False
        return replacement_value == source_value and self._should_preserve_location_verbatim(source_value)

    def _mask_company_core(self, core_text: str) -> str:
        normalized = re.sub(r"\s+", "", core_text)
        if not normalized:
            return ""
        visible_indexes = [
            index
            for index, char in enumerate(normalized)
            if ("\u4e00" <= char <= "\u9fff") or char.isascii() and char.isalnum()
        ]
        if len(visible_indexes) >= 2:
            index = visible_indexes[1]
            return normalized[:index] + "某" + normalized[index + 1 :]
        if visible_indexes:
            index = visible_indexes[0]
            return normalized[: index + 1] + "某" + normalized[index + 1 :]
        return f"{normalized}某"

    def _extract_leading_cjk_char(self, text: str) -> str:
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return char
        return text[:1] if text else ""

    def _mask_identifier(
        self,
        source_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self._is_case_number(source_text, metadata):
            return mask_case_number_digits_only(source_text)
        return "".join("*" if not char.isspace() else char for char in source_text)

    def _is_case_number(
        self,
        source_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        details = metadata or {}
        identifier_kind = str(details.get("identifier_kind") or "").strip().lower()
        if identifier_kind == "case_no":
            return True

        label = str(details.get("label") or "").strip()
        if label_matches(
            label,
            (
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
            ),
        ):
            return True
        return looks_like_case_number(source_text)

    def _next_letter_alias(
        self,
        prefix: str,
        alias_state: Dict[str, Dict[str, int]],
    ) -> str:
        return f"{prefix}{self._next_scoped_alias(prefix, alias_state, style='letter', always_append=True, return_suffix_only=True)}"

    def _next_scoped_alias(
        self,
        prefix: str,
        alias_state: Dict[str, Dict[str, int]],
        *,
        style: str,
        always_append: bool,
        return_suffix_only: bool = False,
    ) -> str:
        counters = alias_state["counters"]
        count = counters.get(prefix, 0) + 1
        counters[prefix] = count

        suffix = self._to_alpha(count - 1) if style == "letter" else self._to_cn_serial(count - 1)
        if return_suffix_only:
            return suffix
        if not always_append and count == 1:
            return prefix
        return f"{prefix}{suffix}"

    def _to_alpha(self, index: int) -> str:
        if index < 0:
            return "A"

        result = ""
        base = len(self.LETTERS)
        current = index
        while True:
            result = self.LETTERS[current % base] + result
            current = current // base - 1
            if current < 0:
                break
        return result

    def _to_cn_serial(self, index: int) -> str:
        if 0 <= index < len(self.CN_SERIALS):
            return self.CN_SERIALS[index]
        return f"第{index + 1}"

    def _build_structured_replacement(
        self,
        *,
        entity: Dict,
        context: Dict,
        amount_cluster_values: Dict[tuple[int, int], float],
    ) -> Optional[str]:
        entity_type = entity["type"]
        source_text = entity["text"]

        if entity_type == "AMOUNT":
            return None
        if entity_type == "DATE":
            return None
        return None

    def _build_amount_cluster_values(
        self,
        entities: List[Dict],
        contexts: List[Dict],
    ) -> Dict[tuple[int, int], float]:
        clusters: Dict[tuple[int, int], float] = {}

        for entity, context in zip(entities, contexts):
            if entity["type"] != "AMOUNT":
                continue
            key = (context["line_start"], context["line_end"])
            if key in clusters:
                continue

            numeric_value = self._extract_numeric_amount(entity["text"])
            if numeric_value is None:
                numeric_value = self._estimate_numeric_from_upper_amount(entity["text"])
            if numeric_value is None:
                numeric_value = 100000.0

            factor = 1.08 + ((self._stable_seed(context["line"]) % 19) / 100)
            new_value = max(1000.0, numeric_value * factor)
            decimals = 2 if re.search(r"\.\d+", entity["text"]) else 0
            clusters[key] = round(new_value, decimals)

        return clusters

    def _extract_numeric_amount(self, text: str) -> Optional[float]:
        match = re.search(r"(\d[\d,]*(?:\.\d+)?)", text)
        if not match:
            return None
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            return None
        unit = text.replace(match.group(1), "")
        if "亿" in unit:
            value *= 100000000
        elif "万" in unit:
            value *= 10000
        return value

    def _estimate_numeric_from_upper_amount(self, text: str) -> Optional[float]:
        numerals = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "壹": 1,
            "二": 2,
            "贰": 2,
            "两": 2,
            "三": 3,
            "叁": 3,
            "四": 4,
            "肆": 4,
            "五": 5,
            "伍": 5,
            "六": 6,
            "陆": 6,
            "七": 7,
            "柒": 7,
            "八": 8,
            "捌": 8,
            "九": 9,
            "玖": 9,
        }
        units = {
            "十": 10,
            "拾": 10,
            "百": 100,
            "佰": 100,
            "千": 1000,
            "仟": 1000,
            "万": 10000,
            "亿": 100000000,
        }

        total = 0
        section = 0
        number = 0
        found = False

        for char in text:
            if char in numerals:
                number = numerals[char]
                found = True
            elif char in units:
                found = True
                unit_value = units[char]
                if unit_value >= 10000:
                    section = (section + max(number, 1)) * unit_value
                    total += section
                    section = 0
                else:
                    section += max(number, 1) * unit_value
                number = 0
        total += section + number
        return float(total) if found and total else None

    def _render_amount(self, source_text: str, amount_value: Optional[float]) -> str:
        amount_value = amount_value or 100000.0
        if re.search(r"\d", source_text):
            return self._format_numeric_amount(source_text, amount_value)
        return self._format_upper_amount(source_text, amount_value)

    def _format_numeric_amount(self, source_text: str, amount_value: float) -> str:
        match = re.search(r"(?P<prefix>.*?)(?P<number>\d[\d,]*(?:\.\d+)?)(?P<suffix>.*)", source_text)
        if not match:
            return source_text

        prefix = match.group("prefix")
        number = match.group("number")
        suffix = match.group("suffix")

        unit_multiplier = 1
        if "亿" in suffix:
            unit_multiplier = 100000000
        elif "万" in suffix:
            unit_multiplier = 10000

        scaled_value = amount_value / unit_multiplier
        decimals = len(number.split(".")[1]) if "." in number else 0
        formatted_number = f"{scaled_value:,.{decimals}f}" if "," in number else f"{scaled_value:.{decimals}f}"
        if decimals == 0:
            formatted_number = formatted_number.split(".")[0]

        return f"{prefix}{formatted_number}{suffix}"

    def _format_upper_amount(self, source_text: str, amount_value: float) -> str:
        integer_value = int(round(amount_value))
        upper_text = self._number_to_upper_amount(integer_value)
        if "人民币" in source_text:
            return f"人民币{upper_text}"
        return upper_text

    def _number_to_upper_amount(self, value: int) -> str:
        digits = ["零", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖"]
        units = ["", "拾", "佰", "仟"]
        big_units = ["", "万", "亿", "兆"]

        if value == 0:
            return "零元整"

        parts: List[str] = []
        unit_index = 0
        while value > 0:
            section = value % 10000
            value //= 10000
            if section == 0:
                unit_index += 1
                continue
            section_text = []
            zero_pending = False
            for i in range(4):
                digit = section % 10
                section //= 10
                if digit == 0:
                    zero_pending = bool(section_text)
                else:
                    if zero_pending:
                        section_text.append("零")
                        zero_pending = False
                    section_text.append(units[i])
                    section_text.append(digits[digit])
            parts.append("".join(reversed(section_text)).rstrip("零") + big_units[unit_index])
            unit_index += 1

        result = "".join(reversed(parts))
        result = re.sub(r"零+", "零", result).rstrip("零")
        return f"{result}元整"

    def _generate_person(self, seed: int) -> str:
        return self.PERSON_POOL[seed % len(self.PERSON_POOL)]

    def _generate_organization(self, seed: int, source_text: str, context: Dict) -> str:
        prefix = self.COMPANY_PREFIX[seed % len(self.COMPANY_PREFIX)]
        role = context.get("role", "")
        suffix = self._extract_org_suffix(source_text)
        theme_key = self._infer_industry_theme(source_text, context)
        theme_map = self.COMPANY_THEME_MAP.get(theme_key, self.COMPANY_THEME_MAP["default"])
        suffix_kind = self._classify_org_suffix(suffix)
        middle = theme_map[suffix_kind]

        if role == "甲方" and suffix_kind == "company" and theme_key == "default":
            middle = "工程投资"
        elif role in {"乙方", "收款"} and suffix_kind == "company" and theme_key == "default":
            middle = "技术服务"

        if suffix in {"服务中心", "技术中心"}:
            middle = re.sub(r"(服务|技术)$", "", middle)
            return f"{prefix}{middle}服务中心"
        if suffix_kind == "institute":
            return f"{prefix}{middle}{suffix}"
        return f"{prefix}{middle}{suffix}"

    def _classify_org_suffix(self, suffix: str) -> str:
        if suffix in {"服务中心", "技术中心"}:
            return "center"
        if suffix in {"研究院", "事务所", "局", "院", "所"}:
            return "institute"
        return "company"

    def _infer_industry_theme(self, source_text: str, context: Dict) -> str:
        priority_segments = [
            " ".join(part for part in [source_text, context.get("label", "")] if part),
            context.get("line", ""),
            " ".join(part for part in [context.get("previous_line", ""), context.get("next_line", "")] if part),
        ]

        for segment in priority_segments:
            theme = self._match_industry_theme(segment)
            if theme != "default":
                return theme
        return "default"

    def _match_industry_theme(self, text: str) -> str:
        if not text:
            return "default"

        theme_scores: Dict[str, int] = {}
        for theme, keywords in self.INDUSTRY_KEYWORDS.items():
            score = sum(text.count(keyword) for keyword in keywords)
            if score:
                theme_scores[theme] = score

        if not theme_scores:
            return "default"

        return max(
            theme_scores,
            key=lambda theme: (theme_scores[theme], -list(self.INDUSTRY_KEYWORDS).index(theme)),
        )

    def _extract_org_suffix(self, source_text: str) -> str:
        for suffix in self.ORG_SUFFIX_PATTERNS:
            if source_text.endswith(suffix):
                return suffix
        if "中心" in source_text:
            return "服务中心"
        return "有限公司"

    def _generate_bank_name(self, seed: int, source_text: str) -> str:
        prefix = self.COMPANY_PREFIX[seed % len(self.COMPANY_PREFIX)]
        branch = self.BRANCH_NAMES[seed % len(self.BRANCH_NAMES)]
        if "股份有限公司" in source_text:
            return f"{prefix}银行股份有限公司{branch}"
        return f"{prefix}银行{branch}"

    def _generate_account_name(self, seed: int, source_text: str, context: Dict) -> str:
        if any(token in source_text for token in ["公司", "中心", "集团", "所", "院"]):
            return self._generate_organization(seed, source_text, context)
        if re.fullmatch(r"[\u4e00-\u9fa5\u00b7]{2,8}", source_text):
            return self._generate_person(seed)
        return f"星泽结算账户{seed % 9 + 1}"

    def _generate_project(self, seed: int, source_text: str, context: Dict) -> str:
        prefix = self.PROJECT_PREFIX[seed % len(self.PROJECT_PREFIX)]
        role = context.get("role", "")
        theme_key = self._infer_industry_theme(source_text, context)
        capacity_text = self._generate_project_capacity(seed, source_text)

        if "风电" in source_text or theme_key == "新能源":
            capacity_prefix = f"{capacity_text}MW" if capacity_text else ""
            return f"{prefix}{capacity_prefix}风电场示范项目一期"
        if "检测" in source_text or theme_key == "检测":
            return f"{prefix}防雷检测服务项目"
        if "工程" in source_text or theme_key == "工程":
            return f"{prefix}建设工程项目"
        if theme_key == "采购":
            return f"{prefix}设备采购项目"
        if theme_key == "信息化":
            return f"{prefix}协同管理平台项目"
        if role == "检测项目":
            return f"{prefix}技术检测项目"
        return f"{prefix}业务协同项目"

    def _generate_project_capacity(self, seed: int, source_text: str) -> str:
        match = re.search(r"(\d+(?:\.\d+)?)\s*MW", source_text, re.I)
        if not match:
            return ""
        base_value = float(match.group(1))
        delta = (seed % 7) - 3
        anonymized = max(10.0, base_value + delta)
        if anonymized.is_integer():
            return str(int(anonymized))
        return f"{anonymized:.1f}".rstrip("0").rstrip(".")

    def _generate_location(self, seed: int, source_text: str) -> str:
        province = self.PROVINCES[seed % len(self.PROVINCES)] if "省" in source_text else ""
        city = self.CITIES[(seed // 3) % len(self.CITIES)] if "市" in source_text else ""
        district = self.DISTRICTS[(seed // 5) % len(self.DISTRICTS)] if "区" in source_text else ""
        county = self.COUNTIES[(seed // 7) % len(self.COUNTIES)] if "县" in source_text else ""
        town = self.TOWNS[(seed // 11) % len(self.TOWNS)] if "镇" in source_text else ""
        second_town = ""
        if source_text.count("镇") >= 2:
            second_town = self._pick_distinct_value(self.TOWNS, (seed // 13) % len(self.TOWNS), town)
        road = self.ROADS[(seed // 13) % len(self.ROADS)] if any(token in source_text for token in ["路", "街", "大道"]) else ""

        town_segment = ""
        if second_town and second_town != town:
            connector = "与" if "与" in source_text else "、"
            town_segment = f"{town}{connector}{second_town}"
        else:
            town_segment = town

        location = "".join(part for part in [province, city, district or county, town_segment, road] if part)
        if not location:
            location = f"{self.CITIES[seed % len(self.CITIES)]}{self.ROADS[(seed // 3) % len(self.ROADS)]}18号"
        return location

    def _pick_distinct_value(self, values: List[str], index: int, current: str) -> str:
        if not values:
            return ""
        candidate = values[index % len(values)]
        if candidate != current:
            return candidate
        return values[(index + 1) % len(values)]

    def _generate_position(self, seed: int, context: Dict) -> str:
        label = context.get("label", "")
        if "法定代表人" in label:
            return "法定代表人"
        if "联系人" in label:
            return "项目联系人"
        if "项目" in label:
            return "项目经理"
        options = ["项目经理", "商务经理", "项目主管", "综合经理"]
        return options[seed % len(options)]

    def _generate_contract_no(self, seed: int, source_text: str) -> str:
        year_match = re.search(r"(20\d{2})", source_text)
        year = year_match.group(1) if year_match else "2026"
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        prefix = letters[seed % len(letters)] + letters[(seed // 3) % len(letters)]
        return f"HT[{year}]{prefix}{seed % 9000 + 1000}号"

    def _generate_project_code(self, seed: int) -> str:
        return f"PRJ-{2026 + seed % 3}-{seed % 900 + 100:03d}"

    def _generate_product_name(self, seed: int, source_text: str) -> str:
        if "系统" in source_text:
            return f"协同管理系统{seed % 5 + 1}型"
        return f"标准化服务组件{seed % 8 + 1}号"

    def _generate_sensitive_term(self, seed: int) -> str:
        options = ["业务执行项", "合规运营项", "内部批准项", "受控信息项"]
        return options[seed % len(options)]

    def _generate_date(self, seed: int, source_text: str) -> str:
        year_match = re.search(r"(20\d{2})", source_text)
        month_match = re.search(r"(?:20\d{2}[年/-])(\d{1,2})", source_text)
        day_match = re.search(r"(?:\d{1,2}[月/-])(\d{1,2}|xx|XX)", source_text)

        base_year = int(year_match.group(1)) if year_match else 2026
        base_month = int(month_match.group(1)) if month_match else 1
        base_day = 1
        if day_match and day_match.group(1).isdigit():
            base_day = int(day_match.group(1))

        base_date = date(base_year, max(1, min(base_month, 12)), max(1, min(base_day, 28)))
        shifted = base_date + timedelta(days=seed % 120 + 7)

        if "年" in source_text:
            month_text = f"{shifted.month:02d}" if month_match and month_match.group(1).startswith("0") else str(shifted.month)
            if day_match and day_match.group(1).lower() == "xx":
                day_text = day_match.group(1)
            else:
                day_text = f"{shifted.day:02d}" if day_match and len(day_match.group(1)) == 2 else str(shifted.day)
            return f"{shifted.year}年{month_text}月{day_text}日"

        separator = "/" if "/" in source_text else "-"
        month_text = f"{shifted.month:02d}" if month_match and len(month_match.group(1)) == 2 else str(shifted.month)
        if day_match and day_match.group(1).lower() == "xx":
            day_text = day_match.group(1)
        else:
            day_length = len(day_match.group(1)) if day_match and day_match.group(1).isdigit() else 2
            day_text = f"{shifted.day:0{day_length}d}" if day_length == 2 else str(shifted.day)
        return f"{shifted.year}{separator}{month_text}{separator}{day_text}"

    def _generate_mobile_phone(self, seed: int) -> str:
        return f"139{seed % 100000000:08d}"

    def _generate_landline_phone(self, seed: int, source_text: str) -> str:
        match = re.match(r"(0\d{2,3})(-?)(\d+)", source_text)
        if not match:
            return f"010-6{seed % 1000000:06d}"
        area_code, dash, number = match.groups()
        new_number = f"6{seed % (10 ** (len(number) - 1)):0{len(number) - 1}d}"
        return f"{area_code}{dash}{new_number}"

    def _generate_bank_card(self, seed: int, source_text: str) -> str:
        length = len(re.sub(r"\D", "", source_text))
        body_length = max(10, length - 6)
        body = f"{seed % (10 ** body_length):0{body_length}d}"
        return f"621700{body}"[:length]

    def _generate_id_card(self, seed: int) -> str:
        area = "110101"
        year = 1985 + seed % 20
        month = seed % 12 + 1
        day = seed % 28 + 1
        seq = seed % 900 + 100
        check = "X" if seed % 2 else str(seed % 10)
        return f"{area}{year:04d}{month:02d}{day:02d}{seq:03d}{check}"

    def _generate_credit_code(self, seed: int) -> str:
        alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
        return "".join(alphabet[(seed + index * 7) % len(alphabet)] for index in range(18))

    def _generate_email(self, seed: int) -> str:
        return f"contact{seed % 997 + 1}@example.com"

    def _stable_seed(self, value: str) -> int:
        return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)

    def _get_ollama_service(self, llm_model: Optional[str] = None) -> Optional[OllamaLLMService]:
        if settings.LLM_BACKEND.lower() != "ollama":
            return None
        model_name = llm_model or settings.OLLAMA_MODEL
        if model_name not in self._ollama_services:
            self._ollama_services[model_name] = OllamaLLMService(
                base_url=settings.OLLAMA_BASE_URL,
                model=model_name,
                timeout=settings.OLLAMA_TIMEOUT,
                num_ctx=settings.OLLAMA_NUM_CTX,
            )
        return self._ollama_services[model_name]
