"""Structured single-document legal review generation."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from app.core.config import settings
from app.core.llm_strategy import get_llm_strategy_profile, get_runtime_llm_strategy_profile
from app.services.ollama_service import OllamaLLMService

logger = logging.getLogger(__name__)


class ReviewService:
    """Generate structured lawyer-facing review cards from one document."""

    LEGAL_DOCUMENT_TYPES = {"legal_application", "enforcement_paper"}
    PARTY_ENTITY_TYPES = {
        "PERSON",
        "PERSON_NAME",
        "ORGANIZATION",
        "COMPANY_NAME",
        "ACCOUNT_NAME",
        "BANK_NAME",
        "PROJECT",
    }
    ENTITY_LABELS = {
        "PERSON": "人名",
        "PERSON_NAME": "人名",
        "ORGANIZATION": "机构主体",
        "COMPANY_NAME": "公司主体",
        "LOCATION": "地址",
        "POSITION": "职务",
        "PROJECT": "项目名称",
        "CONTRACT_NO": "编号",
        "BANK_NAME": "开户行",
        "ACCOUNT_NAME": "户名",
        "AMOUNT": "金额",
        "DATE": "日期",
        "CN_PHONE": "手机号",
        "LANDLINE_PHONE": "联系电话",
        "CN_BANK_CARD": "银行卡号",
        "EMAIL_ADDRESS": "邮箱",
    }
    REVIEW_SEVERITIES = {"low", "medium", "high"}

    def __init__(self) -> None:
        self._ollama_services: Dict[str, OllamaLLMService] = {}

    async def generate_review(
        self,
        *,
        task_id: str,
        text: str,
        entities: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        structure: Optional[Dict[str, Any]] = None,
        quality_metadata: Optional[Dict[str, Any]] = None,
        llm_model: Optional[str] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        quality_metadata = dict(quality_metadata or {})

        self._emit_progress(progress_callback, 1, 3, "正在整理审阅事实片段...")
        fact_bundle = self._build_fact_bundle(
            task_id=task_id,
            text=text,
            entities=entities,
            metadata=metadata,
            structure=structure,
            quality_metadata=quality_metadata,
        )

        llm_payload = None
        if llm_model and settings.LLM_BACKEND.lower() == "ollama":
            self._emit_progress(progress_callback, 2, 3, "正在生成 27B 结构化审阅结论...")
            llm_payload = await asyncio.to_thread(
                self._generate_llm_review,
                fact_bundle,
                llm_model,
            )

        self._emit_progress(progress_callback, 3, 3, "正在整理审阅卡片与证据引用...")
        merged = self._merge_review_payload(fact_bundle, llm_payload, llm_model=llm_model)
        return merged

    def _emit_progress(
        self,
        progress_callback,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "stage": "review_generate",
                    "current": current,
                    "total": total,
                    "message": message,
                }
            )
        except Exception:
            logger.debug("Failed to emit review progress", exc_info=True)

    def _get_ollama_service(self, llm_model: str) -> OllamaLLMService:
        if llm_model not in self._ollama_services:
            self._ollama_services[llm_model] = OllamaLLMService(
                base_url=settings.OLLAMA_BASE_URL,
                model=llm_model,
                timeout=settings.OLLAMA_TIMEOUT,
                num_ctx=settings.OLLAMA_NUM_CTX,
            )
        return self._ollama_services[llm_model]

    def _build_fact_bundle(
        self,
        *,
        task_id: str,
        text: str,
        entities: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        structure: Optional[Dict[str, Any]],
        quality_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        document_type = str(
            metadata.get("llm_document_type")
            or metadata.get("document_type")
            or "other"
        ).strip() or "other"
        document_type_label = str(
            metadata.get("llm_document_type_label")
            or metadata.get("document_type_label")
            or self._document_type_label(document_type)
        ).strip() or self._document_type_label(document_type)

        canonical_groups = self._build_canonical_groups(entities, quality_metadata)
        evidence_catalog = self._build_evidence_catalog(
            text=text,
            entities=entities,
            structure=structure,
            canonical_groups=canonical_groups,
            quality_metadata=quality_metadata,
        )
        evidence_by_id = {item["id"]: item for item in evidence_catalog}
        suspected_misses = self._build_suspected_misses(
            text=text,
            quality_metadata=quality_metadata,
            metadata=metadata,
            evidence_by_id=evidence_by_id,
            structure=structure,
        )
        cards = self._build_fallback_cards(
            text=text,
            document_type=document_type,
            document_type_label=document_type_label,
            entities=entities,
            canonical_groups=canonical_groups,
            evidence_catalog=evidence_catalog,
            suspected_misses=suspected_misses,
            structure=structure,
        )
        summary = self._build_fallback_summary(
            document_type_label=document_type_label,
            canonical_groups=canonical_groups,
            entities=entities,
            text=text,
        )

        return {
            "task_id": task_id,
            "document_type": document_type,
            "document_type_label": document_type_label,
            "summary": summary,
            "cards": cards,
            "canonical_groups": canonical_groups,
            "suspected_misses": suspected_misses,
            "metadata": metadata,
            "quality_metadata": quality_metadata,
            "evidence_catalog": evidence_catalog,
            "text_length": len(text),
            "line_count": max(1, text.count("\n") + 1) if text else 0,
        }

    def _build_fallback_summary(
        self,
        *,
        document_type_label: str,
        canonical_groups: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        text: str,
    ) -> str:
        subjects = [
            group["primary_text"]
            for group in canonical_groups
            if str(group.get("entity_type", "")).strip().upper() in self.PARTY_ENTITY_TYPES
            and group.get("primary_text")
        ][:3]
        subject_text = "、".join(subjects) if subjects else "未形成稳定主体归并"
        amount_count = sum(1 for entity in entities if str(entity.get("type")) == "AMOUNT")
        date_count = sum(1 for entity in entities if str(entity.get("type")) == "DATE")
        length_hint = "篇幅较长" if len(text) >= 6000 else "篇幅中等" if len(text) >= 2400 else "篇幅较短"
        return (
            f"该文档当前识别为 {document_type_label}，{length_hint}。"
            f"已识别的核心主体包括 {subject_text}。"
            f"当前抽取到金额相关信息 {amount_count} 项、日期相关信息 {date_count} 项，"
            "审阅结果仅用于帮助律师快速定位事实、风险和待核实点。"
        )

    def _document_type_label(self, document_type: str) -> str:
        mapping = {
            "contract": "合同/协议类",
            "legal_application": "申请/诉状类",
            "enforcement_paper": "执行/保全类",
            "financial_account_material": "账户/结算材料类",
            "other": "其他正式文书",
        }
        return mapping.get(document_type, "其他正式文书")

    def _build_canonical_groups(
        self,
        entities: List[Dict[str, Any]],
        quality_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_type = str(entity.get("type", "")).strip()
            text = str(entity.get("text", "")).strip()
            if not entity_type or not text:
                continue

            group_id = (
                str(entity.get("group_id") or "").strip()
                or str(entity.get("canonical_key") or "").strip()
                or f"TEXT::{self._normalize_group_text(text)}"
            )
            if not group_id:
                continue

            group = groups.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "group_label": str(entity.get("group_label") or entity.get("context_label") or "").strip(),
                    "primary_text": text,
                    "entity_type": entity_type,
                    "canonical_role": str(entity.get("canonical_role") or "").strip() or None,
                    "aliases": set(),
                    "mentions": 0,
                    "needs_review": False,
                    "review_reasons": set(),
                    "confirmed": bool(entity.get("canonical_key")),
                },
            )
            group["aliases"].add(text)
            group["mentions"] += 1
            group["needs_review"] = bool(group["needs_review"] or entity.get("needs_review"))
            review_reason = str(entity.get("review_reason") or "").strip()
            if review_reason:
                group["review_reasons"].add(review_reason)
            if len(text) > len(str(group.get("primary_text", ""))):
                group["primary_text"] = text
            if not group.get("group_label") and entity.get("context_label"):
                group["group_label"] = str(entity.get("context_label"))
            if not group.get("canonical_role") and entity.get("canonical_role"):
                group["canonical_role"] = str(entity.get("canonical_role"))

        evidence_summary = quality_metadata.get("evidence_summary")
        if isinstance(evidence_summary, list):
            for item in evidence_summary:
                if not isinstance(item, dict):
                    continue
                group_id = str(item.get("canonical_key", "")).strip()
                if not group_id or group_id not in groups:
                    continue
                groups[group_id]["confirmed"] = bool(groups[group_id]["confirmed"] or not item.get("conflict"))
                if item.get("source_layer") and not groups[group_id].get("group_label"):
                    groups[group_id]["group_label"] = str(item.get("source_layer"))

        result: List[Dict[str, Any]] = []
        for group in groups.values():
            aliases = sorted(group["aliases"], key=len, reverse=True)
            result.append(
                {
                    "group_id": group["group_id"],
                    "group_label": group["group_label"] or aliases[0],
                    "primary_text": group["primary_text"],
                    "entity_type": group["entity_type"],
                    "canonical_role": group["canonical_role"],
                    "aliases": aliases,
                    "mentions": group["mentions"],
                    "needs_review": bool(group["needs_review"]),
                    "review_reasons": sorted(group["review_reasons"]),
                    "confirmed": bool(group["confirmed"]),
                }
            )

        result.sort(key=lambda item: (-int(item["mentions"]), -len(str(item["primary_text"])), str(item["group_id"])))
        return result

    def _build_evidence_catalog(
        self,
        *,
        text: str,
        entities: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        quality_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        seen_keys: set[tuple[int, int, str]] = set()

        for entity in sorted(entities, key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0)))):
            start = max(0, int(entity.get("start", 0)))
            end = max(start, int(entity.get("end", start)))
            quote = str(entity.get("text", "")).strip() or text[start:end]
            if not quote:
                continue
            block_type = self._infer_block_type_from_quote(quote)
            evidence = self._make_evidence_ref(
                evidence_id=f"EV{len(catalog) + 1}",
                quote=quote,
                start=start,
                end=end,
                structure=structure,
                fallback_block_type=block_type,
            )
            dedupe_key = (evidence["start"], evidence["end"], evidence["quote"])
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            catalog.append(evidence)
            if len(catalog) >= 24:
                break

        if len(catalog) < 24:
            for group in canonical_groups[:8]:
                primary_text = str(group.get("primary_text", "")).strip()
                if not primary_text:
                    continue
                for start, end in self._find_text_spans(text, primary_text):
                    evidence = self._make_evidence_ref(
                        evidence_id=f"EV{len(catalog) + 1}",
                        quote=text[start:end],
                        start=start,
                        end=end,
                        structure=structure,
                        fallback_block_type=self._infer_block_type_from_quote(primary_text),
                    )
                    dedupe_key = (evidence["start"], evidence["end"], evidence["quote"])
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    catalog.append(evidence)
                    break
                if len(catalog) >= 24:
                    break

        residual_hits = quality_metadata.get("residual_hits")
        if isinstance(residual_hits, list):
            for hit in residual_hits[:6]:
                if not isinstance(hit, dict):
                    continue
                start = max(0, int(hit.get("start", 0)))
                end = max(start, int(hit.get("end", start)))
                if end <= start:
                    continue
                quote = text[start:end]
                evidence = self._make_evidence_ref(
                    evidence_id=f"EV{len(catalog) + 1}",
                    quote=quote,
                    start=start,
                    end=end,
                    structure=structure,
                    fallback_block_type="line",
                )
                dedupe_key = (evidence["start"], evidence["end"], evidence["quote"])
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                catalog.append(evidence)
                if len(catalog) >= 30:
                    break

        return catalog

    def _build_suspected_misses(
        self,
        *,
        text: str,
        quality_metadata: Dict[str, Any],
        metadata: Dict[str, Any],
        evidence_by_id: Dict[str, Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        residual_hits = quality_metadata.get("residual_hits")
        if isinstance(residual_hits, list):
            for hit in residual_hits[:8]:
                if not isinstance(hit, dict):
                    continue
                start = max(0, int(hit.get("start", 0)))
                end = max(start, int(hit.get("end", start)))
                evidence = self._find_matching_evidence(
                    evidence_by_id.values(),
                    start,
                    end,
                    str(text[start:end]),
                )
                items.append(
                    {
                        "title": f"疑似遗漏的{self.ENTITY_LABELS.get(str(hit.get('type', '')), '敏感信息')}提及",
                        "severity": "medium",
                        "reason": str(hit.get("line") or hit.get("window_text") or "当前文本中仍出现未覆盖的主体变体。").strip(),
                        "evidence_refs": [evidence] if evidence else [],
                        "lawyer_action_hint": "建议核对该片段是否应补充进实体列表或与现有主体归并。",
                    }
                )

        consistency_issues = quality_metadata.get("consistency_issues")
        if isinstance(consistency_issues, list):
            for issue in consistency_issues[:4]:
                if not isinstance(issue, dict):
                    continue
                issue_type = str(issue.get("type", "")).strip()
                if issue_type == "canonical_key_multi_replacement":
                    reason = "同一主体在不同位置出现了多个替换表达，可能导致审阅和导出时不一致。"
                elif issue_type == "same_text_multi_replacement":
                    reason = "同一原文文本被映射到多个替换值，可能说明实体归并存在冲突。"
                else:
                    reason = "实体一致性检查发现了需要人工确认的问题。"
                items.append(
                    {
                        "title": "主体归并或替换一致性待确认",
                        "severity": "medium",
                        "reason": reason,
                        "evidence_refs": self._anchor_evidence_refs(list(evidence_by_id.values())),
                        "lawyer_action_hint": "建议人工核对同一主体、简称和角色称谓是否已经统一。",
                    }
                )

        if metadata.get("format") == "pdf" and structure and isinstance(structure.get("pages"), list):
            low_quality_pages = [
                page
                for page in structure["pages"]
                if isinstance(page, dict)
                and str(page.get("source", "")).startswith("ocr")
                and str(page.get("ocr_quality", "")).strip().lower() not in {"", "high"}
            ]
            for page in low_quality_pages[:3]:
                page_number = page.get("page_number")
                quote = str(page.get("text", "")).strip()[:80]
                evidence = self._make_evidence_ref(
                    evidence_id="",
                    quote=quote,
                    start=0,
                    end=0,
                    structure=structure,
                    fallback_block_type="page",
                    forced_page=page_number if isinstance(page_number, int) else None,
                    evidence_quality="low",
                )
                items.append(
                    {
                        "title": f"第 {page_number} 页 OCR 质量偏低",
                        "severity": "low",
                        "reason": "该页文本主要通过 OCR 恢复，质量未达到 high，关键信息可能仍需人工复核。",
                        "evidence_refs": [evidence],
                        "lawyer_action_hint": "建议重点核对该页中的金额、编号、签章、账户和尾部主体信息。",
                    }
                )

        return items

    def _build_fallback_cards(
        self,
        *,
        text: str,
        document_type: str,
        document_type_label: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        suspected_misses: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence_by_id = {item["id"]: item for item in evidence_catalog}
        cards: List[Dict[str, Any]] = [
            {
                "type": "summary",
                "title": "文档摘要",
                "items": [
                    {
                        "label": "摘要",
                        "value": self._build_fallback_summary(
                            document_type_label=document_type_label,
                            canonical_groups=canonical_groups,
                            entities=entities,
                            text=text,
                        ),
                        "evidence_refs": evidence_catalog[:2],
                    }
                ],
            },
            {
                "type": "case_profile",
                "title": "案件要素卡" if document_type in self.LEGAL_DOCUMENT_TYPES else "文书要素卡",
                "items": self._build_case_profile_items(
                    text=text,
                    document_type=document_type,
                    document_type_label=document_type_label,
                    entities=entities,
                    canonical_groups=canonical_groups,
                    evidence_catalog=evidence_catalog,
                    evidence_by_id=evidence_by_id,
                    structure=structure,
                ),
            },
            {
                "type": "parties",
                "title": "主体信息",
                "items": self._build_party_items(canonical_groups, evidence_by_id),
            },
            {
                "type": "pending",
                "title": "待确认项",
                "items": suspected_misses,
            },
        ]

        request_items = self._build_request_breakdown_items(
            text=text,
            document_type=document_type,
            evidence_catalog=evidence_catalog,
            structure=structure,
        )
        if request_items:
            cards.append(
                {
                    "type": "request_breakdown",
                    "title": "请求事项拆解" if document_type in self.LEGAL_DOCUMENT_TYPES else "核心事项拆解",
                    "items": request_items,
                }
            )

        procedure_items = self._build_procedure_check_items(
            text=text,
            document_type=document_type,
            entities=entities,
            canonical_groups=canonical_groups,
            evidence_catalog=evidence_catalog,
            structure=structure,
        )
        if procedure_items:
            cards.append(
                {
                    "type": "procedure_check",
                    "title": "程序信息核对" if document_type in self.LEGAL_DOCUMENT_TYPES else "文书完整性核对",
                    "items": procedure_items,
                }
            )

        if document_type == "contract":
            cards.append(
                {
                    "type": "risk",
                    "title": "风险点",
                    "items": self._build_contract_risks(text, entities, evidence_catalog),
                }
            )

        return [card for card in cards if card.get("items")]

    def _build_party_items(
        self,
        canonical_groups: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for group in canonical_groups[:10]:
            entity_type = str(group.get("entity_type", "")).strip().upper()
            if entity_type not in self.PARTY_ENTITY_TYPES:
                continue
            primary_text = str(group.get("primary_text", "")).strip()
            if not primary_text:
                continue
            evidence = self._find_evidence_for_quote(evidence_by_id.values(), primary_text)
            aliases = [alias for alias in group.get("aliases", []) if alias != primary_text][:4]
            label = str(group.get("group_label") or group.get("canonical_role") or self.ENTITY_LABELS.get(entity_type, "主体"))
            value = primary_text
            if aliases:
                value = f"{primary_text}（别名/简称：{'、'.join(aliases)}）"
            items.append(
                {
                    "label": label,
                    "value": value,
                    "reason": "基于实体归并和上下文角色整理出的主体信息。",
                    "evidence_refs": [evidence] if evidence else [],
                    "lawyer_action_hint": "建议核对主体名称、角色和简称是否与原文保持一致。",
                }
            )
        return items

    def _build_case_profile_items(
        self,
        *,
        text: str,
        document_type: str,
        document_type_label: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        items.append(
            {
                "label": "文书类型",
                "value": document_type_label,
                "reason": "根据文档分类结果整理出的当前文书类型。",
                "evidence_refs": self._anchor_evidence_refs(evidence_catalog),
                "lawyer_action_hint": "建议先确认文书分类是否准确，再继续查看后续拆解结果。",
            }
        )

        top_parties = []
        for group in canonical_groups:
            entity_type = str(group.get("entity_type", "")).strip().upper()
            if entity_type not in self.PARTY_ENTITY_TYPES:
                continue
            primary_text = str(group.get("primary_text", "")).strip()
            if not primary_text:
                continue
            role = str(group.get("canonical_role") or group.get("group_label") or "").strip()
            top_parties.append(f"{role}：{primary_text}" if role and role != primary_text else primary_text)
            if len(top_parties) >= 3:
                break
        if top_parties:
            items.append(
                {
                    "label": "核心主体",
                    "value": "；".join(top_parties),
                    "reason": "根据主体归并结果整理出的当前文书核心主体。",
                    "evidence_refs": self._anchor_evidence_refs(
                        evidence_catalog,
                        *(str(group.get("primary_text", "")).strip() for group in canonical_groups[:3]),
                    ),
                    "lawyer_action_hint": "建议核对主体名称、简称和程序身份是否已经一一对应。",
                }
            )

        identifier_entity = next(
            (
                str(entity.get("text", "")).strip()
                for entity in entities
                if str(entity.get("type", "")).strip().upper() == "CONTRACT_NO"
                and str(entity.get("text", "")).strip()
            ),
            "",
        )
        identifier_match = None if identifier_entity else self._find_first_pattern_match(
            text,
            (
                r"((?:案号|案件编号|执行案号)[:：]?\s*[^\n]{4,40}?号)",
                r"((?:\(|（)\d{4}(?:\)|）)[^\n]{2,40}?号)",
            ),
        )
        identifier_value = identifier_entity or (identifier_match["quote"] if identifier_match else "")
        if identifier_value:
            items.append(
                {
                    "label": "案号/编号",
                    "value": identifier_value,
                    "reason": "从文书编号、案号或项目编号表达中整理出的标识信息。",
                    "evidence_refs": self._resolve_match_evidence_refs(
                        text=text,
                        structure=structure,
                        evidence_catalog=evidence_catalog,
                        quote=identifier_value,
                        start=(identifier_match or {}).get("start"),
                        end=(identifier_match or {}).get("end"),
                    ),
                    "lawyer_action_hint": "建议核对编号是否完整，是否与封面、页眉或落款保持一致。",
                }
            )

        agency_match = self._find_first_pattern_match(
            text,
            (
                r"([\u4e00-\u9fa5]{2,30}(?:人民法院|仲裁委员会|人民检察院|公安局))",
            ),
        )
        if agency_match:
            items.append(
                {
                    "label": "处理机关",
                    "value": agency_match["quote"],
                    "reason": "根据正文或抬头中出现的法院、仲裁机构或处理机关整理。",
                    "evidence_refs": self._resolve_match_evidence_refs(
                        text=text,
                        structure=structure,
                        evidence_catalog=evidence_catalog,
                        quote=agency_match["quote"],
                        start=agency_match["start"],
                        end=agency_match["end"],
                    ),
                    "lawyer_action_hint": "建议核对机关名称是否完整，是否与案号和程序阶段相匹配。",
                }
            )

        date_values = [
            str(entity.get("text", "")).strip()
            for entity in entities
            if str(entity.get("type", "")).strip().upper() == "DATE"
            and str(entity.get("text", "")).strip()
        ][:3]
        if date_values:
            items.append(
                {
                    "label": "关键时间",
                    "value": "、".join(date_values),
                    "reason": "根据文书中的日期表达整理出的关键时间节点。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, *date_values),
                    "lawyer_action_hint": "建议核对这些日期分别对应签署、履行、申请或程序节点。",
                }
            )

        amount_values = [
            str(entity.get("text", "")).strip()
            for entity in entities
            if str(entity.get("type", "")).strip().upper() == "AMOUNT"
            and str(entity.get("text", "")).strip()
        ][:3]
        if amount_values:
            items.append(
                {
                    "label": "金额/标的",
                    "value": "、".join(amount_values),
                    "reason": "根据文书中的金额表达整理出的争议或履行金额信息。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, *amount_values),
                    "lawyer_action_hint": "建议核对金额是否区分本金、利息、违约金或费用。",
                }
            )

        return items[:6]

    def _build_request_breakdown_items(
        self,
        *,
        text: str,
        document_type: str,
        evidence_catalog: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if document_type in self.LEGAL_DOCUMENT_TYPES:
            return self._build_legal_request_breakdown_items(text, evidence_catalog, structure)
        return self._build_document_focus_items(text, evidence_catalog, structure)

    def _build_legal_request_breakdown_items(
        self,
        text: str,
        evidence_catalog: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        request_block = self._find_first_pattern_match(
            text,
            (
                r"((?:诉讼请求|上诉请求|申请事项|请求事项)[:：][\s\S]{0,260})",
            ),
        )
        if not request_block:
            return [
                {
                    "title": "请求事项入口待确认",
                    "severity": "medium",
                    "value": "未稳定定位到显式的请求事项段落。",
                    "reason": "当前文本中没有清晰匹配到“诉讼请求 / 上诉请求 / 申请事项 / 请求事项”标题。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "请求", "申请", "上诉"),
                    "lawyer_action_hint": "建议人工核对文书中是否已完整列明全部请求或申请事项。",
                }
            ]

        block_text = re.split(
            r"(事实与理由|事实和理由|申请理由|理由如下|此致|综上)",
            request_block["quote"],
            maxsplit=1,
        )[0].strip()
        block_body = re.sub(r"^(?:诉讼请求|上诉请求|申请事项|请求事项)[:：]\s*", "", block_text).strip()
        segments: List[str] = []
        for line in block_body.splitlines():
            normalized = re.sub(r"^[一二三四五六七八九十0-9]+[、.．)\）]\s*", "", line).strip("；;。 ")
            if normalized:
                segments.append(normalized)
        if len(segments) <= 1:
            for chunk in re.split(r"[；;]\s*", block_body):
                normalized = re.sub(r"^[一二三四五六七八九十0-9]+[、.．)\）]\s*", "", chunk).strip("；;。 ")
                if normalized:
                    segments.append(normalized)
        if not segments and block_body:
            segments = [block_body[:160]]

        items: List[Dict[str, Any]] = []
        fallback_ref = self._resolve_match_evidence_refs(
            text=text,
            structure=structure,
            evidence_catalog=evidence_catalog,
            quote=request_block["quote"],
            start=request_block["start"],
            end=request_block["end"],
        )
        for index, segment in enumerate(segments[:6], start=1):
            items.append(
                {
                    "title": f"请求 {index}",
                    "value": segment[:180],
                    "reason": "根据请求事项段落拆解出的核心请求。",
                    "evidence_refs": self._resolve_fragment_evidence_refs(
                        text=text,
                        structure=structure,
                        evidence_catalog=evidence_catalog,
                        fragment=segment,
                        fallback_refs=fallback_ref,
                    ),
                    "lawyer_action_hint": "建议核对该请求是否具备对应的事实、金额和证据支撑。",
                }
            )
        return items

    def _build_document_focus_items(
        self,
        text: str,
        evidence_catalog: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        keyword_groups = [
            ("付款安排", ("付款", "支付", "价款", "结算", "收款")),
            ("履行期限", ("履行期限", "交付", "完成", "服务期限", "交货")),
            ("违约责任", ("违约", "赔偿", "违约金")),
            ("争议处理", ("争议解决", "仲裁", "管辖", "人民法院")),
        ]
        items: List[Dict[str, Any]] = []
        for title, keywords in keyword_groups:
            line_match = self._find_keyword_line_match(text, keywords)
            if not line_match:
                continue
            items.append(
                {
                    "title": title,
                    "value": line_match["quote"][:180],
                    "reason": "根据文书中显式出现的关键义务或处理条款整理。",
                    "evidence_refs": self._resolve_match_evidence_refs(
                        text=text,
                        structure=structure,
                        evidence_catalog=evidence_catalog,
                        quote=line_match["quote"],
                        start=line_match["start"],
                        end=line_match["end"],
                    ),
                    "lawyer_action_hint": "建议核对该条款是否完整，是否还存在附条件或附件限制。",
                }
            )
        return items[:4]

    def _build_procedure_check_items(
        self,
        *,
        text: str,
        document_type: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if document_type in self.LEGAL_DOCUMENT_TYPES:
            return self._build_legal_procedure_check_items(
                text=text,
                entities=entities,
                canonical_groups=canonical_groups,
                evidence_catalog=evidence_catalog,
                structure=structure,
            )
        return self._build_document_completeness_items(
            text=text,
            entities=entities,
            canonical_groups=canonical_groups,
            evidence_catalog=evidence_catalog,
            structure=structure,
        )

    def _build_legal_procedure_check_items(
        self,
        *,
        text: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        case_match = self._find_first_pattern_match(
            text,
            (
                r"((?:案号|案件编号|执行案号)[:：]?\s*[^\n]{4,40}?号)",
                r"((?:\(|（)\d{4}(?:\)|）)[^\n]{2,40}?号)",
            ),
        )
        items.append(
            self._build_check_item(
                title="案号核对",
                present=bool(case_match),
                present_value=(case_match["quote"] if case_match else "已定位到案号信息"),
                missing_reason="当前文书未稳定定位到案号，程序定位可能不足。",
                present_reason="已定位到案号或案件编号信息。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=case_match,
                missing_quotes=("案号", "民事", "执行"),
                missing_hint="建议核对首页、抬头或正文说明段中的案号表达。",
                present_hint="建议核对案号是否完整，是否与处理机关保持一致。",
            )
        )

        agency_match = self._find_first_pattern_match(
            text,
            (r"([\u4e00-\u9fa5]{2,30}(?:人民法院|仲裁委员会|人民检察院|公安局))",),
        )
        items.append(
            self._build_check_item(
                title="处理机关核对",
                present=bool(agency_match),
                present_value=(agency_match["quote"] if agency_match else "已识别处理机关"),
                missing_reason="当前文书未明显定位到法院、仲裁机构或处理机关名称。",
                present_reason="已识别处理机关名称。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=agency_match,
                missing_quotes=("法院", "仲裁", "检察院"),
                missing_hint="建议核对抬头、落款或送达段是否明确处理机关。",
                present_hint="建议核对机关名称是否完整，是否与文书类型相匹配。",
            )
        )

        role_keywords = ("上诉人", "被上诉人", "原告", "被告", "申请人", "被申请人", "被执行人", "申请执行人")
        role_groups = [
            group
            for group in canonical_groups
            if any(keyword in str(group.get("group_label", "")) or keyword in str(group.get("canonical_role", "")) for keyword in role_keywords)
        ]
        role_match = self._find_keyword_line_match(text, role_keywords)
        items.append(
            self._build_check_item(
                title="当事人角色核对",
                present=bool(role_groups or role_match),
                present_value="；".join(
                    f"{str(group.get('group_label') or group.get('canonical_role') or '主体')}：{str(group.get('primary_text', '')).strip()}"
                    for group in role_groups[:3]
                    if str(group.get("primary_text", "")).strip()
                ) or (role_match["quote"] if role_match else "已识别角色信息"),
                missing_reason="当前文书未稳定识别到主要当事人角色行，主体身份可能需要补充确认。",
                present_reason="已识别主要当事人角色或程序身份。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=role_match,
                missing_quotes=role_keywords,
                missing_hint="建议核对各主体是否分别对应上诉人、被上诉人、申请人等身份。",
                present_hint="建议核对角色称谓与主体名称是否一一对应。",
                missing_severity="high",
            )
        )

        request_match = self._find_first_pattern_match(
            text,
            (r"((?:诉讼请求|上诉请求|申请事项|请求事项)[:：][\s\S]{0,120})",),
        )
        items.append(
            self._build_check_item(
                title="请求事项核对",
                present=bool(request_match),
                present_value=(request_match["quote"][:140] if request_match else "已识别请求事项"),
                missing_reason="当前文书未明显定位到请求事项段落，请求边界可能不够清晰。",
                present_reason="已定位到请求事项段落。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=request_match,
                missing_quotes=("请求事项", "申请事项", "诉讼请求", "上诉请求"),
                missing_hint="建议核对文书中是否逐项列明全部请求。",
                present_hint="建议核对请求事项是否完整，是否与证据和事实部分一致。",
            )
        )
        return items

    def _build_document_completeness_items(
        self,
        *,
        text: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        has_parties = any(
            str(group.get("entity_type", "")).strip().upper() in self.PARTY_ENTITY_TYPES and str(group.get("primary_text", "")).strip()
            for group in canonical_groups
        )
        party_match = self._find_keyword_line_match(text, ("甲方", "乙方", "委托人", "受托人", "公司"))
        items.append(
            self._build_check_item(
                title="主体信息核对",
                present=has_parties,
                present_value="已识别核心主体信息",
                missing_reason="当前文书的主体信息较弱，后续归并与导出可能不够稳定。",
                present_reason="已识别到主要主体信息。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=party_match,
                missing_quotes=("甲方", "乙方", "公司"),
                missing_hint="建议核对文书抬头、签署页或账户信息中的主体名称。",
                present_hint="建议核对主体名称和简称是否已经统一。",
            )
        )

        identifier_present = any(
            str(entity.get("type", "")).strip().upper() == "CONTRACT_NO" and str(entity.get("text", "")).strip()
            for entity in entities
        )
        identifier_match = self._find_keyword_line_match(text, ("编号", "合同号", "协议号", "项目号"))
        items.append(
            self._build_check_item(
                title="编号信息核对",
                present=bool(identifier_present or identifier_match),
                present_value=(identifier_match["quote"] if identifier_match else "已识别编号信息"),
                missing_reason="当前文书未稳定识别到编号或项目标识信息。",
                present_reason="已识别编号或项目标识信息。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=identifier_match,
                missing_quotes=("编号", "合同号", "项目"),
                missing_hint="建议核对抬头、页眉页脚或项目基础信息中的编号表达。",
                present_hint="建议核对编号是否完整，是否与正文引用保持一致。",
            )
        )

        date_present = any(
            str(entity.get("type", "")).strip().upper() == "DATE" and str(entity.get("text", "")).strip()
            for entity in entities
        )
        date_match = self._find_keyword_line_match(text, ("日期", "签订", "期限", "生效"))
        items.append(
            self._build_check_item(
                title="日期节点核对",
                present=bool(date_present or date_match),
                present_value=(date_match["quote"] if date_match else "已识别日期信息"),
                missing_reason="当前文书未稳定识别到关键日期，履行或生效节点可能需要补充确认。",
                present_reason="已识别关键日期或期限表达。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=date_match,
                missing_quotes=("日期", "期限", "签订"),
                missing_hint="建议核对签订、生效、履行和截止时间是否完整。",
                present_hint="建议核对日期分别对应的业务节点和条件限制。",
            )
        )

        signature_match = self._find_keyword_line_match(text, ("签字", "签章", "盖章", "法定代表人", "授权代表", "附件"))
        items.append(
            self._build_check_item(
                title="签署与附件核对",
                present=bool(signature_match),
                present_value=(signature_match["quote"] if signature_match else "已识别签署或附件线索"),
                missing_reason="当前文书未明显定位到签署页、签章信息或附件线索。",
                present_reason="已识别签署、签章或附件相关表达。",
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                match=signature_match,
                missing_quotes=("签字", "签章", "附件"),
                missing_hint="建议核对尾部签章页、授权代表和附件清单是否完整。",
                present_hint="建议核对签署主体、签署时间和附件引用是否一致。",
            )
        )
        return items

    def _build_check_item(
        self,
        *,
        title: str,
        present: bool,
        present_value: str,
        missing_reason: str,
        present_reason: str,
        text: str,
        structure: Optional[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        match: Optional[Dict[str, Any]],
        missing_quotes: tuple[str, ...],
        missing_hint: str,
        present_hint: str,
        missing_severity: str = "medium",
    ) -> Dict[str, Any]:
        if present:
            evidence_refs = self._resolve_match_evidence_refs(
                text=text,
                structure=structure,
                evidence_catalog=evidence_catalog,
                quote=(match or {}).get("quote") or present_value,
                start=(match or {}).get("start"),
                end=(match or {}).get("end"),
            )
            return {
                "title": title,
                "value": present_value,
                "reason": present_reason,
                "evidence_refs": evidence_refs,
                "lawyer_action_hint": present_hint,
            }
        return {
            "title": title,
            "severity": missing_severity,
            "value": "待补充核对",
            "reason": missing_reason,
            "evidence_refs": self._anchor_evidence_refs(evidence_catalog, *missing_quotes),
            "lawyer_action_hint": missing_hint,
        }

    def _build_key_info_items(
        self,
        entities: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        preferred_types = [
            "CONTRACT_NO",
            "DATE",
            "AMOUNT",
            "PROJECT",
            "LOCATION",
            "BANK_NAME",
            "ACCOUNT_NAME",
            "CN_PHONE",
            "LANDLINE_PHONE",
        ]
        items: List[Dict[str, Any]] = []
        seen_values: set[tuple[str, str]] = set()
        for entity_type in preferred_types:
            typed_entities = [
                entity
                for entity in entities
                if str(entity.get("type", "")).strip().upper() == entity_type
            ]
            for entity in typed_entities[:4]:
                value = str(entity.get("text", "")).strip()
                if not value or (entity_type, value) in seen_values:
                    continue
                seen_values.add((entity_type, value))
                evidence = self._find_matching_evidence(
                    evidence_by_id.values(),
                    int(entity.get("start", 0)),
                    int(entity.get("end", 0)),
                    value,
                )
                items.append(
                    {
                        "label": self.ENTITY_LABELS.get(entity_type, entity_type),
                        "value": value,
                        "reason": "从当前文档中抽取出的关键事实信息。",
                        "evidence_refs": [evidence] if evidence else [],
                        "lawyer_action_hint": "建议核对该信息是否完整、是否存在上下文限制或附条件表述。",
                    }
                )
        return items[:14]

    def _build_contract_risks(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_types = {str(entity.get("type", "")).strip().upper() for entity in entities}
        lower_text = text
        risks: List[Dict[str, Any]] = []
        if "CONTRACT_NO" not in entity_types:
            risks.append(
                {
                    "title": "合同编号未明确识别",
                    "severity": "medium",
                    "reason": "当前文档未稳定识别出合同编号或项目编号，归档和引用时可能不够明确。",
                    "evidence_refs": evidence_catalog[:1],
                    "lawyer_action_hint": "建议核对抬头、页眉页脚或项目基础信息处是否存在编号表达。",
                }
            )
        if not re.search(r"(违约|违约责任|违约金|赔偿)", lower_text):
            risks.append(
                {
                    "title": "违约责任条款识别较弱",
                    "severity": "medium",
                    "reason": "原文中未明显出现违约责任类关键词，后续争议处置条款可能不足。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "付款", "支付", "价款", "结算"),
                    "lawyer_action_hint": "建议核对是否存在违约责任、赔偿范围和违约金计算规则。",
                }
            )
        if not re.search(r"(争议解决|仲裁|人民法院|管辖法院|管辖)", lower_text):
            risks.append(
                {
                    "title": "争议解决路径未明确定位",
                    "severity": "medium",
                    "reason": "原文中未明显识别出争议解决或管辖条款。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "合同", "协议", "纠纷"),
                    "lawyer_action_hint": "建议核对合同是否明确约定仲裁机构、法院管辖或适用方式。",
                }
            )
        if not re.search(r"(盖章|签字|签章|法定代表人|授权代表)", lower_text):
            risks.append(
                {
                    "title": "签章或签署要素不明确",
                    "severity": "high",
                    "reason": "当前文档未明显定位到签章、签字或代表人签署信息，主体生效要件可能需要进一步确认。",
                    "evidence_refs": self._anchor_evidence_refs(
                        evidence_catalog,
                        "法定代表人",
                        "授权代表",
                        "甲方",
                        "乙方",
                        prefer_last=True,
                    ),
                    "lawyer_action_hint": "建议核对尾部签章页、代表人签字和生效条款是否完整。",
                }
            )
        if not re.search(r"(付款|支付|价款|结算|收款)", lower_text):
            risks.append(
                {
                    "title": "付款或结算条款识别不足",
                    "severity": "medium",
                    "reason": "原文中未明显识别出付款、结算或价款表达，履行与收款安排可能不够清晰。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "金额", "价款", "合同"),
                    "lawyer_action_hint": "建议核对价款金额、付款节点、收款账户和发票条件。",
                }
            )
        return risks[:6]

    def _build_claim_items(
        self,
        text: str,
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        patterns = [
            r"(诉讼请求[:：].{0,120})",
            r"(上诉请求[:：].{0,120})",
            r"(申请事项[:：].{0,120})",
            r"(请求事项[:：].{0,120})",
        ]
        items: List[Dict[str, Any]] = []
        for pattern in patterns:
            match = re.search(pattern, text, re.S)
            if not match:
                continue
            quote = match.group(1).strip()
            evidence = self._find_evidence_for_quote(evidence_catalog, quote[:40])
            items.append(
                {
                    "label": "请求事项",
                    "value": quote[:160],
                    "reason": "根据原文中显式出现的请求事项或诉请表达整理。",
                    "evidence_refs": [evidence] if evidence else [],
                    "lawyer_action_hint": "建议核对请求事项是否完整、是否与事实和证据部分相互支撑。",
                }
            )
            break
        if not items:
            items.append(
                {
                    "label": "请求事项",
                    "value": "未稳定定位到请求事项段落。",
                    "severity": "medium",
                    "reason": "文中未明确识别到“诉讼请求 / 上诉请求 / 申请事项”等段落。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "申请", "起诉", "上诉"),
                    "lawyer_action_hint": "建议人工核对文书是否明确列明全部请求或申请事项。",
                }
            )
        return items

    def _build_litigation_risks(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_types = {str(entity.get("type", "")).strip().upper() for entity in entities}
        risks: List[Dict[str, Any]] = []
        if "CONTRACT_NO" not in entity_types and "案号" not in text:
            risks.append(
                {
                    "title": "案号未明确定位",
                    "severity": "medium",
                    "reason": "当前文档没有稳定识别到案号，程序阶段和关联案件定位可能不足。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "民事", "申请", "上诉"),
                    "lawyer_action_hint": "建议核对抬头、首页或说明性段落中是否存在案号。",
                }
            )
        if not re.search(r"(人民法院|仲裁委员会)", text):
            risks.append(
                {
                    "title": "法院或机构信息不明确",
                    "severity": "medium",
                    "reason": "当前文档未明显定位到法院、仲裁机构或处理机关名称。",
                    "evidence_refs": evidence_catalog[:1],
                    "lawyer_action_hint": "建议核对文书抬头、落款或送达段是否已明确处理机关。",
                }
            )
        if not re.search(r"(上诉人|被上诉人|原告|被告|申请人|被申请人|被执行人|申请执行人)", text):
            risks.append(
                {
                    "title": "当事人角色行识别不足",
                    "severity": "high",
                    "reason": "当前文书中未明显定位到主要当事人角色行，主体定位和程序身份可能存在缺口。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "公司", "法院", "申请"),
                    "lawyer_action_hint": "建议逐项核对当事人、代理人和程序身份是否清晰对应。",
                }
            )
        if not re.search(r"(事实与理由|事实和理由|申请理由|理由如下)", text):
            risks.append(
                {
                    "title": "事实与理由段结构不清",
                    "severity": "medium",
                    "reason": "当前文书中未明显识别到事实与理由段落，论证结构可能需要人工确认。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "请求", "申请", "事实"),
                    "lawyer_action_hint": "建议核对事实、法律理由与证据之间是否形成完整链条。",
                }
            )
        return risks[:6]

    def _generate_llm_review(
        self,
        fact_bundle: Dict[str, Any],
        llm_model: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            service = self._get_ollama_service(llm_model)
            if not service.available:
                return None
            prompt = self._build_llm_review_prompt(fact_bundle)
            payload = service.generate_json(prompt, num_predict=2200)
            parsed = service._load_json_object(payload, required_keys={"summary", "cards"})
            if not isinstance(parsed, dict):
                return None
            return parsed
        except Exception as exc:
            logger.warning("Structured review generation fell back to heuristics: %s", exc)
            return None

    def _build_llm_review_prompt(self, fact_bundle: Dict[str, Any]) -> str:
        evidence_lines = []
        for evidence in fact_bundle["evidence_catalog"][:24]:
            evidence_lines.append(
                (
                    f'- {evidence["id"]}: quote="{evidence["quote"]}", '
                    f'page={evidence.get("page")}, block_type={evidence.get("block_type")}, '
                    f'evidence_quality={evidence.get("evidence_quality")}'
                )
            )

        group_lines = []
        for group in fact_bundle["canonical_groups"][:10]:
            aliases = "、".join(group.get("aliases", [])[:4])
            group_lines.append(
                f'- {group["group_id"]}: primary_text="{group.get("primary_text", "")}", '
                f'role="{group.get("canonical_role") or ""}", label="{group.get("group_label") or ""}", '
                f'aliases="{aliases}"'
            )

        missing_lines = []
        for item in fact_bundle["suspected_misses"][:8]:
            evidence_ids = ",".join(ref.get("id", "") for ref in item.get("evidence_refs", []) if ref.get("id"))
            missing_lines.append(
                f'- title="{item.get("title", "")}", severity="{item.get("severity", "")}", '
                f'reason="{item.get("reason", "")}", evidence_ref_ids="{evidence_ids}"'
            )

        strategy = get_llm_strategy_profile(llm_model)
        return f"""
You are generating a structured legal document review for a Chinese lawyer.
This output must stay factual, concise, and grounded in the supplied evidence only.

Document type: {fact_bundle["document_type"]} ({fact_bundle["document_type_label"]})
Model strategy: {strategy.key} / {strategy.label}

Canonical groups:
{chr(10).join(group_lines) or "- none"}

Evidence catalog:
{chr(10).join(evidence_lines) or "- none"}

Potential misses and review cues:
{chr(10).join(missing_lines) or "- none"}

Rules:
1. Output strict JSON only.
2. Do not give legal conclusions, legal advice, or cite laws not present in the source.
3. Stay within the evidence catalog and the provided canonical groups.
4. Every risk or pending item must include a title, severity, reason, lawyer_action_hint, and at least one evidence_ref_id when evidence exists.
5. Keep severities to low, medium, or high.
6. Always include the following card types when relevant:
   - summary
   - case_profile
   - parties
   - procedure_check
   - pending
7. For litigation / application / enforcement documents, also include:
   - request_breakdown
8. For contract or other business documents, include request_breakdown only when the source text contains explicit core obligations or commercial arrangement cues.
9. risk is an optional supplemental card for contract documents only.

Return this exact JSON shape:
{{
  "summary": "one paragraph",
  "cards": [
    {{
      "type": "summary",
      "title": "文档摘要",
      "items": [
        {{
          "label": "摘要",
          "value": "text",
          "reason": "why this summary is grounded",
          "evidence_ref_ids": ["EV1", "EV2"],
          "lawyer_action_hint": "what to verify next"
        }}
      ]
    }}
  ],
  "suspected_misses": [
    {{
      "title": "item title",
      "severity": "medium",
      "reason": "why it needs review",
      "evidence_ref_ids": ["EV3"],
      "lawyer_action_hint": "what the lawyer should verify"
    }}
  ]
}}
""".strip()

    def _merge_review_payload(
        self,
        fact_bundle: Dict[str, Any],
        llm_payload: Optional[Dict[str, Any]],
        *,
        llm_model: Optional[str],
    ) -> Dict[str, Any]:
        evidence_catalog = fact_bundle["evidence_catalog"]
        evidence_by_id = {item["id"]: item for item in evidence_catalog if item.get("id")}
        fallback_cards = {card["type"]: card for card in fact_bundle["cards"]}
        runtime_profile = (
            get_runtime_llm_strategy_profile(
                llm_model,
                text_length=int(fact_bundle.get("text_length") or 0),
                line_count=int(fact_bundle.get("line_count") or 0),
            )
            if llm_model
            else None
        )
        strategy = runtime_profile.strategy if runtime_profile is not None else None

        summary = fact_bundle["summary"]
        cards = list(fact_bundle["cards"])
        suspected_misses = list(fact_bundle["suspected_misses"])

        if isinstance(llm_payload, dict):
            llm_summary = str(llm_payload.get("summary", "")).strip()
            if llm_summary:
                summary = llm_summary

            merged_cards: List[Dict[str, Any]] = []
            if isinstance(llm_payload.get("cards"), list):
                for card in llm_payload["cards"]:
                    parsed_card = self._normalize_llm_card(card, evidence_by_id)
                    if not parsed_card:
                        continue
                    merged_cards.append(parsed_card)
                    fallback_cards.pop(parsed_card["type"], None)
            merged_cards.extend(fallback_cards.values())
            cards = merged_cards

            llm_misses = llm_payload.get("suspected_misses")
            if isinstance(llm_misses, list):
                combined = suspected_misses[:]
                for item in llm_misses:
                    parsed_item = self._normalize_llm_card_item(item, evidence_by_id)
                    if not parsed_item or not parsed_item.get("title"):
                        continue
                    combined.append(parsed_item)
                suspected_misses = self._dedupe_review_items(combined)

        cards = self._backfill_card_evidence(cards, fact_bundle["cards"], evidence_catalog)
        suspected_misses = self._backfill_review_items(
            suspected_misses,
            fact_bundle["suspected_misses"],
            evidence_catalog,
        )

        if llm_model and isinstance(llm_payload, dict):
            review_generation_mode = "llm_structured"
            review_generation_label = "27B 结构化输出"
            precision_summary = "27B 已完成高精度识别与结构化审阅输出，当前卡片来自模型结论与证据归并。"
        elif llm_model:
            review_generation_mode = "heuristic_fallback"
            review_generation_label = "启发式回退"
            precision_summary = (
                "27B 已参与高精度识别、OCR/实体复核，但结构化审阅 JSON 未稳定产出，"
                "当前卡片由启发式规则与证据锚点回退生成。"
            )
        else:
            review_generation_mode = "heuristic_only"
            review_generation_label = "启发式生成"
            precision_summary = "当前结果主要由结构化启发式审阅生成。"

        metadata = {
            **fact_bundle["metadata"],
            "analysis_tier": "precision_review" if llm_model else "standard",
            "ocr_mode": self._infer_ocr_mode(fact_bundle["metadata"]),
            "precision_summary": precision_summary,
            "review_model": llm_model,
            "review_generation_mode": review_generation_mode,
            "review_generation_label": review_generation_label,
            "review_strategy_key": strategy.key if strategy is not None else None,
            "review_strategy_label": strategy.label if strategy is not None else None,
            "review_strategy_description": strategy.description if strategy is not None else None,
            "review_budget_tier": runtime_profile.budget_tier if runtime_profile is not None else None,
            "review_budget_label": runtime_profile.budget_label if runtime_profile is not None else None,
            "specialized_passes": list(strategy.specialized_passes) if strategy is not None else [],
            "definition_recall_enabled": bool(strategy.enable_definition_recall) if strategy is not None else False,
            "residual_scan_enabled": bool(strategy.enable_residual_scan) if strategy is not None else False,
            "review_card_count": len(cards),
            "review_evidence_count": len(evidence_catalog),
            "review_queue_count": len(suspected_misses),
        }

        return {
            "task_id": fact_bundle["task_id"],
            "document_type": fact_bundle["document_type"],
            "document_type_label": fact_bundle["document_type_label"],
            "summary": summary,
            "cards": cards,
            "canonical_groups": fact_bundle["canonical_groups"],
            "suspected_misses": suspected_misses,
            "metadata": metadata,
        }

    def _normalize_llm_card(
        self,
        card: Any,
        evidence_by_id: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(card, dict):
            return None
        card_type = str(card.get("type", "")).strip()
        title = str(card.get("title", "")).strip()
        if not card_type or not title:
            return None
        items = card.get("items")
        if not isinstance(items, list):
            return None
        normalized_items = []
        for item in items:
            normalized_item = self._normalize_llm_card_item(item, evidence_by_id)
            if normalized_item:
                normalized_items.append(normalized_item)
        if not normalized_items:
            return None
        return {
            "type": card_type,
            "title": title,
            "items": normalized_items,
        }

    def _normalize_llm_card_item(
        self,
        item: Any,
        evidence_by_id: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        evidence_refs: List[Dict[str, Any]] = []
        ref_ids = item.get("evidence_ref_ids")
        if isinstance(ref_ids, list):
            for ref_id in ref_ids:
                ref = evidence_by_id.get(str(ref_id).strip())
                if ref:
                    evidence_refs.append(ref)

        severity = str(item.get("severity", "")).strip().lower() or None
        if severity not in self.REVIEW_SEVERITIES:
            severity = None

        normalized = {
            "title": str(item.get("title", "")).strip() or None,
            "label": str(item.get("label", "")).strip() or None,
            "value": str(item.get("value", "")).strip() or None,
            "severity": severity,
            "reason": str(item.get("reason", "")).strip() or None,
            "evidence_refs": evidence_refs,
            "lawyer_action_hint": str(item.get("lawyer_action_hint", "")).strip() or None,
        }
        if not any(normalized.get(key) for key in ("title", "label", "value", "reason")):
            return None
        return normalized

    def _dedupe_review_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        result: List[Dict[str, Any]] = []
        for item in items:
            title = str(item.get("title") or item.get("label") or "").strip()
            reason = str(item.get("reason") or "").strip()
            key = (title, reason)
            if not title or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _backfill_card_evidence(
        self,
        cards: List[Dict[str, Any]],
        fallback_cards: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fallback_by_type = {str(card.get("type", "")): card for card in fallback_cards}
        for card in cards:
            card_type = str(card.get("type", ""))
            if card_type not in {"summary", "case_profile", "request_breakdown", "procedure_check", "pending", "risk", "claims", "procedure_risk"}:
                continue
            fallback_items = list((fallback_by_type.get(card_type) or {}).get("items") or [])
            card["items"] = self._backfill_review_items(
                list(card.get("items") or []),
                fallback_items,
                evidence_catalog,
            )
        return cards

    def _backfill_review_items(
        self,
        items: List[Dict[str, Any]],
        fallback_items: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        for index, item in enumerate(items):
            if item.get("evidence_refs"):
                continue
            fallback_item = fallback_items[index] if index < len(fallback_items) else None
            fallback_refs = list((fallback_item or {}).get("evidence_refs") or [])
            item["evidence_refs"] = fallback_refs or self._anchor_evidence_refs(evidence_catalog)
        return items

    def _make_evidence_ref(
        self,
        *,
        evidence_id: str,
        quote: str,
        start: int,
        end: int,
        structure: Optional[Dict[str, Any]],
        fallback_block_type: str,
        forced_page: Optional[int] = None,
        evidence_quality: Optional[str] = None,
    ) -> Dict[str, Any]:
        page_map = self._build_page_offset_map(structure)
        page_entry = None
        page_number = forced_page
        if forced_page is None and page_map and start >= 0:
            for item in page_map:
                if item["start"] <= start < item["end"]:
                    page_entry = item["page"]
                    page_number = item["page_number"]
                    break
        elif page_map and forced_page is not None:
            page_entry = next(
                (item["page"] for item in page_map if item["page_number"] == forced_page),
                None,
            )

        final_quality = evidence_quality or "high"
        if page_entry and str(page_entry.get("source", "")).startswith("ocr"):
            ocr_quality = str(page_entry.get("ocr_quality", "")).strip().lower()
            if ocr_quality in {"low", "medium"}:
                final_quality = "low" if ocr_quality == "low" else "medium"

        return {
            "id": evidence_id,
            "quote": quote.strip()[:220],
            "start": max(0, start),
            "end": max(max(0, start), end),
            "page": page_number,
            "block_type": fallback_block_type,
            "evidence_quality": final_quality,
        }

    def _build_page_offset_map(self, structure: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(structure, dict):
            return []
        pages = structure.get("pages")
        if not isinstance(pages, list):
            return []

        offset = 0
        page_map: List[Dict[str, Any]] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            text = str(page.get("text", "")).strip()
            if not text:
                continue
            start = offset
            end = start + len(text)
            page_map.append(
                {
                    "page_number": int(page.get("page_number", len(page_map) + 1) or (len(page_map) + 1)),
                    "start": start,
                    "end": end,
                    "page": page,
                }
            )
            offset = end + 2
        return page_map

    def _find_matching_evidence(
        self,
        evidence_catalog: Iterable[Dict[str, Any]],
        start: int,
        end: int,
        quote: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_quote = quote.strip()
        for evidence in evidence_catalog:
            if int(evidence.get("start", -1)) == start and int(evidence.get("end", -1)) == end:
                return evidence
            if normalized_quote and normalized_quote == str(evidence.get("quote", "")).strip():
                return evidence
        return None

    def _find_evidence_for_quote(
        self,
        evidence_catalog: Iterable[Dict[str, Any]],
        quote: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_quote = quote.strip()
        if not normalized_quote:
            return None
        for evidence in evidence_catalog:
            evidence_quote = str(evidence.get("quote", "")).strip()
            if normalized_quote in evidence_quote or evidence_quote in normalized_quote:
                return evidence
        return None

    def _anchor_evidence_refs(
        self,
        evidence_catalog: List[Dict[str, Any]],
        *quotes: str,
        prefer_last: bool = False,
    ) -> List[Dict[str, Any]]:
        for quote in quotes:
            evidence = self._find_evidence_for_quote(evidence_catalog, quote)
            if evidence:
                return [evidence]
        if not evidence_catalog:
            return []
        return [evidence_catalog[-1] if prefer_last else evidence_catalog[0]]

    def _find_first_pattern_match(
        self,
        text: str,
        patterns: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        for pattern in patterns:
            match = re.search(pattern, text, re.S)
            if not match:
                continue
            group_index = 1 if match.lastindex else 0
            quote = str(match.group(group_index) or "").strip()
            start, end = match.span(group_index)
            if not quote:
                continue
            return {
                "quote": quote,
                "start": start,
                "end": end,
            }
        return None

    def _find_keyword_line_match(
        self,
        text: str,
        keywords: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        offset = 0
        normalized_keywords = tuple(str(keyword).strip() for keyword in keywords if str(keyword).strip())
        for raw_line in text.splitlines():
            line = raw_line.strip()
            current_offset = offset
            offset += len(raw_line) + 1
            if not line:
                continue
            if not any(keyword in line for keyword in normalized_keywords):
                continue
            start_in_line = 0
            for keyword in normalized_keywords:
                keyword_index = line.find(keyword)
                if keyword_index != -1:
                    start_in_line = keyword_index
                    break
            return {
                "quote": line[:220],
                "start": max(0, current_offset + start_in_line),
                "end": min(current_offset + len(raw_line), current_offset + len(line)),
            }
        return None

    def _resolve_match_evidence_refs(
        self,
        *,
        text: str,
        structure: Optional[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        quote: Optional[str],
        start: Optional[int],
        end: Optional[int],
    ) -> List[Dict[str, Any]]:
        normalized_quote = str(quote or "").strip()
        normalized_start = int(start) if start is not None else -1
        normalized_end = int(end) if end is not None else normalized_start
        if normalized_quote:
            existing = self._find_matching_evidence(
                evidence_catalog,
                normalized_start,
                normalized_end,
                normalized_quote,
            )
            if existing:
                return [existing]
            if normalized_start >= 0:
                return [
                    self._make_evidence_ref(
                        evidence_id="",
                        quote=normalized_quote,
                        start=normalized_start,
                        end=max(normalized_start, normalized_end),
                        structure=structure,
                        fallback_block_type=self._infer_block_type_from_quote(normalized_quote),
                    )
                ]
            spans = self._find_text_spans(text, normalized_quote)
            if spans:
                start_pos, end_pos = spans[0]
                return [
                    self._make_evidence_ref(
                        evidence_id="",
                        quote=normalized_quote,
                        start=start_pos,
                        end=end_pos,
                        structure=structure,
                        fallback_block_type=self._infer_block_type_from_quote(normalized_quote),
                    )
                ]
        return self._anchor_evidence_refs(evidence_catalog)

    def _resolve_fragment_evidence_refs(
        self,
        *,
        text: str,
        structure: Optional[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        fragment: str,
        fallback_refs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_fragment = str(fragment or "").strip()
        if not normalized_fragment:
            return fallback_refs
        spans = self._find_text_spans(text, normalized_fragment)
        if spans:
            start, end = spans[0]
            existing = self._find_matching_evidence(evidence_catalog, start, end, normalized_fragment)
            if existing:
                return [existing]
            return [
                self._make_evidence_ref(
                    evidence_id="",
                    quote=normalized_fragment,
                    start=start,
                    end=end,
                    structure=structure,
                    fallback_block_type=self._infer_block_type_from_quote(normalized_fragment),
                )
            ]
        return fallback_refs

    def _infer_block_type_from_quote(self, quote: str) -> str:
        normalized = quote.strip()
        if not normalized:
            return "text"
        if "\t" in normalized or re.search(r"\s{3,}", normalized):
            return "table"
        if len(normalized) > 36:
            return "paragraph"
        return "line"

    def _normalize_group_text(self, value: str) -> str:
        return re.sub(r"\s+", "", value or "").strip().lower()

    def _find_text_spans(self, text: str, needle: str) -> List[tuple[int, int]]:
        normalized_needle = str(needle or "").strip()
        if not normalized_needle:
            return []
        start = 0
        results: List[tuple[int, int]] = []
        while True:
            index = text.find(normalized_needle, start)
            if index == -1:
                break
            results.append((index, index + len(normalized_needle)))
            start = index + len(normalized_needle)
        return results

    def _infer_ocr_mode(self, metadata: Dict[str, Any]) -> str:
        ocr_pages = int(metadata.get("ocr_pages", 0) or 0)
        if ocr_pages <= 0:
            return "native_or_mixed"
        effective_review_model = str(
            metadata.get("ocr_review_model")
            or metadata.get("effective_ocr_model")
            or metadata.get("ocr_model")
            or ""
        ).lower()
        return "ocr_pro" if effective_review_model.startswith("qwen3.5:27b") else "ocr_standard"
