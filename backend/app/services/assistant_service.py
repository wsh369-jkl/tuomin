"""Independent lawyer assistant workflow result generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.services.review_service import ReviewService


class AssistantWorkflowService(ReviewService):
    """Build structured assistant sections for litigation/execution workflows."""

    SUPPORTED_TYPES = {
        "legal_application": "申请/诉讼类",
        "enforcement_paper": "执行/保全类",
        "judicial_decision": "裁判文书类",
        "evidence_material": "证据材料类",
    }

    def generate_assistant_result(
        self,
        *,
        assistant_id: str,
        filename: str,
        text: str,
        entities: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        structure: Optional[Dict[str, Any]] = None,
        quality_metadata: Optional[Dict[str, Any]] = None,
        llm_model: Optional[str] = None,
        stage_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        quality_metadata = dict(quality_metadata or {})
        fact_bundle = self._build_fact_bundle(
            task_id=assistant_id,
            text=text,
            entities=entities,
            metadata=metadata,
            structure=structure,
            quality_metadata=quality_metadata,
        )
        support_context = self._resolve_support_context(text, metadata)
        evidence_catalog = fact_bundle["evidence_catalog"]
        evidence_by_id = {item["id"]: item for item in evidence_catalog if item.get("id")}

        if support_context["support_mode"] == "supported":
            sections = self._build_supported_sections(
                text=text,
                document_type=support_context["document_type"],
                document_type_label=support_context["document_type_label"],
                entities=entities,
                canonical_groups=fact_bundle["canonical_groups"],
                evidence_catalog=evidence_catalog,
                evidence_by_id=evidence_by_id,
                suspected_misses=fact_bundle["suspected_misses"],
                structure=structure,
            )
            summary = self._build_supported_summary(
                document_type_label=support_context["document_type_label"],
                sections=sections,
            )
        else:
            sections = self._build_limited_sections(
                text=text,
                document_type_label=support_context["document_type_label"],
                entities=entities,
                canonical_groups=fact_bundle["canonical_groups"],
                evidence_catalog=evidence_catalog,
                evidence_by_id=evidence_by_id,
                structure=structure,
                limited_reason=support_context["limited_reason"],
            )
            summary = (
                f"当前材料识别为 {support_context['document_type_label']}，不属于首期重点支持范围，"
                "已降级为材料概览模式。"
            )

        return {
            "assistant_id": assistant_id,
            "filename": filename,
            "document_type": support_context["document_type"],
            "document_type_label": support_context["document_type_label"],
            "support_mode": support_context["support_mode"],
            "support_notice": support_context["support_notice"],
            "summary": summary,
            "sections": sections,
            "text": text,
            "metadata": {
                **metadata,
                "assistant_model": llm_model,
                "ocr_mode": self._infer_ocr_mode(metadata),
                "classification_stage": support_context["classification_stage"],
                "evidence_count": len(evidence_catalog),
                "limited_reason": support_context["limited_reason"],
                "stage_trace": list(stage_trace or []),
            },
        }

    def _resolve_support_context(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        llm_type = str(metadata.get("llm_document_type") or metadata.get("document_type") or "other").strip() or "other"
        classification_stage = "llm_metadata" if metadata.get("llm_document_type") else "heuristic_fallback"

        if llm_type == "legal_application":
            return self._supported_context("legal_application", self.SUPPORTED_TYPES["legal_application"], classification_stage)
        if llm_type == "enforcement_paper":
            return self._supported_context("enforcement_paper", self.SUPPORTED_TYPES["enforcement_paper"], classification_stage)
        if llm_type == "financial_account_material":
            return self._supported_context("evidence_material", self.SUPPORTED_TYPES["evidence_material"], classification_stage)

        if re.search(r"(判决书|裁定书|调解书)", text):
            return self._supported_context("judicial_decision", self.SUPPORTED_TYPES["judicial_decision"], classification_stage)
        if re.search(r"(证据目录|证据清单|证据材料|质证意见)", text):
            return self._supported_context("evidence_material", self.SUPPORTED_TYPES["evidence_material"], classification_stage)
        if re.search(r"(答辩状)", text):
            return self._supported_context("legal_application", self.SUPPORTED_TYPES["legal_application"], classification_stage)

        limited_label = str(metadata.get("llm_document_type_label") or metadata.get("document_type_label") or self._document_type_label(llm_type))
        limited_reason = "当前材料不属于首期重点支持的诉讼/执行/证据材料范围。"
        return {
            "document_type": llm_type,
            "document_type_label": limited_label,
            "support_mode": "limited",
            "support_notice": "当前材料已降级为材料概览模式，不输出程序核对或合同风险审查。",
            "limited_reason": limited_reason,
            "classification_stage": classification_stage,
        }

    def _supported_context(self, document_type: str, document_type_label: str, classification_stage: str) -> Dict[str, Any]:
        return {
            "document_type": document_type,
            "document_type_label": document_type_label,
            "support_mode": "supported",
            "support_notice": "当前材料属于首期重点支持范围，已生成案件首页、请求拆解、程序核对和缺口清单。",
            "limited_reason": "",
            "classification_stage": classification_stage,
        }

    def _build_supported_summary(
        self,
        *,
        document_type_label: str,
        sections: List[Dict[str, Any]],
    ) -> str:
        request_count = len(next((section["items"] for section in sections if section["type"] == "request_breakdown"), []))
        gap_count = len(next((section["items"] for section in sections if section["type"] == "evidence_gaps"), []))
        return (
            f"当前材料识别为 {document_type_label}，已生成案件首页、请求事项拆解和程序信息核对。"
            f"当前共整理请求事项 {request_count} 项，待进一步核对的证据或材料缺口 {gap_count} 项。"
        )

    def _build_supported_sections(
        self,
        *,
        text: str,
        document_type: str,
        document_type_label: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
        suspected_misses: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        case_items = self._build_case_profile_items(
            text=text,
            document_type=document_type,
            document_type_label=document_type_label,
            entities=entities,
            canonical_groups=canonical_groups,
            evidence_catalog=evidence_catalog,
            evidence_by_id=evidence_by_id,
            structure=structure,
        )
        request_items = self._build_request_breakdown_items(
            text=text,
            document_type=document_type,
            evidence_catalog=evidence_catalog,
            structure=structure,
        )
        procedure_items = self._build_procedure_check_items(
            text=text,
            document_type=document_type,
            entities=entities,
            canonical_groups=canonical_groups,
            evidence_catalog=evidence_catalog,
            structure=structure,
        )
        evidence_gap_items = self._build_evidence_gap_items(
            text=text,
            suspected_misses=suspected_misses,
            request_items=request_items,
            evidence_catalog=evidence_catalog,
        )
        missing_material_items = self._build_missing_material_items(
            text=text,
            document_type=document_type,
            procedure_items=procedure_items,
            evidence_catalog=evidence_catalog,
        )

        sections = [
            {"type": "case_overview", "title": "案件首页", "items": self._convert_items(case_items)},
            {"type": "request_breakdown", "title": "请求事项拆解", "items": self._convert_items(request_items)},
            {"type": "procedure_checks", "title": "程序信息核对", "items": self._convert_items(procedure_items)},
            {"type": "evidence_gaps", "title": "证据缺口", "items": self._convert_items(evidence_gap_items, default_status="needs_review")},
            {"type": "missing_materials", "title": "待补材料清单", "items": self._convert_items(missing_material_items, default_status="missing")},
        ]
        return [section for section in sections if section["items"]]

    def _build_limited_sections(
        self,
        *,
        text: str,
        document_type_label: str,
        entities: List[Dict[str, Any]],
        canonical_groups: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
        limited_reason: str,
    ) -> List[Dict[str, Any]]:
        overview_items = self._build_case_profile_items(
            text=text,
            document_type="other",
            document_type_label=document_type_label,
            entities=entities,
            canonical_groups=canonical_groups,
            evidence_catalog=evidence_catalog,
            evidence_by_id=evidence_by_id,
            structure=structure,
        )
        overview_items.insert(
            0,
            {
                "label": "处理范围",
                "value": "材料概览模式",
                "reason": limited_reason,
                "evidence_refs": self._anchor_evidence_refs(evidence_catalog),
                "lawyer_action_hint": "如需更深入办案辅助，请优先上传诉讼、执行、保全或证据材料。",
            },
        )
        return [
            {
                "type": "limited_overview",
                "title": "材料概览",
                "items": self._convert_items(overview_items),
            }
        ]

    def _build_evidence_gap_items(
        self,
        *,
        text: str,
        suspected_misses: List[Dict[str, Any]],
        request_items: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items = list(suspected_misses)
        if request_items and str(request_items[0].get("title") or "") == "请求事项入口待确认":
            items.append(
                {
                    "title": "请求事项证据入口待确认",
                    "severity": "medium",
                    "reason": "当前材料未稳定定位到显式请求事项段落，请求与证据之间的对应关系仍需人工核对。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "请求", "申请", "上诉"),
                    "lawyer_action_hint": "建议人工定位请求事项段落，并核对每一项请求是否有事实和证据支持。",
                }
            )
        if not items and not re.search(r"(证据|附件|证明|提交如下证据)", text):
            items.append(
                {
                    "title": "证据引用线索偏弱",
                    "severity": "medium",
                    "reason": "当前文本中未明显识别到证据、附件或证明材料的引用线索。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog),
                    "lawyer_action_hint": "建议核对文书是否明确列明证据目录、附件编号或证明目的。",
                }
            )
        return items[:8]

    def _build_missing_material_items(
        self,
        *,
        text: str,
        document_type: str,
        procedure_items: List[Dict[str, Any]],
        evidence_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        title_mapping = {
            "案号核对": "案号或案件编号",
            "处理机关核对": "法院/处理机关名称",
            "当事人角色核对": "当事人身份或程序角色信息",
            "请求事项核对": "明确列示的请求事项",
            "签署与附件核对": "落款、签字、签章或附件材料",
        }
        for item in procedure_items:
            if not item.get("severity"):
                continue
            title = str(item.get("title", "")).strip()
            items.append(
                {
                    "title": title_mapping.get(title, "待补材料"),
                    "severity": item.get("severity"),
                    "reason": item.get("reason"),
                    "evidence_refs": list(item.get("evidence_refs") or []),
                    "lawyer_action_hint": item.get("lawyer_action_hint"),
                }
            )

        if document_type == "enforcement_paper" and not re.search(r"(判决书|裁定书|调解书|执行依据|生效法律文书)", text):
            items.append(
                {
                    "title": "执行依据材料",
                    "severity": "high",
                    "reason": "当前执行材料中未明显定位到判决书、裁定书或其他执行依据表达。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "执行", "判决", "裁定"),
                    "lawyer_action_hint": "建议补充生效法律文书或明确执行依据。",
                }
            )

        if re.search(r"(保全)", text) and not re.search(r"(账户|账号|房产|车辆|股权|冻结)", text):
            items.append(
                {
                    "title": "保全对象明细",
                    "severity": "medium",
                    "reason": "文中提到保全，但未明显识别到账户、财产类型或具体保全对象。",
                    "evidence_refs": self._anchor_evidence_refs(evidence_catalog, "保全"),
                    "lawyer_action_hint": "建议补充具体保全对象、账号或财产线索。",
                }
            )

        deduped = self._dedupe_review_items(items)
        return deduped[:8]

    def _convert_items(
        self,
        items: List[Dict[str, Any]],
        *,
        default_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": str(item.get("title", "")).strip() or None,
                    "label": str(item.get("label", "")).strip() or None,
                    "value": str(item.get("value", "")).strip() or None,
                    "status": default_status or ("missing" if item.get("severity") else "ready"),
                    "severity": str(item.get("severity", "")).strip() or None,
                    "reason": str(item.get("reason", "")).strip() or None,
                    "evidence_refs": list(item.get("evidence_refs") or []),
                    "action_hint": str(item.get("lawyer_action_hint") or item.get("action_hint") or "").strip() or None,
                }
            )
        return normalized
