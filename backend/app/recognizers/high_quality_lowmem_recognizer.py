"""High-quality low-memory recognizer for Chinese desensitization."""

from __future__ import annotations

import logging
import re
import hashlib
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.recognizer_base import BaseRecognizer, RecognizerResult
from app.rules import RuleFirstPipeline
from app.rules.default_subject_policy import DEFAULT_SUBJECT_TYPES
from app.rules.default_subject_policy import canonicalize_default_result
from app.rules.default_subject_policy import projected_default_subject_type
from app.rules.subject_ledger import SubjectLedgerAdjudicationResolver
from app.services.chinese_ner_service import ChineseNERService
from app.services.chinese_uie_service import ChineseUIEService
from app.services.contract_structure_backfill_service import ContractStructureBackfillService
from app.services.coverage_first import build_passive_coverage_first_metadata
from app.services.docx_structure_backfill_service import DocxStructureBackfillService
from app.services.lowmem_entity_utils import (
    NON_ENTITY_ROLE_TERMS,
    ORG_SUFFIX_TERMS,
    docx_structure_unit_inventory,
    iter_exact_matches,
    iter_docx_structure_units,
    looks_like_organization_short_name,
    normalize_entity_text,
    build_recognition_view,
    resolve_docx_unit_spans,
)
from app.services.lowmem_model_assets import (
    build_model_asset,
    primary_models_ready,
)
from app.services.lowmem_memory import release_runtime_memory
from app.services.qwen_fragment_review_service import QwenFragmentReviewService
from app.services.recall_first_entity_merge_service import RecallFirstEntityMergeService
from app.services.risk_snippet_scheduler import RiskSnippet, RiskSnippetScheduler

logger = logging.getLogger(__name__)


class HighQualityLowMemoryRecognizer(BaseRecognizer):
    """Local recall-first workflow replacing full-document 4B LLM extraction."""

    ORG_ALIAS_BUSINESS_SUFFIX_TERMS = (
        "科技",
        "技术",
        "建设",
        "工程",
        "建筑",
        "贸易",
        "商贸",
        "实业",
        "投资",
        "管理",
        "咨询",
        "服务",
        "文化",
        "传媒",
        "信息",
        "网络",
        "电子",
        "材料",
        "能源",
        "地产",
        "置业",
        "发展",
    )

    REVIEW_REJECTION_PROTECTED_TYPES = {
        "CN_ID_CARD",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
        "TAX_NO",
        "URL",
        "DATE",
        "AMOUNT",
        "CONTRACT_NO",
        "CASE_NO",
    }

    SUPPORTED_ENTITIES = [
        "PERSON",
        "PERSON_NAME",
        "ORGANIZATION",
        "COMPANY_NAME",
        "LOCATION",
        "ADDRESS",
        "GOVERNMENT",
        "COURT",
        "PROJECT",
        "CONTRACT_NO",
        "CASE_NO",
        "BANK_NAME",
        "ACCOUNT_NAME",
        "LEGAL_REPRESENTATIVE",
        "CONTACT_PERSON",
        "SIGNATORY",
        "ALIAS",
        "CN_ID_CARD",
        "CN_PHONE",
        "LANDLINE_PHONE",
        "CN_BANK_CARD",
        "CN_CREDIT_CODE",
        "EMAIL_ADDRESS",
        "DATE",
        "AMOUNT",
    ]

    def __init__(self) -> None:
        super().__init__(
            name="high_quality_lowmem",
            supported_entities=self.SUPPORTED_ENTITIES,
            supported_language="zh",
            version="1.0.0",
        )
        self.uie_service = ChineseUIEService(settings.PRIMARY_IE_MODEL, backend=settings.PRIMARY_IE_BACKEND)
        self.ner_service = ChineseNERService(settings.PRIMARY_NER_MODEL, backend=settings.PRIMARY_NER_BACKEND, source_name="ner")
        self.secondary_ner_service = ChineseNERService(
            settings.SECONDARY_NER_MODEL,
            backend=settings.SECONDARY_NER_BACKEND,
            source_name="secondary_ner",
        )
        self.backfill_service = ContractStructureBackfillService()
        self.docx_backfill_service = DocxStructureBackfillService(self.backfill_service)
        self.snippet_scheduler = RiskSnippetScheduler()
        self.review_service = QwenFragmentReviewService()
        self.merge_service = RecallFirstEntityMergeService()
        self.rule_first_pipeline = RuleFirstPipeline()
        self.subject_ledger_resolver = SubjectLedgerAdjudicationResolver()
        self.last_run_metadata: Dict[str, object] = {}
        self.last_run_artifacts: Dict[str, object] = {}
        self.last_rule_first_metadata: Dict[str, object] = {}

    def get_last_run_metadata(self, llm_model: Optional[str] = None) -> Dict[str, object]:
        return dict(self.last_run_metadata)

    def get_last_run_artifacts(self) -> Dict[str, object]:
        artifacts: Dict[str, object] = {}
        for key, value in self.last_run_artifacts.items():
            if isinstance(value, list):
                copied_items = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    copied_item = dict(item)
                    metadata = copied_item.get("metadata")
                    if isinstance(metadata, dict):
                        copied_item["metadata"] = dict(metadata)
                    copied_items.append(copied_item)
                artifacts[key] = copied_items
            else:
                artifacts[key] = value
        return artifacts

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        if not self.enabled or not settings.is_high_quality_desensitize_mode():
            return []

        self.last_run_artifacts = {}
        self.last_coverage_first_metadata: Dict[str, object] = {}
        source_structure = kwargs.get("source_structure") if isinstance(kwargs.get("source_structure"), dict) else None
        recognition_view_metadata = self._build_recognition_view_metadata(text)
        docx_unit_metadata = self._build_docx_unit_diagnostic_metadata(text, source_structure)
        seed_results = self._normalize_existing_results(kwargs.get("existing_results"))
        results: list[RecognizerResult] = []
        results.extend(self._canonicalize_default_results(seed_results))
        stage_counts: dict[str, int] = {
            "seed_existing": len(seed_results),
            "structured_backfill": 0,
            "docx_structure_backfill": 0,
            "docx_structure_uie": 0,
            "docx_structure_ner": 0,
            "primary_uie": 0,
            "primary_ner": 0,
            "secondary_ner": 0,
            **recognition_view_metadata,
            **docx_unit_metadata,
            "rule_first_input_candidates": 0,
            "rule_first_format_candidates": 0,
            "rule_first_type_rule_candidates": 0,
            "rule_first_rejected_candidates": 0,
            "rule_first_rejected_review_only_candidates": 0,
            "rule_first_boundary_repairs": 0,
            "rule_first_output_candidates": 0,
            "pre_review_merged": 0,
            "risk_snippets": 0,
            "review_snippets_selected": 0,
            "missing_candidate_review_snippets": 0,
            "missing_candidate_review_snippets_selected": 0,
            "qwen_discovery_snippets": 0,
            "qwen_discovery_snippets_selected": 0,
            "qwen_discovery_raw_candidates": 0,
            "qwen_discovery_materialized_entities": 0,
            "qwen_discovery_rejected_by_gate": 0,
            "qwen_discovery_span_miss": 0,
            "qwen_raw_candidates": 0,
            "qwen_review": 0,
            "qwen_new_after_merge": 0,
            "qwen_rejected": 0,
            "final_deterministic_adjudication_rejected": 0,
            "post_review_merged": 0,
            "alias_propagation_added": 0,
            "quality_gate_snippets_selected": 0,
            "quality_gate_review": 0,
            "quality_gate_rejected": 0,
            "final": 0,
        }

        quality_flags: list[str] = []
        requires_manual_review = False
        coverage_priority_unit_ids: set[str] = set()

        if settings.STRUCTURED_BACKFILL:
            backfill_results = self.backfill_service.extract(text)
            stage_counts["structured_backfill"] = len(backfill_results)
            if backfill_results:
                quality_flags.append("structured_backfill_applied")
                results.extend(self._canonicalize_default_results(backfill_results))

            docx_backfill_results = self.docx_backfill_service.extract(
                text=text,
                source_structure=source_structure,
            )
            stage_counts["docx_structure_backfill"] = len(docx_backfill_results)
            if docx_backfill_results:
                quality_flags.append("docx_structure_backfill_applied")
            if any(
                bool((item.metadata or {}).get("docx_review_required"))
                for item in docx_backfill_results
            ):
                requires_manual_review = True
                quality_flags.append("docx_structure_review_required")
            results.extend(self._canonicalize_default_results(docx_backfill_results))

        self.last_coverage_first_metadata = build_passive_coverage_first_metadata(
            source_structure=source_structure,
            candidates=results,
            source_text=text,
        )
        self.last_run_artifacts["coverage_first_directory_rows"] = list(
            self.last_coverage_first_metadata.get("coverage_first_directory_rows") or []
        )
        self.last_run_artifacts["coverage_first_rewrite_entries"] = list(
            self.last_coverage_first_metadata.get("coverage_first_rewrite_entries") or []
        )
        coverage_priority_unit_ids = {
            str(unit_id or "").strip()
            for unit_id in self.last_coverage_first_metadata.get("coverage_first_review_priority_unit_ids", [])
            if str(unit_id or "").strip()
        }

        if settings.ENABLE_PRIMARY_UIE:
            try:
                uie_results = self.uie_service.extract(text)
                stage_counts.update(self._safe_int_metadata(self.uie_service.last_extract_metadata))
                stage_counts["primary_uie"] = len(uie_results)
                results.extend(self._canonicalize_default_results(uie_results))
                docx_uie_results = self._extract_docx_unit_model_results(
                    text=text,
                    source_structure=source_structure,
                    extractor=self.uie_service,
                    source_name="docx_structure_uie",
                    priority_unit_ids=coverage_priority_unit_ids,
                )
                stage_counts.update(self._consume_docx_unit_model_metadata("docx_structure_uie"))
                stage_counts["docx_structure_uie"] = len(docx_uie_results)
                if docx_uie_results:
                    quality_flags.append("docx_structure_uie_applied")
                    results.extend(self._canonicalize_default_results(docx_uie_results))
            except Exception as exc:
                logger.warning("Chinese UIE extraction failed: %s", exc)
                quality_flags.append("primary_ie_failed")
                requires_manual_review = True
            finally:
                if settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE:
                    self.uie_service.unload()

        if settings.ENABLE_PRIMARY_NER:
            try:
                ner_results = self.ner_service.extract(text)
                stage_counts.update(self._safe_int_metadata(self.ner_service.last_extract_metadata))
                stage_counts["primary_ner"] = len(ner_results)
                results.extend(self._canonicalize_default_results(ner_results))
                docx_ner_results = self._extract_docx_unit_model_results(
                    text=text,
                    source_structure=source_structure,
                    extractor=self.ner_service,
                    source_name="docx_structure_ner",
                    priority_unit_ids=coverage_priority_unit_ids,
                )
                stage_counts.update(self._consume_docx_unit_model_metadata("docx_structure_ner"))
                stage_counts["docx_structure_ner"] = len(docx_ner_results)
                if docx_ner_results:
                    quality_flags.append("docx_structure_ner_applied")
                    results.extend(self._canonicalize_default_results(docx_ner_results))
            except Exception as exc:
                logger.warning("Chinese NER extraction failed: %s", exc)
                quality_flags.append("primary_ner_failed")
                requires_manual_review = True
            finally:
                if settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE:
                    self.ner_service.unload()

        if settings.ENABLE_SECONDARY_NER:
            try:
                secondary_ner_results = self.secondary_ner_service.extract(text)
                stage_counts.update(self._safe_int_metadata(self.secondary_ner_service.last_extract_metadata))
                stage_counts["secondary_ner"] = len(secondary_ner_results)
                results.extend(self._canonicalize_default_results(secondary_ner_results))
            except Exception as exc:
                logger.warning("Secondary Chinese NER extraction failed: %s", exc)
                quality_flags.append("secondary_ner_failed")
            finally:
                if settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE:
                    self.secondary_ner_service.unload()

        rule_first_result = self.rule_first_pipeline.apply(
            text=text,
            results=results,
            source_structure=source_structure,
        )
        results = list(rule_first_result.results)
        self.last_rule_first_metadata = dict(rule_first_result.metadata)
        self.last_run_artifacts["rule_first_rejected_entities"] = [
            item.to_dict() for item in rule_first_result.rejected_results
        ]
        stage_counts["rule_first_input_candidates"] = int(
            self.last_rule_first_metadata.get("rule_first_input_candidate_count") or 0
        )
        stage_counts["rule_first_format_candidates"] = int(
            self.last_rule_first_metadata.get("rule_first_format_candidate_count") or 0
        )
        stage_counts["rule_first_type_rule_candidates"] = int(
            self.last_rule_first_metadata.get("rule_first_type_rule_candidate_count") or 0
        )
        stage_counts["rule_first_rejected_candidates"] = int(
            self.last_rule_first_metadata.get("rule_first_rejected_candidate_count") or 0
        )
        stage_counts["rule_first_rejected_review_only_candidates"] = int(
            self.last_rule_first_metadata.get("rule_first_rejected_review_only_candidate_count") or 0
        )
        stage_counts["rule_first_boundary_repairs"] = int(
            self.last_rule_first_metadata.get("rule_first_boundary_repair_count") or 0
        )
        stage_counts["rule_first_output_candidates"] = int(
            self.last_rule_first_metadata.get("rule_first_output_candidate_count") or 0
        )
        stage_counts["candidate_ledger_entries"] = int(
            self.last_rule_first_metadata.get("candidate_ledger_entry_count") or 0
        )
        stage_counts["candidate_ledger_accepted"] = int(
            self.last_rule_first_metadata.get("candidate_ledger_accepted") or 0
        )
        stage_counts["candidate_ledger_review_only"] = int(
            self.last_rule_first_metadata.get("candidate_ledger_review_only") or 0
        )
        stage_counts["candidate_ledger_rejected"] = int(
            self.last_rule_first_metadata.get("candidate_ledger_rejected") or 0
        )
        stage_counts["candidate_ledger_lost_before_subject_ledger"] = int(
            self.last_rule_first_metadata.get("candidate_ledger_lost_before_subject_ledger") or 0
        )
        subject_ledger_summary = dict(
            self.last_rule_first_metadata.get("rule_first_subject_ledger_summary") or {}
        )
        stage_counts["subject_ledger_occurrences"] = int(
            subject_ledger_summary.get("occurrence_count") or 0
        )
        stage_counts["subject_ledger_subjects"] = int(
            subject_ledger_summary.get("subject_count") or 0
        )
        stage_counts["subject_ledger_review_queue"] = int(
            subject_ledger_summary.get("review_queue_count") or 0
        )
        if stage_counts["rule_first_rejected_candidates"]:
            quality_flags.append("rule_first_rejections_applied")
        if int(self.last_rule_first_metadata.get("rule_first_review_queue_count") or 0) > 0:
            quality_flags.append("rule_first_unresolved_review_queue")
            if not settings.ENABLE_QWEN_REVIEW:
                requires_manual_review = True
        if int(self.last_rule_first_metadata.get("directory_quality_gate_blocking_issue_count") or 0) > 0:
            quality_flags.append("rule_first_directory_quality_gate_blocked")
            requires_manual_review = True

        self.last_coverage_first_metadata = build_passive_coverage_first_metadata(
            source_structure=source_structure,
            candidates=results,
            source_text=text,
        )
        self.last_run_artifacts["coverage_first_directory_rows"] = list(
            self.last_coverage_first_metadata.get("coverage_first_directory_rows") or []
        )
        self.last_run_artifacts["coverage_first_rewrite_entries"] = list(
            self.last_coverage_first_metadata.get("coverage_first_rewrite_entries") or []
        )
        if self.last_coverage_first_metadata.get("coverage_first_enabled"):
            stage_counts["coverage_first_obligations"] = int(
                self.last_coverage_first_metadata.get("coverage_first_obligation_count") or 0
            )
            stage_counts["coverage_first_uncovered_required_obligations"] = int(
                self.last_coverage_first_metadata.get("coverage_first_uncovered_required_obligation_count") or 0
            )
            for flag in self._coverage_first_blocking_flags(self.last_coverage_first_metadata):
                requires_manual_review = True
                quality_flags.append(flag)

        coverage_first_artifacts = {
            "coverage_first_directory_rows": list(
                self.last_coverage_first_metadata.get("coverage_first_directory_rows") or []
            ),
            "coverage_first_rewrite_entries": list(
                self.last_coverage_first_metadata.get("coverage_first_rewrite_entries") or []
            ),
        }
        self.last_run_artifacts = {
            "pre_review_inputs": [item.to_dict() for item in results],
            "rule_first_rejected_entities": [item.to_dict() for item in rule_first_result.rejected_results],
            **coverage_first_artifacts,
        }
        merged = self.merge_service.merge(results)
        pre_review_merged = list(merged)
        self.last_run_artifacts["pre_review_merged"] = [item.to_dict() for item in pre_review_merged]
        stage_counts["pre_review_merged"] = len(pre_review_merged)
        alias_backscan_candidates = self._build_alias_backscan_review_candidates(text, pre_review_merged)
        self.last_run_artifacts["alias_backscan_review_candidates"] = [
            item.to_dict() for item in alias_backscan_candidates
        ]
        stage_counts["alias_backscan_review_candidates"] = len(alias_backscan_candidates)
        review_only_candidates = [
            *rule_first_result.rejected_results,
            *alias_backscan_candidates,
        ]
        snippets = self.snippet_scheduler.build_snippets(
            text,
            pre_review_merged,
            source_structure=source_structure,
            rejected_entities=review_only_candidates,
            max_snippets=self._ordinary_review_budget(),
        )
        stage_counts.update(self._safe_int_metadata(getattr(self.snippet_scheduler, "last_metadata", {})))
        stage_counts["risk_snippets"] = len(snippets)
        stage_counts["missing_candidate_review_snippets"] = sum(
            1 for snippet in snippets if snippet.snippet_type == "missing_candidate_review"
        )
        stage_counts["qwen_discovery_snippets"] = sum(
            1 for snippet in snippets if str(snippet.snippet_type or "") == "qwen_coverage_discovery"
        )
        stage_counts["docx_structure_snippets"] = sum(
            1 for snippet in snippets if snippet.risk_reason.startswith("docx_structure:")
        )
        stage_counts["docx_table_cell_snippets"] = sum(
            1 for snippet in snippets if snippet.snippet_type == "docx_table_cell_block"
        )
        review_snippets = self._select_review_snippets_requiring_semantic_recovery(snippets, pre_review_merged)
        stage_counts["review_snippets_selected"] = len(review_snippets)
        stage_counts["missing_candidate_review_snippets_selected"] = sum(
            1 for snippet in review_snippets if str(snippet.snippet_type or "") == "missing_candidate_review"
        )
        stage_counts["qwen_discovery_snippets_selected"] = sum(
            1 for snippet in review_snippets if str(snippet.snippet_type or "") == "qwen_coverage_discovery"
        )
        stage_counts["docx_structure_snippets_selected"] = sum(
            1 for snippet in review_snippets if str(snippet.risk_reason or "").startswith("docx_structure:")
        )
        stage_counts["docx_table_cell_snippets_selected"] = sum(
            1 for snippet in review_snippets if str(snippet.snippet_type or "") == "docx_table_cell_block"
        )
        if settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE:
            release_runtime_memory()

        review_result = None
        gate_result = None
        review_skipped_reason = None
        review_trigger_reasons = sorted({snippet.risk_reason for snippet in review_snippets})
        review_available = bool(self.review_service.installed)
        qwen_contribution = {
            "qwen_raw_candidates": 0,
            "qwen_materialized_entities": 0,
            "qwen_new_entities_after_merge": 0,
            "qwen_confirmed_overlaps": 0,
            "qwen_discarded_entities": 0,
            "qwen_value_level": "not_run",
            "qwen_rejected_entities": 0,
        }
        deterministic_review_decisions = self._final_deterministic_adjudication_decisions(pre_review_merged)
        deterministic_review_rejections = list(deterministic_review_decisions.get("rejections") or [])
        deterministic_review_entities = QwenFragmentReviewService()._materialize_candidates(
            text,
            {"entities": deterministic_review_decisions.get("entities") or []},
            RiskSnippet(
                "final_subject_adjudication",
                "final_subject:deterministic_global",
                0,
                len(text),
                text,
            ),
            source_name="review_deterministic_decision",
        )
        if deterministic_review_rejections:
            before_review_count = len(results)
            results = self._apply_review_rejections(results, deterministic_review_rejections)
            deterministic_rejected_count = max(0, before_review_count - len(results))
            stage_counts["final_deterministic_adjudication_rejected"] = deterministic_rejected_count
            if deterministic_rejected_count:
                quality_flags.append("final_deterministic_adjudication_applied")
        if deterministic_review_entities:
            results.extend(self._canonicalize_default_results(deterministic_review_entities))
            stage_counts["final_deterministic_adjudication_entities"] = len(deterministic_review_entities)
            quality_flags.append("final_deterministic_adjudication_entities_applied")
        if settings.ENABLE_QWEN_REVIEW and review_available and review_snippets:
            review_result = await self.review_service.review(
                text,
                review_snippets,
                existing_entities=pre_review_merged,
                max_snippets=self._ordinary_review_budget(),
            )
            qwen_contribution = self._measure_qwen_contribution(
                pre_review_merged,
                review_result.entities,
                review_result.raw_candidate_count,
            )
            qwen_contribution["qwen_rejected_entities"] = len(review_result.rejected_entities)
            if review_result.rejected_entities:
                before_review_count = len(results)
                results = self._apply_review_rejections(results, review_result.rejected_entities)
                rejected_review_count = max(0, before_review_count - len(results))
                if rejected_review_count:
                    quality_flags.append("review_rejections_applied")
                    requires_manual_review = True
            ledger_decisions = list((review_result.metadata or {}).get("ledger_conflict_decisions") or [])
            if ledger_decisions:
                results = self._apply_ledger_adjudication_decisions(results, ledger_decisions)
                self._apply_resolved_subject_ledger_metadata(ledger_decisions)
                quality_flags.append("ledger_conflict_adjudication_applied")
                if any(bool(item.get("requires_manual_review")) for item in ledger_decisions):
                    requires_manual_review = True
                    quality_flags.append("ledger_conflict_manual_review_required")
            if review_result.entities:
                results.extend(self._canonicalize_default_results(review_result.entities))
            requires_manual_review = requires_manual_review or review_result.requires_manual_review
            if review_result.error:
                quality_flags.append(f"review_warning:{review_result.error}")
            if not review_result.model_used:
                requires_manual_review = True
                quality_flags.append("review_model_not_used")
        elif settings.ENABLE_QWEN_REVIEW and review_available:
            review_skipped_reason = "no_review_snippets"
        elif settings.ENABLE_QWEN_REVIEW:
            requires_manual_review = True
            quality_flags.append("review_model_missing")
            review_skipped_reason = "review_model_missing"
        else:
            review_skipped_reason = "review_disabled"

        post_review_merge_input_count = len(results)
        merged = self.merge_service.merge(results)
        stage_counts["post_review_merge_input_count"] = post_review_merge_input_count
        stage_counts["qwen_raw_candidates"] = int(qwen_contribution["qwen_raw_candidates"])
        stage_counts["qwen_review"] = len(review_result.entities) if review_result else 0
        stage_counts["qwen_new_after_merge"] = int(qwen_contribution["qwen_new_entities_after_merge"])
        stage_counts["qwen_discovery_new_after_merge"] = int(
            qwen_contribution.get("qwen_discovery_new_entities_after_merge") or 0
        )
        stage_counts["qwen_discovery_confirmed_overlaps"] = int(
            qwen_contribution.get("qwen_discovery_confirmed_overlaps") or 0
        )
        stage_counts["qwen_rejected"] = int(qwen_contribution.get("qwen_rejected_entities") or 0)
        stage_counts["ledger_conflict_decisions"] = int(
            (review_result.metadata or {}).get("ledger_conflict_decision_count") or 0
        ) if review_result else 0
        stage_counts["ledger_conflict_snippets"] = int(
            (review_result.metadata or {}).get("ledger_conflict_snippet_count") or 0
        ) if review_result else sum(
            1
            for snippet in review_snippets
            if str(snippet.snippet_type or "") == "ledger_conflict_adjudication"
            or str(snippet.risk_reason or "").startswith("subject_ledger:")
        )
        if review_result:
            review_metadata = dict(review_result.metadata or {})
            stage_counts["review_scheduled_ledger_snippets"] = int(
                review_metadata.get("review_scheduled_ledger_snippet_count") or 0
            )
            stage_counts["review_scheduled_standard_snippets"] = int(
                review_metadata.get("review_scheduled_standard_snippet_count") or 0
            )
            stage_counts["review_scheduled_missing_candidate_snippets"] = int(
                review_metadata.get("review_scheduled_missing_candidate_snippet_count") or 0
            )
            stage_counts["qwen_discovery_snippets_selected"] = int(
                review_metadata.get("qwen_discovery_snippet_selected_count")
                or stage_counts.get("qwen_discovery_snippets_selected")
                or 0
            )
            stage_counts["qwen_discovery_raw_candidates"] = int(
                review_metadata.get("qwen_discovery_raw_candidate_count") or 0
            )
            stage_counts["qwen_discovery_materialized_entities"] = int(
                review_metadata.get("qwen_discovery_materialized_entity_count") or 0
            )
            stage_counts["qwen_discovery_rejected_by_gate"] = int(
                review_metadata.get("qwen_discovery_rejected_by_gate_count") or 0
            )
            stage_counts["qwen_discovery_span_miss"] = int(
                review_metadata.get("qwen_discovery_span_miss_count") or 0
            )
            stage_counts["review_deterministic_rejections"] = int(
                review_metadata.get("review_deterministic_rejection_count") or 0
            )
            stage_counts["review_deterministic_entity_decisions"] = int(
                review_metadata.get("review_deterministic_entity_decision_count") or 0
            )
            stage_counts["review_deterministic_entities"] = int(
                review_metadata.get("review_deterministic_entity_count") or 0
            )
            stage_counts["review_entity_decision_rejections"] = int(
                review_metadata.get("review_entity_decision_rejection_count") or 0
            )
            stage_counts["review_entity_decision_entities"] = int(
                review_metadata.get("review_entity_decision_entity_count") or 0
            )
        resolved_subject_ledger_summary = dict(
            self.last_rule_first_metadata.get("resolved_subject_ledger_summary") or {}
        )
        stage_counts["resolved_subject_ledger_review_queue"] = int(
            resolved_subject_ledger_summary.get("review_queue_count") or 0
        )
        stage_counts["resolved_subject_ledger_unresolved_subjects"] = int(
            resolved_subject_ledger_summary.get("unresolved_subject_count") or 0
        )
        stage_counts["post_review_merged"] = len(merged)
        scheduled_ledger_count = int(stage_counts.get("review_scheduled_ledger_snippets") or 0)
        ledger_completion = self._ledger_adjudication_completion(
            review_enabled=bool(settings.ENABLE_QWEN_REVIEW),
            review_result=review_result,
            ledger_decision_count=int(stage_counts.get("ledger_conflict_decisions") or 0),
            ledger_snippet_count=int(stage_counts.get("ledger_conflict_snippets") or 0),
            scheduled_ledger_count=scheduled_ledger_count,
            source_review_queue_count=int(stage_counts.get("subject_ledger_review_queue") or 0),
            resolved_review_queue_count=int(
                resolved_subject_ledger_summary.get("review_queue_count") or 0
            ),
        )
        expected_ledger_decisions = int(ledger_completion["expected_decision_count"])
        ledger_adjudication_incomplete = bool(ledger_completion["incomplete"])
        if ledger_adjudication_incomplete:
            requires_manual_review = True
            quality_flags.append("ledger_conflict_adjudication_incomplete")
        if settings.ALIAS_PROPAGATION:
            before_count = len(merged)
            merged = self.merge_service.propagate_aliases(text, merged)
            alias_added = max(0, len(merged) - before_count)
            stage_counts["alias_propagation_added"] = alias_added
            if alias_added:
                quality_flags.append("alias_propagation_applied")

        if self._needs_quality_review(text, merged):
            requires_manual_review = True
            quality_flags.append("quality_anomaly_detected")

        stage_counts.update(
            self._qwen_discovery_projection_stage_counts(
                merged,
                prefix="qwen_discovery_projection_input",
            )
        )
        stage_counts["projection_input_count"] = len(merged)
        final_results = self._project_default_public_results(merged)
        stage_counts.update(
            self._qwen_discovery_projection_stage_counts(
                final_results,
                prefix="qwen_discovery_projection_output",
            )
        )
        stage_counts["projection_output_count"] = len(final_results)
        final_results = self._filter_entities(final_results, entities)
        stage_counts.update(
            self._qwen_discovery_projection_stage_counts(
                final_results,
                prefix="qwen_discovery_filter_output",
            )
        )
        stage_counts["qwen_discovery_lost_before_projection"] = max(
            0,
            int(stage_counts.get("qwen_discovery_projection_input_count") or 0)
            - int(stage_counts.get("qwen_discovery_projection_output_count") or 0),
        )
        stage_counts["qwen_discovery_lost_before_filter"] = max(
            0,
            int(stage_counts.get("qwen_discovery_projection_output_count") or 0)
            - int(stage_counts.get("qwen_discovery_filter_output_count") or 0),
        )
        stage_counts["filter_output_count"] = len(final_results)
        stage_counts["final"] = len(final_results)
        logger.info(
            "High-quality low-memory recognition finished: stage_counts=%s, review_used=%s, review_error=%s",
            stage_counts,
            bool(review_result and review_result.model_used),
            review_result.error if review_result else None,
        )
        self.last_run_metadata = self._build_metadata(
            review_result=review_result,
            gate_result=gate_result,
            snippet_count=len(review_snippets),
            requires_manual_review=requires_manual_review,
            quality_flags=quality_flags,
            stage_counts=stage_counts,
            review_skipped_reason=review_skipped_reason,
            review_trigger_reasons=review_trigger_reasons,
            qwen_contribution=qwen_contribution,
            final_results=final_results,
            ledger_adjudication_incomplete=ledger_adjudication_incomplete,
        )
        return final_results

    @staticmethod
    def _build_recognition_view_metadata(text: str) -> dict[str, int]:
        recognition_view = build_recognition_view(text or "")
        return {
            "recognition_view_original_length": len(recognition_view.original_text),
            "recognition_view_sanitized_length": len(recognition_view.sanitized_text),
            "recognition_view_removed_inline_space_count": recognition_view.removed_inline_space_count,
            "recognition_view_index_map_length": len(recognition_view.sanitized_to_original),
            "recognition_view_original_to_sanitized_length": len(recognition_view.original_to_sanitized),
            "recognition_view_span_remap_fail_count": recognition_view.span_remap_fail_count,
        }

    def _build_docx_unit_diagnostic_metadata(
        self,
        text: str,
        source_structure: dict[str, object] | None,
    ) -> dict[str, int | dict[str, int]]:
        if not isinstance(source_structure, dict):
            return {
                "docx_unit_count": 0,
                "docx_unit_raw_count": 0,
                "docx_unit_page_view_count": 0,
                "docx_unit_page_duplicate_raw_id_count": 0,
                "docx_unit_page_unique_extra_count": 0,
                "docx_unit_raw_duplicate_id_count": 0,
                "docx_unit_raw_duplicate_key_count": 0,
                "docx_unit_count_by_container": {},
                "docx_unit_span_exact_count": 0,
                "docx_unit_span_mapped_count": 0,
                "docx_unit_span_mismatch_count": 0,
                "docx_unit_span_missing_count": 0,
                "docx_unit_span_duplicate_text_count": 0,
                "docx_unit_span_unresolved_count": 0,
            }
        inventory = docx_structure_unit_inventory(source_structure)
        units = resolve_docx_unit_spans(text or "", self._iter_docx_units(source_structure))
        container_counts: dict[str, int] = {}
        exact_count = 0
        mapped_count = 0
        mismatch_count = 0
        missing_count = 0
        duplicate_text_count = 0
        unresolved_count = 0
        for unit in units:
            unit_text = str(unit.get("text") or "")
            container = str(unit.get("container_type") or unit.get("unit_type") or "unknown")
            container_counts[container] = container_counts.get(container, 0) + 1
            if not unit_text:
                missing_count += 1
                unresolved_count += 1
                continue
            resolution = str(unit.get("_span_resolution") or "")
            if resolution == "exact":
                exact_count += 1
                mapped_count += 1
            elif resolution in {"ordered_forward", "sanitized_ordered_forward"}:
                mismatch_count += 1
                mapped_count += 1
            else:
                mismatch_count += 1
                unresolved_count += 1
            first = (text or "").find(unit_text)
            if first >= 0 and (text or "").find(unit_text, first + 1) >= 0:
                duplicate_text_count += 1
        return {
            "docx_unit_count": len(units),
            "docx_unit_raw_count": int(inventory.get("raw_docx_text_unit_count") or 0),
            "docx_unit_page_view_count": int(inventory.get("page_docx_unit_count") or 0),
            "docx_unit_page_duplicate_raw_id_count": int(inventory.get("page_docx_unit_duplicate_raw_id_count") or 0),
            "docx_unit_page_unique_extra_count": int(inventory.get("page_docx_unit_unique_extra_count") or 0),
            "docx_unit_raw_duplicate_id_count": int(inventory.get("raw_docx_text_unit_duplicate_id_count") or 0),
            "docx_unit_raw_duplicate_key_count": int(inventory.get("raw_docx_text_unit_duplicate_key_count") or 0),
            "docx_unit_count_by_container": dict(sorted(container_counts.items())),
            "docx_unit_span_exact_count": exact_count,
            "docx_unit_span_mapped_count": mapped_count,
            "docx_unit_span_mismatch_count": mismatch_count,
            "docx_unit_span_missing_count": missing_count,
            "docx_unit_span_duplicate_text_count": duplicate_text_count,
            "docx_unit_span_unresolved_count": unresolved_count,
        }

    @staticmethod
    def _canonicalize_default_results(results: List[RecognizerResult]) -> List[RecognizerResult]:
        canonicalized: list[RecognizerResult] = []
        for result in results or []:
            updated = canonicalize_default_result(result)
            if updated is not None:
                canonicalized.append(updated)
        return canonicalized

    @staticmethod
    def _safe_int_metadata(metadata: dict[str, object] | None) -> dict[str, int]:
        safe: dict[str, int] = {}
        if not isinstance(metadata, dict):
            return safe
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool):
                safe[key] = int(value)
                continue
            if isinstance(value, int):
                safe[key] = value
        return safe

    def _consume_docx_unit_model_metadata(self, source_name: str) -> dict[str, int]:
        metadata = getattr(self, "_last_docx_unit_model_metadata", None)
        if not isinstance(metadata, dict):
            return {}
        source_metadata = metadata.pop(source_name, None)
        if not isinstance(source_metadata, dict):
            return {}
        return self._safe_int_metadata(source_metadata)

    @classmethod
    def _build_alias_backscan_review_candidates(
        cls,
        text: str,
        confirmed_results: List[RecognizerResult],
    ) -> List[RecognizerResult]:
        if not text or not confirmed_results:
            return []
        occupied = {
            (int(result.start), int(result.end), str(result.entity_type or "").upper())
            for result in confirmed_results
        }
        candidates: list[RecognizerResult] = []
        seen: set[tuple[str, int, int, str]] = set()
        for source in confirmed_results:
            public_type = projected_default_subject_type(source)
            if public_type not in {"ORGANIZATION", "GOVERNMENT"}:
                continue
            source_text = str(source.text or "")
            for alias in cls._alias_backscan_surfaces_for_source(source):
                for start, end, matched_text in iter_exact_matches(text, alias):
                    if (start, end, public_type) in occupied:
                        continue
                    if source.start <= start and end <= source.end:
                        continue
                    if not cls._alias_backscan_context_supports_candidate(text, start, end):
                        continue
                    key = (public_type, start, end, matched_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    source_metadata = dict(source.metadata or {})
                    metadata = {
                        "source_layer": "structure",
                        "trigger": "alias_backscan_review",
                        "recognition_channel": "alias_backscan",
                        "rule_first_rejected": True,
                        "rule_first_reject_stage": "alias_backscan_review",
                        "rule_first_reject_reasons": ["alias_backscan_review_only"],
                        "rule_first_positive_signals": ["confirmed_subject_alias_surface"],
                        "rule_first_negative_signals": [],
                        "rule_first_action": "review",
                        "rule_first_risk_level": "medium",
                        "rule_first_candidate_id": f"alias_backscan:{public_type}:{start}:{end}",
                        "rule_first_source": "alias_backscan_review",
                        "rule_first_source_layer": "structure",
                        "rule_first_entity_type": public_type,
                        "rule_first_recognition_channel": "alias_backscan",
                        "rule_first_review_only_candidate": True,
                        "rule_first_review_only_reasons": ["alias_backscan_confirmed_subject_surface"],
                        "rule_first_trigger": "alias_backscan_review",
                        "missing_candidate_review": True,
                        "review_only_rejected_candidate": True,
                        "alias_backscan_source_text": source_text,
                        "alias_backscan_surface": alias,
                        "alias_backscan_source_subject_id": source_metadata.get("subject_ledger_subject_id"),
                        "alias_backscan_source_canonical_text": source_metadata.get("subject_ledger_canonical_text")
                        or source_metadata.get("canonical_subject_text")
                        or source_text,
                        "alias_backscan_source_canonical_key": source_metadata.get("subject_ledger_canonical_key")
                        or source_metadata.get("canonical_key"),
                        "normalized_text": normalize_entity_text(matched_text),
                        "identity_surface": normalize_entity_text(matched_text),
                        "short_org_candidate": public_type == "ORGANIZATION",
                        "requires_manual_review": True,
                    }
                    if public_type == "GOVERNMENT":
                        metadata["official_institution"] = True
                        metadata["official_institution_family"] = (
                            source_metadata.get("official_institution_family") or "government"
                        )
                    candidates.append(
                        RecognizerResult(
                            entity_type=public_type,
                            start=start,
                            end=end,
                            score=0.78,
                            text=matched_text,
                            source="alias_backscan_review",
                            metadata=metadata,
                        )
                    )
        return candidates

    @classmethod
    def _alias_backscan_surfaces_for_source(cls, source: RecognizerResult) -> list[str]:
        surfaces: set[str] = set()
        metadata = dict(source.metadata or {})
        for key in (
            "definition_alias",
            "alias_surface",
        ):
            value = normalize_entity_text(str(metadata.get(key) or ""))
            if cls._alias_backscan_surface_is_usable(value):
                surfaces.add(value)
        for surface in metadata.get("subject_surfaces") or []:
            value = normalize_entity_text(str(surface or ""))
            if cls._alias_backscan_surface_is_usable(value):
                surfaces.add(value)
        public_type = projected_default_subject_type(source)
        if public_type == "ORGANIZATION":
            core = cls._organization_alias_core(source.text)
            if cls._alias_backscan_surface_is_usable(core):
                surfaces.add(core)
            brand_core = cls._organization_alias_brand_core(core)
            if cls._alias_backscan_surface_is_usable(brand_core):
                surfaces.add(brand_core)
        return sorted(surfaces, key=lambda item: (-len(item), item))[:4]

    @staticmethod
    def _organization_alias_core(value: str) -> str:
        normalized = normalize_entity_text(value)
        if not normalized:
            return ""
        for suffix in sorted(ORG_SUFFIX_TERMS, key=len, reverse=True):
            if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 2:
                normalized = normalized[: -len(suffix)]
                break
        normalized = re.sub(
            r"^(?:中国|中华人民共和国|全国|中央|北京市|上海市|天津市|重庆市|"
            r"北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|"
            r"河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆)",
            "",
            normalized,
        )
        return normalized

    @classmethod
    def _organization_alias_brand_core(cls, value: str) -> str:
        core = normalize_entity_text(value)
        if len(core) < 3:
            return core
        for token in cls.ORG_ALIAS_BUSINESS_SUFFIX_TERMS:
            if core.endswith(token) and len(core) - len(token) >= 2:
                return core[: -len(token)]
        return core

    @staticmethod
    def _alias_backscan_surface_is_usable(value: str) -> bool:
        normalized = normalize_entity_text(value)
        if not 2 <= len(normalized) <= 8:
            return False
        if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{2,8}", normalized):
            return False
        if normalized in NON_ENTITY_ROLE_TERMS:
            return False
        if re.fullmatch(r"\d+", normalized):
            return False
        if not looks_like_organization_short_name(normalized) and len(normalized) < 4:
            return True
        if not looks_like_organization_short_name(normalized):
            return False
        return True

    @staticmethod
    def _alias_backscan_context_supports_candidate(text: str, start: int, end: int) -> bool:
        left = normalize_entity_text(text[max(0, start - 8) : start])
        right = normalize_entity_text(text[end : min(len(text), end + 12)])
        if not left and not right:
            return False
        if any(token in right for token in ("履约", "履行", "结算", "付款", "收款", "交付", "供货", "施工", "对账", "盖章", "落款", "签署", "签订", "负责", "承担")):
            return True
        if any(token in left for token in ("由", "向", "对", "与", "和", "及", "通过", "经由", "根据", "依据", "甲方", "乙方", "丙方")):
            return True
        if any(token in right for token in ("公司", "集团", "单位", "主体", "账户", "材料", "资料", "文件", "信息")):
            return False
        return False

    @staticmethod
    def _project_default_public_results(results: List[RecognizerResult]) -> List[RecognizerResult]:
        projected: list[RecognizerResult] = []
        seen: set[tuple[str, int, int, str]] = set()
        for result in results or []:
            public_type = projected_default_subject_type(result)
            if public_type not in DEFAULT_SUBJECT_TYPES:
                continue
            metadata = dict(result.metadata or {})
            if public_type != result.entity_type:
                metadata.setdefault("internal_entity_type", result.entity_type)
            item = RecognizerResult(
                entity_type=public_type,
                start=result.start,
                end=result.end,
                score=result.score,
                text=result.text,
                source=result.source,
                metadata=metadata,
            )
            key = (item.entity_type, int(item.start), int(item.end), str(item.text or ""))
            if key in seen:
                continue
            seen.add(key)
            projected.append(item)
        projected.sort(key=lambda item: (item.start, item.end, item.entity_type))
        return projected

    @staticmethod
    def _qwen_discovery_projection_stage_counts(
        results: List[RecognizerResult],
        *,
        prefix: str,
    ) -> dict[str, int]:
        discovery = [
            result
            for result in results or []
            if bool((result.metadata or {}).get("qwen_coverage_discovery"))
        ]
        return {
            f"{prefix}_count": len(discovery),
            f"{prefix}_subject_count": sum(
                1
                for result in discovery
                if projected_default_subject_type(result) in DEFAULT_SUBJECT_TYPES
            ),
        }

    def _extract_docx_unit_model_results(
        self,
        *,
        text: str,
        source_structure: dict[str, object] | None,
        extractor,
        source_name: str,
        priority_unit_ids: set[str] | None = None,
    ) -> List[RecognizerResult]:
        if not hasattr(self, "_last_docx_unit_model_metadata"):
            self._last_docx_unit_model_metadata = {}
        units = self._select_docx_units_for_local_model_pass(
            text,
            source_structure,
            priority_unit_ids=priority_unit_ids,
            source_name=source_name,
        )
        diagnostic_metadata = dict(getattr(self, "_last_docx_unit_model_metadata", {}).get(source_name) or {})
        if not text or not units:
            diagnostic_metadata[f"{source_name}_model_result_count"] = 0
            diagnostic_metadata[f"{source_name}_result_count"] = 0
            self._last_docx_unit_model_metadata[source_name] = diagnostic_metadata
            return []
        results: list[RecognizerResult] = []
        for unit in units:
            unit_text = str(unit.get("text") or "")
            start, end = self._resolve_docx_unit_span_for_model(text, unit)
            if start < 0 or end <= start or text[start:end] != unit_text:
                diagnostic_metadata[f"{source_name}_span_mismatch_skip_count"] = (
                    int(diagnostic_metadata.get(f"{source_name}_span_mismatch_skip_count") or 0) + 1
                )
                continue
            try:
                local_results = extractor.extract(unit_text)
            except Exception as exc:
                logger.warning("%s DOCX unit extraction failed: %s", source_name, exc)
                diagnostic_metadata[f"{source_name}_extract_error_count"] = (
                    int(diagnostic_metadata.get(f"{source_name}_extract_error_count") or 0) + 1
                )
                continue
            diagnostic_metadata[f"{source_name}_local_extract_call_count"] = (
                int(diagnostic_metadata.get(f"{source_name}_local_extract_call_count") or 0) + 1
            )
            diagnostic_metadata[f"{source_name}_local_result_count"] = (
                int(diagnostic_metadata.get(f"{source_name}_local_result_count") or 0) + len(local_results)
            )
            for local in local_results:
                global_start = start + int(local.start)
                global_end = start + int(local.end)
                if global_start < start or global_end > end:
                    continue
                if text[global_start:global_end] != local.text:
                    continue
                entity_metadata = dict(local.metadata or {})
                entity_metadata.update(self._docx_unit_model_metadata(unit, source_name, local.source))
                results.append(
                    RecognizerResult(
                        entity_type=local.entity_type,
                        start=global_start,
                        end=global_end,
                        score=float(local.score or 0.0),
                        text=text[global_start:global_end],
                        source=source_name,
                        metadata=entity_metadata,
                    )
                )
        merged = self.merge_service.merge(results)
        diagnostic_metadata[f"{source_name}_model_result_count"] = len(results)
        diagnostic_metadata[f"{source_name}_result_count"] = len(merged)
        self._last_docx_unit_model_metadata[source_name] = diagnostic_metadata
        return merged

    def _apply_resolved_subject_ledger_metadata(self, ledger_decisions: List[dict]) -> None:
        if not ledger_decisions:
            return
        source_ledger = self.last_rule_first_metadata.get("rule_first_subject_ledger")
        if not isinstance(source_ledger, dict):
            return
        resolved = self.subject_ledger_resolver.apply(source_ledger, ledger_decisions)
        self.last_rule_first_metadata["resolved_subject_ledger"] = resolved.ledger
        self.last_rule_first_metadata["resolved_subject_ledger_summary"] = resolved.summary
        self.last_rule_first_metadata["resolved_subject_ledger_subject_id_map"] = dict(resolved.subject_id_map)
        self.last_run_artifacts["resolved_subject_ledger"] = resolved.ledger

    def _select_docx_units_for_local_model_pass(
        self,
        text: str,
        source_structure: dict[str, object] | None,
        *,
        priority_unit_ids: set[str] | None = None,
        source_name: str = "docx_structure_model",
    ) -> list[dict[str, object]]:
        if not isinstance(source_structure, dict):
            if not hasattr(self, "_last_docx_unit_model_metadata"):
                self._last_docx_unit_model_metadata = {}
            self._last_docx_unit_model_metadata[source_name] = {
                f"{source_name}_unit_total_count": 0,
                f"{source_name}_unit_selected_count": 0,
            }
            return []
        candidates: list[tuple[int, int, dict[str, object]]] = []
        seen: set[str] = set()
        priority_unit_ids = {str(unit_id or "").strip() for unit_id in (priority_unit_ids or set()) if str(unit_id or "").strip()}
        inventory = docx_structure_unit_inventory(source_structure)
        units = resolve_docx_unit_spans(text or "", self._iter_docx_units(source_structure))
        supported_containers = {
            "paragraph",
            "table_cell",
            "textbox",
            "header",
            "footer",
            "footnote",
            "endnote",
            "comment",
            "chart",
            "diagram",
            "glossary",
            "xml_text",
        }
        metadata: dict[str, int] = {
            f"{source_name}_unit_total_count": len(units),
            f"{source_name}_unit_raw_count": int(inventory.get("raw_docx_text_unit_count") or 0),
            f"{source_name}_unit_page_view_count": int(inventory.get("page_docx_unit_count") or 0),
            f"{source_name}_unit_page_duplicate_raw_id_count": int(
                inventory.get("page_docx_unit_duplicate_raw_id_count") or 0
            ),
            f"{source_name}_unit_page_unique_extra_count": int(
                inventory.get("page_docx_unit_unique_extra_count") or 0
            ),
            f"{source_name}_unit_raw_duplicate_id_count": int(
                inventory.get("raw_docx_text_unit_duplicate_id_count") or 0
            ),
            f"{source_name}_unit_raw_duplicate_key_count": int(
                inventory.get("raw_docx_text_unit_duplicate_key_count") or 0
            ),
            f"{source_name}_unit_selected_count": 0,
            f"{source_name}_unit_priority_selected_count": 0,
            f"{source_name}_unit_cap_excluded_count": 0,
            f"{source_name}_unit_skip_empty_count": 0,
            f"{source_name}_unit_skip_too_short_count": 0,
            f"{source_name}_unit_long_selected_count": 0,
            f"{source_name}_unit_skip_unsupported_container_count": 0,
            f"{source_name}_unit_skip_unresolved_span_count": 0,
            f"{source_name}_unit_skip_duplicate_count": 0,
        }
        for order, unit in enumerate(units):
            unit_text = str(unit.get("text") or "")
            compact = "".join(unit_text.split())
            if not compact:
                metadata[f"{source_name}_unit_skip_empty_count"] += 1
                continue
            if len(compact) < 2:
                metadata[f"{source_name}_unit_skip_too_short_count"] += 1
                continue
            container_type = str(unit.get("container_type") or unit.get("unit_type") or "")
            unit_id = str(unit.get("unit_id") or "")
            priority_selected = bool(unit_id and unit_id in priority_unit_ids)
            if container_type not in supported_containers:
                metadata[f"{source_name}_unit_skip_unsupported_container_count"] += 1
                continue
            unit_start, unit_end = self._resolve_docx_unit_span_for_model(text, unit)
            if unit_start < 0 or unit_end <= unit_start:
                metadata[f"{source_name}_unit_skip_unresolved_span_count"] += 1
                continue
            key = f"{container_type}:{unit_start}:{unit_end}:{unit_text[:40]}"
            if unit_id:
                key = f"{unit_id}:{key}"
            if key in seen:
                metadata[f"{source_name}_unit_skip_duplicate_count"] += 1
                continue
            seen.add(key)
            priority = self._docx_unit_model_pass_priority(unit, unit_text, priority_selected)
            if len(compact) > 1000:
                metadata[f"{source_name}_unit_long_selected_count"] += 1
            candidates.append((priority, order, dict(unit)))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        max_units = max(1, int(getattr(settings, "DOCX_UNIT_MODEL_PASS_MAX_UNITS", 320) or 320))
        selected = [unit for _priority, _order, unit in candidates[:max_units]]
        metadata[f"{source_name}_unit_selected_count"] = len(selected)
        metadata[f"{source_name}_unit_priority_selected_count"] = sum(
            1
            for unit in selected
            if str(unit.get("unit_id") or "").strip() in priority_unit_ids
        )
        metadata[f"{source_name}_unit_cap_excluded_count"] = max(0, len(candidates) - len(selected))
        if not hasattr(self, "_last_docx_unit_model_metadata"):
            self._last_docx_unit_model_metadata = {}
        self._last_docx_unit_model_metadata[source_name] = metadata
        return selected

    @classmethod
    def _resolve_docx_unit_span_for_model(cls, text: str, unit: dict[str, object]) -> tuple[int, int]:
        start = cls._coerce_int(unit.get("_resolved_start"), -1)
        end = cls._coerce_int(unit.get("_resolved_end"), -1)
        unit_text = str(unit.get("text") or "")
        if start >= 0 and end > start and (text or "")[start:end] == unit_text:
            return start, end
        start = cls._coerce_int(unit.get("start"), -1)
        end = cls._coerce_int(unit.get("end"), -1)
        if start >= 0 and end > start and (text or "")[start:end] == unit_text:
            return start, end
        return -1, -1

    @staticmethod
    def _docx_unit_model_pass_priority(unit: dict[str, object], unit_text: str, priority_selected: bool) -> int:
        compact = re.sub(r"\s+", "", unit_text or "")
        container_type = str(unit.get("container_type") or unit.get("unit_type") or "")
        score = 1000 if priority_selected else 0
        if container_type in {"textbox", "header", "footer", "footnote", "endnote", "comment"}:
            score += 120
        elif container_type == "table_cell":
            score += 80
        elif container_type in {"chart", "diagram", "glossary", "xml_text"}:
            score += 70
        if any(token in compact for token in ("甲方", "乙方", "丙方", "申请人", "被申请人", "原告", "被告", "第三人")):
            score += 80
        if any(token in compact for token in ("法定代表人", "负责人", "联系人", "开户行", "户名", "账户", "账号", "签章", "盖章", "落款")):
            score += 70
        if any(token in compact for token in ("以下简称", "下称", "简称", "又称")):
            score += 60
        if re.search(
            r"[\u4e00-\u9fa5A-Za-z0-9·]{2,}(?:公司|集团|银行|法院|检察院|仲裁委员会|项目|工程|中心|研究院|事务所|商行|工作室|合作社)",
            compact,
        ):
            score += 50
        if re.search(r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|经理|主任|负责人|联系人)", compact):
            score += 35
        return score

    @staticmethod
    def _docx_unit_has_sensitive_or_entity_cue(unit_text: str, container_type: str) -> bool:
        compact = re.sub(r"\s+", "", unit_text or "")
        if not compact:
            return False
        cue_keywords = {
            "甲方",
            "乙方",
            "丙方",
            "委托方",
            "受托方",
            "发包人",
            "承包人",
            "供应商",
            "申请人",
            "被申请人",
            "原告",
            "被告",
            "第三人",
            "联系人",
            "联系电话",
            "法定代表人",
            "负责人",
            "地址",
            "住所",
            "开户行",
            "开户银行",
            "账户",
            "账号",
            "户名",
            "项目",
            "工程",
            "合同",
            "签章",
            "盖章",
            "落款",
            "以下简称",
            "下称",
            "简称",
            "又称",
        }
        if any(keyword in compact for keyword in cue_keywords):
            return True
        entity_shape = bool(
            re.search(
                r"[\u4e00-\u9fa5A-Za-z0-9·]{2,}(?:公司|集团|银行|法院|检察院|仲裁委员会|项目|工程|中心|研究院|事务所|商行|工作室|合作社)",
                compact,
            )
        )
        if container_type in {"table_cell", "textbox", "header", "footer", "footnote", "endnote"}:
            return entity_shape
        if container_type in {"comment", "chart", "diagram", "glossary", "xml_text"}:
            return entity_shape or bool(re.search(r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|经理|主任|负责人|联系人)", compact))
        return False

    def _iter_docx_units(self, source_structure: dict[str, object]):
        yield from iter_docx_structure_units(source_structure)

    def _docx_unit_model_metadata(
        self,
        unit: dict[str, object],
        source_name: str,
        base_source: str,
    ) -> dict[str, object]:
        container_type = str(unit.get("container_type") or unit.get("unit_type") or "")
        rewrite_policy = str(unit.get("rewrite_policy") or "exact")
        return {
            "source": source_name,
            "base_source": base_source,
            "trigger": "docx_structure_unit_model_pass",
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
    def _coerce_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_metadata(
        self,
        *,
        review_result,
        snippet_count: int,
        requires_manual_review: bool,
        quality_flags: list[str],
        stage_counts: dict[str, int],
        review_skipped_reason: Optional[str],
        review_trigger_reasons: list[str],
        qwen_contribution: dict[str, object],
        gate_result=None,
        final_results: Optional[List[RecognizerResult]] = None,
        ledger_adjudication_incomplete: bool = False,
    ) -> Dict[str, object]:
        primary_ie = build_model_asset(settings.PRIMARY_IE_MODEL, role="primary_ie", backend=settings.PRIMARY_IE_BACKEND)
        primary_ner = build_model_asset(settings.PRIMARY_NER_MODEL, role="primary_ner", backend=settings.PRIMARY_NER_BACKEND)
        secondary_ner = build_model_asset(settings.SECONDARY_NER_MODEL, role="secondary_ner", backend=settings.SECONDARY_NER_BACKEND)
        review = build_model_asset(settings.REVIEW_MODEL, role="review", backend=settings.REVIEW_BACKEND)
        fallback = build_model_asset(settings.REVIEW_MODEL_FALLBACK, role="review_fallback", backend=settings.REVIEW_MODEL_FALLBACK_BACKEND)
        blocking_quality_flags = [
            item
            for item in quality_flags
            if item.startswith(
                (
                    "review_warning:",
                    "review_model_missing",
                    "quality_anomaly",
                )
            )
        ]
        docx_front_stage_metadata = self._build_docx_front_stage_metadata(final_results or [])
        coverage_first_metadata = dict(getattr(self, "last_coverage_first_metadata", {}) or {})
        rule_first_metadata = dict(getattr(self, "last_rule_first_metadata", {}) or {})
        subject_ledger_summary = dict(rule_first_metadata.get("rule_first_subject_ledger_summary") or {})
        resolved_subject_ledger_summary = dict(rule_first_metadata.get("resolved_subject_ledger_summary") or {})
        active_subject_ledger_summary = resolved_subject_ledger_summary or subject_ledger_summary
        ledger_decision_count = int(
            (review_result.metadata or {}).get("ledger_conflict_decision_count") or 0
        ) if review_result else 0
        ledger_snippet_count = int(stage_counts.get("ledger_conflict_snippets") or 0)
        scheduled_ledger_count = int(
            (review_result.metadata or {}).get("review_scheduled_ledger_snippet_count") or 0
        ) if review_result else int(stage_counts.get("review_scheduled_ledger_snippets") or 0)
        scheduled_missing_candidate_count = int(
            (review_result.metadata or {}).get("review_scheduled_missing_candidate_snippet_count") or 0
        ) if review_result else int(stage_counts.get("review_scheduled_missing_candidate_snippets") or 0)
        ledger_completion = self._ledger_adjudication_completion(
            review_enabled=bool(settings.ENABLE_QWEN_REVIEW),
            review_result=review_result,
            ledger_decision_count=ledger_decision_count,
            ledger_snippet_count=ledger_snippet_count,
            scheduled_ledger_count=scheduled_ledger_count,
            source_review_queue_count=int(active_subject_ledger_summary.get("review_queue_count") or 0),
            resolved_review_queue_count=int(resolved_subject_ledger_summary.get("review_queue_count") or 0),
        )
        expected_ledger_decisions = int(ledger_completion["expected_decision_count"])
        return {
            "recognition_profile": settings.get_high_quality_profile_key(),
            "workflow_variant": "lowmem_mid_review",
            "review_configured": bool(settings.ENABLE_QWEN_REVIEW),
            "review_dispatched": bool(settings.ENABLE_QWEN_REVIEW and int(stage_counts.get("risk_snippets") or 0) > 0),
            "review_started": bool(
                review_result and review_result.model_used
            ),
            "review_completed": bool(
                not ledger_adjudication_incomplete
                and (
                    settings.ENABLE_QWEN_REVIEW
                    and (
                        review_result is not None
                        or review_skipped_reason in {"review_model_missing", "review_disabled"}
                    )
                )
                or not settings.ENABLE_QWEN_REVIEW
            ),
            "primary_model": "rule/uie/ner",
            "primary_ie_model": settings.PRIMARY_IE_MODEL,
            "primary_ie_model_path": str(primary_ie.path) if primary_ie.path else None,
            "primary_ie_backend": settings.PRIMARY_IE_BACKEND,
            "primary_ie_backend_available": bool(self.uie_service.backend_available),
            "primary_ie_backend_error": self.uie_service.backend_error,
            "primary_ner_model": settings.PRIMARY_NER_MODEL,
            "primary_ner_model_path": str(primary_ner.path) if primary_ner.path else None,
            "primary_ner_backend": settings.PRIMARY_NER_BACKEND,
            "primary_ner_backend_available": bool(self.ner_service.backend_available),
            "primary_ner_backend_error": self.ner_service.backend_error,
            "secondary_ner_model": settings.SECONDARY_NER_MODEL,
            "secondary_ner_model_path": str(secondary_ner.path) if secondary_ner.path else None,
            "secondary_ner_backend": settings.SECONDARY_NER_BACKEND,
            "secondary_ner_backend_available": bool(self.secondary_ner_service.backend_available),
            "secondary_ner_backend_error": self.secondary_ner_service.backend_error,
            "review_model": (
                review_result.model_name
                if review_result and review_result.model_name
                else settings.get_default_review_llm_model()
            ),
            "review_backend": (
                review_result.review_backend
                if review_result
                else None
            ),
            "active_review_stage": (
                "standard_review"
                if review_result and review_result.model_used
                else None
            ),
            "active_review_model": (
                review_result.model_name
                if review_result and review_result.model_name
                else None
            ),
            "active_review_backend": (
                review_result.review_backend
                if review_result
                else None
            ),
            "review_model_configured": settings.get_default_review_llm_model() or settings.REVIEW_MODEL,
            "review_model_path": str(review.path) if review.path else None,
            "review_model_used": bool(review_result and review_result.model_used),
            "review_model_fallback_used": bool(review_result and review_result.fallback_used),
            "review_model_fallback": settings.REVIEW_MODEL_FALLBACK,
            "review_model_fallback_path": str(fallback.path) if fallback.path else None,
            "review_model_loaded": bool(self.review_service.loaded),
            "review_error": review_result.error if review_result else None,
            "review_snippet_count": snippet_count,
            "review_snippet_scheduled_count": int(stage_counts.get("risk_snippets") or 0),
            "ledger_conflict_adjudication_enabled": bool(
                review_result and (review_result.metadata or {}).get("ledger_conflict_adjudication_enabled")
            ),
            "ledger_conflict_decision_count": ledger_decision_count,
            "ledger_conflict_snippet_count": ledger_snippet_count,
            "ledger_conflict_expected_decision_count": expected_ledger_decisions,
            "ledger_conflict_remaining_review_queue_count": int(
                ledger_completion["remaining_review_queue_count"]
            ),
            "ledger_conflict_decisions": list(
                (review_result.metadata or {}).get("ledger_conflict_decisions") or []
            ) if review_result else [],
            "review_scheduled_ledger_snippet_count": int(
                (review_result.metadata or {}).get("review_scheduled_ledger_snippet_count") or 0
            ) if review_result else 0,
            "review_scheduled_standard_snippet_count": int(
                (review_result.metadata or {}).get("review_scheduled_standard_snippet_count") or 0
            ) if review_result else 0,
            "missing_candidate_review_snippet_count": int(
                stage_counts.get("missing_candidate_review_snippets") or 0
            ),
            "missing_candidate_review_snippet_selected_count": int(
                stage_counts.get("missing_candidate_review_snippets_selected") or 0
            ),
            "review_scheduled_missing_candidate_snippet_count": scheduled_missing_candidate_count,
            "qwen_discovery_snippet_count": int(stage_counts.get("qwen_discovery_snippets") or 0),
            "qwen_discovery_snippet_selected_count": int(stage_counts.get("qwen_discovery_snippets_selected") or 0),
            "qwen_discovery_unit_count": int(stage_counts.get("qwen_discovery_snippets_selected") or 0),
            "qwen_discovery_raw_candidate_count": int(stage_counts.get("qwen_discovery_raw_candidates") or 0),
            "qwen_discovery_materialized_entity_count": int(
                stage_counts.get("qwen_discovery_materialized_entities") or 0
            ),
            "qwen_discovery_new_entity_count": int(stage_counts.get("qwen_discovery_new_after_merge") or 0),
            "qwen_discovery_confirmed_overlap_count": int(
                stage_counts.get("qwen_discovery_confirmed_overlaps") or 0
            ),
            "qwen_discovery_projection_input_count": int(
                stage_counts.get("qwen_discovery_projection_input_count") or 0
            ),
            "qwen_discovery_projection_input_subject_count": int(
                stage_counts.get("qwen_discovery_projection_input_subject_count") or 0
            ),
            "qwen_discovery_projection_output_count": int(
                stage_counts.get("qwen_discovery_projection_output_count") or 0
            ),
            "qwen_discovery_projection_output_subject_count": int(
                stage_counts.get("qwen_discovery_projection_output_subject_count") or 0
            ),
            "qwen_discovery_filter_output_count": int(
                stage_counts.get("qwen_discovery_filter_output_count") or 0
            ),
            "qwen_discovery_filter_output_subject_count": int(
                stage_counts.get("qwen_discovery_filter_output_subject_count") or 0
            ),
            "qwen_discovery_lost_before_projection_count": int(
                stage_counts.get("qwen_discovery_lost_before_projection") or 0
            ),
            "qwen_discovery_lost_before_filter_count": int(
                stage_counts.get("qwen_discovery_lost_before_filter") or 0
            ),
            "qwen_discovery_rejected_by_gate_count": int(
                stage_counts.get("qwen_discovery_rejected_by_gate") or 0
            ),
            "qwen_discovery_span_miss_count": int(stage_counts.get("qwen_discovery_span_miss") or 0),
            "coverage_discovery_unit_total_count": int(
                stage_counts.get("coverage_discovery_unit_total_count") or 0
            ),
            "coverage_discovery_signal_unit_count": int(
                stage_counts.get("coverage_discovery_signal_unit_count") or 0
            ),
            "coverage_discovery_fully_covered_unit_count": int(
                stage_counts.get("coverage_discovery_fully_covered_unit_count") or 0
            ),
            "coverage_discovery_partial_unit_count": int(
                stage_counts.get("coverage_discovery_partial_unit_count") or 0
            ),
            "coverage_discovery_uncovered_signal_unit_count": int(
                stage_counts.get("coverage_discovery_uncovered_signal_unit_count") or 0
            ),
            "coverage_discovery_snippet_raw_count": int(
                stage_counts.get("coverage_discovery_snippet_raw_count") or 0
            ),
            "risk_snippet_coverage_discovery_raw_count": int(
                stage_counts.get("risk_snippet_coverage_discovery_raw_count") or 0
            ),
            "risk_snippet_coverage_discovery_deduped_count": int(
                stage_counts.get("risk_snippet_coverage_discovery_deduped_count") or 0
            ),
            "alias_backscan_review_candidate_count": int(
                stage_counts.get("alias_backscan_review_candidates") or 0
            ),
            "review_deterministic_rejection_count": int(
                (review_result.metadata or {}).get("review_deterministic_rejection_count") or 0
            ) if review_result else 0,
            "review_deterministic_entity_decision_count": int(
                (review_result.metadata or {}).get("review_deterministic_entity_decision_count") or 0
            ) if review_result else 0,
            "review_deterministic_entity_count": int(
                (review_result.metadata or {}).get("review_deterministic_entity_count") or 0
            ) if review_result else 0,
            "review_entity_decision_rejection_count": int(
                (review_result.metadata or {}).get("review_entity_decision_rejection_count") or 0
            ) if review_result else 0,
            "review_entity_decision_entity_count": int(
                (review_result.metadata or {}).get("review_entity_decision_entity_count") or 0
            ) if review_result else 0,
            "ledger_conflict_adjudication_incomplete": bool(ledger_adjudication_incomplete),
            "review_skipped_reason": review_skipped_reason,
            "review_quality_mode": "rule_first",
            "subject_ledger_enabled": bool(active_subject_ledger_summary.get("subject_ledger_enabled")),
            "subject_ledger_mode": active_subject_ledger_summary.get("subject_ledger_mode"),
            "subject_ledger_occurrence_count": int(active_subject_ledger_summary.get("occurrence_count") or 0),
            "subject_ledger_subject_count": int(active_subject_ledger_summary.get("subject_count") or 0),
            "subject_ledger_review_queue_count": int(active_subject_ledger_summary.get("review_queue_count") or 0),
            "subject_ledger_unresolved_subject_count": int(active_subject_ledger_summary.get("unresolved_subject_count") or 0),
            "resolved_subject_ledger_enabled": bool(resolved_subject_ledger_summary),
            "resolved_subject_ledger_summary": resolved_subject_ledger_summary,
            "resolved_subject_ledger_review_queue_count": int(
                resolved_subject_ledger_summary.get("review_queue_count") or 0
            ),
            "resolved_subject_ledger_unresolved_subject_count": int(
                resolved_subject_ledger_summary.get("unresolved_subject_count") or 0
            ),
            "resolved_subject_ledger_adjudicated_merge_count": int(
                resolved_subject_ledger_summary.get("adjudicated_merge_count") or 0
            ),
            "resolved_subject_ledger_adjudicated_keep_separate_count": int(
                resolved_subject_ledger_summary.get("adjudicated_keep_separate_count") or 0
            ),
            "high_risk_subject_unreviewed_count": 0,
            "same_surface_multi_type_unresolved_count": int(
                rule_first_metadata.get("same_surface_multi_type_unresolved_count") or 0
            ),
            "subject_multi_replacement_count": 0,
            "replacement_reused_by_multi_subject_count": 0,
            **rule_first_metadata,
            **coverage_first_metadata,
            "qwen_trigger_reasons": review_trigger_reasons,
            **docx_front_stage_metadata,
            **qwen_contribution,
            "stage_counts": dict(stage_counts),
            "quality_policy": settings.QUALITY_POLICY,
            "primary_models_unloaded_after_stage": bool(settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE),
            "requires_manual_review": bool(requires_manual_review),
            "quality_gate_passed": not bool(requires_manual_review) and not blocking_quality_flags,
            "quality_flags": sorted(set(quality_flags)),
            "primary_models_ready": primary_models_ready(),
            "review_model_installed": bool(self.review_service.installed),
        }

    @staticmethod
    def _coverage_first_blocking_flags(metadata: Dict[str, object]) -> list[str]:
        if not isinstance(metadata, dict) or not metadata.get("coverage_first_enabled"):
            return []
        checks = (
            ("coverage_first_uncovered_required_obligation_count", "coverage_first_uncovered_required_obligation"),
            ("coverage_first_unrewritable_obligation_count", "coverage_first_unrewritable_obligation"),
            ("coverage_first_hard_arbitration_task_count", "coverage_first_hard_arbitration_required"),
            ("coverage_first_hard_identity_conflict_count", "coverage_first_hard_identity_conflict"),
            ("coverage_first_directory_replacement_conflict_count", "coverage_first_directory_replacement_conflict"),
            ("coverage_first_blocked_rewrite_entry_count", "coverage_first_blocked_rewrite_entry"),
            ("coverage_first_post_verifier_blocking_issue_count", "coverage_first_post_verifier_blocked"),
        )
        flags: list[str] = []
        for metric, flag in checks:
            try:
                count = int(metadata.get(metric) or 0)
            except Exception:
                count = 0
            if count > 0 and flag not in flags:
                flags.append(flag)
        if metadata.get("coverage_first_prewrite_verification_passed") is False:
            flags.append("coverage_first_prewrite_verification_failed")
        if metadata.get("coverage_first_post_verifier_ready_to_export") is False:
            flags.append("coverage_first_post_verifier_not_ready")
        return flags

    @staticmethod
    def _build_docx_front_stage_metadata(results: List[RecognizerResult]) -> Dict[str, object]:
        source_counts = {"docx_structure_backfill": 0, "docx_structure_uie": 0, "docx_structure_ner": 0}
        container_counts: dict[str, int] = {}
        part_counts: dict[str, int] = {}
        review_required_count = 0
        exact_rewrite_count = 0
        entity_count = 0
        for result in results:
            source = str(result.source or "")
            metadata = dict(result.metadata or {})
            if source not in source_counts and not str(metadata.get("source") or "").startswith("docx_structure"):
                continue
            if source in source_counts:
                source_counts[source] += 1
            entity_count += 1
            container = str(metadata.get("docx_container_type") or "").strip() or "unknown"
            part_name = str(metadata.get("docx_part_name") or "").strip() or "unknown"
            container_counts[container] = container_counts.get(container, 0) + 1
            part_counts[part_name] = part_counts.get(part_name, 0) + 1
            if bool(metadata.get("docx_review_required")):
                review_required_count += 1
            if str(metadata.get("docx_rewrite_policy") or "exact") == "exact":
                exact_rewrite_count += 1
        return {
            "docx_front_stage_enabled": bool(entity_count),
            "docx_front_stage_entity_count": entity_count,
            "docx_front_stage_source_counts": source_counts,
            "docx_front_stage_container_counts": dict(sorted(container_counts.items())),
            "docx_front_stage_part_counts": dict(sorted(part_counts.items())),
            "docx_front_stage_review_required_count": review_required_count,
            "docx_front_stage_exact_rewrite_count": exact_rewrite_count,
        }

    def _select_review_snippets_requiring_semantic_recovery(
        self,
        snippets,
        existing_results: List[RecognizerResult],
    ) -> list:
        ledger_snippets = []
        missing_candidate_snippets = []
        discovery_snippets = []
        standard_required = []
        standard_fallback = []
        for index, snippet in enumerate(snippets):
            if self._is_ledger_review_snippet(snippet):
                if self._snippet_requires_review(snippet, existing_results):
                    ledger_snippets.append(snippet)
                continue
            if str(getattr(snippet, "snippet_type", "") or "") == "missing_candidate_review":
                if self._snippet_requires_review(snippet, existing_results):
                    missing_candidate_snippets.append(snippet)
                continue
            if (
                str(getattr(snippet, "snippet_type", "") or "") == "qwen_coverage_discovery"
                or str(getattr(snippet, "risk_reason", "") or "").startswith("qwen_discovery:")
            ):
                if self._snippet_requires_review(snippet, existing_results):
                    discovery_snippets.append(snippet)
                continue
            if self._snippet_requires_review(snippet, existing_results):
                standard_required.append((index, snippet))
            elif self._snippet_is_standard_review_fallback(snippet):
                standard_fallback.append((index, snippet))

        ordinary_budget = self._ordinary_review_budget()
        ordered_standard = [
            snippet
            for _, snippet in sorted(
                standard_required,
                key=lambda item: self._standard_review_priority(item[1], item[0], existing_results),
            )
        ]
        if len(ordered_standard) < ordinary_budget:
            ordered_standard.extend(
                snippet
                for _, snippet in sorted(
                    standard_fallback,
                    key=lambda item: self._standard_review_priority(item[1], item[0], existing_results),
                )
                if snippet not in ordered_standard
            )
        structure_required = [
            snippet for snippet in ordered_standard if str(getattr(snippet, "risk_reason", "") or "").startswith("docx_structure:")
        ]
        non_structure_required = [
            snippet for snippet in ordered_standard if not str(getattr(snippet, "risk_reason", "") or "").startswith("docx_structure:")
        ]
        structure_limit = max(2, min(len(structure_required), max(ordinary_budget // 2, 6)))
        remaining_budget = max(1, ordinary_budget - min(structure_limit, len(structure_required)))
        return [
            *ledger_snippets,
            *missing_candidate_snippets,
            *discovery_snippets,
            *structure_required[:structure_limit],
            *non_structure_required[:remaining_budget],
        ]

    def _snippet_requires_review(self, snippet, existing_results: List[RecognizerResult]) -> bool:
        # Quality-first mode: these blocks are where the 4B baseline used to add
        # the most value. Review them even when the primary models found some
        # entities, because "found something" is not the same as "found the
        # right span and type".
        if snippet.snippet_type in {
            "ledger_conflict_adjudication",
            "header_party_block",
            "legal_party_block",
            "definition_block",
            "account_block",
            "address_block",
            "signature_block",
            "conflict_block",
            "ocr_anomaly_block",
            "rule_first_review_block",
            "missing_candidate_review",
            "qwen_coverage_discovery",
            "docx_table_cell_block",
            "docx_textbox_block",
            "docx_header_block",
            "docx_footer_block",
            "docx_footnote_block",
            "docx_endnote_block",
        }:
            return True
        if str(snippet.risk_reason or "").startswith("subject_ledger:"):
            return True
        if snippet.snippet_type in {"ocr_anomaly_block", "conflict_block"}:
            return True
        if snippet.risk_reason in {"long_document_low_entity_density", "uie_ner_overlap_conflict"}:
            return True
        if self._snippet_has_suspicious_candidate(snippet, existing_results):
            return True

        snippet_text = snippet.text or ""
        if snippet.snippet_type == "definition_block":
            return self._definition_block_requires_review(snippet_text) and not self._snippet_has_entity(snippet, existing_results, {"ORGANIZATION"})
        if snippet.snippet_type == "account_block":
            return False
        if snippet.snippet_type == "address_block" or snippet.risk_reason == "residence_address_cue":
            return not self._snippet_has_entity(snippet, existing_results, {"LOCATION"})
        if snippet.snippet_type == "legal_party_block" or snippet.risk_reason == "legal_party_cue":
            return not self._snippet_has_entity(snippet, existing_results, {"PERSON", "ORGANIZATION", "GOVERNMENT"})
        if snippet.risk_reason == "role_person_cue":
            return not self._snippet_has_entity(snippet, existing_results, {"PERSON"})
        if snippet.snippet_type == "header_party_block":
            return not self._snippet_has_entity(snippet, existing_results, {"PERSON", "ORGANIZATION", "GOVERNMENT"})
        if snippet.snippet_type == "narrative_hotspot":
            return True
        return False

    @staticmethod
    def _ordinary_review_budget() -> int:
        return max(1, int(settings.MID_REVIEW_MAX_SNIPPETS or settings.REVIEW_MAX_SNIPPETS or 1))

    @staticmethod
    def _is_ledger_review_snippet(snippet) -> bool:
        return (
            str(getattr(snippet, "snippet_type", "") or "") == "ledger_conflict_adjudication"
            or str(getattr(snippet, "risk_reason", "") or "").startswith("subject_ledger:")
        )

    def _snippet_is_standard_review_fallback(self, snippet) -> bool:
        snippet_type = str(getattr(snippet, "snippet_type", "") or "")
        risk_reason = str(getattr(snippet, "risk_reason", "") or "")
        if self._is_ledger_review_snippet(snippet):
            return False
        if snippet_type in {
            "rule_first_review_block",
            "header_party_block",
            "legal_party_block",
            "definition_block",
            "address_block",
            "signature_block",
            "conflict_block",
            "ocr_anomaly_block",
            "docx_table_cell_block",
            "docx_textbox_block",
            "docx_header_block",
            "docx_footer_block",
            "docx_footnote_block",
            "docx_endnote_block",
            "narrative_hotspot",
            "qwen_coverage_discovery",
        }:
            return True
        return risk_reason.startswith(("rule_first:", "docx_structure:", "qwen_discovery:", "organization_action_cue"))

    def _standard_review_priority(
        self,
        snippet,
        index: int,
        existing_results: List[RecognizerResult],
    ) -> tuple[int, int]:
        snippet_type = str(getattr(snippet, "snippet_type", "") or "")
        risk_reason = str(getattr(snippet, "risk_reason", "") or "")
        if self._snippet_has_suspicious_candidate(snippet, existing_results):
            return (0, index)
        if snippet_type in {"rule_first_review_block", "missing_candidate_review"}:
            return (1, index)
        if snippet_type == "qwen_coverage_discovery" or risk_reason.startswith("qwen_discovery:"):
            return (2, index)
        if snippet_type in {"conflict_block", "ocr_anomaly_block"}:
            return (3, index)
        if risk_reason.startswith("docx_structure:"):
            return (4, index)
        if snippet_type in {"header_party_block", "legal_party_block", "definition_block"}:
            return (5, index)
        if snippet_type in {"address_block", "signature_block"}:
            return (6, index)
        if snippet_type == "narrative_hotspot":
            return (7, index)
        return (8, index)

    @staticmethod
    def _snippet_has_suspicious_candidate(snippet, existing_results: List[RecognizerResult]) -> bool:
        for entity in existing_results:
            if entity.start >= snippet.end or entity.end <= snippet.start:
                continue
            normalized = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", entity.text or "")
            if not normalized:
                continue
            if normalized in {
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
                "法院",
                "人民法院",
                *NON_ENTITY_ROLE_TERMS,
            }:
                return True
            if entity.entity_type == "PERSON" and len(normalized) > 8:
                if any(token in normalized for token in ("就", "对", "向", "请求", "认为", "法院")):
                    return True
            if entity.entity_type in {"ORGANIZATION", "GOVERNMENT"} and len(normalized) > 18:
                if any(token in normalized for token in ("对被告", "不服", "请求", "认为", "提交", "证明")):
                    return True
        return False

    @staticmethod
    def _snippet_has_entity(snippet, existing_results: List[RecognizerResult], entity_types: set[str]) -> bool:
        return any(
            item.entity_type in entity_types and item.start < snippet.end and item.end > snippet.start
            for item in existing_results
        )

    @staticmethod
    def _definition_block_requires_review(snippet_text: str) -> bool:
        aliases = []
        for match in re.finditer(
            r"(?:以下简称|下称|简称|又称)\s*[“\"'‘’]?(?P<alias>[\u4e00-\u9fa5A-Za-z0-9]{2,20})[”\"'‘’]?",
            snippet_text or "",
        ):
            aliases.append(match.group("alias"))
        if not aliases:
            return "以下简称" in snippet_text or "简称" in snippet_text or "下称" in snippet_text or "又称" in snippet_text
        non_sensitive_alias_tokens = ("判决", "裁定", "决定", "协议", "合同", "本案", "原审", "一审", "二审")
        return any(not any(token in alias for token in non_sensitive_alias_tokens) for alias in aliases)

    @staticmethod
    def _measure_qwen_contribution(
        pre_review_results: List[RecognizerResult],
        review_entities: List[RecognizerResult],
        raw_candidate_count: int,
    ) -> dict[str, object]:
        materialized = len(review_entities)
        new_entities = 0
        confirmed_overlaps = 0
        discovery_materialized = 0
        discovery_new_entities = 0
        discovery_confirmed_overlaps = 0
        for entity in review_entities:
            is_discovery = bool((entity.metadata or {}).get("qwen_coverage_discovery"))
            if is_discovery:
                discovery_materialized += 1
            covered = HighQualityLowMemoryRecognizer._covered_by_existing(entity, pre_review_results)
            if covered:
                confirmed_overlaps += 1
                if is_discovery:
                    discovery_confirmed_overlaps += 1
            else:
                new_entities += 1
                if is_discovery:
                    discovery_new_entities += 1
        if new_entities >= 3:
            value_level = "material"
        elif confirmed_overlaps >= 3:
            value_level = "confirmation"
        elif new_entities > 0 or confirmed_overlaps > 0:
            value_level = "low"
        else:
            value_level = "none"
        return {
            "qwen_raw_candidates": int(raw_candidate_count or 0),
            "qwen_materialized_entities": materialized,
            "qwen_new_entities_after_merge": new_entities,
            "qwen_confirmed_overlaps": confirmed_overlaps,
            "qwen_discovery_materialized_entities": discovery_materialized,
            "qwen_discovery_new_entities_after_merge": discovery_new_entities,
            "qwen_discovery_confirmed_overlaps": discovery_confirmed_overlaps,
            "qwen_discarded_entities": max(0, int(raw_candidate_count or 0) - materialized),
            "qwen_value_level": value_level,
            "qwen_rejected_entities": 0,
        }

    @staticmethod
    def _ledger_adjudication_completion(
        *,
        review_enabled: bool,
        review_result,
        ledger_decision_count: int,
        ledger_snippet_count: int,
        scheduled_ledger_count: int = 0,
        source_review_queue_count: int = 0,
        resolved_review_queue_count: int = 0,
    ) -> dict[str, int | bool]:
        """Separate scheduled model review completion from remaining ledger debt."""

        scheduled_count = int(scheduled_ledger_count or 0)
        snippet_count = int(ledger_snippet_count or 0)
        decision_count = int(ledger_decision_count or 0)
        expected_decision_count = scheduled_count if scheduled_count > 0 else snippet_count
        review_expected = bool(review_enabled and expected_decision_count > 0)
        incomplete = bool(
            review_expected
            and (
                not review_result
                or not getattr(review_result, "model_used", False)
                or snippet_count <= 0
                or decision_count < expected_decision_count
            )
        )
        remaining_queue_count = int(resolved_review_queue_count or 0)
        if remaining_queue_count <= 0 and int(source_review_queue_count or 0) > expected_decision_count:
            remaining_queue_count = max(0, int(source_review_queue_count or 0) - expected_decision_count)
        return {
            "expected_decision_count": expected_decision_count,
            "review_expected": review_expected,
            "incomplete": incomplete,
            "remaining_review_queue_count": remaining_queue_count,
        }

    @staticmethod
    def _final_deterministic_adjudication_decisions(results: List[RecognizerResult]) -> dict:
        rows = [QwenFragmentReviewService._entity_dict_for_review(item) for item in results]
        return QwenFragmentReviewService._deterministic_review_entity_decisions(rows)

    @staticmethod
    def _final_deterministic_adjudication_rejections(results: List[RecognizerResult]) -> List[dict]:
        decisions = HighQualityLowMemoryRecognizer._final_deterministic_adjudication_decisions(results)
        return list(decisions.get("rejections") or [])

    @staticmethod
    def _apply_review_rejections(
        results: List[RecognizerResult],
        rejected_entities: List[dict],
    ) -> List[RecognizerResult]:
        if not rejected_entities:
            return results
        return [
            result
            for result in results
            if not HighQualityLowMemoryRecognizer._matches_review_rejection(result, rejected_entities)
        ]

    @staticmethod
    def _matches_review_rejection(result: RecognizerResult, rejected_entities: List[dict]) -> bool:
        result_public_type = projected_default_subject_type(result)
        if result_public_type not in DEFAULT_SUBJECT_TYPES:
            return False
        if result.source in {"regex", "custom"}:
            return False
        if result.entity_type in HighQualityLowMemoryRecognizer.REVIEW_REJECTION_PROTECTED_TYPES:
            return False
        for rejected in rejected_entities:
            rejected_type = projected_default_subject_type(
                RecognizerResult(
                    entity_type=str(rejected.get("type") or ""),
                    start=int(rejected.get("start") or 0),
                    end=int(rejected.get("end") or 0),
                    score=0.0,
                    text=str(rejected.get("text") or ""),
                    source=str(rejected.get("source") or "review_rejection"),
                )
            )
            rejected_text = str(rejected.get("text") or "")
            try:
                rejected_start = int(rejected.get("start"))
            except (TypeError, ValueError):
                rejected_start = -1
            try:
                rejected_end = int(rejected.get("end"))
            except (TypeError, ValueError):
                rejected_end = -1
            if rejected_type and result_public_type != rejected_type:
                continue
            if rejected_start >= 0 and rejected_end >= 0:
                if result.start == rejected_start and result.end == rejected_end:
                    return True
            if (
                rejected_text
                and result.text == rejected_text
                and rejected_start >= 0
                and rejected_end >= 0
                and result.start == rejected_start
                and result.end == rejected_end
            ):
                return True
        return False

    @staticmethod
    def _apply_ledger_adjudication_decisions(
        results: List[RecognizerResult],
        decisions: List[dict],
    ) -> List[RecognizerResult]:
        if not decisions:
            return results
        by_occurrence = {
            str(item.get("occurrence_id") or ""): item
            for item in decisions
            if str(item.get("occurrence_id") or "")
        }
        by_edge = {
            str(item.get("edge_id") or ""): item
            for item in decisions
            if str(item.get("edge_id") or "")
        }
        annotated: List[RecognizerResult] = []
        for result in results:
            metadata = dict(result.metadata or {})
            occurrence_id = str(metadata.get("subject_ledger_occurrence_id") or "")
            edge_id = str(metadata.get("subject_ledger_edge_id") or "")
            decision = by_occurrence.get(occurrence_id) or by_edge.get(edge_id)
            if not decision:
                annotated.append(result)
                continue
            decision_edge_id = str(decision.get("edge_id") or "").strip()
            if decision_edge_id and edge_id and decision_edge_id != edge_id:
                annotated.append(result)
                continue
            source_subject_id = str(
                decision.get("source_subject_id")
                or decision.get("subject_id")
                or ""
            ).strip()
            current_subject_id = str(metadata.get("subject_ledger_subject_id") or "").strip()
            if source_subject_id and current_subject_id and source_subject_id != current_subject_id:
                annotated.append(result)
                continue
            target_subject_id = str(
                decision.get("target_subject_id")
                or metadata.get("subject_ledger_edge_target_subject_id")
                or ""
            ).strip()
            metadata["ledger_adjudication"] = {
                "decision_scope": str(decision.get("decision_scope") or ("edge" if decision_edge_id else "occurrence")),
                "edge_id": decision_edge_id,
                "source_subject_id": source_subject_id,
                "target_subject_id": target_subject_id,
                "edge_relation": str(decision.get("edge_relation") or metadata.get("subject_ledger_edge_relation") or ""),
                "action": str(decision.get("action") or ""),
                "subject_id": str(decision.get("subject_id") or ""),
                "canonical_subject_id": str(decision.get("canonical_subject_id") or ""),
                "confidence": float(decision.get("confidence") or 0.0),
                "reason": str(decision.get("reason") or ""),
                "requires_manual_review": bool(decision.get("requires_manual_review")),
            }
            action = str(decision.get("action") or "").strip().lower()
            if action in {"confirm", "merge_to_canonical"}:
                canonical_subject_id = str(decision.get("canonical_subject_id") or "").strip()
                if action == "merge_to_canonical" and target_subject_id:
                    canonical_subject_id = target_subject_id
                if canonical_subject_id:
                    metadata["subject_ledger_subject_id"] = canonical_subject_id
                    metadata["subject_ledger_adjudicated_subject_id"] = canonical_subject_id
                    metadata["subject_ledger_replacement_key"] = (
                        HighQualityLowMemoryRecognizer._ledger_replacement_key_for_subject(canonical_subject_id)
                    )
                if decision_edge_id:
                    metadata["subject_ledger_adjudicated_edge_id"] = decision_edge_id
                    metadata["subject_ledger_edge_adjudication_action"] = action
                metadata["subject_ledger_status"] = "confirmed_subject"
                metadata["subject_ledger_subject_status"] = "confirmed_subject"
                metadata["requires_manual_review"] = False
            elif action == "keep_separate":
                if decision_edge_id and current_subject_id:
                    separate_subject_id = current_subject_id
                else:
                    separate_subject_id = HighQualityLowMemoryRecognizer._separate_ledger_subject_id(
                        result,
                        decision,
                    )
                metadata["subject_ledger_subject_id"] = separate_subject_id
                metadata["subject_ledger_adjudicated_subject_id"] = separate_subject_id
                metadata["subject_ledger_replacement_key"] = (
                    HighQualityLowMemoryRecognizer._ledger_replacement_key_for_subject(separate_subject_id)
                )
                if not decision_edge_id:
                    metadata["subject_ledger_canonical_text"] = str(result.text or "")
                metadata["subject_ledger_canonical_key"] = metadata["subject_ledger_replacement_key"]
                metadata["subject_ledger_status"] = "confirmed_separate_subject"
                metadata["subject_ledger_subject_status"] = "confirmed_separate_subject"
                if decision_edge_id:
                    metadata["subject_ledger_adjudicated_edge_id"] = decision_edge_id
                    metadata["subject_ledger_edge_adjudication_action"] = "keep_separate"
                metadata["requires_manual_review"] = False
            elif action == "manual_review":
                if decision_edge_id:
                    metadata["subject_ledger_adjudicated_edge_id"] = decision_edge_id
                    metadata["subject_ledger_edge_adjudication_action"] = "manual_review"
                metadata["requires_manual_review"] = True
            annotated.append(
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text=result.text,
                    source=result.source,
                    metadata=metadata,
                )
            )
        return annotated

    @staticmethod
    def _ledger_replacement_key_for_subject(subject_id: str) -> str:
        subject = re.sub(r"[^A-Za-z0-9_]+", "_", str(subject_id or "").strip()).strip("_")
        return f"LEDGER_SUBJECT_{subject}" if subject else ""

    @staticmethod
    def _separate_ledger_subject_id(result: RecognizerResult, decision: dict) -> str:
        occurrence_id = str(decision.get("occurrence_id") or "").strip()
        if occurrence_id:
            return f"{occurrence_id}::SEPARATE"
        digest = hashlib.sha1(
            f"{result.entity_type}:{result.start}:{result.end}:{result.text}".encode("utf-8")
        ).hexdigest()[:12]
        return f"ADJUDICATED_SEPARATE::{digest}"

    @staticmethod
    def _covered_by_existing(entity: RecognizerResult, existing_results: List[RecognizerResult]) -> bool:
        for existing in existing_results:
            same_text = entity.text == existing.text
            same_type = entity.entity_type == existing.entity_type
            overlaps = entity.start < existing.end and entity.end > existing.start
            if same_text and same_type:
                return True
            if overlaps and same_type:
                existing_len = existing.end - existing.start
                entity_len = entity.end - entity.start
                if existing_len >= entity_len:
                    return True
        return False

    @staticmethod
    def _normalize_existing_results(value) -> List[RecognizerResult]:
        if not value:
            return []
        normalized: list[RecognizerResult] = []
        for item in value:
            if isinstance(item, RecognizerResult):
                normalized.append(item)
                continue
            if not isinstance(item, dict):
                continue
            entity_type = str(item.get("entity_type") or item.get("type") or "").strip()
            text = str(item.get("text") or "")
            try:
                start = int(item.get("start"))
                end = int(item.get("end"))
            except (TypeError, ValueError):
                continue
            if not entity_type or not text or end <= start:
                continue
            try:
                score = float(item.get("score") or 0.8)
            except (TypeError, ValueError):
                score = 0.8
            normalized.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=score,
                    text=text,
                    source=str(item.get("source") or "existing"),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        return normalized

    @staticmethod
    def _filter_entities(results: List[RecognizerResult], entities: Optional[List[str]]) -> List[RecognizerResult]:
        if not entities:
            return results
        allowed = set(entities)
        return [item for item in results if item.entity_type in allowed]

    @staticmethod
    def _needs_quality_review(text: str, results: List[RecognizerResult]) -> bool:
        if len(text) > 1500 and len(results) <= 2:
            return True
        labels = ["法定代表人", "签章", "盖章"]
        if any(label in text for label in labels):
            matched_types = {item.entity_type for item in results}
            if not {"PERSON", "ORGANIZATION", "GOVERNMENT"} & matched_types:
                return True
        return False
