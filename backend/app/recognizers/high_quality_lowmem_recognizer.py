"""High-quality low-memory recognizer for Chinese desensitization."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.recognizer_base import BaseRecognizer, RecognizerResult
from app.services.chinese_ner_service import ChineseNERService
from app.services.chinese_uie_service import ChineseUIEService
from app.services.contract_structure_backfill_service import ContractStructureBackfillService
from app.services.lowmem_entity_utils import NON_ENTITY_ROLE_TERMS
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
        "POSITION",
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
        "DATE",
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
        self.snippet_scheduler = RiskSnippetScheduler()
        self.review_service = QwenFragmentReviewService()
        self.merge_service = RecallFirstEntityMergeService()
        self.last_run_metadata: Dict[str, object] = {}
        self.last_run_artifacts: Dict[str, object] = {}

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
        seed_results = self._normalize_existing_results(kwargs.get("existing_results"))
        results: list[RecognizerResult] = []
        results.extend(seed_results)
        stage_counts: dict[str, int] = {
            "seed_existing": len(seed_results),
            "structured_backfill": 0,
            "primary_uie": 0,
            "primary_ner": 0,
            "secondary_ner": 0,
            "pre_review_merged": 0,
            "risk_snippets": 0,
            "review_snippets_selected": 0,
            "qwen_raw_candidates": 0,
            "qwen_review": 0,
            "qwen_new_after_merge": 0,
            "qwen_rejected": 0,
            "post_review_merged": 0,
            "alias_propagation_added": 0,
            "quality_gate_snippets_selected": 0,
            "quality_gate_review": 0,
            "quality_gate_rejected": 0,
            "final": 0,
        }

        quality_flags: list[str] = []
        requires_manual_review = False

        if settings.STRUCTURED_BACKFILL:
            backfill_results = self.backfill_service.extract(text)
            stage_counts["structured_backfill"] = len(backfill_results)
            if backfill_results:
                quality_flags.append("structured_backfill_applied")
            results.extend(backfill_results)

        if settings.ENABLE_PRIMARY_UIE:
            try:
                uie_results = self.uie_service.extract(text)
                stage_counts["primary_uie"] = len(uie_results)
                results.extend(uie_results)
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
                stage_counts["primary_ner"] = len(ner_results)
                results.extend(ner_results)
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
                stage_counts["secondary_ner"] = len(secondary_ner_results)
                results.extend(secondary_ner_results)
            except Exception as exc:
                logger.warning("Secondary Chinese NER extraction failed: %s", exc)
                quality_flags.append("secondary_ner_failed")
            finally:
                if settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE:
                    self.secondary_ner_service.unload()

        self.last_run_artifacts = {
            "pre_review_inputs": [item.to_dict() for item in results],
        }
        merged = self.merge_service.merge(results)
        pre_review_merged = list(merged)
        self.last_run_artifacts["pre_review_merged"] = [item.to_dict() for item in pre_review_merged]
        stage_counts["pre_review_merged"] = len(pre_review_merged)
        snippets = self.snippet_scheduler.build_snippets(text, pre_review_merged)
        stage_counts["risk_snippets"] = len(snippets)
        review_snippets = self._select_review_snippets_requiring_semantic_recovery(snippets, pre_review_merged)
        stage_counts["review_snippets_selected"] = len(review_snippets)
        if settings.LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE:
            release_runtime_memory()

        review_result = None
        review_skipped_reason = None
        review_trigger_reasons = sorted({snippet.risk_reason for snippet in review_snippets})
        qwen_contribution = {
            "qwen_raw_candidates": 0,
            "qwen_materialized_entities": 0,
            "qwen_new_entities_after_merge": 0,
            "qwen_confirmed_overlaps": 0,
            "qwen_discarded_entities": 0,
            "qwen_value_level": "not_run",
            "qwen_rejected_entities": 0,
        }
        if settings.ENABLE_QWEN_REVIEW and review_snippets:
            review_result = await self.review_service.review(text, review_snippets, existing_entities=pre_review_merged)
            qwen_contribution = self._measure_qwen_contribution(pre_review_merged, review_result.entities, review_result.raw_candidate_count)
            stage_counts["qwen_raw_candidates"] = int(qwen_contribution["qwen_raw_candidates"])
            stage_counts["qwen_review"] = len(review_result.entities)
            stage_counts["qwen_new_after_merge"] = int(qwen_contribution["qwen_new_entities_after_merge"])
            if review_result.rejected_entities:
                before_reject_count = len(results)
                results = self._apply_review_rejections(results, review_result.rejected_entities)
                rejected_count = max(0, before_reject_count - len(results))
                stage_counts["qwen_rejected"] = rejected_count
                qwen_contribution["qwen_rejected_entities"] = rejected_count
                if rejected_count:
                    qwen_contribution["qwen_value_level"] = "corrective"
                    quality_flags.append("qwen_rejections_applied")
            results.extend(review_result.entities)
            requires_manual_review = requires_manual_review or review_result.requires_manual_review
            if review_result.error:
                quality_flags.append(f"review_warning:{review_result.error}")
            if review_result.arbitration_used:
                quality_flags.append("heavy_arbitration_used")
            if review_result.arbitration_error:
                quality_flags.append(f"arbitration_warning:{review_result.arbitration_error}")
            if settings.REVIEW_UNLOAD_AFTER_TASK:
                self.review_service.unload()
        elif settings.ENABLE_QWEN_REVIEW and not snippets:
            review_skipped_reason = "no_risk_snippets"
        elif settings.ENABLE_QWEN_REVIEW and snippets and not review_snippets:
            review_skipped_reason = "primary_pipeline_sufficient"
        elif settings.ENABLE_QWEN_REVIEW and not self.review_service.installed:
            requires_manual_review = True
            quality_flags.append("review_model_missing")
            review_skipped_reason = "review_model_missing"
        else:
            review_skipped_reason = "review_disabled"

        merged = self.merge_service.merge(results)
        stage_counts["post_review_merged"] = len(merged)
        if settings.ALIAS_PROPAGATION:
            before_count = len(merged)
            merged = self.merge_service.propagate_aliases(text, merged)
            alias_added = max(0, len(merged) - before_count)
            stage_counts["alias_propagation_added"] = alias_added
            if alias_added:
                quality_flags.append("alias_propagation_applied")

        if settings.ENABLE_QWEN_REVIEW and self.review_service.installed:
            quality_gate_snippets = self._build_final_quality_gate_snippets(text, merged)
            stage_counts["quality_gate_snippets_selected"] = len(quality_gate_snippets)
            if quality_gate_snippets:
                gate_result = await self.review_service.review(
                    text,
                    quality_gate_snippets,
                    existing_entities=merged,
                    max_snippets=len(quality_gate_snippets),
                )
                stage_counts["quality_gate_review"] = len(gate_result.entities)
                if gate_result.rejected_entities:
                    before_gate_count = len(merged)
                    merged = self._apply_review_rejections(merged, gate_result.rejected_entities)
                    rejected_gate_count = max(0, before_gate_count - len(merged))
                    stage_counts["quality_gate_rejected"] = rejected_gate_count
                    if rejected_gate_count:
                        quality_flags.append("quality_gate_rejections_applied")
                if gate_result.entities:
                    merged = self.merge_service.merge([*merged, *gate_result.entities])
                    quality_flags.append("quality_gate_entities_added")
                requires_manual_review = requires_manual_review or gate_result.requires_manual_review
                if gate_result.error:
                    quality_flags.append(f"quality_gate_warning:{gate_result.error}")
                if settings.REVIEW_UNLOAD_AFTER_TASK:
                    self.review_service.unload()

        if self._needs_quality_review(text, merged):
            requires_manual_review = True
            quality_flags.append("quality_anomaly_detected")

        final_results = self._filter_entities(merged, entities)
        stage_counts["final"] = len(final_results)
        logger.info(
            "High-quality low-memory recognition finished: stage_counts=%s, review_used=%s, review_error=%s",
            stage_counts,
            bool(review_result and review_result.model_used),
            review_result.error if review_result else None,
        )
        self.last_run_metadata = self._build_metadata(
            review_result=review_result,
            snippet_count=len(review_snippets),
            requires_manual_review=requires_manual_review,
            quality_flags=quality_flags,
            stage_counts=stage_counts,
            review_skipped_reason=review_skipped_reason,
            review_trigger_reasons=review_trigger_reasons,
            qwen_contribution=qwen_contribution,
        )
        return final_results

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
                    "quality_gate_warning:",
                    "arbitration_warning:",
                    "review_model_missing",
                    "quality_anomaly",
                )
            )
        ]
        return {
            "recognition_profile": settings.get_high_quality_profile_key(),
            "workflow_variant": (
                "local_high_quality_mid_review"
                if settings.is_local_high_quality_mode()
                else "lowmem_mid_review"
            ),
            "review_configured": bool(settings.ENABLE_QWEN_REVIEW),
            "review_dispatched": bool(settings.ENABLE_QWEN_REVIEW and int(stage_counts.get("risk_snippets") or 0) > 0),
            "review_started": bool(review_result is not None),
            "review_completed": bool(
                review_result is not None
                or (
                    settings.ENABLE_QWEN_REVIEW
                    and review_skipped_reason in {"no_risk_snippets", "primary_pipeline_sufficient", "review_model_missing"}
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
                else (settings.get_default_review_llm_model() or settings.MID_REVIEW_MODEL)
            ),
            "review_backend": review_result.review_backend if review_result else None,
            "review_model_configured": settings.get_default_review_llm_model() or settings.MID_REVIEW_MODEL,
            "fast_review_model_configured": settings.FAST_REVIEW_MODEL,
            "review_model_path": str(review.path) if review.path else None,
            "review_model_used": bool(review_result and review_result.model_used),
            "review_model_fallback_used": bool(review_result and review_result.fallback_used),
            "review_model_fallback": settings.REVIEW_MODEL_FALLBACK,
            "review_model_fallback_path": str(fallback.path) if fallback.path else None,
            "review_model_loaded": bool(self.review_service.loaded),
            "review_error": review_result.error if review_result else None,
            "review_snippet_count": snippet_count,
            "review_snippet_scheduled_count": int(stage_counts.get("risk_snippets") or 0),
            "arbitration_model": review_result.arbitration_model if review_result else None,
            "arbitration_model_configured": settings.HEAVY_ARBITRATION_MODEL,
            "arbitration_used": bool(review_result and review_result.arbitration_used),
            "arbitration_snippet_count": int(review_result.arbitration_snippet_count) if review_result else 0,
            "arbitration_error": review_result.arbitration_error if review_result else None,
            "review_skipped_reason": review_skipped_reason,
            "qwen_trigger_reasons": review_trigger_reasons,
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

    def _select_review_snippets_requiring_semantic_recovery(
        self,
        snippets,
        existing_results: List[RecognizerResult],
    ) -> list:
        selected = []
        for snippet in snippets:
            if self._snippet_requires_review(snippet, existing_results):
                selected.append(snippet)
        return selected

    def _snippet_requires_review(self, snippet, existing_results: List[RecognizerResult]) -> bool:
        # Quality-first mode: these blocks are where the 4B baseline used to add
        # the most value. Review them even when the primary models found some
        # entities, because "found something" is not the same as "found the
        # right span and type".
        if snippet.snippet_type in {
            "header_party_block",
            "legal_party_block",
            "definition_block",
            "account_block",
            "address_block",
            "signature_block",
            "conflict_block",
            "ocr_anomaly_block",
        }:
            return True
        if snippet.snippet_type in {"ocr_anomaly_block", "conflict_block"}:
            return True
        if snippet.risk_reason in {"long_document_low_entity_density", "uie_ner_overlap_conflict"}:
            return True
        if self._snippet_has_suspicious_candidate(snippet, existing_results):
            return True

        snippet_text = snippet.text or ""
        if snippet.snippet_type == "definition_block":
            return self._definition_block_requires_review(snippet_text) and not self._snippet_has_entity(snippet, existing_results, {"ALIAS"})
        if snippet.snippet_type == "account_block":
            return not self._snippet_has_entity(snippet, existing_results, {"BANK_NAME", "ACCOUNT_NAME", "CN_BANK_CARD"})
        if snippet.snippet_type == "address_block" or snippet.risk_reason == "residence_address_cue":
            return not self._snippet_has_entity(snippet, existing_results, {"ADDRESS", "LOCATION"})
        if snippet.snippet_type == "legal_party_block" or snippet.risk_reason == "legal_party_cue":
            return not self._snippet_has_entity(snippet, existing_results, {"PERSON", "ORGANIZATION", "COMPANY_NAME", "COURT"})
        if snippet.risk_reason == "role_person_cue":
            return not self._snippet_has_entity(snippet, existing_results, {"PERSON", "PERSON_NAME"})
        if snippet.snippet_type == "header_party_block":
            return not self._snippet_has_entity(snippet, existing_results, {"PERSON", "ORGANIZATION", "COMPANY_NAME", "COURT"})
        if snippet.snippet_type == "narrative_hotspot":
            return True
        return False

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
            if entity.entity_type in {"CASE_NO", "CONTRACT_NO"}:
                if len(normalized) > (48 if entity.entity_type == "CASE_NO" else 80):
                    return True
                if any(token in normalized for token in ("上诉人", "被上诉人", "不服", "提起上诉", "事实与理由", "以下简称", "判决书")):
                    return True
            if entity.entity_type in {"PERSON", "PERSON_NAME"} and len(normalized) > 8:
                if any(token in normalized for token in ("就", "对", "向", "请求", "认为", "法院")):
                    return True
            if entity.entity_type in {"ORGANIZATION", "COMPANY_NAME"} and len(normalized) > 18:
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
        for entity in review_entities:
            if HighQualityLowMemoryRecognizer._covered_by_existing(entity, pre_review_results):
                confirmed_overlaps += 1
            else:
                new_entities += 1
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
            "qwen_discarded_entities": max(0, int(raw_candidate_count or 0) - materialized),
            "qwen_value_level": value_level,
            "qwen_rejected_entities": 0,
        }

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
        if result.source in {"regex", "custom"}:
            return False
        if result.entity_type in HighQualityLowMemoryRecognizer.REVIEW_REJECTION_PROTECTED_TYPES:
            return False
        for rejected in rejected_entities:
            rejected_type = str(rejected.get("type") or "")
            rejected_text = str(rejected.get("text") or "")
            rejected_start = int(rejected.get("start") or -1)
            rejected_end = int(rejected.get("end") or -1)
            if rejected_type and result.entity_type != rejected_type:
                continue
            if rejected_start >= 0 and rejected_end >= 0:
                if result.start == rejected_start and result.end == rejected_end:
                    return True
            if rejected_text and result.text == rejected_text:
                return True
        return False

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
        return normalized

    @staticmethod
    def _filter_entities(results: List[RecognizerResult], entities: Optional[List[str]]) -> List[RecognizerResult]:
        if not entities:
            return results
        allowed = set(entities)
        return [item for item in results if item.entity_type in allowed]

    @staticmethod
    def _needs_quality_review(text: str, results: List[RecognizerResult]) -> bool:
        if len(text) > 1500 and len([item for item in results if item.entity_type not in {"DATE", "AMOUNT"}]) <= 2:
            return True
        labels = ["法定代表人", "开户行", "户名", "签章", "盖章"]
        if any(label in text for label in labels):
            matched_types = {item.entity_type for item in results}
            if not {"PERSON", "BANK_NAME", "ACCOUNT_NAME", "ORGANIZATION"} & matched_types:
                return True
        return False

    @staticmethod
    def _build_final_quality_gate_snippets(text: str, results: List[RecognizerResult]) -> List[RiskSnippet]:
        if not text.strip():
            return []

        max_chars = min(int(settings.REVIEW_MAX_CHARS_PER_SNIPPET or 420), 720)
        candidates = sorted(results, key=lambda item: (item.start, item.end))
        if settings.is_high_quality_lowmem_mode():
            max_snippets = max(
                1,
                len(candidates) + max(1, (len(text) + max_chars - 1) // max_chars),
            )
        else:
            max_snippets = max(1, min(12, int(settings.REVIEW_MAX_SNIPPETS or 8) + 2))
        if not candidates:
            snippets: list[RiskSnippet] = []
            window_count = max(1, min(max_snippets, (len(text) + max_chars - 1) // max_chars))
            if window_count == 1:
                return [RiskSnippet("quality_gate_block", "final_entity_quality_gate", 0, min(len(text), max_chars), text[:max_chars])]
            if len(text) <= max_chars:
                return [RiskSnippet("quality_gate_block", "final_entity_quality_gate", 0, len(text), text)]
            step = max(1, (len(text) - max_chars) // max(window_count - 1, 1))
            for index in range(window_count):
                start = min(max(0, index * step), max(0, len(text) - max_chars))
                end = min(len(text), start + max_chars)
                snippet_text = text[start:end]
                if snippet_text.strip():
                    snippets.append(
                        RiskSnippet(
                            "quality_gate_block",
                            "final_entity_quality_gate_gap",
                            start,
                            end,
                            snippet_text,
                        )
                    )
            return snippets[:max_snippets]

        snippets: list[RiskSnippet] = []
        coverage_windows: list[tuple[int, int]] = []
        covered_end = -1
        for entity in candidates:
            if entity.end <= covered_end:
                continue
            start = max(0, entity.start - 120)
            end = min(len(text), start + max_chars)
            while True:
                next_entity = next(
                    (
                        item
                        for item in candidates
                        if item.start >= entity.start and item.start < end and item.end > end and item.end - start <= max_chars
                    ),
                    None,
                )
                if next_entity is None:
                    break
                end = next_entity.end
            snippets.append(
                RiskSnippet(
                    "quality_gate_block",
                    "final_entity_quality_gate",
                    start,
                    end,
                    text[start:end],
                )
            )
            coverage_windows.append((start, end))
            covered_end = end
            if len(snippets) >= max_snippets:
                break

        if len(snippets) < max_snippets:
            uncovered_ranges: list[tuple[int, int]] = []
            cursor = 0
            for start, end in sorted(coverage_windows):
                if start > cursor:
                    uncovered_ranges.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < len(text):
                uncovered_ranges.append((cursor, len(text)))

            for start, end in sorted(uncovered_ranges, key=lambda item: (-(item[1] - item[0]), item[0])):
                if len(snippets) >= max_snippets:
                    break
                if end - start <= 0:
                    continue
                window_start = start
                while window_start < end and len(snippets) < max_snippets:
                    window_end = min(end, window_start + max_chars)
                    snippet_text = text[window_start:window_end]
                    if snippet_text.strip():
                        snippets.append(
                            RiskSnippet(
                                "quality_gate_block",
                                "final_entity_quality_gate_gap",
                                window_start,
                                window_end,
                                snippet_text,
                            )
                        )
                    window_start = window_end

        deduplicated: list[RiskSnippet] = []
        seen_spans: set[tuple[int, int]] = set()
        for snippet in sorted(snippets, key=lambda item: (item.start, item.end)):
            key = (snippet.start, snippet.end)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            deduplicated.append(snippet)
        return deduplicated[:max_snippets]
