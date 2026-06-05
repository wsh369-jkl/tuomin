from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import normalize_text, similarity
from .models import ConversionDiffCandidate, ConversionPreflightResult


BAD_MODEL_FLAG_FRAGMENTS = (
    "schema_incomplete",
    "schema_repaired",
    "parse_error",
    "model_parse_error",
    "preferred_text_not_in_candidate_context",
    "budget_exhausted",
    "failure_circuit_open",
    "missing_crop",
    "preflight_failed",
    "unavailable",
    "failed",
)


@dataclass
class GuardEvent:
    source: str
    review_id: str
    diff_id: str = ""
    page_no: Optional[int] = None
    original_decision: str = ""
    final_decision: str = ""
    original_verdict: str = ""
    final_verdict: str = ""
    reason: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "review_id": self.review_id,
            "diff_id": self.diff_id,
            "page_no": self.page_no,
            "original_decision": self.original_decision,
            "final_decision": self.final_decision,
            "original_verdict": self.original_verdict,
            "final_verdict": self.final_verdict,
            "reason": self.reason,
            "flags": list(self.flags),
        }


class ModelOutputGuard:
    """Prevent model-only or malformed outputs from becoming confirmed errors.

    v4 treats local LLM/VL calls as reviewers, not truth sources. This guard is
    intentionally conservative: an unsafe model result is preserved in evidence
    but downgraded to a deferred/suspected state so it cannot write a confirmed
    DOCX comment or close a coverage gap.
    """

    VERSION = "model_output_guard_v1"

    def apply(self, *, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        diff_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        events: List[GuardEvent] = []

        for review in preflight_result.qwen_gate_reviews:
            diff = diff_by_id.get(review.diff_id)
            reason = self._qwen_gate_block_reason(review=review, diff=diff)
            if reason:
                original_decision = review.decision
                original_verdict = review.verdict
                review.decision = "defer"
                if review.verdict == "confirmed_error":
                    review.verdict = "suspected_error"
                review.next_route = review.next_route or "needs_human_mapping_review"
                review.reason = self._append_guard_reason(review.reason, reason)
                self._append_flag(review.flags, "model_output_guard_deferred")
                self._append_flag(review.flags, reason)
                events.append(
                    GuardEvent(
                        source="qwen_gate_review",
                        review_id=review.gate_id,
                        diff_id=review.diff_id,
                        page_no=diff.pdf_page_no if diff else None,
                        original_decision=original_decision,
                        final_decision=review.decision,
                        original_verdict=original_verdict,
                        final_verdict=review.verdict,
                        reason=reason,
                        flags=list(review.flags),
                    )
                )

        for review in preflight_result.qwen_vl_reviews:
            diff = diff_by_id.get(review.diff_id)
            reason = self._qwen_vl_block_reason(review=review, diff=diff)
            if reason:
                original_decision = review.decision
                original_verdict = review.verdict
                review.decision = "defer"
                if review.verdict == "supports_pdf":
                    review.verdict = "conflict" if reason == "model_output_guard_conflict" else "unreadable"
                review.next_route = review.next_route or "needs_human_visual_review"
                review.reason = self._append_guard_reason(review.reason, reason)
                self._append_flag(review.flags, "model_output_guard_deferred")
                self._append_flag(review.flags, reason)
                events.append(
                    GuardEvent(
                        source="qwen_vl_review",
                        review_id=review.vl_id,
                        diff_id=review.diff_id,
                        page_no=review.page_no or (diff.pdf_page_no if diff else None),
                        original_decision=original_decision,
                        final_decision=review.decision,
                        original_verdict=original_verdict,
                        final_verdict=review.verdict,
                        reason=reason,
                        flags=list(review.flags),
                    )
                )

        for review in preflight_result.page_text_qwen_reviews:
            reason = self._page_text_qwen_block_reason(review=review)
            if reason:
                original_decision = review.decision
                original_verdict = review.verdict
                review.decision = "defer"
                review.status = "model_output_guard_deferred"
                review.next_route = review.next_route or "needs_human_page_review"
                review.reason = self._append_guard_reason(review.reason, reason)
                self._append_flag(review.flags, "model_output_guard_deferred")
                self._append_flag(review.flags, reason)
                events.append(
                    GuardEvent(
                        source="page_text_qwen_review",
                        review_id=review.review_id,
                        page_no=review.page_no,
                        original_decision=original_decision,
                        final_decision=review.decision,
                        original_verdict=original_verdict,
                        final_verdict=review.verdict,
                        reason=reason,
                        flags=list(review.flags),
                    )
                )

        for review in preflight_result.table_page_vl_reviews:
            reason = self._page_vl_block_reason(review=review, decision="", suspicious_values=review.suspicious_values)
            if reason:
                original_verdict = review.verdict
                review.verdict = "uncertain"
                review.next_route = review.next_route or "needs_human_table_review"
                review.reason = self._append_guard_reason(review.reason, reason)
                review.suspicious_values = []
                self._append_flag(review.flags, "model_output_guard_deferred")
                self._append_flag(review.flags, reason)
                events.append(
                    GuardEvent(
                        source="table_page_vl_review",
                        review_id=review.review_id,
                        page_no=review.page_no,
                        original_verdict=original_verdict,
                        final_verdict=review.verdict,
                        reason=reason,
                        flags=list(review.flags),
                    )
                )

        for review in preflight_result.image_text_vl_reviews:
            reason = self._page_vl_block_reason(
                review=review,
                decision=review.decision,
                suspicious_values=review.suspicious_values,
            )
            if reason:
                original_decision = review.decision
                original_verdict = review.verdict
                review.decision = "defer"
                review.verdict = "uncertain"
                review.next_route = review.next_route or "needs_human_visual_review"
                review.reason = self._append_guard_reason(review.reason, reason)
                review.suspicious_values = []
                self._append_flag(review.flags, "model_output_guard_deferred")
                self._append_flag(review.flags, reason)
                events.append(
                    GuardEvent(
                        source="image_text_vl_review",
                        review_id=review.review_id,
                        page_no=review.page_no,
                        original_decision=original_decision,
                        final_decision=review.decision,
                        original_verdict=original_verdict,
                        final_verdict=review.verdict,
                        reason=reason,
                        flags=list(review.flags),
                    )
                )

        for result in preflight_result.specialist_review_results:
            reason = self._specialist_result_block_reason(result=result)
            if reason:
                original_decision = result.decision
                result.decision = "suspected_error"
                result.status = "model_output_guard_deferred"
                result.next_route = result.next_route or "needs_human_review"
                result.comment_policy = "report_only_until_confirmed"
                result.reason = self._append_guard_reason(result.reason, reason)
                self._append_flag(result.flags, "model_output_guard_deferred")
                self._append_flag(result.flags, reason)
                events.append(
                    GuardEvent(
                        source="specialist_review_result",
                        review_id=result.result_id,
                        page_no=result.page_no,
                        original_decision=original_decision,
                        final_decision=result.decision,
                        reason=reason,
                        flags=list(result.flags),
                    )
                )

        existing_events = [
            item
            for item in (preflight_result.model_output_guard or {}).get("events", [])
            if isinstance(item, dict)
        ]
        new_events = [item.to_dict() for item in events]
        all_events = existing_events + new_events
        summary = {
            "enabled": True,
            "version": self.VERSION,
            "guarded_count": len(all_events),
            "total_guarded_count": len(all_events),
            "events": all_events,
            "source_counts": self._source_counts_from_dicts(all_events),
            "latest_guarded_count": len(all_events),
            "latest_pass_guarded_count": len(new_events),
        }
        preflight_result.model_output_guard = summary
        return summary

    def _qwen_gate_block_reason(self, *, review: Any, diff: Optional[ConversionDiffCandidate]) -> str:
        if review.decision != "allow_report_candidate":
            return ""
        if review.verdict == "model_conflict":
            return "model_output_guard_conflict"
        if review.verdict not in {"confirmed_error", "suspected_error"}:
            return "model_output_guard_non_error_verdict"
        if float(review.confidence or 0.0) <= 0:
            return "model_output_guard_zero_confidence"
        if self._has_bad_model_flags(review.flags):
            return "model_output_guard_bad_model_flags"
        preferred = str(review.preferred_text or "")
        if not self._compact(preferred):
            return "model_output_guard_missing_preferred_text"
        if self._looks_like_free_rewrite(old_text=diff.docx_text if diff else "", new_text=preferred, source_text=diff.pdf_text if diff else ""):
            return "model_output_guard_free_rewrite"
        if diff and not self._preferred_supported_by_diff(preferred=preferred, diff=diff):
            return "model_output_guard_preferred_not_supported_by_pdf_context"
        return ""

    def _qwen_vl_block_reason(self, *, review: Any, diff: Optional[ConversionDiffCandidate]) -> str:
        if review.decision != "allow_report_candidate":
            return ""
        if review.verdict == "conflict":
            return "model_output_guard_conflict"
        if review.verdict != "supports_pdf":
            return "model_output_guard_non_pdf_verdict"
        if float(review.confidence or 0.0) <= 0:
            return "model_output_guard_zero_confidence"
        if self._has_bad_model_flags(review.flags):
            return "model_output_guard_bad_model_flags"
        preferred = str(review.preferred_text or review.visible_text or "")
        if not self._compact(preferred):
            return "model_output_guard_missing_preferred_text"
        if not str(review.crop_path or "").strip():
            return "model_output_guard_missing_visual_evidence"
        if self._looks_like_free_rewrite(old_text=diff.docx_text if diff else "", new_text=preferred, source_text=review.visible_text or (diff.pdf_text if diff else "")):
            return "model_output_guard_free_rewrite"
        if diff and not self._preferred_supported_by_visual_or_diff(preferred=preferred, visible_text=review.visible_text, diff=diff):
            return "model_output_guard_preferred_not_supported_by_visual_context"
        return ""

    def _page_text_qwen_block_reason(self, *, review: Any) -> str:
        if review.decision != "allow_exact_replacements":
            return ""
        if float(review.confidence or 0.0) <= 0:
            return "model_output_guard_zero_confidence"
        if self._has_bad_model_flags(review.flags) or str(review.model_parse_error or "").strip():
            return "model_output_guard_bad_model_flags"
        if not review.suspicious_values:
            return "model_output_guard_missing_replacement_values"
        for value in review.suspicious_values:
            old_text = str(value.get("docx_text") or "")
            new_text = str(value.get("pdf_text") or value.get("visible_text") or "")
            if self._looks_like_free_rewrite(old_text=old_text, new_text=new_text, source_text=review.pdf_text_excerpt):
                return "model_output_guard_free_rewrite"
        return ""

    def _page_vl_block_reason(self, *, review: Any, decision: str, suspicious_values: List[Dict[str, Any]]) -> str:
        if decision and decision != "allow_exact_replacements":
            return ""
        if not suspicious_values:
            return ""
        if not bool(getattr(review, "available", False)):
            return "model_output_guard_unavailable"
        if float(getattr(review, "confidence", 0.0) or 0.0) <= 0:
            return "model_output_guard_zero_confidence"
        if self._has_bad_model_flags(getattr(review, "flags", [])) or str(getattr(review, "model_parse_error", "") or "").strip():
            return "model_output_guard_bad_model_flags"
        image_path = str(getattr(review, "review_image_path", "") or getattr(review, "page_image_path", "") or "").strip()
        if not image_path:
            return "model_output_guard_missing_visual_evidence"
        for value in suspicious_values:
            old_text = str(value.get("docx_text") or "")
            new_text = str(value.get("visible_text") or value.get("pdf_text") or "")
            visible_context = str(getattr(review, "visible_text_excerpt", "") or "")
            if self._looks_like_free_rewrite(old_text=old_text, new_text=new_text, source_text=visible_context):
                return "model_output_guard_free_rewrite"
        return ""

    def _specialist_result_block_reason(self, *, result: Any) -> str:
        if result.decision != "confirmed_error":
            return ""
        flags = list(getattr(result, "flags", []) or [])
        model = str(getattr(result, "model", "") or "")
        refs = [str(ref.get("source") or "") for ref in getattr(result, "evidence_refs", []) or [] if isinstance(ref, dict)]
        model_derived = (
            "qwen" in model.lower()
            or "vl" in model.lower()
            or any("qwen" in ref or "vl" in ref for ref in refs)
            or any("qwen" in flag or "vl" in flag for flag in flags)
        )
        if not model_derived:
            return ""
        if float(result.confidence or 0.0) <= 0:
            return "model_output_guard_zero_confidence"
        if self._has_bad_model_flags(flags):
            return "model_output_guard_bad_model_flags"
        if not self._compact(result.old_text) or not self._compact(result.new_text):
            return "model_output_guard_missing_replacement_text"
        if self._looks_like_free_rewrite(old_text=result.old_text, new_text=result.new_text, source_text=""):
            return "model_output_guard_free_rewrite"
        return ""

    def _preferred_supported_by_diff(self, *, preferred: str, diff: ConversionDiffCandidate) -> bool:
        preferred_key = self._compact(preferred)
        candidates = [
            diff.pdf_text,
            diff.pdf_value,
            diff.reason,
        ]
        return self._supported_by_candidates(preferred_key=preferred_key, candidates=candidates)

    def _preferred_supported_by_visual_or_diff(
        self,
        *,
        preferred: str,
        visible_text: str,
        diff: ConversionDiffCandidate,
    ) -> bool:
        preferred_key = self._compact(preferred)
        visible_key = self._compact(visible_text)
        docx_key = self._compact(diff.docx_value or diff.docx_text)
        if visible_key and docx_key and visible_key == docx_key:
            if preferred_key != visible_key and preferred_key not in visible_key and visible_key not in preferred_key:
                return False
        candidates = [
            visible_text,
            diff.pdf_text,
            diff.pdf_value,
        ]
        return self._supported_by_candidates(preferred_key=preferred_key, candidates=candidates)

    def _supported_by_candidates(self, *, preferred_key: str, candidates: List[Any]) -> bool:
        if not preferred_key:
            return False
        for candidate in candidates:
            candidate_key = self._compact(candidate)
            if not candidate_key:
                continue
            if preferred_key == candidate_key or preferred_key in candidate_key or candidate_key in preferred_key:
                return True
            if len(preferred_key) >= 4 and len(candidate_key) >= 4 and similarity(preferred_key, candidate_key) >= 0.82:
                return True
        return False

    def _looks_like_free_rewrite(self, *, old_text: Any, new_text: Any, source_text: Any) -> bool:
        old_key = self._compact(old_text)
        new_key = self._compact(new_text)
        source_key = self._compact(source_text)
        if not new_key:
            return False
        base_len = max(len(old_key), len(source_key), 1)
        if len(new_key) >= 24 and len(new_key) > base_len * 2.4 and len(new_key) > base_len + 18:
            return True
        if old_key and source_key and len(new_key) >= 18:
            if similarity(new_key, old_key) < 0.18 and similarity(new_key, source_key) < 0.18:
                return True
        return False

    def _has_bad_model_flags(self, flags: Any) -> bool:
        values = [str(flag or "").lower() for flag in (flags or [])]
        return any(fragment in value for value in values for fragment in BAD_MODEL_FLAG_FRAGMENTS)

    def _append_guard_reason(self, reason: str, guard_reason: str) -> str:
        text = str(reason or "").strip()
        suffix = f"模型输出守门降级：{guard_reason}"
        if suffix in text:
            return text
        return f"{text}；{suffix}" if text else suffix

    def _append_flag(self, flags: List[str], flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    def _compact(self, value: Any) -> str:
        text = normalize_text(str(value or ""))
        return re.sub(r"\s+", "", text)

    def _source_counts(self, events: List[GuardEvent]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in events:
            counts[event.source] = counts.get(event.source, 0) + 1
        return counts

    def _source_counts_from_dicts(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in events:
            source = str(event.get("source") or "")
            if not source:
                continue
            counts[source] = counts.get(source, 0) + 1
        return counts
