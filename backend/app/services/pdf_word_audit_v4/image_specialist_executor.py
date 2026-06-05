from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings

from .models import ConversionPreflightResult, DocxEvidenceUnit, ImagePdfPageReview, PageOcrTextEvidenceReview, PageTextCoverageProfile, SpecialistReviewResult, SpecialistReviewTask


class ImagePageSpecialistExecutor:
    """Promote image/scan page evidence into specialist results.

    Rule-only text artifact hints are retained as routing candidates, but they
    are not allowed to become confirmed replacement comments without PDF OCR or
    VL evidence. This keeps the image workflow model/evidence-led instead of
    turning local cleanup rules into the final judge.
    """

    TASK_TYPE = "image_page_specialist_review"
    COMMENT_POLICY = "comment_if_exact_replacement"
    TARGET_STATUSES = {
        "covered",
        "covered_by_page_ocr",
        "covered_by_nearby_page_ocr",
        "uncovered_docx_content",
        "mapping_uncertain",
        "needs_pdf_ocr",
        "needs_full_page_ocr",
        "needs_region_segmentation",
        "low_confidence_page_review",
        "needs_text_alignment",
        "diff_candidate",
        "table_pending",
    }

    def __init__(self, *, enabled: Optional[bool] = None, max_results: Optional[int] = None) -> None:
        self.enabled = bool(
            getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_ENABLED", True)
            if enabled is None
            else enabled
        )
        self.max_results = max(0, int(max_results if max_results is not None else getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_RESULTS", 9999) or 9999))

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        text_hint_fn: Callable[..., str],
        priority_fn: Callable[..., int],
        context_terms: Sequence[str],
    ) -> List[SpecialistReviewResult]:
        if not self.enabled or self.max_results <= 0:
            return []
        tasks = [task for task in preflight_result.specialist_review_tasks if task.task_type == self.TASK_TYPE and task.page_no]
        if not tasks:
            return []
        self._task_by_page = {int(task.page_no or 0): task for task in tasks}
        self._review_by_page = {int(review.page_no or 0): review for review in preflight_result.image_page_reviews}
        self._docx_by_unit = {unit.unit_id: unit for unit in preflight_result.docx_units}
        self._used_signatures: set[Tuple[int, str, str]] = set()
        self._used_units: set[str] = set()
        results: List[SpecialistReviewResult] = []

        self._append_vl_text_results(preflight_result=preflight_result, results=results)
        self._append_page_ocr_text_evidence_results(preflight_result=preflight_result, results=results)
        self._append_page_text_qwen_results(preflight_result=preflight_result, results=results)
        self._append_page_text_gap_results(preflight_result=preflight_result, results=results)
        candidates = []
        for review in preflight_result.content_coverage_reviews:
            if review.side != "docx" or review.status not in self.TARGET_STATUSES:
                continue
            page_no = int(review.page_no or 0)
            task = self._task_by_page.get(page_no)
            if task is None:
                continue
            unit = self._docx_by_unit.get(review.unit_id)
            if unit is None or unit.container_type == "table_cell":
                continue
            if review.unit_id in self._used_units:
                continue
            hint = self._call_text_hint(text_hint_fn=text_hint_fn, text=review.text, context_terms=context_terms)
            suggestion = self._clean_suggestion(hint)
            if not self._valid_exact_replacement(old_text=review.text, new_text=suggestion):
                continue
            priority = self._call_priority(priority_fn=priority_fn, text=review.text, hint=hint)
            candidates.append((page_no, priority, review, unit, task, suggestion, hint))
        candidates.sort(key=lambda item: (item[0], item[1], item[3].order_index, item[2].review_id))
        deduped_candidates = []
        seen_replacements: set[Tuple[int, str]] = set()
        for candidate in candidates:
            key = (candidate[0], self._compact(candidate[5]))
            if key in seen_replacements:
                continue
            seen_replacements.add(key)
            deduped_candidates.append(candidate)
        candidates = deduped_candidates

        per_page_count: Dict[int, int] = {}
        completed_pages: set[int] = set()
        for result in results:
            page_no = int(result.page_no or 0)
            if page_no > 0 and result.decision == "confirmed_error":
                completed_pages.add(page_no)
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
        for page_no, priority, review, unit, task, suggestion, hint in candidates:
            if len(results) >= self.max_results:
                break
            if per_page_count.get(page_no, 0) >= 5:
                continue
            if self._seen(page_no=page_no, old_text=review.text, new_text=suggestion):
                continue
            image_review = self._review_by_page.get(page_no)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            self._used_units.add(review.unit_id)
            results.append(
                SpecialistReviewResult(
                    result_id="",
                    task_id=task.task_id,
                    task_type=self.TASK_TYPE,
                    page_no=page_no,
                    status="rule_candidate_needs_model_confirmation",
                    decision="suspected_error",
                    confidence=max(float(review.confidence or 0.0), min(0.59, self._confidence_for_priority(priority))),
                    reason=self._reason_for_hint(hint=hint, image_review=image_review),
                    issue_type=self._issue_type(old_text=review.text, new_text=suggestion),
                    wps_unit_id=review.unit_id,
                    old_text=self._clean_text(review.text),
                    new_text=suggestion,
                    next_route=task.next_route or "needs_qwen_vl",
                    model="image_page_candidate_rules",
                    comment_policy="report_only_until_confirmed",
                    evidence_refs=self._evidence_refs(task=task, review_id=review.review_id, image_review=image_review),
                    flags=list(review.flags)
                    + [
                        "image_page_specialist",
                        "rule_candidate_only",
                        "needs_pdf_or_model_confirmation",
                        f"coverage_status={review.status}",
                        f"artifact_priority={priority}",
                    ],
                )
            )

        for task in tasks:
            if len(results) >= self.max_results:
                break
            page_no = int(task.page_no or 0)
            if page_no in completed_pages:
                continue
            image_review = self._review_by_page.get(page_no)
            results.append(
                SpecialistReviewResult(
                    result_id="",
                    task_id=task.task_id,
                    task_type=self.TASK_TYPE,
                    page_no=page_no,
                    status="deferred_no_exact_replacement",
                    decision="defer",
                    confidence=0.0,
                    reason=task.reason or "图片页专项未找到可定位的精确替换值，保持报告记录。",
                    next_route=task.next_route or "needs_full_page_review",
                    model=task.model_hint,
                    comment_policy="report_only_until_confirmed",
                    evidence_refs=self._evidence_refs(task=task, review_id="", image_review=image_review),
                    flags=list(task.flags) + ["image_page_specialist", "no_exact_replacement"],
                )
            )

        for index, result in enumerate(results[: self.max_results], start=1):
            result.result_id = f"image_specialist_{index:04d}"
        return results[: self.max_results]

    def _append_vl_text_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        per_page_limit = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_RESULTS_PER_PAGE", 9999) or 9999))
        per_page_count: Dict[int, int] = {}
        for review in preflight_result.image_text_vl_reviews:
            if len(results) >= self.max_results:
                return
            page_no = int(review.page_no or 0)
            task = self._task_by_page.get(page_no)
            if task is None or not review.available or review.decision != "allow_exact_replacements":
                continue
            for value in review.suspicious_values:
                if len(results) >= self.max_results:
                    return
                if per_page_count.get(page_no, 0) >= per_page_limit:
                    break
                unit_id = str(value.get("unit_id") or "")
                unit = self._docx_by_unit.get(unit_id)
                if unit is None or unit.container_type == "table_cell" or unit_id in self._used_units:
                    continue
                old_text = self._clean_text(value.get("docx_text") or unit.text)
                new_text = self._clean_suggestion(value.get("visible_text") or "")
                if not self._valid_exact_replacement(old_text=old_text, new_text=new_text):
                    continue
                if self._seen(page_no=page_no, old_text=old_text, new_text=new_text):
                    continue
                self._used_units.add(unit_id)
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
                results.append(
                    SpecialistReviewResult(
                        result_id="",
                        task_id=task.task_id,
                        task_type=self.TASK_TYPE,
                        page_no=page_no,
                        status="executed",
                        decision="confirmed_error",
                        confidence=max(float(review.confidence or 0.0), 0.72),
                        reason=self._reason_for_vl_text(review_reason=review.reason, value_reason=str(value.get("reason") or "")),
                        issue_type=str(value.get("issue_type") or "text_substitution"),
                        wps_unit_id=unit_id,
                        old_text=old_text,
                        new_text=new_text,
                        next_route="",
                        model=review.model,
                        comment_policy=self.COMMENT_POLICY,
                        evidence_refs=[
                            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
                            {"source": "image_text_vl_review", "id": review.review_id, "raw": "image_text_vl_reviews.json"},
                        ],
                        flags=list(review.flags) + ["image_page_specialist", "image_text_vl_confirmed", "exact_replacement"],
                    )
                )

    def _append_page_ocr_text_evidence_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        per_page_limit = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_RESULTS_PER_PAGE", 9999) or 9999))
        per_page_count: Dict[int, int] = {}
        reviews = [
            review
            for review in preflight_result.page_ocr_text_evidence_reviews
            if review.decision == "confirmed_error"
        ]
        reviews.sort(key=lambda item: (int(item.page_no or 9999), -float(item.confidence or 0.0), item.review_id))
        for review in reviews:
            if len(results) >= self.max_results:
                return
            page_no = int(review.page_no or 0)
            if per_page_count.get(page_no, 0) >= per_page_limit:
                continue
            task = self._task_by_page.get(page_no)
            unit = self._docx_by_unit.get(review.docx_unit_id)
            if task is None or unit is None or unit.container_type == "table_cell":
                continue
            if unit.unit_id in self._used_units:
                continue
            old_text = self._clean_text(review.docx_text or unit.text)
            new_text = self._clean_suggestion(review.suggested_text)
            if not self._valid_exact_replacement(old_text=old_text, new_text=new_text):
                continue
            if self._format_only_page_ocr_replacement(old_text=old_text, new_text=new_text, issue_type=review.issue_type):
                continue
            if self._seen(page_no=page_no, old_text=old_text, new_text=new_text):
                continue
            self._used_units.add(unit.unit_id)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            image_review = self._review_by_page.get(page_no)
            results.append(
                SpecialistReviewResult(
                    result_id="",
                    task_id=task.task_id,
                    task_type=self.TASK_TYPE,
                    page_no=page_no,
                    status="executed",
                    decision="confirmed_error",
                    confidence=max(float(review.confidence or 0.0), 0.7),
                    reason=self._reason_for_page_ocr_text(review=review, image_review=image_review),
                    issue_type=review.issue_type or self._issue_type(old_text=old_text, new_text=new_text),
                    wps_unit_id=unit.unit_id,
                    old_text=old_text,
                    new_text=new_text,
                    next_route="",
                    model="page_ocr_text_evidence_rules",
                    comment_policy=self.COMMENT_POLICY,
                    evidence_refs=self._page_ocr_text_evidence_refs(task=task, review=review, image_review=image_review),
                    flags=list(review.flags)
                    + [
                        "image_page_specialist",
                        "page_ocr_text_evidence_confirmed",
                        "exact_replacement",
                    ],
                )
            )

    def _append_page_text_qwen_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        per_page_limit = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_RESULTS_PER_PAGE", 9999) or 9999))
        per_page_count: Dict[int, int] = {}
        for review in preflight_result.page_text_qwen_reviews:
            if len(results) >= self.max_results:
                return
            page_no = int(review.page_no or 0)
            task = self._task_by_page.get(page_no)
            if task is None or not review.available or review.decision != "allow_exact_replacements":
                continue
            for value in review.suspicious_values:
                if len(results) >= self.max_results:
                    return
                if per_page_count.get(page_no, 0) >= per_page_limit:
                    break
                unit_id = str(value.get("unit_id") or "")
                unit = self._docx_by_unit.get(unit_id)
                if unit is None or unit.container_type == "table_cell" or unit_id in self._used_units:
                    continue
                old_text = self._clean_text(value.get("docx_text") or unit.text)
                new_text = self._clean_suggestion(value.get("pdf_text") or "")
                if not self._valid_exact_replacement(old_text=old_text, new_text=new_text):
                    continue
                if self._format_only_page_ocr_replacement(
                    old_text=old_text,
                    new_text=new_text,
                    issue_type=str(value.get("issue_type") or ""),
                ):
                    continue
                if self._seen(page_no=page_no, old_text=old_text, new_text=new_text):
                    continue
                confidence = max(float(review.confidence or 0.0), float(value.get("ocr_support_score") or 0.0), 0.72)
                self._used_units.add(unit_id)
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
                results.append(
                    SpecialistReviewResult(
                        result_id="",
                        task_id=task.task_id,
                        task_type=self.TASK_TYPE,
                        page_no=page_no,
                        status="executed",
                        decision="confirmed_error",
                        confidence=min(0.9, confidence),
                        reason=self._reason_for_page_text_qwen(review=review, value=value),
                        issue_type=str(value.get("issue_type") or "page_text_qwen_replacement"),
                        wps_unit_id=unit_id,
                        old_text=old_text,
                        new_text=new_text,
                        next_route="",
                        model=review.model or "qwen_page_text_compare",
                        comment_policy=self.COMMENT_POLICY,
                        evidence_refs=[
                            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
                            {"source": "page_text_qwen_review", "id": review.review_id, "raw": "page_text_qwen_reviews.json"},
                        ],
                        context={
                            "page_text_qwen_review": review.to_dict(),
                            "suspicious_value": dict(value),
                        },
                        flags=list(review.flags)
                        + [
                            "image_page_specialist",
                            "page_text_qwen_confirmed",
                            "exact_replacement",
                        ],
                    )
                )

    def _append_page_text_gap_results(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        results: List[SpecialistReviewResult],
    ) -> None:
        max_results = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_GAP_MAX_RESULTS", 9999) or 9999))
        if max_results <= 0:
            return
        per_page_limit = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_GAP_MAX_PER_PAGE", 9999) or 9999))
        per_page_count: Dict[int, int] = {}
        appended = 0
        profiles = [
            profile
            for profile in preflight_result.page_text_coverage_profiles
            if profile.risk_level in {"high", "medium"}
            and profile.status in {"page_text_coverage_gap", "page_text_coverage_uncertain", "table_page_text_coverage_uncertain"}
        ]
        profiles.sort(key=lambda item: (0 if item.risk_level == "high" else 1, item.page_no))
        for profile in profiles:
            if len(results) >= self.max_results or appended >= max_results:
                return
            page_no = int(profile.page_no or 0)
            task = self._task_by_page.get(page_no)
            if task is None:
                continue
            for candidate in self._page_text_gap_candidates(profile=profile):
                if len(results) >= self.max_results or appended >= max_results:
                    return
                if per_page_count.get(page_no, 0) >= per_page_limit:
                    break
                unit = self._locate_gap_docx_unit(
                    page_no=page_no,
                    docx_text=str(candidate.get("docx_text") or ""),
                )
                if unit is None or unit.unit_id in self._used_units:
                    continue
                old_text = self._clean_text(candidate.get("docx_text") or unit.text)
                new_text = self._clean_suggestion(candidate.get("pdf_text") or "")
                if not self._valid_exact_replacement(old_text=old_text, new_text=new_text):
                    continue
                if self._seen(page_no=page_no, old_text=old_text, new_text=new_text):
                    continue
                self._used_units.add(unit.unit_id)
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
                appended += 1
                image_review = self._review_by_page.get(page_no)
                results.append(
                    SpecialistReviewResult(
                        result_id="",
                        task_id=task.task_id,
                        task_type=self.TASK_TYPE,
                        page_no=page_no,
                        status="page_text_gap_candidate_needs_confirmation",
                        decision="suspected_error",
                        confidence=max(0.5, min(0.64, float(candidate.get("confidence") or 0.0))),
                        reason=self._page_text_gap_reason(candidate=candidate, profile=profile, image_review=image_review),
                        issue_type=str(candidate.get("issue_type") or "page_text_gap_near_match"),
                        wps_unit_id=unit.unit_id,
                        old_text=old_text,
                        new_text=new_text,
                        next_route=task.next_route or "needs_full_page_review",
                        model="page_text_coverage_rules",
                        comment_policy="report_only_page_gap_candidate",
                        evidence_refs=self._page_text_gap_refs(task=task, profile=profile, image_review=image_review),
                        context={
                            "page_text_coverage": profile.to_dict(),
                            "gap_candidate": dict(candidate),
                        },
                        flags=list(task.flags)
                        + list(profile.flags)
                        + [
                            "image_page_specialist",
                            "page_text_gap_candidate",
                            "rule_candidate_only",
                            "needs_pdf_or_model_confirmation",
                        ],
                    )
                )

    def _page_text_gap_candidates(self, *, profile: PageTextCoverageProfile) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for docx_text in profile.docx_gap_samples:
            for pdf_text in profile.pdf_gap_samples:
                candidate = self._page_text_gap_pair(docx_text=docx_text, pdf_text=pdf_text)
                if candidate:
                    rows.append(candidate)
        rows = self._drop_unstable_page_text_gap_candidates(rows)
        rows.sort(
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                -len(self._compact(item.get("docx_text") or "")),
                str(item.get("docx_text") or ""),
            )
        )
        deduped: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (self._compact(row.get("docx_text") or ""), self._compact(row.get("pdf_text") or ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_TEXT_GAP_MAX_PER_PAGE", 9999) or 9999)):
                break
        return deduped

    def _page_text_gap_pair(self, *, docx_text: Any, pdf_text: Any) -> Dict[str, Any]:
        old_text = self._clean_text(docx_text)
        new_text = self._clean_text(pdf_text)
        old_key = self._compact(old_text)
        new_key = self._compact(new_text)
        if not old_key or not new_key or old_key == new_key:
            return {}
        if not (4 <= len(old_key) <= 48 and 4 <= len(new_key) <= 48):
            return {}
        if self._page_text_gap_noise(old_text) or self._page_text_gap_noise(new_text):
            return {}
        if self._page_text_gap_weak_numeric_fragment(old_key=old_key, new_key=new_key):
            return {}
        if self._page_text_gap_date_fragment(old_text=old_text, new_text=new_text):
            return {}
        if min(len(old_key), len(new_key)) / max(len(old_key), len(new_key)) < 0.72:
            return {}
        if not self._page_text_gap_has_review_value(old_key=old_key, new_key=new_key):
            return {}
        ratio = SequenceMatcher(None, old_key, new_key).ratio()
        common = self._longest_common_substring_length(old_key, new_key)
        digit_old = re.sub(r"\D+", "", old_key)
        digit_new = re.sub(r"\D+", "", new_key)
        same_digits = bool(len(digit_old) >= 3 and digit_old == digit_new)
        if same_digits:
            threshold = 0.52
        elif any(ch.isdigit() for ch in old_key + new_key):
            threshold = 0.68
        else:
            threshold = 0.72
        if ratio < threshold and common < min(6, max(4, min(len(old_key), len(new_key)) - 1)):
            return {}
        return {
            "docx_text": old_text,
            "pdf_text": new_text,
            "confidence": max(0.5, min(0.64, 0.38 + ratio * 0.24 + (0.06 if same_digits else 0.0))),
            "similarity": round(ratio, 4),
            "common_span": common,
            "issue_type": "page_text_gap_near_match",
            "reason": "页级覆盖画像发现同页 DOCX 缺口与 PDF OCR 缺口高度相似，疑似 WPS 识别错字、漏字或形近字替换。",
        }

    def _drop_unstable_page_text_gap_candidates(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_old: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_old.setdefault(self._compact(row.get("docx_text") or ""), []).append(row)
        stable: List[Dict[str, Any]] = []
        for old_key, group in by_old.items():
            new_keys = {self._compact(item.get("pdf_text") or "") for item in group}
            if len(new_keys) <= 1:
                stable.extend(group)
                continue
            old_digits = re.sub(r"\D+", "", old_key)
            new_digit_keys = {re.sub(r"\D+", "", key) for key in new_keys}
            if old_digits and len(old_digits) >= 3 and len(new_digit_keys) == 1 and old_digits in new_digit_keys:
                continue
            if len(new_keys) >= 3:
                continue
            stable.extend(group)
        return stable

    def _locate_gap_docx_unit(self, *, page_no: int, docx_text: str) -> Optional[DocxEvidenceUnit]:
        needle = self._compact(docx_text)
        if not needle:
            return None
        candidates = [
            unit
            for unit in self._docx_by_unit.values()
            if int(unit.estimated_page_no or 0) == page_no
            and unit.container_type != "table_cell"
            and unit.unit_id not in self._used_units
            and self._compact(unit.text)
        ]
        exact: List[DocxEvidenceUnit] = []
        fuzzy: List[tuple[float, DocxEvidenceUnit]] = []
        for unit in candidates:
            key = self._compact(unit.text)
            if needle in key or key in needle:
                exact.append(unit)
                continue
            if len(needle) >= 6:
                score = SequenceMatcher(None, needle, key[: max(len(needle) + 8, 16)]).ratio()
                if score >= 0.72:
                    fuzzy.append((score, unit))
        if exact:
            exact.sort(key=lambda item: (len(self._compact(item.text)), item.order_index))
            return exact[0]
        if fuzzy:
            fuzzy.sort(key=lambda item: (-item[0], item[1].order_index))
            return fuzzy[0][1]
        return None

    def _page_text_gap_noise(self, text: str) -> bool:
        value = self._clean_text(text)
        if any(marker in value for marker in ("?", "？", "_", "*", "%", "[", "]", "［", "］", "无法", "看不清")):
            return True
        compact = self._compact(value)
        if re.search(r"[\u4e00-\u9fff][A-Za-z]|[A-Za-z][\u4e00-\u9fff]", compact):
            return True
        if len(compact) >= 18 and not re.search(r"[\u4e00-\u9fff]", compact):
            return True
        if re.fullmatch(r"\d{8,}", compact):
            return True
        return False

    def _page_text_gap_weak_numeric_fragment(self, *, old_key: str, new_key: str) -> bool:
        old_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", old_key))
        new_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", new_key))
        old_digits = re.sub(r"\D+", "", old_key)
        new_digits = re.sub(r"\D+", "", new_key)
        if not old_has_cjk and not new_has_cjk:
            return True
        if old_digits and new_digits and old_digits != new_digits:
            old_digit_ratio = len(old_digits) / max(1, len(old_key))
            new_digit_ratio = len(new_digits) / max(1, len(new_key))
            if old_digit_ratio >= 0.45 or new_digit_ratio >= 0.45:
                return True
        if old_has_cjk != new_has_cjk and (old_digits or new_digits):
            return True
        return False

    def _page_text_gap_date_fragment(self, *, old_text: str, new_text: str) -> bool:
        full_date = r"\d{4}年\d{1,2}月\d{1,2}日"
        old_has_full_date = bool(re.search(full_date, old_text))
        new_has_full_date = bool(re.search(full_date, new_text))
        if old_has_full_date != new_has_full_date:
            return True
        if re.search(r"\d{4}\s*年|\d{4}\s*月", old_text + new_text) and not (old_has_full_date and new_has_full_date):
            old_digits = re.sub(r"\D+", "", old_text)
            new_digits = re.sub(r"\D+", "", new_text)
            if old_digits != new_digits:
                return True
        return False

    def _page_text_gap_has_review_value(self, *, old_key: str, new_key: str) -> bool:
        if old_key == new_key:
            return False
        if len(set(old_key) | set(new_key)) <= 2:
            return False
        if not (re.search(r"[\u4e00-\u9fff]", old_key + new_key) or re.search(r"\d", old_key + new_key)):
            return False
        return True

    def _longest_common_substring_length(self, left: str, right: str) -> int:
        if not left or not right:
            return 0
        previous = [0] * (len(right) + 1)
        best = 0
        for left_char in left:
            current = [0]
            for index, right_char in enumerate(right, start=1):
                value = previous[index - 1] + 1 if left_char == right_char else 0
                current.append(value)
                if value > best:
                    best = value
            previous = current
        return best

    def _page_text_gap_reason(
        self,
        *,
        candidate: Dict[str, Any],
        profile: PageTextCoverageProfile,
        image_review: Optional[ImagePdfPageReview],
    ) -> str:
        page_context = (
            f"页级 DOCX token 覆盖 {float(profile.docx_token_coverage_ratio or 0.0):.0%}，"
            f"PDF token 覆盖 {float(profile.pdf_token_coverage_ratio or 0.0):.0%}。"
        )
        risk = f"页级风险：{image_review.risk_level}。" if image_review and image_review.risk_level else ""
        return f"{candidate.get('reason') or ''}{page_context}{risk}"[:220]

    def _page_text_gap_refs(
        self,
        *,
        task: SpecialistReviewTask,
        profile: PageTextCoverageProfile,
        image_review: Optional[ImagePdfPageReview],
    ) -> List[Dict[str, Any]]:
        refs = [
            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
            {"source": "page_text_coverage_profile", "id": f"page_{int(profile.page_no or 0):04d}", "raw": "page_text_coverage_profiles.json"},
        ]
        if image_review:
            refs.append({"source": "image_page_review", "id": image_review.review_id, "raw": "image_pdf_page_reviews.json"})
        return refs

    def _reason_for_page_text_qwen(self, *, review: Any, value: Dict[str, Any]) -> str:
        value_reason = " ".join(str(value.get("reason") or "").split())
        review_reason = " ".join(str(getattr(review, "reason", "") or "").split())
        support = value.get("ocr_support_score")
        if value_reason:
            base = value_reason
        elif review_reason:
            base = review_reason
        else:
            base = "Qwen 页级文本复核在 PDF OCR 文本中定位到与 DOCX 不一致的精确替换值。"
        if support not in (None, ""):
            return f"{base} OCR支持分 {float(support):.2f}。"[:220]
        return base[:220]

    def _reason_for_page_ocr_text(
        self,
        *,
        review: PageOcrTextEvidenceReview,
        image_review: Optional[ImagePdfPageReview],
    ) -> str:
        reason = review.reason or "同页 PDF OCR 支持建议替换值。"
        if image_review and image_review.risk_level:
            return f"页级 OCR 证据确认该 DOCX 文本与 PDF 可见内容不一致。{reason} 页级风险：{image_review.risk_level}。"[:220]
        return f"页级 OCR 证据确认该 DOCX 文本与 PDF 可见内容不一致。{reason}"[:220]

    def _page_ocr_text_evidence_refs(
        self,
        *,
        task: SpecialistReviewTask,
        review: PageOcrTextEvidenceReview,
        image_review: Optional[ImagePdfPageReview],
    ) -> List[Dict[str, Any]]:
        refs = [
            {"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"},
            {"source": "page_ocr_text_evidence_review", "id": review.review_id, "raw": "page_ocr_text_evidence_reviews.json"},
        ]
        if review.coverage_review_id:
            refs.append({"source": "content_coverage_review", "id": review.coverage_review_id, "raw": "full_content_coverage_reviews.json"})
        if image_review:
            refs.append({"source": "image_page_review", "id": image_review.review_id, "raw": "image_pdf_page_reviews.json"})
        return refs

    def _reason_for_vl_text(self, *, review_reason: str, value_reason: str) -> str:
        reason = value_reason or review_reason
        if reason:
            return f"Qwen3-VL 图片页全文复核确认该 DOCX 文本与 PDF 可见内容不一致。{reason}"[:220]
        return "Qwen3-VL 图片页全文复核确认该 DOCX 文本与 PDF 可见内容不一致。"

    def _call_text_hint(self, *, text_hint_fn: Callable[..., str], text: str, context_terms: Sequence[str]) -> str:
        try:
            return str(text_hint_fn(text, context_terms=context_terms) or "")
        except TypeError:
            return str(text_hint_fn(text) or "")

    def _call_priority(self, *, priority_fn: Callable[..., int], text: str, hint: str) -> int:
        try:
            value = priority_fn(text=text, hint=hint)
        except TypeError:
            value = priority_fn(text, hint)
        if value is None or value == "":
            return 5
        return int(value)

    def _clean_suggestion(self, value: Any) -> str:
        text = self._clean_text(value)
        for prefix in ("疑似应核对为：", "建议核对为：", "建议改为：", "疑似存在前缀污染字符，建议核对："):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        return self._clean_cjk_digit_spacing(text)

    def _clean_cjk_digit_spacing(self, text: str) -> str:
        value = str(text or "")
        value = re.sub(r"\s+([，。；：、）)])", r"\1", value)
        value = re.sub(r"(?<=[，。；：、])\s+(?=[\u4e00-\u9fff\d])", "", value)
        value = re.sub(r"([（(])\s+", r"\1", value)
        value = re.sub(r"(?<=[\u4e00-\u9fff\d])\s+(?=[\u4e00-\u9fff\d])", "", value)
        return self._clean_text(value)

    def _valid_exact_replacement(self, *, old_text: str, new_text: str) -> bool:
        old_value = self._clean_text(old_text)
        new_value = self._clean_text(new_text)
        if not old_value or not new_value:
            return False
        if old_value == new_value:
            return False
        if self._compact(old_value) == self._compact(new_value) and " " not in old_value:
            return False
        if len(new_value) > 180:
            return False
        if any(marker in new_value for marker in ("?", "？", "_", "看不清", "无法", "不确定", "参考", "仍需", "核对", "疑似")):
            return False
        if self._looks_noisy(new_value):
            return False
        return True

    def _format_only_page_ocr_replacement(self, *, old_text: str, new_text: str, issue_type: str) -> bool:
        """Keep OCR-backed exact comments for content changes, not pure cleanup.

        Page OCR often supports removing WPS-inserted spaces from long scan
        text. That is useful evidence, but it should not become a "must change"
        content-error comment unless the normalized text actually changes.
        """

        old_key = self._format_key(old_text)
        new_key = self._format_key(new_text)
        if not old_key or old_key != new_key:
            return False
        issue = str(issue_type or "")
        if issue in {"paragraph_text_ocr_artifact", "spaced_text_artifact", "date_or_digit_spacing_artifact"}:
            return True
        return bool(re.search(r"\s", str(old_text or "")))

    def _format_key(self, text: str) -> str:
        value = str(text or "")
        value = value.replace("（", "(").replace("）", ")").replace("，", ",").replace("。", ".").replace("：", ":")
        return re.sub(r"[\s,.;:，。；：、/／\\-]+", "", value)

    def _looks_noisy(self, text: str) -> bool:
        value = "".join(str(text or "").split())
        if not value:
            return True
        if len(value) >= 40:
            digit_ratio = sum(ch.isdigit() for ch in value) / max(1, len(value))
            ascii_ratio = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in value) / max(1, len(value))
            if digit_ratio > 0.35 or ascii_ratio > 0.20:
                return True
        if re.search(r"\d{4}年\d{1,2}月\d{3,}日|(?<!\d)\d{3}年", value):
            return True
        return False

    def _seen(self, *, page_no: int, old_text: str, new_text: str) -> bool:
        key = (int(page_no or 0), self._compact(old_text), self._compact(new_text))
        if key in self._used_signatures:
            return True
        self._used_signatures.add(key)
        return False

    def _issue_type(self, *, old_text: str, new_text: str) -> str:
        old_value = self._clean_text(old_text)
        new_value = self._clean_text(new_text)
        if re.search(r"\d+\s+年|\d+\s+月|\d+\s+日|[0-9]\s+[0-9]", old_value) and re.search(r"\d+年|\d+月|\d+日", new_value):
            return "date_or_digit_spacing_artifact"
        if " " in old_value and " " not in new_value and len(new_value) <= 30:
            return "spaced_text_artifact"
        if len(old_value) <= 30:
            return "short_text_ocr_artifact"
        return "paragraph_text_ocr_artifact"

    def _confidence_for_priority(self, priority: int) -> float:
        if priority <= 1:
            return 0.68
        if priority <= 2:
            return 0.64
        return 0.6

    def _reason_for_hint(self, *, hint: str, image_review: Optional[ImagePdfPageReview]) -> str:
        base = "图片型/扫描型页面规则发现 DOCX 文本疑似 OCR 字符污染、日期空格或形近字错误；尚未取得 PDF OCR/VL 确认证据。"
        if image_review and image_review.risk_level:
            return f"{base} 页级风险：{image_review.risk_level}；{image_review.reason}"
        return base

    def _evidence_refs(
        self,
        *,
        task: SpecialistReviewTask,
        review_id: str,
        image_review: Optional[ImagePdfPageReview],
    ) -> List[Dict[str, Any]]:
        refs = [{"source": "specialist_review_task", "id": task.task_id, "raw": "specialist_review_tasks.json"}]
        if review_id:
            refs.append({"source": "content_coverage_review", "id": review_id, "raw": "full_content_coverage_reviews.json"})
        if image_review:
            refs.append({"source": "image_page_review", "id": image_review.review_id, "raw": "image_pdf_page_reviews.json"})
        return refs

    def _clean_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _compact(self, value: Any) -> str:
        return "".join(str(value or "").split()).lower()
