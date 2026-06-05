from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .models import ConversionDiffCandidate, ConversionPreflightResult, SpecialistReviewTask


class SpecialistReviewPlanBuilder:
    """Build page-level follow-up tasks for unresolved conversion-fidelity risks."""

    VISUAL_ROUTES = {
        "needs_qwen_vl",
        "needs_region_ocr",
        "needs_human_visual_review",
        "needs_region_segmentation",
        "needs_qwen_vl_page_gate",
    }
    MAPPING_ROUTES = {"needs_human_mapping_review"}
    TABLE_ROUTES = {"needs_table_parser"}
    RECALL_ROUTES = {"needs_recall_guard_review"}
    FULL_PAGE_ROUTES = {"needs_full_page_ocr"}
    TEXT_ALIGNMENT_ROUTES = {"needs_text_alignment"}

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[SpecialistReviewTask]:
        self._diff_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        self._tasks: Dict[Tuple[str, int], SpecialistReviewTask] = {}

        self._from_review_routes(preflight_result)
        self._from_focused_reviews(preflight_result)
        self._from_table_reviews(preflight_result)
        self._from_table_cell_evidence(preflight_result)
        self._from_table_grid_evidence(preflight_result)
        self._from_page_text_qwen_reviews(preflight_result)
        self._from_content_coverage(preflight_result)
        self._from_page_reviews(preflight_result)
        self._from_model_gates(preflight_result)

        tasks = list(self._tasks.values())
        tasks.sort(key=lambda item: (-(item.priority or 0), int(item.page_no or 0), item.task_type))
        for index, task in enumerate(tasks, start=1):
            page = int(task.page_no or 0)
            task.task_id = f"specialist_p{page:04d}_{task.task_type}_{index:04d}"
            task.candidate_ids = task.candidate_ids[:9999]
            task.coverage_review_ids = task.coverage_review_ids[:9999]
            task.backfill_ids = task.backfill_ids[:9999]
            task.review_ids = task.review_ids[:9999]
            task.evidence_refs = task.evidence_refs[:9999]
            task.flags = task.flags[:9999]
        return tasks

    def _from_review_routes(self, preflight_result: ConversionPreflightResult) -> None:
        for route in preflight_result.review_routes:
            task_type = self._task_type_for_route(route.route)
            if not task_type:
                continue
            self._add(
                task_type=task_type,
                page_no=route.page_no,
                priority=int(route.priority or 0),
                status=self._status_for_route(route.route),
                reason=route.reason,
                next_route=route.route,
                candidate_ids=[route.diff_id],
                evidence_ref={"source": "review_route", "id": route.route_id, "raw": "review_routes.json"},
                source="review_route",
                flags=route.flags,
            )

    def _from_focused_reviews(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.focused_reviews:
            route = review.next_route or ""
            if review.status == "recall_guard" or route in self.RECALL_ROUTES:
                self._add(
                    task_type="recall_guard_specialist_review",
                    page_no=self._page_for_diff(review.diff_id, review.page_no),
                    priority=88,
                    status="pending_recall_guard_review",
                    reason=review.reason or "高风险字段未被主对齐链路可靠覆盖。",
                    next_route=route or "needs_recall_guard_review",
                    model_hint="qwen3-vl:8b_or_qwen3.5:9b",
                    candidate_ids=[review.diff_id],
                    evidence_ref={"source": "focused_review", "id": review.review_id, "raw": "focused_candidate_reviews.json"},
                    source="focused_review",
                    flags=review.flags + ["recall_guard"],
                )
                continue
            task_type = self._task_type_for_route(route)
            if not task_type:
                continue
            self._add(
                task_type=task_type,
                page_no=self._page_for_diff(review.diff_id, review.page_no),
                priority=self._priority_for_route(route),
                status=self._status_for_route(route),
                reason=review.reason,
                next_route=route,
                model_hint=self._model_hint_for_route(route),
                candidate_ids=[review.diff_id],
                evidence_ref={"source": "focused_review", "id": review.review_id, "raw": "focused_candidate_reviews.json"},
                source="focused_review",
                flags=review.flags + [f"focused_status={review.status}", f"focused_decision={review.decision}"],
            )

    def _from_table_reviews(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.table_reviews:
            if review.status == "parseable_grid" and not review.related_diff_ids:
                continue
            if review.status == "parseable_grid":
                continue
            self._add(
                task_type="table_cell_specialist_review",
                page_no=review.page_no,
                priority=86 if review.status in {"table_parse_failed", "missing_page_image"} else 78,
                status="pending_table_parser",
                reason=f"表格结构未稳定解析：{review.status}",
                next_route="needs_table_parser",
                model_hint="table_parser_then_qwen3-vl:8b",
                candidate_ids=review.related_diff_ids,
                evidence_ref={"source": "table_review", "id": review.table_id, "raw": "table_reviews.json"},
                source="table_review",
                flags=review.flags + [f"table_status={review.status}"],
            )
        for review in preflight_result.table_page_vl_reviews:
            needs_task = not review.available or review.verdict in {"conflict", "unreadable", "uncertain"} or bool(review.suspicious_values)
            if not needs_task:
                continue
            status = "model_unavailable" if not review.available else "model_reviewed_with_open_items"
            self._add(
                task_type="table_visual_specialist_review",
                page_no=review.page_no,
                priority=87,
                status=status,
                reason=review.reason or "表格页视觉复核仍有未解决可疑值。",
                next_route=review.next_route or "needs_table_parser",
                model_hint=review.model or "qwen3-vl:8b",
                evidence_ref={"source": "table_page_vl_review", "id": review.review_id, "raw": "table_page_vl_reviews.json"},
                source="table_page_vl_review",
                flags=review.flags + [f"table_vl_verdict={review.verdict}"],
            )

    def _from_table_cell_evidence(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.table_cell_evidence_reviews:
            if review.decision != "confirmed_error":
                continue
            self._add(
                task_type="table_cell_specialist_review",
                page_no=review.page_no,
                priority=91,
                status="ready_exact_table_cell_replacement",
                reason=review.reason or "表格单元证据已经给出可定位的 PDF 可见替换值。",
                next_route="comment_if_exact_replacement",
                model_hint=review.evidence_source or "pdf_page_ocr",
                coverage_review_ids=[review.coverage_review_id] if review.coverage_review_id else [],
                review_ids=[review.review_id],
                evidence_ref={"source": "table_cell_evidence_review", "id": review.review_id, "raw": "table_cell_evidence_reviews.json"},
                source="table_cell_evidence_review",
                flags=review.flags + [f"table_cell_status={review.status}", f"table_cell_issue={review.issue_type}"],
            )

    def _from_table_grid_evidence(self, preflight_result: ConversionPreflightResult) -> None:
        for grid in preflight_result.table_grid_evidence:
            if grid.status in {"grid_sampled_no_issue", "grid_covered_or_low_risk"}:
                continue
            page_no = int(grid.page_no or 0)
            if page_no <= 0:
                continue
            coverage_review_ids = [
                str(item.get("coverage_review_id") or "")
                for item in grid.anomaly_cells[:9999]
                if str(item.get("coverage_review_id") or "")
            ]
            review_ids = [
                str(item.get("evidence_review_id") or "")
                for item in grid.anomaly_cells[:9999]
                if str(item.get("evidence_review_id") or "")
            ]
            if int(grid.confirmed_error_count or 0) > 0:
                priority = 89
                status = "ready_table_grid_confirmed_cells"
                next_route = "comment_if_exact_replacement"
                model_hint = "table_cell_evidence_rules"
            elif int(grid.suspected_error_count or 0) > 0 or grid.anomaly_cells:
                priority = 84
                status = "pending_table_grid_pattern_review"
                next_route = "needs_qwen_vl_table_cell_review"
                model_hint = "qwen3-vl:8b_table_grid_review"
            else:
                priority = 80
                status = "pending_table_grid_structure_review"
                next_route = "needs_table_parser"
                model_hint = "table_parser_then_qwen3-vl:8b"
            self._add(
                task_type="table_cell_specialist_review",
                page_no=page_no,
                priority=priority,
                status=status,
                reason=(
                    f"表格结构证据显示第 {page_no} 页表 {grid.table_index}："
                    f"确认错误 {grid.confirmed_error_count} 个、疑似异常 {grid.suspected_error_count} 个、"
                    f"未可靠覆盖单元 {grid.unresolved_cell_count} 个。"
                ),
                next_route=next_route,
                model_hint=model_hint,
                coverage_review_ids=coverage_review_ids,
                review_ids=review_ids,
                evidence_ref={"source": "table_grid_evidence", "id": grid.grid_id, "raw": "table_grid_evidence.json"},
                source="table_grid_evidence",
                flags=list(grid.flags)
                + list(grid.route_hints)
                + [
                    f"table_grid_status={grid.status}",
                    f"table_grid_id={grid.grid_id}",
                    f"table_index={grid.table_index}",
                ],
            )

    def _from_page_text_qwen_reviews(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.page_text_qwen_reviews:
            if not review.available or review.decision != "allow_exact_replacements":
                continue
            table_values = [
                value
                for value in review.suspicious_values
                if str(value.get("page_kind") or "") == "table_page"
            ]
            if not table_values:
                continue
            self._add(
                task_type="table_cell_specialist_review",
                page_no=review.page_no,
                priority=90,
                status="ready_page_text_qwen_table_replacement",
                reason=review.reason or "页级 OCR 文本 Qwen 复核已给出表格页精确替换候选。",
                next_route="comment_if_exact_replacement",
                model_hint=review.model or "qwen3.5:9b",
                review_ids=[review.review_id],
                evidence_ref={"source": "page_text_qwen_review", "id": review.review_id, "raw": "page_text_qwen_reviews.json"},
                source="page_text_qwen_review",
                flags=list(review.flags) + ["page_text_qwen_table_value", f"page_text_qwen_verdict={review.verdict}"],
            )

    def _from_content_coverage(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered":
                continue
            route = review.next_route or "needs_content_coverage_review"
            task_type = self._task_type_for_route(route) or "full_content_coverage_review"
            self._add(
                task_type=task_type,
                page_no=review.page_no,
                priority=self._priority_for_route(route, default=72),
                status=self._coverage_status_for_route(route),
                reason=review.reason or "DOCX/PDF 内容单元尚未被可靠覆盖。",
                next_route=route,
                model_hint=self._model_hint_for_route(route),
                coverage_review_ids=[review.review_id],
                evidence_ref={"source": "content_coverage_review", "id": review.review_id, "raw": "full_content_coverage_reviews.json"},
                source="content_coverage_review",
                flags=review.flags + [f"coverage_status={review.status}", f"coverage_side={review.side}"],
            )
            self._add(
                task_type="full_content_coverage_review",
                page_no=review.page_no,
                priority=70,
                status="pending_content_alignment",
                reason="本页存在未覆盖内容，需在专项复核中确认是否为 WPS 漏识别/多识别/错位。",
                next_route=route,
                model_hint=self._model_hint_for_route(route),
                coverage_review_ids=[review.review_id],
                evidence_ref={"source": "content_coverage_review", "id": review.review_id, "raw": "full_content_coverage_reviews.json"},
                source="content_coverage_review",
                flags=review.flags + ["full_content_unresolved"],
            )
        for backfill in preflight_result.content_coverage_backfills:
            if backfill.available and backfill.next_route not in {"needs_text_alignment", "needs_human_visual_review"}:
                continue
            route = backfill.next_route or ("needs_text_alignment" if backfill.available else "needs_region_ocr")
            task_type = "full_content_coverage_review" if backfill.available else "visual_region_specialist_review"
            self._add(
                task_type=task_type,
                page_no=backfill.page_no,
                priority=73 if backfill.available else 82,
                status="needs_text_alignment" if backfill.available else "pending_region_ocr",
                reason=backfill.reason or ("局部回填已有文本但还没完成段落级对齐。" if backfill.available else "局部回填 OCR 不可用或质量不足。"),
                next_route=route,
                model_hint=self._model_hint_for_route(route),
                coverage_review_ids=[backfill.coverage_review_id],
                backfill_ids=[backfill.backfill_id],
                evidence_ref={"source": "content_coverage_backfill", "id": backfill.backfill_id, "raw": "full_content_backfill_reviews.json"},
                source="content_coverage_backfill",
                flags=backfill.flags + [f"backfill_status={backfill.status}", f"backfill_method={backfill.method}"],
            )

    def _from_page_reviews(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.fragment_anomaly_reviews:
            self._add(
                task_type="fragment_page_specialist_review",
                page_no=review.page_no,
                priority=84,
                status="pending_fragment_review",
                reason=review.reason or "本页存在短碎片/重复片段风险。",
                next_route=review.next_route or "needs_human_page_review",
                model_hint="qwen3.5:9b_with_page_context",
                review_ids=[review.review_id],
                evidence_ref={"source": "fragment_anomaly_review", "id": review.review_id, "raw": "fragment_anomaly_reviews.json"},
                source="fragment_anomaly_review",
                flags=review.flags,
            )
        for review in preflight_result.image_page_reviews:
            if review.risk_level not in {"high", "medium"}:
                continue
            self._add(
                task_type="image_page_specialist_review",
                page_no=review.page_no,
                priority=90 if review.risk_level == "high" else 78,
                status="pending_full_page_review",
                reason=review.reason or "图片型/扫描型页面需要整页覆盖复核。",
                next_route=review.next_route or "needs_full_page_review",
                model_hint="qwen3-vl:8b_then_qwen3.5:9b",
                review_ids=[review.review_id],
                evidence_ref={"source": "image_page_review", "id": review.review_id, "raw": "image_pdf_page_reviews.json"},
                source="image_page_review",
                flags=review.flags + [f"image_risk={review.risk_level}", f"ocr_quality={review.ocr_quality}"],
            )

    def _from_model_gates(self, preflight_result: ConversionPreflightResult) -> None:
        for review in preflight_result.qwen_vl_reviews:
            if review.decision == "allow_report_candidate" or review.decision == "block_candidate":
                continue
            self._add(
                task_type="visual_region_specialist_review",
                page_no=self._page_for_diff(review.diff_id, None),
                priority=92 if not review.available else 86,
                status="model_unavailable" if not review.available else "model_deferred",
                reason=review.reason or "Qwen-VL 未能确认候选差异。",
                next_route=review.next_route or "needs_human_visual_review",
                model_hint=review.model or "qwen3-vl:8b",
                candidate_ids=[review.diff_id],
                evidence_ref={"source": "qwen_vl_review", "id": review.vl_id, "raw": "qwen_vl_reviews.json"},
                source="qwen_vl_review",
                flags=review.flags + [f"qwen_vl_verdict={review.verdict}", f"qwen_vl_decision={review.decision}"],
            )
        for review in preflight_result.qwen_gate_reviews:
            if review.decision == "allow_report_candidate" or review.decision == "block_candidate":
                continue
            self._add(
                task_type="text_gate_specialist_review",
                page_no=self._page_for_diff(review.diff_id, None),
                priority=84,
                status="model_unavailable" if not review.available else "model_deferred",
                reason=review.reason or "Qwen 文本门槛未能确认候选差异。",
                next_route=review.next_route or "needs_human_mapping_review",
                model_hint=review.model or "qwen3.5:9b",
                candidate_ids=[review.diff_id],
                evidence_ref={"source": "qwen_gate_review", "id": review.gate_id, "raw": "qwen_gate_reviews.json"},
                source="qwen_gate_review",
                flags=review.flags + [f"qwen_verdict={review.verdict}", f"qwen_decision={review.decision}"],
            )

    def _add(
        self,
        *,
        task_type: str,
        page_no: Optional[int],
        priority: int,
        status: str,
        reason: str,
        next_route: str,
        model_hint: str = "",
        candidate_ids: Optional[List[str]] = None,
        coverage_review_ids: Optional[List[str]] = None,
        backfill_ids: Optional[List[str]] = None,
        review_ids: Optional[List[str]] = None,
        evidence_ref: Optional[Dict[str, Any]] = None,
        source: str = "",
        flags: Optional[List[str]] = None,
    ) -> None:
        page = int(page_no or 0)
        key = (task_type, page)
        if key not in self._tasks:
            self._tasks[key] = SpecialistReviewTask(
                task_id="",
                task_type=task_type,
                page_no=page if page > 0 else None,
                priority=int(priority or 0),
                status=status,
                reason=reason,
                next_route=next_route,
                model_hint=model_hint,
            )
        task = self._tasks[key]
        if int(priority or 0) > int(task.priority or 0):
            task.priority = int(priority or 0)
            task.status = status or task.status
            task.reason = reason or task.reason
            task.next_route = next_route or task.next_route
            task.model_hint = model_hint or task.model_hint
        elif not task.reason and reason:
            task.reason = reason
        for value in candidate_ids or []:
            self._append_unique(task.candidate_ids, value)
        for value in coverage_review_ids or []:
            self._append_unique(task.coverage_review_ids, value)
        for value in backfill_ids or []:
            self._append_unique(task.backfill_ids, value)
        for value in review_ids or []:
            self._append_unique(task.review_ids, value)
        if evidence_ref:
            self._append_unique_ref(task.evidence_refs, evidence_ref)
        if source:
            task.source_counts[source] = int(task.source_counts.get(source, 0)) + 1
        for flag in flags or []:
            self._append_unique(task.flags, flag)

    def _task_type_for_route(self, route: str) -> str:
        if route in self.TABLE_ROUTES:
            return "table_cell_specialist_review"
        if route in self.FULL_PAGE_ROUTES:
            return "image_page_specialist_review"
        if route in self.TEXT_ALIGNMENT_ROUTES:
            return "full_content_coverage_review"
        if route in self.VISUAL_ROUTES:
            return "visual_region_specialist_review"
        if route in self.MAPPING_ROUTES:
            return "mapping_specialist_review"
        if route in self.RECALL_ROUTES:
            return "recall_guard_specialist_review"
        return ""

    def _status_for_route(self, route: str) -> str:
        if route == "needs_table_parser":
            return "pending_table_parser"
        if route == "needs_full_page_ocr":
            return "pending_full_page_ocr"
        if route == "needs_text_alignment":
            return "needs_text_alignment"
        if route == "needs_region_segmentation":
            return "pending_region_segmentation"
        if route == "needs_qwen_vl_page_gate":
            return "pending_page_vl_review"
        if route in {"needs_qwen_vl", "needs_region_ocr", "needs_human_visual_review"}:
            return "pending_visual_review"
        if route == "needs_human_mapping_review":
            return "pending_mapping_review"
        if route == "needs_recall_guard_review":
            return "pending_recall_guard_review"
        return "pending_specialist_review"

    def _coverage_status_for_route(self, route: str) -> str:
        if route == "needs_table_parser":
            return "pending_table_parser"
        if route == "needs_full_page_ocr":
            return "pending_full_page_ocr"
        if route == "needs_text_alignment":
            return "needs_text_alignment"
        if route == "needs_region_segmentation":
            return "pending_region_segmentation"
        if route == "needs_qwen_vl_page_gate":
            return "pending_page_vl_review"
        if route in {"needs_qwen_vl", "needs_region_ocr", "needs_human_visual_review"}:
            return "pending_region_ocr"
        if route == "needs_human_mapping_review":
            return "pending_mapping_review"
        if route == "needs_recall_guard_review":
            return "pending_recall_guard_review"
        return "pending_content_alignment"

    def _priority_for_route(self, route: str, *, default: int = 75) -> int:
        if route == "needs_table_parser":
            return 84
        if route == "needs_full_page_ocr":
            return 86
        if route == "needs_region_segmentation":
            return 84
        if route == "needs_qwen_vl_page_gate":
            return 88
        if route == "needs_text_alignment":
            return 74
        if route in {"needs_qwen_vl", "needs_region_ocr", "needs_human_visual_review"}:
            return 82
        if route == "needs_recall_guard_review":
            return 88
        if route == "needs_human_mapping_review":
            return 76
        return default

    def _model_hint_for_route(self, route: str) -> str:
        if route == "needs_full_page_ocr":
            return "full_page_ocr_then_qwen3.5:9b_alignment"
        if route == "needs_region_segmentation":
            return "layout_segmentation_then_region_ocr"
        if route == "needs_qwen_vl_page_gate":
            return "qwen3-vl:8b_full_page_gate"
        if route == "needs_text_alignment":
            return "qwen3.5:9b_text_alignment"
        if route == "needs_qwen_vl":
            return "qwen3-vl:8b"
        if route == "needs_region_ocr":
            return "local_region_ocr_then_qwen3-vl:8b"
        if route == "needs_table_parser":
            return "table_parser_then_qwen3-vl:8b"
        if route == "needs_recall_guard_review":
            return "qwen3-vl:8b_or_qwen3.5:9b"
        if route == "needs_human_mapping_review":
            return "anchor_mapping_then_qwen3.5:9b"
        if route == "needs_human_visual_review":
            return "qwen3-vl:8b_retry_or_human_visual_review"
        return ""

    def _page_for_diff(self, diff_id: str, fallback: Optional[int]) -> Optional[int]:
        diff: Optional[ConversionDiffCandidate] = self._diff_by_id.get(diff_id)
        if diff:
            return diff.pdf_page_no or diff.docx_estimated_page_no or fallback
        return fallback

    def _append_unique(self, target: List[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)

    def _append_unique_ref(self, target: List[Dict[str, Any]], value: Dict[str, Any]) -> None:
        ref = {str(k): v for k, v in dict(value).items() if v not in {None, ""}}
        if not ref:
            return
        key = (str(ref.get("source") or ""), str(ref.get("id") or ""), str(ref.get("raw") or ""))
        for item in target:
            if (str(item.get("source") or ""), str(item.get("id") or ""), str(item.get("raw") or "")) == key:
                return
        target.append(ref)
