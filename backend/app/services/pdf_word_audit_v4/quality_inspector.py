from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .common import CorrectionCandidate
from .models import ConversionPreflightResult


class QualityInspectorBuilder:
    """Summarize why the current v4 run did or did not produce useful comments."""

    def build(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
    ) -> Dict[str, Any]:
        coverage_resolution = self._coverage_resolution_context(
            preflight_result=preflight_result,
            corrections=corrections,
        )
        page_rows = self._page_rows(
            preflight_result=preflight_result,
            corrections=corrections,
            resolved_coverage_ids=coverage_resolution["coverage_review_ids"],
        )
        bottlenecks = self._bottlenecks(
            preflight_result=preflight_result,
            page_rows=page_rows,
            corrections=corrections,
            resolved_coverage_ids=coverage_resolution["coverage_review_ids"],
        )
        unresolved = sum(int(row.get("unresolved_candidate_count") or 0) for row in page_rows)
        coverage_unresolved_raw = sum(1 for item in preflight_result.content_coverage_reviews if item.decision != "covered")
        coverage_resolved = sum(
            1
            for item in preflight_result.content_coverage_reviews
            if item.decision != "covered" and item.review_id in coverage_resolution["coverage_review_ids"]
        )
        coverage_unresolved = max(0, coverage_unresolved_raw - coverage_resolved)
        coverage_backfilled = sum(1 for item in preflight_result.content_coverage_backfills if item.available)
        allow_count = sum(1 for item in preflight_result.qwen_gate_reviews if item.decision == "allow_report_candidate")
        allow_count += sum(1 for item in preflight_result.qwen_vl_reviews if item.decision == "allow_report_candidate")
        summary = {
            "enabled": True,
            "version": "v4_quality_inspection_v1",
            "overall_status": self._overall_status(
                preflight_result=preflight_result,
                unresolved=unresolved,
                bottlenecks=bottlenecks,
                comments=corrections,
            ),
            "comment_count": len(corrections),
            "confirmed_candidate_count": allow_count,
            "unresolved_candidate_count": unresolved,
            "full_content_unresolved_count": coverage_unresolved,
            "full_content_unresolved_raw_count": coverage_unresolved_raw,
            "full_content_resolved_by_specialist_count": coverage_resolved,
            "full_content_backfilled_count": coverage_backfilled,
            "high_risk_page_coverage_review_count": len(preflight_result.high_risk_page_coverage_reviews),
            "high_risk_page_coverage_unresolved_count": sum(
                int(item.unresolved_count or 0) for item in preflight_result.high_risk_page_coverage_reviews
            ),
            "page_text_coverage_review_count": len(preflight_result.page_text_coverage_profiles),
            "page_text_coverage_high_risk_count": sum(1 for item in preflight_result.page_text_coverage_profiles if item.risk_level == "high"),
            "page_text_coverage_medium_risk_count": sum(1 for item in preflight_result.page_text_coverage_profiles if item.risk_level == "medium"),
            "page_text_coverage_low_docx_ratio_count": sum(
                1 for item in preflight_result.page_text_coverage_profiles if float(item.docx_token_coverage_ratio or 0.0) < 0.72
            ),
            "page_text_coverage_low_pdf_ratio_count": sum(
                1 for item in preflight_result.page_text_coverage_profiles if float(item.pdf_token_coverage_ratio or 0.0) < 0.72
            ),
            "canonical_page_type_counts": self._count_page_profile_value(preflight_result.page_profiles, "audit_canonical_page_type"),
            "pdf_source_type_counts": self._count_page_profile_value(preflight_result.page_profiles, "pdf_source_type"),
            "recognition_risk_counts": self._count_page_profile_value(preflight_result.page_profiles, "recognition_risk"),
            "quality_first_page_count": sum(1 for item in preflight_result.page_profiles.values() if item.get("quality_first")),
            "bottleneck_count": len(bottlenecks),
            "highest_risk_pages": [row["page_no"] for row in page_rows if row.get("risk_level") == "high"][:20],
        }
        return {
            "enabled": True,
            "version": "v4_quality_inspection_v1",
            "overall_status": summary["overall_status"],
            "summary": summary,
            "bottlenecks": bottlenecks,
            "page_summaries": page_rows,
            "recommended_next_actions": self._recommended_next_actions(bottlenecks=bottlenecks),
        }

    def _page_rows(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
        resolved_coverage_ids: set[str],
    ) -> List[Dict[str, Any]]:
        by_page: Dict[int, Dict[str, Any]] = {}

        def row_for(page_no: int | None) -> Dict[str, Any]:
            page = int(page_no or 0)
            if page <= 0:
                page = 0
            if page not in by_page:
                profile = preflight_result.page_profiles.get(str(page)) or {}
                by_page[page] = {
                    "page_no": page,
                    "labels": list(profile.get("labels") or []),
                    "canonical_page_type": str(profile.get("audit_canonical_page_type") or ""),
                    "pdf_source_type": str(profile.get("pdf_source_type") or ""),
                    "recognition_strategy": str(profile.get("recognition_strategy") or ""),
                    "recognition_risk": str(profile.get("recognition_risk") or ""),
                    "quality_first": bool(profile.get("quality_first")),
                    "diff_count": 0,
                    "comment_count": 0,
                    "mapping_uncertain_count": 0,
                    "table_candidate_count": 0,
                    "visual_candidate_count": 0,
                    "recall_guard_count": 0,
                    "fragment_anomaly_count": 0,
                    "image_page_review_count": 0,
                    "image_page_high_risk": 0,
                    "qwen_vl_deferred_count": 0,
                    "qwen_vl_allowed_count": 0,
                    "qwen_text_allowed_count": 0,
                    "content_coverage_unresolved_count": 0,
                    "content_coverage_unresolved_raw_count": 0,
                    "content_coverage_resolved_by_specialist_count": 0,
                    "content_coverage_backfilled_count": 0,
                    "page_text_coverage_status": "",
                    "page_text_coverage_risk": "",
                    "page_text_docx_ratio": 0.0,
                    "page_text_pdf_ratio": 0.0,
                    "page_text_similarity": 0.0,
                    "unresolved_candidate_count": 0,
                    "dominant_blocker": "",
                    "risk_level": "low",
                    "candidate_ids": [],
                }
            return by_page[page]

        diff_page_by_id: Dict[str, int] = {}
        for diff in preflight_result.diff_candidates:
            page_no = int(diff.pdf_page_no or diff.docx_estimated_page_no or 0)
            diff_page_by_id[diff.diff_id] = page_no
            row = row_for(page_no)
            row["diff_count"] += 1
            row["candidate_ids"].append(diff.diff_id)
            if diff.category == "mapping_uncertain":
                row["mapping_uncertain_count"] += 1
            if diff.category in {"table_structure_suspect", "table_cell_mismatch_suspect"}:
                row["table_candidate_count"] += 1
            if diff.category in {"text_substitution", "visual_region_unresolved"} or "needs_ocr" in set(diff.flags):
                row["visual_candidate_count"] += 1
            if "recall_guard" in set(diff.flags):
                row["recall_guard_count"] += 1

        for review in preflight_result.focused_reviews:
            row = row_for(review.page_no or diff_page_by_id.get(review.diff_id))
            if review.next_route in {"needs_qwen_vl", "needs_region_ocr"}:
                row["visual_candidate_count"] += 1
            if review.status in {"blocked_mapping_uncertain", "needs_table_parser", "recall_guard"} or review.next_route in {
                "needs_human_mapping_review",
                "needs_table_parser",
                "needs_qwen_vl",
                "needs_region_ocr",
            }:
                row["unresolved_candidate_count"] += 1

        for review in preflight_result.qwen_vl_reviews:
            row = row_for(diff_page_by_id.get(review.diff_id))
            if review.decision == "allow_report_candidate":
                row["qwen_vl_allowed_count"] += 1
            elif review.decision != "block_candidate":
                row["qwen_vl_deferred_count"] += 1

        for review in preflight_result.qwen_gate_reviews:
            row = row_for(diff_page_by_id.get(review.diff_id))
            if review.decision == "allow_report_candidate":
                row["qwen_text_allowed_count"] += 1

        for review in preflight_result.content_coverage_reviews:
            if review.decision != "covered":
                row = row_for(review.page_no)
                row["content_coverage_unresolved_raw_count"] += 1
                if review.review_id in resolved_coverage_ids:
                    row["content_coverage_resolved_by_specialist_count"] += 1
                else:
                    row["content_coverage_unresolved_count"] += 1

        for review in preflight_result.content_coverage_backfills:
            if review.available:
                row_for(review.page_no)["content_coverage_backfilled_count"] += 1

        for profile in preflight_result.page_text_coverage_profiles:
            row = row_for(profile.page_no)
            row["page_text_coverage_status"] = profile.status
            row["page_text_coverage_risk"] = profile.risk_level
            row["page_text_docx_ratio"] = round(float(profile.docx_token_coverage_ratio or 0.0), 4)
            row["page_text_pdf_ratio"] = round(float(profile.pdf_token_coverage_ratio or 0.0), 4)
            row["page_text_similarity"] = round(float(profile.page_text_similarity or 0.0), 4)

        for review in preflight_result.fragment_anomaly_reviews:
            row_for(review.page_no)["fragment_anomaly_count"] += 1

        for review in preflight_result.image_page_reviews:
            row = row_for(review.page_no)
            row["image_page_review_count"] += 1
            if review.risk_level == "high":
                row["image_page_high_risk"] += 1

        for correction in corrections:
            row = row_for(correction.page_no)
            row["comment_count"] += 1

        rows = [row for page, row in by_page.items() if page > 0]
        for row in rows:
            blockers = {
                "mapping": int(row["mapping_uncertain_count"]),
                "table": int(row["table_candidate_count"]),
                "visual": int(row["visual_candidate_count"]) + int(row["qwen_vl_deferred_count"]),
                "recall": int(row["recall_guard_count"]),
                "fragment": int(row["fragment_anomaly_count"]),
                "image_page": int(row["image_page_high_risk"]) or int(row["image_page_review_count"]),
                "coverage": int(row["content_coverage_unresolved_count"]),
                "page_text": 2 if row.get("page_text_coverage_risk") == "high" else (1 if row.get("page_text_coverage_risk") == "medium" else 0),
            }
            row["dominant_blocker"] = max(blockers.items(), key=lambda item: item[1])[0] if any(blockers.values()) else ""
            if (
                row["image_page_high_risk"]
                or row["fragment_anomaly_count"]
                or row["unresolved_candidate_count"] >= 3
                or row["mapping_uncertain_count"] >= 2
                or row["table_candidate_count"] >= 2
                or row.get("page_text_coverage_risk") == "high"
            ):
                row["risk_level"] = "high"
            elif (
                row["unresolved_candidate_count"]
                or row["recall_guard_count"]
                or row["qwen_vl_deferred_count"]
                or row.get("page_text_coverage_risk") == "medium"
            ):
                row["risk_level"] = "medium"
            elif row["content_coverage_unresolved_count"]:
                row["risk_level"] = "medium"
            row["candidate_ids"] = row["candidate_ids"][:30]
        rows.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(item["risk_level"], 3), item["page_no"]))
        return rows

    def _count_page_profile_value(self, page_profiles: Dict[str, Dict[str, Any]], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for profile in page_profiles.values():
            value = str(profile.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    def _coverage_resolution_context(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
    ) -> Dict[str, set[str]]:
        resolved_unit_ids = {item.wps_unit_id for item in corrections if item.wps_unit_id}
        resolved_coverage_ids: set[str] = set()
        for result in preflight_result.specialist_review_results:
            if result.decision != "confirmed_error" or result.comment_policy != "comment_if_exact_replacement":
                continue
            if result.wps_unit_id:
                resolved_unit_ids.add(result.wps_unit_id)
            for ref in result.evidence_refs:
                if ref.get("source") == "content_coverage_review" and ref.get("id"):
                    resolved_coverage_ids.add(str(ref["id"]))
        for review in preflight_result.table_cell_evidence_reviews:
            if review.decision != "confirmed_error":
                continue
            if (
                review.status == "pdf_ocr_context_confirmed_mismatch"
                or "table_context_confirmed_decimal_missing" in set(review.flags or [])
            ):
                continue
            if review.docx_unit_id:
                resolved_unit_ids.add(review.docx_unit_id)
            if review.coverage_review_id:
                resolved_coverage_ids.add(review.coverage_review_id)
        for review in preflight_result.content_coverage_reviews:
            if review.unit_id in resolved_unit_ids:
                resolved_coverage_ids.add(review.review_id)
        return {
            "unit_ids": resolved_unit_ids,
            "coverage_review_ids": resolved_coverage_ids,
        }

    def _bottlenecks(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        page_rows: Sequence[Dict[str, Any]],
        corrections: Sequence[CorrectionCandidate],
        resolved_coverage_ids: set[str],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        mapping_count = sum(1 for item in preflight_result.diff_candidates if item.category == "mapping_uncertain")
        if mapping_count:
            rows.append(
                {
                    "type": "mapping_uncertain",
                    "severity": "high" if mapping_count >= 5 else "medium",
                    "count": mapping_count,
                    "reason": "PDF-DOCX 页/段映射不稳，字段比较被正确阻断，但会造成漏检。",
                }
            )
        table_pending = sum(1 for item in preflight_result.focused_reviews if item.next_route == "needs_table_parser")
        if table_pending:
            rows.append(
                {
                    "type": "table_parser_gap",
                    "severity": "high" if table_pending >= 3 else "medium",
                    "count": table_pending,
                    "reason": "表格候选仍停在结构解析阶段，无法形成可批注的单元格级证据。",
                }
            )
        visual_pending = sum(1 for item in preflight_result.focused_reviews if item.next_route in {"needs_qwen_vl", "needs_region_ocr"})
        vl_deferred = sum(1 for item in preflight_result.qwen_vl_reviews if item.decision == "defer")
        vl_unavailable = sum(1 for item in preflight_result.qwen_vl_reviews if not item.available)
        if visual_pending or vl_deferred or vl_unavailable:
            rows.append(
                {
                    "type": "visual_gate_gap",
                    "severity": "high" if vl_unavailable else "medium",
                    "count": visual_pending + vl_deferred + vl_unavailable,
                    "reason": "视觉候选需要 Qwen-VL/局部 OCR 二次确认；未确认前不应直接报错。",
                }
            )
        recall_count = sum(1 for item in preflight_result.diff_candidates if "recall_guard" in set(item.flags))
        if recall_count:
            rows.append(
                {
                    "type": "recall_guard_gap",
                    "severity": "medium",
                    "count": recall_count,
                    "reason": "高风险字段进入兜底链路，说明主对齐/候选生成仍有漏召回。",
                }
            )
        coverage_unresolved_raw = sum(1 for item in preflight_result.content_coverage_reviews if item.decision != "covered")
        coverage_resolved = sum(
            1
            for item in preflight_result.content_coverage_reviews
            if item.decision != "covered" and item.review_id in resolved_coverage_ids
        )
        coverage_unresolved = max(0, coverage_unresolved_raw - coverage_resolved)
        coverage_backfilled = sum(1 for item in preflight_result.content_coverage_backfills if item.available)
        coverage_attempted = sum(1 for item in preflight_result.content_coverage_backfills if item.attempted)
        if coverage_unresolved:
            rows.append(
                {
                    "type": "full_content_coverage_gap",
                    "severity": "high" if coverage_unresolved >= 20 else "medium",
                    "count": coverage_unresolved,
                    "raw_count": coverage_unresolved_raw,
                    "resolved_by_specialist_count": coverage_resolved,
                    "backfilled_count": coverage_backfilled,
                    "attempted_backfill_count": coverage_attempted,
                    "reason": "存在未被 PDF 侧可靠覆盖的普通内容单元；已回填的局部 OCR 只能补证据，仍不能直接等同全内容放行。",
                }
            )
        if preflight_result.high_risk_page_coverage_reviews:
            rows.append(
                {
                    "type": "high_risk_page_coverage_review",
                    "severity": "high",
                    "count": len(preflight_result.high_risk_page_coverage_reviews),
                    "pages": [item.page_no for item in preflight_result.high_risk_page_coverage_reviews[:20]],
                    "unresolved_count": sum(int(item.unresolved_count or 0) for item in preflight_result.high_risk_page_coverage_reviews),
                    "reason": "高风险页未覆盖内容已被分组；这些页需要整页级核查，避免未定位候选造成漏检。",
                }
            )
        page_text_high = [item for item in preflight_result.page_text_coverage_profiles if item.risk_level == "high"]
        page_text_medium = [item for item in preflight_result.page_text_coverage_profiles if item.risk_level == "medium"]
        if page_text_high or page_text_medium:
            rows.append(
                {
                    "type": "page_text_coverage_gap",
                    "severity": "high" if page_text_high else "medium",
                    "count": len(page_text_high) + len(page_text_medium),
                    "high_risk_count": len(page_text_high),
                    "medium_risk_count": len(page_text_medium),
                    "pages": [item.page_no for item in [*page_text_high, *page_text_medium][:20]],
                    "reason": "页级 DOCX/PDF OCR token 覆盖画像显示仍有整页文本缺口；这类缺口会造成候选未定位和漏检。",
                }
            )
        fragment_count = sum(1 for item in preflight_result.fragment_anomaly_reviews)
        if fragment_count:
            rows.append(
                {
                    "type": "fragment_anomaly_gap",
                    "severity": "high",
                    "count": fragment_count,
                    "reason": "扫描/图片页存在大量 DOCX 短碎片和重复片段，说明 WPS 可能把同一可见区域拆碎、重复或错位识别。",
                }
            )
        image_high = sum(1 for item in preflight_result.image_page_reviews if item.risk_level == "high")
        image_medium = sum(1 for item in preflight_result.image_page_reviews if item.risk_level == "medium")
        if image_high or image_medium:
            rows.append(
                {
                    "type": "image_pdf_page_risk",
                    "severity": "high" if image_high else "medium",
                    "count": image_high + image_medium,
                    "high_risk_count": image_high,
                    "medium_risk_count": image_medium,
                    "reason": "图片型/扫描型 PDF 页不能按普通文字 PDF 放行；这些页存在 OCR 质量、覆盖缺口、表格或碎片化风险。",
                }
            )
        if preflight_result.diff_candidates and len(corrections) <= max(1, len(preflight_result.diff_candidates) // 10):
            rows.append(
                {
                    "type": "low_comment_yield",
                    "severity": "medium",
                    "count": len(corrections),
                    "reason": "候选很多但可批注结果少，说明证据链仍卡在映射、表格或视觉确认。",
                }
            )
        high_pages = [row["page_no"] for row in page_rows if row.get("risk_level") == "high"]
        if high_pages:
            rows.append(
                {
                    "type": "high_risk_pages",
                    "severity": "medium",
                    "count": len(high_pages),
                    "pages": high_pages[:20],
                    "reason": "这些页聚集了未解决候选，应优先优化局部证据。",
                }
            )
        return rows

    def _recommended_next_actions(self, *, bottlenecks: Sequence[Dict[str, Any]]) -> List[str]:
        types = {str(item.get("type") or "") for item in bottlenecks}
        actions: List[str] = []
        if "visual_gate_gap" in types:
            actions.append("优先检查 qwen_vl_gate_reviews.json：确认 qwen3-vl 是否可用，以及哪些截图 unreadable/conflict。")
        if "table_parser_gap" in types:
            actions.append("优先增强表格页的单元格定位和局部 OCR，不要把整页 OCR 结果直接拿来比较。")
        if "mapping_uncertain" in types:
            actions.append("增强页级锚点和段落窗口对齐；mapping_uncertain 仍不得进入字段错误批注。")
        if "recall_guard_gap" in types:
            actions.append("把召回兜底候选送入视觉/表格专项门槛，能确认的再升级为确认错误批注；不能确认的只保留核查参考。")
        if "full_content_coverage_gap" in types:
            actions.append("优先看 full_content_backfill_reviews.json：先判断未覆盖区域是否已被局部 OCR/Qwen-VL 读出，再做段落级对齐和批注升级。")
        if "high_risk_page_coverage_review" in types:
            actions.append("优先看 high_risk_page_coverage_reviews.json：按页核查 unresolved 分布、DOCX/PDF 样例和整页批注锚点。")
        if "page_text_coverage_gap" in types:
            actions.append("优先看 page_text_coverage_profiles.json：按页查看 DOCX/PDF token 覆盖率和缺口样例，定位候选未召回页面。")
        if not actions:
            actions.append("当前预审没有明显阻塞，下一步可以提高批注准入覆盖率。")
        return actions

    def _overall_status(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        unresolved: int,
        bottlenecks: Sequence[Dict[str, Any]],
        comments: Sequence[CorrectionCandidate],
    ) -> str:
        severities = {str(item.get("severity") or "") for item in bottlenecks}
        if "high" in severities:
            if self._structured_review_queue_ready(preflight_result=preflight_result):
                return "usable_with_review_queue"
            return "needs_pipeline_improvement"
        if unresolved and self._structured_review_queue_ready(preflight_result=preflight_result):
            return "usable_with_review_queue"
        if unresolved:
            return "needs_more_review"
        if comments:
            return "usable_with_comments"
        return "no_actionable_candidates"

    def _structured_review_queue_ready(self, *, preflight_result: ConversionPreflightResult) -> bool:
        tasks = [task for task in list(preflight_result.coverage_review_tasks or []) if task.requires_human_review()]
        if not tasks:
            return False
        unresolved_high_risk_pages = {
            int(item.page_no or 0)
            for item in preflight_result.high_risk_page_coverage_reviews
            if int(item.page_no or 0) > 0 and int(item.unresolved_count or 0) > 0
        }
        task_pages = {int(item.page_no or 0) for item in tasks if int(item.page_no or 0) > 0}
        if unresolved_high_risk_pages and not unresolved_high_risk_pages.issubset(task_pages):
            return False
        return True
