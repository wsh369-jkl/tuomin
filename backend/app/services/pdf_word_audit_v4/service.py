from __future__ import annotations

import json
import logging
import re
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.runtime_security import ensure_private_directory, ensure_private_file

from .anchor_ocr import AnchorOcrExtractor
from .common import (
    CorrectionCandidate,
    DocxAuditCommentWriter,
    ORGANIZATION_SUFFIXES,
    has_high_value_field_content,
    looks_like_document_title,
    looks_like_organization_name,
    looks_like_table_header,
    looks_like_table_title,
    normalize_text,
    similarity,
    table_text_artifact_replacement,
)
from .content_coverage_backfill import ContentCoverageBackfillBuilder, ContentCoverageBackfillResolver
from .coverage_scheduler import CoverageGapScheduler
from .coverage_task_execution import CoverageTaskExecutionBridge
from .docx_evidence import DocxEvidenceExtractor
from .docx_page_remap import DocxPageRemapper
from .fragment_anomaly import FragmentAnomalyBuilder
from .focused_review import FocusedReviewBuilder
from .full_content_coverage import FullContentCoverageBuilder
from .full_page_review_closure import FullPageReviewCoverageCloser
from .high_risk_page_coverage import HighRiskPageCoverageBuilder
from .image_page_review import ImagePdfPageReviewBuilder
from .image_page_vl_text import ImagePageVlTextBuilder
from .image_specialist_executor import ImagePageSpecialistExecutor
from .mapping_stabilizer import MappingStabilizer
from .model_candidate_recall import ModelCandidateRecallBuilder
from .model_output_guard import ModelOutputGuard
from .models import ConversionPreflightResult
from .page_ocr_text_evidence import PageOcrTextEvidenceBuilder
from .page_text_qwen_review import PageTextQwenReviewBuilder
from .page_text_coverage import PageTextCoverageProfileBuilder
from .pdf_evidence import PdfEvidenceExtractor
from .page_orientation import PageOrientationNormalizer
from .preflight import ConversionPreflightBuilder
from .quality_inspector import QualityInspectorBuilder
from .qwen_gate import FocusedQwenGateBuilder
from .qwen_vl_gate import OllamaQwenVlClient, QwenVlGateBuilder
from .report_builder import ProductReportBuilder
from .renderer import PageRenderer
from .route_planner import AuditRoutePlanner
from .specialist_review_plan import SpecialistReviewPlanBuilder
from .table_audit_engine import TableAuditEngine
from .table_cell_evidence import TableCellEvidenceBuilder
from .table_grid_evidence import TableGridEvidenceBuilder
from .table_heavy_page_closure import TableHeavyPageCoverageCloser
from .table_specialist_executor import TableSpecialistExecutor
from .table_page_vl import TablePageVlBuilder
from .table_review import TableReviewBuilder

logger = logging.getLogger(__name__)


class PdfWordAuditV4Service:
    """PDF->DOCX conversion fidelity review with conservative local gates.

    v4 first builds traceable PDF/DOCX evidence and routes. Only the small
    candidate set then enters table, visual, and Qwen gates for comment-only
    output; it never edits the DOCX body.
    """

    PROFILE = "pdf_word_audit_v4_conversion_fidelity_preflight"
    GENERIC_TITLE_SUFFIXES = (
        "书",
        "合同",
        "协议",
        "证明",
        "申请书",
        "声明",
        "通知",
        "函",
        "清单",
        "明细",
        "明细表",
        "一览表",
        "汇总表",
        "统计表",
        "台账",
        "目录",
    )
    GENERIC_FIELD_LABEL_PHRASES = (
        "身份号码",
        "身份证号码",
        "身份证",
        "住址",
        "地址",
        "出生日期",
        "联系电话",
        "手机号",
        "账号",
        "账户",
    )
    CONTEXT_PLACE_SUFFIX_SPECS = (
        ("办事处", 2),
        ("大道", 2),
        ("道路", 2),
        ("小区", 2),
        ("花园", 2),
        ("公寓", 2),
        ("广场", 2),
        ("省", 2),
        ("市", 2),
        ("区", 2),
        ("县", 2),
        ("镇", 2),
        ("乡", 2),
        ("村", 2),
        ("路", 2),
        ("街", 2),
        ("苑", 2),
        ("园", 2),
        ("城", 2),
        ("府", 2),
        ("湾", 2),
        ("庭", 2),
    )
    CONTEXT_PLACE_SUFFIXES = (
        "办事处",
        "大道",
        "道路",
        "小区",
        "花园",
        "公寓",
        "广场",
        "省",
        "市",
        "区",
        "县",
        "镇",
        "乡",
        "村",
        "路",
        "街",
        "苑",
        "园",
        "城",
        "府",
        "湾",
        "庭",
    )
    CONTEXT_PLACE_EXTRACTION_SUFFIX_SPECS = (
        ("办事处", 2, 8),
        ("大道", 2, 8),
        ("道路", 2, 8),
        ("小区", 2, 8),
        ("花园", 2, 8),
        ("公寓", 2, 8),
        ("广场", 2, 8),
        ("省", 2, 4),
        ("市", 2, 4),
        ("区", 2, 4),
        ("县", 2, 4),
        ("镇", 2, 5),
        ("乡", 2, 5),
        ("村", 2, 6),
        ("路", 2, 6),
        ("街", 2, 6),
        ("苑", 2, 6),
        ("园", 2, 6),
        ("城", 2, 6),
        ("府", 2, 6),
        ("湾", 2, 6),
        ("庭", 2, 6),
    )
    CONTEXT_STRONG_PLACE_SUFFIXES = (
        "大道",
        "道路",
        "办事处",
        "小区",
        "花园",
        "公寓",
        "广场",
        "苑",
        "园",
        "城",
        "府",
        "湾",
        "庭",
    )

    def __init__(self, *, progress_path: Optional[str | Path] = None) -> None:
        self.progress_path = Path(progress_path).expanduser().resolve() if progress_path else None
        self.renderer = PageRenderer()

    def audit(
        self,
        *,
        pdf_path: str | Path,
        wps_docx_path: str | Path,
        output_dir: str | Path,
        audit_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_pdf = Path(pdf_path).expanduser().resolve()
        template_docx = Path(wps_docx_path).expanduser().resolve()
        output_root = Path(output_dir).expanduser().resolve()
        ensure_private_directory(output_root)
        audit_id = audit_id or uuid.uuid4().hex
        prefix = f"{source_pdf.stem}_{audit_id[:10]}"
        reviewed_docx_path = output_root / f"{prefix}_reviewed.docx"
        report_path = output_root / f"{prefix}_audit_report.json"
        evidence_zip_path = output_root / f"{prefix}_evidence.zip"
        work_dir = output_root / "v4_evidence" / audit_id[:12]
        raw_dir = work_dir / "evidence" / "raw"
        ensure_private_directory(raw_dir)
        total_steps = 17

        warnings: List[str] = []
        timings: Dict[str, float] = {}
        self._emit_progress("workflow_preflight", 1, total_steps, "正在做 PDF→DOCX 转换忠实度 v4 预审...")
        preflight = self._preflight(source_pdf=source_pdf, template_docx=template_docx)
        if preflight["errors"]:
            raise RuntimeError("pdf_word_audit_v4_preflight_failed:" + ",".join(preflight["errors"]))
        warnings.extend(preflight.get("warnings") or [])

        self._emit_progress("page_render", 2, total_steps, "正在渲染 PDF 页面作为复杂文档视觉基准...")
        start = time.perf_counter()
        page_count = self._page_count(source_pdf, warnings=warnings)
        all_pages = list(range(1, page_count + 1))
        rendered_pages = self.renderer.render_pages(
            pdf_path=source_pdf,
            page_numbers=all_pages,
            output_dir=work_dir / "pages",
            dpi=max(120, int(getattr(settings, "PDF_WORD_AUDIT_V4_RENDER_DPI", 180) or 180)),
            max_edge=max(1200, int(getattr(settings, "PDF_WORD_AUDIT_V4_RENDER_MAX_EDGE", 1900) or 1900)),
        )
        timings["page_render"] = round(time.perf_counter() - start, 4)

        self._emit_progress("page_orientation", 3, total_steps, "正在把 PDF 渲染页统一旋转到正向...")
        start = time.perf_counter()
        rendered_pages, page_orientation_payload = PageOrientationNormalizer(work_dir=work_dir).normalize(
            rendered_pages=rendered_pages,
        )
        warnings.extend(str(item) for item in (page_orientation_payload.get("summary") or {}).get("warnings") or [] if str(item))
        timings["page_orientation"] = round(time.perf_counter() - start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="page_orientation",
            current=3,
            total=total_steps,
            timings=timings,
            warnings=warnings,
            extra={"page_orientation": page_orientation_payload},
        )

        self._emit_progress("evidence_extract", 4, total_steps, "正在提取 PDF 可见证据和 DOCX XML 内容单元...")
        start = time.perf_counter()
        pdf_units, page_profiles, pdf_warnings = PdfEvidenceExtractor().extract(
            pdf_path=source_pdf,
            rendered_pages=rendered_pages,
            page_count=page_count,
        )
        self._apply_page_orientation_profiles(page_profiles=page_profiles, page_orientation=page_orientation_payload)
        warnings.extend(pdf_warnings)
        anchor_units, anchor_ocr_payload, anchor_warnings = AnchorOcrExtractor(work_dir=work_dir).extract(
            rendered_pages=rendered_pages,
            page_profiles=page_profiles,
            page_count=page_count,
        )
        pdf_units.extend(anchor_units)
        warnings.extend(anchor_warnings)
        docx_units = DocxEvidenceExtractor().extract(docx_path=template_docx, estimated_page_count=page_count)
        docx_page_remap = DocxPageRemapper().remap(
            docx_units=docx_units,
            pdf_units=pdf_units,
            page_profiles=page_profiles,
            pdf_page_count=page_count,
        )
        warnings.extend(str(item) for item in (docx_page_remap.get("summary") or {}).get("warnings") or [] if str(item))
        timings["evidence_extract"] = round(time.perf_counter() - start, 4)

        self._emit_progress("v4_alignment", 5, total_steps, "正在做 PDF-DOCX 页/段对齐和差异候选召回...")
        start = time.perf_counter()
        step_start = time.perf_counter()
        preflight_result = ConversionPreflightBuilder().build(
            pdf_page_count=page_count,
            pdf_units=pdf_units,
            docx_units=docx_units,
            page_profiles=page_profiles,
            warnings=warnings,
            anchor_ocr_payload=anchor_ocr_payload,
        )
        preflight_result.page_orientation = page_orientation_payload
        preflight_result.docx_page_remap = docx_page_remap
        MappingStabilizer().apply(preflight_result=preflight_result)
        AuditRoutePlanner().build(preflight_result=preflight_result)
        timings["v4_alignment"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_alignment",
            current=5,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_focused_review", 6, total_steps, "正在做候选局部 crop 和本地证据复核...")
        step_start = time.perf_counter()
        preflight_result.focused_reviews = FocusedReviewBuilder(work_dir=work_dir).build(
            preflight_result=preflight_result,
            rendered_pages=rendered_pages,
        )
        timings["v4_focused_review"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_focused_review",
            current=6,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_table_structure", 7, total_steps, "正在做表格结构和单元格级风险预审...")
        step_start = time.perf_counter()
        preflight_result.table_reviews = TableReviewBuilder(work_dir=work_dir).build(
            preflight_result=preflight_result,
            rendered_pages=rendered_pages,
        )
        timings["v4_table_structure"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_table_structure",
            current=7,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_content_coverage", 8, total_steps, "正在做全内容覆盖检查和局部 OCR 回填...")
        step_start = time.perf_counter()
        preflight_result.content_coverage_reviews = FullContentCoverageBuilder().build(preflight_result=preflight_result)
        preflight_result.content_coverage_backfills = ContentCoverageBackfillBuilder(
            work_dir=work_dir,
            progress_callback=self._builder_progress_callback(stage="v4_content_coverage", current=8, total=total_steps),
        ).build(
            preflight_result=preflight_result,
            rendered_pages=rendered_pages,
        )
        ContentCoverageBackfillResolver().apply(preflight_result=preflight_result)
        preflight_result.table_cell_evidence_reviews = TableCellEvidenceBuilder().build(
            preflight_result=preflight_result,
        )
        preflight_result.table_grid_evidence = TableGridEvidenceBuilder().build(
            preflight_result=preflight_result,
        )
        timings["v4_content_coverage"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_content_coverage",
            current=8,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_page_risk_aggregation", 9, total_steps, "正在聚合扫描页碎片和页级覆盖风险...")
        step_start = time.perf_counter()
        preflight_result.fragment_anomaly_reviews = FragmentAnomalyBuilder().build(
            preflight_result=preflight_result,
        )
        preflight_result.image_page_reviews = ImagePdfPageReviewBuilder().build(
            preflight_result=preflight_result,
        )
        document_context_terms = self._document_context_terms(preflight_result=preflight_result)
        preflight_result.page_ocr_text_evidence_reviews = PageOcrTextEvidenceBuilder().build(
            preflight_result=preflight_result,
            text_hint_fn=self._docx_text_artifact_hint,
            priority_fn=self._docx_text_artifact_priority,
            context_terms=document_context_terms,
        )
        preflight_result.page_text_coverage_profiles = PageTextCoverageProfileBuilder().build(
            preflight_result=preflight_result,
        )
        preflight_result.audit_route_plan["model_candidate_recall"] = ModelCandidateRecallBuilder(work_dir=work_dir).apply(
            preflight_result=preflight_result,
            rendered_pages=rendered_pages,
        )
        timings["v4_page_risk_aggregation"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_page_risk_aggregation",
            current=9,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        qwen_vl_shared_session = bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_SHARED_SESSION_ENABLED", False))
        self._emit_progress("v4_image_text_vl", 10, total_steps, "正在执行图片页 Qwen3-VL 全文视觉复核...")
        step_start = time.perf_counter()
        preflight_result.image_text_vl_reviews = ImagePageVlTextBuilder(
            work_dir=work_dir,
            unload_after=not qwen_vl_shared_session,
            progress_callback=self._builder_progress_callback(stage="v4_image_text_vl", current=10, total=total_steps),
        ).build(
            preflight_result=preflight_result,
            rendered_pages=rendered_pages,
        )
        timings["v4_image_text_vl"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_image_text_vl",
            current=10,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_table_page_vl", 11, total_steps, "正在执行表格页 Qwen3-VL 视觉复核...")
        step_start = time.perf_counter()
        preflight_result.table_page_vl_reviews = TablePageVlBuilder(
            work_dir=work_dir,
            unload_after=not qwen_vl_shared_session,
            progress_callback=self._builder_progress_callback(stage="v4_table_page_vl", current=11, total=total_steps),
        ).build(
            preflight_result=preflight_result,
            rendered_pages=rendered_pages,
        )
        preflight_result.table_grid_evidence = TableGridEvidenceBuilder().build(
            preflight_result=preflight_result,
        )
        timings["v4_table_page_vl"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_table_page_vl",
            current=11,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_qwen_vl_gate", 12, total_steps, "正在执行 Qwen3-VL 视觉门槛复核...")
        step_start = time.perf_counter()
        preflight_result.qwen_vl_reviews = QwenVlGateBuilder(
            work_dir=work_dir,
            unload_after=not qwen_vl_shared_session,
            progress_callback=self._builder_progress_callback(stage="v4_qwen_vl_gate", current=12, total=total_steps),
        ).build(preflight_result=preflight_result)
        if qwen_vl_shared_session:
            self._unload_qwen_vl_model()
        timings["v4_qwen_vl_gate"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_qwen_vl_gate",
            current=12,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_qwen_text_gate", 13, total_steps, "正在执行 Qwen 文本最终门槛复核...")
        step_start = time.perf_counter()
        preflight_result.qwen_gate_reviews = FocusedQwenGateBuilder().build(preflight_result=preflight_result)
        preflight_result.page_text_qwen_reviews = PageTextQwenReviewBuilder(
            work_dir=work_dir,
            progress_callback=self._builder_progress_callback(stage="v4_qwen_text_gate", current=13, total=total_steps),
        ).build(preflight_result=preflight_result)
        model_output_guard = ModelOutputGuard()
        model_output_guard.apply(preflight_result=preflight_result)
        timings["v4_qwen_text_gate"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_qwen_text_gate",
            current=13,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_specialist_review_plan", 14, total_steps, "正在整理未确认高风险项专项复核计划...")
        step_start = time.perf_counter()
        preflight_result.specialist_review_tasks = SpecialistReviewPlanBuilder().build(preflight_result=preflight_result)
        timings["v4_specialist_review_plan"] = round(time.perf_counter() - step_start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_specialist_review_plan",
            current=14,
            total=total_steps,
            preflight_result=preflight_result,
            timings=timings,
            warnings=warnings,
        )
        self._emit_progress("v4_table_specialist_review", 15, total_steps, "正在执行表格/图片页专项复核并筛选可批注结果...")
        step_start = time.perf_counter()
        table_specialist_results = TableSpecialistExecutor().build(preflight_result=preflight_result)
        image_specialist_results = ImagePageSpecialistExecutor().build(
            preflight_result=preflight_result,
            text_hint_fn=self._docx_text_artifact_hint,
            priority_fn=self._docx_text_artifact_priority,
            context_terms=document_context_terms,
        )
        preflight_result.specialist_review_results = table_specialist_results + image_specialist_results
        model_output_guard.apply(preflight_result=preflight_result)
        self._apply_confirmed_review_coverage(preflight_result)
        TableHeavyPageCoverageCloser().apply(preflight_result=preflight_result)
        FullPageReviewCoverageCloser().apply(preflight_result=preflight_result)
        preflight_result.table_grid_evidence = TableGridEvidenceBuilder().build(
            preflight_result=preflight_result,
        )
        preflight_result.table_audit_summary = TableAuditEngine().build(preflight_result=preflight_result)
        preflight_result.high_risk_page_coverage_reviews = HighRiskPageCoverageBuilder().build(
            preflight_result=preflight_result,
        )
        coverage_scheduler = CoverageGapScheduler()
        preflight_result.coverage_review_tasks = coverage_scheduler.build(preflight_result=preflight_result)
        CoverageTaskExecutionBridge().apply(preflight_result=preflight_result)
        timings["v4_table_specialist_review"] = round(time.perf_counter() - step_start, 4)
        corrections = self._comment_corrections(preflight_result)
        preflight_result.quality_inspection = QualityInspectorBuilder().build(
            preflight_result=preflight_result,
            corrections=corrections,
        )
        timings["conversion_preflight"] = round(time.perf_counter() - start, 4)
        self._write_stage_checkpoint(
            raw_dir=raw_dir,
            stage="v4_table_specialist_review",
            current=15,
            total=total_steps,
            preflight_result=preflight_result,
            corrections=corrections,
            timings=timings,
            warnings=warnings,
        )

        self._emit_progress("audit_write_report", 16, total_steps, "正在写入 v4 复核 DOCX 批注、报告和证据包...")
        start = time.perf_counter()
        writer_summary = DocxAuditCommentWriter().write(
            template_docx_path=template_docx,
            output_docx_path=reviewed_docx_path,
            corrections=corrections,
        )
        findings = self._build_findings(preflight_result=preflight_result, corrections=corrections)
        product_report = ProductReportBuilder().build(
            audit_id=audit_id,
            preflight_result=preflight_result,
            findings=findings,
            corrections=corrections,
            reviewed_docx_path=reviewed_docx_path,
            report_path=report_path,
            evidence_zip_path=evidence_zip_path,
        )
        raw_payload_paths = self._write_raw_payloads(
            raw_dir=raw_dir,
            preflight_result=preflight_result,
            corrections=corrections,
            product_report=product_report,
        )
        product_report["artifact_manifest"]["raw_payload_paths"] = raw_payload_paths
        metadata = self._metadata(
            audit_id=audit_id,
            source_pdf=source_pdf,
            template_docx=template_docx,
            reviewed_docx_path=reviewed_docx_path,
            report_path=report_path,
            evidence_zip_path=evidence_zip_path,
            page_count=page_count,
            preflight=preflight,
            preflight_result=preflight_result,
            corrections=corrections,
            findings=findings,
            writer_summary=writer_summary,
            raw_payload_paths=raw_payload_paths,
            warnings=warnings,
            timings=timings,
        )
        report = {
            "audit_id": audit_id,
            "mode": "comment_only" if corrections else "report_only",
            "comment_count": len(corrections),
            "correction_count": len(corrections),
            "finding_count": len(findings),
            "metadata": metadata,
            "product_report": product_report,
            "page_risk_summary": product_report["page_risk_summary"],
            "page_terminal_summary": dict(product_report.get("page_terminal_summary") or {}),
            "table_summary": product_report["table_summary"],
            "coverage_summary": product_report["coverage_summary"],
            "artifact_manifest": product_report["artifact_manifest"],
            "report_truncation": self._report_truncation_summary(preflight_result),
            "findings": findings,
            "corrections": [item.to_dict() for item in corrections],
            "writer_summary": dict(writer_summary),
            "conversion_review_summary": self._review_summary(preflight_result, corrections=corrections),
            "conversion_preflight": preflight_result.raw_payload(),
            "page_orientation": dict(preflight_result.page_orientation or {}),
            "docx_page_remap": dict(preflight_result.docx_page_remap or {}),
            "mapping_stabilization": dict(preflight_result.mapping_stabilization or {}),
            "model_output_guard": dict(preflight_result.model_output_guard or {}),
            "audit_route_plan": dict(preflight_result.audit_route_plan),
            "conversion_diff_candidates": self._report_diff_candidates(preflight_result)[:500],
            "focused_candidate_reviews": [item.to_dict() for item in preflight_result.focused_reviews[:500]],
            "table_reviews": [item.to_dict() for item in preflight_result.table_reviews[:500]],
            "table_grid_evidence": [item.to_dict() for item in preflight_result.table_grid_evidence[:500]],
            "full_content_coverage_reviews": [item.to_dict() for item in preflight_result.content_coverage_reviews[:1000]],
            "full_content_backfill_reviews": [item.to_dict() for item in preflight_result.content_coverage_backfills[:500]],
            "fragment_anomaly_reviews": [item.to_dict() for item in preflight_result.fragment_anomaly_reviews[:200]],
            "image_pdf_page_reviews": [item.to_dict() for item in preflight_result.image_page_reviews[:500]],
            "image_text_vl_reviews": [item.to_dict() for item in preflight_result.image_text_vl_reviews[:300]],
            "page_ocr_text_evidence_reviews": [item.to_dict() for item in preflight_result.page_ocr_text_evidence_reviews[:500]],
            "page_text_qwen_reviews": [item.to_dict() for item in preflight_result.page_text_qwen_reviews[:300]],
            "page_text_coverage_profiles": [item.to_dict() for item in preflight_result.page_text_coverage_profiles[:300]],
            "high_risk_page_coverage_reviews": [item.to_dict() for item in preflight_result.high_risk_page_coverage_reviews[:200]],
            "coverage_review_tasks": [item.to_dict() for item in preflight_result.coverage_review_tasks[:500]],
            "review_task_summary": coverage_scheduler.summary(tasks=preflight_result.coverage_review_tasks),
            "human_review_queue": coverage_scheduler.human_review_queue(
                tasks=preflight_result.coverage_review_tasks,
                limit=max(9999, len(preflight_result.coverage_review_tasks)),
            ),
            "table_page_vl_reviews": [item.to_dict() for item in preflight_result.table_page_vl_reviews[:200]],
            "table_cell_evidence_reviews": [item.to_dict() for item in preflight_result.table_cell_evidence_reviews[:1000]],
            "table_audit_summary": dict(preflight_result.table_audit_summary or {}),
            "qwen_vl_reviews": [item.to_dict() for item in preflight_result.qwen_vl_reviews[:500]],
            "qwen_gate_reviews": [item.to_dict() for item in preflight_result.qwen_gate_reviews[:500]],
            "quality_inspection": dict(preflight_result.quality_inspection),
            "specialist_review_tasks": [item.to_dict() for item in preflight_result.specialist_review_tasks[:500]],
            "specialist_review_results": [item.to_dict() for item in preflight_result.specialist_review_results[:500]],
            "review_routes": [item.to_dict() for item in preflight_result.review_routes[:500]],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ensure_private_file(report_path)
        self._build_evidence_zip(evidence_zip_path=evidence_zip_path, report_path=report_path, work_dir=work_dir)
        metadata["audit_report_path"] = str(report_path)
        metadata["evidence_zip_path"] = str(evidence_zip_path)
        timings["write_outputs"] = round(time.perf_counter() - start, 4)

        self._emit_progress("audit_done", 17, total_steps, "PDF→DOCX 转换忠实度 v4 预审完成。")
        return {
            "audit_id": audit_id,
            "audited_docx_path": str(reviewed_docx_path),
            "reviewed_docx_path": str(reviewed_docx_path),
            "audit_report_path": str(report_path),
            "evidence_zip_path": str(evidence_zip_path),
            "metadata": metadata,
            "product_report": product_report,
            "findings": findings,
            "corrections": report["corrections"],
            "comment_count": len(corrections),
            "correction_count": len(corrections),
            "finding_count": len(findings),
        }

    def _preflight(self, *, source_pdf: Path, template_docx: Path) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        if not source_pdf.exists() or source_pdf.suffix.lower() != ".pdf":
            errors.append("source_pdf_missing_or_invalid")
        if not template_docx.exists() or template_docx.suffix.lower() != ".docx":
            errors.append("wps_docx_template_missing_or_invalid")
        return {"errors": errors, "warnings": warnings}

    def _page_count(self, source_pdf: Path, *, warnings: List[str]) -> int:
        try:
            return max(1, int(self.renderer.page_count(source_pdf)))
        except Exception:
            logger.debug("Failed to count PDF pages for v4.", exc_info=True)
            warnings.append("pdf_page_count_failed_default_1")
            return 1

    def _unload_qwen_vl_model(self) -> None:
        """Release the visual model before the text-only Qwen gate loads."""
        try:
            model = str(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_MODEL", "qwen3-vl:8b") or "qwen3-vl:8b").strip()
            OllamaQwenVlClient(model=model).unload()
        except Exception:
            logger.debug("Failed to unload shared Qwen-VL session.", exc_info=True)

    def _apply_page_orientation_profiles(self, *, page_profiles: Dict[str, Dict[str, Any]], page_orientation: Dict[str, Any]) -> None:
        for page in page_orientation.get("pages") or []:
            if not isinstance(page, dict):
                continue
            try:
                page_no = int(page.get("page_no") or 0)
            except Exception:
                page_no = 0
            if page_no <= 0:
                continue
            profile = page_profiles.setdefault(str(page_no), {})
            profile["orientation_normalized"] = bool(page.get("normalized"))
            profile["orientation_rotation_degrees"] = int(page.get("rotation_degrees") or 0)
            profile["orientation_detected_rotation_degrees"] = int(page.get("detected_rotation_degrees") or 0)
            profile["orientation_confidence"] = float(page.get("confidence") or 0.0)
            profile["orientation_reason"] = str(page.get("reason") or "")[:220]

    def _write_stage_checkpoint(
        self,
        *,
        raw_dir: Path,
        stage: str,
        current: int,
        total: int,
        preflight_result: Optional[ConversionPreflightResult] = None,
        corrections: Sequence[CorrectionCandidate] = (),
        timings: Optional[Dict[str, float]] = None,
        warnings: Sequence[str] = (),
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist inspectable partial state after expensive v4 stages.

        The v4 workflow can spend a long time in serial OCR/VL/Qwen calls. If
        the user pauses or the worker exits mid-run, completed stages should
        still leave structured evidence on disk instead of only image crops.
        """

        try:
            ensure_private_directory(raw_dir)
            raw_payload_paths: List[str] = []
            if preflight_result is not None:
                raw_payload_paths = self._write_raw_payloads(
                    raw_dir=raw_dir,
                    preflight_result=preflight_result,
                    corrections=corrections,
                )
            for filename, payload in (extra or {}).items():
                if not filename.endswith(".json"):
                    filename = f"{filename}.json"
                path = raw_dir / filename
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                ensure_private_file(path)
                rel = f"evidence/raw/{filename}"
                if rel not in raw_payload_paths:
                    raw_payload_paths.append(rel)

            checkpoint_dir = raw_dir / "checkpoints"
            ensure_private_directory(checkpoint_dir)
            payload = {
                "enabled": True,
                "version": "v4_stage_checkpoint_v1",
                "stage": str(stage),
                "current": int(current),
                "total": int(total),
                "updated_at": time.time(),
                "timings": dict(timings or {}),
                "warnings": [str(item) for item in warnings if str(item)],
                "raw_payload_paths": raw_payload_paths,
            }
            if preflight_result is not None:
                payload["conversion_fidelity"] = preflight_result.summary()
                payload["counts"] = self._checkpoint_counts(preflight_result=preflight_result, corrections=corrections)
            if extra:
                payload["extra_keys"] = sorted(str(key) for key in extra.keys())

            stage_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage)).strip("_") or "stage"
            for path in (
                checkpoint_dir / f"checkpoint_{int(current):02d}_{stage_name}.json",
                checkpoint_dir / "checkpoint_latest.json",
            ):
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                ensure_private_file(path)
        except Exception:
            logger.debug("Failed to write v4 stage checkpoint for %s.", stage, exc_info=True)

    def _checkpoint_counts(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
    ) -> Dict[str, Any]:
        return {
            "pdf_page_count": int(preflight_result.pdf_page_count or 0),
            "pdf_unit_count": len(preflight_result.pdf_units),
            "docx_unit_count": len(preflight_result.docx_units),
            "diff_candidate_count": len(preflight_result.diff_candidates),
            "focused_review_count": len(preflight_result.focused_reviews),
            "content_coverage_review_count": len(preflight_result.content_coverage_reviews),
            "content_coverage_backfill_count": len(preflight_result.content_coverage_backfills),
            "table_page_vl_review_count": len(preflight_result.table_page_vl_reviews),
            "table_cell_evidence_review_count": len(preflight_result.table_cell_evidence_reviews),
            "table_grid_evidence_count": len(preflight_result.table_grid_evidence),
            "image_text_vl_review_count": len(preflight_result.image_text_vl_reviews),
            "qwen_vl_review_count": len(preflight_result.qwen_vl_reviews),
            "qwen_gate_review_count": len(preflight_result.qwen_gate_reviews),
            "page_text_qwen_review_count": len(preflight_result.page_text_qwen_reviews),
            "coverage_review_task_count": len(preflight_result.coverage_review_tasks),
            "specialist_task_count": len(preflight_result.specialist_review_tasks),
            "specialist_result_count": len(preflight_result.specialist_review_results),
            "correction_count": len(corrections),
        }

    def _write_raw_payloads(
        self,
        *,
        raw_dir: Path,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate] = (),
        product_report: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        qwen_vl_candidate_count = preflight_result.qwen_vl_candidate_count()
        payloads = {
            "conversion_preflight.json": preflight_result.raw_payload(),
            "product_report.json": product_report
            or {
                "enabled": False,
                "version": "product_report_v1",
            },
            "anchor_ocr_pages.json": preflight_result.anchor_ocr
            or {
                "enabled": False,
                "version": "anchor_ocr_v1",
                "summary": {"enabled": False, "attempted_page_count": 0, "succeeded_page_count": 0},
                "pages": [],
            },
            "page_orientation.json": preflight_result.page_orientation
            or {
                "enabled": False,
                "version": "page_orientation_v1",
                "summary": {"enabled": False},
                "pages": [],
            },
            "pdf_evidence_units.json": {
                "enabled": True,
                "version": "pdf_evidence_units_v1",
                "units": [item.to_dict() for item in preflight_result.pdf_units],
            },
            "docx_evidence_units.json": {
                "enabled": True,
                "version": "docx_evidence_units_v1",
                "units": [item.to_dict() for item in preflight_result.docx_units],
            },
            "docx_page_remap.json": preflight_result.docx_page_remap
            or {
                "enabled": False,
                "version": "docx_page_remap_v1",
                "summary": {"enabled": False},
                "table_reviews": [],
            },
            "mapping_stabilization.json": preflight_result.mapping_stabilization
            or {
                "enabled": False,
                "version": "mapping_stabilizer_v1",
            },
            "alignment_links.json": {
                "enabled": True,
                "version": "alignment_links_v1",
                "links": [item.to_dict() for item in preflight_result.alignment_links],
            },
            "conversion_diff_candidates.json": {
                "enabled": True,
                "version": "conversion_diff_candidates_v1",
                "candidates": [item.to_dict() for item in preflight_result.diff_candidates],
            },
            "audit_route_plan.json": preflight_result.audit_route_plan
            or {
                "enabled": False,
                "version": "audit_route_plan_v1",
                "summary": {"enabled": False},
                "pages": [],
            },
            "focused_candidate_reviews.json": {
                "enabled": True,
                "version": "focused_review_v1",
                "reviews": [item.to_dict() for item in preflight_result.focused_reviews],
            },
            "table_reviews.json": {
                "enabled": True,
                "version": "table_parse_review_v1",
                "reviews": [item.to_dict() for item in preflight_result.table_reviews],
            },
            "full_content_coverage_reviews.json": {
                "enabled": True,
                "version": "full_content_coverage_v1",
                "reviews": [item.to_dict() for item in preflight_result.content_coverage_reviews],
            },
            "full_content_backfill_reviews.json": {
                "enabled": True,
                "version": "full_content_backfill_v1",
                "reviews": [item.to_dict() for item in preflight_result.content_coverage_backfills],
            },
            "full_content_reconciliation.json": preflight_result.content_coverage_reconciliation
            or {
                "enabled": False,
                "version": "content_coverage_backfill_reconciliation_v1",
            },
            "table_page_vl_reviews.json": {
                "enabled": True,
                "version": "table_page_vl_v1",
                "reviews": [item.to_dict() for item in preflight_result.table_page_vl_reviews],
            },
            "table_cell_evidence_reviews.json": {
                "enabled": True,
                "version": "table_cell_evidence_v1",
                "reviews": [item.to_dict() for item in preflight_result.table_cell_evidence_reviews],
            },
            "table_audit_summary.json": preflight_result.table_audit_summary
            or {
                "enabled": False,
                "version": "table_audit_engine_v1",
            },
            "table_grid_evidence.json": {
                "enabled": True,
                "version": "table_grid_evidence_v1",
                "grids": [item.to_dict() for item in preflight_result.table_grid_evidence],
            },
            "fragment_anomaly_reviews.json": {
                "enabled": True,
                "version": "fragment_anomaly_v1",
                "reviews": [item.to_dict() for item in preflight_result.fragment_anomaly_reviews],
            },
            "image_pdf_page_reviews.json": {
                "enabled": True,
                "version": "image_pdf_page_review_v1",
                "reviews": [item.to_dict() for item in preflight_result.image_page_reviews],
            },
            "image_text_vl_reviews.json": {
                "enabled": True,
                "version": "image_text_vl_v1",
                "reviews": [item.to_dict() for item in preflight_result.image_text_vl_reviews],
            },
            "page_ocr_text_evidence_reviews.json": {
                "enabled": True,
                "version": "page_ocr_text_evidence_v1",
                "reviews": [item.to_dict() for item in preflight_result.page_ocr_text_evidence_reviews],
            },
            "page_text_qwen_reviews.json": {
                "enabled": True,
                "version": "page_text_qwen_review_v1",
                "reviews": [item.to_dict() for item in preflight_result.page_text_qwen_reviews],
            },
            "page_text_coverage_profiles.json": {
                "enabled": True,
                "version": "page_text_coverage_v1",
                "profiles": [item.to_dict() for item in preflight_result.page_text_coverage_profiles],
            },
            "high_risk_page_coverage_reviews.json": {
                "enabled": True,
                "version": "high_risk_page_coverage_v1",
                "reviews": [item.to_dict() for item in preflight_result.high_risk_page_coverage_reviews],
            },
            "coverage_review_tasks.json": {
                "enabled": True,
                "version": "coverage_gap_scheduler_v1",
                "summary": CoverageGapScheduler().summary(tasks=preflight_result.coverage_review_tasks),
                "tasks": [item.to_dict() for item in preflight_result.coverage_review_tasks],
            },
            "qwen_gate_reviews.json": {
                "enabled": True,
                "version": "focused_qwen_gate_v1",
                "reviews": [item.to_dict() for item in preflight_result.qwen_gate_reviews],
            },
            "model_output_guard.json": preflight_result.model_output_guard
            or {
                "enabled": False,
                "version": "model_output_guard_v1",
            },
            "qwen_vl_reviews.json": {
                "enabled": True,
                "version": "qwen_vl_gate_v1",
                "candidate_count": qwen_vl_candidate_count,
                "skipped_candidate_count": max(0, qwen_vl_candidate_count - len(preflight_result.qwen_vl_reviews)),
                "reviews": [item.to_dict() for item in preflight_result.qwen_vl_reviews],
            },
            "quality_inspection.json": preflight_result.quality_inspection
            or {
                "enabled": False,
                "version": "v4_quality_inspection_v1",
                "summary": {"enabled": False},
            },
            "specialist_review_tasks.json": {
                "enabled": True,
                "version": "specialist_review_plan_v1",
                "tasks": [item.to_dict() for item in preflight_result.specialist_review_tasks],
            },
            "specialist_review_results.json": {
                "enabled": True,
                "version": "specialist_review_results_v1",
                "results": [item.to_dict() for item in preflight_result.specialist_review_results],
            },
            "review_routes.json": {
                "enabled": True,
                "version": "review_routes_v1",
                "routes": [item.to_dict() for item in preflight_result.review_routes],
            },
            "comment_corrections.json": {
                "enabled": True,
                "version": "v4_comment_corrections_v1",
                "corrections": [item.to_dict() for item in corrections],
            },
        }
        written: List[str] = []
        for filename, payload in payloads.items():
            path = raw_dir / filename
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            ensure_private_file(path)
            written.append(f"evidence/raw/{filename}")
        return written

    def _report_diff_candidates(self, preflight_result: ConversionPreflightResult) -> List[Dict[str, Any]]:
        reviews_by_diff = {item.diff_id: item.to_dict() for item in preflight_result.focused_reviews}
        qwen_by_diff = {item.diff_id: item.to_dict() for item in preflight_result.qwen_gate_reviews}
        qwen_vl_by_diff = {item.diff_id: item.to_dict() for item in preflight_result.qwen_vl_reviews}
        table_by_diff: Dict[str, Dict[str, Any]] = {}
        for table in preflight_result.table_reviews:
            payload = table.to_dict()
            for diff_id in table.related_diff_ids:
                table_by_diff[diff_id] = payload
        rows: List[Dict[str, Any]] = []
        for diff in preflight_result.diff_candidates:
            row = diff.to_dict()
            review = reviews_by_diff.get(diff.diff_id)
            if review:
                row["focused_status"] = review.get("status", "")
                row["focused_decision"] = review.get("decision", "")
                row["focused_next_route"] = review.get("next_route", "")
                row["focused_reason"] = review.get("reason", "")
                row["focused_crop_path"] = review.get("crop_path", "")
                row["focused_crop_ocr_quality"] = review.get("crop_ocr_quality", "")
                ocr_text = str(review.get("crop_ocr_text") or "")
                row["focused_crop_ocr_excerpt"] = ocr_text[:160]
            else:
                row["focused_status"] = "not_reviewed"
                row["focused_decision"] = ""
                row["focused_next_route"] = ""
            qwen = qwen_by_diff.get(diff.diff_id)
            if qwen:
                row["qwen_gate_verdict"] = qwen.get("verdict", "")
                row["qwen_gate_decision"] = qwen.get("decision", "")
                row["qwen_gate_reason"] = qwen.get("reason", "")
                row["qwen_gate_next_route"] = qwen.get("next_route", "")
                row["qwen_gate_confidence"] = qwen.get("confidence", 0.0)
            else:
                row["qwen_gate_verdict"] = ""
                row["qwen_gate_decision"] = ""
                row["qwen_gate_next_route"] = ""
            qwen_vl = qwen_vl_by_diff.get(diff.diff_id)
            if qwen_vl:
                row["qwen_vl_verdict"] = qwen_vl.get("verdict", "")
                row["qwen_vl_decision"] = qwen_vl.get("decision", "")
                row["qwen_vl_reason"] = qwen_vl.get("reason", "")
                row["qwen_vl_visible_text"] = qwen_vl.get("visible_text", "")
                row["qwen_vl_preferred_text"] = qwen_vl.get("preferred_text", "")
                row["qwen_vl_next_route"] = qwen_vl.get("next_route", "")
                row["qwen_vl_confidence"] = qwen_vl.get("confidence", 0.0)
            else:
                row["qwen_vl_verdict"] = ""
                row["qwen_vl_decision"] = ""
                row["qwen_vl_next_route"] = ""
            table = table_by_diff.get(diff.diff_id)
            if table:
                row["table_review_status"] = table.get("status", "")
                row["table_review_rows"] = table.get("row_count", 0)
                row["table_review_cols"] = table.get("col_count", 0)
                row["table_review_confidence"] = table.get("confidence", 0.0)
            else:
                row["table_review_status"] = ""
            rows.append(row)
        return rows

    def _comment_corrections(self, preflight_result: ConversionPreflightResult) -> List[CorrectionCandidate]:
        diffs_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        focused_by_id = {item.diff_id: item for item in preflight_result.focused_reviews}
        qwen_vl_by_id = {item.diff_id: item for item in preflight_result.qwen_vl_reviews}
        backfill_by_coverage_id = {item.coverage_review_id: item.to_dict() for item in preflight_result.content_coverage_backfills}
        docx_unit_ids = {item.unit_id for item in preflight_result.docx_units}
        corrections: List[CorrectionCandidate] = []
        used_docx_units: set[str] = set()
        vl_comment_spans: List[Dict[str, Any]] = []
        for vl in preflight_result.qwen_vl_reviews:
            diff = diffs_by_id.get(vl.diff_id)
            if diff is None or not diff.docx_unit_id or diff.docx_unit_id not in docx_unit_ids:
                continue
            if diff.docx_unit_id in used_docx_units:
                continue
            if vl.decision != "allow_report_candidate":
                continue
            focused = focused_by_id.get(vl.diff_id)
            comment_kind = self._comment_kind_for_vl(vl.to_dict())
            if not comment_kind:
                continue
            suggested = vl.preferred_text or vl.visible_text or diff.pdf_text or diff.pdf_value or ""
            page_no = int(diff.docx_estimated_page_no or diff.pdf_page_no or vl.page_no or 0)
            if self._is_redundant_visual_exact_comment(
                page_no=page_no,
                old_text=diff.docx_text,
                new_text=suggested,
                existing=vl_comment_spans,
            ):
                continue
            used_docx_units.add(diff.docx_unit_id)
            vl_comment_spans.append({"page_no": page_no, "old_text": diff.docx_text, "new_text": suggested})
            comment_text = self._vl_comment_text(diff=diff.to_dict(), focused=focused.to_dict() if focused else {}, vl=vl.to_dict(), comment_kind=comment_kind)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_vl_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=diff.docx_unit_id,
                    page_no=diff.docx_estimated_page_no or diff.pdf_page_no,
                    old_text=diff.docx_text,
                    new_text=suggested,
                    action="review",
                    confidence=max(float(vl.confidence or 0.0), float(diff.confidence or 0.0)),
                    alignment_score=float(diff.alignment_confidence or 0.0),
                    reason=vl.reason or diff.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
        for gate in preflight_result.qwen_gate_reviews:
            diff = diffs_by_id.get(gate.diff_id)
            if diff is None or not diff.docx_unit_id or diff.docx_unit_id not in docx_unit_ids:
                continue
            if diff.docx_unit_id in used_docx_units:
                continue
            focused = focused_by_id.get(gate.diff_id)
            comment_kind = self._comment_kind_for_gate(gate.to_dict())
            if not comment_kind:
                continue
            used_docx_units.add(diff.docx_unit_id)
            suggested = gate.preferred_text or diff.pdf_text or ""
            comment_text = self._comment_text(diff=diff.to_dict(), focused=focused.to_dict() if focused else {}, gate=gate.to_dict(), comment_kind=comment_kind)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_comment_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=diff.docx_unit_id,
                    page_no=diff.docx_estimated_page_no or diff.pdf_page_no,
                    old_text=diff.docx_text,
                    new_text=suggested,
                    action="review",
                    confidence=max(float(gate.confidence or 0.0), float(diff.confidence or 0.0)),
                    alignment_score=float(diff.alignment_confidence or 0.0),
                    reason=gate.reason or diff.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
        self._append_specialist_review_result_comments(
            corrections,
            preflight_result=preflight_result,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
        )
        self._append_visual_pending_comments(
            corrections,
            preflight_result=preflight_result,
            diffs_by_id=diffs_by_id,
            focused_by_id=focused_by_id,
            qwen_vl_by_id=qwen_vl_by_id,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
        )
        self._append_recall_guard_comments(
            corrections,
            preflight_result=preflight_result,
            diffs_by_id=diffs_by_id,
            focused_by_id=focused_by_id,
            qwen_vl_by_id=qwen_vl_by_id,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
        )
        self._append_table_page_vl_comments(
            corrections,
            preflight_result=preflight_result,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
        )
        self._append_fragment_anomaly_comments(
            corrections,
            preflight_result=preflight_result,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
        )
        if bool(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_PAGE_COMMENTS_ENABLED", False)):
            self._append_image_pdf_page_comments(
                corrections,
                preflight_result=preflight_result,
                docx_unit_ids=docx_unit_ids,
                used_docx_units=used_docx_units,
            )
        self._append_content_coverage_comments(
            corrections,
            preflight_result=preflight_result,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
            backfill_by_coverage_id=backfill_by_coverage_id,
        )
        self._append_high_risk_page_coverage_comments(
            corrections,
            preflight_result=preflight_result,
            docx_unit_ids=docx_unit_ids,
            used_docx_units=used_docx_units,
        )
        corrections = self._dedupe_comment_corrections(corrections)
        self._compact_comment_texts(corrections)
        return self._filter_docx_comment_corrections(corrections)

    def _filter_docx_comment_corrections(self, corrections: Sequence[CorrectionCandidate]) -> List[CorrectionCandidate]:
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_CONFIRMED_COMMENTS_ONLY", True)):
            return list(corrections)
        return [correction for correction in corrections if self._is_confirmed_docx_comment(correction)]

    def _is_confirmed_docx_comment(self, correction: CorrectionCandidate) -> bool:
        title = self._comment_title(correction.comment_text)
        confirmed_titles = {
            "转换确认错误",
            "视觉确认转换错误",
            "表格专项确认错误",
            "图片页正文专项确认错误",
        }
        if title not in confirmed_titles:
            return False
        text = str(correction.comment_text or "")
        if "疑似" in title or "提示" in title or "核查参考" in text:
            return False
        if "必须改为：" not in text:
            return False
        old_key = self._compact_for_match(correction.old_text)
        new_key = self._compact_for_match(correction.new_text)
        if not old_key or not new_key or old_key == new_key:
            return False
        if self._looks_like_incomplete_confusable_numeric_replacement(
            old_text=correction.old_text,
            new_text=correction.new_text,
        ):
            return False
        return True

    def _looks_like_incomplete_confusable_numeric_replacement(self, *, old_text: Any, new_text: Any) -> bool:
        old_value = self._compact_for_match(old_text)
        new_value = self._compact_for_match(new_text).replace(",", "").replace("，", "")
        if not re.fullmatch(r"\d{4,8}", new_value):
            return False
        if "." in old_value or "." in new_value:
            return False
        if not re.search(r"[A-Za-z]", old_value) or not re.search(r"\d", old_value):
            return False
        simple_confusable_numeric = bool(re.fullmatch(r"[bcdegiloqsz]\d{3,7}", old_value))
        if not simple_confusable_numeric and re.fullmatch(r"[a-z]{1,8}[-_/]?[a-z0-9][-_/a-z0-9]{2,}", old_value):
            return False
        return self._table_page_vl_numeric_key(old_value) == new_value

    def _is_redundant_visual_exact_comment(
        self,
        *,
        page_no: int,
        old_text: str,
        new_text: str,
        existing: Sequence[Dict[str, Any]],
    ) -> bool:
        old_key = self._compact_for_match(old_text)
        new_key = self._compact_for_match(new_text)
        if len(old_key) < 6 or len(new_key) < 6:
            return False
        for item in existing:
            if int(item.get("page_no") or 0) != int(page_no or 0):
                continue
            existing_old = self._compact_for_match(item.get("old_text"))
            existing_new = self._compact_for_match(item.get("new_text"))
            if not existing_old or not existing_new:
                continue
            if old_key == existing_old and new_key == existing_new:
                return True
            if self._contains_substantive_span(existing_old, old_key) and self._contains_substantive_span(existing_new, new_key):
                return True
        return False

    def _contains_substantive_span(self, container: str, candidate: str) -> bool:
        if len(candidate) < 6 or len(container) < len(candidate) + 4:
            return False
        return candidate in container

    def _dedupe_comment_corrections(self, corrections: Sequence[CorrectionCandidate]) -> List[CorrectionCandidate]:
        rows: List[CorrectionCandidate] = []
        seen: set[Tuple[int, str, str]] = set()
        for correction in corrections:
            old_key = self._compact_for_match(correction.old_text)
            new_key = self._compact_for_match(correction.new_text)
            if old_key and new_key:
                key = (int(correction.page_no or 0), old_key, new_key)
                if key in seen:
                    continue
                seen.add(key)
            rows.append(correction)
        return rows

    def _apply_confirmed_review_coverage(self, preflight_result: ConversionPreflightResult) -> None:
        """Mark coverage items resolved once a stricter layer confirmed them.

        "Resolved" here does not mean the DOCX is correct. It means the item is
        no longer an unprocessed coverage gap because a downstream gate has
        already produced a concrete review result/comment for the same unit.
        """

        resolved_unit_ids: set[str] = set()
        resolved_coverage_ids: set[str] = set()
        direct_resolved_unit_ids: set[str] = set()
        direct_resolved_coverage_ids: set[str] = set()
        no_issue_unit_ids: set[str] = set()
        no_issue_coverage_ids: set[str] = set()
        diff_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}

        for result in preflight_result.specialist_review_results:
            if result.decision != "confirmed_error" or result.comment_policy != "comment_if_exact_replacement":
                continue
            if result.wps_unit_id:
                resolved_unit_ids.add(result.wps_unit_id)
                direct_resolved_unit_ids.add(result.wps_unit_id)
            for ref in result.evidence_refs:
                if ref.get("source") == "content_coverage_review" and ref.get("id"):
                    resolved_coverage_ids.add(str(ref["id"]))
                    direct_resolved_coverage_ids.add(str(ref["id"]))

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
                direct_resolved_unit_ids.add(review.docx_unit_id)
            if review.coverage_review_id:
                resolved_coverage_ids.add(review.coverage_review_id)
                direct_resolved_coverage_ids.add(review.coverage_review_id)

        for review in preflight_result.page_ocr_text_evidence_reviews:
            if review.decision != "confirmed_error":
                continue
            if review.docx_unit_id:
                resolved_unit_ids.add(review.docx_unit_id)
                direct_resolved_unit_ids.add(review.docx_unit_id)
            if review.coverage_review_id:
                resolved_coverage_ids.add(review.coverage_review_id)
                direct_resolved_coverage_ids.add(review.coverage_review_id)

        for review in preflight_result.qwen_vl_reviews:
            if review.decision != "allow_report_candidate":
                continue
            diff = diff_by_id.get(review.diff_id)
            if diff and diff.docx_unit_id:
                resolved_unit_ids.add(diff.docx_unit_id)
        for review in preflight_result.qwen_gate_reviews:
            if review.decision != "allow_report_candidate":
                continue
            diff = diff_by_id.get(review.diff_id)
            if diff and diff.docx_unit_id:
                resolved_unit_ids.add(diff.docx_unit_id)

        for review in preflight_result.table_page_vl_reviews:
            if not review.available or review.verdict != "no_obvious_issue" or review.suspicious_values:
                continue
            for sample in review.docx_samples:
                coverage_id = str(sample.get("coverage_review_id") or sample.get("sample_id") or sample.get("id") or "").strip()
                unit_id = str(sample.get("unit_id") or "").strip()
                if coverage_id:
                    no_issue_coverage_ids.add(coverage_id)
                if unit_id:
                    no_issue_unit_ids.add(unit_id)

        resolved_unit_ids.update(no_issue_unit_ids)
        resolved_coverage_ids.update(no_issue_coverage_ids)
        direct_resolved_unit_ids.update(no_issue_unit_ids)
        direct_resolved_coverage_ids.update(no_issue_coverage_ids)

        coverage_by_id = {review.review_id: review for review in preflight_result.content_coverage_reviews}
        coverage_ids_by_link: Dict[str, set[str]] = {}
        coverage_ids_by_diff: Dict[str, set[str]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.related_link_id:
                coverage_ids_by_link.setdefault(review.related_link_id, set()).add(review.review_id)
            if review.related_diff_id:
                coverage_ids_by_diff.setdefault(review.related_diff_id, set()).add(review.review_id)

        pending_review_ids = list(direct_resolved_coverage_ids)
        pending_review_ids.extend(
            review.review_id
            for review in preflight_result.content_coverage_reviews
            if review.unit_id in direct_resolved_unit_ids
        )
        seen_review_ids: set[str] = set()
        while pending_review_ids:
            review_id = pending_review_ids.pop()
            if review_id in seen_review_ids:
                continue
            seen_review_ids.add(review_id)
            review = coverage_by_id.get(review_id)
            if review is None:
                continue
            for related_ids in (
                coverage_ids_by_link.get(review.related_link_id, set()) if review.related_link_id else set(),
                coverage_ids_by_diff.get(review.related_diff_id, set()) if review.related_diff_id else set(),
            ):
                for related_review_id in related_ids:
                    if related_review_id in resolved_coverage_ids:
                        continue
                    resolved_coverage_ids.add(related_review_id)
                    pending_review_ids.append(related_review_id)

        if not resolved_unit_ids and not resolved_coverage_ids:
            return
        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered":
                continue
            if review.review_id not in resolved_coverage_ids and review.unit_id not in resolved_unit_ids:
                continue
            review.decision = "covered"
            no_issue_resolved = review.review_id in no_issue_coverage_ids or review.unit_id in no_issue_unit_ids
            direct_confirmed_resolved = review.review_id in direct_resolved_coverage_ids or review.unit_id in direct_resolved_unit_ids
            if no_issue_resolved:
                review.status = "resolved_by_table_page_vl_no_issue"
            elif direct_confirmed_resolved:
                review.status = "resolved_by_confirmed_review"
            else:
                review.status = "resolved_by_linked_review_pair"
            review.confidence = max(float(review.confidence or 0.0), 0.8)
            review.reason = (
                "该表格内容样例已由表格页视觉复核检查，未发现可定位转换差异，不再计入未处理覆盖缺口。"
                if no_issue_resolved
                else (
                    "该内容单元已由更严格的视觉/表格/专项证据确认并生成具体复核结果，不再计入未处理覆盖缺口。"
                    if direct_confirmed_resolved
                    else "该内容单元与已确认的 PDF/DOCX 成对覆盖项共享同一对齐/差异链路，作为同一转换问题已完成处理，不再计入未处理覆盖缺口。"
                )
            )
            if no_issue_resolved:
                flag = "resolved_by_table_page_vl_no_issue"
            elif direct_confirmed_resolved:
                flag = "resolved_by_confirmed_review"
            else:
                flag = "resolved_by_linked_review_pair"
            if flag not in review.flags:
                review.flags.append(flag)

    def _append_specialist_review_result_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not preflight_result.specialist_review_results:
            return
        appended = 0
        type_limits = {
            "table_cell_specialist_review": max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_COMMENTS", 20) or 20)),
            "table_visual_specialist_review": max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_COMMENTS", 20) or 20)),
            "image_page_specialist_review": max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_COMMENTS", 16) or 16)),
        }
        type_counts: Dict[str, int] = {}
        per_page_count: Dict[int, int] = {}
        for result in preflight_result.specialist_review_results:
            is_confirmed_exact = result.decision == "confirmed_error" and result.comment_policy == "comment_if_exact_replacement"
            is_image_rule_candidate = (
                result.task_type == "image_page_specialist_review"
                and result.decision == "suspected_error"
                and result.comment_policy == "report_only_until_confirmed"
                and "rule_candidate_only" in set(result.flags or [])
            )
            if not (is_confirmed_exact or is_image_rule_candidate):
                continue
            if not result.wps_unit_id or result.wps_unit_id not in docx_unit_ids:
                continue
            related_unit_ids = [
                str(item.get("unit_id") or "")
                for item in (result.context or {}).get("related_units", [])
                if str(item.get("unit_id") or "")
            ]
            if result.wps_unit_id in used_docx_units:
                continue
            type_limit = type_limits.get(result.task_type, 0)
            if type_counts.get(result.task_type, 0) >= type_limit:
                continue
            if not result.old_text or not result.new_text:
                continue
            page_no = int(result.page_no or 0)
            page_limit = self._specialist_result_comment_page_limit(result=result)
            if page_no > 0 and per_page_count.get(page_no, 0) >= page_limit:
                continue
            used_docx_units.add(result.wps_unit_id)
            used_docx_units.update(related_unit_ids)
            if page_no > 0:
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            type_counts[result.task_type] = type_counts.get(result.task_type, 0) + 1
            comment_text = self._specialist_review_result_comment_text(result=result.to_dict())
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_specialist_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=result.wps_unit_id,
                    page_no=result.page_no,
                    old_text=result.old_text,
                    new_text=result.new_text,
                    action="review",
                    confidence=max(0.62, float(result.confidence or 0.0)),
                    alignment_score=0.0,
                    reason=result.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1

    def _specialist_result_comment_page_limit(self, *, result: Any) -> int:
        if str(result.task_type or "").startswith("table_"):
            return max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_COMMENTS_PER_PAGE", 20) or 20))
        return 10

    def _specialist_review_result_comment_text(self, *, result: Dict[str, Any]) -> str:
        if result.get("task_type") == "image_page_specialist_review":
            title = "图片页正文专项确认错误" if result.get("decision") == "confirmed_error" else "正文文字疑似识别错误"
        else:
            title = "表格专项确认错误" if result.get("decision") == "confirmed_error" else "表格文字疑似识别错误"
        context = dict(result.get("context") or {})
        location = self._table_context_location(context=context) if result.get("task_type") != "image_page_specialist_review" else ""
        group_summary = self._table_group_summary_text(context=context) if result.get("task_type") != "image_page_specialist_review" else ""
        related = self._table_related_units_text(context=context) if result.get("task_type") != "image_page_specialist_review" else ""
        lines = [
            f"WPS PDF转DOCX审查：{title}",
            f"页码：{result.get('page_no') or ''}",
        ]
        if location:
            lines.append(f"位置：{location}")
        lines.extend(
            [
                f"当前：{result.get('old_text') or ''}",
                f"建议改为：{result.get('new_text') or ''}",
            ]
        )
        if group_summary:
            lines.append(f"同列同类：{group_summary}")
        if related:
            lines.append(f"同类单元：{related}")
        if result.get("reason"):
            lines.append(f"依据：{result.get('reason')}")
        if result.get("task_type") == "image_page_specialist_review":
            if result.get("decision") == "confirmed_error":
                lines.append("说明：图片页专项已定位到具体 DOCX 文本单元；不自动改正文，请按原 PDF 确认后修改。")
            else:
                lines.append("说明：这是规则候选，不是确认错误；需 PDF OCR、Qwen-VL 或人工对照原 PDF 后确认。")
        else:
            if result.get("decision") == "confirmed_error":
                lines.append("说明：表格专项已定位到具体 DOCX 表格单元；不自动改正文，请按原 PDF 确认后修改。")
            else:
                lines.append("说明：这是规则候选，不是确认错误；需表格解析、PDF OCR、Qwen-VL 或人工对照原 PDF 后确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _table_context_location(self, *, context: Dict[str, Any]) -> str:
        position = dict(context.get("table_position") or {})
        parts = []
        if position.get("table_index"):
            parts.append(f"表{position.get('table_index')}")
        if position.get("row_index"):
            parts.append(f"行{position.get('row_index')}")
        if position.get("col_index"):
            parts.append(f"列{position.get('col_index')}")
        return " / ".join(parts)

    def _table_group_summary_text(self, *, context: Dict[str, Any]) -> str:
        summary = dict(context.get("group_summary") or {})
        count = int(summary.get("count") or 0)
        if count <= 1:
            return ""
        parts = [f"共{count}处"]
        if summary.get("table_index"):
            parts.append(f"表{summary.get('table_index')}")
        if summary.get("col_index"):
            parts.append(f"列{summary.get('col_index')}")
        parts.append("完整清单见报告")
        return "，".join(parts)

    def _table_related_units_text(self, *, context: Dict[str, Any]) -> str:
        rows = []
        for item in list(context.get("related_units") or [])[:4]:
            loc = []
            if item.get("row_index"):
                loc.append(f"行{item.get('row_index')}")
            if item.get("col_index"):
                loc.append(f"列{item.get('col_index')}")
            old_text = self._clip_comment_value(str(item.get("old_text") or ""), limit=24)
            new_text = self._clip_comment_value(str(item.get("new_text") or ""), limit=24)
            if not old_text or not new_text:
                continue
            rows.append(f"{''.join(loc) or item.get('unit_id')}：{old_text}->{new_text}")
        return "；".join(rows)

    def _compact_comment_texts(self, corrections: Sequence[CorrectionCandidate]) -> None:
        for correction in corrections:
            correction.comment_text = self._compact_comment_text(correction=correction)

    def _compact_comment_text(self, *, correction: CorrectionCandidate) -> str:
        title = self._comment_title(correction.comment_text)
        current = self._clip_comment_value(correction.old_text, limit=72)
        suggestion = self._clean_comment_suggestion(correction.new_text)
        if title == "PDF内容覆盖提示" and self._is_noisy_pdf_backfill_suggestion(correction.new_text):
            suggestion = ""
        action = self._compact_comment_action(title=title, suggestion=suggestion)
        basis = self._compact_comment_basis(title=title, correction=correction)
        lines = [title]
        if correction.page_no:
            lines.append(f"第{correction.page_no}页")
        location_line = self._comment_line(correction.comment_text, prefix="位置：")
        if location_line and title in {"表格专项确认错误", "表格文字疑似识别错误", "表格数字疑似识别错误"}:
            lines.append(location_line)
        if current:
            lines.append(f"当前：{current}")
        if action:
            lines.append(action)
        if title == "页级全量核查提示":
            focus_line = self._comment_line(correction.comment_text, prefix="重点：")
            coverage_line = self._comment_line(correction.comment_text, prefix="页级文本覆盖：")
            docx_gap_line = self._comment_line(correction.comment_text, prefix="DOCX缺口样例：")
            pdf_gap_line = self._comment_line(correction.comment_text, prefix="PDF缺口样例：")
            for extra_line, limit in (
                (focus_line, 100),
                (coverage_line, 100),
                (docx_gap_line, 100),
                (pdf_gap_line, 100),
            ):
                if extra_line:
                    lines.append(self._clip_comment_value(extra_line, limit=limit))
        group_line = self._comment_line(correction.comment_text, prefix="同列同类：")
        if group_line and title in {"表格专项确认错误", "表格文字疑似识别错误"}:
            lines.append(self._clip_comment_value(group_line, limit=72))
        related_line = self._comment_line(correction.comment_text, prefix="同类单元：")
        if related_line and title in {"表格专项确认错误", "表格文字疑似识别错误"}:
            lines.append(self._clip_comment_value(related_line, limit=90))
        if basis:
            lines.append(f"依据：{basis}")
        return "\n".join(line for line in lines if line.strip())

    def _comment_line(self, text: str, *, prefix: str) -> str:
        for line in str(text or "").splitlines():
            value = line.strip()
            if value.startswith(prefix):
                return value
        return ""

    def _comment_title(self, comment_text: str) -> str:
        first = str(comment_text or "").splitlines()[0].strip()
        if "：" in first:
            first = first.split("：", 1)[1].strip()
        return first or "转换复核提示"

    def _clean_comment_suggestion(self, value: str) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            return ""
        for prefix in ("疑似应核对为：", "建议核对为：", "建议改为："):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        if text.startswith("可疑字符替换参考："):
            text = text[len("可疑字符替换参考："):].split("；", 1)[0].strip()
        if text.startswith("包含 ") or text.startswith("PDF页OCR匹配值："):
            return ""
        if " | " in text and len(text) > 80:
            return ""
        text = self._clean_cjk_digit_spacing(text)
        return self._clip_comment_value(text, limit=110)

    def _clean_cjk_digit_spacing(self, text: str) -> str:
        value = str(text or "")
        value = re.sub(r"\s+([，。；：、）)])", r"\1", value)
        value = re.sub(r"(?<=[，。；：、])\s+(?=[\u4e00-\u9fff\d])", "", value)
        value = re.sub(r"([（(])\s+", r"\1", value)
        value = re.sub(r"(?<=[\u4e00-\u9fff\d])\s+(?=[\u4e00-\u9fff\d])", "", value)
        return " ".join(value.split()).strip()

    def _is_noisy_pdf_backfill_suggestion(self, value: str) -> bool:
        text = " ".join(str(value or "").split())
        if len(text) > 120:
            return True
        if re.search(r"(?<!\d)\d{3}年|\d{4}年\d{1,2}月\d{3,}日|[>]{1,}|\){1,}\d", text):
            return True
        compact = self._compact_for_match(text)
        if len(compact) >= 60:
            digit_ratio = sum(ch.isdigit() for ch in compact) / max(1, len(compact))
            ascii_ratio = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in compact) / max(1, len(compact))
            if digit_ratio > 0.28 or ascii_ratio > 0.18:
                return True
        return False

    def _compact_comment_action(self, *, title: str, suggestion: str) -> str:
        exact_titles = {
            "转换确认错误",
            "视觉确认转换错误",
            "表格视觉确认错误",
            "表格专项确认错误",
            "图片页正文专项确认错误",
        }
        if suggestion and title in exact_titles:
            return f"必须改为：{suggestion}"
        if suggestion and title in {"疑似转换错误", "表格数字疑似识别错误", "表格文字疑似识别错误", "正文文字疑似识别错误"}:
            return f"核查参考：{suggestion}"
        if suggestion and title in {"表格复核提示", "视觉复核提示", "映射复核提示"}:
            return f"核查参考：{suggestion}"
        if suggestion and title == "PDF内容覆盖提示":
            return f"局部PDF回填/核查是否补入：{suggestion}"
        if title == "漏检兜底提示":
            return "处理：对照原 PDF 核查该高风险字段。"
        if suggestion and title == "全内容覆盖提示":
            return f"核查对应内容：{suggestion}"
        if title == "页级全量核查提示":
            return "处理：按原 PDF 全量核查本页；未确认前不直接替换。"
        if "页级" in title or "整页" in title or title == "图片型PDF整页审查提示":
            return "处理：按原 PDF 对本页逐项核查正文、表格、数字和零散文字。"
        if "表格" in title:
            return "处理：按原 PDF 表格逐格核查该单元及同页相邻单元。"
        if "扫描页碎片" in title:
            return "处理：核查该页是否有重复、错序、漏行或碎片化文字。"
        if "视觉复核" in title:
            return "视觉门槛：未调用；处理：对照证据截图和原 PDF 核查该处文字。"
        return "处理：对照原 PDF 核查该处内容。"

    def _compact_comment_basis(self, *, title: str, correction: CorrectionCandidate) -> str:
        if title in {
            "表格专项确认错误",
            "图片页正文专项确认错误",
            "表格文字疑似识别错误",
            "正文文字疑似识别错误",
            "表格数字疑似识别错误",
        }:
            return ""
        if title == "表格视觉确认错误":
            return "PDF 页级 OCR 与 DOCX 表格值存在高风险差异。"
        if title == "转换确认错误":
            return "Qwen门槛确认 PDF 侧值。"
        if title == "疑似转换错误":
            return "Qwen门槛认为疑似错误，尚未达到确认错误门槛。"
        if title == "视觉确认转换错误":
            return "Qwen视觉门槛确认 PDF 侧值。"
        if title == "图片型PDF整页审查提示":
            return "该页为图片/扫描型 PDF，WPS 转换错误风险高。"
        if title == "页级全量核查提示":
            return "本页存在大量未可靠覆盖或映射不确定内容。"
        if title == "PDF内容覆盖提示":
            return "PDF 侧内容未找到可靠 DOCX 覆盖。"
        if title == "全内容覆盖提示":
            return "不是确认错误；DOCX 内容未被 PDF 侧可靠覆盖。"
        if title == "漏检兜底提示":
            return "不是确认错误；主候选链路未覆盖到同字段，需要兜底核查。"
        if title == "扫描页碎片/重复异常提示":
            return "同页出现重复片段或碎片化 OCR 文本。"
        if title == "视觉复核提示":
            return "局部 OCR 证据不足，未作为确认错误。"
        reason = " ".join(str(correction.reason or "").split())
        return self._clip_comment_value(reason, limit=90)

    def _clip_comment_value(self, value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _append_table_page_vl_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not preflight_result.table_page_vl_reviews:
            return
        if any(
            item.decision == "confirmed_error"
            and item.task_type == "table_visual_specialist_review"
            and item.comment_policy == "comment_if_exact_replacement"
            for item in preflight_result.specialist_review_results
        ):
            return
        docx_by_page: Dict[int, List[Any]] = {}
        docx_by_id = {item.unit_id: item for item in preflight_result.docx_units}
        coverage_by_id = {item.review_id: item for item in preflight_result.content_coverage_reviews if item.side == "docx"}
        for unit in preflight_result.docx_units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no > 0 and unit.unit_id in docx_unit_ids:
                docx_by_page.setdefault(page_no, []).append(unit)
        appended = 0
        per_page_count: Dict[int, int] = {}
        for review in preflight_result.table_page_vl_reviews:
            if appended >= 6:
                break
            if not review.available or not review.suspicious_values:
                continue
            for value in review.suspicious_values:
                if appended >= 6:
                    break
                if not self._valid_table_page_vl_comment_value(value=value):
                    continue
                page_no = int(review.page_no or 0)
                if per_page_count.get(page_no, 0) >= 3:
                    break
                anchor = self._pick_docx_unit_for_table_page_vl(
                    page_no=page_no,
                    value=value,
                    docx_by_page=docx_by_page,
                    docx_by_id=docx_by_id,
                    coverage_by_id=coverage_by_id,
                    used_docx_units=used_docx_units,
                )
                if anchor is None:
                    continue
                used_docx_units.add(anchor.unit_id)
                per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
                comment_text = self._table_page_vl_comment_text(review=review.to_dict(), value=value)
                corrections.append(
                    CorrectionCandidate(
                        id=f"v4_table_vl_{uuid.uuid4().hex[:12]}",
                        wps_unit_id=anchor.unit_id,
                        page_no=anchor.estimated_page_no or review.page_no,
                        old_text=str(value.get("docx_text") or anchor.text),
                        new_text=str(value.get("visible_text") or ""),
                        action="review",
                        confidence=max(float(review.confidence or 0.0), 0.62),
                        alignment_score=0.0,
                        reason=str(value.get("reason") or review.reason),
                        comment_text=comment_text,
                        sensitive_low_priority=False,
                    )
                )
                appended += 1

    def _pick_docx_unit_for_table_page_vl(
        self,
        *,
        page_no: int,
        value: Dict[str, Any],
        docx_by_page: Dict[int, List[Any]],
        docx_by_id: Dict[str, Any],
        coverage_by_id: Dict[str, Any],
        used_docx_units: set[str],
    ) -> Any | None:
        if not self._table_page_vl_value_has_anchor(value=value):
            return None
        for key in ("unit_id", "docx_unit_id"):
            unit_id = str(value.get(key) or "")
            unit = docx_by_id.get(unit_id)
            if unit is not None and unit.unit_id not in used_docx_units:
                return unit
        sample_id = str(value.get("sample_id") or value.get("coverage_review_id") or value.get("id") or "")
        coverage = coverage_by_id.get(sample_id)
        if coverage is not None:
            unit = docx_by_id.get(coverage.unit_id)
            if unit is not None and unit.unit_id not in used_docx_units:
                return unit
        row_key = self._table_position_key(value.get("row") or value.get("row_index"))
        col_key = self._table_position_key(value.get("col") or value.get("col_index"))
        if row_key and col_key:
            positional = [
                unit
                for unit in docx_by_page.get(page_no, [])
                if unit.unit_id not in used_docx_units
                and self._table_position_key(unit.row_index) == row_key
                and self._table_position_key(unit.col_index) == col_key
            ]
            if len(positional) == 1:
                return positional[0]
        wanted = self._compact_for_match(value.get("docx_text") or "")
        candidates = [item for item in docx_by_page.get(page_no, []) if item.unit_id not in used_docx_units]
        if wanted and self._table_page_vl_value_has_strong_anchor(value=value):
            for unit in candidates:
                candidate = self._compact_for_match(unit.text)
                if candidate and (candidate == wanted or wanted in candidate or candidate in wanted):
                    return unit
        return None

    def _table_page_vl_value_has_anchor(self, *, value: Dict[str, Any]) -> bool:
        if any(str(value.get(key) or "").strip() for key in ("unit_id", "docx_unit_id", "coverage_review_id", "sample_id", "id", "anchor_status", "grid_id")):
            return True
        return bool(self._table_position_key(value.get("row") or value.get("row_index")) and self._table_position_key(value.get("col") or value.get("col_index")))

    def _table_page_vl_value_has_strong_anchor(self, *, value: Dict[str, Any]) -> bool:
        return any(str(value.get(key) or "").strip() for key in ("unit_id", "docx_unit_id", "coverage_review_id", "sample_id", "id"))

    def _table_position_key(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"\d+", text)
        return str(int(match.group(0))) if match else ""

    def _valid_table_page_vl_comment_value(self, *, value: Dict[str, Any]) -> bool:
        if not self._table_page_vl_value_has_anchor(value=value):
            return False
        source = str(value.get("source") or "").strip()
        issue_type = str(value.get("issue_type") or "").strip()
        if source == "pdf_page_ocr":
            if issue_type not in {"digit_letter_confusion", "value_mismatch"}:
                return False
        elif issue_type not in {
            "digit_letter_confusion",
            "value_mismatch",
            "missing_or_extra_cell",
            "table_value_mismatch",
            "decimal_point_missing",
            "decimal_or_punctuation_pollution",
        }:
            return False
        old_text = " ".join(str(value.get("docx_text") or "").split())
        new_text = " ".join(str(value.get("visible_text") or "").split())
        if not old_text or not new_text:
            return False
        if self._compact_for_match(old_text) == self._compact_for_match(new_text):
            return False
        if len(new_text) > 80 or any(marker in new_text for marker in ("?", "？", "_", "看不清", "无法", "不确定")):
            return False
        old_digits = re.sub(r"\D+", "", old_text)
        new_digits = re.sub(r"\D+", "", new_text)
        numeric_like = bool((len(old_digits) >= 2 or len(new_digits) >= 2) and not re.search(r"[\u4e00-\u9fff]", old_text + new_text))
        if numeric_like and re.search(r"[A-Za-z]", new_text):
            return False
        if numeric_like and not self._valid_table_page_vl_numeric_replacement(old_text=old_text, new_text=new_text):
            return False
        if source != "pdf_page_ocr":
            reason = str(value.get("reason") or "").strip()
            if not reason:
                return False
            if self._noisy_table_page_vl_comment_text(old_text) or self._noisy_table_page_vl_comment_text(new_text):
                return False
        return True

    def _noisy_table_page_vl_comment_text(self, text: Any) -> bool:
        value = " ".join(str(text or "").split())
        compact = self._compact_for_match(value)
        if not compact:
            return True
        if len(value) > 80:
            return True
        if re.search(r"(\d)\1{7,}", compact):
            return True
        if re.search(r"([\u4e00-\u9fffA-Za-z])\1{5,}", compact):
            return True
        if compact.count("0") >= 10 and len(set(compact)) <= 4:
            return True
        if any(marker in value for marker in ("?", "？", "_", "看不清", "无法", "不确定", "疑似")):
            return True
        return False

    def _valid_table_page_vl_numeric_replacement(self, *, old_text: str, new_text: str) -> bool:
        new_value = str(new_text or "").replace(",", "").replace("，", "").strip()
        if not re.fullmatch(r"[¥￥]?\d{1,8}(?:\.\d{1,4})?", new_value):
            return False
        old_key = self._table_page_vl_numeric_key(old_text)
        new_key = self._table_page_vl_numeric_key(new_text)
        if len(old_key) < 2 or len(new_key) < 2:
            return False
        return old_key == new_key

    def _table_page_vl_numeric_key(self, text: Any) -> str:
        mapping = str.maketrans(
            {
                "B": "8",
                "b": "8",
                "D": "0",
                "d": "0",
                "O": "0",
                "o": "0",
                "Q": "0",
                "q": "0",
                "C": "0",
                "c": "0",
                "G": "6",
                "g": "6",
                "E": "6",
                "e": "6",
                "I": "1",
                "i": "1",
                "L": "1",
                "l": "1",
                "S": "5",
                "s": "5",
                "Z": "2",
                "z": "2",
            }
        )
        return re.sub(r"[^0-9]", "", str(text or "").translate(mapping))

    def _compact_for_match(self, text: Any) -> str:
        return "".join(str(text or "").split()).lower()

    def _table_page_vl_comment_text(self, *, review: Dict[str, Any], value: Dict[str, Any]) -> str:
        source = str(value.get("source") or "")
        page_ocr_only = bool(source == "pdf_page_ocr" or review.get("model") == "pdf_page_ocr" or not review.get("attempted"))
        title = "表格数字疑似识别错误" if page_ocr_only else "表格视觉确认错误"
        lines = [
            f"WPS PDF转DOCX审查：{title}",
            f"页码：{review.get('page_no') or ''}",
            f"DOCX表格单元：{value.get('docx_text') or '（空）'}",
        ]
        if value.get("visible_text"):
            visible_label = "PDF页OCR匹配" if value.get("source") == "pdf_page_ocr" else "PDF视觉可见"
            lines.append(f"{visible_label}：{value.get('visible_text')}")
        lines.append(f"模型判断：{review.get('verdict') or ''} / {value.get('issue_type') or ''} / {value.get('severity') or ''} / {review.get('confidence') or 0}")
        if value.get("reason"):
            lines.append(f"原因：{value.get('reason')}")
        elif review.get("reason"):
            lines.append(f"原因：{review.get('reason')}")
        if review.get("page_image_path"):
            lines.append(f"页面截图：{review.get('page_image_path')}")
        if review.get("next_route"):
            lines.append(f"后续：{review.get('next_route')}")
        lines.append("说明：这是 Qwen3-VL 正向页图复核和 PDF 页 OCR 匹配形成的表格提示；未自动修改正文，请按原 PDF 表格确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _append_fragment_anomaly_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not preflight_result.fragment_anomaly_reviews:
            return
        max_comments = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_FRAGMENT_ANOMALY_MAX_COMMENTS", 4) or 4))
        if max_comments <= 0:
            return
        docx_by_id = {item.unit_id: item for item in preflight_result.docx_units}
        appended = 0
        for review in preflight_result.fragment_anomaly_reviews:
            if appended >= max_comments:
                break
            payload = review.to_dict()
            if self._fragment_anomaly_covered_by_image_specialist(corrections=corrections, review=payload):
                continue
            anchor = self._pick_docx_unit_for_fragment_anomaly(
                review=payload,
                docx_by_id=docx_by_id,
                docx_unit_ids=docx_unit_ids,
                used_docx_units=used_docx_units,
            )
            if anchor is None:
                continue
            used_docx_units.add(anchor.unit_id)
            comment_text = self._fragment_anomaly_comment_text(review=payload, anchor_text=anchor.text)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_fragment_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=anchor.unit_id,
                    page_no=anchor.estimated_page_no or review.page_no,
                    old_text=anchor.text,
                    new_text=self._fragment_anomaly_hint_text(payload),
                    action="review",
                    confidence=max(float(review.confidence or 0.0), 0.58),
                    alignment_score=0.0,
                    reason=review.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1

    def _fragment_anomaly_covered_by_image_specialist(
        self,
        *,
        corrections: Sequence[CorrectionCandidate],
        review: Dict[str, Any],
    ) -> bool:
        page_no = int(review.get("page_no") or 0)
        if page_no <= 0:
            return False
        terms = self._fragment_anomaly_compact_terms(review)
        if not terms:
            return False
        for correction in corrections:
            if int(correction.page_no or 0) != page_no:
                continue
            if "图片页正文专项确认错误" not in str(correction.comment_text or ""):
                continue
            suggestion = self._compact_for_match(self._clean_comment_suggestion(correction.new_text))
            if not suggestion:
                continue
            if any(term and len(term) >= 5 and (term in suggestion or suggestion in term) for term in terms):
                return True
        return False

    def _fragment_anomaly_compact_terms(self, review: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for item in list(review.get("repeated_terms") or [])[:8]:
            for key in ("display", "term"):
                value = self._compact_for_match(item.get(key) or "")
                if value and value not in terms:
                    terms.append(value)
        return terms

    def _pick_docx_unit_for_fragment_anomaly(
        self,
        *,
        review: Dict[str, Any],
        docx_by_id: Dict[str, Any],
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> Any | None:
        anchor_id = str(review.get("anchor_unit_id") or "")
        anchor = docx_by_id.get(anchor_id)
        if anchor is not None and anchor.unit_id in docx_unit_ids and anchor.unit_id not in used_docx_units:
            return anchor
        examples = list(review.get("docx_examples") or [])
        example_ids = [str(item.get("unit_id") or "") for item in examples]
        candidates = [
            docx_by_id[unit_id]
            for unit_id in example_ids
            if unit_id in docx_by_id and unit_id in docx_unit_ids and unit_id not in used_docx_units
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-len(self._compact_for_match(item.text)), item.order_index))
        return candidates[0]

    def _fragment_anomaly_hint_text(self, review: Dict[str, Any]) -> str:
        terms = self._dedupe_fragment_hint_terms([
            str(item.get("display") or item.get("term") or "")
            for item in list(review.get("repeated_terms") or [])[:3]
            if str(item.get("display") or item.get("term") or "").strip()
        ])
        if not terms:
            return "删除或合并重复碎片；按原 PDF 只保留可见内容。"
        return f"删除或合并重复碎片；仅保留一次：{'；'.join(terms)}"[:180]

    def _dedupe_fragment_hint_terms(self, terms: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        for term in terms:
            value = " ".join(str(term or "").split())
            if not value:
                continue
            compact = self._compact_for_match(value)
            if any(compact and compact == self._compact_for_match(existing) for existing in cleaned):
                continue
            cleaned.append(value)
        result: List[str] = []
        compact_pairs = [(term, self._compact_for_match(term)) for term in cleaned]
        for term, compact in compact_pairs:
            if any(compact and compact != other and compact in other for _, other in compact_pairs):
                continue
            result.append(term)
        return result

    def _fragment_anomaly_comment_text(self, *, review: Dict[str, Any], anchor_text: str) -> str:
        terms = []
        for item in list(review.get("repeated_terms") or [])[:4]:
            label = str(item.get("display") or item.get("term") or "").strip()
            if not label:
                continue
            terms.append(f"{label}（{item.get('unit_count') or 0}处）")
        examples = [
            str(item.get("text") or "").strip()
            for item in list(review.get("docx_examples") or [])[:5]
            if str(item.get("text") or "").strip()
        ]
        lines = [
            "WPS PDF转DOCX审查：扫描页碎片/重复异常提示",
            f"页码：{review.get('page_no') or ''}",
            f"批注位置：{anchor_text or '（空）'}",
            f"页级异常：普通段落 {review.get('fragment_count') or 0} 个；短碎片 {review.get('short_fragment_count') or 0} 个；重复命中 {review.get('repeated_fragment_count') or 0} 个。",
        ]
        if terms:
            lines.append(f"重复片段：{' | '.join(terms)}")
        if examples:
            lines.append(f"DOCX样例：{' | '.join(examples)}")
        excerpt = str(review.get("pdf_anchor_excerpt") or "").strip()
        if excerpt:
            lines.append(f"PDF页级OCR摘要：{excerpt[:180]}")
        if review.get("reason"):
            lines.append(f"原因：{review.get('reason')}")
        if review.get("next_route"):
            lines.append(f"后续：{review.get('next_route')}")
        lines.append("说明：这是页级转换风险提示，不是确认字段错误；表示 WPS 可能把扫描页同一可见区域拆碎、重复或错位识别，需要按原 PDF 整页核查。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _append_image_pdf_page_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not preflight_result.image_page_reviews:
            return
        max_comments = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_IMAGE_PAGE_REVIEW_MAX_COMMENTS", 8) or 8))
        if max_comments <= 0:
            return
        docx_by_id = {item.unit_id: item for item in preflight_result.docx_units}
        appended = 0
        rows = [
            review
            for review in preflight_result.image_page_reviews
            if review.risk_level in {"high", "medium"} and review.anchor_unit_id
        ]
        rows.sort(key=lambda item: (0 if item.risk_level == "high" else 1, -int(item.unresolved_count or 0), item.page_no))
        for review in rows:
            if appended >= max_comments:
                break
            anchor = docx_by_id.get(review.anchor_unit_id)
            if anchor is None or anchor.unit_id not in docx_unit_ids or anchor.unit_id in used_docx_units:
                anchor = self._pick_docx_unit_for_image_page_review(
                    review=review.to_dict(),
                    docx_by_id=docx_by_id,
                    docx_unit_ids=docx_unit_ids,
                    used_docx_units=used_docx_units,
                )
            if anchor is None:
                continue
            used_docx_units.add(anchor.unit_id)
            payload = review.to_dict()
            comment_text = self._image_pdf_page_comment_text(review=payload, anchor_text=anchor.text)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_image_page_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=anchor.unit_id,
                    page_no=anchor.estimated_page_no or review.page_no,
                    old_text=anchor.text,
                    new_text=self._image_pdf_page_hint(payload),
                    action="review",
                    confidence=max(float(review.confidence or 0.0), 0.55),
                    alignment_score=0.0,
                    reason=review.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1

    def _pick_docx_unit_for_image_page_review(
        self,
        *,
        review: Dict[str, Any],
        docx_by_id: Dict[str, Any],
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> Any | None:
        for sample in list(review.get("docx_samples") or []):
            unit_id = str(sample.get("unit_id") or "")
            unit = docx_by_id.get(unit_id)
            if unit is not None and unit.unit_id in docx_unit_ids and unit.unit_id not in used_docx_units:
                return unit
        return None

    def _image_pdf_page_hint(self, review: Dict[str, Any]) -> str:
        samples = [
            str(item.get("text") or "").strip()
            for item in list(review.get("docx_samples") or [])[:6]
            if str(item.get("text") or "").strip()
        ]
        return " | ".join(samples)[:220]

    def _image_pdf_page_comment_text(self, *, review: Dict[str, Any], anchor_text: str) -> str:
        counts = dict(review.get("coverage_status_counts") or {})
        count_text = "；".join(f"{key}={value}" for key, value in counts.items())
        samples = [
            str(item.get("text") or "").strip()
            for item in list(review.get("docx_samples") or [])[:6]
            if str(item.get("text") or "").strip()
        ]
        lines = [
            "WPS PDF转DOCX审查：图片型PDF整页审查提示",
            f"页码：{review.get('page_no') or ''}",
            f"页类型：{review.get('page_kind') or ''}；风险：{review.get('risk_level') or ''}；结论：{review.get('verdict') or ''}",
            f"批注位置：{anchor_text or '（空）'}",
            f"OCR质量：{review.get('ocr_quality') or 'unknown'} / {review.get('ocr_confidence') or 0}；OCR字数：{review.get('ocr_text_chars') or 0}；OCR行数：{review.get('ocr_line_count') or 0}",
            f"DOCX内容单元：{review.get('docx_unit_count') or 0}；未可靠覆盖：{review.get('unresolved_count') or 0}",
        ]
        if count_text:
            lines.append(f"未覆盖分布：{count_text}")
        if review.get("fragment_anomaly_count"):
            lines.append(f"碎片/重复异常：{review.get('fragment_anomaly_count')}")
        if samples:
            lines.append(f"DOCX样例：{' | '.join(samples)}")
        excerpt = str(review.get("pdf_ocr_excerpt") or "").strip()
        if excerpt:
            lines.append(f"PDF页级OCR摘要：{excerpt[:180]}")
        if review.get("reason"):
            lines.append(f"原因：{review.get('reason')}")
        if review.get("next_route"):
            lines.append(f"后续：{review.get('next_route')}")
        lines.append("说明：图片型/扫描型 PDF 是本审查的高风险主线；该页不能按普通文字 PDF 放行，需要对照原图整页核查正文、表格、数字和零散文字。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _append_content_coverage_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_unit_ids: set[str],
        used_docx_units: set[str],
        backfill_by_coverage_id: Dict[str, Dict[str, Any]],
    ) -> None:
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_FULL_CONTENT_COVERAGE_ENABLED", True)):
            return
        max_comments = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_FULL_CONTENT_MAX_COMMENTS", 36) or 36))
        if max_comments <= 0:
            return
        priority = {
            "diff_candidate": 0,
            "uncovered_docx_content": 1,
            "needs_pdf_ocr": 2,
            "mapping_uncertain": 3,
            "table_pending": 4,
            "low_priority_uncovered": 8,
        }
        per_page_count: Dict[int, int] = {}
        appended = 0
        commented_pdf_review_ids: set[str] = set()
        emitted_pdf_backfill_texts: set[str] = set()
        docx_by_page = self._docx_comment_anchors_by_page(preflight_result=preflight_result, used_docx_units=used_docx_units)
        specialist_suppressed_docx_units = {
            item.wps_unit_id
            for item in preflight_result.specialist_review_results
            if item.task_type == "image_page_specialist_review"
            and (
                (item.decision == "confirmed_error" and item.comment_policy == "comment_if_exact_replacement")
                or (item.decision == "suspected_error" and item.comment_policy == "report_only_until_confirmed")
            )
            and item.wps_unit_id
        }
        specialist_suppressed_replacements = {
            (int(item.page_no or 0), self._compact_for_match(item.new_text))
            for item in preflight_result.specialist_review_results
            if item.task_type == "image_page_specialist_review"
            and (
                (item.decision == "confirmed_error" and item.comment_policy == "comment_if_exact_replacement")
                or (item.decision == "suspected_error" and item.comment_policy == "report_only_until_confirmed")
            )
            and int(item.page_no or 0) > 0
            and self._compact_for_match(item.new_text)
        }
        appended += self._append_suspicious_table_value_comments(
            corrections,
            preflight_result=preflight_result,
            used_docx_units=used_docx_units,
            per_page_count=per_page_count,
            max_to_append=min(8, max_comments - appended),
        )
        if appended >= max_comments:
            return
        appended += self._append_suspicious_table_text_comments(
            corrections,
            preflight_result=preflight_result,
            used_docx_units=used_docx_units,
            per_page_count=per_page_count,
            max_to_append=min(8, max_comments - appended),
        )
        if appended >= max_comments:
            return
        appended += self._append_suspicious_docx_text_comments(
            corrections,
            preflight_result=preflight_result,
            used_docx_units=used_docx_units,
            specialist_suppressed_units=specialist_suppressed_docx_units,
            specialist_suppressed_replacements=specialist_suppressed_replacements,
            per_page_count=per_page_count,
            max_to_append=min(16, max_comments - appended),
        )
        if appended >= max_comments:
            return
        backfilled_pdf_pages = {
            int(item.page_no or 0)
            for item in preflight_result.content_coverage_backfills
            if item.side == "pdf" and item.available and int(item.page_no or 0) > 0
        }
        pdf_backfill_max = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_FULL_CONTENT_BACKFILL_MAX_COMMENTS", 12) or 12))
        if pdf_backfill_max > 0:
            appended += self._append_pdf_coverage_comment_batch(
                corrections,
                preflight_result=preflight_result,
                docx_by_page=docx_by_page,
                used_docx_units=used_docx_units,
                per_page_count=per_page_count,
                backfill_by_coverage_id=backfill_by_coverage_id,
                commented_pdf_review_ids=commented_pdf_review_ids,
                emitted_pdf_backfill_texts=emitted_pdf_backfill_texts,
                only_available_backfill=True,
                max_to_append=min(pdf_backfill_max, max_comments - appended),
            )
        if appended >= max_comments:
            return
        if bool(getattr(settings, "PDF_WORD_AUDIT_V4_TABLE_GAP_GROUP_COMMENTS_ENABLED", False)):
            appended += self._append_table_gap_group_comments(
                corrections,
                preflight_result=preflight_result,
                used_docx_units=used_docx_units,
                per_page_count=per_page_count,
                max_to_append=min(6, max_comments - appended),
            )
        if appended >= max_comments:
            return
        page_cluster_max = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_COVERAGE_MAX_COMMENTS", 8) or 8))
        if page_cluster_max > 0 and bool(getattr(settings, "PDF_WORD_AUDIT_V4_PAGE_COVERAGE_GROUP_COMMENTS_ENABLED", False)):
            appended += self._append_page_coverage_group_comments(
                corrections,
                preflight_result=preflight_result,
                docx_by_page=docx_by_page,
                used_docx_units=used_docx_units,
                per_page_count=per_page_count,
                max_to_append=min(page_cluster_max, max_comments - appended),
            )
        if appended >= max_comments:
            return
        candidates = [
            review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx"
            and review.decision != "covered"
            and review.unit_id in docx_unit_ids
            and review.unit_id not in used_docx_units
            and not self._is_unlocated_recall_coverage_review(review)
            and not (review.status == "needs_pdf_ocr" and int(review.page_no or 0) in backfilled_pdf_pages)
            and self._should_emit_docx_coverage_comment(review=review, preflight_result=preflight_result)
        ]
        candidates.sort(key=lambda item: (priority.get(item.status, 9), item.page_no or 9999, item.review_id))
        for review in candidates:
            if appended >= max_comments:
                break
            page_no = int(review.page_no or 0)
            if per_page_count.get(page_no, 0) >= 4 and review.status not in {"uncovered_docx_content", "diff_candidate"}:
                continue
            used_docx_units.add(review.unit_id)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            payload = review.to_dict()
            backfill = self._best_docx_page_backfill(review=review, preflight_result=preflight_result, backfill_by_coverage_id=backfill_by_coverage_id)
            if backfill:
                payload["coverage_backfill"] = backfill
            suggestion = self._docx_coverage_suggestion(review=review)
            if not suggestion:
                continue
            comment_text = self._content_coverage_comment_text(review=payload)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_coverage_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=review.unit_id,
                    page_no=review.page_no,
                    old_text=review.text,
                    new_text=suggestion,
                    action="review",
                    confidence=float(review.confidence or 0.0),
                    alignment_score=0.0,
                    reason=review.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1
        if appended >= max_comments:
            return

        appended += self._append_pdf_coverage_comment_batch(
            corrections,
            preflight_result=preflight_result,
            docx_by_page=docx_by_page,
            used_docx_units=used_docx_units,
            per_page_count=per_page_count,
            backfill_by_coverage_id=backfill_by_coverage_id,
            commented_pdf_review_ids=commented_pdf_review_ids,
            emitted_pdf_backfill_texts=emitted_pdf_backfill_texts,
            only_available_backfill=False,
            max_to_append=max_comments - appended,
        )

    def _append_high_risk_page_coverage_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_COMMENTS_ENABLED", True)):
            return
        max_comments = max(0, int(getattr(settings, "PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MAX_COMMENTS", 6) or 6))
        if max_comments <= 0:
            return
        appended = 0
        pages_with_exact_specialist_comments = {
            int(item.page_no or 0)
            for item in preflight_result.specialist_review_results
            if item.decision == "confirmed_error"
            and item.comment_policy == "comment_if_exact_replacement"
            and item.old_text
            and item.new_text
            and int(item.page_no or 0) > 0
        }
        reviews = sorted(
            preflight_result.high_risk_page_coverage_reviews,
            key=lambda item: (-int(item.priority or 0), int(item.page_no or 0)),
        )
        for review in reviews:
            if appended >= max_comments:
                break
            if int(review.page_no or 0) in pages_with_exact_specialist_comments:
                continue
            anchor_unit_id = str(review.comment_anchor_unit_id or "")
            if not anchor_unit_id or anchor_unit_id not in docx_unit_ids or anchor_unit_id in used_docx_units:
                continue
            used_docx_units.add(anchor_unit_id)
            hint = self._high_risk_page_coverage_hint(review=review)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_high_risk_page_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=anchor_unit_id,
                    page_no=review.page_no,
                    old_text=review.comment_anchor_text,
                    new_text="按原 PDF 全量核查本页；未确认前不直接替换。",
                    action="review",
                    confidence=0.55,
                    alignment_score=0.0,
                    reason=review.reason,
                    comment_text=self._high_risk_page_coverage_comment_text(review=review, hint=hint),
                    sensitive_low_priority=False,
                )
            )
            appended += 1

    def _high_risk_page_coverage_hint(self, *, review: Any) -> str:
        parts = [
            f"未覆盖{int(review.unresolved_count or 0)}处",
            f"DOCX{int(review.docx_unresolved_count or 0)}",
            f"PDF{int(review.pdf_unresolved_count or 0)}",
        ]
        if int(review.table_unresolved_count or 0):
            parts.append(f"表格{int(review.table_unresolved_count or 0)}")
        if int(review.mapping_uncertain_count or 0):
            parts.append(f"映射不稳{int(review.mapping_uncertain_count or 0)}")
        if int(review.visual_unresolved_count or 0):
            parts.append(f"OCR/视觉{int(review.visual_unresolved_count or 0)}")
        if int(review.backfilled_count or 0):
            parts.append(f"已OCR回填{int(review.backfilled_count or 0)}")
        text_profile = dict(getattr(review, "page_text_coverage", {}) or {})
        if text_profile:
            docx_ratio = float(text_profile.get("docx_token_coverage_ratio") or 0.0)
            pdf_ratio = float(text_profile.get("pdf_token_coverage_ratio") or 0.0)
            status = str(text_profile.get("status") or "")
            if status:
                parts.append(f"页级覆盖{docx_ratio:.0%}/{pdf_ratio:.0%}")
        return "；".join(parts)[:90]

    def _high_risk_page_coverage_comment_text(self, *, review: Any, hint: str) -> str:
        text_profile = dict(getattr(review, "page_text_coverage", {}) or {})
        lines = [
            "页级全量核查提示",
            f"第{int(review.page_no or 0)}页",
            "处理：按原 PDF 全量核查本页；未确认前不直接替换。",
            f"重点：{hint}",
            "原因：本页仍有未可靠覆盖或映射不稳内容。",
        ]
        if text_profile:
            lines.append(
                "页级文本覆盖："
                f"DOCX token {float(text_profile.get('docx_token_coverage_ratio') or 0.0):.0%}；"
                f"PDF token {float(text_profile.get('pdf_token_coverage_ratio') or 0.0):.0%}；"
                f"相似度 {float(text_profile.get('page_text_similarity') or 0.0):.0%}。"
            )
            docx_samples = [str(item) for item in list(text_profile.get("docx_gap_samples") or [])[:4] if str(item).strip()]
            pdf_samples = [str(item) for item in list(text_profile.get("pdf_gap_samples") or [])[:4] if str(item).strip()]
            if docx_samples:
                lines.append(f"DOCX缺口样例：{' | '.join(docx_samples)}")
            if pdf_samples:
                lines.append(f"PDF缺口样例：{' | '.join(pdf_samples)}")
        return "\n".join(str(line)[:120] for line in lines if str(line).strip())

    def _is_unlocated_recall_coverage_review(self, review: Any) -> bool:
        flags = set(getattr(review, "flags", []) or [])
        return bool({"diff_category=unlocated_hard_field", "diff_category=page_coverage_gap"} & flags)

    def _should_emit_docx_coverage_comment(self, *, review: Any, preflight_result: ConversionPreflightResult) -> bool:
        page_no = int(getattr(review, "page_no", 0) or 0)
        if page_no > 0 and self._page_is_image_or_scan(preflight_result=preflight_result, page_no=page_no):
            return False
        flags = set(getattr(review, "flags", []) or [])
        status = str(getattr(review, "status", "") or "")
        if status != "uncovered_docx_content" and not (status == "diff_candidate" and "diff_category=extra_content" in flags):
            return False
        text = " ".join(str(getattr(review, "text", "") or "").split())
        if len(self._compact_for_match(text)) < int(getattr(settings, "PDF_WORD_AUDIT_V4_FULL_CONTENT_MIN_TEXT_CHARS", 4) or 4):
            return False
        return True

    def _docx_coverage_suggestion(self, *, review: Any) -> str:
        text = " ".join(str(getattr(review, "text", "") or "").split())
        if not text:
            return ""
        flags = set(getattr(review, "flags", []) or [])
        status = str(getattr(review, "status", "") or "")
        if status == "uncovered_docx_content" or (status == "diff_candidate" and "diff_category=extra_content" in flags):
            return "建议删除或核对该段：PDF 未找到对应内容。"
        return ""

    def _append_pdf_coverage_comment_batch(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_by_page: Dict[int, List[Any]],
        used_docx_units: set[str],
        per_page_count: Dict[int, int],
        backfill_by_coverage_id: Dict[str, Dict[str, Any]],
        commented_pdf_review_ids: set[str],
        emitted_pdf_backfill_texts: set[str],
        only_available_backfill: bool,
        max_to_append: int,
    ) -> int:
        if max_to_append <= 0:
            return 0
        pdf_priority = {
            "diff_candidate": 0,
            "uncovered_pdf_content": 1,
            "visual_pending": 2,
            "table_pending": 3,
            "mapping_uncertain": 4,
        }
        pdf_candidates = [
            review
            for review in preflight_result.content_coverage_reviews
            if review.side == "pdf" and review.decision != "covered"
            and review.review_id not in commented_pdf_review_ids
        ]
        if only_available_backfill:
            pdf_candidates = [
                review
                for review in pdf_candidates
                if bool((backfill_by_coverage_id.get(review.review_id) or {}).get("available"))
            ]
        pdf_candidates.sort(
            key=lambda item: (
                0 if bool((backfill_by_coverage_id.get(item.review_id) or {}).get("available")) else 1,
                item.page_no or 9999,
                pdf_priority.get(item.status, 9),
                item.review_id,
            )
        )
        appended = 0
        for review in pdf_candidates:
            if appended >= max_to_append:
                break
            page_no = int(review.page_no or 0)
            anchor = self._pick_docx_anchor_for_pdf_coverage(page_no=page_no, docx_by_page=docx_by_page, used_docx_units=used_docx_units)
            if anchor is None:
                continue
            backfill = backfill_by_coverage_id.get(review.review_id)
            backfill_text = str((backfill or {}).get("extracted_text") or "").strip()
            if not self._should_emit_pdf_coverage_comment(
                anchor_text=anchor.text,
                backfill_text=backfill_text,
                review_text=review.text,
                emitted_texts=emitted_pdf_backfill_texts,
            ):
                continue
            used_docx_units.add(anchor.unit_id)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            payload = review.to_dict()
            payload["anchor_docx_text"] = anchor.text
            if backfill:
                payload["coverage_backfill"] = backfill
            comment_text = self._content_coverage_comment_text(review=payload)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_pdf_coverage_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=anchor.unit_id,
                    page_no=anchor.estimated_page_no or review.page_no,
                    old_text=anchor.text,
                    new_text=backfill_text or review.text,
                    action="review",
                    confidence=float(review.confidence or 0.0),
                    alignment_score=0.0,
                    reason=review.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            emitted_pdf_backfill_texts.add(self._compact_for_match(backfill_text))
            commented_pdf_review_ids.add(review.review_id)
            appended += 1
        return appended

    def _should_emit_pdf_coverage_comment(
        self,
        *,
        anchor_text: str,
        backfill_text: str,
        review_text: str,
        emitted_texts: set[str],
    ) -> bool:
        text = " ".join(str(backfill_text or "").split())
        if not text:
            return False
        if self._is_noisy_pdf_backfill_suggestion(text):
            return False
        compact_text = self._compact_for_match(text)
        if not compact_text or compact_text in emitted_texts:
            return False
        anchor_compact = self._compact_for_match(anchor_text)
        if len(anchor_compact) < 3 and len(compact_text) > 12:
            return False
        if len(anchor_compact) >= 3 and compact_text in anchor_compact:
            return False
        review_compact = self._compact_for_match(review_text)
        if len(anchor_compact) >= 3 and anchor_compact in compact_text:
            return len(compact_text) - len(anchor_compact) >= 6
        if len(review_compact) >= 3 and review_compact == compact_text:
            return len(compact_text) >= 8 and not anchor_compact
        return False

    def _append_suspicious_table_value_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        used_docx_units: set[str],
        per_page_count: Dict[int, int],
        max_to_append: int,
    ) -> int:
        if max_to_append <= 0:
            return 0
        candidates = [
            review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx"
            and review.status in {"table_pending", "mapping_uncertain", "diff_candidate"}
            and review.unit_id not in used_docx_units
            and self._table_value_has_confusable_letter(review.text)
        ]
        candidates.sort(key=lambda item: (item.page_no or 9999, item.review_id))
        appended = 0
        per_page_added: Dict[int, int] = {}
        for review in candidates:
            if appended >= max_to_append:
                break
            hint = self._numeric_confusable_hint(review.text)
            if not hint:
                continue
            page_no = int(review.page_no or 0)
            if self._has_existing_replacement_comment(corrections, page_no=page_no, old_text=review.text, new_text=hint):
                continue
            if per_page_added.get(page_no, 0) >= 3:
                continue
            used_docx_units.add(review.unit_id)
            per_page_added[page_no] = per_page_added.get(page_no, 0) + 1
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            comment_text = self._suspicious_table_value_comment_text(review=review)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_table_value_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=review.unit_id,
                    page_no=review.page_no,
                    old_text=review.text,
                    new_text=hint,
                    action="review",
                    confidence=max(0.62, float(review.confidence or 0.0)),
                    alignment_score=0.0,
                    reason="表格数字类单元混入疑似 OCR 混淆字母，需要优先对照原 PDF。",
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1
        return appended

    def _suspicious_table_value_comment_text(self, *, review: Any) -> str:
        hint = self._numeric_confusable_hint(review.text)
        lines = [
            "WPS PDF转DOCX审查：表格数字疑似识别错误",
            f"覆盖状态：{review.status} / {review.decision}",
            f"DOCX表格单元：{review.text}",
            f"疑点：数字类单元包含易混字母，可能是 WPS 将数字识别成字母。",
        ]
        if hint:
            lines.append(f"字符提示：{hint}")
        lines.append("说明：未自动修改正文；该批注只标记高风险数字单元，请按原 PDF 表格核查准确值。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _table_value_has_confusable_letter(self, text: str) -> bool:
        compact = "".join(str(text or "").split()).upper()
        if len(compact) < 3 or not any(ch.isdigit() for ch in compact):
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in compact):
            return False
        confusable = set("BDEGILOQSZ")
        return any(ch in confusable for ch in compact)

    def _numeric_confusable_hint(self, text: str) -> str:
        compact = " ".join(str(text or "").split())
        if not self._confusable_letters_are_fractional(compact):
            return ""
        replacements = str.maketrans({
            "O": "0",
            "o": "0",
            "D": "0",
            "B": "8",
            "b": "8",
            "I": "1",
            "i": "1",
            "L": "1",
            "l": "1",
            "S": "5",
            "s": "5",
            "Z": "2",
            "z": "2",
        })
        hinted = compact.translate(replacements)
        if re.search(r"[A-Za-z]", hinted):
            return ""
        if hinted == compact:
            return ""
        return hinted

    def _confusable_letters_are_fractional(self, text: str) -> bool:
        compact = "".join(str(text or "").split())
        if not compact or not re.search(r"[A-Za-z]", compact):
            return False
        if "." not in compact:
            return False
        integer_part, fractional_part = compact.rsplit(".", 1)
        if re.search(r"[A-Za-z]", integer_part):
            return False
        return bool(re.search(r"[A-Za-z]", fractional_part))

    def _append_suspicious_table_text_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        used_docx_units: set[str],
        per_page_count: Dict[int, int],
        max_to_append: int,
    ) -> int:
        if max_to_append <= 0:
            return 0
        docx_by_unit = {item.unit_id: item for item in preflight_result.docx_units}
        candidates = []
        grouped_candidates: Dict[Tuple[int, int, int, str], List[Any]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.side != "docx" or review.unit_id in used_docx_units:
                continue
            if review.status not in {"covered", "table_pending", "mapping_uncertain", "diff_candidate", "uncovered_docx_content"}:
                continue
            unit = docx_by_unit.get(review.unit_id)
            if unit is None or unit.container_type != "table_cell":
                continue
            hint = self._table_text_artifact_hint(review.text)
            if not hint:
                continue
            page_no = int(review.page_no or 0)
            if self._has_existing_replacement_comment(corrections, page_no=page_no, old_text=review.text, new_text=hint):
                continue
            suggestion_key = self._compact_for_match(self._clean_comment_suggestion(hint))
            group_key = (page_no, int(unit.table_index or 0), int(unit.row_index or 0), suggestion_key)
            if group_key[0] > 0 and group_key[1] > 0 and group_key[2] > 0 and group_key[3]:
                if group_key in grouped_candidates:
                    grouped_candidates[group_key][3].append((review, unit, hint))
                    continue
                grouped_candidates[group_key] = [review, unit, hint, []]
            else:
                candidates.append((review, unit, hint, []))
        candidates.extend(tuple(item) for item in grouped_candidates.values())
        candidates.sort(
            key=lambda item: (
                int(item[0].page_no or 9999),
                int(item[1].table_index or 9999),
                int(item[1].row_index or 9999),
                int(item[1].col_index or 9999),
                item[0].review_id,
            )
        )
        appended = 0
        per_page_added: Dict[int, int] = {}
        for review, unit, hint, related_items in candidates:
            if appended >= max_to_append:
                break
            page_no = int(review.page_no or 0)
            if per_page_added.get(page_no, 0) >= 3:
                continue
            used_docx_units.add(review.unit_id)
            for _related_review, related_unit, _related_hint in related_items:
                used_docx_units.add(related_unit.unit_id)
            per_page_added[page_no] = per_page_added.get(page_no, 0) + 1
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            comment_text = self._suspicious_table_text_comment_text(
                review=review,
                unit=unit,
                hint=hint,
                related_items=related_items,
            )
            suggestion = self._clean_comment_suggestion(hint) or hint
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_table_text_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=review.unit_id,
                    page_no=review.page_no,
                    old_text=review.text,
                    new_text=suggestion,
                    action="review",
                    confidence=max(0.6, float(review.confidence or 0.0)),
                    alignment_score=0.0,
                    reason="表格短文本出现图片型 PDF/WPS OCR 常见形近误识别，需要对照原 PDF。",
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1
        return appended

    def _has_existing_replacement_comment(
        self,
        corrections: Sequence[CorrectionCandidate],
        *,
        page_no: int,
        old_text: str,
        new_text: str,
    ) -> bool:
        wanted_old = self._compact_for_match(old_text)
        wanted_new = self._compact_for_match(self._clean_comment_suggestion(new_text))
        if not wanted_old or not wanted_new:
            return False
        for correction in corrections:
            if int(correction.page_no or 0) != int(page_no or 0):
                continue
            current_old = self._compact_for_match(correction.old_text)
            current_new = self._compact_for_match(self._clean_comment_suggestion(correction.new_text))
            if current_old == wanted_old and current_new == wanted_new:
                return True
        return False

    def _suspicious_table_text_comment_text(
        self,
        *,
        review: Any,
        unit: Any,
        hint: str,
        related_items: Sequence[Tuple[Any, Any, str]] = (),
    ) -> str:
        location = []
        if unit.table_index is not None:
            location.append(f"表{unit.table_index}")
        if unit.row_index is not None:
            location.append(f"行{unit.row_index}")
        if unit.col_index is not None:
            location.append(f"列{unit.col_index}")
        lines = [
            "WPS PDF转DOCX审查：表格文字疑似识别错误",
            f"覆盖状态：{review.status} / {review.decision}",
            f"位置：{' / '.join(location) if location else review.unit_id}",
            f"DOCX表格单元：{review.text}",
            f"字符提示：{hint}",
            "疑点：图片型 PDF 表格短文本出现常见 OCR 形近字/污染字符，可能不是原 PDF 可见文本。",
        ]
        related_context = {
            "related_units": [
                {
                    "unit_id": related_unit.unit_id,
                    "old_text": related_review.text,
                    "new_text": self._clean_comment_suggestion(related_hint),
                    "row_index": int(related_unit.row_index or 0) or None,
                    "col_index": int(related_unit.col_index or 0) or None,
                }
                for related_review, related_unit, related_hint in related_items
            ]
        }
        related_text = self._table_related_units_text(context=related_context)
        if related_text:
            lines.append(f"同类单元：{related_text}")
        if review.next_route:
            lines.append(f"后续：{review.next_route}")
        lines.append("说明：未自动修改正文；该批注只标记表格文字高风险单元，请按原 PDF 表头/单元格逐字确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _table_text_artifact_hint(self, text: str) -> str:
        replacement = table_text_artifact_replacement(text)
        if not replacement:
            return ""
        compact = " ".join(str(text or "").split())
        if compact[:1] in {"!", "！", "|", "丨"} and replacement == compact[1:]:
            return f"疑似存在前缀污染字符，建议核对：{replacement}"
        return f"疑似应核对为：{replacement}"

    def _append_suspicious_docx_text_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        used_docx_units: set[str],
        per_page_count: Dict[int, int],
        max_to_append: int,
        specialist_suppressed_units: Optional[set[str]] = None,
        specialist_suppressed_replacements: Optional[set[Tuple[int, str]]] = None,
    ) -> int:
        if max_to_append <= 0:
            return 0
        specialist_suppressed_units = specialist_suppressed_units or set()
        specialist_suppressed_replacements = specialist_suppressed_replacements or set()
        docx_by_unit = {item.unit_id: item for item in preflight_result.docx_units}
        context_terms = self._document_context_terms(preflight_result=preflight_result)
        candidates = []
        seen_replacements: set[Tuple[int, str]] = set()
        for review in preflight_result.content_coverage_reviews:
            if review.side != "docx":
                continue
            if review.status not in {
                "covered",
                "covered_by_page_ocr",
                "covered_by_nearby_page_ocr",
                "uncovered_docx_content",
                "mapping_uncertain",
                "needs_pdf_ocr",
                "diff_candidate",
                "table_pending",
            }:
                continue
            unit = docx_by_unit.get(review.unit_id)
            if unit is None or unit.container_type == "table_cell":
                continue
            if not self._page_is_image_or_scan(preflight_result=preflight_result, page_no=int(review.page_no or 0)):
                continue
            hint = self._docx_text_artifact_hint(review.text, context_terms=context_terms)
            if not hint:
                continue
            artifact_priority = self._docx_text_artifact_priority(text=review.text, hint=hint)
            suggestion = self._clean_comment_suggestion(hint)
            replacement_key = (int(review.page_no or 0), self._compact_for_match(suggestion))
            if replacement_key[1] and any(
                page_no == replacement_key[0]
                and self._specialist_replacement_covers_docx_artifact(
                    candidate_replacement=replacement_key[1],
                    specialist_replacement=specialist_value,
                )
                for page_no, specialist_value in specialist_suppressed_replacements
            ):
                continue
            if review.unit_id in specialist_suppressed_units:
                continue
            if review.unit_id in used_docx_units and artifact_priority > 2:
                continue
            if replacement_key[1] and replacement_key in seen_replacements:
                continue
            if replacement_key[1]:
                seen_replacements.add(replacement_key)
            candidates.append((review, unit, hint, suggestion, artifact_priority))
        candidates.sort(
            key=lambda item: (
                int(item[0].page_no or 9999),
                item[4],
                item[1].order_index,
                item[0].review_id,
            )
        )
        appended = 0
        per_page_added: Dict[int, int] = {}
        for review, unit, hint, suggestion, artifact_priority in candidates:
            if appended >= max_to_append:
                break
            page_no = int(review.page_no or 0)
            if per_page_added.get(page_no, 0) >= 4:
                continue
            used_docx_units.add(review.unit_id)
            per_page_added[page_no] = per_page_added.get(page_no, 0) + 1
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            comment_text = self._suspicious_docx_text_comment_text(review=review, unit=unit, hint=hint)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_docx_text_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=review.unit_id,
                    page_no=review.page_no,
                    old_text=review.text,
                    new_text=suggestion,
                    action="review",
                    confidence=max(0.58, float(review.confidence or 0.0)),
                    alignment_score=0.0,
                    reason="图片型 PDF 页面中的 DOCX 文本出现常见 OCR 字符污染或日期误识别。",
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1
        return appended

    def _specialist_replacement_covers_docx_artifact(
        self,
        *,
        candidate_replacement: str,
        specialist_replacement: str,
    ) -> bool:
        candidate = self._compact_for_match(candidate_replacement)
        specialist = self._compact_for_match(specialist_replacement)
        if not candidate or not specialist:
            return False
        if candidate == specialist:
            return True
        if candidate in specialist:
            return True
        if specialist in candidate:
            if self._replacement_extra_has_review_value(candidate=candidate, covered=specialist):
                return False
            extra = candidate.replace(specialist, "", 1)
            return len(extra) <= 3 or self._looks_like_duplicate_address_fragment(candidate=candidate, covered=specialist)
        return False

    def _looks_like_duplicate_address_fragment(self, *, candidate: str, covered: str) -> bool:
        if not candidate or not covered or covered not in candidate:
            return False
        if not re.search(r"\d+号楼\d{3,4}", covered):
            return False
        extra = candidate.replace(covered, "", 1)
        if not extra:
            return True
        if len(extra) > len(covered):
            return False
        suffix = re.search(r"\d+号楼\d{3,4}", covered)
        return bool(suffix and suffix.group(0) in extra or extra in covered)

    def _suspicious_docx_text_comment_text(self, *, review: Any, unit: Any, hint: str) -> str:
        lines = [
            "WPS PDF转DOCX审查：正文文字疑似识别错误",
            f"覆盖状态：{review.status} / {review.decision}",
            f"DOCX：{review.text}",
            f"字符提示：{hint}",
            "疑点：图片型 PDF 页面中出现常见 OCR 字符污染、日期误识别或错字，不应只按页级风险提示放过。",
        ]
        if review.next_route:
            lines.append(f"后续：{review.next_route}")
        lines.append("说明：未自动修改正文；该批注只标记正文/标题高风险文本，请按原 PDF 对照确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _document_context_terms(self, *, preflight_result: ConversionPreflightResult) -> List[str]:
        texts = [str(item.text or "") for item in preflight_result.docx_units]
        texts.extend(str(item.text or "") for item in preflight_result.pdf_units)
        counts = Counter(term for text in texts for term in self._extract_context_terms_from_text(text))
        seen: set[str] = set()
        terms: List[str] = []
        for term, count in counts.items():
            if self._context_term_needs_repeat(term) and count < 2:
                continue
            if term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    def _context_term_needs_repeat(self, term: str) -> bool:
        if re.search(r"\d", term):
            return False
        if term.endswith(self.CONTEXT_PLACE_SUFFIXES):
            return True
        return False

    def _extract_context_terms_from_text(self, text: str) -> List[str]:
        value = " ".join(str(text or "").split())
        compact = self._compact_for_match(value)
        terms: List[str] = []
        for match in re.finditer(r"[\u4e00-\u9fff]{2,30}(?:律师事务所|人民法院|有限公司|有限责任公司|股份有限公司|服务所|委员会|管理有限公司)", compact):
            terms.append(match.group(0))
        for line in re.split(r"[\r\n]+", value):
            candidate = "".join(str(line or "").split())
            if looks_like_document_title(candidate) or looks_like_table_title(candidate):
                terms.append(candidate)
        for match in re.finditer(r"[\u4e00-\u9fff]{2,20}\d+号楼\d{3,4}", compact):
            term = self._clean_context_address_term(match.group(0))
            if term:
                terms.append(term)
        terms.extend(term for term in self._extract_place_context_terms(compact) if self._is_plausible_place_context_term(term))
        return terms

    def _context_address_noise_tokens(self) -> Tuple[str, ...]:
        return (
            "记录",
            "金额",
            "姓名",
            "单价",
            "数量",
            "合计",
            "总计",
            "编号",
            "编码",
            "序号",
            "日期",
            "时间",
            "账号",
            "账户",
            "电话",
            "完票",
            "销账",
            "销帐",
            "未销",
            "预销",
            "清单",
            "明细",
            "统计",
            "台账",
        )

    def _clean_context_address_term(self, term: str) -> str:
        value = self._compact_for_match(term)
        if not value or self._has_context_address_noise(value):
            return ""
        match = re.fullmatch(r"(?P<place>[\u4e00-\u9fff]{2,20})(?P<building>\d+号楼\d{3,4})", value)
        if not match:
            return ""
        place = self._strip_context_place_leading_label(match.group("place"))
        if not self._is_plausible_place_context_term(place):
            return ""
        candidate = f"{place}{match.group('building')}"
        return candidate if self._is_plausible_context_address_term(candidate) else ""

    def _has_context_address_noise(self, value: str) -> bool:
        compact = self._compact_for_match(value)
        return any(noise in compact for noise in self._context_address_noise_tokens())

    def _strip_context_place_leading_label(self, value: str) -> str:
        place = self._compact_for_match(value)
        for label in (
            "证明",
            "地址",
            "住址",
            "住所",
            "坐落于",
            "坐落",
            "位于",
            "位於",
            "房产位于",
            "房屋位于",
            "涉案房产",
            "涉案房屋",
        ):
            if place.startswith(label) and len(place) > len(label) + 2:
                place = place[len(label) :]
        return place

    def _is_plausible_place_context_term(self, term: str) -> bool:
        value = self._strip_context_place_leading_label(term)
        if not value or self._has_context_address_noise(value):
            return False
        if not re.fullmatch(r"[\u4e00-\u9fff]{3,20}", value):
            return False
        if any(char in value for char in "账账号码费额量款税¥￥0123456789"):
            return False
        if value.endswith(("法庭", "家庭")):
            return False
        for suffix, min_stem in self.CONTEXT_PLACE_SUFFIX_SPECS:
            if value.endswith(suffix):
                return len(value[: -len(suffix)]) >= min_stem
        return False

    def _is_plausible_context_address_term(self, term: str) -> bool:
        value = self._compact_for_match(term)
        if not value or self._has_context_address_noise(value):
            return False
        match = re.fullmatch(r"(?P<place>[\u4e00-\u9fff]{3,20})(?P<building>\d+号楼\d{3,4})", value)
        if not match:
            return False
        return self._is_plausible_place_context_term(match.group("place"))

    def _clean_context_address_terms(self, terms: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        rows: List[str] = []
        for term in terms:
            cleaned = self._clean_context_address_term(term)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            rows.append(cleaned)
        return rows

    def _clean_context_place_terms(self, terms: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        rows: List[str] = []
        for term in terms:
            cleaned = self._strip_context_place_leading_label(term)
            if not self._is_plausible_place_context_term(cleaned) or cleaned in seen:
                continue
            seen.add(cleaned)
            rows.append(cleaned)
        return rows

    def _extract_place_context_terms(self, compact_text: str) -> List[str]:
        value = str(compact_text or "")
        if not value:
            return []
        rows: List[str] = []
        seen: set[str] = set()
        for suffix, min_stem, max_stem in self.CONTEXT_PLACE_EXTRACTION_SUFFIX_SPECS:
            suffix_len = len(suffix)
            for end in range(suffix_len, len(value) + 1):
                if value[end - suffix_len : end] != suffix:
                    continue
                for stem_len in range(min_stem, max_stem + 1):
                    start = end - suffix_len - stem_len
                    if start < 0:
                        continue
                    term = value[start:end]
                    if not re.fullmatch(r"[\u4e00-\u9fff]+", term):
                        continue
                    if term in seen:
                        continue
                    seen.add(term)
                    rows.append(term)
        return rows

    def _docx_text_artifact_hint(self, text: str, *, context_terms: Sequence[str] = ()) -> str:
        compact = " ".join(str(text or "").split())
        if not compact or len(compact) > 180:
            return ""
        if len(compact) > 140 and not self._long_docx_text_has_high_value_artifact(compact):
            return ""
        normalized = compact
        changed = False
        date_separator_fixed = self._fix_date_separator_artifacts(normalized)
        if date_separator_fixed != normalized:
            normalized = date_separator_fixed
            changed = True
        phrase_schema_fixed = self._fix_high_value_phrase_schema_artifacts(normalized, context_terms=context_terms)
        if phrase_schema_fixed != normalized:
            normalized = phrase_schema_fixed
            changed = True
        label_fixed = self._fix_known_spaced_labels(normalized, context_terms=context_terms)
        if label_fixed != normalized:
            normalized = label_fixed
            changed = True
        spaced_org_fixed = self._fix_spaced_organization_name(normalized)
        if spaced_org_fixed != normalized:
            normalized = spaced_org_fixed
            changed = True
        context_place_fixed = self._fix_context_place_near_miss(normalized, context_terms=context_terms)
        if context_place_fixed != normalized:
            normalized = context_place_fixed
            changed = True
        address_component_fixed = self._fix_address_component_artifacts(normalized)
        if address_component_fixed != normalized:
            normalized = address_component_fixed
            changed = True
        embedded_date_fixed = self._fix_embedded_spaced_dates(normalized)
        if embedded_date_fixed != normalized:
            normalized = embedded_date_fixed
            changed = True
        datetime_fixed = self._fix_datetime_digit_artifacts(normalized)
        if datetime_fixed != normalized:
            normalized = datetime_fixed
            changed = True
        chinese_date_fixed = self._fix_chinese_date_spacing_artifact(normalized)
        if chinese_date_fixed != normalized:
            normalized = chinese_date_fixed
            changed = True
        date_fixed = re.sub(r"(\d{4}年\d{1,2}月\d{1,2})B", r"\1日", normalized)
        if date_fixed != normalized:
            normalized = date_fixed
            changed = True
        missing_day_fixed = re.sub(r"(\d{4}年\d{1,2}月\d{1,2})(?=$|[。；，,、\\s])", r"\1日", normalized)
        if missing_day_fixed != normalized and re.search(r"\d{4}年\d{1,2}月\d{1,2}$", normalized):
            normalized = missing_day_fixed
            changed = True
        id_fixed = re.sub(r"(?<!\d)(\d{17}[0-9Xx])\d(?=[）)])", r"\1", normalized)
        if id_fixed != normalized:
            normalized = id_fixed
            changed = True
        year_fixed = re.sub(
            r"(出生日期)(?P<year>9\d{2})年",
            lambda match: f"{match.group(1)}1{match.group('year')}年",
            normalized,
        )
        if year_fixed != normalized:
            normalized = year_fixed
            changed = True
        spaced_fixed = self._fix_spaced_digits_artifact(normalized)
        if spaced_fixed != normalized:
            normalized = spaced_fixed
            changed = True
        high_signal_fixed = self._fix_high_signal_spaced_phrase_artifact(normalized, context_terms=context_terms)
        if high_signal_fixed != normalized:
            normalized = high_signal_fixed
            changed = True
        address_hint = self._docx_address_artifact_hint(normalized, context_terms=context_terms)
        if address_hint:
            address_value = self._clean_comment_suggestion(address_hint)
            if changed and self._docx_artifact_has_meaningful_extra(normalized, address_value):
                return f"疑似应核对为：{self._normalize_full_docx_artifact_suggestion(normalized)}"
            return address_hint
        if changed and normalized != compact:
            return f"疑似应核对为：{normalized}"
        return ""

    def _long_docx_text_has_high_value_artifact(self, text: str) -> bool:
        value = str(text or "")
        if re.search(r"\d\s+\d.*年.*月.*日", value):
            return True
        compact = normalize_text(value)
        if looks_like_document_title(compact) or looks_like_table_title(compact) or looks_like_organization_name(compact):
            return True
        if has_high_value_field_content(value):
            return True
        if self._compact_has_near_phrase_schema(value, schemas=self._generic_phrase_schemas(text=value)):
            return True
        return bool(re.search(r"(?<!\d)\d{17}[0-9Xx]\d(?=[）)])", value))

    def _docx_text_artifact_priority(self, *, text: str, hint: str) -> int:
        merged = f"{text} {hint}"
        suggestion = normalize_text(self._clean_comment_suggestion(hint))
        if re.search(r"\d\s+\d.*年.*月.*日|\d{3}年", merged):
            return 0
        if has_high_value_field_content(merged) or (suggestion and has_high_value_field_content(suggestion)):
            return 1
        if re.search(r"(?:19|20)\d{2}年\d{1,2}月\d{1,2}日", merged):
            return 1
        compact = normalize_text(merged)
        if (
            (suggestion and (looks_like_document_title(suggestion) or looks_like_organization_name(suggestion)))
            or looks_like_document_title(compact)
            or looks_like_organization_name(compact)
        ):
            return 2
        if (
            (suggestion and (looks_like_table_title(suggestion) or looks_like_table_header(suggestion)))
            or looks_like_table_title(compact)
            or looks_like_table_header(compact)
        ):
            return 2
        return 5

    def _fix_known_spaced_labels(self, text: str, *, context_terms: Sequence[str] = ()) -> str:
        return self._fix_high_value_phrase_schema_artifacts(str(text or ""), context_terms=context_terms)

    def _fix_date_separator_artifacts(self, text: str) -> str:
        value = str(text or "")
        value = re.sub(r"(?<=\d月)[/／](?=\d{1,2}日)", "", value)
        value = re.sub(r"((?:19|20)?\d{2}年\d{1,2}月\d{1,2})旧(?=$|[。；;，,、\\s])", r"\1日", value)
        return value

    def _fix_high_value_phrase_schema_artifacts(self, text: str, *, context_terms: Sequence[str] = ()) -> str:
        return self._replace_near_phrase_schema(
            str(text or ""),
            schemas=self._generic_phrase_schemas(text=text, context_terms=context_terms),
        )

    def _generic_phrase_schemas(self, *, text: str, context_terms: Sequence[str] = ()) -> List[str]:
        compact = normalize_text(text)
        schemas = {phrase for phrase in self.GENERIC_FIELD_LABEL_PHRASES if phrase}
        if compact and (looks_like_document_title(compact) or looks_like_table_title(compact) or looks_like_organization_name(compact)):
            schemas.add(compact)
        for term in context_terms:
            candidate = normalize_text(term)
            if not candidate or len(candidate) < 2 or len(candidate) > 36:
                continue
            if (
                looks_like_document_title(candidate)
                or looks_like_table_title(candidate)
                or looks_like_organization_name(candidate)
                or self._is_plausible_context_address_term(candidate)
                or self._is_plausible_place_context_term(candidate)
            ):
                schemas.add(candidate)
        return sorted(schemas, key=lambda item: (-len(item), item))

    def _replace_near_phrase_schema(self, text: str, *, schemas: Sequence[str]) -> str:
        value = str(text or "")
        compact_chars: List[str] = []
        index_map: List[int] = []
        had_space = False
        for index, char in enumerate(value):
            if char.isspace():
                had_space = True
                continue
            compact_chars.append(char)
            index_map.append(index)
        compact = "".join(compact_chars)
        if not compact:
            return value
        candidates: List[Tuple[int, int, int, int, str, int]] = []
        for schema in schemas:
            size = len(schema)
            if size <= 1 or len(compact) < size:
                continue
            for start in range(0, len(compact) - size + 1):
                candidate = compact[start : start + size]
                if not re.fullmatch(r"[\u4e00-\u9fff\d]+", candidate):
                    continue
                distance = self._same_length_distance(candidate, schema)
                if candidate == schema and not self._window_had_space(index_map=index_map, start=start, size=size):
                    continue
                if not self._phrase_schema_match_allowed(candidate=candidate, schema=schema, distance=distance, had_space=had_space):
                    continue
                original_start = index_map[start]
                original_end = index_map[start + size - 1] + 1
                candidates.append((start, start + size, original_start, original_end, schema, distance))
        selected: List[Tuple[int, int, int, int, str, int]] = []
        occupied: List[Tuple[int, int]] = []
        for candidate in sorted(set(candidates), key=lambda item: (item[5], -(item[1] - item[0]), item[0], item[4])):
            start, end = candidate[0], candidate[1]
            if any(not (end <= left or start >= right) for left, right in occupied):
                continue
            selected.append(candidate)
            occupied.append((start, end))
        for _, _, original_start, original_end, replacement, _ in sorted(selected, key=lambda item: item[2], reverse=True):
            value = value[:original_start] + replacement + value[original_end:]
        return value

    def _phrase_schema_match_allowed(self, *, candidate: str, schema: str, distance: int, had_space: bool) -> bool:
        if candidate == schema:
            return had_space
        score = similarity(candidate, schema)
        if schema in self.GENERIC_FIELD_LABEL_PHRASES:
            return distance <= 1 and score >= 0.75
        if looks_like_organization_name(schema):
            return (had_space or distance <= max(1, len(schema) // 8)) and score >= 0.72
        if looks_like_document_title(schema) or looks_like_table_title(schema):
            return (had_space or distance <= max(1, len(schema) // 6)) and score >= 0.7
        if self._is_plausible_context_address_term(schema):
            return distance <= max(1, len(schema) // 8) and score >= 0.78
        if self._is_plausible_place_context_term(schema):
            return distance == 1 and candidate[:1] == schema[:1] and candidate[-1:] == schema[-1:] and score >= 0.8
        return False

    def _compact_has_near_phrase_schema(self, text: str, *, schemas: Sequence[str]) -> bool:
        value = str(text or "")
        compact = re.sub(r"\s+", "", value)
        if not compact:
            return False
        had_space = compact != value
        for schema in schemas:
            if schema in compact:
                return True
            size = len(schema)
            if len(compact) < size:
                continue
            for start in range(0, len(compact) - size + 1):
                candidate = compact[start : start + size]
                distance = self._same_length_distance(candidate, schema)
                if self._phrase_schema_match_allowed(candidate=candidate, schema=schema, distance=distance, had_space=had_space):
                    return True
        return False

    def _same_length_distance(self, left: str, right: str) -> int:
        if len(left) != len(right):
            return max(len(left), len(right))
        return sum(1 for left_char, right_char in zip(left, right) if left_char != right_char)

    def _window_had_space(self, *, index_map: Sequence[int], start: int, size: int) -> bool:
        if size <= 1:
            return False
        original_span = index_map[start + size - 1] - index_map[start] + 1
        return original_span > size

    def _fix_spaced_organization_name(self, text: str) -> str:
        value = str(text or "")
        if not re.search(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", value):
            return value
        compact = re.sub(r"\s+", "", value)
        if any(compact.endswith(suffix) for suffix in ORGANIZATION_SUFFIXES):
            return compact
        return value

    def _fix_context_place_near_miss(self, text: str, *, context_terms: Sequence[str]) -> str:
        value = str(text or "")
        if not value:
            return value
        terms = [
            term
            for term in self._clean_context_place_terms(context_terms)
            if 3 <= len(term) <= 8 and term.endswith(self.CONTEXT_PLACE_SUFFIXES) and re.fullmatch(r"[\u4e00-\u9fff]+", term)
        ]
        if not terms:
            return value
        term_set = set(terms)
        compact_chars: List[str] = []
        index_map: List[int] = []
        for index, char in enumerate(value):
            if char.isspace():
                continue
            compact_chars.append(char)
            index_map.append(index)
        compact_value = "".join(compact_chars)
        if not compact_value:
            return value
        candidates: List[tuple[int, int, int, int, str]] = []
        for term in terms:
            size = len(term)
            for start in range(0, max(0, len(compact_value) - size + 1)):
                candidate = compact_value[start : start + size]
                if not re.fullmatch(r"[\u4e00-\u9fff]+", candidate):
                    continue
                if candidate in term_set or candidate == term:
                    continue
                if not self._place_candidate_has_left_boundary(compact_value, start):
                    continue
                same_boundary = candidate[0] == term[0] and candidate[-1] == term[-1]
                same_strong_suffix = any(
                    term.endswith(suffix) and candidate.endswith(suffix)
                    for suffix in self.CONTEXT_STRONG_PLACE_SUFFIXES
                )
                if not (same_boundary or same_strong_suffix):
                    continue
                diff_count = sum(1 for left, right in zip(candidate, term) if left != right)
                if diff_count == 1:
                    end = start + size
                    candidates.append((start, end, index_map[start], index_map[end - 1] + 1, term))
        selected: List[tuple[int, int, int, int, str]] = []
        occupied: List[tuple[int, int]] = []
        for candidate in sorted(set(candidates), key=lambda item: (-(item[1] - item[0]), item[0], item[4])):
            start, end = candidate[0], candidate[1]
            if any(not (end <= left or start >= right) for left, right in occupied):
                continue
            selected.append(candidate)
            occupied.append((start, end))
        for _, _, original_start, original_end, replacement in sorted(selected, key=lambda item: item[2], reverse=True):
            value = value[:original_start] + replacement + value[original_end:]
        return value

    def _place_candidate_has_left_boundary(self, text: str, start: int) -> bool:
        if start <= 0:
            return True
        prefix = text[:start]
        previous = prefix[-1]
        if not re.fullmatch(r"[\u4e00-\u9fff]", previous):
            return True
        label_boundaries = (
            "位于",
            "住址",
            "地址",
            "住所",
            "坐落",
            "坐落于",
            "户籍地",
            "所在地",
            "通讯地址",
            "联系地址",
            "于",
            "至",
            "到",
            "在",
        )
        if prefix.endswith(label_boundaries):
            return True
        return previous in {"省", "市", "区", "县", "镇", "乡", "村", "街", "路", "道", "处", "园", "庭", "寓", "场", "号"}

    def _fix_address_component_artifacts(self, text: str) -> str:
        value = str(text or "")
        value = re.sub(r"(?<=\d)[弹蝉]元", "单元", value)
        value = re.sub(r"(?<=\d)木楼", "楼", value)
        value = re.sub(r"(?<=号楼)[栋幢]\s*(?=\d+\s*(?:单元|楼|室|号))", "", value)
        value = re.sub(r"(?<=\d)[栋幢]\s*(?=\d+\s*(?:单元|楼|室|号))", "号楼", value)
        return value

    def _fix_embedded_spaced_dates(self, text: str) -> str:
        value = str(text or "")

        def replace(match: re.Match[str]) -> str:
            year = re.sub(r"\s+", "", match.group("year"))
            month = re.sub(r"\s+", "", match.group("month"))
            day = re.sub(r"\s+", "", match.group("day"))
            if len(year) != 4 or not (1 <= len(month) <= 2) or not (1 <= len(day) <= 2):
                return match.group(0)
            try:
                month_num = int(month)
                day_num = int(day)
            except ValueError:
                return match.group(0)
            if not (1 <= month_num <= 12 and 1 <= day_num <= 31):
                return match.group(0)
            return f"{year}年{month}月{day}日"

        return re.sub(
            r"(?P<year>(?:\d\s*){4})年\s*(?P<month>(?:\d\s*){1,4})月\s*(?P<day>(?:\d\s*){1,4})日",
            replace,
            value,
        )

    def _fix_chinese_date_spacing_artifact(self, text: str) -> str:
        value = str(text or "")
        chinese_number = "零〇○一二三四五六七八九十廿两"

        def replace(match: re.Match[str]) -> str:
            year = re.sub(r"\s+", "", match.group("year"))
            month = re.sub(r"\s+", "", match.group("month"))
            day = re.sub(r"\s+", "", match.group("day"))
            return f"{year}年{month}月{day}日"

        return re.sub(
            rf"(?P<year>[零〇○一二三四五六七八九]\s*[零〇○一二三四五六七八九]\s*[零〇○一二三四五六七八九]\s*[零〇○一二三四五六七八九])年\s*(?P<month>[{chinese_number}]\s*[{chinese_number}]?\s*[{chinese_number}]?)\s*月\s*(?P<day>[{chinese_number}]\s*[{chinese_number}]?\s*[{chinese_number}]?\s*[{chinese_number}]?)\s*日",
            replace,
            value,
        )

    def _fix_datetime_digit_artifacts(self, text: str) -> str:
        value = str(text or "")

        def replace(match: re.Match[str]) -> str:
            year = re.sub(r"\s+", "", match.group("year"))
            month = re.sub(r"\s+", "", match.group("month"))
            day = re.sub(r"\s+", "", match.group("day"))
            hour = re.sub(r"\s+", "", match.group("hour"))
            minute = re.sub(r"\s+", "", match.group("minute"))
            try:
                year_num = int(year)
                month_num = int(month)
                day_num = int(day)
                hour_num = int(hour)
                minute_num = int(minute)
            except ValueError:
                return match.group(0)
            if not (1900 <= year_num <= 2099 and 1 <= month_num <= 12 and 1 <= day_num <= 31 and 0 <= hour_num <= 23 and 0 <= minute_num <= 59):
                return match.group(0)
            return f"{year}年{month}月{day}日{hour}时{minute}分"

        value = re.sub(
            r"(?P<year>(?:\d\s*){4})年\s*(?P<month>(?:\d\s*){1,2})月\s*(?P<day>(?:\d\s*){1,2})[1lI]\s+(?P<hour>(?:\d\s*){1,2})时\s*(?P<minute>(?:\d\s*){1,2})分",
            replace,
            value,
        )
        value = re.sub(
            r"(?P<year>(?:\d\s*){4})年\s*(?P<month>(?:\d\s*){1,2})月\s*(?P<day>(?:\d\s*){1,2})\s+日\s*(?P<hour>(?:\d\s*){1,2})时\s*(?P<minute>(?:\d\s*){1,2})分",
            replace,
            value,
        )
        return value

    def _fix_spaced_digits_artifact(self, text: str) -> str:
        value = str(text or "")
        if not re.search(r"\d\s+\d", value) or "年" not in value or "月" not in value or "日" not in value:
            return value
        match = re.fullmatch(
            r"\s*((?:\d\s*){4})年\s*((?:\d\s*){1,2})月\s*((?:\d\s*){1,2})日\s*",
            value,
        )
        if not match:
            return value
        year = re.sub(r"\s+", "", match.group(1))
        month = re.sub(r"\s+", "", match.group(2))
        day = re.sub(r"\s+", "", match.group(3))
        return f"{year}年{month}月{day}日"

    def _fix_high_signal_spaced_phrase_artifact(self, text: str, *, context_terms: Sequence[str] = ()) -> str:
        value = str(text or "")
        no_space = re.sub(r"\s+", "", value)
        if no_space == value:
            normalized = self._fix_high_value_phrase_schema_artifacts(value, context_terms=context_terms)
            if normalized != value and self._normalized_phrase_has_review_signal(normalized):
                return normalized
            return value
        normalized = self._fix_high_value_phrase_schema_artifacts(no_space, context_terms=context_terms)
        if self._normalized_phrase_has_review_signal(normalized):
            return normalized
        return value

    def _normalize_full_docx_artifact_suggestion(self, text: str) -> str:
        value = " ".join(str(text or "").split())
        compact = normalize_text(value)
        if self._normalized_phrase_has_review_signal(compact):
            return compact
        return value

    def _docx_artifact_has_meaningful_extra(self, candidate_text: str, covered_text: str) -> bool:
        candidate = self._compact_for_match(candidate_text)
        covered = self._compact_for_match(covered_text)
        if not candidate or not covered or covered not in candidate or candidate == covered:
            return False
        return self._replacement_extra_has_review_value(candidate=candidate, covered=covered)

    def _replacement_extra_has_review_value(self, *, candidate: str, covered: str) -> bool:
        extra = candidate.replace(covered, "", 1)
        if not extra:
            return False
        if self._normalized_phrase_has_review_signal(extra):
            return True
        if re.search(r"\d{4}年\d{1,2}月\d{1,2}日", extra):
            return True
        if re.search(r"(?<!\d)\d{17}[0-9x](?!\d)", extra):
            return True
        return False

    def _normalized_phrase_has_review_signal(self, text: str) -> bool:
        compact = normalize_text(text)
        if not compact:
            return False
        return bool(
            has_high_value_field_content(compact)
            or looks_like_document_title(compact)
            or looks_like_table_title(compact)
            or looks_like_organization_name(compact)
            or looks_like_table_header(compact)
        )

    def _docx_address_artifact_hint(self, text: str, *, context_terms: Sequence[str] = ()) -> str:
        value = str(text or "")
        no_space = re.sub(r"\s+", "", value)
        building_match = re.search(r"[\u4e00-\u9fff]{2,20}\d+号楼\d{3,4}", no_space)
        has_spaced_address = no_space != value and bool(building_match)
        context_addresses = self._clean_context_address_terms(context_terms)
        has_context_address_pollution = self._has_context_address_pollution(no_space, context_addresses)
        if not has_context_address_pollution and not has_spaced_address:
            return ""
        for term in sorted(context_addresses, key=len, reverse=True):
            suffix_match = re.search(r"\d+号楼\d{3,4}", term)
            if suffix_match and suffix_match.group(0) in no_space:
                return f"疑似应核对为：{term}"
        if building_match:
            candidate = building_match.group(0)
            if self._is_plausible_context_address_term(candidate):
                return f"疑似应核对为：{candidate}"
        return ""

    def _has_context_address_pollution(self, text: str, context_addresses: Sequence[str]) -> bool:
        compact = self._compact_for_match(text)
        if not compact or not context_addresses:
            return False
        for term in context_addresses:
            cleaned = self._compact_for_match(term)
            if not cleaned or cleaned == compact:
                continue
            suffix_match = re.search(r"\d+号楼\d{3,4}", cleaned)
            if not suffix_match or suffix_match.group(0) not in compact:
                continue
            if compact in cleaned or cleaned in compact:
                return compact != cleaned
            if re.search(r"[A-Za-z+]{2,}|[^\u4e00-\u9fff0-9号楼栋幢单元室室座区县市镇乡村路街道园小区]", compact):
                return True
            compact_place = compact[: compact.find(suffix_match.group(0))]
            cleaned_place = cleaned[: cleaned.find(suffix_match.group(0))]
            if compact_place and cleaned_place and self._edit_similarity(compact_place, cleaned_place) < 0.55:
                return True
        return False

    def _edit_similarity(self, left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        max_len = max(len(left), len(right), 1)
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_char in enumerate(right, start=1):
                insert_cost = current[right_index - 1] + 1
                delete_cost = previous[right_index] + 1
                replace_cost = previous[right_index - 1] + (0 if left_char == right_char else 1)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        distance = previous[-1]
        return max(0.0, 1.0 - distance / max_len)

    def _page_is_image_or_scan(self, *, preflight_result: ConversionPreflightResult, page_no: int) -> bool:
        if page_no <= 0:
            return False
        profile = preflight_result.page_profiles.get(str(page_no)) or {}
        labels = set(profile.get("labels") or [])
        primary_route = str(profile.get("primary_route") or "")
        if primary_route == "native_text_compare":
            return False
        if primary_route in {"image_text_compare", "image_table_cell_compare", "image_form_field_compare", "mixed_region_compare"}:
            return True
        return bool(
            labels & {"scan_like", "image_text_heavy", "table_heavy"}
            or profile.get("needs_ocr")
            or not profile.get("native_text_reliable", True)
        )

    def _append_table_gap_group_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        used_docx_units: set[str],
        per_page_count: Dict[int, int],
        max_to_append: int,
    ) -> int:
        if max_to_append <= 0:
            return 0
        by_page: Dict[int, List[Any]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.side != "docx" or review.status != "table_pending" or review.decision == "covered":
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0 or review.unit_id in used_docx_units:
                continue
            by_page.setdefault(page_no, []).append(review)
        page_rows = sorted(by_page.items(), key=lambda item: (-len(item[1]), item[0]))
        appended = 0
        for page_no, rows in page_rows:
            if appended >= max_to_append:
                break
            if len(rows) < 12:
                continue
            anchor = self._table_gap_anchor(rows=rows, used_docx_units=used_docx_units)
            if anchor is None:
                continue
            used_docx_units.add(anchor.unit_id)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            comment_text = self._table_gap_group_comment_text(page_no=page_no, rows=rows)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_table_gap_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=anchor.unit_id,
                    page_no=anchor.page_no,
                    old_text=anchor.text,
                    new_text=self._table_gap_samples_text(rows=rows, limit=10),
                    action="review",
                    confidence=0.54,
                    alignment_score=0.0,
                    reason=f"第 {page_no} 页存在 {len(rows)} 个未被 PDF 页级 OCR 保守覆盖的表格单元。",
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1
        return appended

    def _table_gap_anchor(self, *, rows: Sequence[Any], used_docx_units: set[str]) -> Any | None:
        candidates = [item for item in rows if item.unit_id not in used_docx_units and str(item.text or "").strip()]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (0 if self._table_gap_is_high_signal(str(item.text or "")) else 1, item.review_id))
        return candidates[0]

    def _table_gap_group_comment_text(self, *, page_no: int, rows: Sequence[Any]) -> str:
        high_signal = [item for item in rows if self._table_gap_is_high_signal(str(item.text or ""))]
        short_count = sum(1 for item in rows if len("".join(str(item.text or "").split())) <= 3)
        digit_count = sum(1 for item in rows if any(ch.isdigit() for ch in str(item.text or "")))
        samples = self._table_gap_samples(rows=rows, limit=14)
        lines = [
            "WPS PDF转DOCX审查：表格页集中复核提示",
            f"页码：{page_no}",
            f"未覆盖表格单元：{len(rows)}；含数字单元：{digit_count}；短碎片单元：{short_count}",
            f"高信号样例：{' | '.join(samples) if samples else '（无）'}",
        ]
        if high_signal:
            lines.append(f"优先核查：{self._table_gap_samples_text(rows=high_signal, limit=8)}")
        lines.append("原因：这些 DOCX 表格单元未在同页或邻近页 PDF 页级 OCR 中保守命中，可能是 WPS 误识别、表格行列错位、OCR 噪声或页映射偏移。")
        lines.append("说明：这是表格页聚合复核提示，不是确认错误；请优先对照原 PDF 表格核查样例值和同一行/列。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _table_gap_samples(self, *, rows: Sequence[Any], limit: int) -> List[str]:
        unique: List[str] = []
        seen: set[str] = set()
        ordered = sorted(rows, key=lambda item: (0 if self._table_gap_is_high_signal(str(item.text or "")) else 1, item.review_id))
        for item in ordered:
            text = " ".join(str(item.text or "").split())
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(text[:40])
            if len(unique) >= limit:
                break
        return unique

    def _table_gap_samples_text(self, *, rows: Sequence[Any], limit: int) -> str:
        return " | ".join(self._table_gap_samples(rows=rows, limit=limit))

    def _table_gap_is_high_signal(self, text: str) -> bool:
        compact = "".join(str(text or "").split())
        if any(ch.isdigit() for ch in compact):
            return len(compact) >= 3
        return len(compact) >= 4

    def _append_page_coverage_group_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        docx_by_page: Dict[int, List[Any]],
        used_docx_units: set[str],
        per_page_count: Dict[int, int],
        max_to_append: int,
    ) -> int:
        if max_to_append <= 0:
            return 0
        by_page: Dict[int, List[Any]] = {}
        for review in preflight_result.content_coverage_reviews:
            if review.decision == "covered":
                continue
            page_no = int(review.page_no or 0)
            if page_no <= 0:
                continue
            by_page.setdefault(page_no, []).append(review)
        page_rows = [
            (page_no, rows)
            for page_no, rows in by_page.items()
            if self._page_coverage_group_is_actionable(page_no=page_no, rows=rows)
        ]
        page_rows.sort(key=lambda item: (-self._page_coverage_group_score(rows=item[1]), item[0]))
        appended = 0
        for page_no, rows in page_rows:
            if appended >= max_to_append:
                break
            anchor = self._pick_docx_anchor_for_pdf_coverage(
                page_no=page_no,
                docx_by_page=docx_by_page,
                used_docx_units=used_docx_units,
            )
            if anchor is None:
                continue
            used_docx_units.add(anchor.unit_id)
            per_page_count[page_no] = per_page_count.get(page_no, 0) + 1
            comment_text = self._page_coverage_group_comment_text(page_no=page_no, rows=rows, anchor_text=anchor.text)
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_page_coverage_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=anchor.unit_id,
                    page_no=anchor.estimated_page_no or page_no,
                    old_text=anchor.text,
                    new_text=self._page_coverage_group_hint(rows=rows),
                    action="review",
                    confidence=0.53,
                    alignment_score=0.0,
                    reason=f"第 {page_no} 页存在 {len(rows)} 个未被可靠覆盖的内容单元，需要整页复核。",
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1
        return appended

    def _page_coverage_group_is_actionable(self, *, page_no: int, rows: Sequence[Any]) -> bool:
        if len(rows) >= 25:
            return True
        status_counts = self._page_coverage_status_counts(rows=rows)
        visual = status_counts.get("pdf/visual_pending", 0) + status_counts.get("docx/needs_pdf_ocr", 0)
        table = status_counts.get("docx/table_pending", 0) + status_counts.get("pdf/table_pending", 0)
        mapping = status_counts.get("docx/mapping_uncertain", 0) + status_counts.get("pdf/mapping_uncertain", 0)
        return visual >= 10 or table >= 20 or mapping >= 12

    def _page_coverage_group_score(self, *, rows: Sequence[Any]) -> int:
        status_counts = self._page_coverage_status_counts(rows=rows)
        score = len(rows)
        score += 3 * (status_counts.get("docx/table_pending", 0) + status_counts.get("pdf/table_pending", 0))
        score += 2 * (status_counts.get("docx/mapping_uncertain", 0) + status_counts.get("pdf/mapping_uncertain", 0))
        score += 2 * (status_counts.get("pdf/visual_pending", 0) + status_counts.get("docx/needs_pdf_ocr", 0))
        score += 2 * sum(1 for item in rows if item.status == "diff_candidate")
        return score

    def _page_coverage_status_counts(self, *, rows: Sequence[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in rows:
            key = f"{item.side}/{item.status}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _page_coverage_group_comment_text(self, *, page_no: int, rows: Sequence[Any], anchor_text: str) -> str:
        counts = self._page_coverage_status_counts(rows=rows)
        docx_rows = [item for item in rows if item.side == "docx"]
        pdf_rows = [item for item in rows if item.side == "pdf"]
        docx_samples = self._page_coverage_samples(rows=docx_rows, limit=8)
        pdf_samples = self._page_coverage_samples(rows=pdf_rows, limit=6)
        count_text = "；".join(f"{key}={value}" for key, value in counts.items())
        lines = [
            "WPS PDF转DOCX审查：页级全量核查提示",
            f"页码：{page_no}",
            f"批注位置：{anchor_text or '（空）'}",
            f"未可靠覆盖内容单元：{len(rows)}；状态分布：{count_text}",
        ]
        if docx_samples:
            lines.append(f"DOCX未覆盖样例：{' | '.join(docx_samples)}")
        if pdf_samples:
            lines.append(f"PDF未覆盖样例：{' | '.join(pdf_samples)}")
        lines.append("原因：该页存在大量普通内容、表格单元、视觉区域或映射不确定内容尚未被 PDF-DOCX 证据链可靠覆盖，不能只依赖少量字段候选判断转换质量。")
        lines.append("后续：needs_full_page_review")
        lines.append("说明：这是整页级召回提示，不是确认错误；用于提示该页需要按原 PDF 全量逐项核查，避免前序候选未定位导致漏检。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _page_coverage_group_hint(self, *, rows: Sequence[Any]) -> str:
        samples = self._page_coverage_samples(rows=rows, limit=8)
        return " | ".join(samples)[:220]

    def _page_coverage_samples(self, *, rows: Sequence[Any], limit: int) -> List[str]:
        priority = {
            "diff_candidate": 0,
            "mapping_uncertain": 1,
            "table_pending": 2,
            "visual_pending": 3,
            "uncovered_docx_content": 4,
            "uncovered_pdf_content": 5,
        }
        ordered = sorted(rows, key=lambda item: (priority.get(item.status, 9), item.review_id))
        samples: List[str] = []
        seen: set[str] = set()
        for item in ordered:
            text = " ".join(str(item.text or "").split())
            if not text:
                text = str(item.unit_id or "")
            compact = self._compact_for_match(text)
            if not compact or compact in seen:
                continue
            seen.add(compact)
            samples.append(text[:48])
            if len(samples) >= limit:
                break
        return samples

    def _best_docx_page_backfill(
        self,
        *,
        review: Any,
        preflight_result: ConversionPreflightResult,
        backfill_by_coverage_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if review.status not in {"needs_pdf_ocr", "table_pending", "mapping_uncertain"}:
            return {}
        if not review.page_no:
            return {}
        own = backfill_by_coverage_id.get(review.review_id)
        if own:
            return own
        page_no = int(review.page_no)
        candidates = [
            item.to_dict()
            for item in preflight_result.content_coverage_backfills
            if int(item.page_no or 0) == page_no and item.available
        ]
        if not candidates:
            return {}
        candidates.sort(key=lambda item: (0 if item.get("method") == "qwen_vl" else 1, -float(item.get("confidence") or 0.0), item.get("backfill_id") or ""))
        return candidates[0]

    def _docx_comment_anchors_by_page(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        used_docx_units: set[str],
    ) -> Dict[int, List[Any]]:
        rows: Dict[int, List[Any]] = {}
        for unit in preflight_result.docx_units:
            if unit.unit_id in used_docx_units:
                continue
            if not str(unit.text or "").strip():
                continue
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0:
                continue
            rows.setdefault(page_no, []).append(unit)
        for page_no, units in rows.items():
            rows[page_no] = sorted(
                units,
                key=lambda item: (
                    1 if item.container_type in {"header", "footer", "footnote", "endnote"} else 0,
                    item.order_index,
                ),
            )
        return rows

    def _pick_docx_anchor_for_pdf_coverage(
        self,
        *,
        page_no: int,
        docx_by_page: Dict[int, List[Any]],
        used_docx_units: set[str],
    ) -> Any | None:
        for unit in docx_by_page.get(page_no, []):
            if unit.unit_id not in used_docx_units:
                return unit
        return None

    def _append_visual_pending_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        diffs_by_id: Dict[str, Any],
        focused_by_id: Dict[str, Any],
        qwen_vl_by_id: Dict[str, Any],
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_VL_ENABLED", True)):
            return
        appended = 0
        for review in preflight_result.focused_reviews:
            if appended >= 3:
                break
            if review.next_route != "needs_qwen_vl" or review.diff_id in qwen_vl_by_id:
                continue
            if not review.crop_path:
                continue
            diff = diffs_by_id.get(review.diff_id)
            if diff is None or not diff.docx_unit_id or diff.docx_unit_id not in docx_unit_ids:
                continue
            if diff.docx_unit_id in used_docx_units:
                continue
            used_docx_units.add(diff.docx_unit_id)
            focused = focused_by_id.get(review.diff_id)
            suggested = diff.pdf_text or diff.pdf_value or ""
            if not self._has_concrete_review_suggestion(suggested):
                continue
            fallback_gate = {
                "decision": "manual_review",
                "verdict": "not_model_reviewed",
                "confidence": review.confidence,
                "reason": self._visual_pending_reason(review=review, diff=diff),
                "preferred_text": suggested,
                "next_route": "needs_human_visual_review",
                "gate_label": "视觉门槛",
                "attempted": False,
            }
            comment_text = self._comment_text(diff=diff.to_dict(), focused=focused.to_dict() if focused else {}, gate=fallback_gate, comment_kind="视觉复核提示")
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_visual_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=diff.docx_unit_id,
                    page_no=diff.docx_estimated_page_no or diff.pdf_page_no,
                    old_text=diff.docx_text,
                    new_text=suggested,
                    action="review",
                    confidence=max(float(diff.confidence or 0.0), float(review.confidence or 0.0)),
                    alignment_score=float(diff.alignment_confidence or 0.0),
                    reason=review.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1

    def _has_concrete_review_suggestion(self, value: str) -> bool:
        text = " ".join(str(value or "").split())
        if not text:
            return False
        compact = self._compact_for_match(text)
        if len(compact) < 2:
            return False
        if compact in {"pdf", "docx", "unknown", "none", "null"}:
            return False
        return True

    def _visual_pending_reason(self, *, review: Any, diff: Any) -> str:
        quality = str(getattr(review, "crop_ocr_quality", "") or "").strip().lower()
        visual_text = dict(getattr(review, "visual_text", {}) or {})
        support = str(visual_text.get("support") or "").strip()
        stable = bool(visual_text.get("stable"))
        flags = set(getattr(diff, "flags", []) or [])
        category = str(getattr(diff, "category", "") or getattr(review, "category", "") or "")
        if quality == "low" and not (stable and support == "pdf"):
            return "局部 OCR 质量低或文字被拆分污染，未调用 Qwen-VL 自动确认；该候选只作为人工对照截图复核任务。"
        if category in {"unlocated_hard_field", "page_coverage_gap"} and "recall_pdf_field_not_matched" in flags:
            return "该候选属于召回兜底，缺少可直接比较的 PDF 侧候选文本，未调用 Qwen-VL 自动确认；需要人工对照原 PDF 页面。"
        return "候选未进入 Qwen-VL 调用队列，当前只保留为人工视觉复核任务，不作为确认错误。"

    def _append_recall_guard_comments(
        self,
        corrections: List[CorrectionCandidate],
        *,
        preflight_result: ConversionPreflightResult,
        diffs_by_id: Dict[str, Any],
        focused_by_id: Dict[str, Any],
        qwen_vl_by_id: Dict[str, Any],
        docx_unit_ids: set[str],
        used_docx_units: set[str],
    ) -> None:
        if not bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_GATE_ENABLED", True)):
            return
        appended = 0
        priority = {"unlocated_hard_field": 0, "page_coverage_gap": 1}
        guard_reviews = [
            review
            for review in preflight_result.focused_reviews
            if review.status == "recall_guard" and review.decision == "needs_recall_review"
        ]
        guard_reviews.sort(key=lambda item: (priority.get(item.category, 99), item.page_no or 9999, item.diff_id))
        for review in guard_reviews:
            if appended >= 6:
                break
            diff = diffs_by_id.get(review.diff_id)
            if diff is None or not diff.docx_unit_id or diff.docx_unit_id not in docx_unit_ids:
                continue
            vl = qwen_vl_by_id.get(review.diff_id)
            if vl is not None and getattr(vl, "decision", "") == "block_candidate":
                continue
            if diff.docx_unit_id in used_docx_units:
                continue
            if not self._should_emit_recall_guard_comment(diff=diff):
                continue
            used_docx_units.add(diff.docx_unit_id)
            focused = focused_by_id.get(review.diff_id)
            comment_text = self._recall_guard_comment_text(diff=diff.to_dict(), focused=focused.to_dict() if focused else {})
            corrections.append(
                CorrectionCandidate(
                    id=f"v4_recall_{uuid.uuid4().hex[:12]}",
                    wps_unit_id=diff.docx_unit_id,
                    page_no=diff.docx_estimated_page_no or diff.pdf_page_no,
                    old_text=diff.docx_text,
                    new_text=diff.pdf_text or diff.pdf_value or "",
                    action="review",
                    confidence=max(float(diff.confidence or 0.0), float(getattr(focused, "confidence", 0.0) or 0.0)),
                    alignment_score=float(diff.alignment_confidence or 0.0),
                    reason=diff.reason,
                    comment_text=comment_text,
                    sensitive_low_priority=False,
                )
            )
            appended += 1

    def _should_emit_recall_guard_comment(self, *, diff: Any) -> bool:
        flags = set(getattr(diff, "flags", []) or [])
        category = str(getattr(diff, "category", "") or "")
        if category in {"unlocated_hard_field", "page_coverage_gap"} or "recall_pdf_field_not_matched" in flags:
            return False
        pdf_text = str(getattr(diff, "pdf_value", "") or getattr(diff, "pdf_text", "") or "").strip()
        docx_text = str(getattr(diff, "docx_value", "") or getattr(diff, "docx_text", "") or "").strip()
        if not pdf_text or not docx_text:
            return False
        if self._is_noisy_pdf_backfill_suggestion(pdf_text):
            return False
        if len(" ".join(pdf_text.split())) > 90 or len(" ".join(docx_text.split())) > 140:
            return False
        return self._compact_for_match(pdf_text) != self._compact_for_match(docx_text)

    def _comment_kind_for_gate(self, gate: Dict[str, Any]) -> str:
        if gate.get("decision") == "allow_report_candidate":
            if gate.get("verdict") == "confirmed_error":
                return "转换确认错误"
            return "疑似转换错误"
        if gate.get("next_route") == "needs_human_table_review":
            return "表格复核提示"
        if gate.get("next_route") == "needs_qwen_vl":
            return "视觉复核提示"
        if gate.get("next_route") == "needs_human_mapping_review" and gate.get("verdict") in {"coverage_gap", "model_conflict"}:
            return "映射复核提示"
        return ""

    def _comment_kind_for_vl(self, vl: Dict[str, Any]) -> str:
        if vl.get("decision") == "allow_report_candidate":
            return "视觉确认转换错误"
        if vl.get("decision") == "defer" and vl.get("next_route") in {"needs_region_ocr", "needs_human_visual_review", "needs_qwen_vl"}:
            return "视觉复核提示"
        return ""

    def _comment_text(self, *, diff: Dict[str, Any], focused: Dict[str, Any], gate: Dict[str, Any], comment_kind: str) -> str:
        lines = [
            f"WPS PDF转DOCX审查：{comment_kind}",
            f"候选：{diff.get('category') or ''} / {diff.get('diff_id') or ''}",
            f"DOCX：{diff.get('docx_text') or '（空）'}",
        ]
        pdf_text = str(diff.get("pdf_text") or "").strip()
        preferred = str(gate.get("preferred_text") or "").strip()
        if preferred:
            lines.append(f"PDF证据：{preferred}")
        elif pdf_text:
            lines.append(f"PDF证据：{pdf_text}")
        if gate:
            gate_label = str(gate.get("gate_label") or "Qwen门槛")
            verdict = str(gate.get("verdict") or "")
            decision = str(gate.get("decision") or "")
            if gate.get("attempted") is False:
                lines.append(f"{gate_label}：未调用 / {decision} / {gate.get('confidence') or 0}")
            else:
                lines.append(f"{gate_label}：{verdict} / {decision} / {gate.get('confidence') or 0}")
            if gate.get("reason"):
                lines.append(f"原因：{gate.get('reason')}")
            if gate.get("next_route"):
                lines.append(f"后续：{gate.get('next_route')}")
        if focused.get("crop_path"):
            lines.append(f"证据截图：{focused.get('crop_path')}")
        visual_text = dict(focused.get("visual_text") or {})
        if visual_text.get("support"):
            lines.append(f"视觉文本证据：{visual_text.get('support')} / {visual_text.get('reason')}")
        table_cell = dict(focused.get("table_cell") or {})
        if table_cell.get("cell_ocr_text"):
            lines.append(f"表格单元OCR：{table_cell.get('cell_ocr_text')}")
        if table_cell.get("crop_path"):
            lines.append(f"表格单元截图：{table_cell.get('crop_path')}")
        lines.append("说明：未自动修改正文，请按批注对照原 PDF/证据截图确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _vl_comment_text(self, *, diff: Dict[str, Any], focused: Dict[str, Any], vl: Dict[str, Any], comment_kind: str) -> str:
        lines = [
            f"WPS PDF转DOCX审查：{comment_kind}",
            f"候选：{diff.get('category') or ''} / {diff.get('diff_id') or ''}",
            f"DOCX：{diff.get('docx_text') or '（空）'}",
        ]
        visible = str(vl.get("visible_text") or "").strip()
        preferred = str(vl.get("preferred_text") or "").strip()
        if visible:
            lines.append(f"截图可见：{visible}")
        if preferred:
            lines.append(f"PDF证据：{preferred}")
        elif diff.get("pdf_text") or diff.get("pdf_value"):
            lines.append(f"PDF候选：{diff.get('pdf_text') or diff.get('pdf_value')}")
        lines.append(f"Qwen视觉门槛：{vl.get('verdict') or ''} / {vl.get('decision') or ''} / {vl.get('confidence') or 0}")
        if vl.get("reason"):
            lines.append(f"原因：{vl.get('reason')}")
        if vl.get("next_route"):
            lines.append(f"后续：{vl.get('next_route')}")
        if vl.get("crop_path") or focused.get("crop_path"):
            lines.append(f"证据截图：{vl.get('crop_path') or focused.get('crop_path')}")
        visual_text = dict(focused.get("visual_text") or {})
        if visual_text.get("support"):
            lines.append(f"本地OCR证据：{visual_text.get('support')} / {visual_text.get('reason')}")
        lines.append("说明：未自动修改正文，请按批注对照原 PDF/证据截图确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _content_coverage_comment_text(self, *, review: Dict[str, Any]) -> str:
        side = str(review.get("side") or "")
        backfill = dict(review.get("coverage_backfill") or {})
        backfill_text = str(backfill.get("extracted_text") or "").strip()
        if side == "pdf":
            lines = [
                "WPS PDF转DOCX审查：PDF内容覆盖提示",
                f"覆盖状态：{review.get('status') or ''} / {review.get('decision') or ''}",
                f"PDF证据：{review.get('text') or backfill_text or review.get('unit_id') or '（无可提取文本）'}",
            ]
            if review.get("anchor_docx_text"):
                lines.append(f"批注挂载位置：{review.get('anchor_docx_text')}")
        else:
            lines = [
                "WPS PDF转DOCX审查：全内容覆盖提示",
                f"覆盖状态：{review.get('status') or ''} / {review.get('decision') or ''}",
                f"DOCX：{review.get('text') or '（空）'}",
            ]
        if backfill:
            method = str(backfill.get("method") or "").strip()
            status = str(backfill.get("status") or "").strip()
            confidence = backfill.get("confidence") or 0
            if backfill_text:
                lines.append(f"局部PDF回填：{backfill_text}")
            else:
                lines.append(f"局部PDF回填：{status or '未读出稳定文本'}")
            if method or status:
                lines.append(f"回填来源：{method or 'unknown'} / {status or ''} / {confidence}")
            if backfill.get("crop_path"):
                lines.append(f"回填截图：{backfill.get('crop_path')}")
        if review.get("reason"):
            lines.append(f"原因：{review.get('reason')}")
        if review.get("next_route"):
            lines.append(f"后续：{review.get('next_route')}")
        if review.get("related_link_id"):
            lines.append(f"关联映射：{review.get('related_link_id')}")
        if review.get("related_diff_id"):
            lines.append(f"关联候选：{review.get('related_diff_id')}")
        if side == "pdf":
            lines.append("说明：这是 PDF 侧漏覆盖提示，不是确认错误；表示原 PDF 该内容未找到可靠 DOCX 覆盖，需要对照原 PDF 确认是否漏转。")
        else:
            lines.append("说明：这是全内容覆盖审查，不是确认错误；表示该段/单元尚未被 PDF 侧可靠覆盖，需要对照原 PDF 逐字确认。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _recall_guard_comment_text(self, *, diff: Dict[str, Any], focused: Dict[str, Any]) -> str:
        lines = [
            "WPS PDF转DOCX审查：漏检兜底提示",
            f"候选：{diff.get('category') or ''} / {diff.get('diff_id') or ''}",
            f"DOCX：{diff.get('docx_text') or '（空）'}",
        ]
        field_type = str(diff.get("field_type") or "").strip()
        docx_value = str(diff.get("docx_value") or "").strip()
        if field_type or docx_value:
            lines.append(f"高风险字段：{field_type or 'unknown'} / {docx_value or '（未提取）'}")
        flags = set(diff.get("flags") or [])
        pdf_text = str(diff.get("pdf_value") or "").strip()
        if not pdf_text and diff.get("category") != "unlocated_hard_field":
            pdf_text = str(diff.get("pdf_text") or "").strip()
        if pdf_text and ("recall_pdf_field_not_matched" not in flags):
            lines.append(f"PDF侧可用证据：{pdf_text}")
        elif diff.get("category") == "unlocated_hard_field":
            lines.append("PDF侧可用证据：主候选链路未覆盖到同字段，需对照原PDF或局部截图复核。")
        if focused.get("reason"):
            lines.append(f"原因：{focused.get('reason')}")
        if focused.get("next_route"):
            lines.append(f"后续：{focused.get('next_route')}")
        if focused.get("crop_path"):
            lines.append(f"证据截图：{focused.get('crop_path')}")
        lines.append("说明：这是召回兜底，不是确认错误；表示该位置未被主候选链路可靠覆盖，需要按原 PDF 复核。")
        return "\n".join(str(line)[:220] for line in lines if str(line).strip())

    def _report_truncation_summary(self, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        """Describe display-only truncation in the main JSON report.

        The full evidence lists are written to ``evidence/raw/*.json``.  The
        report keeps selected previews bounded so loading the result page stays
        usable even when the quality-first profile produces thousands of rows.
        """

        limits = {
            "conversion_diff_candidates": 500,
            "focused_candidate_reviews": 500,
            "table_reviews": 500,
            "table_grid_evidence": 500,
            "full_content_coverage_reviews": 1000,
            "full_content_backfill_reviews": 500,
            "fragment_anomaly_reviews": 200,
            "image_pdf_page_reviews": 500,
            "image_text_vl_reviews": 300,
            "page_ocr_text_evidence_reviews": 500,
            "page_text_qwen_reviews": 300,
            "page_text_coverage_profiles": 300,
            "high_risk_page_coverage_reviews": 200,
            "coverage_review_tasks": 500,
            "table_page_vl_reviews": 200,
            "table_cell_evidence_reviews": 1000,
            "qwen_vl_reviews": 500,
            "qwen_gate_reviews": 500,
            "specialist_review_tasks": 500,
            "specialist_review_results": 500,
            "review_routes": 500,
        }
        counts = {
            "conversion_diff_candidates": len(preflight_result.diff_candidates),
            "focused_candidate_reviews": len(preflight_result.focused_reviews),
            "table_reviews": len(preflight_result.table_reviews),
            "table_grid_evidence": len(preflight_result.table_grid_evidence),
            "full_content_coverage_reviews": len(preflight_result.content_coverage_reviews),
            "full_content_backfill_reviews": len(preflight_result.content_coverage_backfills),
            "fragment_anomaly_reviews": len(preflight_result.fragment_anomaly_reviews),
            "image_pdf_page_reviews": len(preflight_result.image_page_reviews),
            "image_text_vl_reviews": len(preflight_result.image_text_vl_reviews),
            "page_ocr_text_evidence_reviews": len(preflight_result.page_ocr_text_evidence_reviews),
            "page_text_qwen_reviews": len(preflight_result.page_text_qwen_reviews),
            "page_text_coverage_profiles": len(preflight_result.page_text_coverage_profiles),
            "high_risk_page_coverage_reviews": len(preflight_result.high_risk_page_coverage_reviews),
            "coverage_review_tasks": len(preflight_result.coverage_review_tasks),
            "table_page_vl_reviews": len(preflight_result.table_page_vl_reviews),
            "table_cell_evidence_reviews": len(preflight_result.table_cell_evidence_reviews),
            "qwen_vl_reviews": len(preflight_result.qwen_vl_reviews),
            "qwen_gate_reviews": len(preflight_result.qwen_gate_reviews),
            "specialist_review_tasks": len(preflight_result.specialist_review_tasks),
            "specialist_review_results": len(preflight_result.specialist_review_results),
            "review_routes": len(preflight_result.review_routes),
        }
        fields: Dict[str, Dict[str, Any]] = {}
        for name, total in counts.items():
            limit = limits[name]
            fields[name] = {
                "total_count": int(total),
                "report_preview_limit": int(limit),
                "truncated_in_main_report": int(total) > int(limit),
                "raw_payload_complete": True,
            }
        return {
            "display_only": True,
            "raw_payloads_are_complete": True,
            "raw_payload_dir": "evidence/raw",
            "fields": fields,
        }

    def _review_summary(
        self,
        preflight_result: ConversionPreflightResult,
        *,
        corrections: Sequence[CorrectionCandidate] = (),
    ) -> Dict[str, Any]:
        summary = preflight_result.summary()
        focused = summary.get("focused_review") or {}
        no_confirmed = [
            item.diff_id
            for item in preflight_result.focused_reviews
            if item.decision == "no_confirmed_error"
        ]
        needs_table = [
            item.diff_id
            for item in preflight_result.focused_reviews
            if item.next_route == "needs_table_parser"
        ]
        needs_mapping = [
            item.diff_id
            for item in preflight_result.focused_reviews
            if item.next_route == "needs_human_mapping_review"
        ]
        needs_visual = [
            item.diff_id
            for item in preflight_result.focused_reviews
            if item.next_route in {"needs_qwen_vl", "needs_region_ocr"}
        ]
        possible = [
            item.diff_id
            for item in preflight_result.focused_reviews
            if item.decision == "possible_conversion_error"
        ]
        qwen_allow = [
            item.diff_id
            for item in preflight_result.qwen_gate_reviews
            if item.decision == "allow_report_candidate"
        ]
        qwen_defer = [
            item.diff_id
            for item in preflight_result.qwen_gate_reviews
            if item.decision == "defer"
        ]
        qwen_block = [
            item.diff_id
            for item in preflight_result.qwen_gate_reviews
            if item.decision == "block_candidate"
        ]
        qwen_vl_allow = [
            item.diff_id
            for item in preflight_result.qwen_vl_reviews
            if item.decision == "allow_report_candidate"
        ]
        qwen_vl_defer = [
            item.diff_id
            for item in preflight_result.qwen_vl_reviews
            if item.decision == "defer"
        ]
        qwen_vl_block = [
            item.diff_id
            for item in preflight_result.qwen_vl_reviews
            if item.decision == "block_candidate"
        ]
        qwen_next_route_counts: Dict[str, int] = {}
        for item in preflight_result.qwen_gate_reviews:
            if item.next_route:
                qwen_next_route_counts[item.next_route] = qwen_next_route_counts.get(item.next_route, 0) + 1
        qwen_vl_next_route_counts: Dict[str, int] = {}
        for item in preflight_result.qwen_vl_reviews:
            if item.next_route:
                qwen_vl_next_route_counts[item.next_route] = qwen_vl_next_route_counts.get(item.next_route, 0) + 1
        table_ready = [
            item.diff_id
            for item in preflight_result.focused_reviews
            if item.status == "ready_for_table_gate"
        ]
        coverage_status_counts: Dict[str, int] = {}
        backfill_status_counts: Dict[str, int] = {}
        coverage_commentable = []
        for item in preflight_result.content_coverage_reviews:
            coverage_status_counts[item.status] = coverage_status_counts.get(item.status, 0) + 1
            if item.side == "docx" and item.decision != "covered":
                coverage_commentable.append(item.review_id)
        for item in preflight_result.content_coverage_backfills:
            backfill_status_counts[item.status] = backfill_status_counts.get(item.status, 0) + 1
        specialist_type_counts: Dict[str, int] = {}
        specialist_status_counts: Dict[str, int] = {}
        specialist_page_nos: set[int] = set()
        for task in preflight_result.specialist_review_tasks:
            specialist_type_counts[task.task_type] = specialist_type_counts.get(task.task_type, 0) + 1
            specialist_status_counts[task.status] = specialist_status_counts.get(task.status, 0) + 1
            if task.page_no:
                specialist_page_nos.add(int(task.page_no))
        specialist_result_decision_counts: Dict[str, int] = {}
        specialist_result_type_counts: Dict[str, int] = {}
        for result in preflight_result.specialist_review_results:
            specialist_result_decision_counts[result.decision] = specialist_result_decision_counts.get(result.decision, 0) + 1
            specialist_result_type_counts[result.task_type] = specialist_result_type_counts.get(result.task_type, 0) + 1
        return {
            "mode": "comment_only" if corrections else "report_only",
            "safe_to_generate_findings": False,
            "candidate_count": len(preflight_result.diff_candidates),
            "comment_count": len(corrections),
            "comment_correction_ids": [item.id for item in corrections[:80]],
            "focused_reviewed_count": len(preflight_result.focused_reviews),
            "post_gate_route_counts": dict(focused.get("post_gate_route_counts") or {}),
            "no_confirmed_error_count": int(focused.get("no_confirmed_error_count") or 0),
            "possible_conversion_error_ids": possible[:80],
            "qwen_allow_report_candidate_ids": qwen_allow[:80],
            "qwen_defer_ids": qwen_defer[:80],
            "qwen_block_ids": qwen_block[:80],
            "qwen_next_route_counts": dict(sorted(qwen_next_route_counts.items())),
            "qwen_vl_allow_report_candidate_ids": qwen_vl_allow[:80],
            "qwen_vl_defer_ids": qwen_vl_defer[:80],
            "qwen_vl_block_ids": qwen_vl_block[:80],
            "qwen_vl_next_route_counts": dict(sorted(qwen_vl_next_route_counts.items())),
            "full_content_coverage_status_counts": dict(sorted(coverage_status_counts.items())),
            "full_content_backfill_status_counts": dict(sorted(backfill_status_counts.items())),
            "full_content_backfilled_review_ids": [
                item.coverage_review_id
                for item in preflight_result.content_coverage_backfills
                if item.available
            ][:120],
            "full_content_unresolved_docx_review_ids": coverage_commentable[:120],
            "specialist_review_task_count": len(preflight_result.specialist_review_tasks),
            "specialist_review_page_count": len(specialist_page_nos),
            "specialist_review_type_counts": dict(sorted(specialist_type_counts.items())),
            "specialist_review_status_counts": dict(sorted(specialist_status_counts.items())),
            "specialist_review_task_ids": [item.task_id for item in preflight_result.specialist_review_tasks[:120]],
            "specialist_review_result_count": len(preflight_result.specialist_review_results),
            "specialist_review_confirmed_error_count": sum(1 for item in preflight_result.specialist_review_results if item.decision == "confirmed_error"),
            "specialist_review_deferred_count": sum(1 for item in preflight_result.specialist_review_results if item.decision != "confirmed_error"),
            "specialist_review_result_type_counts": dict(sorted(specialist_result_type_counts.items())),
            "specialist_review_result_decision_counts": dict(sorted(specialist_result_decision_counts.items())),
            "table_ready_for_qwen_ids": table_ready[:80],
            "no_confirmed_error_ids": no_confirmed[:80],
            "needs_table_parser_ids": needs_table[:80],
            "needs_mapping_review_ids": needs_mapping[:80],
            "needs_visual_review_ids": needs_visual[:80],
            "explanation": (
                "v4 当前只做转换忠实度预审和局部证据门控；"
                "no_confirmed_error 表示局部 crop OCR 更支持 DOCX 或证据冲突，不能当成 WPS 转换错误；"
                "needs_table_parser/needs_mapping_review/needs_visual_review 表示必须进入下一阶段专项复核。"
            ),
        }

    def _build_findings(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
    ) -> List[Dict[str, Any]]:
        """Expose the v4 evidence in the public finding contract.

        v4 still keeps ``corrections`` for backward compatibility and DOCX
        comment writing.  The UI/API, however, should consume the unified
        finding list so confirmed comments, model conflicts, and coverage gaps
        are all visible from the result endpoint.
        """

        findings: List[Dict[str, Any]] = []
        for correction in corrections:
            findings.append(self._finding_from_correction(correction))
        findings.extend(
            self._specialist_result_findings(
                preflight_result=preflight_result,
                corrections=corrections,
                existing_count=len(findings),
            )
        )
        findings.extend(self._model_conflict_findings(preflight_result=preflight_result, existing_count=len(findings)))
        findings.extend(
            self._coverage_gap_findings(
                preflight_result=preflight_result,
                existing_count=len(findings),
            )
        )
        return findings

    def _specialist_result_findings(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
        existing_count: int,
    ) -> List[Dict[str, Any]]:
        """Report specialist evidence even when DOCX comment budgets are capped."""

        commented_keys = {
            (
                int(correction.page_no or 0),
                str(correction.wps_unit_id or ""),
                self._compact_for_match(correction.old_text),
                self._compact_for_match(correction.new_text),
            )
            for correction in corrections
        }
        commented_rows = [
            {
                "page_no": int(correction.page_no or 0),
                "unit_id": str(correction.wps_unit_id or ""),
                "old_text": self._compact_for_match(correction.old_text),
                "new_text": self._compact_for_match(correction.new_text),
            }
            for correction in corrections
        ]
        rows: List[Dict[str, Any]] = []
        seen_rows: List[Dict[str, Any]] = []
        for result in preflight_result.specialist_review_results:
            if result.decision not in {"confirmed_error", "suspected_error"}:
                continue
            if not result.old_text or not result.new_text:
                continue
            key = (
                int(result.page_no or 0),
                str(result.wps_unit_id or ""),
                self._compact_for_match(result.old_text),
                self._compact_for_match(result.new_text),
            )
            if key in commented_keys:
                continue
            if self._replacement_already_commented(
                page_no=int(result.page_no or 0),
                unit_id=str(result.wps_unit_id or ""),
                old_text=result.old_text,
                new_text=result.new_text,
                existing_rows=commented_rows,
            ):
                continue
            if self._replacement_already_commented(
                page_no=int(result.page_no or 0),
                unit_id=str(result.wps_unit_id or ""),
                old_text=result.old_text,
                new_text=result.new_text,
                existing_rows=seen_rows,
            ):
                continue
            confirmed = (
                result.decision == "confirmed_error"
                and result.comment_policy == "comment_if_exact_replacement"
                and not self._looks_like_incomplete_confusable_numeric_replacement(
                    old_text=result.old_text,
                    new_text=result.new_text,
                )
            )
            status = "confirmed_error" if confirmed else "suspected_error"
            category = "missing_text" if result.task_type == "image_page_specialist_review" else "table_cell_mismatch"
            rows.append(
                {
                    "id": f"v4_specialist_finding_{existing_count + len(rows) + 1:04d}",
                    "severity": "high" if confirmed else "medium",
                    "category": category,
                    "page_no": result.page_no,
                    "wps_text": result.old_text,
                    "suggested_text": result.new_text,
                    "diff_ops": self._simple_diff_ops(result.old_text, result.new_text),
                    "confidence": round(float(result.confidence or 0.0), 4),
                    "status": status,
                    "evidence_sources": [result.task_type or "specialist_review"],
                    "bbox_refs": [],
                    "crop_refs": self._specialist_crop_refs(result.evidence_refs),
                    "wps_anchor": {
                        "unit_id": result.wps_unit_id,
                        "result_id": result.result_id,
                        "task_id": result.task_id,
                        "comment_policy": result.comment_policy,
                    },
                    "reason": result.reason or "专项复核发现 WPS 转换文本存在风险。",
                    "requires_human_review": True,
                }
            )
            seen_rows.append(
                {
                    "page_no": int(result.page_no or 0),
                    "unit_id": str(result.wps_unit_id or ""),
                    "old_text": self._compact_for_match(result.old_text),
                    "new_text": self._compact_for_match(result.new_text),
                }
            )
        return rows

    def _replacement_already_commented(
        self,
        *,
        page_no: int,
        unit_id: str,
        old_text: str,
        new_text: str,
        existing_rows: Sequence[Dict[str, Any]],
    ) -> bool:
        old_key = self._compact_for_match(old_text)
        new_key = self._compact_for_match(new_text)
        if not old_key or not new_key:
            return False
        for item in existing_rows:
            if int(item.get("page_no") or 0) != int(page_no or 0):
                continue
            existing_old = str(item.get("old_text") or "")
            existing_new = str(item.get("new_text") or "")
            existing_unit = str(item.get("unit_id") or "")
            if old_key == existing_old and new_key == existing_new:
                return True
            if unit_id and existing_unit == unit_id and self._replacement_texts_equivalent(
                old_left=old_key,
                old_right=existing_old,
                new_left=new_key,
                new_right=existing_new,
            ):
                return True
        return False

    def _replacement_texts_equivalent(
        self,
        *,
        old_left: str,
        old_right: str,
        new_left: str,
        new_right: str,
    ) -> bool:
        if not old_left or not old_right or not new_left or not new_right:
            return False
        if old_left == old_right and new_left == new_right:
            return True
        old_score = float(similarity(old_left, old_right))
        new_score = float(similarity(new_left, new_right))
        return old_score >= 0.92 and new_score >= 0.9

    def _specialist_crop_refs(self, evidence_refs: Sequence[Dict[str, Any]]) -> List[str]:
        refs: List[str] = []
        for item in evidence_refs:
            for key in ("crop_path", "image_path", "page_image_path", "path"):
                value = str(item.get(key) or "").strip()
                if value and value not in refs:
                    refs.append(value)
        return refs[:8]

    def _finding_from_correction(self, correction: CorrectionCandidate) -> Dict[str, Any]:
        title = self._comment_title(correction.comment_text)
        confirmed = self._is_confirmed_docx_comment(correction)
        status = "confirmed_error" if confirmed else "suspected_error"
        return {
            "id": correction.id,
            "severity": "high" if confirmed else ("low" if correction.sensitive_low_priority else "medium"),
            "category": self._finding_category_for_comment(title, correction.comment_text),
            "page_no": correction.page_no,
            "wps_text": correction.old_text or "",
            "suggested_text": correction.new_text or "",
            "diff_ops": self._simple_diff_ops(correction.old_text, correction.new_text),
            "confidence": round(float(correction.confidence or 0.0), 4),
            "status": status,
            "evidence_sources": self._evidence_sources_for_comment(title, correction.comment_text),
            "bbox_refs": [],
            "crop_refs": self._crop_refs_from_comment(correction.comment_text),
            "wps_anchor": {
                "unit_id": correction.wps_unit_id,
                "correction_id": correction.id,
                "action": correction.action,
                "alignment_score": round(float(correction.alignment_score or 0.0), 4),
            },
            "reason": correction.reason or title or "v4 已生成 DOCX 复核批注。",
            "requires_human_review": True,
        }

    def _model_conflict_findings(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        existing_count: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        diffs_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        for review in preflight_result.qwen_vl_reviews:
            if review.verdict != "conflict":
                continue
            diff = diffs_by_id.get(review.diff_id)
            rows.append(
                self._model_review_finding(
                    index=existing_count + len(rows) + 1,
                    source="qwen_vl_review",
                    review_id=review.vl_id,
                    page_no=review.page_no or (diff.pdf_page_no if diff else None),
                    wps_text=diff.docx_text if diff else "",
                    suggested_text=review.preferred_text or review.visible_text or (diff.pdf_text if diff else ""),
                    confidence=review.confidence,
                    reason=review.reason or "Qwen-VL 视觉证据与文本候选存在冲突。",
                    crop_ref=review.crop_path,
                    extra_anchor={"diff_id": review.diff_id, "verdict": review.verdict, "decision": review.decision},
                )
            )
        for review in preflight_result.qwen_gate_reviews:
            if review.verdict != "model_conflict":
                continue
            diff = diffs_by_id.get(review.diff_id)
            rows.append(
                self._model_review_finding(
                    index=existing_count + len(rows) + 1,
                    source="qwen_gate_review",
                    review_id=review.gate_id,
                    page_no=diff.pdf_page_no if diff else None,
                    wps_text=diff.docx_text if diff else "",
                    suggested_text=review.preferred_text or (diff.pdf_text if diff else ""),
                    confidence=review.confidence,
                    reason=review.reason or "Qwen 文本门槛判断模型证据存在冲突。",
                    crop_ref="",
                    extra_anchor={"diff_id": review.diff_id, "verdict": review.verdict, "decision": review.decision},
                )
            )
        return rows

    def _model_review_finding(
        self,
        *,
        index: int,
        source: str,
        review_id: str,
        page_no: Optional[int],
        wps_text: str,
        suggested_text: str,
        confidence: float,
        reason: str,
        crop_ref: str,
        extra_anchor: Dict[str, Any],
    ) -> Dict[str, Any]:
        crop_refs = [crop_ref] if crop_ref else []
        return {
            "id": f"v4_model_conflict_{index:04d}",
            "severity": "medium",
            "category": "model_conflict",
            "page_no": page_no,
            "wps_text": wps_text or "",
            "suggested_text": suggested_text or "",
            "diff_ops": self._simple_diff_ops(wps_text, suggested_text),
            "confidence": round(float(confidence or 0.0), 4),
            "status": "model_conflict",
            "evidence_sources": [source],
            "bbox_refs": [],
            "crop_refs": crop_refs,
            "wps_anchor": {"review_id": review_id, **extra_anchor},
            "reason": reason,
            "requires_human_review": True,
        }

    def _coverage_gap_findings(
        self,
        *,
        preflight_result: ConversionPreflightResult,
        existing_count: int,
    ) -> List[Dict[str, Any]]:
        quality = dict(preflight_result.quality_inspection or {})
        bottlenecks = [item for item in quality.get("bottlenecks") or [] if isinstance(item, dict)]
        rows: List[Dict[str, Any]] = []
        for item in bottlenecks:
            gap_type = str(item.get("type") or "coverage_gap")
            if gap_type in {"visual_gate_gap"}:
                # Visual conflicts get their own model_conflict findings above.
                continue
            pages = item.get("pages")
            page_no = None
            if isinstance(pages, list) and pages:
                try:
                    page_no = int(pages[0])
                except Exception:
                    page_no = None
            rows.append(
                {
                    "id": f"v4_coverage_gap_{existing_count + len(rows) + 1:04d}",
                    "severity": str(item.get("severity") or "medium"),
                    "category": gap_type,
                    "page_no": page_no,
                    "wps_text": "",
                    "suggested_text": "",
                    "diff_ops": [],
                    "confidence": 0.0,
                    "status": "coverage_gap",
                    "evidence_sources": ["quality_inspection"],
                    "bbox_refs": [],
                    "crop_refs": [],
                    "wps_anchor": {
                        "quality_bottleneck_type": gap_type,
                        "pages": pages if isinstance(pages, list) else [],
                        "count": item.get("count"),
                        "unresolved_count": item.get("unresolved_count"),
                    },
                    "reason": str(item.get("reason") or "v4 质量自检发现仍存在未覆盖高风险区域。"),
                    "requires_human_review": True,
                }
            )
        return rows

    def _finding_category_for_comment(self, title: str, comment_text: str) -> str:
        text = f"{title}\n{comment_text}"
        if "表格" in text:
            return "table_cell_mismatch"
        if "图片页" in text:
            return "missing_text"
        if "视觉" in text:
            return "substitution"
        if "顺序" in text:
            return "reading_order_mismatch"
        if "缺失" in text:
            return "missing_text"
        if "多余" in text:
            return "extra_text"
        return "substitution"

    def _evidence_sources_for_comment(self, title: str, comment_text: str) -> List[str]:
        text = f"{title}\n{comment_text}".lower()
        sources = ["docx_comment"]
        if "qwen-vl" in text or "qwen3-vl" in text or "视觉" in text:
            sources.append("qwen_vl_review")
        if "qwen" in text and "qwen_vl_review" not in sources:
            sources.append("qwen_gate_review")
        if "表格" in text:
            sources.append("table_specialist_review")
        if "图片页" in text:
            sources.append("image_page_specialist_review")
        if "ocr" in text:
            sources.append("ocr_evidence")
        return sources

    def _crop_refs_from_comment(self, comment_text: str) -> List[str]:
        refs: List[str] = []
        for match in re.finditer(r"(?:证据截图|截图证据|crop)[:：]\s*([^\n，,；;]+)", str(comment_text or ""), flags=re.IGNORECASE):
            value = match.group(1).strip()
            if value:
                refs.append(value)
        return refs[:8]

    def _simple_diff_ops(self, old_text: str, new_text: str) -> List[Dict[str, Any]]:
        old = str(old_text or "")
        new = str(new_text or "")
        if old == new:
            return []
        return [
            {
                "op": "replace" if old and new else ("delete" if old else "insert"),
                "old_text": old,
                "new_text": new,
            }
        ]

    def _finding_counts(self, findings: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        status_counts = Counter(str(item.get("status") or "") for item in findings)
        return {
            "confirmed_count": int(status_counts.get("confirmed_error", 0)),
            "suspected_count": int(status_counts.get("suspected_error", 0)),
            "model_conflict_count": int(status_counts.get("model_conflict", 0)),
            "coverage_gap_count": int(status_counts.get("coverage_gap", 0)),
        }

    def _coverage_metadata(self, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        summary = preflight_result.summary()
        anchor = dict(summary.get("anchor_ocr") or {})
        full_content = dict(summary.get("full_content_coverage") or {})
        full_backfill = dict(summary.get("full_content_backfill") or {})
        focused = dict(summary.get("focused_review") or {})
        qwen_vl = dict(summary.get("qwen_vl_gate") or {})
        image_text_vl = dict(summary.get("image_text_vl") or {})
        table_page_vl = dict(summary.get("table_page_vl") or {})
        page_text_qwen = dict(summary.get("page_text_qwen_review") or {})
        table_review = dict(summary.get("table_review") or {})
        table_audit = dict(preflight_result.table_audit_summary or {})
        mapping = dict(preflight_result.mapping_stabilization or {})
        return {
            "pages_ocr_covered": int(anchor.get("succeeded_page_count") or 0),
            "pages_structure_covered": int(table_review.get("reviewed_page_count") or 0),
            "pages_glm_reviewed": 0,
            "pages_glm_attempted": 0,
            "pages_vl_reviewed": int(qwen_vl.get("available_count") or 0)
            + int(image_text_vl.get("available_count") or 0)
            + int(table_page_vl.get("available_count") or 0),
            "pages_vl_attempted": int(qwen_vl.get("reviewed_candidate_count") or 0)
            + int(image_text_vl.get("reviewed_page_count") or 0)
            + int(table_page_vl.get("reviewed_page_count") or 0),
            "region_count": int(full_content.get("reviewed_unit_count") or 0),
            "region_crop_count": int(full_backfill.get("crop_count") or 0),
            "crop_ocr_reviewed": int(full_backfill.get("available_count") or 0),
            "crop_ocr_attempted": int(full_backfill.get("attempted_count") or 0),
            "vl_crop_reviewed": int(qwen_vl.get("available_count") or 0),
            "vl_crop_attempted": int(qwen_vl.get("reviewed_candidate_count") or 0),
            "glm_crop_reviewed": 0,
            "glm_crop_attempted": 0,
            "qwen_text_reviewed": int(page_text_qwen.get("reviewed_page_count") or 0),
            "qwen_text_available": int(page_text_qwen.get("available_page_count") or 0),
            "table_reviewed_pages": int(table_review.get("reviewed_page_count") or 0),
            "table_reviewed_cell_count": int(table_audit.get("reviewed_cell_count") or 0),
            "table_unresolved_cell_count": int(table_audit.get("unresolved_cell_count") or 0),
            "table_confirmed_cell_count": int(table_audit.get("confirmed_cell_count") or 0),
            "table_suspected_cell_count": int(table_audit.get("suspected_cell_count") or 0),
            "table_pattern_cluster_count": int(table_audit.get("cluster_count") or 0),
            "table_merged_comment_count": int(table_audit.get("merged_comment_count") or 0),
            "focused_reviewed_candidates": int(focused.get("reviewed_candidate_count") or 0),
            "mapping_stabilized_count": int(mapping.get("resolved_count") or 0),
            "mapping_unresolved_count": int(mapping.get("unresolved_count") or 0),
            "coverage_review_task_count": len(preflight_result.coverage_review_tasks),
            "human_review_required_count": sum(1 for item in preflight_result.coverage_review_tasks if item.requires_human_review()),
        }

    def _metadata(
        self,
        *,
        audit_id: str,
        source_pdf: Path,
        template_docx: Path,
        reviewed_docx_path: Path,
        report_path: Path,
        evidence_zip_path: Path,
        page_count: int,
        preflight: Dict[str, Any],
        preflight_result: ConversionPreflightResult,
        corrections: Sequence[CorrectionCandidate],
        findings: Sequence[Dict[str, Any]],
        writer_summary: Dict[str, Any],
        raw_payload_paths: Sequence[str],
        warnings: Sequence[str],
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        comment_count = int(writer_summary.get("comment_count") or 0)
        finding_counts = self._finding_counts(findings)
        metadata = {
            "audit_id": audit_id,
            "audit_profile": self.PROFILE,
            "engine": "v4",
            "mode": "comment_only" if comment_count else "report_only",
            "pdf": str(source_pdf),
            "wps_template_docx": str(template_docx),
            "audited_docx_path": str(reviewed_docx_path),
            "reviewed_docx_path": str(reviewed_docx_path),
            "audit_report_path": str(report_path),
            "evidence_zip_path": str(evidence_zip_path),
            "page_count": int(page_count),
            "comment_count": comment_count,
            "confirmed_count": finding_counts["confirmed_count"],
            "suspected_count": finding_counts["suspected_count"],
            "model_conflict_count": finding_counts["model_conflict_count"],
            "coverage_gap_count": finding_counts["coverage_gap_count"],
            "auto_replace_count": 0,
            "finding_count": len(findings),
            "findings_count": len(findings),
            "corrections_count": len(corrections),
            "comment_policy": {
                "enabled": True,
                "version": "v4_conservative_comment_v1",
                "comment_only": True,
                "auto_replace": False,
                "comment_candidate_count": len(corrections),
                "writer_summary": dict(writer_summary),
            },
            "conversion_fidelity": preflight_result.summary(),
            "raw_payload_paths": list(raw_payload_paths),
            "preflight": dict(preflight),
            "warnings": list(warnings),
            "model_timings": dict(timings),
        }
        metadata.update(self._coverage_metadata(preflight_result))
        metadata.update(self._quality_metadata(preflight_result))
        return metadata

    def _quality_metadata(self, preflight_result: ConversionPreflightResult) -> Dict[str, Any]:
        quality = dict(preflight_result.quality_inspection or {})
        summary = dict(quality.get("summary") or {})
        return {
            "quality_status": quality.get("overall_status") or summary.get("overall_status") or "",
            "quality_bottleneck_count": len(quality.get("bottlenecks") or []),
            "quality_highest_risk_pages": list(summary.get("highest_risk_pages") or []),
            "quality_recommended_next_actions": list(quality.get("recommended_next_actions") or [])[:12],
            "full_content_unresolved_count": int(summary.get("full_content_unresolved_count") or 0),
            "high_risk_page_coverage_unresolved_count": int(summary.get("high_risk_page_coverage_unresolved_count") or 0),
        }

    def _build_evidence_zip(self, *, evidence_zip_path: Path, report_path: Path, work_dir: Path) -> None:
        ensure_private_directory(evidence_zip_path.parent)
        with zipfile.ZipFile(evidence_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(report_path, "audit_report.json")
            for path in sorted(work_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(work_dir).as_posix())
        ensure_private_file(evidence_zip_path)

    def _builder_progress_callback(self, *, stage: str, current: int, total: int):
        def _callback(event: Dict[str, Any]) -> None:
            message = str(event.get("message") or "").strip() or stage
            self._emit_progress(stage, current, total, message, details=event)

        return _callback

    def _emit_progress(self, stage: str, current: int, total: int, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        if self.progress_path is None:
            return
        payload = {
            "stage": stage,
            "current": int(current),
            "total": int(total),
            "message": message,
            "updated_at": time.time(),
        }
        if details:
            payload["details"] = dict(details)
        try:
            ensure_private_directory(self.progress_path.parent)
            temp_path = self.progress_path.with_name(f".{self.progress_path.name}.{uuid.uuid4().hex}.tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            ensure_private_file(temp_path)
            temp_path.replace(self.progress_path)
            ensure_private_file(self.progress_path)
        except Exception:
            logger.debug("Failed to write v4 audit progress.", exc_info=True)
