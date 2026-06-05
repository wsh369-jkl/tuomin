"""PDF-to-Word OCR audit APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.desensitize import (
    _delete_directory_if_managed,
    _delete_if_managed,
    _peak_memory_mib_from_worker,
    _run_worker_process_blocking,
)
from app.core.config import settings
from app.core.runtime_security import ensure_private_directory, ensure_private_file
from app.schemas.desensitize import (
    PdfWordAuditCorrection,
    PdfWordAuditFinding,
    PdfWordAuditResult,
    PdfWordAuditStatus,
)
from app.services.pdf_word_audit_v4.page_terminal_state import PageTerminalStateMachine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf-word-audit", tags=["pdf-word-audit"])

audit_tasks: Dict[str, Dict] = {}
audit_task_runners: Dict[str, asyncio.Task] = {}

AUDIT_FAILURE_DETAIL = "PDF 转 Word 核查失败，请检查原 PDF、WPS 转换 DOCX 和 OCR 模型状态。"
AUDIT_INTERRUPTED_DETAIL = "PDF 转 Word 核查任务在应用重启或异常退出时中断，请重新发起核查。"
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
ZIP_MIME_TYPE = "application/zip"
TASK_MANIFEST_VERSION = "pdf_word_audit_task_v1"


def _simple_diff_ops(old_text: str, new_text: str) -> List[Dict]:
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


def _comment_title(comment_text: str) -> str:
    value = str(comment_text or "").strip()
    if not value:
        return ""
    return re.split(r"\r?\n", value, maxsplit=1)[0].strip()


def _finding_category_from_comment(title: str, comment_text: str) -> str:
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


def _evidence_sources_from_comment(title: str, comment_text: str) -> List[str]:
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


def _crop_refs_from_comment(comment_text: str) -> List[str]:
    refs: List[str] = []
    for match in re.finditer(r"(?:证据截图|截图证据|crop)[:：]\s*([^\n，,；;]+)", str(comment_text or ""), flags=re.IGNORECASE):
        value = match.group(1).strip()
        if value:
            refs.append(value)
    return refs[:8]


def _finding_from_correction_payload(correction: Dict) -> Dict:
    comment_text = str(correction.get("comment_text") or "")
    title = _comment_title(comment_text)
    confirmed = "确认错误" in title or "必须改为" in comment_text or "确认" in str(correction.get("reason") or "")
    old_text = str(correction.get("old_text") or "")
    new_text = str(correction.get("new_text") or "")
    return {
        "id": str(correction.get("id") or ""),
        "severity": "high" if confirmed else ("low" if correction.get("sensitive_low_priority") else "medium"),
        "category": _finding_category_from_comment(title, comment_text),
        "page_no": correction.get("page_no"),
        "wps_text": old_text,
        "suggested_text": new_text,
        "diff_ops": _simple_diff_ops(old_text, new_text),
        "confidence": float(correction.get("confidence") or 0.0),
        "status": "confirmed_error" if confirmed else "suspected_error",
        "evidence_sources": _evidence_sources_from_comment(title, comment_text),
        "bbox_refs": [],
        "crop_refs": _crop_refs_from_comment(comment_text),
        "wps_anchor": {
            "unit_id": str(correction.get("wps_unit_id") or ""),
            "correction_id": str(correction.get("id") or ""),
            "action": str(correction.get("action") or "review"),
            "alignment_score": float(correction.get("alignment_score") or 0.0),
        },
        "reason": str(correction.get("reason") or title or "v4 已生成 DOCX 复核批注。"),
        "requires_human_review": True,
    }


def _finding_status_counts(findings: List[Dict]) -> Dict[str, int]:
    status_counts = Counter(str(item.get("status") or "") for item in findings)
    return {
        "confirmed_count": int(status_counts.get("confirmed_error", 0)),
        "suspected_count": int(status_counts.get("suspected_error", 0)),
        "model_conflict_count": int(status_counts.get("model_conflict", 0)),
        "coverage_gap_count": int(status_counts.get("coverage_gap", 0)),
    }


def _legacy_next_engine(*, task_type: str = "", next_route: str = "") -> str:
    route = str(next_route or "").strip()
    route_map = {
        "needs_table_parser": "table_audit_engine",
        "needs_qwen_vl_table_cell_review": "table_audit_engine",
        "comment_if_exact_replacement": "docx_comment_writer",
        "needs_full_page_review": "image_page_specialist_review",
        "needs_full_page_ocr": "full_page_ocr",
        "needs_qwen_vl_page_gate": "qwen_vl_page_gate",
        "needs_qwen_vl": "qwen_vl_local_gate",
        "needs_region_ocr": "focused_crop_review",
        "needs_human_visual_review": "human_review",
        "needs_human_page_review": "human_review",
        "needs_human_mapping_review": "mapping_stabilizer",
        "needs_recall_guard_review": "recall_guard_review",
        "needs_text_alignment": "text_alignment",
    }
    if route in route_map:
        return route_map[route]

    task_type_map = {
        "table_cell_specialist_review": "table_audit_engine",
        "table_visual_specialist_review": "table_audit_engine",
        "image_page_specialist_review": "image_page_specialist_review",
        "visual_region_specialist_review": "focused_crop_review",
        "mapping_specialist_review": "mapping_stabilizer",
        "recall_guard_specialist_review": "recall_guard_review",
        "full_content_coverage_review": "text_alignment",
        "fragment_page_specialist_review": "docx_fragment_merger",
        "text_gate_specialist_review": "qwen_text_gate",
    }
    return task_type_map.get(str(task_type or "").strip(), "")


def _legacy_human_review_queue(report_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    coverage_tasks = [
        dict(item)
        for item in list(report_payload.get("coverage_review_tasks") or [])
        if isinstance(item, dict)
    ]
    if coverage_tasks:
        for item in coverage_tasks:
            if not item.get("next_engine"):
                item["next_engine"] = _legacy_next_engine(
                    task_type=str(item.get("task_type") or ""),
                    next_route=str(item.get("next_route") or ""),
                )
        return coverage_tasks

    specialist_tasks = [
        dict(item)
        for item in list(report_payload.get("specialist_review_tasks") or [])
        if isinstance(item, dict)
    ]
    if not specialist_tasks:
        return []

    queue: List[Dict[str, Any]] = []
    for item in specialist_tasks:
        row = dict(item)
        row["next_engine"] = str(
            row.get("next_engine")
            or _legacy_next_engine(
                task_type=str(row.get("task_type") or ""),
                next_route=str(row.get("next_route") or ""),
            )
        )
        row.setdefault("fallback", "human_review_required")
        queue.append(row)
    queue.sort(
        key=lambda item: (
            -int(item.get("priority") or 0),
            int(item.get("page_no") or 999999),
            str(item.get("task_type") or ""),
        )
    )
    return queue[:120]


def _legacy_review_task_summary(report_payload: Dict[str, Any], queue: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts = Counter(str(item.get("task_type") or "unknown") for item in queue)
    status_counts = Counter(str(item.get("status") or "unknown") for item in queue)
    engine_counts = Counter(str(item.get("next_engine") or "") for item in queue if str(item.get("next_engine") or ""))
    pages = sorted({int(item.get("page_no") or 0) for item in queue if int(item.get("page_no") or 0) > 0})
    return {
        "enabled": bool(queue),
        "version": "legacy_review_task_summary_v1",
        "task_count": len(queue),
        "page_count": len(pages),
        "pages": pages[:80],
        "high_priority_task_count": sum(1 for item in queue if int(item.get("priority") or 0) >= 85),
        "human_required_count": sum(1 for item in queue if str(item.get("fallback") or "") == "human_review_required"),
        "type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "next_engine_counts": dict(sorted(engine_counts.items())),
    }


def _legacy_table_summary(report_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = [
        dict(item)
        for item in list(report_payload.get("table_grid_evidence") or [])
        if isinstance(item, dict)
    ]
    audit = dict(report_payload.get("table_audit_summary") or {})
    grid_status_counts = Counter(str(item.get("status") or "unknown") for item in rows)
    return {
        "enabled": bool(rows or audit),
        "reviewed_table_count": int(audit.get("reviewed_table_count") or len(rows)),
        "reviewed_cell_count": int(audit.get("reviewed_cell_count") or sum(int(item.get("cell_count") or 0) for item in rows)),
        "confirmed_cell_count": int(audit.get("confirmed_cell_count") or sum(int(item.get("confirmed_error_count") or 0) for item in rows)),
        "suspected_cell_count": int(audit.get("suspected_cell_count") or sum(int(item.get("suspected_error_count") or 0) for item in rows)),
        "unresolved_cell_count": int(audit.get("unresolved_cell_count") or sum(int(item.get("unresolved_cell_count") or 0) for item in rows)),
        "pattern_cluster_count": int(audit.get("cluster_count") or 0),
        "merged_comment_count": int(audit.get("merged_comment_count") or 0),
        "grid_status_counts": dict(sorted(grid_status_counts.items())),
        "high_risk_pages": sorted(
            {
                int(item.get("page_no") or 0)
                for item in rows
                if int(item.get("page_no") or 0) > 0
                and (
                    int(item.get("unresolved_cell_count") or 0) > 0
                    or int(item.get("confirmed_error_count") or 0) > 0
                )
            }
        ),
    }


def _legacy_coverage_summary(report_payload: Dict[str, Any], queue: List[Dict[str, Any]]) -> Dict[str, Any]:
    reviews = [
        dict(item)
        for item in list(report_payload.get("full_content_coverage_reviews") or [])
        if isinstance(item, dict)
    ]
    unresolved_rows = [item for item in reviews if str(item.get("decision") or "") != "covered"]
    coverage_status_counts = Counter(str(item.get("status") or "unknown") for item in reviews)
    coverage_decision_counts = Counter(str(item.get("decision") or "unknown") for item in reviews)
    task_type_counts = Counter(str(item.get("task_type") or "unknown") for item in queue)
    task_status_counts = Counter(str(item.get("status") or "unknown") for item in queue)
    unresolved_pages = sorted({int(item.get("page_no") or 0) for item in unresolved_rows if int(item.get("page_no") or 0) > 0})
    return {
        "coverage_review_count": len(reviews),
        "unresolved_count": len(unresolved_rows),
        "unresolved_pages": unresolved_pages,
        "coverage_status_counts": dict(sorted(coverage_status_counts.items())),
        "coverage_decision_counts": dict(sorted(coverage_decision_counts.items())),
        "review_task_count": len(queue),
        "review_task_type_counts": dict(sorted(task_type_counts.items())),
        "review_task_status_counts": dict(sorted(task_status_counts.items())),
        "quality_bottlenecks": list((report_payload.get("quality_inspection") or {}).get("bottlenecks") or [])[:80],
    }


def _legacy_page_risk_summary(
    report_payload: Dict[str, Any],
    *,
    findings: List[Dict[str, Any]],
    queue: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pages: Dict[int, Dict[str, Any]] = {}

    def ensure_page(page_no: int) -> Dict[str, Any]:
        page = pages.get(page_no)
        if page is None:
            page = {
                "page_no": page_no,
                "risk_level": "low",
                "labels": [],
                "finding_count": 0,
                "confirmed_count": 0,
                "suspected_count": 0,
                "model_conflict_count": 0,
                "coverage_gap_count": 0,
                "review_task_count": 0,
                "table_grid_count": 0,
                "unresolved_coverage_count": 0,
                "reasons": [],
            }
            pages[page_no] = page
        return page

    for item in list(report_payload.get("high_risk_page_coverage_reviews") or []):
        if not isinstance(item, dict):
            continue
        page_no = int(item.get("page_no") or 0)
        if page_no <= 0:
            continue
        page = ensure_page(page_no)
        page["risk_level"] = str(item.get("risk_level") or page["risk_level"] or "high")
        page["review_task_count"] = max(page["review_task_count"], len(item.get("coverage_review_ids") or []))
        page["unresolved_coverage_count"] = max(page["unresolved_coverage_count"], int(item.get("unresolved_count") or 0))
        labels = item.get("flags") or []
        if isinstance(labels, list):
            page["labels"] = [str(value) for value in labels[:20]]
        reason = str(item.get("reason") or "")
        if reason:
            page["reasons"].append(reason)

    for item in list(report_payload.get("table_grid_evidence") or []):
        if not isinstance(item, dict):
            continue
        page_no = int(item.get("page_no") or 0)
        if page_no <= 0:
            continue
        page = ensure_page(page_no)
        page["table_grid_count"] += 1
        page["unresolved_coverage_count"] += int(item.get("unresolved_cell_count") or 0)
        if int(item.get("unresolved_cell_count") or 0) > 0:
            page["risk_level"] = "high"
            page["reasons"].append(str(item.get("status") or "grid_pending_review"))

    for item in list(report_payload.get("full_content_coverage_reviews") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("decision") or "") == "covered":
            continue
        page_no = int(item.get("page_no") or 0)
        if page_no <= 0:
            continue
        page = ensure_page(page_no)
        page["unresolved_coverage_count"] += 1
        status = str(item.get("status") or "")
        if status:
            page["reasons"].append(status)

    for finding in findings:
        page_no = int(finding.get("page_no") or 0)
        if page_no <= 0:
            continue
        page = ensure_page(page_no)
        page["finding_count"] += 1
        status = str(finding.get("status") or "")
        if status == "confirmed_error":
            page["confirmed_count"] += 1
        elif status == "suspected_error":
            page["suspected_count"] += 1
        elif status == "model_conflict":
            page["model_conflict_count"] += 1
        elif status == "coverage_gap":
            page["coverage_gap_count"] += 1
        reason = str(finding.get("category") or status or "")
        if reason:
            page["reasons"].append(reason)

    for task in queue:
        page_no = int(task.get("page_no") or 0)
        if page_no <= 0:
            continue
        page = ensure_page(page_no)
        page["review_task_count"] += 1
        reason = str(task.get("task_type") or task.get("reason") or "")
        if reason:
            page["reasons"].append(reason)

    rows: List[Dict[str, Any]] = []
    for page_no in sorted(pages):
        page = pages[page_no]
        unique_reasons = []
        for reason in page["reasons"]:
            text = str(reason or "").strip()
            if text and text not in unique_reasons:
                unique_reasons.append(text)
        page["reasons"] = unique_reasons[:12]
        if page["confirmed_count"] or page["coverage_gap_count"] or page["model_conflict_count"] or page["unresolved_coverage_count"]:
            page["risk_level"] = "high"
        elif page["suspected_count"] or page["review_task_count"]:
            page["risk_level"] = "medium"
        rows.append(page)
    return PageTerminalStateMachine().apply(pages=rows)


def _legacy_product_report(
    *,
    audit_id: str,
    metadata: Dict[str, Any],
    findings: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    page_risk_summary: List[Dict[str, Any]],
    table_summary: Dict[str, Any],
    coverage_summary: Dict[str, Any],
    human_review_queue: List[Dict[str, Any]],
    artifact_manifest: Dict[str, Any],
    quality_inspection: Dict[str, Any],
    page_terminal_summary: Dict[str, Any],
) -> Dict[str, Any]:
    counts = _finding_status_counts(findings)
    quality_status = str(quality_inspection.get("overall_status") or "")
    if quality_status in {"needs_pipeline_improvement", "needs_more_review"} or coverage_summary.get("unresolved_count") or human_review_queue:
        status = "needs_human_review"
    elif counts["confirmed_count"] > 0:
        status = "confirmed_errors_found"
    else:
        status = "no_confirmed_error"
    page_count = int(metadata.get("page_count") or max((int(item.get("page_no") or 0) for item in page_risk_summary), default=0))
    return {
        "enabled": True,
        "version": "product_report_compat_v1",
        "audit_id": audit_id,
        "status": status,
        "summary": {
            "page_count": page_count,
            "finding_count": len(findings),
            "comment_count": len(corrections),
            "confirmed_count": counts["confirmed_count"],
            "suspected_count": counts["suspected_count"],
            "model_conflict_count": counts["model_conflict_count"],
            "coverage_gap_count": counts["coverage_gap_count"],
            "finding_status_counts": dict(sorted(Counter(str(item.get("status") or "unknown") for item in findings).items())),
            "finding_category_counts": dict(sorted(Counter(str(item.get("category") or "unknown") for item in findings).items())),
            "human_review_task_count": len(human_review_queue),
            "model_guarded_count": int(quality_inspection.get("guarded_count") or 0),
            "page_terminal_state_counts": dict(page_terminal_summary.get("terminal_state_counts") or {}),
            "resolved_page_count": int(page_terminal_summary.get("resolved_page_count") or 0),
            "review_required_page_count": int(page_terminal_summary.get("review_required_page_count") or 0),
            "high_risk_review_required_page_count": int(
                page_terminal_summary.get("high_risk_review_required_page_count") or 0
            ),
        },
        "page_risk_summary": page_risk_summary,
        "page_terminal_summary": page_terminal_summary,
        "table_summary": table_summary,
        "coverage_summary": coverage_summary,
        "model_guard_summary": dict(quality_inspection.get("model_output_guard") or {}),
        "human_review_queue": human_review_queue,
        "artifact_manifest": artifact_manifest,
    }


def _backfill_legacy_product_contract(task: Dict[str, Any], report_payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(report_payload.get("metadata") or {})
    audit_id = str(report_payload.get("audit_id") or task.get("audit_id") or "")
    findings = [dict(item) for item in list(report_payload.get("findings") or []) if isinstance(item, dict)]
    corrections = [dict(item) for item in list(report_payload.get("corrections") or []) if isinstance(item, dict)]
    artifact_manifest = {
        "reviewed_docx_path": str(
            metadata.get("reviewed_docx_path")
            or metadata.get("audited_docx_path")
            or task.get("output_path")
            or ""
        ),
        "audit_report_path": str(metadata.get("audit_report_path") or task.get("report_path") or ""),
        "evidence_zip_path": str(metadata.get("evidence_zip_path") or task.get("evidence_zip_path") or ""),
        "raw_payload_paths": list(metadata.get("raw_payload_paths") or []),
    }
    human_review_queue = _legacy_human_review_queue(report_payload)
    review_task_summary = _legacy_review_task_summary(report_payload, human_review_queue)
    table_summary = _legacy_table_summary(report_payload)
    page_risk_summary = _legacy_page_risk_summary(
        report_payload,
        findings=findings,
        queue=human_review_queue,
    )
    page_terminal_summary = PageTerminalStateMachine().summary(pages=page_risk_summary)
    coverage_summary = _legacy_coverage_summary(report_payload, human_review_queue)
    coverage_summary["page_terminal_state_counts"] = dict(page_terminal_summary.get("terminal_state_counts") or {})
    coverage_summary["high_risk_review_required_page_count"] = int(
        page_terminal_summary.get("high_risk_review_required_page_count") or 0
    )
    product_report = _legacy_product_report(
        audit_id=audit_id,
        metadata=metadata,
        findings=findings,
        corrections=corrections,
        page_risk_summary=page_risk_summary,
        table_summary=table_summary,
        coverage_summary=coverage_summary,
        human_review_queue=human_review_queue,
        artifact_manifest=artifact_manifest,
        quality_inspection=dict(report_payload.get("quality_inspection") or {}),
        page_terminal_summary=page_terminal_summary,
    )
    return {
        "product_report": product_report,
        "page_risk_summary": page_risk_summary,
        "page_terminal_summary": page_terminal_summary,
        "table_summary": table_summary,
        "coverage_summary": coverage_summary,
        "artifact_manifest": artifact_manifest,
        "review_task_summary": review_task_summary,
        "human_review_queue": human_review_queue,
    }


def _normalize_audit_result_contract(result: Dict) -> Dict:
    """Keep the public result contract usable for both new and legacy v4 runs."""

    normalized = dict(result or {})
    corrections = [dict(item) for item in list(normalized.get("corrections") or []) if isinstance(item, dict)]
    findings = [dict(item) for item in list(normalized.get("findings") or []) if isinstance(item, dict)]
    if not findings and corrections:
        findings = [_finding_from_correction_payload(item) for item in corrections]

    metadata = dict(normalized.get("metadata") or {})
    counts = _finding_status_counts(findings)
    metadata.update(counts)
    metadata["finding_count"] = len(findings)
    metadata["findings_count"] = len(findings)
    metadata["corrections_count"] = len(corrections)
    metadata["comment_count"] = int(metadata.get("comment_count") or normalized.get("comment_count") or len(corrections))

    normalized["metadata"] = metadata
    normalized["findings"] = findings
    normalized["corrections"] = corrections
    normalized["finding_count"] = len(findings)
    normalized["correction_count"] = int(normalized.get("correction_count") or len(corrections))
    normalized["comment_count"] = int(normalized.get("comment_count") or metadata["comment_count"])
    return normalized


def _audit_output_dir() -> Path:
    output_dir = Path(settings.RUNTIME_ROOT) / "pdf_word_audit"
    ensure_private_directory(output_dir)
    return output_dir


def _audit_task_manifest_dir() -> Path:
    manifest_dir = Path(settings.RUNTIME_ROOT) / "pdf_word_audit_tasks"
    ensure_private_directory(manifest_dir)
    return manifest_dir


def _audit_task_manifest_path(audit_id: str) -> Path:
    return _audit_task_manifest_dir() / f"{audit_id}.json"


def _audit_work_dir(audit_id: str) -> Path:
    return _audit_output_dir() / "v4_evidence" / audit_id[:12]


def _serialize_task_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_task_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_task_value(item) for item in value]
    return value


def _parse_task_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.min
    return datetime.min


def _task_manifest_payload(task: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        str(key): _serialize_task_value(value)
        for key, value in task.items()
        if key != "result"
    }
    payload["manifest_version"] = TASK_MANIFEST_VERSION
    return payload


def _write_task_manifest(task: Dict[str, Any]) -> None:
    audit_id = str(task.get("audit_id") or "").strip()
    if not audit_id:
        return
    path = _audit_task_manifest_path(audit_id)
    payload = _task_manifest_payload(task)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_private_file(path)


def _read_task_manifest(audit_id: str) -> Optional[Dict[str, Any]]:
    path = _audit_task_manifest_path(audit_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read PDF word audit task manifest: %s", path, exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    payload["audit_id"] = str(payload.get("audit_id") or audit_id)
    payload["created_at"] = _parse_task_datetime(payload.get("created_at"))
    updated_at = _parse_task_datetime(payload.get("updated_at"))
    if updated_at != datetime.min:
        payload["updated_at"] = updated_at
    else:
        payload.pop("updated_at", None)
    return payload


def _mark_interrupted_task_failed(task: Dict[str, Any]) -> Dict[str, Any]:
    status = str(task.get("status") or "").strip().lower()
    if status in {"completed", "failed"}:
        return task
    task["status"] = "failed"
    task["progress"] = 100
    task["message"] = AUDIT_INTERRUPTED_DETAIL
    task["error_message"] = AUDIT_INTERRUPTED_DETAIL
    task["updated_at"] = datetime.now()
    _write_task_manifest(task)
    return task


def _get_task(audit_id: str) -> Optional[Dict[str, Any]]:
    task = audit_tasks.get(audit_id)
    if isinstance(task, dict):
        return task
    task = _read_task_manifest(audit_id)
    if not isinstance(task, dict):
        return None
    task = _mark_interrupted_task_failed(task)
    audit_tasks[audit_id] = task
    return task


def _remove_task_manifest(audit_id: str) -> None:
    path = _audit_task_manifest_path(audit_id)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to remove PDF word audit task manifest: %s", path, exc_info=True)


def _result_from_report_payload(task: Dict[str, Any], report_payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(report_payload.get("metadata") or {})
    product_report = dict(report_payload.get("product_report") or {})
    artifact_manifest = dict(report_payload.get("artifact_manifest") or product_report.get("artifact_manifest") or {})
    output_path = str(
        task.get("output_path")
        or metadata.get("audited_docx_path")
        or metadata.get("reviewed_docx_path")
        or artifact_manifest.get("reviewed_docx_path")
        or ""
    )
    report_path = str(
        task.get("report_path")
        or metadata.get("audit_report_path")
        or artifact_manifest.get("audit_report_path")
        or ""
    )
    evidence_path = str(
        task.get("evidence_zip_path")
        or metadata.get("evidence_zip_path")
        or artifact_manifest.get("evidence_zip_path")
        or ""
    )
    page_risk_summary = list(report_payload.get("page_risk_summary") or product_report.get("page_risk_summary") or [])
    table_summary = dict(report_payload.get("table_summary") or product_report.get("table_summary") or {})
    coverage_summary = dict(report_payload.get("coverage_summary") or product_report.get("coverage_summary") or {})
    human_review_queue = list(report_payload.get("human_review_queue") or product_report.get("human_review_queue") or [])
    review_task_summary = dict(report_payload.get("review_task_summary") or {})
    if not product_report or not page_risk_summary or not table_summary or not coverage_summary or not human_review_queue or not review_task_summary:
        legacy = _backfill_legacy_product_contract(task, report_payload)
        if not product_report:
            product_report = dict(legacy.get("product_report") or {})
        if not page_risk_summary:
            page_risk_summary = list(legacy.get("page_risk_summary") or [])
        if not table_summary:
            table_summary = dict(legacy.get("table_summary") or {})
        if not coverage_summary:
            coverage_summary = dict(legacy.get("coverage_summary") or {})
        if not artifact_manifest:
            artifact_manifest = dict(legacy.get("artifact_manifest") or {})
        if not review_task_summary:
            review_task_summary = dict(legacy.get("review_task_summary") or {})
        if not human_review_queue:
            human_review_queue = list(legacy.get("human_review_queue") or [])
    page_risk_summary = PageTerminalStateMachine().apply(pages=page_risk_summary)
    page_terminal_summary = PageTerminalStateMachine().summary(pages=page_risk_summary)
    coverage_summary["page_terminal_state_counts"] = dict(page_terminal_summary.get("terminal_state_counts") or {})
    coverage_summary["high_risk_review_required_page_count"] = int(
        page_terminal_summary.get("high_risk_review_required_page_count") or 0
    )
    product_summary = dict(product_report.get("summary") or {})
    product_summary["page_terminal_state_counts"] = dict(page_terminal_summary.get("terminal_state_counts") or {})
    product_summary["resolved_page_count"] = int(page_terminal_summary.get("resolved_page_count") or 0)
    product_summary["review_required_page_count"] = int(page_terminal_summary.get("review_required_page_count") or 0)
    product_summary["high_risk_review_required_page_count"] = int(
        page_terminal_summary.get("high_risk_review_required_page_count") or 0
    )
    product_report["summary"] = product_summary
    product_report["page_risk_summary"] = page_risk_summary
    product_report["page_terminal_summary"] = page_terminal_summary
    product_report["coverage_summary"] = coverage_summary
    product_report["table_summary"] = table_summary
    product_report["human_review_queue"] = human_review_queue
    product_report["artifact_manifest"] = artifact_manifest
    return {
        "audit_id": str(report_payload.get("audit_id") or task.get("audit_id") or ""),
        "audited_docx_path": output_path,
        "reviewed_docx_path": output_path,
        "audit_report_path": report_path,
        "evidence_zip_path": evidence_path,
        "metadata": metadata,
        "product_report": product_report,
        "page_risk_summary": page_risk_summary,
        "table_summary": table_summary,
        "coverage_summary": coverage_summary,
        "artifact_manifest": artifact_manifest,
        "review_task_summary": review_task_summary,
        "human_review_queue": human_review_queue,
        "findings": list(report_payload.get("findings") or []),
        "corrections": list(report_payload.get("corrections") or []),
        "comment_count": int(report_payload.get("comment_count") or metadata.get("comment_count") or 0),
        "correction_count": int(report_payload.get("correction_count") or len(report_payload.get("corrections") or [])),
        "finding_count": int(report_payload.get("finding_count") or len(report_payload.get("findings") or [])),
    }


def _load_task_result(task: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(task.get("result") or {})
    if result:
        return result
    report_path = str(task.get("report_path") or "")
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load PDF word audit report payload: %s", path, exc_info=True)
        return {}
    if not isinstance(payload, dict):
        return {}
    result = _result_from_report_payload(task, payload)
    before = (
        str(task.get("output_path") or ""),
        str(task.get("report_path") or ""),
        str(task.get("evidence_zip_path") or ""),
        str(task.get("output_filename") or ""),
    )
    task["output_path"] = str(task.get("output_path") or result.get("reviewed_docx_path") or "")
    task["report_path"] = str(task.get("report_path") or result.get("audit_report_path") or "")
    task["evidence_zip_path"] = str(task.get("evidence_zip_path") or result.get("evidence_zip_path") or "")
    task["output_filename"] = _task_output_filename(task)
    task["result"] = result
    after = (
        str(task.get("output_path") or ""),
        str(task.get("report_path") or ""),
        str(task.get("evidence_zip_path") or ""),
        str(task.get("output_filename") or ""),
    )
    if after != before:
        _write_task_manifest(task)
    return result


def _task_output_filename(task: Dict[str, Any]) -> str:
    filename = str(task.get("output_filename") or "").strip()
    if filename:
        return filename
    source_name = str(task.get("filename") or "document")
    return f"{Path(source_name).stem}_reviewed.docx"


def _task_created_at(task: Dict) -> datetime:
    created_at = task.get("created_at")
    if isinstance(created_at, datetime):
        return created_at
    if isinstance(created_at, str):
        return _parse_task_datetime(created_at)
    return datetime.min


def _set_task_state(
    task: Dict,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    if status is not None:
        task["status"] = status
    if progress is not None:
        task["progress"] = max(0, min(100, int(progress)))
    if message is not None:
        task["message"] = message
    if error_message is not None:
        task["error_message"] = error_message
    task["updated_at"] = datetime.now()
    _write_task_manifest(task)


def _status_response(audit_id: str, task: Dict) -> PdfWordAuditStatus:
    return PdfWordAuditStatus(
        audit_id=audit_id,
        filename=task.get("filename"),
        template_filename=task.get("template_filename"),
        status=str(task.get("status") or "queued"),
        progress=max(0, min(100, int(task.get("progress") or 0))),
        message=task.get("message"),
        error_message=task.get("error_message"),
        created_at=_task_created_at(task),
    )


def _save_upload(audit_id: str, filename: str, content: bytes, *, expected_suffix: str, label: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix != expected_suffix:
        raise HTTPException(status_code=400, detail=f"{label} 必须是 {expected_suffix} 文件。")
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"{label} 文件过大，单文件最大 {settings.MAX_UPLOAD_SIZE // (1024 * 1024)}MB。",
        )
    path = Path(settings.UPLOAD_DIR) / f"pdf_word_audit_{audit_id}_{label}{suffix}"
    path.write_bytes(content)
    ensure_private_file(path)
    return str(path)


def _progress_callback(audit_id: str):
    bands = {
        "workflow_preflight": (5, 8),
        "page_render": (8, 16),
        "page_orientation": (16, 22),
        "evidence_extract": (22, 30),
        "v4_alignment": (30, 40),
        "v4_focused_review": (40, 48),
        "v4_table_structure": (48, 56),
        "v4_content_coverage": (56, 68),
        "v4_page_risk_aggregation": (68, 74),
        "v4_image_text_vl": (74, 80),
        "v4_table_page_vl": (80, 86),
        "v4_qwen_vl_gate": (86, 90),
        "v4_qwen_text_gate": (90, 93),
        "v4_specialist_review_plan": (93, 95),
        "v4_table_specialist_review": (95, 97),
        "audit_write_report": (97, 99),
        "audit_done": (99, 100),
    }

    def _callback(payload: Dict[str, object]) -> None:
        task = audit_tasks.get(audit_id)
        if task is None or str(task.get("status") or "").lower() in {"completed", "failed"}:
            return
        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        progress = int(task.get("progress") or 5)
        band = bands.get(stage)
        if band is not None:
            start, end = band
            if total > 0:
                progress = start + int((end - start) * min(max(current / total, 0.0), 1.0))
            else:
                progress = max(progress, start)
        message = str(payload.get("message") or "").strip() or task.get("message") or "正在执行 PDF 转 Word 核查..."
        _set_task_state(task, status="processing", progress=progress, message=message)

    return _callback


def run_pdf_word_audit_worker_blocking(
    *,
    audit_id: str,
    pdf_path: str,
    wps_docx_path: str,
    progress_callback=None,
) -> Dict[str, object]:
    audit_threads = max(1, int(getattr(settings, "PDF_WORD_AUDIT_V4_WORKER_THREADS", 1) or 1))
    result = _run_worker_process_blocking(
        module_name="app.workers.pdf_word_audit_worker",
        payload={
            "audit_id": audit_id,
            "pdf_path": str(Path(pdf_path).expanduser().resolve()),
            "wps_docx_path": str(Path(wps_docx_path).expanduser().resolve()),
            "output_dir": str(_audit_output_dir()),
        },
        input_prefix="pdf-word-audit-in-",
        output_prefix="pdf-word-audit-out-",
        timeout_seconds=max(int(settings.LOCAL_PDF_FRONTLINE_WORKER_TIMEOUT or 0), 7200),
        progress_callback=progress_callback,
        env_overrides={
            "OMP_NUM_THREADS": str(audit_threads),
            "OPENBLAS_NUM_THREADS": str(audit_threads),
            "MKL_NUM_THREADS": str(audit_threads),
            "VECLIB_MAXIMUM_THREADS": str(audit_threads),
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    audit = dict(result.get("audit") or {})
    metadata = dict(audit.get("metadata") or {})
    worker = dict(result.get("_worker_process") or {})
    peak_mib, peak_source = _peak_memory_mib_from_worker(worker)
    metadata.update(
        {
            "audit_worker_process": True,
            "audit_worker_pid": worker.get("pid"),
            "audit_worker_seconds": worker.get("seconds"),
            "audit_worker_peak_rss_mib": worker.get("peak_rss_mib"),
            "audit_worker_peak_footprint_mib": worker.get("peak_footprint_mib"),
            "audit_worker_peak_mib": round(peak_mib, 1) if peak_mib is not None else None,
            "audit_worker_peak_source": peak_source,
        }
    )
    sidecar_peak = None
    ocr_metadata = dict(metadata.get("ocr_metadata") or {})
    try:
        sidecar_peak = float(
            metadata.get("paddle_text_sidecar_peak_rss_mib")
            or ocr_metadata.get("paddle_text_sidecar_peak_rss_mib")
            or 0.0
        )
    except (TypeError, ValueError):
        sidecar_peak = None
    pipeline_peaks = [value for value in (peak_mib, sidecar_peak) if value is not None and value > 0]
    if pipeline_peaks:
        metadata["audit_pipeline_peak_mib"] = round(max(pipeline_peaks), 1)
    audit["metadata"] = metadata
    return audit


async def _run_audit_task(audit_id: str) -> None:
    task = _get_task(audit_id)
    if task is None:
        return
    try:
        task["work_dir"] = str(_audit_work_dir(audit_id))
        _set_task_state(task, status="processing", progress=5, message="正在准备 PDF 转 Word 核查...")
        audit = await asyncio.to_thread(
            run_pdf_word_audit_worker_blocking,
            audit_id=audit_id,
            pdf_path=str(task["pdf_path"]),
            wps_docx_path=str(task["wps_docx_path"]),
            progress_callback=_progress_callback(audit_id),
        )
        output_path = str(audit.get("reviewed_docx_path") or audit.get("audited_docx_path") or "")
        if not output_path or not Path(output_path).exists():
            raise RuntimeError("pdf_word_audit_missing_output_docx")
        task["result"] = audit
        task["output_path"] = output_path
        task["output_filename"] = f"{Path(str(task['filename'])).stem}_reviewed.docx"
        task["report_path"] = str(audit.get("audit_report_path") or (audit.get("metadata") or {}).get("audit_report_path") or "")
        task["evidence_zip_path"] = str(audit.get("evidence_zip_path") or (audit.get("metadata") or {}).get("evidence_zip_path") or "")
        _set_task_state(task, status="completed", progress=100, message="PDF 转 Word 核查完成。")
    except Exception as exc:
        logger.error("PDF word audit failed: %s", exc, exc_info=True)
        _set_task_state(task, status="failed", progress=100, message=AUDIT_FAILURE_DETAIL, error_message=AUDIT_FAILURE_DETAIL)
    finally:
        audit_task_runners.pop(audit_id, None)


def _purge_task(audit_id: str) -> None:
    task = audit_tasks.pop(audit_id, None)
    if not isinstance(task, dict):
        task = _read_task_manifest(audit_id)

    audit_task_runners.pop(audit_id, None)
    if isinstance(task, dict):
        output_parent = str(_audit_output_dir())
        _delete_if_managed(task.get("output_path"), allowed_parent=output_parent)
        _delete_if_managed(task.get("report_path"), allowed_parent=output_parent)
        _delete_if_managed(task.get("evidence_zip_path"), allowed_parent=output_parent)
        _delete_directory_if_managed(task.get("work_dir"), allowed_parent=output_parent)
        _delete_if_managed(task.get("pdf_path"), allowed_parent=settings.UPLOAD_DIR)
        _delete_if_managed(task.get("wps_docx_path"), allowed_parent=settings.UPLOAD_DIR)
    _remove_task_manifest(audit_id)


def _purge_terminal_tasks(now: Optional[datetime] = None) -> None:
    current_time = now or datetime.now()
    retention = timedelta(hours=max(settings.TASK_RETENTION_HOURS, 1))
    terminal_tasks: Dict[str, Dict[str, Any]] = {}

    for audit_id, task in list(audit_tasks.items()):
        if not isinstance(task, dict):
            continue
        if str(task.get("status") or "").lower() in {"completed", "failed"}:
            terminal_tasks[audit_id] = task

    for manifest_path in _audit_task_manifest_dir().glob("*.json"):
        audit_id = manifest_path.stem
        if audit_id in terminal_tasks:
            continue
        task = _read_task_manifest(audit_id)
        if not isinstance(task, dict):
            continue
        if str(task.get("status") or "").lower() in {"completed", "failed"}:
            terminal_tasks[audit_id] = task

    expired_task_ids = [
        audit_id
        for audit_id, task in terminal_tasks.items()
        if current_time - _task_created_at(task) > retention
    ]
    for audit_id in expired_task_ids:
        _purge_task(audit_id)

    remaining_finished = [
        (_task_created_at(task), audit_id)
        for audit_id, task in terminal_tasks.items()
        if audit_id not in set(expired_task_ids)
    ]
    overflow = len(remaining_finished) - max(settings.MAX_PRESERVED_TASKS, 1)
    if overflow > 0:
        for _, audit_id in sorted(remaining_finished, key=lambda item: item[0])[:overflow]:
            _purge_task(audit_id)


@router.post("/upload", response_model=PdfWordAuditStatus)
async def upload_for_pdf_word_audit(
    pdf_file: UploadFile = File(...),
    wps_docx_file: UploadFile = File(...),
):
    _purge_terminal_tasks()
    audit_id = str(uuid.uuid4())
    pdf_name = pdf_file.filename or "source.pdf"
    docx_name = wps_docx_file.filename or "wps-template.docx"
    pdf_content = await pdf_file.read()
    docx_content = await wps_docx_file.read()
    pdf_path = ""
    docx_path = ""
    try:
        pdf_path = _save_upload(audit_id, pdf_name, pdf_content, expected_suffix=".pdf", label="pdf")
        docx_path = _save_upload(audit_id, docx_name, docx_content, expected_suffix=".docx", label="wps")
    except Exception:
        _delete_if_managed(pdf_path, allowed_parent=settings.UPLOAD_DIR)
        _delete_if_managed(docx_path, allowed_parent=settings.UPLOAD_DIR)
        raise
    task = {
        "audit_id": audit_id,
        "filename": pdf_name,
        "template_filename": docx_name,
        "pdf_path": pdf_path,
        "wps_docx_path": docx_path,
        "work_dir": str(_audit_work_dir(audit_id)),
        "status": "queued",
        "progress": 5,
        "message": "文件已上传，正在准备 PDF 转 Word 核查...",
        "created_at": datetime.now(),
    }
    audit_tasks[audit_id] = task
    _write_task_manifest(task)
    audit_task_runners[audit_id] = asyncio.create_task(_run_audit_task(audit_id))
    return _status_response(audit_id, task)


@router.get("/status/{audit_id}", response_model=PdfWordAuditStatus)
async def get_pdf_word_audit_status(audit_id: str):
    task = _get_task(audit_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Audit task was not found.")
    return _status_response(audit_id, task)


@router.get("/result/{audit_id}", response_model=PdfWordAuditResult)
async def get_pdf_word_audit_result(audit_id: str):
    task = _get_task(audit_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Audit task was not found.")
    status = str(task.get("status") or "").lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or AUDIT_FAILURE_DETAIL)
    if status != "completed":
        raise HTTPException(status_code=409, detail="PDF 转 Word 核查仍在运行。")
    result = _normalize_audit_result_contract(_load_task_result(task))
    if not result:
        raise HTTPException(status_code=500, detail="PDF 转 Word 核查结果不可用。")
    return PdfWordAuditResult(
        audit_id=audit_id,
        filename=str(task.get("filename") or ""),
        template_filename=str(task.get("template_filename") or ""),
        status="completed",
        metadata=dict(result.get("metadata") or {}),
        product_report=dict(result.get("product_report") or {}),
        page_risk_summary=list(result.get("page_risk_summary") or []),
        table_summary=dict(result.get("table_summary") or {}),
        coverage_summary=dict(result.get("coverage_summary") or {}),
        artifact_manifest=dict(result.get("artifact_manifest") or {}),
        review_task_summary=dict(result.get("review_task_summary") or {}),
        human_review_queue=list(result.get("human_review_queue") or []),
        findings=[PdfWordAuditFinding(**item) for item in list(result.get("findings") or [])],
        corrections=[PdfWordAuditCorrection(**item) for item in list(result.get("corrections") or [])],
        download_url=f"/api/v1/pdf-word-audit/download/{audit_id}",
        report_url=f"/api/v1/pdf-word-audit/report/{audit_id}",
        evidence_url=f"/api/v1/pdf-word-audit/evidence/{audit_id}",
        output_filename=_task_output_filename(task),
    )


@router.get("/download/{audit_id}")
async def download_pdf_word_audit_result(audit_id: str):
    task = _get_task(audit_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Audit task was not found.")
    result = _load_task_result(task)
    output_path = str(task.get("output_path") or result.get("reviewed_docx_path") or result.get("audited_docx_path") or "")
    if not output_path:
        raise HTTPException(status_code=400, detail="Audit output is not available yet.")
    path = Path(output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit output file no longer exists.")
    return FileResponse(
        path,
        media_type=DOCX_MIME_TYPE,
        filename=_task_output_filename(task),
    )


@router.get("/report/{audit_id}")
async def download_pdf_word_audit_report(audit_id: str):
    task = _get_task(audit_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Audit task was not found.")
    result = _load_task_result(task)
    report_path = str(task.get("report_path") or result.get("audit_report_path") or "")
    if not report_path:
        raise HTTPException(status_code=400, detail="Audit report is not available yet.")
    path = Path(report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit report file no longer exists.")
    return FileResponse(path, media_type="application/json", filename=f"{Path(str(task.get('filename') or 'document')).stem}_audit_report.json")


@router.get("/evidence/{audit_id}")
async def download_pdf_word_audit_evidence(audit_id: str):
    task = _get_task(audit_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Audit task was not found.")
    result = _load_task_result(task)
    evidence_path = str(task.get("evidence_zip_path") or result.get("evidence_zip_path") or "")
    if not evidence_path:
        raise HTTPException(status_code=400, detail="Audit evidence is not available yet.")
    path = Path(evidence_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit evidence file no longer exists.")
    return FileResponse(path, media_type=ZIP_MIME_TYPE, filename=f"{Path(str(task.get('filename') or 'document')).stem}_evidence.zip")
