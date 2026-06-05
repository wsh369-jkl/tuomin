from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET


EVAL_BASELINE_VERSION = "v4_eval_baseline_v1"

CORE_EVIDENCE_ZIP_ENTRIES = (
    "audit_report.json",
    "evidence/raw/comment_corrections.json",
    "evidence/raw/quality_inspection.json",
    "evidence/raw/specialist_review_results.json",
    "evidence/raw/table_cell_evidence_reviews.json",
)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_eval_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_audit_files(
    *,
    worker_output_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> Dict[str, Any]:
    worker_output = load_json(worker_output_path) if worker_output_path else None
    report_payload: Optional[Dict[str, Any]] = None
    resolved_report_path = Path(report_path).expanduser() if report_path else None

    if resolved_report_path is None and worker_output:
        audit_payload = _audit_payload(worker_output)
        raw_report_path = audit_payload.get("audit_report_path") or (audit_payload.get("metadata") or {}).get("audit_report_path")
        if raw_report_path:
            resolved_report_path = Path(str(raw_report_path)).expanduser()

    if resolved_report_path and resolved_report_path.exists():
        report_payload = load_json(resolved_report_path)

    return summarize_audit_payloads(
        worker_output=worker_output,
        report_payload=report_payload,
        worker_output_path=Path(worker_output_path).expanduser() if worker_output_path else None,
        report_path=resolved_report_path,
    )


def summarize_audit_payloads(
    *,
    worker_output: Optional[Mapping[str, Any]] = None,
    report_payload: Optional[Mapping[str, Any]] = None,
    worker_output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    audit_payload = _audit_payload(worker_output or {})
    metadata = _metadata(audit_payload=audit_payload, report_payload=report_payload)
    report = report_payload or {}
    corrections = _corrections(audit_payload=audit_payload, report_payload=report)
    conversion_summary = _dict(report.get("conversion_review_summary"))
    conversion_fidelity = _dict(metadata.get("conversion_fidelity"))
    audit_route_plan = _dict(conversion_fidelity.get("audit_route_plan"))
    quality = _dict(report.get("quality_inspection") or conversion_fidelity.get("quality_inspection"))
    quality_summary = _dict(quality.get("summary"))

    reviewed_docx_path = _path_from(audit_payload, metadata, "reviewed_docx_path")
    wps_template_docx_path = _path_from(audit_payload, metadata, "wps_template_docx")
    audit_report_path = Path(report_path).expanduser() if report_path else _path_from(audit_payload, metadata, "audit_report_path")
    evidence_zip_path = _path_from(audit_payload, metadata, "evidence_zip_path")

    title_counts = _comment_title_counts(corrections)
    counts = _counts(
        metadata=metadata,
        conversion_fidelity=conversion_fidelity,
        audit_route_plan=audit_route_plan,
        conversion_summary=conversion_summary,
        quality=quality,
        quality_summary=quality_summary,
        corrections=corrections,
    )
    comment_quality = _comment_quality(corrections=corrections, title_counts=title_counts)
    artifacts = _artifact_summary(
        worker_output_path=worker_output_path,
        wps_template_docx_path=wps_template_docx_path,
        reviewed_docx_path=reviewed_docx_path,
        audit_report_path=audit_report_path,
        evidence_zip_path=evidence_zip_path,
    )

    summary: Dict[str, Any] = {
        "version": EVAL_BASELINE_VERSION,
        "ok": bool((worker_output or {}).get("ok", True)) if worker_output else True,
        "audit_id": audit_payload.get("audit_id") or metadata.get("audit_id") or report.get("audit_id"),
        "engine": metadata.get("engine"),
        "mode": metadata.get("mode"),
        "paths": {
            "worker_output_path": str(worker_output_path) if worker_output_path else "",
            "reviewed_docx_path": str(reviewed_docx_path) if reviewed_docx_path else "",
            "audit_report_path": str(audit_report_path) if audit_report_path else "",
            "evidence_zip_path": str(evidence_zip_path) if evidence_zip_path else "",
        },
        "counts": counts,
        "quality": {
            "overall_status": quality.get("overall_status") or quality_summary.get("overall_status"),
            "bottleneck_count": len(_list(quality.get("bottlenecks"))),
            "highest_risk_pages": _list(quality_summary.get("highest_risk_pages")),
            "recommended_next_actions": _list(quality.get("recommended_next_actions"))[:12],
        },
        "comment_quality": comment_quality,
        "artifacts": artifacts,
    }
    summary["checks"] = _checks(summary)
    summary["status"] = _overall_eval_status(summary["checks"])
    return summary


def _audit_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    audit = payload.get("audit")
    if isinstance(audit, Mapping):
        return dict(audit)
    return dict(payload)


def _metadata(*, audit_payload: Mapping[str, Any], report_payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    report_metadata = _dict((report_payload or {}).get("metadata"))
    audit_metadata = _dict(audit_payload.get("metadata"))
    merged = dict(audit_metadata)
    merged.update({key: value for key, value in report_metadata.items() if value is not None})
    return merged


def _corrections(*, audit_payload: Mapping[str, Any], report_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    corrections = report_payload.get("corrections")
    if isinstance(corrections, list):
        return [dict(item) for item in corrections if isinstance(item, Mapping)]
    corrections = audit_payload.get("corrections")
    if isinstance(corrections, list):
        return [dict(item) for item in corrections if isinstance(item, Mapping)]
    return []


def _counts(
    *,
    metadata: Mapping[str, Any],
    conversion_fidelity: Mapping[str, Any],
    audit_route_plan: Mapping[str, Any],
    conversion_summary: Mapping[str, Any],
    quality: Mapping[str, Any],
    quality_summary: Mapping[str, Any],
    corrections: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    table_cell = _dict(conversion_fidelity.get("table_cell_evidence"))
    specialist = _dict(conversion_fidelity.get("specialist_review_results"))
    qwen_gate = _dict(conversion_fidelity.get("qwen_gate"))
    qwen_vl_gate = _dict(conversion_fidelity.get("qwen_vl_gate"))
    image_text_vl = _dict(conversion_fidelity.get("image_text_vl"))
    page_text_qwen = _dict(conversion_fidelity.get("page_text_qwen_review"))
    high_risk = _dict(conversion_fidelity.get("high_risk_page_coverage"))
    full_content = _dict(conversion_fidelity.get("full_content_coverage"))

    return {
        "page_count": _int(_first_present(metadata.get("page_count"), conversion_fidelity.get("pdf_page_count"))),
        "comment_count": _int(_first_present(metadata.get("comment_count"), conversion_summary.get("comment_count"), len(corrections))),
        "correction_count": len(corrections),
        "diff_candidate_count": _int(_first_present(conversion_fidelity.get("diff_candidate_count"), conversion_summary.get("candidate_count"))),
        "route_counts": _dict(conversion_fidelity.get("route_counts")),
        "document_source_type": audit_route_plan.get("document_source_type"),
        "canonical_page_type_counts": _dict(audit_route_plan.get("canonical_page_type_counts")),
        "pdf_source_type_counts": _dict(audit_route_plan.get("pdf_source_type_counts")),
        "recognition_strategy_counts": _dict(audit_route_plan.get("recognition_strategy_counts")),
        "recognition_risk_counts": _dict(audit_route_plan.get("recognition_risk_counts")),
        "quality_first_page_count": _int(audit_route_plan.get("quality_first_page_count")),
        "low_confidence_page_count": _int(audit_route_plan.get("low_confidence_page_count")),
        "table_image_page_count": _int(audit_route_plan.get("table_image_page_count")),
        "scan_text_page_count": _int(audit_route_plan.get("scan_text_page_count")),
        "full_content_unresolved_count": _int(
            _first_present(
                quality_summary.get("full_content_unresolved_count"),
                full_content.get("unresolved_count"),
                full_content.get("unresolved_docx_count"),
            )
        ),
        "high_risk_page_coverage_unresolved_count": _int(
            _first_present(quality_summary.get("high_risk_page_coverage_unresolved_count"), high_risk.get("unresolved_count"))
        ),
        "table_cell_reviewed_count": _int(_first_present(table_cell.get("reviewed_cell_count"), table_cell.get("review_count"))),
        "table_cell_confirmable_count": _int(table_cell.get("confirmable_cell_count")),
        "table_cell_status_counts": _dict(table_cell.get("status_counts")),
        "table_cell_decision_counts": _dict(table_cell.get("decision_counts")),
        "specialist_result_count": _int(
            _first_present(specialist.get("result_count"), conversion_summary.get("specialist_review_result_count"))
        ),
        "specialist_confirmed_error_count": _int(
            _first_present(
                specialist.get("confirmed_error_count"),
                conversion_summary.get("specialist_review_confirmed_error_count"),
            )
        ),
        "specialist_deferred_count": _int(
            _first_present(specialist.get("deferred_count"), conversion_summary.get("specialist_review_deferred_count"))
        ),
        "qwen_text_reviewed_count": _int(qwen_gate.get("reviewed_candidate_count")),
        "qwen_text_available_count": _int(qwen_gate.get("available_count")),
        "page_text_qwen_reviewed_page_count": _int(page_text_qwen.get("reviewed_page_count")),
        "page_text_qwen_available_page_count": _int(page_text_qwen.get("available_page_count")),
        "page_text_qwen_exact_replacement_count": _int(page_text_qwen.get("exact_replacement_count")),
        "page_text_qwen_status_counts": _dict(page_text_qwen.get("status_counts")),
        "qwen_vl_candidate_count": _int(qwen_vl_gate.get("candidate_count")),
        "qwen_vl_available_count": _int(qwen_vl_gate.get("available_count")),
        "image_text_vl_reviewed_page_count": _int(image_text_vl.get("reviewed_page_count")),
        "image_text_vl_available_count": _int(image_text_vl.get("available_count")),
        "quality_bottleneck_count": len(_list(quality.get("bottlenecks"))),
    }


def _comment_title_counts(corrections: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in corrections:
        text = str(item.get("comment_text") or "")
        first_line = text.splitlines()[0].strip() if text else ""
        if not first_line:
            first_line = str(item.get("reason") or "unknown").strip()[:40] or "unknown"
        counter[first_line] += 1
    return dict(counter)


def _comment_quality(*, corrections: Sequence[Mapping[str, Any]], title_counts: Mapping[str, int]) -> Dict[str, Any]:
    mandatory_alpha_numeric: List[Dict[str, Any]] = []
    mandatory_incomplete_numeric: List[Dict[str, Any]] = []
    noisy_token_comments: List[Dict[str, Any]] = []
    suspected_count = 0
    confirmed_count = 0
    for item in corrections:
        comment_text = str(item.get("comment_text") or "")
        new_text = str(item.get("new_text") or "")
        old_text = str(item.get("old_text") or "")
        title = comment_text.splitlines()[0].strip() if comment_text else ""
        if "疑似" in title or "核查参考" in comment_text:
            suspected_count += 1
        if "确认错误" in title or "必须改为" in comment_text:
            confirmed_count += 1
        if ("必须改为" in comment_text or "确认错误" in title) and _looks_like_noisy_alpha_numeric(new_text, old_text):
            mandatory_alpha_numeric.append(_comment_issue(item))
        if ("必须改为" in comment_text or "确认错误" in title) and _looks_like_incomplete_confusable_numeric(new_text, old_text):
            mandatory_incomplete_numeric.append(_comment_issue(item))
        if _has_noisy_token(new_text) or _has_noisy_token(old_text):
            noisy_token_comments.append(_comment_issue(item))
    return {
        "title_counts": dict(title_counts),
        "confirmed_comment_count": confirmed_count,
        "suspected_comment_count": suspected_count,
        "mandatory_alpha_numeric_replacement_count": len(mandatory_alpha_numeric),
        "mandatory_alpha_numeric_replacements": mandatory_alpha_numeric[:20],
        "mandatory_incomplete_numeric_replacement_count": len(mandatory_incomplete_numeric),
        "mandatory_incomplete_numeric_replacements": mandatory_incomplete_numeric[:20],
        "noisy_token_comment_count": len(noisy_token_comments),
        "noisy_token_comments": noisy_token_comments[:20],
    }


def _comment_issue(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "page_no": item.get("page_no"),
        "old_text": item.get("old_text"),
        "new_text": item.get("new_text"),
        "comment_title": str(item.get("comment_text") or "").splitlines()[0].strip(),
    }


def _looks_like_noisy_alpha_numeric(new_text: str, old_text: str) -> bool:
    compact_new = _compact(new_text)
    compact_old = _compact(old_text)
    if not re.search(r"[A-Za-z]", compact_new) or not re.search(r"\d", compact_new):
        return False
    if _has_expected_identifier_shape(compact_new):
        return False
    if compact_old and _digit_punctuation_ratio(compact_old) >= 0.75:
        return True
    return _digit_punctuation_ratio(compact_new) >= 0.70


def _looks_like_incomplete_confusable_numeric(new_text: str, old_text: str) -> bool:
    compact_new = _compact(new_text).replace(",", "").replace("，", "")
    compact_old = _compact(old_text)
    if not re.fullmatch(r"\d{4,8}", compact_new):
        return False
    if "." in compact_new or "." in compact_old:
        return False
    if not re.search(r"[A-Za-z]", compact_old) or not re.search(r"\d", compact_old):
        return False
    simple_confusable_numeric = bool(re.fullmatch(r"[BCDEGILOQSZbcdegiloqsz]\d{3,7}", compact_old))
    if _has_expected_identifier_shape(compact_old) and not simple_confusable_numeric:
        return False
    old_key = _confusable_digit_key(compact_old)
    return bool(old_key and old_key == compact_new)


def _confusable_digit_key(value: str) -> str:
    mapping = str.maketrans(
        {
            "B": "8",
            "b": "8",
            "C": "0",
            "c": "0",
            "D": "0",
            "d": "0",
            "E": "6",
            "e": "6",
            "G": "6",
            "g": "6",
            "I": "1",
            "i": "1",
            "L": "1",
            "l": "1",
            "O": "0",
            "o": "0",
            "Q": "0",
            "q": "0",
            "S": "5",
            "s": "5",
            "Z": "2",
            "z": "2",
        }
    )
    return re.sub(r"\D+", "", str(value or "").translate(mapping))


def _has_expected_identifier_shape(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{1,8}[-_/]?[A-Za-z0-9][-_/A-Za-z0-9]{2,}", value))


def _has_noisy_token(value: str) -> bool:
    compact = _compact(value)
    if not compact:
        return False
    if _has_expected_identifier_shape(compact):
        return False
    return bool(re.search(r"\d[A-Za-z]\d", compact) or re.search(r"\d+[A-Za-z]+[.,]\d+", compact))


def _digit_punctuation_ratio(value: str) -> float:
    if not value:
        return 0.0
    allowed = sum(1 for char in value if char.isdigit() or char in ".,，。:：-_/()（）%+")
    return allowed / max(1, len(value))


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _artifact_summary(
    *,
    worker_output_path: Optional[Path],
    wps_template_docx_path: Optional[Path],
    reviewed_docx_path: Optional[Path],
    audit_report_path: Optional[Path],
    evidence_zip_path: Optional[Path],
) -> Dict[str, Any]:
    evidence_zip_entries: List[str] = []
    if evidence_zip_path and evidence_zip_path.exists():
        try:
            with zipfile.ZipFile(evidence_zip_path) as archive:
                evidence_zip_entries = archive.namelist()
        except Exception:
            evidence_zip_entries = []
    missing_core_entries = [entry for entry in CORE_EVIDENCE_ZIP_ENTRIES if entry not in set(evidence_zip_entries)]
    return {
        "worker_output": _file_stat(worker_output_path),
        "wps_template_docx": _file_stat(wps_template_docx_path),
        "reviewed_docx": _file_stat(reviewed_docx_path),
        "docx_text_integrity": _docx_text_integrity(
            original_path=wps_template_docx_path,
            reviewed_path=reviewed_docx_path,
        ),
        "audit_report": _file_stat(audit_report_path),
        "evidence_zip": {
            **_file_stat(evidence_zip_path),
            "entry_count": len(evidence_zip_entries),
            "required_entries": list(CORE_EVIDENCE_ZIP_ENTRIES),
            "missing_required_entries": missing_core_entries,
        },
    }


def _file_stat(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "size_bytes": 0}
    expanded = Path(path).expanduser()
    exists = expanded.exists()
    return {
        "path": str(expanded),
        "exists": exists,
        "size_bytes": expanded.stat().st_size if exists and expanded.is_file() else 0,
    }


def _docx_text_integrity(*, original_path: Optional[Path], reviewed_path: Optional[Path]) -> Dict[str, Any]:
    if original_path is None or reviewed_path is None:
        return {"checked": False, "reason": "missing_original_or_reviewed_path"}
    original = Path(original_path).expanduser()
    reviewed = Path(reviewed_path).expanduser()
    if not original.exists() or not reviewed.exists():
        return {
            "checked": False,
            "reason": "docx_missing",
            "original_exists": original.exists(),
            "reviewed_exists": reviewed.exists(),
        }
    try:
        original_parts = _docx_visible_text_parts(original)
        reviewed_parts = _docx_visible_text_parts(reviewed)
        comment_counts = _docx_comment_counts(reviewed)
        return {
            "checked": True,
            "text_equal": original_parts == reviewed_parts,
            "original_text_part_count": len(original_parts),
            "reviewed_text_part_count": len(reviewed_parts),
            "comment_range_start_count": comment_counts["comment_range_start_count"],
            "comment_reference_count": comment_counts["comment_reference_count"],
            "comments_part_exists": comment_counts["comments_part_exists"],
        }
    except Exception as exc:
        return {"checked": False, "reason": f"docx_integrity_error:{exc}"}


def _docx_visible_text_parts(path: Path) -> List[tuple[str, List[str]]]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rows: List[tuple[str, List[str]]] = []
    with zipfile.ZipFile(path) as archive:
        part_names = [
            name
            for name in archive.namelist()
            if name.startswith("word/")
            and name.endswith(".xml")
            and not name.startswith("word/comments")
            and not name.startswith("word/_rels/")
        ]
        for part_name in sorted(part_names):
            try:
                root = ET.fromstring(archive.read(part_name))
            except Exception:
                continue
            texts = [node.text or "" for node in root.findall(".//w:t", ns)]
            if texts:
                rows.append((part_name, texts))
    return rows


def _docx_comment_counts(path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        document_xml = archive.read("word/document.xml").decode("utf-8", errors="ignore") if "word/document.xml" in names else ""
        return {
            "comments_part_exists": "word/comments.xml" in names,
            "comment_range_start_count": document_xml.count("commentRangeStart"),
            "comment_reference_count": document_xml.count("commentReference"),
        }


def _checks(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    counts = _dict(summary.get("counts"))
    artifacts = _dict(summary.get("artifacts"))
    comment_quality = _dict(summary.get("comment_quality"))
    evidence_zip = _dict(artifacts.get("evidence_zip"))
    docx_text_integrity = _dict(artifacts.get("docx_text_integrity"))
    text_integrity_checked = bool(docx_text_integrity.get("checked"))
    checks = [
        _check("worker_output_ok", bool(summary.get("ok")), "fail"),
        _check("engine_is_v4", summary.get("engine") == "v4", "fail"),
        _check("comment_count_matches_corrections", counts.get("comment_count") == counts.get("correction_count"), "warn"),
        _check("reviewed_docx_exists", bool(_dict(artifacts.get("reviewed_docx")).get("exists")), "fail"),
        _check(
            "reviewed_docx_text_unchanged",
            (not text_integrity_checked) or bool(docx_text_integrity.get("text_equal")),
            "fail",
        ),
        _check("audit_report_exists", bool(_dict(artifacts.get("audit_report")).get("exists")), "fail"),
        _check("evidence_zip_exists", bool(evidence_zip.get("exists")), "fail"),
        _check("evidence_zip_has_core_payloads", not evidence_zip.get("missing_required_entries"), "fail"),
        _check("no_mandatory_alpha_numeric_replacements", _int(comment_quality.get("mandatory_alpha_numeric_replacement_count")) == 0, "fail"),
        _check("no_mandatory_incomplete_numeric_replacements", _int(comment_quality.get("mandatory_incomplete_numeric_replacement_count")) == 0, "fail"),
        _check(
            "quality_inspection_passable",
            str(_dict(summary.get("quality")).get("overall_status") or "")
            in {"", "no_actionable_candidates", "usable_with_comments", "usable_with_review_queue"},
            "fail",
        ),
        _check("quality_metrics_present", counts.get("full_content_unresolved_count") is not None, "warn"),
        _check("model_gate_metrics_present", counts.get("qwen_text_reviewed_count") is not None, "warn"),
        _check("page_taxonomy_metrics_present", bool(counts.get("canonical_page_type_counts")), "warn"),
    ]
    return checks


def _check(name: str, passed: bool, severity: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity}


def _overall_eval_status(checks: Sequence[Mapping[str, Any]]) -> str:
    if any(not check.get("passed") and check.get("severity") == "fail" for check in checks):
        return "failed"
    if any(not check.get("passed") for check in checks):
        return "needs_attention"
    return "passed"


def _path_from(audit_payload: Mapping[str, Any], metadata: Mapping[str, Any], key: str) -> Optional[Path]:
    raw = audit_payload.get(key) or metadata.get(key)
    return Path(str(raw)).expanduser() if raw else None


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)) else []


def _int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
