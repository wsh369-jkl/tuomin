"""One-shot worker for prepare/anonymize/export."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.anonymization_strategy import DEFAULT_ANONYMIZATION_STRATEGY, get_anonymization_strategy_profile
from app.core.config import desensitize_mode_context, settings
from app.core.document_workflow import LARGE_DOCUMENT_TEXT_THRESHOLD, classify_document_workflow
from app.core.runtime_security import ensure_private_directory, ensure_private_file
from app.engine.desensitization_engine import get_engine
from app.processors.document_exporter import DocumentExporter
from app.services.coverage_first.final_export import build_coverage_first_final_export_bundle
from app.processors.docx_merge import merge_docx_documents
from app.processors.docx_xml_utils import extract_docx_text


LARGE_DOCUMENT_GROUP_PAGE_COUNT = 10
GLOBAL_SUBJECT_REPLACEMENT_FAMILIES = {
    "person",
    "organization",
    "alias",
    "bank",
    "project",
    "location",
    "address",
    "court",
}
LARGE_DOCUMENT_MODEL_REVIEW_MAX_CANDIDATES = 24
LARGE_DOCUMENT_MODEL_REVIEW_MAX_CONTEXT_CHARS = 12000
LARGE_DOCUMENT_MODEL_REVIEW_MAX_WINDOWS_PER_FAMILY = 80
LARGE_DOCUMENT_RULE_REVIEW_MAX_BUCKET_CANDIDATES = 80
LARGE_DOCUMENT_RULE_REVIEW_MAX_EDGES_PER_FAMILY = 2000
LOCAL_CANONICAL_KEY_PATTERN = re.compile(
    r"^(?:ORG|PERSON|PROJECT|LOCATION|ADDRESS|COURT|BANK|ALIAS|COMPANY)_\d+$"
)
LARGE_DOCUMENT_SUBJECT_CARD_KEY_PREFIX = "GLOBAL_SUBJECT_CARD_"
LARGE_DOCUMENT_MODEL_SUBJECT_KEY_PREFIX = "GLOBAL_MODEL_SUBJECT::"


def _write_progress(progress_path: str | None, payload: Dict[str, Any]) -> None:
    if not progress_path:
        return
    path = Path(progress_path)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    ensure_private_file(path)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_subject_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = text.strip("：:，,。.；;（）()《》<>“”\"'`")
    return text


def _normalize_replacement(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return ""
    return text


def _is_local_generated_canonical_key(value: Any) -> bool:
    key = str(value or "").strip()
    if not key:
        return False
    return bool(
        LOCAL_CANONICAL_KEY_PATTERN.match(key)
        or key.startswith("ORG_OCC_")
        or key.startswith("ORG_LEDGER_")
        or key.startswith("ORG_SHORT::")
        or key.startswith(LARGE_DOCUMENT_SUBJECT_CARD_KEY_PREFIX)
    )


def _entity_family(contextual_service: Any, entity: Dict[str, Any]) -> str:
    entity_type = str(entity.get("type") or "").strip().upper()
    families = getattr(contextual_service, "GROUP_FAMILIES", {}) or {}
    return str(families.get(entity_type) or "").strip()


def _global_subject_family(contextual_service: Any, entity: Dict[str, Any]) -> str:
    family = _entity_family(contextual_service, entity)
    if family != "alias":
        return family

    metadata = dict(entity.get("metadata") or {})
    alias_evidence = any(
        _normalize_subject_token(metadata.get(key))
        for key in (
            "canonical",
            "definition_full_text",
            "definition_alias",
            "memory_primary_text",
            "identity_surface",
        )
    )
    canonical_role = str(entity.get("canonical_role") or metadata.get("canonical_role") or "").strip().upper()
    if alias_evidence or canonical_role in {"PARTY_A", "PARTY_B", "PARTY_C", "ORGANIZATION"}:
        return "organization"
    return ""


def _normalize_large_document_pages_from_structure(
    *,
    text: str,
    source_structure: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(source_structure, dict):
        return []
    raw_pages = source_structure.get("pages")
    if not isinstance(raw_pages, list) or len(raw_pages) <= 1:
        return []

    pages: List[Dict[str, Any]] = []
    cursor = 0
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            return []

        page_text = str(raw_page.get("text") or "")
        start = _coerce_int(raw_page.get("start"), -1)
        end = _coerce_int(raw_page.get("end"), -1)
        if (start < 0 or end < start) and page_text:
            found_at = text.find(page_text, cursor)
            if found_at < 0:
                found_at = text.find(page_text)
            if found_at >= 0:
                start = found_at
                end = found_at + len(page_text)
        if start < 0 or end < start or end > len(text):
            return []

        page_number = _coerce_int(raw_page.get("page_number"), index)
        if page_number <= 0:
            page_number = index
        pages.append(
            {
                "page_number": page_number,
                "text": text[start:end],
                "start": start,
                "end": end,
                "raw": dict(raw_page),
            }
        )
        cursor = max(cursor, end)

    return pages


def _build_large_document_page_groups(
    *,
    text: str,
    source_structure: Any,
    group_page_count: int = LARGE_DOCUMENT_GROUP_PAGE_COUNT,
) -> List[Dict[str, Any]]:
    pages = _normalize_large_document_pages_from_structure(
        text=text,
        source_structure=source_structure,
    )
    if len(pages) <= 1:
        return []

    groups: List[Dict[str, Any]] = []
    max_pages = max(1, int(group_page_count or LARGE_DOCUMENT_GROUP_PAGE_COUNT))
    max_text_length = max(1, int(LARGE_DOCUMENT_TEXT_THRESHOLD) - 1)
    current_pages: List[Dict[str, Any]] = []

    def append_group(group_pages: List[Dict[str, Any]]) -> None:
        if not group_pages:
            return
        group_start = int(group_pages[0]["start"])
        group_end = int(group_pages[-1]["end"])
        groups.append(
            {
                "index": len(groups),
                "page_numbers": [int(page["page_number"]) for page in group_pages],
                "pages": [dict(page) for page in group_pages],
                "start": group_start,
                "end": group_end,
                "text": text[group_start:group_end],
                "page_count": len(group_pages),
            }
        )

    for page in pages:
        page_text_length = len(str(page.get("text") or ""))
        if page_text_length > max_text_length:
            raise RuntimeError(
                "large_document_single_page_exceeds_default_text_threshold:"
                f"{int(page.get('page_number') or 0)}"
            )

        candidate_pages = [*current_pages, page]
        candidate_start = int(candidate_pages[0]["start"])
        candidate_end = int(candidate_pages[-1]["end"])
        candidate_length = len(text[candidate_start:candidate_end])
        if current_pages and (
            len(candidate_pages) > max_pages
            or candidate_length > max_text_length
        ):
            append_group(current_pages)
            current_pages = [page]
        else:
            current_pages = candidate_pages

    append_group(current_pages)
    return groups


def _project_entities_to_group(
    *,
    entities: List[Dict[str, Any]],
    group: Dict[str, Any],
) -> List[Dict[str, Any]]:
    group_start = int(group.get("start", 0))
    group_end = int(group.get("end", group_start))
    group_text = str(group.get("text") or "")
    local_entities: List[Dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        start = _coerce_int(entity.get("start"), -1)
        end = _coerce_int(entity.get("end"), -1)
        if start < group_start or end > group_end or end <= start:
            continue
        item = dict(entity)
        item["start"] = start - group_start
        item["end"] = end - group_start
        if 0 <= item["start"] < item["end"] <= len(group_text):
            item["text"] = group_text[item["start"]:item["end"]]
        local_entities.append(item)
    return local_entities


def _restore_group_entities_to_global(
    *,
    entities: List[Dict[str, Any]],
    group: Dict[str, Any],
    full_text: str,
) -> List[Dict[str, Any]]:
    group_start = int(group.get("start", 0))
    group_index = int(group.get("index", 0))
    restored: List[Dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        local_start = _coerce_int(entity.get("start"), -1)
        local_end = _coerce_int(entity.get("end"), -1)
        if local_start < 0 or local_end <= local_start:
            continue
        start = group_start + local_start
        end = group_start + local_end
        if start < 0 or end > len(full_text) or end <= start:
            continue
        item = dict(entity)
        metadata = dict(item.get("metadata") or {})
        metadata["large_document_group_index"] = group_index
        metadata["large_document_group_pages"] = list(group.get("page_numbers") or [])
        item["metadata"] = metadata
        item["start"] = start
        item["end"] = end
        item["text"] = full_text[start:end]
        restored.append(item)
    return restored


def _offset_entities_for_source_text(
    *,
    entities: List[Dict[str, Any]],
    offset: int,
) -> List[Dict[str, Any]]:
    if offset <= 0:
        return [dict(entity) for entity in entities if isinstance(entity, dict)]
    remapped: List[Dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        item = dict(entity)
        start = _coerce_int(item.get("start"), -1)
        end = _coerce_int(item.get("end"), -1)
        if start >= 0 and end >= start:
            item["start"] = start + offset
            item["end"] = end + offset
        remapped.append(item)
    return remapped


def _large_document_task_slug(payload: Dict[str, Any]) -> str:
    raw_task_id = str(payload.get("task_id") or "large-document").strip() or "large-document"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_task_id).strip("._")
    return slug or "large-document"


def _attach_docx_export_quality_metadata(
    quality_metadata: Any,
    *,
    source_metadata: Any,
    export_result: Any,
) -> Dict[str, Any]:
    metadata = dict(quality_metadata) if isinstance(quality_metadata, dict) else {}
    if not isinstance(export_result, dict) or str(export_result.get("file_type") or "").lower() != "docx":
        return metadata

    source_meta = source_metadata if isinstance(source_metadata, dict) else {}
    audit = source_meta.get("docx_text_coverage_audit") if isinstance(source_meta, dict) else None
    if not isinstance(audit, dict):
        audit = {}

    flags = list(metadata.get("quality_flags") or [])
    blocking = False

    precise_complete = export_result.get("docx_precise_rewrite_complete")
    if precise_complete is False:
        flags.append("docx_precise_rewrite_incomplete")
        blocking = True

    def audit_count(name: str) -> int:
        return _coerce_int(audit.get(name), 0)

    audit_count_fields = {
        "docx_total_text_node_count": "total_text_node_count",
        "docx_covered_text_node_count": "covered_text_node_count",
        "docx_uncovered_text_node_count": "uncovered_text_node_count",
        "docx_unhandled_text_part_count": "unhandled_text_part_count",
        "docx_unknown_text_part_count": "unknown_text_part_count",
        "docx_hidden_text_node_count": "hidden_text_node_count",
        "docx_field_instruction_node_count": "field_instruction_node_count",
        "docx_review_required_text_node_count": "review_required_text_node_count",
    }
    for output_key, audit_key in audit_count_fields.items():
        if audit_key in audit:
            metadata[output_key] = audit_count(audit_key)

    if audit_count("unhandled_text_part_count") > 0:
        flags.append("docx_unhandled_text_parts")
        blocking = True
    if audit_count("unknown_text_part_count") > 0:
        flags.append("docx_unknown_text_parts")
        blocking = True
    if audit_count("field_instruction_node_count") > 0:
        flags.append("docx_field_instruction_text_present")
        blocking = True
    if audit_count("hidden_text_node_count") > 0:
        flags.append("docx_hidden_text_present")
        blocking = True
    if audit_count("review_required_text_node_count") > 0:
        flags.append("docx_secondary_story_review_required")
        blocking = True

    metadata["docx_rewrite_method"] = export_result.get("docx_rewrite_method")
    metadata["docx_precise_rewrite_complete"] = export_result.get("docx_precise_rewrite_complete")
    metadata["docx_range_rewrite_applied_count"] = _coerce_int(
        export_result.get("docx_range_rewrite_applied_count"),
        0,
    )
    metadata["docx_range_rewrite_required_count"] = _coerce_int(
        export_result.get("docx_range_rewrite_required_count"),
        0,
    )
    metadata["docx_range_rewrite_preflight_blocked_count"] = _coerce_int(
        export_result.get("docx_range_rewrite_preflight_blocked_count"),
        0,
    )
    metadata["docx_range_rewrite_unapplied_count"] = _coerce_int(
        export_result.get("docx_range_rewrite_unapplied_count"),
        0,
    )
    metadata["docx_chinese_inline_space_normalized"] = bool(
        export_result.get("docx_chinese_inline_space_normalized")
    )
    metadata["docx_chinese_inline_space_removed_count"] = _coerce_int(
        export_result.get("docx_chinese_inline_space_removed_count"),
        0,
    )
    if metadata["docx_range_rewrite_unapplied_count"] > 0:
        flags.append("docx_range_rewrite_unapplied")
        blocking = True
    if metadata["docx_range_rewrite_preflight_blocked_count"] > 0:
        flags.append("docx_range_rewrite_preflight_blocked")
        blocking = True
    metadata["docx_post_rewrite_residual_count"] = _coerce_int(
        export_result.get("docx_post_rewrite_residual_count"),
        0,
    )
    if metadata["docx_post_rewrite_residual_count"] > 0:
        flags.append("docx_post_rewrite_residual")
        blocking = True
    residual_samples = export_result.get("docx_post_rewrite_residual_samples")
    if isinstance(residual_samples, list):
        metadata["docx_post_rewrite_residual_samples"] = residual_samples[:20]
    if audit:
        metadata["docx_text_coverage_audit"] = audit
    if blocking:
        metadata["requires_manual_review"] = True
        metadata["quality_gate_passed"] = False
        existing_reason = str(metadata.get("quality_gate_reason") or "").strip()
        docx_reason = "DOCX 存在未覆盖文本部件、隐藏/字段文本或精确回写未完成，需复核后才能标记质量通过。"
        metadata["quality_gate_reason"] = (
            f"{existing_reason}; {docx_reason}" if existing_reason else docx_reason
        )
    metadata["quality_flags"] = sorted(set(str(flag) for flag in flags if str(flag).strip()))
    return metadata


def _attach_directory_quality_metadata(metadata: Any, entities: Any) -> Dict[str, Any]:
    quality = dict(metadata) if isinstance(metadata, dict) else {}
    entity_list = [entity for entity in entities or [] if isinstance(entity, dict)]
    flags = list(quality.get("quality_flags") or [])
    by_subject: Dict[str, set[str]] = {}
    by_replacement: Dict[str, set[str]] = {}
    missing_replacement_count = 0
    for entity in entity_list:
        source_text = str(entity.get("text") or "").strip()
        replacement = _normalize_replacement(entity.get("replacement"))
        entity_type = str(entity.get("type") or entity.get("entity_type") or "").strip()
        metadata_obj = entity.get("metadata") if isinstance(entity.get("metadata"), dict) else {}
        subject_key = str(
            metadata_obj.get("global_large_document_subject_id")
            or metadata_obj.get("canonical_key")
            or f"{entity_type}:{source_text}"
        ).strip()
        if source_text and not replacement:
            missing_replacement_count += 1
        if subject_key and replacement:
            by_subject.setdefault(subject_key, set()).add(replacement)
        if replacement:
            by_replacement.setdefault(replacement, set()).add(subject_key or f"{entity_type}:{source_text}")

    subject_multi_replacement_count = sum(1 for values in by_subject.values() if len(values) > 1)
    replacement_reused_by_multi_subject_count = sum(1 for values in by_replacement.values() if len(values) > 1)
    quality["final_directory_missing_replacement_count"] = missing_replacement_count
    quality["final_directory_subject_multi_replacement_count"] = subject_multi_replacement_count
    quality["final_directory_replacement_reused_by_multi_subject_count"] = replacement_reused_by_multi_subject_count

    if missing_replacement_count > 0:
        flags.append("final_directory_missing_replacement")
    if subject_multi_replacement_count > 0:
        flags.append("final_directory_subject_multi_replacement")
    if replacement_reused_by_multi_subject_count > 0:
        flags.append("final_directory_replacement_reused_by_multi_subject")
    if any(flag.startswith("final_directory_") for flag in flags):
        quality["requires_manual_review"] = True
        quality["quality_gate_passed"] = False
        existing_reason = str(quality.get("quality_gate_reason") or "").strip()
        directory_reason = "最终目录存在 replacement 缺失或主体/代号一致性冲突，不能导出为质量通过结果。"
        quality["quality_gate_reason"] = (
            f"{existing_reason}; {directory_reason}" if existing_reason else directory_reason
        )
    quality["quality_flags"] = sorted(set(str(flag) for flag in flags if str(flag).strip()))
    return quality


def _attach_coverage_first_final_export_metadata(
    metadata: Any,
    coverage_first_final_export: Any,
) -> Dict[str, Any]:
    quality = dict(metadata) if isinstance(metadata, dict) else {}
    if not isinstance(coverage_first_final_export, dict) or not coverage_first_final_export.get("enabled"):
        return quality
    summary = dict(coverage_first_final_export.get("summary") or {})
    flags = list(quality.get("quality_flags") or [])
    quality["coverage_first_final_export_used"] = True
    for key in (
        "final_entity_input_count",
        "final_desensitized_entity_input_count",
        "final_directory_subject_count",
        "final_directory_occurrence_count",
        "final_missing_directory_entity_count",
        "final_mapping_entity_count",
        "final_rewrite_entry_count",
        "final_blocked_rewrite_entry_count",
        "final_replacement_reused_by_multi_subject_count",
        "final_subject_multi_replacement_count",
        "qwen_discovery_final_entity_count",
        "qwen_discovery_desensitized_entity_count",
        "qwen_discovery_directory_row_count",
        "qwen_discovery_directory_occurrence_count",
        "qwen_discovery_mapping_entity_count",
        "qwen_discovery_rewrite_entry_count",
    ):
        quality[f"coverage_first_{key}"] = _coerce_int(summary.get(key), 0)
    quality["coverage_first_final_export_ready"] = bool(summary.get("final_export_ready"))
    failure_counts = summary.get("final_rewrite_failure_counts")
    if isinstance(failure_counts, dict):
        quality["coverage_first_final_rewrite_failure_counts"] = dict(failure_counts)

    if not quality["coverage_first_final_export_ready"]:
        flags.append("coverage_first_final_export_not_ready")
    if quality["coverage_first_final_blocked_rewrite_entry_count"] > 0:
        flags.append("coverage_first_final_blocked_rewrite_entry")
    if quality["coverage_first_final_replacement_reused_by_multi_subject_count"] > 0:
        flags.append("coverage_first_final_replacement_reused_by_multi_subject")
    if quality["coverage_first_final_subject_multi_replacement_count"] > 0:
        flags.append("coverage_first_final_subject_multi_replacement")
    if quality["coverage_first_final_missing_directory_entity_count"] > 0:
        flags.append("coverage_first_final_missing_directory_entity")
    if (
        quality["coverage_first_qwen_discovery_desensitized_entity_count"]
        > quality["coverage_first_qwen_discovery_mapping_entity_count"]
    ):
        flags.append("coverage_first_final_qwen_discovery_mapping_missing")

    if any(str(flag).startswith("coverage_first_final_") for flag in flags):
        quality["requires_manual_review"] = True
        quality["quality_gate_passed"] = False
        existing_reason = str(quality.get("quality_gate_reason") or "").strip()
        reason = "coverage-first 最终目录或精确回写账本未闭合，不能导出为质量通过结果。"
        quality["quality_gate_reason"] = f"{existing_reason}; {reason}" if existing_reason else reason
    quality["quality_flags"] = sorted(set(str(flag) for flag in flags if str(flag).strip()))
    return quality


def _large_document_group_output_root(payload: Dict[str, Any]) -> Path:
    task_id = _large_document_task_slug(payload)
    root = Path(settings.OUTPUT_DIR) / f"{task_id}_large_document_groups"
    if root.exists():
        shutil.rmtree(root)
    ensure_private_directory(root)
    return root


def _large_document_group_dir(root: Path, group: Dict[str, Any]) -> Path:
    group_index = int(group.get("index", 0))
    page_numbers = [int(value) for value in (group.get("page_numbers") or []) if isinstance(value, int)]
    if page_numbers:
        suffix = f"pages_{page_numbers[0]:04d}_{page_numbers[-1]:04d}"
    else:
        suffix = "pages_unknown"
    group_dir = root / f"group_{group_index + 1:04d}_{suffix}"
    ensure_private_directory(group_dir)
    return group_dir


def _write_group_source_file(
    *,
    payload: Dict[str, Any],
    group: Dict[str, Any],
    group_dir: Path,
) -> tuple[str, str]:
    source_path = Path(str(payload.get("source_path") or ""))
    source_suffix = source_path.suffix.lower()
    group_index = int(group.get("index", 0))
    source_stem = source_path.stem or str(payload.get("task_id") or "large_document")
    group_text = str(group.get("text") or "")

    if source_suffix == ".docx":
        if not source_path.exists():
            raise FileNotFoundError(f"large_document_group_source_missing: {source_path}")
        from app.processors.docx_merge import write_docx_body_slice

        group_source_path = group_dir / f"{source_stem}_group_{group_index + 1:04d}.docx"
        page_texts = [str(page.get("text") or "") for page in (group.get("pages") or []) if isinstance(page, dict)]
        write_docx_body_slice(
            source_path,
            group_source_path,
            group_text=group_text,
            page_texts=page_texts,
        )
        ensure_private_file(group_source_path)
        return str(group_source_path), group_source_path.name

    group_source_path = group_dir / f"{source_stem}_group_{group_index + 1:04d}.txt"
    group_source_path.write_text(group_text, encoding="utf-8")
    ensure_private_file(group_source_path)
    return str(group_source_path), group_source_path.name


def _move_group_export_file(
    *,
    path_value: Any,
    download_name: Any,
    group_dir: Path,
    fallback_name: str,
) -> str:
    source = Path(str(path_value or ""))
    if not source.exists():
        return ""
    target_name = str(download_name or "").strip() or fallback_name or source.name
    target = group_dir / target_name
    if source.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
    ensure_private_file(target)
    return str(target)


def _materialize_group_export_result(
    *,
    export_result: Dict[str, Any],
    group_dir: Path,
) -> Dict[str, Any]:
    result = dict(export_result)
    output_path = _move_group_export_file(
        path_value=export_result.get("output_path"),
        download_name=export_result.get("download_name"),
        group_dir=group_dir,
        fallback_name="anonymized_output",
    )
    if output_path:
        result["output_path"] = output_path

    mapping_output_path = _move_group_export_file(
        path_value=export_result.get("mapping_output_path"),
        download_name=export_result.get("mapping_download_name"),
        group_dir=group_dir,
        fallback_name="mapping.docx",
    )
    if mapping_output_path:
        result["mapping_output_path"] = mapping_output_path
    return result


def _group_use_custom(payload: Dict[str, Any]) -> bool:
    if "use_custom" in payload:
        return bool(payload.get("use_custom"))
    task_id = str(payload.get("task_id") or "").strip()
    if task_id:
        task_state_path = Path(settings.RUNTIME_ROOT) / "task_state" / f"{task_id}.json"
        try:
            task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
        except Exception:
            task_state = {}
        config = task_state.get("config") if isinstance(task_state, dict) else None
        if isinstance(config, dict) and "use_custom" in config:
            return bool(config.get("use_custom"))
    return True


def _large_document_group_task_id(payload: Dict[str, Any], group: Dict[str, Any]) -> str:
    return f"{_large_document_task_slug(payload)}_group_{int(group.get('index', 0)) + 1:04d}"


async def _run_group_source_through_default_line(
    *,
    payload: Dict[str, Any],
    group: Dict[str, Any],
    group_source_path: str,
    group_source_name: str,
) -> Dict[str, Any]:
    from app.api import desensitize as desensitize_api

    group_task_id = _large_document_group_task_id(payload, group)
    selected_desensitize_mode = str(payload.get("desensitize_mode") or settings.get_effective_desensitize_mode())
    group_task = {
        "task_id": group_task_id,
        "filename": group_source_name,
        "file_path": group_source_path,
        "created_at": datetime.now(),
        "suffix": Path(group_source_name).suffix.lower() or Path(group_source_path).suffix.lower(),
        "status": "queued",
        "progress": 5,
        "message": "文件已上传，正在准备识别任务...",
        "config": {
            "use_llm": bool(payload.get("use_llm")),
            "use_custom": _group_use_custom(payload),
            "llm_model": payload.get("llm_model"),
            "anonymization_strategy": payload.get("anonymization_strategy"),
            "desensitize_mode": selected_desensitize_mode,
        },
    }
    desensitize_api.tasks[group_task_id] = group_task

    try:
        await desensitize_api._run_analysis_task(group_task_id)
        analyzed_task = desensitize_api.tasks.get(group_task_id)
        if not isinstance(analyzed_task, dict):
            raise RuntimeError(f"large_document_group_default_task_missing_after_analysis:{group_task_id}")
        if str(analyzed_task.get("status") or "").strip().lower() != "ready":
            detail = str(analyzed_task.get("error_message") or analyzed_task.get("message") or "")
            raise RuntimeError(f"large_document_group_default_analysis_failed:{group_task_id}:{detail}")

        process_request = {
            "task_id": group_task_id,
            "entities": list(analyzed_task.get("entities") or []),
            "operator_config": payload.get("operator_config"),
            "llm_model": payload.get("llm_model"),
            "anonymization_strategy": payload.get("anonymization_strategy"),
            "desensitize_mode": selected_desensitize_mode,
            "async_mode": False,
        }
        analyzed_task["pending_process_request"] = process_request
        await desensitize_api._run_process_task(group_task_id)
        completed_task = desensitize_api.tasks.get(group_task_id)
        if not isinstance(completed_task, dict):
            raise RuntimeError(f"large_document_group_default_task_missing_after_process:{group_task_id}")
        if str(completed_task.get("status") or "").strip().lower() != "completed":
            detail = str(completed_task.get("error_message") or completed_task.get("message") or "")
            raise RuntimeError(f"large_document_group_default_process_failed:{group_task_id}:{detail}")

        export_result = {
            "output_path": completed_task.get("output_path"),
            "download_name": completed_task.get("output_filename"),
            "file_type": completed_task.get("output_file_type"),
            "media_type": completed_task.get("output_media_type"),
            "preserves_format": bool(completed_task.get("preserves_format")),
            "warning": completed_task.get("export_warning"),
            "mapping_output_path": completed_task.get("mapping_output_path"),
            "mapping_download_name": completed_task.get("mapping_output_filename"),
            "mapping_file_type": completed_task.get("mapping_output_file_type"),
            "mapping_media_type": completed_task.get("mapping_output_media_type"),
        }
        return {
            "text": str(completed_task.get("text") or ""),
            "metadata": dict(completed_task.get("metadata") or {}),
            "structure": completed_task.get("structure"),
            "detected_entities": list(analyzed_task.get("entities") or []),
            "prepared_entities": list(completed_task.get("entities") or []),
            "quality_metadata": dict(completed_task.get("quality_metadata") or {}),
            "anonymized_text": str(completed_task.get("anonymized_text") or ""),
            "export_result": export_result,
            "statistics": completed_task.get("statistics") or {},
        }
    finally:
        desensitize_api.tasks.pop(group_task_id, None)
        desensitize_api.analysis_task_runners.pop(group_task_id, None)
        desensitize_api.process_task_runners.pop(group_task_id, None)
        if hasattr(desensitize_api, "_delete_task_state"):
            desensitize_api._delete_task_state(group_task_id)


def _write_large_document_group_manifest(
    *,
    payload: Dict[str, Any],
    group_root: Path,
    group_runs: List[Dict[str, Any]],
    quality_metadata: Dict[str, Any],
    final_export_result: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest_path = group_root / "manifest.json"
    manifest_payload = {
        "task_id": str(payload.get("task_id") or ""),
        "front_half_only": bool(quality_metadata.get("large_document_front_half_only")),
        "group_count": len(group_runs),
        "group_page_count": LARGE_DOCUMENT_GROUP_PAGE_COUNT,
        "execution_mode": quality_metadata.get("execution_mode"),
        "large_document_execution_mode": quality_metadata.get("large_document_execution_mode"),
        "global_subject_count": quality_metadata.get("global_subject_count"),
        "replacement_conflict_count": quality_metadata.get("replacement_conflict_count"),
        "final_export_result": final_export_result or {},
        "groups": group_runs,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_private_file(manifest_path)
    return manifest_path


def _create_large_document_group_archive(
    *,
    payload: Dict[str, Any],
    group_root: Path,
) -> tuple[str, str]:
    task_id = _large_document_task_slug(payload)
    archive_name = f"{task_id}_large_document_groups.zip"
    archive_path = Path(settings.OUTPUT_DIR) / archive_name
    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(group_root.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=str(file_path.relative_to(group_root.parent)))

    ensure_private_file(archive_path)
    return str(archive_path), archive_name


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _subject_tokens_for_entity(contextual_service: Any, entity: Dict[str, Any], family: str) -> List[str]:
    tokens: List[str] = []
    entity_text = _normalize_subject_token(entity.get("text"))
    if entity_text:
        tokens.append(f"{family}:text:{entity_text}")
        tokens.append(f"{family}:identity:{entity_text}")

    metadata = dict(entity.get("metadata") or {})
    for key in (
        "canonical",
        "definition_full_text",
        "definition_alias",
        "identity_surface",
        "residual_variant",
        "materialized_from",
        "memory_primary_text",
        "group_label",
    ):
        token = _normalize_subject_token(metadata.get(key))
        if token:
            tokens.append(f"{family}:evidence:{token}")
            tokens.append(f"{family}:identity:{token}")

    for key in ("group_label", "canonical", "definition_full_text"):
        token = _normalize_subject_token(entity.get(key))
        if token:
            tokens.append(f"{family}:evidence:{token}")
            tokens.append(f"{family}:identity:{token}")

    canonical_key = (
        str(entity.get("canonical_key") or metadata.get("canonical_key") or entity.get("group_id") or "").strip()
    )
    if canonical_key and not _is_local_generated_canonical_key(canonical_key):
        tokens.append(f"{family}:canonical:{canonical_key}")

    canonical_role = str(entity.get("canonical_role") or metadata.get("canonical_role") or "").strip().upper()
    if canonical_role in {"PARTY_A", "PARTY_B", "PARTY_C"}:
        tokens.append(f"{family}:role:{canonical_role}")

    if family == "organization":
        derive_aliases = getattr(contextual_service, "_derive_organization_identity_aliases", None)
        if callable(derive_aliases):
            for alias in derive_aliases(str(entity.get("text") or "")):
                token = _normalize_subject_token(alias)
                if token:
                    tokens.append(f"{family}:derived:{token}")
                    tokens.append(f"{family}:identity:{token}")
    elif family == "person":
        derive_aliases = getattr(contextual_service, "_derive_person_identity_aliases", None)
        if callable(derive_aliases):
            for alias in derive_aliases(str(entity.get("text") or "")):
                token = _normalize_subject_token(alias)
                if token:
                    tokens.append(f"{family}:derived:{token}")
                    tokens.append(f"{family}:identity:{token}")
    elif family == "project":
        derive_aliases = getattr(contextual_service, "_derive_project_identity_aliases", None)
        if callable(derive_aliases):
            for alias in derive_aliases(str(entity.get("text") or "")):
                token = _normalize_subject_token(alias)
                if token:
                    tokens.append(f"{family}:derived:{token}")
                    tokens.append(f"{family}:identity:{token}")

    return sorted(set(tokens))


def _build_large_document_identity_contexts(
    *,
    contextual_service: Any,
    text: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    build_context = getattr(contextual_service, "_build_minimal_context", None)
    contexts: List[Dict[str, Any]] = []
    for entity in entities:
        if callable(build_context):
            try:
                context = build_context(text, entity)
            except Exception:
                context = {}
        else:
            context = {}
        if not isinstance(context, dict):
            context = {}
        metadata = dict(entity.get("metadata") or {})
        if not context.get("canonical_role") and entity.get("canonical_role"):
            context["canonical_role"] = str(entity.get("canonical_role") or "")
        if not context.get("canonical_role") and metadata.get("canonical_role"):
            context["canonical_role"] = str(metadata.get("canonical_role") or "")
        if not context.get("label") and entity.get("context_label"):
            context["label"] = str(entity.get("context_label") or "")
        if not context.get("role") and entity.get("context_role"):
            context["role"] = str(entity.get("context_role") or "")
        contexts.append(context)
    return contexts


def _large_document_directory_pools(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
) -> Dict[str, List[int]]:
    pools: Dict[str, List[int]] = {}
    for index, entity in enumerate(entities):
        family = _global_subject_family(contextual_service, entity)
        if family not in GLOBAL_SUBJECT_REPLACEMENT_FAMILIES:
            continue
        if not _normalize_replacement(entity.get("replacement")):
            continue
        pools.setdefault(family, []).append(index)
    return pools


def _large_document_model_review_tokens(
    contextual_service: Any,
    *,
    entity: Dict[str, Any],
    family: str,
) -> List[str]:
    tokens = []
    for token in _subject_tokens_for_entity(contextual_service, entity, family):
        if any(marker in token for marker in (":identity:", ":derived:", ":canonical:", ":role:")):
            tokens.append(token)
    return sorted(set(tokens))


def _large_document_alias_values_from_tokens(tokens: List[str]) -> List[str]:
    values: List[str] = []
    for token in tokens:
        if ":identity:" in token or ":derived:" in token or ":evidence:" in token or ":text:" in token:
            value = token.rsplit(":", 1)[-1]
        else:
            continue
        value = _normalize_subject_token(value)
        if value:
            values.append(value)
    return sorted(set(values), key=lambda item: (-len(item), item))


def _large_document_card_alias_values(
    contextual_service: Any,
    card: Dict[str, Any],
    family: str,
) -> List[str]:
    metadata = dict(card.get("metadata") or {})
    aliases: set[str] = set()
    for value in [card.get("text"), *list(metadata.get("variants") or [])]:
        token = _normalize_subject_token(value)
        if token:
            aliases.add(token)

        if family == "organization":
            derive_aliases = getattr(contextual_service, "_derive_organization_identity_aliases", None)
        elif family == "person":
            derive_aliases = getattr(contextual_service, "_derive_person_identity_aliases", None)
        elif family == "project":
            derive_aliases = getattr(contextual_service, "_derive_project_identity_aliases", None)
        else:
            derive_aliases = None
        if callable(derive_aliases):
            try:
                aliases.update(_normalize_subject_token(alias) for alias in derive_aliases(str(value or "")))
            except Exception:
                pass

    for token in _large_document_model_review_tokens(
        contextual_service,
        entity=card,
        family=family,
    ):
        aliases.update(_large_document_alias_values_from_tokens([token]))

    return sorted({alias for alias in aliases if alias}, key=lambda item: (-len(item), item))


def _large_document_candidate_link_tokens(
    contextual_service: Any,
    *,
    entity: Dict[str, Any],
    family: str,
) -> List[str]:
    if family not in {"organization", "project", "bank", "court"}:
        return _large_document_model_review_tokens(
            contextual_service,
            entity=entity,
            family=family,
        )

    tokens = set(
        _large_document_model_review_tokens(
            contextual_service,
            entity=entity,
            family=family,
        )
    )
    alias_values = _large_document_alias_values_from_tokens(list(tokens))
    entity_text = _normalize_subject_token(entity.get("text"))
    if entity_text:
        alias_values.append(entity_text)

    for alias in sorted(set(alias_values), key=lambda item: (-len(item), item)):
        if len(alias) >= 2:
            tokens.add(f"{family}:alias_candidate:{alias}")
        companyish = alias.removesuffix("公司")
        if len(companyish) >= 2:
            tokens.add(f"{family}:alias_candidate:{companyish}")
        if family == "organization" and len(companyish) >= 4:
            tokens.add(f"{family}:alias_candidate:{companyish[:4]}")
            tokens.add(f"{family}:alias_candidate:{companyish[-4:]}")
        if family == "project" and len(alias) >= 4:
            tokens.add(f"{family}:alias_candidate:{alias[:6]}")
    return sorted(tokens)


def _large_document_rule_identity_tokens(
    contextual_service: Any,
    *,
    entity: Dict[str, Any],
    family: str,
) -> List[str]:
    tokens: List[str] = []
    metadata = dict(entity.get("metadata") or {})
    for token in _subject_tokens_for_entity(contextual_service, entity, family):
        if ":canonical:" in token or ":role:" in token:
            tokens.append(token)
            continue
        if ":identity:" not in token:
            continue
        raw_value = token.rsplit(":", 1)[-1]
        if not raw_value:
            continue
        if raw_value == _normalize_subject_token(entity.get("text")):
            tokens.append(token)
            continue
        for key in (
            "canonical",
            "definition_full_text",
            "definition_alias",
            "identity_surface",
            "memory_primary_text",
        ):
            if raw_value == _normalize_subject_token(metadata.get(key)):
                    tokens.append(token)
                    break
    return sorted(set(tokens))


def _large_document_strong_alias_values(
    contextual_service: Any,
    *,
    entity: Dict[str, Any],
    family: str,
) -> List[str]:
    if family not in {"organization", "project", "bank", "court"}:
        return []
    aliases = _large_document_alias_values_from_tokens(
        _large_document_candidate_link_tokens(
            contextual_service,
            entity=entity,
            family=family,
        )
    )
    text = _normalize_subject_token(entity.get("text"))
    if text:
        aliases.append(text)
        aliases.append(text.removesuffix("公司"))
    return sorted({alias for alias in aliases if len(alias) >= 2}, key=lambda item: (-len(item), item))


def _large_document_is_full_subject_name(
    contextual_service: Any,
    *,
    entity: Dict[str, Any],
    family: str,
) -> bool:
    text = _normalize_subject_token(entity.get("text"))
    if not text:
        return False
    if family == "organization":
        looks_like_company = getattr(contextual_service, "_looks_like_company_subject", None)
        if callable(looks_like_company):
            try:
                if bool(looks_like_company(text)):
                    return True
            except Exception:
                pass
        return len(text) >= 6 and any(token in text for token in ("公司", "集团", "中心", "银行", "院", "局", "委员会"))
    if family == "project":
        return len(text) >= 6 and any(token in text for token in ("项目", "工程", "标段"))
    if family in {"bank", "court"}:
        return len(text) >= 4
    return False


def _large_document_apply_unique_anchor_alias_merges(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    pool_indexes: List[int],
    family: str,
    candidate_indexes: List[int],
    local_pos_by_entity_index: Dict[int, int],
    uf: _UnionFind,
) -> int:
    if family not in {"organization", "project", "bank", "court"}:
        return 0

    alias_anchor_positions: Dict[str, set[int]] = {}
    for entity_index in pool_indexes:
        entity = entities[entity_index]
        if not _large_document_is_full_subject_name(
            contextual_service,
            entity=entity,
            family=family,
        ):
            continue
        pos = local_pos_by_entity_index[entity_index]
        for alias in _large_document_strong_alias_values(
            contextual_service,
            entity=entity,
            family=family,
        ):
            alias_anchor_positions.setdefault(alias, set()).add(uf.find(pos))

    unique_anchor_by_alias = {
        alias: next(iter(roots))
        for alias, roots in alias_anchor_positions.items()
        if len(roots) == 1
    }
    if not unique_anchor_by_alias:
        return 0

    merge_edges = 0
    for entity_index in pool_indexes:
        entity = entities[entity_index]
        pos = local_pos_by_entity_index[entity_index]
        entity_aliases = _large_document_strong_alias_values(
            contextual_service,
            entity=entity,
            family=family,
        )
        candidate_anchor_roots = {
            unique_anchor_by_alias[alias]
            for alias in entity_aliases
            if alias in unique_anchor_by_alias
        }
        candidate_anchor_roots.discard(uf.find(pos))
        if len(candidate_anchor_roots) != 1:
            continue

        anchor_root = next(iter(candidate_anchor_roots))
        anchor_member_pos = next(
            (
                member_pos
                for member_pos, member_entity_index in enumerate(candidate_indexes)
                if uf.find(member_pos) == anchor_root
            ),
            -1,
        )
        if anchor_member_pos < 0:
            continue
        anchor_entity_index = candidate_indexes[anchor_member_pos]
        if _large_document_entities_have_identity_conflict(
            contextual_service=contextual_service,
            left_entity=entities[anchor_entity_index],
            left_context=contexts[anchor_entity_index] if anchor_entity_index < len(contexts) else {},
            right_entity=entity,
            right_context=contexts[entity_index] if entity_index < len(contexts) else {},
        ):
            continue
        if not _large_document_can_union_components(
            contextual_service=contextual_service,
            entities=entities,
            candidate_indexes=candidate_indexes,
            uf=uf,
            left_pos=anchor_member_pos,
            right_pos=pos,
        ):
            continue
        if uf.find(anchor_member_pos) != uf.find(pos):
            merge_edges += 1
        uf.union(anchor_member_pos, pos)

    return merge_edges


def _chunk_large_document_indexes(indexes: List[int], size: int) -> List[List[int]]:
    if size <= 0:
        size = LARGE_DOCUMENT_MODEL_REVIEW_MAX_CANDIDATES
    return [indexes[offset : offset + size] for offset in range(0, len(indexes), size)]


def _large_document_model_review_batches(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    family: str,
    global_indexes: List[int],
    max_candidates: int = LARGE_DOCUMENT_MODEL_REVIEW_MAX_CANDIDATES,
) -> List[List[int]]:
    ordered_indexes = sorted(
        global_indexes,
        key=lambda index: (
            _coerce_int(entities[index].get("start"), 10**12),
            _coerce_int(entities[index].get("end"), 10**12),
            index,
        ),
    )
    if len(ordered_indexes) < 2:
        return []

    token_buckets: Dict[str, List[int]] = {}
    for global_index in ordered_indexes:
        for token in _large_document_candidate_link_tokens(
            contextual_service,
            entity=entities[global_index],
            family=family,
        ):
            token_buckets.setdefault(token, []).append(global_index)

    batches: List[List[int]] = []
    seen_signatures: set[tuple[int, ...]] = set()

    def add_batch(indexes: List[int], *, representative_only: bool = False) -> None:
        unique_ordered = sorted(
            set(indexes),
            key=lambda index: (
                _coerce_int(entities[index].get("start"), 10**12),
                _coerce_int(entities[index].get("end"), 10**12),
                index,
            ),
        )
        if len(unique_ordered) < 2:
            return
        if representative_only and len(unique_ordered) > max_candidates:
            first_indexes = unique_ordered[: max(1, max_candidates // 2)]
            tail_count = max_candidates - len(first_indexes)
            tail_indexes = unique_ordered[-tail_count:] if tail_count > 0 else []
            unique_ordered = sorted(
                set([*first_indexes, *tail_indexes]),
                key=lambda index: (
                    _coerce_int(entities[index].get("start"), 10**12),
                    _coerce_int(entities[index].get("end"), 10**12),
                    index,
                ),
            )
        for chunk in _chunk_large_document_indexes(unique_ordered, max_candidates):
            if len(chunk) < 2:
                continue
            signature = tuple(chunk)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            batches.append(chunk)

    for token, bucket_indexes in sorted(
        token_buckets.items(),
        key=lambda item: (-len(set(item[1])), item[0]),
    ):
        if len(set(bucket_indexes)) < 2:
            continue
        add_batch(bucket_indexes, representative_only=":alias_candidate:" in token)
        if len(batches) >= LARGE_DOCUMENT_MODEL_REVIEW_MAX_WINDOWS_PER_FAMILY:
            break

    return batches[:LARGE_DOCUMENT_MODEL_REVIEW_MAX_WINDOWS_PER_FAMILY]


def _large_document_model_review_text(
    *,
    contexts: List[Dict[str, Any]],
    candidate_indexes: List[int],
    max_chars: int = LARGE_DOCUMENT_MODEL_REVIEW_MAX_CONTEXT_CHARS,
) -> str:
    blocks: List[str] = []
    seen_blocks: set[str] = set()
    for global_index in candidate_indexes:
        context = contexts[global_index] if 0 <= global_index < len(contexts) else {}
        if not isinstance(context, dict):
            continue
        for key in ("previous_line", "line", "window_text", "next_line"):
            block = str(context.get(key) or "").strip()
            if not block or block in seen_blocks:
                continue
            seen_blocks.add(block)
            blocks.append(block)
            if sum(len(item) + 1 for item in blocks) >= max_chars:
                break
        if sum(len(item) + 1 for item in blocks) >= max_chars:
            break
    review_text = "\n".join(blocks).strip()
    if len(review_text) > max_chars:
        review_text = review_text[:max_chars]
    return review_text


def _large_document_subject_card_contexts(
    contextual_service: Any,
    subject_cards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    card_contexts: List[Dict[str, Any]] = []
    for card in subject_cards:
        metadata = dict(card.get("metadata") or {})
        family = str(metadata.get("family") or "").strip()
        variants = [
            str(item).strip()
            for item in metadata.get("variants") or []
            if str(item).strip()
        ]
        aliases = _large_document_card_alias_values(contextual_service, card, family)
        local_replacements = [
            str(value).strip()
            for value in metadata.get("local_replacements") or []
            if str(value).strip()
        ]
        group_indexes = [
            str(index + 1)
            for index in metadata.get("group_indexes") or []
            if isinstance(index, int) and index >= 0
        ]
        line_parts = [
            f"类型:{family or str(card.get('type') or '').strip()}",
            f"主名称:{str(card.get('text') or '').strip()}",
        ]
        if variants:
            line_parts.append(f"变体:{'、'.join(variants[:8])}")
        if aliases:
            line_parts.append(f"核心别名:{'、'.join(aliases[:10])}")
        if local_replacements:
            line_parts.append(f"原组代号:{'、'.join(local_replacements[:8])}")
        if group_indexes:
            line_parts.append(f"组:{'、'.join(group_indexes[:12])}")
        mention_count = _coerce_int(metadata.get("mention_count"), 0)
        if mention_count > 0:
            line_parts.append(f"出现:{mention_count}")
        line = "；".join(part for part in line_parts if part)
        card_contexts.append(
            {
                "label": "大文件合并目录主体卡片",
                "role": str(card.get("canonical_role") or "").strip(),
                "line": line,
                "window_text": line,
                "previous_line": "",
                "next_line": "",
            }
        )
    return card_contexts


async def _run_large_document_subject_card_model_review(
    *,
    contextual_service: Any,
    text: str,
    entities: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    pools: Dict[str, List[int]],
    subject_members: Dict[int, List[int]],
    run_review: Any,
    llm_model: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reviewed_entities = [dict(entity) for entity in entities]
    subject_cards: List[Dict[str, Any]] = []
    card_member_indexes: List[List[int]] = []
    original_key_by_card_index: Dict[int, str] = {}
    for root, member_indexes in subject_members.items():
        if not member_indexes:
            continue
        first_index = min(
            member_indexes,
            key=lambda index: (_coerce_int(reviewed_entities[index].get("start"), 10**12), index),
        )
        first_entity = reviewed_entities[first_index]
        family = _global_subject_family(contextual_service, first_entity)
        if family not in pools:
            continue
        variants = sorted(
            {
                str(reviewed_entities[index].get("text") or "").strip()
                for index in member_indexes
                if 0 <= index < len(reviewed_entities)
                and str(reviewed_entities[index].get("text") or "").strip()
            },
            key=lambda value: (-len(value), value),
        )
        group_indexes = sorted(
            {
                _coerce_int((reviewed_entities[index].get("metadata") or {}).get("large_document_group_index"), -1)
                for index in member_indexes
                if 0 <= index < len(reviewed_entities)
                and _coerce_int((reviewed_entities[index].get("metadata") or {}).get("large_document_group_index"), -1)
                >= 0
            }
        )
        local_replacements = sorted(
            {
                _normalize_replacement(reviewed_entities[index].get("replacement"))
                for index in member_indexes
                if 0 <= index < len(reviewed_entities)
                and _normalize_replacement(reviewed_entities[index].get("replacement"))
            }
        )
        card_text = variants[0] if variants else str(first_entity.get("text") or "").strip()
        first_start = _coerce_int(first_entity.get("start"), 0)
        first_metadata = dict(first_entity.get("metadata") or {})
        original_key = str(first_entity.get("canonical_key") or first_metadata.get("canonical_key") or "").strip()
        if _is_local_generated_canonical_key(original_key):
            original_key = ""
        card_index = len(subject_cards)
        card = {
            "type": str(first_entity.get("type") or "").strip().upper(),
            "text": card_text,
            "start": first_start,
            "end": first_start + len(card_text),
            "replacement": str(first_entity.get("replacement") or "").strip(),
            "canonical_key": original_key or f"{LARGE_DOCUMENT_SUBJECT_CARD_KEY_PREFIX}{card_index}",
            "canonical_role": str(first_entity.get("canonical_role") or "").strip().upper(),
            "metadata": {
                "large_document_subject_card": True,
                "source_subject_root": root,
                "family": family,
                "subject_card_id": f"{LARGE_DOCUMENT_SUBJECT_CARD_KEY_PREFIX}{card_index}",
                "original_canonical_key": original_key,
                "variants": variants[:12],
                "identity_surface": " ".join(variants[:6]),
                "identity_aliases": [],
                "local_replacements": local_replacements[:12],
                "group_indexes": group_indexes,
                "mention_count": len(member_indexes),
            },
        }
        card["metadata"]["identity_aliases"] = _large_document_card_alias_values(
            contextual_service,
            card,
            family,
        )[:16]
        subject_cards.append(card)
        card_member_indexes.append(list(member_indexes))
        original_key_by_card_index[card_index] = original_key

    if len(subject_cards) < 2:
        return reviewed_entities, {
            "model_review_requested": True,
            "model_review_applied": False,
            "model_review_reason": "subject_card_count_lt_2",
            "model_review_pool_count": len([pool for pool in pools.values() if len(pool) >= 2]),
            "model_review_updated_entity_count": 0,
            "model_review_batching": "subject_cards_by_family_token_windows",
        }

    card_contexts = _large_document_subject_card_contexts(contextual_service, subject_cards)
    card_pools = _large_document_directory_pools(
        contextual_service=contextual_service,
        entities=subject_cards,
    )
    review_results: List[Dict[str, Any]] = []
    review_window_count = 0
    max_candidate_count = 0
    max_review_text_length = 0

    for family, card_indexes in card_pools.items():
        if len(card_indexes) < 2:
            continue
        batches = _large_document_model_review_batches(
            contextual_service=contextual_service,
            entities=subject_cards,
            family=family,
            global_indexes=card_indexes,
            max_candidates=12,
        )
        for batch_indexes in batches:
            review_cards = [dict(subject_cards[index]) for index in batch_indexes]
            review_contexts = [dict(card_contexts[index] if index < len(card_contexts) else {}) for index in batch_indexes]
            candidate_indexes = list(range(len(review_cards)))
            review_text = _large_document_model_review_text(
                contexts=card_contexts,
                candidate_indexes=batch_indexes,
                max_chars=6000,
            )
            if not review_text:
                review_text = "\n".join(
                    f"{card.get('text')} / {'、'.join((card.get('metadata') or {}).get('variants') or [])}"
                    for card in review_cards
                )[:6000]
            review_window_count += 1
            max_candidate_count = max(max_candidate_count, len(candidate_indexes))
            max_review_text_length = max(max_review_text_length, len(review_text))
            try:
                review_result = await run_review(
                    text=review_text,
                    refined_entities=review_cards,
                    contexts=review_contexts,
                    candidate_indexes=candidate_indexes,
                    extra_review_entities=[],
                    llm_model=llm_model,
                    review_text_override=review_text,
                    candidate_limit=len(candidate_indexes),
                    allow_drop=False,
                )
            except Exception as exc:
                review_results.append(
                    {
                        "family": family,
                        "candidate_count": len(candidate_indexes),
                        "review_text_length": len(review_text),
                        "applied": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for local_index, review_card in enumerate(review_cards):
                card_index = batch_indexes[local_index]
                subject_cards[card_index] = dict(review_card)
            payload = dict(review_result or {})
            payload["family"] = family
            payload["candidate_count"] = len(candidate_indexes)
            payload["review_text_length"] = len(review_text)
            review_results.append(payload)

    reviewed_key_by_card_index = {
        index: str(card.get("canonical_key") or "").strip()
        for index, card in enumerate(subject_cards)
        if str(card.get("canonical_key") or "").strip()
    }
    card_indexes_by_reviewed_key: Dict[str, List[int]] = {}
    for card_index, reviewed_key in reviewed_key_by_card_index.items():
        card_indexes_by_reviewed_key.setdefault(reviewed_key, []).append(card_index)

    propagated_key_by_card_index: Dict[int, str] = {}
    for reviewed_key, card_indexes in card_indexes_by_reviewed_key.items():
        if not reviewed_key:
            continue
        if _is_local_generated_canonical_key(reviewed_key):
            if len(card_indexes) < 2:
                continue
            propagated_key = f"{LARGE_DOCUMENT_MODEL_SUBJECT_KEY_PREFIX}{min(card_indexes)}"
        else:
            propagated_key = reviewed_key
        for card_index in card_indexes:
            original_key = original_key_by_card_index.get(card_index, "")
            if propagated_key == original_key and len(card_indexes) < 2:
                continue
            propagated_key_by_card_index[card_index] = propagated_key

    updated_entity_count = 0
    for card_index, member_indexes in enumerate(card_member_indexes):
        canonical_key = propagated_key_by_card_index.get(card_index)
        if not canonical_key:
            continue
        for member_index in member_indexes:
            if member_index < 0 or member_index >= len(reviewed_entities):
                continue
            entity = reviewed_entities[member_index]
            before_key = str(entity.get("canonical_key") or (entity.get("metadata") or {}).get("canonical_key") or "")
            entity["canonical_key"] = canonical_key
            metadata = dict(entity.get("metadata") or {})
            metadata["subject_card_model_review"] = True
            metadata["subject_card_index"] = card_index
            metadata["subject_card_reviewed_key"] = reviewed_key_by_card_index.get(card_index, "")
            entity["metadata"] = metadata
            if canonical_key and canonical_key != before_key:
                updated_entity_count += 1

    subject_card_merge_count = len(
        {
            key
            for key, card_indexes in card_indexes_by_reviewed_key.items()
            if key and len(card_indexes) >= 2
        }
    )
    model_review_applied = (
        any(bool(item.get("applied")) for item in review_results)
        or bool(propagated_key_by_card_index)
        or updated_entity_count > 0
        or subject_card_merge_count > 0
    )

    return reviewed_entities, {
        "model_review_requested": True,
        "model_review_applied": model_review_applied,
        "model_review_reason": "",
        "model_review_pool_count": len([pool for pool in pools.values() if len(pool) >= 2]),
        "model_review_window_count": review_window_count,
        "model_review_max_candidate_count": max_candidate_count,
        "model_review_max_review_text_length": max_review_text_length,
        "model_review_batching": "subject_cards_by_family_token_windows",
        "model_review_subject_card_count": len(subject_cards),
        "model_review_subject_card_merge_count": subject_card_merge_count,
        "model_review_updated_entity_count": updated_entity_count,
        "model_review_results": review_results[:20],
    }


def _large_document_stable_canonical_key(contextual_service: Any, entity: Dict[str, Any]) -> str:
    canonical_key_from_entity = getattr(contextual_service, "_canonical_key_from_entity", None)
    if callable(canonical_key_from_entity):
        try:
            canonical_key = str(canonical_key_from_entity(entity) or "").strip()
        except Exception:
            canonical_key = ""
    else:
        metadata = dict(entity.get("metadata") or {})
        canonical_key = str(entity.get("canonical_key") or metadata.get("canonical_key") or "").strip()
    if canonical_key and _is_local_generated_canonical_key(canonical_key):
        return ""
    return canonical_key


def _large_document_rule_identity_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(entity)
    if _is_local_generated_canonical_key(item.get("canonical_key")):
        item.pop("canonical_key", None)
    metadata = dict(item.get("metadata") or {})
    if _is_local_generated_canonical_key(metadata.get("canonical_key")):
        metadata.pop("canonical_key", None)
    item["metadata"] = metadata
    return item


def _large_document_entities_have_identity_conflict(
    *,
    contextual_service: Any,
    left_entity: Dict[str, Any],
    left_context: Dict[str, Any],
    right_entity: Dict[str, Any],
    right_context: Dict[str, Any],
) -> bool:
    left_key = _large_document_stable_canonical_key(contextual_service, left_entity)
    right_key = _large_document_stable_canonical_key(contextual_service, right_entity)
    if left_key and right_key and left_key != right_key:
        return True

    party_role = getattr(contextual_service, "_party_role_from_context", None)
    if callable(party_role):
        try:
            left_role = str(party_role(left_context) or "").strip()
            right_role = str(party_role(right_context) or "").strip()
        except Exception:
            left_role = ""
            right_role = ""
        if left_role and right_role and left_role != right_role:
            return True
    return False


def _large_document_rule_share_gate(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    pool_indexes: List[int],
    left_entity: Dict[str, Any],
    left_context: Dict[str, Any],
    right_entity: Dict[str, Any],
    right_context: Dict[str, Any],
    family: str,
) -> bool:
    left_key = _large_document_stable_canonical_key(contextual_service, left_entity)
    right_key = _large_document_stable_canonical_key(contextual_service, right_entity)
    if left_key and right_key:
        return left_key == right_key

    left_text = _normalize_subject_token(left_entity.get("text"))
    right_text = _normalize_subject_token(right_entity.get("text"))
    if left_text and right_text and left_text == right_text:
        return True

    left_tokens = set(
        _large_document_rule_identity_tokens(
            contextual_service,
            entity=left_entity,
            family=family,
        )
    )
    right_tokens = set(
        _large_document_rule_identity_tokens(
            contextual_service,
            entity=right_entity,
            family=family,
        )
    )
    shared_tokens = left_tokens & right_tokens
    return any(
        (":canonical:" in token or ":identity:" in token or ":role:" in token)
        for token in shared_tokens
    )


def _large_document_component_stable_keys(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    candidate_indexes: List[int],
    uf: _UnionFind,
    pos: int,
) -> set[str]:
    root = uf.find(pos)
    keys: set[str] = set()
    for candidate_pos, entity_index in enumerate(candidate_indexes):
        if uf.find(candidate_pos) != root:
            continue
        key = _large_document_stable_canonical_key(contextual_service, entities[entity_index])
        if key:
            keys.add(key)
    return keys


def _large_document_can_union_components(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    candidate_indexes: List[int],
    uf: _UnionFind,
    left_pos: int,
    right_pos: int,
) -> bool:
    left_keys = _large_document_component_stable_keys(
        contextual_service=contextual_service,
        entities=entities,
        candidate_indexes=candidate_indexes,
        uf=uf,
        pos=left_pos,
    )
    right_keys = _large_document_component_stable_keys(
        contextual_service=contextual_service,
        entities=entities,
        candidate_indexes=candidate_indexes,
        uf=uf,
        pos=right_pos,
    )
    return not (left_keys and right_keys and left_keys.isdisjoint(right_keys))


def _apply_large_document_rule_identity_merge(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    pools: Dict[str, List[int]],
) -> tuple[Dict[int, List[int]], Dict[str, Any]]:
    candidate_indexes = [entity_index for pool_indexes in pools.values() for entity_index in pool_indexes]
    if not candidate_indexes:
        return {}, {
            "rule_pool_count": 0,
            "rule_identity_edge_count": 0,
            "token_identity_edge_count": 0,
            "local_replacement_edge_count": 0,
        }

    uf = _UnionFind(len(candidate_indexes))
    local_pos_by_entity_index = {entity_index: pos for pos, entity_index in enumerate(candidate_indexes)}
    token_identity_edges = 0
    rule_identity_edges = 0
    local_replacement_edges = 0
    unique_anchor_alias_edges = 0
    token_owner: Dict[str, int] = {}
    local_replacement_owner: Dict[tuple[int, str, str], int] = {}
    should_share_identity = getattr(contextual_service, "_should_share_identity", None)

    for family, pool_indexes in pools.items():
        for entity_index in pool_indexes:
            entity = entities[entity_index]
            pos = local_pos_by_entity_index[entity_index]
            for token in _large_document_rule_identity_tokens(
                contextual_service,
                entity=entity,
                family=family,
            ):
                owner = token_owner.get(token)
                if owner is None:
                    token_owner[token] = pos
                else:
                    owner_entity_index = candidate_indexes[owner]
                    if _large_document_entities_have_identity_conflict(
                        contextual_service=contextual_service,
                        left_entity=entities[owner_entity_index],
                        left_context=contexts[owner_entity_index] if owner_entity_index < len(contexts) else {},
                        right_entity=entity,
                        right_context=contexts[entity_index] if entity_index < len(contexts) else {},
                    ):
                        continue
                    if not _large_document_can_union_components(
                        contextual_service=contextual_service,
                        entities=entities,
                        candidate_indexes=candidate_indexes,
                        uf=uf,
                        left_pos=owner,
                        right_pos=pos,
                    ):
                        continue
                    if uf.find(owner) != uf.find(pos):
                        token_identity_edges += 1
                    uf.union(owner, pos)

            metadata = dict(entity.get("metadata") or {})
            group_index = _coerce_int(metadata.get("large_document_group_index"), -1)
            replacement = _normalize_replacement(entity.get("replacement"))
            if group_index >= 0 and replacement:
                key = (group_index, family, replacement)
                owner = local_replacement_owner.get(key)
                if owner is None:
                    local_replacement_owner[key] = pos
                else:
                    owner_entity_index = candidate_indexes[owner]
                    if _large_document_entities_have_identity_conflict(
                        contextual_service=contextual_service,
                        left_entity=entities[owner_entity_index],
                        left_context=contexts[owner_entity_index] if owner_entity_index < len(contexts) else {},
                        right_entity=entity,
                        right_context=contexts[entity_index] if entity_index < len(contexts) else {},
                    ):
                        continue
                    if not _large_document_can_union_components(
                        contextual_service=contextual_service,
                        entities=entities,
                        candidate_indexes=candidate_indexes,
                        uf=uf,
                        left_pos=owner,
                        right_pos=pos,
                    ):
                        continue
                    if uf.find(owner) != uf.find(pos):
                        local_replacement_edges += 1
                    uf.union(owner, pos)

        if callable(should_share_identity):
            review_token_buckets: Dict[str, List[int]] = {}
            for entity_index in pool_indexes:
                for token in _large_document_model_review_tokens(
                    contextual_service,
                    entity=entities[entity_index],
                    family=family,
                ):
                    review_token_buckets.setdefault(token, []).append(entity_index)
            reviewed_pairs: set[tuple[int, int]] = set()
            family_review_edges = 0
            for _, bucket_indexes in sorted(
                review_token_buckets.items(),
                key=lambda item: (-len(set(item[1])), item[0]),
            ):
                if family_review_edges >= LARGE_DOCUMENT_RULE_REVIEW_MAX_EDGES_PER_FAMILY:
                    break
                unique_bucket = sorted(
                    set(bucket_indexes),
                    key=lambda index: (
                        _coerce_int(entities[index].get("start"), 10**12),
                        _coerce_int(entities[index].get("end"), 10**12),
                        index,
                    ),
                )[:LARGE_DOCUMENT_RULE_REVIEW_MAX_BUCKET_CANDIDATES]
                if len(unique_bucket) < 2:
                    continue
                for left_offset, left_index in enumerate(unique_bucket):
                    left_entity = entities[left_index]
                    left_context = contexts[left_index] if left_index < len(contexts) else {}
                    for right_index in unique_bucket[left_offset + 1:]:
                        pair = (min(left_index, right_index), max(left_index, right_index))
                        if pair in reviewed_pairs:
                            continue
                        reviewed_pairs.add(pair)
                        family_review_edges += 1
                        if family_review_edges > LARGE_DOCUMENT_RULE_REVIEW_MAX_EDGES_PER_FAMILY:
                            break
                        right_entity = entities[right_index]
                        right_context = contexts[right_index] if right_index < len(contexts) else {}
                        try:
                            shares_identity = bool(
                                should_share_identity(
                                    _large_document_rule_identity_entity(left_entity),
                                    left_context,
                                    _large_document_rule_identity_entity(right_entity),
                                    right_context,
                                )
                            )
                        except Exception:
                            shares_identity = False
                        if not shares_identity:
                            continue
                        if not _large_document_rule_share_gate(
                            contextual_service=contextual_service,
                            entities=entities,
                            contexts=contexts,
                            pool_indexes=pool_indexes,
                            left_entity=left_entity,
                            left_context=left_context,
                            right_entity=right_entity,
                            right_context=right_context,
                            family=family,
                        ):
                            continue
                        left_pos = local_pos_by_entity_index[left_index]
                        right_pos = local_pos_by_entity_index[right_index]
                        if not _large_document_can_union_components(
                            contextual_service=contextual_service,
                            entities=entities,
                            candidate_indexes=candidate_indexes,
                            uf=uf,
                            left_pos=left_pos,
                            right_pos=right_pos,
                        ):
                            continue
                        if uf.find(left_pos) != uf.find(right_pos):
                            rule_identity_edges += 1
                        uf.union(left_pos, right_pos)
                    if family_review_edges >= LARGE_DOCUMENT_RULE_REVIEW_MAX_EDGES_PER_FAMILY:
                        break

        unique_anchor_alias_edges += _large_document_apply_unique_anchor_alias_merges(
            contextual_service=contextual_service,
            entities=entities,
            contexts=contexts,
            pool_indexes=pool_indexes,
            family=family,
            candidate_indexes=candidate_indexes,
            local_pos_by_entity_index=local_pos_by_entity_index,
            uf=uf,
        )

    subject_members: Dict[int, List[int]] = {}
    for entity_index in candidate_indexes:
        root = uf.find(local_pos_by_entity_index[entity_index])
        subject_members.setdefault(root, []).append(entity_index)

    return subject_members, {
        "rule_pool_count": len(pools),
        "rule_identity_edge_count": rule_identity_edges,
        "token_identity_edge_count": token_identity_edges,
        "local_replacement_edge_count": local_replacement_edges,
        "unique_anchor_alias_edge_count": unique_anchor_alias_edges,
    }


async def _run_large_document_directory_model_review(
    *,
    contextual_service: Any,
    text: str,
    entities: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    pools: Dict[str, List[int]],
    subject_members: Optional[Dict[int, List[int]]] = None,
    use_llm: bool,
    llm_model: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not use_llm:
        return entities, {
            "model_review_requested": False,
            "model_review_applied": False,
            "model_review_reason": "use_llm_disabled",
            "model_review_pool_count": 0,
            "model_review_updated_entity_count": 0,
        }

    run_review = getattr(contextual_service, "_run_resolution_subset_review", None)
    backend_available = getattr(contextual_service, "_resolution_backend_available", None)
    if not callable(run_review) or not callable(backend_available):
        return entities, {
            "model_review_requested": True,
            "model_review_applied": False,
            "model_review_reason": "resolution_review_unavailable",
            "model_review_pool_count": 0,
            "model_review_updated_entity_count": 0,
        }
    try:
        if not bool(backend_available(llm_model=llm_model)):
            return entities, {
                "model_review_requested": True,
                "model_review_applied": False,
                "model_review_reason": "resolution_backend_unavailable",
                "model_review_pool_count": 0,
                "model_review_updated_entity_count": 0,
            }
    except Exception:
        return entities, {
            "model_review_requested": True,
            "model_review_applied": False,
            "model_review_reason": "resolution_backend_check_failed",
            "model_review_pool_count": 0,
            "model_review_updated_entity_count": 0,
        }

    reviewed_entities = [dict(entity) for entity in entities]
    if subject_members:
        return await _run_large_document_subject_card_model_review(
            contextual_service=contextual_service,
            text=text,
            entities=reviewed_entities,
            contexts=contexts,
            pools=pools,
            subject_members=subject_members,
            run_review=run_review,
            llm_model=llm_model,
        )

    marker_key = "__large_document_global_entity_index"
    review_results: List[Dict[str, Any]] = []
    updated_entity_count = 0
    review_window_count = 0
    max_candidate_count = 0
    max_review_text_length = 0

    for family, global_indexes in pools.items():
        if len(global_indexes) < 2:
            continue
        family_batches = _large_document_model_review_batches(
            contextual_service=contextual_service,
            entities=reviewed_entities,
            family=family,
            global_indexes=global_indexes,
        )
        if not family_batches:
            continue

        for batch_indexes in family_batches:
            if len(batch_indexes) < 2:
                continue
            pool_entities: List[Dict[str, Any]] = []
            pool_contexts: List[Dict[str, Any]] = []
            for global_index in batch_indexes:
                pool_entity = dict(reviewed_entities[global_index])
                pool_entity[marker_key] = global_index
                pool_entities.append(pool_entity)
                pool_contexts.append(dict(contexts[global_index] if global_index < len(contexts) else {}))

            before_keys = {
                global_index: str(
                    reviewed_entities[global_index].get("canonical_key")
                    or (reviewed_entities[global_index].get("metadata") or {}).get("canonical_key")
                    or ""
                )
                for global_index in batch_indexes
            }
            candidate_indexes = list(range(len(pool_entities)))
            review_text = _large_document_model_review_text(
                contexts=contexts,
                candidate_indexes=batch_indexes,
            )
            if not review_text:
                review_text = "\n".join(
                    str(pool_entity.get("text") or "").strip()
                    for pool_entity in pool_entities
                    if str(pool_entity.get("text") or "").strip()
                )[:LARGE_DOCUMENT_MODEL_REVIEW_MAX_CONTEXT_CHARS]
            review_window_count += 1
            max_candidate_count = max(max_candidate_count, len(candidate_indexes))
            max_review_text_length = max(max_review_text_length, len(review_text))
            try:
                review_result = await run_review(
                    text=review_text,
                    refined_entities=pool_entities,
                    contexts=pool_contexts,
                    candidate_indexes=candidate_indexes,
                    extra_review_entities=[],
                    llm_model=llm_model,
                    review_text_override=review_text,
                    candidate_limit=len(candidate_indexes),
                    allow_drop=False,
                )
            except Exception as exc:
                review_results.append(
                    {
                        "family": family,
                        "candidate_count": len(candidate_indexes),
                        "review_text_length": len(review_text),
                        "applied": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            for pool_entity in pool_entities:
                global_index = _coerce_int(pool_entity.get(marker_key), -1)
                if global_index < 0 or global_index >= len(reviewed_entities):
                    continue
                pool_entity.pop(marker_key, None)
                reviewed_entities[global_index] = dict(pool_entity)
                after_key = str(
                    pool_entity.get("canonical_key")
                    or (pool_entity.get("metadata") or {}).get("canonical_key")
                    or ""
                )
                if after_key and after_key != before_keys.get(global_index, ""):
                    updated_entity_count += 1

            review_payload = dict(review_result or {})
            review_payload["family"] = family
            review_payload["candidate_count"] = len(candidate_indexes)
            review_payload["review_text_length"] = len(review_text)
            review_results.append(review_payload)

    return reviewed_entities, {
        "model_review_requested": True,
        "model_review_applied": any(bool(item.get("applied")) for item in review_results),
        "model_review_reason": "",
        "model_review_pool_count": len([pool for pool in pools.values() if len(pool) >= 2]),
        "model_review_window_count": review_window_count,
        "model_review_max_candidate_count": max_candidate_count,
        "model_review_max_review_text_length": max_review_text_length,
        "model_review_batching": "family_pool_token_windows",
        "model_review_updated_entity_count": updated_entity_count,
        "model_review_results": review_results[:20],
    }


def _generated_global_replacement(
    contextual_service: Any,
    *,
    strategy_key: str,
    family: str,
    index: int,
) -> str:
    render = getattr(contextual_service, "_render_symbolic_index", None)
    to_cn = getattr(contextual_service, "_to_cn_serial", None)
    to_alpha = getattr(contextual_service, "_to_alpha", None)
    if strategy_key == "symbolic_codes" and callable(render):
        if family == "person":
            return str(render(index, style="lower_letter"))
        if family in {"organization", "bank", "court"}:
            return str(render(index, style="greek"))
        if family in {"location", "address"}:
            return f"{render(index, style='cn')}地"
        if family == "project":
            return f"PROJECT-{render(index, style='upper_letter')}"
    if strategy_key == "serial_roles":
        if callable(to_cn):
            serial = str(to_cn(index))
        else:
            serial = f"第{index + 1}"
        prefix_by_family = {
            "person": "人员",
            "organization": "单位",
            "bank": "开户行",
            "project": "项目",
            "location": "地区",
            "address": "地址",
            "court": "法院",
        }
        return f"{prefix_by_family.get(family, '主体')}{serial}"
    if callable(to_alpha):
        return f"主体{to_alpha(index)}"
    return f"主体{index + 1}"


def _next_unused_generated_replacement(
    contextual_service: Any,
    *,
    strategy_key: str,
    family: str,
    used_replacements: set[str],
    per_family_sequence_index: Dict[str, int],
) -> str:
    sequence_index = per_family_sequence_index.get(family, 0)
    while True:
        candidate = _generated_global_replacement(
            contextual_service,
            strategy_key=strategy_key,
            family=family,
            index=sequence_index,
        )
        sequence_index += 1
        if candidate and candidate not in used_replacements:
            per_family_sequence_index[family] = sequence_index
            return candidate


def _build_large_document_subjects_from_members(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    subject_members: Dict[int, List[int]],
) -> tuple[List[Dict[str, Any]], Dict[int, int]]:
    subjects: List[Dict[str, Any]] = []
    subject_id_by_entity_index: Dict[int, int] = {}
    ordered_member_groups = sorted(
        subject_members.values(),
        key=lambda indexes: (
            min(_coerce_int(entities[index].get("start"), 10**12) for index in indexes),
            min(indexes),
        ),
    )

    for subject_index, member_indexes in enumerate(ordered_member_groups):
        member_indexes = sorted(
            member_indexes,
            key=lambda idx: (_coerce_int(entities[idx].get("start"), 10**12), idx),
        )
        first_entity = entities[member_indexes[0]]
        family = _global_subject_family(contextual_service, first_entity)
        replacements: Dict[str, int] = {}
        variants: set[str] = set()
        entity_types: set[str] = set()
        group_indexes: set[int] = set()
        for member_index in member_indexes:
            member = entities[member_index]
            replacement = _normalize_replacement(member.get("replacement"))
            if replacement:
                replacements.setdefault(replacement, _coerce_int(member.get("start"), 10**12))
            text_token = str(member.get("text") or "").strip()
            if text_token:
                variants.add(text_token)
            entity_type = str(member.get("type") or "").strip().upper()
            if entity_type:
                entity_types.add(entity_type)
            metadata = dict(member.get("metadata") or {})
            group_index = _coerce_int(metadata.get("large_document_group_index"), -1)
            if group_index >= 0:
                group_indexes.add(group_index)
        earliest_replacement = ""
        if replacements:
            earliest_replacement = sorted(replacements.items(), key=lambda item: (item[1], item[0]))[0][0]
        subject_payload = {
            "id": subject_index,
            "family": family,
            "member_indexes": member_indexes,
            "first_start": _coerce_int(first_entity.get("start"), 10**12),
            "primary_text": str(first_entity.get("text") or "").strip(),
            "primary_type": str(first_entity.get("type") or "").strip().upper(),
            "entity_types": sorted(entity_types),
            "earliest_replacement": earliest_replacement,
            "variants": sorted(variants, key=lambda item: (-len(item), item)),
            "group_indexes": sorted(group_indexes),
        }
        subjects.append(subject_payload)
        for member_index in member_indexes:
            subject_id_by_entity_index[member_index] = subject_index

    return subjects, subject_id_by_entity_index


def _apply_large_document_subject_replacements(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    subjects: List[Dict[str, Any]],
    strategy_key: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not subjects:
        return entities, {
            "global_subject_count": 0,
            "replacement_conflict_count": 0,
            "global_replacement_directory": [],
        }

    subjects.sort(key=lambda item: (int(item["first_start"]), int(item["id"])))
    used_replacements: set[str] = set()
    per_family_sequence_index: Dict[str, int] = {}
    replacement_conflicts = 0
    replacement_by_subject_id: Dict[int, str] = {}

    for subject in subjects:
        family = str(subject["family"])
        observed = str(subject["earliest_replacement"])
        if observed and observed not in used_replacements:
            replacement = observed
        else:
            if observed:
                replacement_conflicts += 1
            replacement = _next_unused_generated_replacement(
                contextual_service,
                strategy_key=strategy_key,
                family=family,
                used_replacements=used_replacements,
                per_family_sequence_index=per_family_sequence_index,
            )
        used_replacements.add(replacement)
        replacement_by_subject_id[int(subject["id"])] = replacement

    subject_id_by_entity_index: Dict[int, int] = {}
    for subject in subjects:
        for member_index in subject["member_indexes"]:
            subject_id_by_entity_index[member_index] = int(subject["id"])

    remapped_entities: List[Dict[str, Any]] = []
    for index, entity in enumerate(entities):
        item = dict(entity)
        subject_id = subject_id_by_entity_index.get(index)
        if subject_id is not None:
            replacement = replacement_by_subject_id[subject_id]
            item["replacement"] = replacement
            item["replacement_method"] = item.get("replacement_method") or "contextual"
            metadata = dict(item.get("metadata") or {})
            metadata["global_large_document_subject_id"] = subject_id
            metadata["global_large_document_replacement"] = replacement
            item["metadata"] = metadata
        remapped_entities.append(item)

    directory = []
    subject_by_id = {int(subject["id"]): subject for subject in subjects}
    for subject_id, replacement in replacement_by_subject_id.items():
        subject = subject_by_id[subject_id]
        directory.append(
            {
                "subject_id": subject_id,
                "family": subject["family"],
                "primary_type": subject.get("primary_type") or "",
                "entity_types": list(subject.get("entity_types") or []),
                "primary_text": subject["primary_text"],
                "replacement": replacement,
                "first_start": subject["first_start"],
                "variants": list(subject["variants"])[:50],
                "group_indexes": list(subject.get("group_indexes") or []),
                "mention_count": len(subject["member_indexes"]),
            }
        )
    directory.sort(key=lambda item: (int(item["first_start"]), int(item["subject_id"])))

    return remapped_entities, {
        "global_subject_count": len(subjects),
        "replacement_conflict_count": replacement_conflicts,
        "global_replacement_directory": directory,
    }


async def _merge_large_document_global_directory(
    *,
    contextual_service: Any,
    text: str,
    entities: List[Dict[str, Any]],
    strategy_key: str,
    use_llm: bool,
    llm_model: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    apply_hints = getattr(contextual_service, "_apply_metadata_identity_hints", None)
    if callable(apply_hints):
        try:
            working_entities = apply_hints([dict(entity) for entity in entities])
        except Exception:
            working_entities = [dict(entity) for entity in entities]
    else:
        working_entities = [dict(entity) for entity in entities]

    contexts = _build_large_document_identity_contexts(
        contextual_service=contextual_service,
        text=text,
        entities=working_entities,
    )
    initial_pools = _large_document_directory_pools(
        contextual_service=contextual_service,
        entities=working_entities,
    )
    initial_subject_members, initial_rule_metadata = _apply_large_document_rule_identity_merge(
        contextual_service=contextual_service,
        entities=working_entities,
        contexts=contexts,
        pools=initial_pools,
    )

    reviewed_entities, model_metadata = await _run_large_document_directory_model_review(
        contextual_service=contextual_service,
        text=text,
        entities=working_entities,
        contexts=contexts,
        pools=initial_pools,
        subject_members=initial_subject_members,
        use_llm=use_llm,
        llm_model=llm_model,
    )
    reviewed_contexts = _build_large_document_identity_contexts(
        contextual_service=contextual_service,
        text=text,
        entities=reviewed_entities,
    )
    final_pools = _large_document_directory_pools(
        contextual_service=contextual_service,
        entities=reviewed_entities,
    )
    subject_members, final_rule_metadata = _apply_large_document_rule_identity_merge(
        contextual_service=contextual_service,
        entities=reviewed_entities,
        contexts=reviewed_contexts,
        pools=final_pools,
    )
    subjects, _ = _build_large_document_subjects_from_members(
        contextual_service=contextual_service,
        entities=reviewed_entities,
        subject_members=subject_members,
    )
    remapped_entities, replacement_metadata = _apply_large_document_subject_replacements(
        contextual_service=contextual_service,
        entities=reviewed_entities,
        subjects=subjects,
        strategy_key=strategy_key,
    )

    pool_sizes = {family: len(indexes) for family, indexes in final_pools.items()}
    metadata = {
        **replacement_metadata,
        "global_directory_pool_sizes": pool_sizes,
        "global_directory_pool_count": len(final_pools),
        "global_directory_rule_initial": initial_rule_metadata,
        "global_directory_rule_final": final_rule_metadata,
        "global_directory_model_review": model_metadata,
        "global_directory_merge_order": "pool_by_subject_family_then_rules_model_then_rules",
    }
    return remapped_entities, metadata


def _reassign_large_document_global_replacements(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    strategy_key: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    contexts = _build_large_document_identity_contexts(
        contextual_service=contextual_service,
        text="",
        entities=entities,
    )
    pools = _large_document_directory_pools(
        contextual_service=contextual_service,
        entities=entities,
    )
    subject_members, rule_metadata = _apply_large_document_rule_identity_merge(
        contextual_service=contextual_service,
        entities=entities,
        contexts=contexts,
        pools=pools,
    )
    subjects, _ = _build_large_document_subjects_from_members(
        contextual_service=contextual_service,
        entities=entities,
        subject_members=subject_members,
    )
    remapped_entities, metadata = _apply_large_document_subject_replacements(
        contextual_service=contextual_service,
        entities=entities,
        subjects=subjects,
        strategy_key=strategy_key,
    )
    metadata["global_directory_rule_final"] = rule_metadata
    metadata["global_directory_pool_sizes"] = {family: len(indexes) for family, indexes in pools.items()}
    return remapped_entities, metadata


def _build_large_document_group_directory(
    *,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in sorted(
        (item for item in entities if isinstance(item, dict)),
        key=lambda item: (_coerce_int(item.get("start"), 10**12), _coerce_int(item.get("end"), 10**12)),
    ):
        source_text = str(entity.get("text") or "").strip()
        replacement = _normalize_replacement(entity.get("replacement"))
        entity_type = str(entity.get("type") or "").strip()
        if not source_text or not replacement or replacement == source_text:
            continue
        key = (entity_type, source_text, replacement)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "type": entity_type,
                "source_text": source_text,
                "replacement": replacement,
                "first_start": _coerce_int(entity.get("start"), 10**12),
            }
        )
    return rows


def _global_entities_to_group(
    *,
    entities: List[Dict[str, Any]],
    group: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _project_entities_to_group(entities=entities, group=group)


async def _anonymize_large_document_groups_in_order(
    *,
    engine: Any,
    text: str,
    groups: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    operator_config: Any,
    progress_path: Optional[str],
) -> str:
    parts: List[str] = []
    cursor = 0
    total = len(groups)
    for group in groups:
        group_index = int(group.get("index", 0))
        group_start = int(group.get("start", cursor))
        group_end = int(group.get("end", group_start))
        if group_start > cursor:
            parts.append(text[cursor:group_start])
        local_entities = _global_entities_to_group(entities=entities, group=group)
        _write_progress(
            progress_path,
            {
                "stage": "anonymize_text",
                "current": group_index + 1,
                "total": total,
                "message": f"正在按全局对照目录回写大文件第 {group_index + 1}/{total} 组...",
            },
        )
        parts.append(
            await engine.anonymize(
                text=str(group.get("text") or ""),
                entities=local_entities,
                operator_config=operator_config,
            )
        )
        cursor = max(cursor, group_end)
    if cursor < len(text):
        parts.append(text[cursor:])
    return "".join(parts)


def _large_document_strategy_key(payload: Dict[str, Any]) -> str:
    return get_anonymization_strategy_profile(
        payload.get("anonymization_strategy") or DEFAULT_ANONYMIZATION_STRATEGY
    ).key


def _export_large_document_mapping_docx(
    *,
    payload: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    exporter = DocumentExporter()
    return dict(
        exporter._export_mapping_docx(
            task_id=f"{_large_document_task_slug(payload)}_global",
            original_filename=str(payload.get("original_filename") or payload.get("source_path") or "large_document"),
            entities=entities,
        )
    )


async def _rewrite_large_document_group_from_directory(
    *,
    payload: Dict[str, Any],
    group: Dict[str, Any],
    group_run: Dict[str, Any],
    entities: List[Dict[str, Any]],
    engine: Any,
    exporter: DocumentExporter,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    group_index = int(group.get("index", 0))
    group_dir = Path(str(group_run.get("group_dir") or ""))
    group_source_path = str(group_run.get("group_source_path") or "")
    group_source_name = str(group_run.get("group_source_filename") or Path(group_source_path).name)
    local_entities = _global_entities_to_group(entities=entities, group=group)
    group_text = str(group.get("text") or "")
    anonymized_text = await engine.anonymize(
        text=group_text,
        entities=local_entities,
        operator_config=payload.get("operator_config"),
    )
    coverage_first_final_export = build_coverage_first_final_export_bundle(
        entities=local_entities,
        source_text=group_text,
    )
    export_result = exporter.export(
        task_id=f"{_large_document_task_slug(payload)}_group_{group_index + 1:04d}_global",
        source_path=group_source_path,
        original_filename=group_source_name,
        source_text=group_text,
        source_metadata={
            "large_document_group_index": group_index,
            "large_document_group_pages": list(group.get("page_numbers") or []),
        },
        source_structure={
            "page_count": int(group.get("page_count") or 1),
            "pages": list(group.get("pages") or []),
        },
        entities=local_entities,
        anonymized_text=anonymized_text,
        operator_config=payload.get("operator_config"),
        coverage_first_final_export=coverage_first_final_export,
    )
    materialized = _materialize_group_export_result(
        export_result=export_result,
        group_dir=group_dir,
    )
    return materialized, local_entities, anonymized_text


def _merge_large_document_rewritten_outputs(
    *,
    payload: Dict[str, Any],
    groups: List[Dict[str, Any]],
    group_runs: List[Dict[str, Any]],
    source_text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    source_path = Path(str(payload.get("source_path") or ""))
    original_filename = str(payload.get("original_filename") or source_path.name or "large_document")
    task_id = _large_document_task_slug(payload)
    source_suffix = source_path.suffix.lower()
    output_stem = Path(original_filename).stem or task_id

    if source_suffix == ".docx":
        rewritten_docx_paths = [
            Path(str(run.get("output_path") or ""))
            for run in group_runs
            if str(run.get("output_path") or "").strip()
            and Path(str(run.get("output_path") or "")).suffix.lower() == ".docx"
            and Path(str(run.get("output_path") or "")).exists()
        ]
        if len(rewritten_docx_paths) == len(group_runs) and rewritten_docx_paths:
            final_path = Path(settings.OUTPUT_DIR) / f"{task_id}_anonymized.docx"
            merge_docx_documents(rewritten_docx_paths, final_path)
            ensure_private_file(final_path)
            return {
                "output_path": str(final_path),
                "download_name": f"{output_stem}_anonymized.docx",
                "file_type": "docx",
                "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "preserves_format": True,
                "warning": None,
                "docx_rewrite_method": "merged_rewritten_group_docx",
                "docx_precise_rewrite_complete": True,
                "docx_range_rewrite_applied_count": 0,
                "docx_range_rewrite_unapplied_count": 0,
            }

    parts: List[str] = []
    cursor = 0
    for group, run in zip(groups, group_runs):
        group_start = int(group.get("start", cursor))
        group_end = int(group.get("end", group_start))
        if group_start > cursor:
            parts.append(source_text[cursor:group_start])
        output_path = Path(str(run.get("output_path") or ""))
        if output_path.exists():
            parts.append(output_path.read_text(encoding="utf-8"))
        else:
            parts.append(str(run.get("anonymized_text") or ""))
        cursor = max(cursor, group_end)
    if cursor < len(source_text):
        parts.append(source_text[cursor:])

    final_path = Path(settings.OUTPUT_DIR) / f"{task_id}_anonymized.txt"
    final_path.write_text("".join(parts), encoding="utf-8")
    ensure_private_file(final_path)
    warning = None
    if source_suffix not in {"", ".txt"}:
        warning = "Large-document final export fell back to TXT because rewritten group outputs were not all DOCX."
    return {
        "output_path": str(final_path),
        "download_name": f"{output_stem}_anonymized.txt",
        "file_type": "txt",
        "media_type": "text/plain",
        "preserves_format": source_suffix in {"", ".txt"},
        "warning": warning,
    }


def _join_large_document_rewritten_text(
    *,
    source_text: str,
    groups: List[Dict[str, Any]],
    rewritten_group_texts: List[str],
) -> str:
    parts: List[str] = []
    cursor = 0
    for group, rewritten_text in zip(groups, rewritten_group_texts):
        group_start = int(group.get("start", cursor))
        group_end = int(group.get("end", group_start))
        if group_start > cursor:
            parts.append(source_text[cursor:group_start])
        parts.append(rewritten_text)
        cursor = max(cursor, group_end)
    if cursor < len(source_text):
        parts.append(source_text[cursor:])
    return "".join(parts)


def _should_run_large_document_grouped_workflow(payload: Dict[str, Any]) -> bool:
    if not settings.is_high_quality_lowmem_mode():
        return False
    text = str(payload.get("source_text") or "")
    source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
    source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
    has_preroute_marker = (
        str((source_metadata or {}).get("analysis_workflow_mode") or "").strip() == "large_document_pre_routed"
        or (source_metadata or {}).get("_large_document_pre_routed") is True
        or str((source_structure or {}).get("analysis_workflow_mode") or "").strip() == "large_document_pre_routed"
        or (source_structure or {}).get("_large_document_pre_routed") is True
    )
    if not has_preroute_marker:
        return False
    workflow = classify_document_workflow(
        text=text,
        source_metadata=source_metadata,
        source_structure=source_structure,
    )
    return bool(workflow.get("enabled"))


async def _run_large_document_grouped_workflow(
    *,
    payload: Dict[str, Any],
    progress_path: Optional[str],
) -> Dict[str, Any]:
    source_text = str(payload.get("source_text") or "")
    source_structure = payload.get("source_structure")
    groups = _build_large_document_page_groups(
        text=source_text,
        source_structure=source_structure,
    )
    if not groups:
        raise RuntimeError("large_document_grouped_workflow_missing_real_pages")

    all_entities: List[Dict[str, Any]] = []
    group_runs: List[Dict[str, Any]] = []
    group_root = _large_document_group_output_root(payload)
    group_sources: List[Dict[str, Any]] = []

    for group in groups:
        group_dir = _large_document_group_dir(group_root, group)
        group_source_path, group_source_name = _write_group_source_file(
            payload=payload,
            group=group,
            group_dir=group_dir,
        )
        group_sources.append(
            {
                "group": group,
                "group_dir": group_dir,
                "group_source_path": group_source_path,
                "group_source_name": group_source_name,
            }
        )

    for group_source in group_sources:
        group = dict(group_source["group"])
        group_index = int(group.get("index", 0))
        group_dir = Path(group_source["group_dir"])
        group_source_path = str(group_source["group_source_path"])
        group_source_name = str(group_source["group_source_name"])
        _write_progress(
            progress_path,
            {
                "stage": "large_document_group_default_flow",
                "current": group_index + 1,
                "total": len(groups),
                "message": f"正在将大文件第 {group_index + 1}/{len(groups)} 组按默认流程独立处理...",
            },
        )
        group_result = await _run_group_source_through_default_line(
            payload=payload,
            group=group,
            group_source_path=group_source_path,
            group_source_name=group_source_name,
        )
        group_export_result = _materialize_group_export_result(
            export_result=dict(group_result.get("export_result") or {}),
            group_dir=group_dir,
        )
        if not group_export_result.get("output_path") or not group_export_result.get("mapping_output_path"):
            raise RuntimeError(f"large_document_group_default_export_missing_files:{group_index + 1}")

        group_text = str(group_result.get("text") or "")
        detected_group_entities = list(group_result.get("detected_entities") or [])
        prepared_group_entities = list(group_result.get("prepared_entities") or [])
        group_quality = dict(group_result.get("quality_metadata") or {})
        group_anonymized_text = str(group_result.get("anonymized_text") or "")
        restored_entities = _restore_group_entities_to_global(
            entities=prepared_group_entities,
            group=group,
            full_text=source_text,
        )
        all_entities.extend(restored_entities)
        group_directory = _build_large_document_group_directory(entities=prepared_group_entities)
        group_statistics = dict(group_result.get("statistics") or {})
        group_runs.append(
            {
                "group_index": group_index,
                "page_numbers": list(group.get("page_numbers") or []),
                "start": int(group.get("start", 0)),
                "end": int(group.get("end", 0)),
                "group_dir": str(group_dir),
                "group_source_path": group_source_path,
                "group_source_filename": group_source_name,
                "output_path": group_export_result.get("output_path"),
                "output_filename": group_export_result.get("download_name"),
                "mapping_output_path": group_export_result.get("mapping_output_path"),
                "mapping_output_filename": group_export_result.get("mapping_download_name"),
                "detected_entity_count": len(detected_group_entities),
                "prepared_entity_count": len(prepared_group_entities),
                "restored_entity_count": len(restored_entities),
                "mapping_row_count": len(group_directory),
                "mapping_directory": group_directory[:100],
                "statistics": group_statistics,
                "anonymized_text_length": len(group_anonymized_text),
                "preserves_format": bool(group_export_result.get("preserves_format")),
                "quality_gate_passed": bool(group_quality.get("quality_gate_passed", True)),
                "quality_gate_reason": str(group_quality.get("quality_gate_reason") or ""),
            }
        )

    all_entities.sort(key=lambda item: (_coerce_int(item.get("start"), 10**12), _coerce_int(item.get("end"), 10**12)))

    _write_progress(
        progress_path,
        {
            "stage": "large_document_global_directory_merge",
            "current": 1,
            "total": 4,
            "message": "正在合并大文件总目录并按主体类型分池识别同一主体...",
        },
    )
    engine = get_engine()
    contextual_service = engine.pipeline_manager.contextual_desensitizer
    strategy_key = _large_document_strategy_key(payload)
    remapped_entities, global_directory_metadata = await _merge_large_document_global_directory(
        contextual_service=contextual_service,
        text=source_text,
        entities=all_entities,
        strategy_key=strategy_key,
        use_llm=bool(payload.get("use_llm")),
        llm_model=payload.get("llm_model"),
    )
    remapped_entities.sort(
        key=lambda item: (_coerce_int(item.get("start"), 10**12), _coerce_int(item.get("end"), 10**12))
    )

    _write_progress(
        progress_path,
        {
            "stage": "large_document_global_mapping_export",
            "current": 2,
            "total": 4,
            "message": "正在导出合并后的大文件总目录...",
        },
    )
    global_mapping_result = _export_large_document_mapping_docx(
        payload=payload,
        entities=remapped_entities,
    )

    _write_progress(
        progress_path,
        {
            "stage": "large_document_group_rewrite",
            "current": 3,
            "total": 4,
            "message": "正在按合并总目录回写每组子目录和子文件...",
        },
    )
    exporter = DocumentExporter()
    rewritten_anonymized_parts: List[str] = []
    for index, (group, group_run) in enumerate(zip(groups, group_runs), start=1):
        _write_progress(
            progress_path,
            {
                "stage": "large_document_group_rewrite",
                "current": index,
                "total": len(group_runs),
                "message": f"正在按合并目录回写大文件第 {index}/{len(group_runs)} 组...",
            },
        )
        rewritten_export_result, local_entities, rewritten_text = await _rewrite_large_document_group_from_directory(
            payload=payload,
            group=group,
            group_run=group_run,
            entities=remapped_entities,
            engine=engine,
            exporter=exporter,
        )
        updated_group_directory = _build_large_document_group_directory(entities=local_entities)
        group_run["output_path"] = rewritten_export_result.get("output_path")
        group_run["output_filename"] = rewritten_export_result.get("download_name")
        group_run["mapping_output_path"] = rewritten_export_result.get("mapping_output_path")
        group_run["mapping_output_filename"] = rewritten_export_result.get("mapping_download_name")
        group_run["mapping_row_count"] = len(updated_group_directory)
        group_run["mapping_directory"] = updated_group_directory[:100]
        group_run["global_directory_rewritten"] = True
        group_run["global_subject_ids"] = sorted(
            {
                _coerce_int((entity.get("metadata") or {}).get("global_large_document_subject_id"), -1)
                for entity in local_entities
                if _coerce_int((entity.get("metadata") or {}).get("global_large_document_subject_id"), -1) >= 0
            }
        )
        group_run["anonymized_text_length"] = len(rewritten_text)
        group_run["preserves_format"] = bool(rewritten_export_result.get("preserves_format"))
        rewritten_anonymized_parts.append(rewritten_text)

    _write_progress(
        progress_path,
        {
            "stage": "large_document_final_merge",
            "current": 4,
            "total": 4,
            "message": "正在合并每组回写后的文件形成最终导出文件...",
        },
    )
    final_primary_result = _merge_large_document_rewritten_outputs(
        payload=payload,
        groups=groups,
        group_runs=group_runs,
        source_text=source_text,
        entities=remapped_entities,
    )
    export_result = dict(final_primary_result)
    export_result.update(global_mapping_result)
    anonymized_text = _join_large_document_rewritten_text(
        source_text=source_text,
        groups=groups,
        rewritten_group_texts=rewritten_anonymized_parts,
    )

    quality_metadata = {
        "quality_gate_passed": all(bool(item.get("quality_gate_passed", True)) for item in group_runs),
        "quality_gate_reason": "; ".join(
            item["quality_gate_reason"]
            for item in group_runs
            if str(item.get("quality_gate_reason") or "").strip()
        ),
        "large_document_mode": True,
        "execution_mode": "large_document_grouped_default_line_full_export",
        "large_document_execution_mode": "page_groups_default_line_then_global_directory_rewrite",
        "large_document_front_half_only": False,
        "large_document_group_page_count": LARGE_DOCUMENT_GROUP_PAGE_COUNT,
        "large_document_group_count": len(groups),
        "large_document_group_runs": group_runs[:50],
        "large_document_group_outputs_materialized": True,
        "large_document_group_output_root": str(group_root),
        "large_document_group_rewrite_order": [int(group.get("index", 0)) for group in groups],
        "large_document_global_directory_merge_pending": False,
        "large_document_global_directory_exported": True,
        "large_document_group_directories_rewritten_from_global_directory": True,
        "large_document_group_files_rewritten_from_group_directories": True,
        "large_document_final_file_merged_from_rewritten_groups": True,
        **global_directory_metadata,
    }
    quality_metadata = _attach_directory_quality_metadata(quality_metadata, remapped_entities)
    quality_metadata = _attach_docx_export_quality_metadata(
        quality_metadata,
        source_metadata=payload.get("source_metadata"),
        export_result=export_result,
    )
    manifest_path = _write_large_document_group_manifest(
        payload=payload,
        group_root=group_root,
        group_runs=group_runs,
        quality_metadata=quality_metadata,
        final_export_result=export_result,
    )
    archive_path, archive_name = _create_large_document_group_archive(
        payload=payload,
        group_root=group_root,
    )
    quality_metadata["large_document_group_manifest_path"] = str(manifest_path)
    quality_metadata["large_document_group_archive_path"] = archive_path

    _write_progress(
        progress_path,
        {
            "stage": "finalize",
            "current": 1,
            "total": 1,
            "message": "大文件总目录与最终导出文件已生成，正在整理组产物包...",
        },
    )
    return {
        "entities": remapped_entities,
        "quality_metadata": quality_metadata,
        "anonymized_text": anonymized_text,
        "export_result": export_result,
    }


async def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    progress_path = str(payload.get("progress_path") or "").strip() or None
    with desensitize_mode_context(payload.get("desensitize_mode")):
        if _should_run_large_document_grouped_workflow(payload):
            return await _run_large_document_grouped_workflow(
                payload=payload,
                progress_path=progress_path,
            )

        exporter = DocumentExporter()
        engine = get_engine()
        _write_progress(progress_path, {"stage": "prepare_entities", "current": 1, "total": 4, "message": "正在准备实体与一致性修复..."})
        entities = await engine.prepare_entities_for_anonymization(
            text=str(payload.get("source_text") or ""),
            entities=list(payload.get("entities") or []),
            use_llm=bool(payload.get("use_llm")),
            operator_config=payload.get("operator_config"),
            llm_model=payload.get("llm_model"),
            anonymization_strategy=payload.get("anonymization_strategy"),
            source_metadata=payload.get("source_metadata") or {},
            source_structure=payload.get("source_structure"),
        )
        quality_metadata = engine.get_last_quality_metadata()
        coverage_first_final_export = build_coverage_first_final_export_bundle(
            entities=entities,
            source_text=str(payload.get("source_text") or ""),
        )
        if coverage_first_final_export.get("enabled"):
            quality_metadata = _attach_coverage_first_final_export_metadata(
                quality_metadata,
                coverage_first_final_export,
            )

        _write_progress(progress_path, {"stage": "anonymize_text", "current": 2, "total": 4, "message": "正在生成脱敏文本..."})
        anonymized_text = await engine.anonymize(
            text=str(payload.get("source_text") or ""),
            entities=entities,
            operator_config=payload.get("operator_config"),
        )

        _write_progress(progress_path, {"stage": "export_file", "current": 3, "total": 4, "message": "正在导出结果文件..."})
        export_result = exporter.export(
            task_id=str(payload.get("task_id") or ""),
            source_path=str(payload.get("source_path") or ""),
            original_filename=str(payload.get("original_filename") or ""),
            source_text=str(payload.get("source_text") or ""),
            source_metadata=payload.get("source_metadata") or {},
            source_structure=payload.get("source_structure"),
            entities=entities,
            anonymized_text=anonymized_text,
            operator_config=payload.get("operator_config"),
            coverage_first_final_export=coverage_first_final_export,
        )
        quality_metadata = _attach_docx_export_quality_metadata(
            quality_metadata,
            source_metadata=payload.get("source_metadata"),
            export_result=export_result,
        )
        quality_metadata = _attach_directory_quality_metadata(quality_metadata, entities)

        _write_progress(progress_path, {"stage": "finalize", "current": 4, "total": 4, "message": "正在整理导出结果..."})
        return {
            "entities": entities,
            "quality_metadata": quality_metadata,
            "anonymized_text": anonymized_text,
            "export_result": export_result,
        }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.process_worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = asyncio.run(_run(payload))
        output_path.write_text(json.dumps({"ok": True, **result}, ensure_ascii=False), encoding="utf-8")
        ensure_private_file(output_path)
        return 0
    except Exception as exc:
        error_payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            output_path.write_text(json.dumps(error_payload, ensure_ascii=False), encoding="utf-8")
            ensure_private_file(output_path)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
