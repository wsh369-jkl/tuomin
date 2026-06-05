"""One-shot worker for prepare/anonymize/export."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import desensitize_mode_context, settings
from app.core.document_workflow import classify_document_workflow
from app.core.runtime_security import ensure_private_file
from app.engine.desensitization_engine import get_engine
from app.processors.document_exporter import DocumentExporter


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
LOCAL_CANONICAL_KEY_PATTERN = re.compile(
    r"^(?:ORG|PERSON|PROJECT|LOCATION|ADDRESS|COURT|BANK|ALIAS|COMPANY)_\d+$"
)


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
    step = max(1, int(group_page_count or LARGE_DOCUMENT_GROUP_PAGE_COUNT))
    for offset in range(0, len(pages), step):
        group_pages = pages[offset:offset + step]
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


def _build_group_metadata(
    *,
    source_metadata: Any,
    group: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = dict(source_metadata or {})
    page_count = int(group.get("page_count", 1) or 1)
    metadata["pages"] = page_count
    metadata["page_count"] = page_count
    metadata["_force_standard_document_workflow"] = True
    metadata["large_document_group_index"] = int(group.get("index", 0))
    metadata["large_document_group_pages"] = list(group.get("page_numbers") or [])
    return metadata


def _build_group_structure(group: Dict[str, Any]) -> Dict[str, Any]:
    group_start = int(group.get("start", 0))
    group_text = str(group.get("text") or "")
    local_pages: List[Dict[str, Any]] = []
    for page in group.get("pages") or []:
        start = int(page.get("start", group_start))
        end = int(page.get("end", start))
        local_start = max(0, start - group_start)
        local_end = max(local_start, min(len(group_text), end - group_start))
        local_pages.append(
            {
                "page_number": int(page.get("page_number", len(local_pages) + 1) or (len(local_pages) + 1)),
                "text": group_text[local_start:local_end],
                "start": local_start,
                "end": local_end,
            }
        )
    return {
        "page_count": max(1, len(local_pages)),
        "pages": local_pages,
        "_force_standard_document_workflow": True,
    }


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


def _reassign_large_document_global_replacements(
    *,
    contextual_service: Any,
    entities: List[Dict[str, Any]],
    strategy_key: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_indexes: List[int] = []
    families_by_index: Dict[int, str] = {}
    for index, entity in enumerate(entities):
        family = _global_subject_family(contextual_service, entity)
        if family not in GLOBAL_SUBJECT_REPLACEMENT_FAMILIES:
            continue
        replacement = _normalize_replacement(entity.get("replacement"))
        if not replacement:
            continue
        candidate_indexes.append(index)
        families_by_index[index] = family

    if not candidate_indexes:
        return entities, {
            "global_subject_count": 0,
            "replacement_conflict_count": 0,
            "global_replacement_directory": [],
        }

    uf = _UnionFind(len(candidate_indexes))
    local_pos_by_entity_index = {entity_index: pos for pos, entity_index in enumerate(candidate_indexes)}
    token_owner: Dict[str, int] = {}
    local_replacement_owner: Dict[tuple[int, str, str], int] = {}

    for entity_index in candidate_indexes:
        entity = entities[entity_index]
        pos = local_pos_by_entity_index[entity_index]
        family = families_by_index[entity_index]
        for token in _subject_tokens_for_entity(contextual_service, entity, family):
            owner = token_owner.get(token)
            if owner is None:
                token_owner[token] = pos
            else:
                uf.union(owner, pos)

        metadata = dict(entity.get("metadata") or {})
        group_index = _coerce_int(metadata.get("large_document_group_index"), -1)
        replacement = _normalize_replacement(entity.get("replacement"))
        if strategy_key in {"symbolic_codes", "serial_roles"} and group_index >= 0 and replacement:
            key = (group_index, family, replacement)
            owner = local_replacement_owner.get(key)
            if owner is None:
                local_replacement_owner[key] = pos
            else:
                uf.union(owner, pos)

    subject_members: Dict[int, List[int]] = {}
    for entity_index in candidate_indexes:
        root = uf.find(local_pos_by_entity_index[entity_index])
        subject_members.setdefault(root, []).append(entity_index)

    subjects: List[Dict[str, Any]] = []
    for subject_index, member_indexes in enumerate(subject_members.values()):
        member_indexes.sort(key=lambda idx: (_coerce_int(entities[idx].get("start"), 10**12), idx))
        first_entity = entities[member_indexes[0]]
        family = families_by_index[member_indexes[0]]
        replacements: Dict[str, int] = {}
        variants: set[str] = set()
        for member_index in member_indexes:
            member = entities[member_index]
            replacement = _normalize_replacement(member.get("replacement"))
            if replacement:
                replacements.setdefault(replacement, _coerce_int(member.get("start"), 10**12))
            text_token = str(member.get("text") or "").strip()
            if text_token:
                variants.add(text_token)
        earliest_replacement = sorted(replacements.items(), key=lambda item: (item[1], item[0]))[0][0]
        subjects.append(
            {
                "id": subject_index,
                "family": family,
                "member_indexes": member_indexes,
                "first_start": _coerce_int(first_entity.get("start"), 10**12),
                "primary_text": str(first_entity.get("text") or "").strip(),
                "earliest_replacement": earliest_replacement,
                "variants": sorted(variants, key=lambda item: (-len(item), item))[:12],
            }
        )

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
                "primary_text": subject["primary_text"],
                "replacement": replacement,
                "first_start": subject["first_start"],
                "variants": subject["variants"],
                "mention_count": len(subject["member_indexes"]),
            }
        )
    directory.sort(key=lambda item: (int(item["first_start"]), int(item["subject_id"])))

    return remapped_entities, {
        "global_subject_count": len(subjects),
        "replacement_conflict_count": replacement_conflicts,
        "global_replacement_directory": directory[:100],
    }


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


def _should_run_large_document_grouped_workflow(payload: Dict[str, Any]) -> bool:
    if not settings.is_high_quality_lowmem_mode():
        return False
    text = str(payload.get("source_text") or "")
    source_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else None
    source_structure = payload.get("source_structure") if isinstance(payload.get("source_structure"), dict) else None
    workflow = classify_document_workflow(
        text=text,
        source_metadata=source_metadata,
        source_structure=source_structure,
    )
    if not workflow.get("enabled"):
        return False
    return bool(_build_large_document_page_groups(text=text, source_structure=source_structure))


async def _run_large_document_grouped_workflow(
    *,
    payload: Dict[str, Any],
    engine: Any,
    exporter: DocumentExporter,
    progress_path: Optional[str],
) -> Dict[str, Any]:
    source_text = str(payload.get("source_text") or "")
    source_metadata = payload.get("source_metadata") or {}
    source_structure = payload.get("source_structure")
    groups = _build_large_document_page_groups(
        text=source_text,
        source_structure=source_structure,
    )
    if not groups:
        raise RuntimeError("large_document_grouped_workflow_missing_real_pages")

    input_entities = [dict(item) for item in list(payload.get("entities") or []) if isinstance(item, dict)]
    all_entities: List[Dict[str, Any]] = []
    group_runs: List[Dict[str, Any]] = []
    contextual_service = engine.pipeline_manager.contextual_desensitizer

    for group in groups:
        group_index = int(group.get("index", 0))
        _write_progress(
            progress_path,
            {
                "stage": "prepare_entities",
                "current": group_index + 1,
                "total": len(groups),
                "message": f"正在按真实分页处理大文件第 {group_index + 1}/{len(groups)} 组...",
            },
        )
        local_entities = _project_entities_to_group(
            entities=input_entities,
            group=group,
        )
        group_metadata = _build_group_metadata(
            source_metadata=source_metadata,
            group=group,
        )
        group_structure = _build_group_structure(group)
        prepared_group_entities = await engine.prepare_entities_for_anonymization(
            text=str(group.get("text") or ""),
            entities=local_entities,
            use_llm=bool(payload.get("use_llm")),
            operator_config=payload.get("operator_config"),
            llm_model=payload.get("llm_model"),
            anonymization_strategy=payload.get("anonymization_strategy"),
            source_metadata=group_metadata,
            source_structure=group_structure,
        )
        group_quality = dict(engine.get_last_quality_metadata() or {})
        group_anonymized_text = await engine.anonymize(
            text=str(group.get("text") or ""),
            entities=prepared_group_entities,
            operator_config=payload.get("operator_config"),
        )
        restored_entities = _restore_group_entities_to_global(
            entities=prepared_group_entities,
            group=group,
            full_text=source_text,
        )
        all_entities.extend(restored_entities)
        group_directory = _build_large_document_group_directory(entities=restored_entities)
        group_runs.append(
            {
                "group_index": group_index,
                "page_numbers": list(group.get("page_numbers") or []),
                "start": int(group.get("start", 0)),
                "end": int(group.get("end", 0)),
                "input_entity_count": len(local_entities),
                "prepared_entity_count": len(prepared_group_entities),
                "restored_entity_count": len(restored_entities),
                "mapping_row_count": len(group_directory),
                "mapping_directory": group_directory[:100],
                "anonymized_text_length": len(group_anonymized_text),
                "quality_gate_passed": bool(group_quality.get("quality_gate_passed", True)),
                "quality_gate_reason": str(group_quality.get("quality_gate_reason") or ""),
            }
        )

    all_entities.sort(key=lambda item: (_coerce_int(item.get("start"), 10**12), _coerce_int(item.get("end"), 10**12)))
    strategy_key = str(payload.get("anonymization_strategy") or "").strip()
    all_entities, global_mapping_metadata = _reassign_large_document_global_replacements(
        contextual_service=contextual_service,
        entities=all_entities,
        strategy_key=strategy_key,
    )

    anonymized_text = await _anonymize_large_document_groups_in_order(
        engine=engine,
        text=source_text,
        groups=groups,
        entities=all_entities,
        operator_config=payload.get("operator_config"),
        progress_path=progress_path,
    )

    quality_metadata = {
        "quality_gate_passed": all(bool(item.get("quality_gate_passed", True)) for item in group_runs),
        "quality_gate_reason": "; ".join(
            item["quality_gate_reason"]
            for item in group_runs
            if str(item.get("quality_gate_reason") or "").strip()
        ),
        "large_document_mode": True,
        "execution_mode": "large_document_grouped_standard_workflow",
        "large_document_execution_mode": "page_groups_of_10_standard_then_global_directory",
        "large_document_group_page_count": LARGE_DOCUMENT_GROUP_PAGE_COUNT,
        "large_document_group_count": len(groups),
        "large_document_group_runs": group_runs[:50],
        "large_document_group_outputs_materialized": True,
        "large_document_group_rewrite_order": [int(group.get("index", 0)) for group in groups],
        **global_mapping_metadata,
    }

    _write_progress(
        progress_path,
        {
            "stage": "export_file",
            "current": 1,
            "total": 1,
            "message": "正在导出合并后的脱敏文件和汇总对照目录...",
        },
    )
    export_result = exporter.export(
        task_id=str(payload.get("task_id") or ""),
        source_path=str(payload.get("source_path") or ""),
        original_filename=str(payload.get("original_filename") or ""),
        source_text=source_text,
        source_metadata=source_metadata,
        source_structure=source_structure,
        entities=all_entities,
        anonymized_text=anonymized_text,
        operator_config=payload.get("operator_config"),
    )
    return {
        "entities": all_entities,
        "quality_metadata": quality_metadata,
        "anonymized_text": anonymized_text,
        "export_result": export_result,
    }


async def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    progress_path = str(payload.get("progress_path") or "").strip() or None
    exporter = DocumentExporter()
    with desensitize_mode_context(payload.get("desensitize_mode")):
        engine = get_engine()
        if _should_run_large_document_grouped_workflow(payload):
            return await _run_large_document_grouped_workflow(
                payload=payload,
                engine=engine,
                exporter=exporter,
                progress_path=progress_path,
            )

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
        )

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
