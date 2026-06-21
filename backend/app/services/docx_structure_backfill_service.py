"""DOCX structure-aware deterministic backfill for the normal recognition flow."""

from __future__ import annotations

import re
from typing import Any, List

from app.core.recognizer_base import RecognizerResult
from app.services.contract_structure_backfill_service import ContractStructureBackfillService
from app.services.lowmem_entity_utils import (
    ORG_PATTERN,
    deduplicate_results,
    infer_semantic_type,
    iter_docx_structure_units,
    is_probable_person,
    looks_like_organization_short_name,
    remap_sanitized_span,
    resolve_docx_unit_spans,
)


class DocxStructureBackfillService:
    """Run contract field backfill on DOCX structural units with exact source offsets."""

    SOURCE = "docx_structure_backfill"
    STRUCTURE_CONTAINERS = {
        "table_cell",
        "textbox",
        "header",
        "footer",
        "footnote",
        "endnote",
        "paragraph",
    }
    REVIEW_REQUIRED_CONTAINERS = {
        "comment",
        "chart",
        "diagram",
        "glossary",
        "xml_text",
    }

    def __init__(self, backfill_service: ContractStructureBackfillService | None = None) -> None:
        self.backfill_service = backfill_service or ContractStructureBackfillService()

    def extract(
        self,
        *,
        text: str,
        source_structure: dict[str, Any] | None,
    ) -> List[RecognizerResult]:
        units = resolve_docx_unit_spans(text or "", self._extract_units(source_structure))
        if not text or not units:
            return []

        results: list[RecognizerResult] = []
        for unit in units:
            results.extend(self._extract_from_unit(text, unit))
        results.extend(self._extract_cross_unit_table_label_values(text, units))

        return deduplicate_results(results)

    def _extract_from_unit(self, source_text: str, unit: dict[str, Any]) -> List[RecognizerResult]:
        unit_text = str(unit.get("text") or "")
        if not unit_text.strip():
            return []
        start = self._coerce_int(unit.get("_resolved_start"), -1)
        end = self._coerce_int(unit.get("_resolved_end"), -1)
        if start < 0 or end <= start:
            start = self._coerce_int(unit.get("start"), -1)
            end = self._coerce_int(unit.get("end"), -1)
        if start < 0 or end <= start or source_text[start:end] != unit_text:
            return []

        container_type = str(unit.get("container_type") or unit.get("unit_type") or "")
        if container_type not in self.STRUCTURE_CONTAINERS and container_type not in self.REVIEW_REQUIRED_CONTAINERS:
            return []
        if not self._has_sensitive_cue(unit_text, container_type):
            return []

        local_results = [
            *self._extract_structural_label_values(unit_text, container_type),
            *self._extract_org_suffix_entities(unit_text, container_type),
            *self.backfill_service.extract(unit_text),
        ]
        mapped: list[RecognizerResult] = []
        for local in local_results:
            global_start = start + local.start
            global_end = start + local.end
            if global_start < start or global_end > end or source_text[global_start:global_end] != local.text:
                continue
            metadata = dict(local.metadata or {})
            metadata.update(self._unit_metadata(unit, container_type))
            mapped.append(
                RecognizerResult(
                    entity_type=local.entity_type,
                    start=global_start,
                    end=global_end,
                    score=max(float(local.score or 0.0), self._score_for_container(container_type)),
                    text=source_text[global_start:global_end],
                    source=self.SOURCE,
                    metadata=metadata,
                )
            )
        return mapped

    def _extract_structural_label_values(
        self,
        unit_text: str,
        container_type: str,
    ) -> List[RecognizerResult]:
        if container_type not in {"table_cell", "textbox", "header", "footer", "footnote", "endnote", "paragraph"}:
            return []
        label_specs = {str(spec["label"]): dict(spec) for spec in self.backfill_service.LABEL_SPECS}
        results: list[RecognizerResult] = []
        results.extend(self._extract_adjacent_cell_label_values(unit_text, label_specs))
        results.extend(self._extract_single_space_label_values(unit_text, label_specs))
        return deduplicate_results(results)

    def _extract_cross_unit_table_label_values(
        self,
        source_text: str,
        units: List[dict[str, Any]],
    ) -> List[RecognizerResult]:
        label_specs = {str(spec["label"]): dict(spec) for spec in self.backfill_service.LABEL_SPECS}
        table_units = [unit for unit in units if self._is_table_unit(unit)]
        if len(table_units) < 2:
            return []

        by_table_row: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for unit in table_units:
            table_index = self._coerce_int(unit.get("table_index"), 0)
            row_index = self._coerce_int(unit.get("row_index"), 0)
            if table_index <= 0 or row_index <= 0:
                continue
            part_name = str(unit.get("part_name") or "")
            by_table_row.setdefault((part_name, table_index, row_index), []).append(unit)

        results: list[RecognizerResult] = []
        ordered_row_keys = sorted(by_table_row, key=lambda item: (item[0], item[1], item[2]))
        row_lookup = {key: self._ordered_row_units(value) for key, value in by_table_row.items()}
        for row_key in ordered_row_keys:
            row_units = row_lookup.get(row_key) or []
            if not row_units:
                continue
            results.extend(
                self._extract_same_row_unit_label_values(
                    source_text=source_text,
                    row_units=row_units,
                    label_specs=label_specs,
                )
            )
            next_row_key = (row_key[0], row_key[1], row_key[2] + 1)
            next_row_units = row_lookup.get(next_row_key) or []
            if next_row_units:
                results.extend(
                    self._extract_stacked_row_label_values(
                        source_text=source_text,
                        label_units=row_units,
                        value_units=next_row_units,
                        label_specs=label_specs,
                    )
                )
        return deduplicate_results(results)

    def _extract_same_row_unit_label_values(
        self,
        *,
        source_text: str,
        row_units: list[dict[str, Any]],
        label_specs: dict[str, dict[str, Any]],
    ) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        for index, unit in enumerate(row_units[:-1]):
            label = self._normalize_label(str(unit.get("text") or ""), label_specs)
            if not label:
                continue
            value_unit = row_units[index + 1]
            result = self._make_cross_unit_label_value_result(
                source_text=source_text,
                label=label,
                value_unit=value_unit,
                trigger="docx_adjacent_table_unit",
            )
            if result:
                results.append(result)
        return results

    def _extract_stacked_row_label_values(
        self,
        *,
        source_text: str,
        label_units: list[dict[str, Any]],
        value_units: list[dict[str, Any]],
        label_specs: dict[str, dict[str, Any]],
    ) -> List[RecognizerResult]:
        results: list[RecognizerResult] = []
        max_count = min(len(label_units), len(value_units))
        for index in range(max_count):
            label = self._normalize_label(str(label_units[index].get("text") or ""), label_specs)
            if not label:
                continue
            result = self._make_cross_unit_label_value_result(
                source_text=source_text,
                label=label,
                value_unit=value_units[index],
                trigger="docx_stacked_table_unit",
            )
            if result:
                results.append(result)
        return results

    def _make_cross_unit_label_value_result(
        self,
        *,
        source_text: str,
        label: str,
        value_unit: dict[str, Any],
        trigger: str,
    ) -> RecognizerResult | None:
        unit_text = str(value_unit.get("text") or "")
        if not unit_text.strip():
            return None
        unit_start, unit_end = self._resolve_unit_span(source_text, value_unit)
        if unit_start < 0 or unit_end <= unit_start or source_text[unit_start:unit_end] != unit_text:
            return None
        local = self._make_label_value_result(
            unit_text,
            label=label,
            raw_value=unit_text,
            value_start=0,
            value_end=len(unit_text),
            trigger=trigger,
        )
        if not local:
            return None
        global_start = unit_start + local.start
        global_end = unit_start + local.end
        if global_start < unit_start or global_end > unit_end or source_text[global_start:global_end] != local.text:
            return None
        container_type = str(value_unit.get("container_type") or value_unit.get("unit_type") or "")
        metadata = dict(local.metadata or {})
        metadata.update(self._unit_metadata(value_unit, container_type))
        return RecognizerResult(
            entity_type=local.entity_type,
            start=global_start,
            end=global_end,
            score=max(float(local.score or 0.0), self._score_for_container(container_type)),
            text=source_text[global_start:global_end],
            source=self.SOURCE,
            metadata=metadata,
        )

    @classmethod
    def _resolve_unit_span(cls, source_text: str, unit: dict[str, Any]) -> tuple[int, int]:
        unit_text = str(unit.get("text") or "")
        start = cls._coerce_int(unit.get("_resolved_start"), -1)
        end = cls._coerce_int(unit.get("_resolved_end"), -1)
        if start >= 0 and end > start and source_text[start:end] == unit_text:
            return start, end
        start = cls._coerce_int(unit.get("start"), -1)
        end = cls._coerce_int(unit.get("end"), -1)
        if start >= 0 and end > start and source_text[start:end] == unit_text:
            return start, end
        return -1, -1

    def _extract_org_suffix_entities(
        self,
        unit_text: str,
        container_type: str,
    ) -> List[RecognizerResult]:
        if container_type not in {
            "paragraph",
            "table_cell",
            "textbox",
            "header",
            "footer",
            "footnote",
            "endnote",
            "comment",
            "chart",
            "diagram",
            "glossary",
            "xml_text",
        }:
            return []
        results: list[RecognizerResult] = []
        sanitized_text, index_map = self._sanitize_docx_unit_for_org_matching(unit_text)
        for match in ORG_PATTERN.finditer(sanitized_text):
            sanitized_start, sanitized_end = self.backfill_service._trim_org_noise_prefix_span(
                sanitized_text,
                match.start(),
                match.end(),
            )
            sanitized_start, sanitized_end = self._trim_org_context_prefix_span(
                sanitized_text,
                sanitized_start,
                sanitized_end,
            )
            span = remap_sanitized_span(index_map, sanitized_start, sanitized_end)
            if span is None:
                continue
            start, end = span
            start, end = self._trim_org_original_delimited_prefix_span(unit_text, start, end)
            if end <= start:
                continue
            entity_text = unit_text[start:end]
            normalized_entity_text = "".join(entity_text.split())
            entity_type = "COURT" if "法院" in normalized_entity_text else "ORGANIZATION"
            result = RecognizerResult(
                entity_type=entity_type,
                start=start,
                end=end,
                score=0.9 if container_type in {"paragraph", "table_cell", "textbox"} else 0.86,
                text=entity_text,
                source="contract_structure_backfill",
                metadata={
                    "source": "contract_structure_backfill",
                    "trigger": "docx_structure_org_suffix",
                    "label": "组织机构后缀",
                    "normalized_text": normalized_entity_text,
                },
            )
            results.append(result)
        return results

    @staticmethod
    def _trim_org_original_delimited_prefix_span(text: str, start: int, end: int) -> tuple[int, int]:
        value = text[start:end]
        for match in re.finditer(r"[\s\t\r\n，,。；;：:、（）()《》【】]", value):
            candidate = value[match.end() :].lstrip(" \t\r\n，,。；;：:、")
            if len(candidate) < 2:
                continue
            compact_candidate = re.sub(r"\s+", "", candidate)
            if ORG_PATTERN.fullmatch(compact_candidate):
                if DocxStructureBackfillService._looks_like_inline_spaced_entity(candidate):
                    continue
                return end - len(candidate), end
        return start, end

    @staticmethod
    def _looks_like_inline_spaced_entity(value: str) -> bool:
        tokens = [token for token in re.split(r"\s+", value.strip()) if token]
        if len(tokens) < 3:
            return False
        short_token_count = sum(1 for token in tokens if len(token) <= 2 and re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]{1,2}", token))
        return short_token_count >= max(3, len(tokens) - 1)

    @staticmethod
    def _sanitize_docx_unit_for_org_matching(text: str) -> tuple[str, list[int]]:
        if not text:
            return "", []

        sanitized_chars: list[str] = []
        index_map: list[int] = []
        cursor = 0
        text_length = len(text)
        while cursor < text_length:
            char = text[cursor]
            if char in "\r\n\t":
                sanitized_chars.append(char)
                index_map.append(cursor)
                cursor += 1
                continue
            if char.isspace():
                space_end = cursor + 1
                while space_end < text_length and text[space_end].isspace() and text[space_end] not in "\r\n\t":
                    space_end += 1
                left_char = sanitized_chars[-1] if sanitized_chars else ""
                right_char = text[space_end] if space_end < text_length else ""
                if DocxStructureBackfillService._is_inline_content_space(left_char, right_char):
                    cursor = space_end
                    continue
                for original_index in range(cursor, space_end):
                    sanitized_chars.append(text[original_index])
                    index_map.append(original_index)
                cursor = space_end
                continue
            sanitized_chars.append(char)
            index_map.append(cursor)
            cursor += 1
        return "".join(sanitized_chars), index_map

    @staticmethod
    def _is_inline_content_space(left_char: str, right_char: str) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]", left_char or ""):
            return False
        if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z0-9]", right_char or ""):
            return False
        return bool(
            re.fullmatch(r"[\u4e00-\u9fa5]", left_char)
            or re.fullmatch(r"[\u4e00-\u9fa5]", right_char)
            or re.fullmatch(r"\d", left_char)
            or re.fullmatch(r"\d", right_char)
        )

    @staticmethod
    def _trim_org_context_prefix_span(text: str, start: int, end: int) -> tuple[int, int]:
        value = text[start:end]
        if len(value) <= 4:
            return start, end
        token_match = re.search(r"[\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,48}$", value)
        if token_match:
            candidate = token_match.group()
            if token_match.start() > 0 and ORG_PATTERN.fullmatch(candidate):
                return end - len(candidate), end
        boundary_pattern = re.compile(
            r"(?:^|[\s\t\r\n，,。；;：:、（）()《》【】])"
            r"(?:[\u4e00-\u9fa5]{0,12}?(?:过程中|期间|阶段|环节|项下|之下|内|后|前|时|由|与|和|及|向|对))$"
        )
        for index in range(1, len(value) - 1):
            candidate = value[index:]
            prefix = value[:index]
            if not ORG_PATTERN.fullmatch(candidate):
                continue
            if boundary_pattern.search(prefix):
                return end - len(candidate), end
        return start, end

    def _extract_adjacent_cell_label_values(
        self,
        unit_text: str,
        label_specs: dict[str, dict[str, Any]],
    ) -> List[RecognizerResult]:
        cells = self._split_table_cells(unit_text)
        if len(cells) < 2:
            return []
        results: list[RecognizerResult] = []
        for index, (cell_text, _cell_start, _cell_end) in enumerate(cells[:-1]):
            label = self._normalize_label(cell_text, label_specs)
            if not label:
                continue
            value_text, value_start, value_end = cells[index + 1]
            result = self._make_label_value_result(
                unit_text,
                label=label,
                raw_value=value_text,
                value_start=value_start,
                value_end=value_end,
                trigger="docx_adjacent_table_cell",
            )
            if result:
                results.append(result)
        return results

    def _extract_single_space_label_values(
        self,
        unit_text: str,
        label_specs: dict[str, dict[str, Any]],
    ) -> List[RecognizerResult]:
        label_pattern = "|".join(re.escape(label) for label in sorted(label_specs, key=len, reverse=True))
        if not label_pattern:
            return []
        pattern = re.compile(
            rf"(?<![\u4e00-\u9fa5A-Za-z0-9])(?P<label>{label_pattern})"
            rf"{self.backfill_service.LABEL_VARIANT_SUFFIX}"
            r"(?:[（(][^）)]{1,30}[）)])?"
            r"(?:[:：]|\s+)"
            r"(?P<value>[^\t\n\r；;。]{2,100})"
        )
        results: list[RecognizerResult] = []
        for match in pattern.finditer(unit_text):
            label = self._normalize_label(match.group("label"), label_specs)
            if not label:
                continue
            result = self._make_label_value_result(
                unit_text,
                label=label,
                raw_value=match.group("value"),
                value_start=match.start("value"),
                value_end=match.end("value"),
                trigger="docx_inline_structure_label",
            )
            if result:
                results.append(result)
        return results

    def _make_label_value_result(
        self,
        unit_text: str,
        *,
        label: str,
        raw_value: str,
        value_start: int,
        value_end: int,
        trigger: str,
    ) -> RecognizerResult | None:
        spec = next((item for item in self.backfill_service.LABEL_SPECS if str(item["label"]) == label), None)
        if not spec:
            return None
        entity_type = str(spec["type"])
        value = self.backfill_service._clean_label_value(raw_value, entity_type, label)
        if not value and entity_type == "PERSON":
            compact_value = re.sub(r"\s+", "", raw_value or "")
            if is_probable_person(compact_value):
                value = compact_value
        if not value:
            return None
        local_offset = raw_value.find(value)
        if local_offset < 0:
            compact_raw = re.sub(r"\s+", "", raw_value)
            compact_value = re.sub(r"\s+", "", value)
            compact_index = compact_raw.find(compact_value)
            if compact_index < 0:
                return None
            cursor = 0
            mapped_start = None
            mapped_end = None
            for raw_index, char in enumerate(raw_value):
                if char.isspace():
                    continue
                if cursor == compact_index:
                    mapped_start = raw_index
                cursor += 1
                if cursor == compact_index + len(compact_value):
                    mapped_end = raw_index + 1
                    break
            if mapped_start is None or mapped_end is None:
                return None
            start = value_start + mapped_start
            end = value_start + mapped_end
        else:
            start = value_start + local_offset
            end = start + len(value)
        if start < value_start or end > value_end or start < 0 or end <= start:
            return None

        resolved_type = entity_type
        if entity_type == "ROLE_SUBJECT":
            resolved_type = infer_semantic_type(value, label)
            if not resolved_type and looks_like_organization_short_name(value):
                resolved_type = "ORGANIZATION"
        elif entity_type == "ORGANIZATION" and is_probable_person(value):
            resolved_type = "PERSON"
        elif entity_type == "ORGANIZATION":
            resolved_type = infer_semantic_type(value, label)
            if not resolved_type and looks_like_organization_short_name(value):
                resolved_type = "ORGANIZATION"
        if not resolved_type:
            return None
        metadata = {
            "source": "contract_structure_backfill",
            "trigger": trigger,
            "label": label,
            "role": spec.get("role"),
            "normalized_text": re.sub(r"\s+", "", unit_text[start:end]),
        }
        if resolved_type == "ORGANIZATION" and looks_like_organization_short_name(value):
            metadata["short_org_candidate"] = True
            metadata["requires_manual_review"] = True

        return RecognizerResult(
            entity_type=resolved_type,
            start=start,
            end=end,
            score=0.95,
            text=unit_text[start:end],
            source="contract_structure_backfill",
            metadata=metadata,
        )

    def _split_table_cells(self, unit_text: str) -> list[tuple[str, int, int]]:
        cells: list[tuple[str, int, int]] = []
        cursor = 0
        for piece in unit_text.split("\t"):
            raw_start = cursor
            raw_end = cursor + len(piece)
            stripped = piece.strip()
            if stripped:
                leading = len(piece) - len(piece.lstrip())
                start = raw_start + leading
                cells.append((stripped, start, start + len(stripped)))
            cursor = raw_end + 1
        return cells

    def _normalize_label(self, text: str, label_specs: dict[str, dict[str, Any]]) -> str:
        compact = re.sub(r"[\s:：，,。；;（）()《》【】\"“”'`]", "", text or "")
        compact = re.sub(r"[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3}$", "", compact)
        if compact in label_specs:
            return compact
        for label in sorted(label_specs, key=len, reverse=True):
            if compact.startswith(label):
                suffix = compact[len(label):]
                if not suffix or re.fullmatch(r"[一二三四五六七八九十甲乙丙丁ABCDEFabcdef\d]{1,3}", suffix):
                    return label
        return ""

    def _unit_metadata(self, unit: dict[str, Any], container_type: str) -> dict[str, Any]:
        rewrite_policy = str(unit.get("rewrite_policy") or "exact")
        review_required = (
            rewrite_policy != "exact"
            or container_type in self.REVIEW_REQUIRED_CONTAINERS
        )
        return {
            "source": self.SOURCE,
            "trigger": "docx_structure_unit",
            "base_source": "contract_structure_backfill",
            "docx_unit_id": unit.get("unit_id"),
            "docx_part_name": unit.get("part_name"),
            "docx_container_type": container_type,
            "docx_unit_type": unit.get("unit_type"),
            "docx_table_index": unit.get("table_index"),
            "docx_row_index": unit.get("row_index"),
            "docx_col_index": unit.get("col_index"),
            "docx_rewrite_policy": rewrite_policy,
            "docx_span_resolution": unit.get("_span_resolution"),
            "docx_resolved_order_index": unit.get("_resolved_order_index"),
            "docx_review_required": bool(review_required),
        }

    def _extract_units(self, source_structure: dict[str, Any] | None) -> List[dict[str, Any]]:
        return [dict(unit) for unit in iter_docx_structure_units(source_structure)]

    def _append_unit(self, units: list[dict[str, Any]], seen: set[str], unit: Any) -> None:
        if not isinstance(unit, dict):
            return
        copied = dict(unit)
        unit_id = str(copied.get("unit_id") or "")
        if unit_id:
            if unit_id in seen:
                return
            seen.add(unit_id)
        units.append(copied)

    def _is_table_unit(self, unit: dict[str, Any]) -> bool:
        container_type = str(unit.get("container_type") or unit.get("unit_type") or "")
        if container_type not in {"table_cell", "table_row"}:
            return False
        return bool(str(unit.get("text") or "").strip())

    def _ordered_row_units(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            units,
            key=lambda unit: (
                self._coerce_int(unit.get("col_index"), 0),
                self._coerce_int(unit.get("start"), 0),
                str(unit.get("unit_id") or ""),
            ),
        )

    def _has_sensitive_cue(self, unit_text: str, container_type: str) -> bool:
        compact = "".join(unit_text.split())
        if not compact:
            return False
        if any(label in compact for label in self.backfill_service.FOLLOWING_FIELD_LABELS):
            return True
        if any(keyword in compact for keyword in ("以下简称", "下称", "简称", "又称", "签章", "盖章", "落款")):
            return True
        if container_type in self.STRUCTURE_CONTAINERS:
            return bool(ORG_PATTERN.search(compact))
        return container_type in self.REVIEW_REQUIRED_CONTAINERS and len(compact) >= 2

    def _score_for_container(self, container_type: str) -> float:
        if container_type in self.REVIEW_REQUIRED_CONTAINERS:
            return 0.86
        if container_type in {"table_cell", "textbox"}:
            return 0.94
        if container_type in {"header", "footer", "footnote", "endnote"}:
            return 0.9
        return 0.88

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
