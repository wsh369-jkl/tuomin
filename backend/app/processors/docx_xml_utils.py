"""Utilities for reading and updating DOCX XML content, including tracked changes."""

from __future__ import annotations

import fnmatch
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWINGML_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"
CHART_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/chart"
MARKUP_COMPATIBILITY_NAMESPACE = "http://schemas.openxmlformats.org/markup-compatibility/2006"
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
NAMESPACES = {
    "a": DRAWINGML_NAMESPACE,
    "c": CHART_NAMESPACE,
    "m": MATH_NAMESPACE,
    "w": WORD_NAMESPACE,
}

PARAGRAPH_TAG = f"{{{WORD_NAMESPACE}}}p"
TABLE_TAG = f"{{{WORD_NAMESPACE}}}tbl"
TABLE_ROW_TAG = f"{{{WORD_NAMESPACE}}}tr"
TABLE_CELL_TAG = f"{{{WORD_NAMESPACE}}}tc"
TEXTBOX_CONTENT_TAG = f"{{{WORD_NAMESPACE}}}txbxContent"
ALTERNATE_CONTENT_TAG = f"{{{MARKUP_COMPATIBILITY_NAMESPACE}}}AlternateContent"
ALTERNATE_CHOICE_TAG = f"{{{MARKUP_COMPATIBILITY_NAMESPACE}}}Choice"
ALTERNATE_FALLBACK_TAG = f"{{{MARKUP_COMPATIBILITY_NAMESPACE}}}Fallback"
TEXT_TAGS = {
    f"{{{WORD_NAMESPACE}}}t",
    f"{{{WORD_NAMESPACE}}}delText",
    f"{{{DRAWINGML_NAMESPACE}}}t",
    f"{{{MATH_NAMESPACE}}}t",
    f"{{{CHART_NAMESPACE}}}v",
}
TAB_TAG = f"{{{WORD_NAMESPACE}}}tab"
BREAK_TAGS = {
    f"{{{WORD_NAMESPACE}}}br",
    f"{{{WORD_NAMESPACE}}}cr",
}
TRACKED_INSERTION_TAG = f"{{{WORD_NAMESPACE}}}ins"
TRACKED_DELETION_TAG = f"{{{WORD_NAMESPACE}}}del"
FIELD_INSTRUCTION_TAG = f"{{{WORD_NAMESPACE}}}instrText"
HIDDEN_TEXT_TAG = f"{{{WORD_NAMESPACE}}}vanish"
XML_SPACE_ATTR = f"{{{XML_NAMESPACE}}}space"

DOCX_TEXT_PART_PATTERNS = (
    "word/document.xml",
    "word/header*.xml",
    "word/footer*.xml",
    "word/footnotes.xml",
    "word/endnotes.xml",
    "word/comments*.xml",
    "word/glossary/document.xml",
    "word/charts/chart*.xml",
    "word/charts/style*.xml",
    "word/charts/colors*.xml",
    "word/diagrams/data*.xml",
)

_REGISTERED_NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": CHART_NAMESPACE,
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "docPropsVTypes": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "dcterms": "http://purl.org/dc/terms/",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "m": MATH_NAMESPACE,
    "o": "urn:schemas-microsoft-com:office:office",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "ve": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "w": WORD_NAMESPACE,
    "w10": "urn:schemas-microsoft-com:office:word",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "wne": "http://schemas.microsoft.com/office/word/2006/wordml",
    "xml": XML_NAMESPACE,
}

for prefix, uri in _REGISTERED_NAMESPACES.items():
    ET.register_namespace(prefix, uri)


@dataclass
class DocxTextFragment:
    """A contiguous character fragment inside one DOCX visible-text unit."""

    part_name: str
    text: str
    local_start: int
    local_end: int
    node_index: Optional[int] = None
    node: Optional[ET.Element] = field(default=None, repr=False, compare=False)
    virtual: bool = False
    source_indexes: Optional[List[int]] = field(default=None, repr=False, compare=False)
    start: int = 0
    end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_name": self.part_name,
            "node_index": self.node_index,
            "start": int(self.start),
            "end": int(self.end),
            "local_start": int(self.local_start),
            "local_end": int(self.local_end),
            "virtual": bool(self.virtual),
            "text_chars": len(self.text or ""),
        }


@dataclass
class DocxVisibleTextUnit:
    """A visible DOCX text unit with global offsets in the parser text."""

    unit_id: str
    part_name: str
    unit_type: str
    container_type: str
    text: str
    order_index: int
    fragments: List[DocxTextFragment] = field(default_factory=list)
    start: int = 0
    end: int = 0
    estimated_page_no: Optional[int] = None
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    flags: List[str] = field(default_factory=list)
    rewrite_policy: str = "exact"

    def to_dict(self, *, include_fragments: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "unit_id": self.unit_id,
            "part_name": self.part_name,
            "unit_type": self.unit_type,
            "container_type": self.container_type,
            "text": self.text,
            "normalized_text": "".join(str(self.text or "").split()),
            "text_chars": len("".join(str(self.text or "").split())),
            "start": int(self.start),
            "end": int(self.end),
            "order_index": int(self.order_index or 0),
            "estimated_page_no": self.estimated_page_no,
            "table_index": self.table_index,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "flags": list(self.flags),
            "rewrite_policy": self.rewrite_policy,
            "fragment_count": len(self.fragments),
        }
        if include_fragments:
            payload["fragments"] = [fragment.to_dict() for fragment in self.fragments]
        return payload


def normalize_docx_unit_virtual_inline_spaces(units: Sequence[DocxVisibleTextUnit]) -> int:
    """Remove Chinese inline spaces before global offsets are assigned.

    Tabs/newlines remain because they carry table/paragraph structure. For real
    XML text nodes, each kept character retains a source-index map so export can
    still rewrite the original node text even when spaces were removed from the
    analysis view.
    """

    removed_count = 0
    for unit in units:
        removed_count += _normalize_unit_virtual_inline_spaces(unit)
    return removed_count


def _normalize_unit_virtual_inline_spaces(unit: DocxVisibleTextUnit) -> int:
    fragments = list(unit.fragments or [])
    if not fragments:
        return 0
    kept: List[DocxTextFragment] = []
    removed_count = 0
    content_started = False
    for index, fragment in enumerate(fragments):
        normalized_fragment, removed_from_fragment, content_started = _normalize_fragment_chinese_inline_spaces(
            fragment,
            content_started=content_started,
        )
        removed_count += removed_from_fragment
        if normalized_fragment.text:
            kept.append(normalized_fragment)
    if not removed_count:
        return 0
    unit.fragments = []
    for fragment in kept:
        _extend_unit_fragments(unit.fragments, [fragment])
    unit.text = "".join(fragment.text for fragment in unit.fragments)
    flags = list(unit.flags or [])
    if "parser_virtual_inline_spaces_removed" not in flags:
        flags.append("parser_virtual_inline_spaces_removed")
    unit.flags = flags
    return removed_count


def _normalize_fragment_chinese_inline_spaces(
    fragment: DocxTextFragment,
    *,
    content_started: bool,
) -> tuple[DocxTextFragment, int, bool]:
    text = fragment.text or ""
    if not text or text in {"\t", "\n", "\r"}:
        return fragment, 0, content_started
    source_indexes = _fragment_source_indexes(fragment)
    kept_chars: List[str] = []
    kept_source_indexes: List[int] = []
    removed_count = 0
    for char_index, char in enumerate(text):
        if char in {" ", "\u3000"} and content_started:
            removed_count += 1
            continue
        kept_chars.append(char)
        kept_source_indexes.append(source_indexes[char_index] if char_index < len(source_indexes) else char_index)
        if char not in {" ", "\u3000"}:
            content_started = True
    if not removed_count:
        return fragment, 0, content_started
    updated = DocxTextFragment(
        part_name=fragment.part_name,
        text="".join(kept_chars),
        local_start=fragment.local_start,
        local_end=fragment.local_start + len(kept_chars),
        node_index=fragment.node_index,
        node=fragment.node,
        virtual=fragment.virtual,
        source_indexes=kept_source_indexes if fragment.node is not None else None,
    )
    return updated, removed_count, content_started


def _fragment_source_indexes(fragment: DocxTextFragment) -> List[int]:
    if fragment.source_indexes is not None and len(fragment.source_indexes) == len(fragment.text or ""):
        return list(fragment.source_indexes)
    return list(range(len(fragment.text or "")))


@dataclass
class _DocxPartBuildState:
    part_name: str
    node_indexes: Dict[int, int] = field(default_factory=dict)
    text_nodes: List[ET.Element] = field(default_factory=list)
    next_unit_index: int = 0
    current_page: int = 1
    estimated_page_count: int = 0
    order_offset: int = 0

    def index_text_node(self, node: ET.Element) -> int:
        key = id(node)
        existing = self.node_indexes.get(key)
        if existing is not None:
            return existing
        index = len(self.text_nodes)
        self.node_indexes[key] = index
        self.text_nodes.append(node)
        return index

    def next_unit_id(self) -> str:
        unit_id = f"{self.part_name}#{self.next_unit_index}"
        self.next_unit_index += 1
        return unit_id


def list_docx_text_parts(names: Iterable[str]) -> List[str]:
    """Return relevant DOCX XML part names in a stable order."""
    matched: List[str] = []
    for pattern in DOCX_TEXT_PART_PATTERNS:
        for name in sorted(names):
            if name in matched:
                continue
            if fnmatch.fnmatch(name, pattern):
                matched.append(name)
    return matched


def list_docx_visible_text_parts(archive: zipfile.ZipFile) -> List[str]:
    """Return DOCX package parts that can contribute visible text.

    The static list covers normal Word stories. Complex layouts can place
    visible text in additional Word XML/VML parts, so discovery must inspect
    package content instead of relying only on filenames.
    """

    names = archive.namelist()
    configured = list_docx_text_parts(names)
    matched: List[str] = []
    for name in configured:
        if name not in matched:
            matched.append(name)

    for name in sorted(names):
        if name in matched or not _is_docx_candidate_text_part_name(name):
            continue
        if _docx_part_has_visible_text(archive, name):
            matched.append(name)
    return matched


def _is_docx_candidate_text_part_name(part_name: str) -> bool:
    name = str(part_name or "")
    lower = name.lower()
    if not (lower.endswith(".xml") or lower.endswith(".vml")):
        return False
    if not name.startswith("word/"):
        return False
    if name.startswith("word/_rels/") or "/_rels/" in name:
        return False
    if name.startswith("word/media/") or name.startswith("word/embeddings/"):
        return False
    if name.startswith("word/theme/"):
        return False
    if name in {
        "word/settings.xml",
        "word/styles.xml",
        "word/stylesWithEffects.xml",
        "word/fontTable.xml",
        "word/webSettings.xml",
        "word/numbering.xml",
        "word/documentProtection.xml",
    }:
        return False
    return True


def _docx_part_has_visible_text(archive: zipfile.ZipFile, part_name: str) -> bool:
    try:
        root = ET.fromstring(archive.read(part_name))
    except Exception:
        return False
    parent_map = {child: parent for parent in root.iter() for child in parent}
    return any(
        _is_docx_visible_text_node(node, parent_map=parent_map)
        for node in _iter_docx_effective_descendants(root)
    )


def docx_contains_tracked_changes(file_path: str | Path) -> bool:
    """Quick check for tracked-change nodes inside a DOCX package."""
    with zipfile.ZipFile(file_path) as archive:
        for part_name in list_docx_visible_text_parts(archive):
            xml_bytes = archive.read(part_name)
            if b"<w:ins" in xml_bytes or b"<w:del" in xml_bytes or b"<w:delText" in xml_bytes:
                return True
    return False


def extract_docx_text(file_path: str | Path) -> Tuple[str, Dict[str, int | bool]]:
    """Extract visible text from a DOCX package, including tracked insertions/deletions."""
    text, metadata, _units = extract_docx_visible_text_units(file_path)
    return text, metadata


def extract_docx_visible_text_units(
    file_path: str | Path,
    *,
    estimated_page_count: int = 0,
    include_fragments: bool = False,
) -> Tuple[str, Dict[str, Any], List[DocxVisibleTextUnit]]:
    """Extract DOCX visible text and keep global offsets for XML-level rewrite."""
    tracked_insertions = 0
    tracked_deletions = 0
    tracked_deleted_text_nodes = 0
    units: List[DocxVisibleTextUnit] = []

    with zipfile.ZipFile(file_path) as archive:
        part_names = list_docx_visible_text_parts(archive)
        for part_name in part_names:
            root = ET.fromstring(archive.read(part_name))
            tracked_insertions += len(root.findall(".//w:ins", NAMESPACES))
            tracked_deletions += len(root.findall(".//w:del", NAMESPACES))
            tracked_deleted_text_nodes += len(root.findall(".//w:delText", NAMESPACES))

            units.extend(
                collect_docx_visible_text_units_from_root(
                    root,
                    part_name=part_name,
                    start_order=len(units),
                    estimated_page_count=estimated_page_count,
                )
            )

    coverage_audit = audit_docx_text_coverage(
        file_path,
        estimated_page_count=estimated_page_count,
        sample_limit=0,
    )
    tracked_changes_detected = bool(
        tracked_insertions or tracked_deletions or tracked_deleted_text_nodes
    )
    removed_virtual_space_count = normalize_docx_unit_virtual_inline_spaces(units)
    text = assign_docx_unit_global_offsets(units)
    metadata: Dict[str, Any] = {
        "tracked_changes_detected": tracked_changes_detected,
        "tracked_insertions": tracked_insertions,
        "tracked_deletions": tracked_deletions,
        "tracked_deleted_text_nodes": tracked_deleted_text_nodes,
        "docx_text_extraction": "ooxml_visible_text_units",
        "docx_text_unit_count": len(units),
        "docx_coverage_backfill_unit_count": sum(
            1 for unit in units if "parser_coverage_backfill" in unit.flags
        ),
        "docx_coverage_backfill_text_node_count": sum(
            len([fragment for fragment in unit.fragments if not fragment.virtual])
            for unit in units
            if "parser_coverage_backfill" in unit.flags
        ),
        "docx_parser_virtual_inline_space_removed_count": removed_virtual_space_count,
        "docx_table_unit_count": sum(
            1 for unit in units if unit.container_type in {"table_cell", "table_row"}
        ),
        "docx_header_footer_unit_count": sum(
            1 for unit in units if unit.container_type in {"header", "footer"}
        ),
        "docx_comment_unit_count": sum(
            1 for unit in units if unit.container_type == "comment"
        ),
        "docx_drawing_unit_count": sum(
            1 for unit in units if "drawing_text" in unit.flags
        ),
        "docx_math_unit_count": sum(
            1 for unit in units if "math_text" in unit.flags
        ),
        "docx_chart_unit_count": sum(
            1 for unit in units if unit.container_type == "chart"
        ),
        "docx_diagram_unit_count": sum(
            1 for unit in units if unit.container_type == "diagram"
        ),
        "docx_text_part_count": len(part_names),
        "docx_text_parts": part_names,
        "docx_text_unit_coverage_by_part": _summarize_unit_coverage_by_part(units),
        "docx_text_coverage_audit": coverage_audit,
        "docx_total_text_node_count": coverage_audit.get("total_text_node_count", 0),
        "docx_covered_text_node_count": coverage_audit.get("covered_text_node_count", 0),
        "docx_uncovered_text_node_count": coverage_audit.get("uncovered_text_node_count", 0),
        "docx_unhandled_text_part_count": coverage_audit.get("unhandled_text_part_count", 0),
        "docx_unknown_text_part_count": coverage_audit.get("unknown_text_part_count", 0),
        "docx_hidden_text_node_count": coverage_audit.get("hidden_text_node_count", 0),
        "docx_field_instruction_node_count": coverage_audit.get("field_instruction_node_count", 0),
        "docx_review_required_text_node_count": coverage_audit.get("review_required_text_node_count", 0),
    }
    if include_fragments:
        metadata["docx_text_fragment_count"] = sum(len(unit.fragments) for unit in units)
    return text, metadata, units


def audit_docx_text_coverage(
    file_path: str | Path,
    *,
    estimated_page_count: int = 0,
    sample_limit: int = 25,
) -> Dict[str, Any]:
    """Audit which DOCX XML text nodes are covered by the visible-text source map."""
    report: Dict[str, Any] = {
        "audit_version": "docx_ooxml_text_coverage_v1",
        "xml_part_count": 0,
        "configured_text_part_count": 0,
        "text_xml_part_count": 0,
        "total_text_node_count": 0,
        "covered_text_node_count": 0,
        "uncovered_text_node_count": 0,
        "review_required_text_node_count": 0,
        "hidden_text_node_count": 0,
        "field_instruction_node_count": 0,
        "metadata_text_node_count": 0,
        "custom_xml_text_node_count": 0,
        "unknown_text_node_count": 0,
        "unhandled_text_part_count": 0,
        "unknown_text_part_count": 0,
        "uncovered_text_by_category": {},
        "coverage_by_part": {},
        "uncovered_samples": [],
        "unknown_text_parts": [],
        "unhandled_text_parts": [],
    }
    source_path = Path(file_path)
    try:
        with zipfile.ZipFile(source_path) as archive:
            names = archive.namelist()
            xml_part_names = sorted(name for name in names if name.lower().endswith(".xml"))
            configured_text_parts = set(list_docx_visible_text_parts(archive))
            report["xml_part_count"] = len(xml_part_names)
            report["configured_text_part_count"] = len(configured_text_parts)
            for part_name in xml_part_names:
                try:
                    root = ET.fromstring(archive.read(part_name))
                except ET.ParseError:
                    continue
                part_report = _audit_docx_xml_part_text_coverage(
                    root,
                    part_name=part_name,
                    configured_text_parts=configured_text_parts,
                    estimated_page_count=estimated_page_count,
                    sample_limit=max(0, int(sample_limit or 0)),
                )
                if part_report["total_text_node_count"] <= 0:
                    continue
                report["text_xml_part_count"] += 1
                _merge_docx_coverage_part_report(report, part_name, part_report)
    except Exception as exc:
        report["audit_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _audit_docx_xml_part_text_coverage(
    root: ET.Element,
    *,
    part_name: str,
    configured_text_parts: set[str],
    estimated_page_count: int,
    sample_limit: int,
) -> Dict[str, Any]:
    units: List[DocxVisibleTextUnit] = []
    if part_name in configured_text_parts:
        units = collect_docx_visible_text_units_from_root(
            root,
            part_name=part_name,
            estimated_page_count=estimated_page_count,
        )
        assign_docx_unit_global_offsets(units)

    covered_nodes = {
        id(fragment.node)
        for unit in units
        for fragment in unit.fragments
        if fragment.node is not None and not fragment.virtual
    }
    review_required_nodes = {
        id(fragment.node)
        for unit in units
        if unit.rewrite_policy == "review_required"
        for fragment in unit.fragments
        if fragment.node is not None and not fragment.virtual
    }
    parent_map = {child: parent for parent in root.iter() for child in parent}
    part_report: Dict[str, Any] = {
        "total_text_node_count": 0,
        "covered_text_node_count": 0,
        "uncovered_text_node_count": 0,
        "review_required_text_node_count": 0,
        "hidden_text_node_count": 0,
        "field_instruction_node_count": 0,
        "metadata_text_node_count": 0,
        "custom_xml_text_node_count": 0,
        "unknown_text_node_count": 0,
        "uncovered_text_by_category": {},
        "uncovered_samples": [],
        "unit_count": len(units),
        "configured_text_part": part_name in configured_text_parts,
    }
    for node_index, node in enumerate(_iter_docx_effective_descendants(root)):
        node_text = str(node.text or "")
        if not node_text.strip():
            continue
        category = _classify_docx_text_node(
            part_name=part_name,
            node=node,
            parent_map=parent_map,
        )
        if category in {"non_visible_xml_text", "package_metadata"}:
            continue
        covered = id(node) in covered_nodes
        part_report["total_text_node_count"] += 1
        if id(node) in review_required_nodes:
            part_report["review_required_text_node_count"] += 1
        if category == "hidden_text":
            part_report["hidden_text_node_count"] += 1
        elif category == "field_instruction":
            part_report["field_instruction_node_count"] += 1
        elif category == "metadata":
            part_report["metadata_text_node_count"] += 1
        elif category == "custom_xml":
            part_report["custom_xml_text_node_count"] += 1
        elif category == "unknown_xml_text":
            part_report["unknown_text_node_count"] += 1
        if covered:
            part_report["covered_text_node_count"] += 1
            continue
        part_report["uncovered_text_node_count"] += 1
        _increment_counter(part_report["uncovered_text_by_category"], category)
        if len(part_report["uncovered_samples"]) < sample_limit:
            part_report["uncovered_samples"].append(
                {
                    "part_name": part_name,
                    "node_index": node_index,
                    "tag": _local_name(node.tag),
                    "category": category,
                    "text_excerpt": _compact_excerpt(node_text, 80),
                }
            )
    return part_report


def _merge_docx_coverage_part_report(
    report: Dict[str, Any],
    part_name: str,
    part_report: Dict[str, Any],
) -> None:
    for key in (
        "total_text_node_count",
        "covered_text_node_count",
        "uncovered_text_node_count",
        "review_required_text_node_count",
        "hidden_text_node_count",
        "field_instruction_node_count",
        "metadata_text_node_count",
        "custom_xml_text_node_count",
        "unknown_text_node_count",
    ):
        report[key] = int(report.get(key, 0) or 0) + int(part_report.get(key, 0) or 0)

    uncovered_by_category = report.setdefault("uncovered_text_by_category", {})
    for category, count in dict(part_report.get("uncovered_text_by_category") or {}).items():
        _increment_counter(uncovered_by_category, category, int(count or 0))

    part_payload = {
        "total_text_node_count": part_report.get("total_text_node_count", 0),
        "covered_text_node_count": part_report.get("covered_text_node_count", 0),
        "uncovered_text_node_count": part_report.get("uncovered_text_node_count", 0),
        "review_required_text_node_count": part_report.get("review_required_text_node_count", 0),
        "unit_count": part_report.get("unit_count", 0),
        "configured_text_part": part_report.get("configured_text_part", False),
        "uncovered_text_by_category": dict(part_report.get("uncovered_text_by_category") or {}),
    }
    report["coverage_by_part"][part_name] = part_payload

    uncovered_count = int(part_report.get("uncovered_text_node_count", 0) or 0)
    unknown_count = int(part_report.get("unknown_text_node_count", 0) or 0)
    categories = set((part_report.get("uncovered_text_by_category") or {}).keys())
    if uncovered_count > 0 and categories - {"metadata", "package_metadata", "custom_xml", "field_instruction"}:
        report["unhandled_text_part_count"] += 1
        report["unhandled_text_parts"].append(part_name)
    if unknown_count > 0:
        report["unknown_text_part_count"] += 1
        report["unknown_text_parts"].append(part_name)

    samples = report.setdefault("uncovered_samples", [])
    for sample in part_report.get("uncovered_samples") or []:
        if len(samples) >= 25:
            break
        samples.append(sample)


def collect_docx_visible_text_units_from_root(
    root: ET.Element,
    *,
    part_name: str,
    start_order: int = 0,
    estimated_page_count: int = 0,
) -> List[DocxVisibleTextUnit]:
    """Collect visible-ish text units from a parsed WordprocessingML part."""
    state = _DocxPartBuildState(
        part_name=part_name,
        estimated_page_count=max(0, int(estimated_page_count or 0)),
        order_offset=start_order,
    )
    parent_map = {child: parent for parent in root.iter() for child in parent}
    table_index = {table: index for index, table in enumerate(root.iter(TABLE_TAG), start=1)}
    units: List[DocxVisibleTextUnit] = []

    body = root.find(".//w:body", NAMESPACES)
    if body is not None:
        for child in _iter_docx_effective_children(body):
            units.extend(
                _collect_docx_block_units(
                    child,
                    state=state,
                    parent_map=parent_map,
                    table_index=table_index,
                    container_override=None,
                )
            )
        _append_docx_coverage_backfill_units(
            root,
            units=units,
            state=state,
            parent_map=parent_map,
            table_index=table_index,
        )
        return units

    for child in _iter_docx_effective_children(root):
        units.extend(
            _collect_docx_block_units(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
                container_override=_part_container_type(part_name),
            )
        )

    if not units:
        part_container = _part_container_type(part_name) or "xml_text"
        units.extend(
            _collect_generic_text_units(
                root,
                state=state,
                container_type=part_container,
            )
        )
    else:
        _append_docx_coverage_backfill_units(
            root,
            units=units,
            state=state,
            parent_map=parent_map,
            table_index=table_index,
        )
    return units


def _append_docx_coverage_backfill_units(
    root: ET.Element,
    *,
    units: List[DocxVisibleTextUnit],
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> None:
    """Append visible text nodes missed by structural traversal.

    Word's complex layouts can hide visible text in wrappers that are not normal
    body paragraphs/tables, especially compatibility branches, shapes, and
    drawing containers. The normal traversal remains the primary source of
    coherent blocks; this pass only backfills still-uncovered visible text nodes
    so recognition sees every visible subject at least once.
    """

    covered_nodes = {
        id(fragment.node)
        for unit in units
        for fragment in unit.fragments
        if fragment.node is not None and not fragment.virtual
    }
    for group in _iter_uncovered_visible_text_node_groups(
        root,
        parent_map=parent_map,
        covered_nodes=covered_nodes,
    ):
        fragments: List[DocxTextFragment] = []
        for node in group:
            _append_text_node_fragment(fragments, node, state=state)
        text = "".join(fragment.text for fragment in fragments)
        if not text.strip():
            continue
        anchor = group[0]
        container_type = _container_type_for_node(
            state.part_name,
            anchor,
            parent_map,
        )
        table_meta = _table_position_for_node(
            anchor,
            parent_map=parent_map,
            table_index=table_index,
        )
        unit_id = state.next_unit_id()
        flags = _container_flags(container_type)
        flags.extend(_fragment_source_flags(fragments))
        flags.append("parser_coverage_backfill")
        units.append(
            DocxVisibleTextUnit(
                unit_id=unit_id,
                part_name=state.part_name,
                unit_type="coverage_backfill",
                container_type=container_type,
                text=text,
                order_index=state.order_offset + state.next_unit_index - 1,
                fragments=fragments,
                estimated_page_no=_page_for_current_unit(state),
                table_index=table_meta.get("table_index"),
                row_index=table_meta.get("row_index"),
                col_index=table_meta.get("col_index"),
                flags=_unique_strings(flags),
                rewrite_policy=_rewrite_policy_for_container(container_type),
            )
        )
        covered_nodes.update(id(fragment.node) for fragment in fragments if fragment.node is not None)


def _iter_uncovered_visible_text_node_groups(
    root: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
    covered_nodes: set[int],
) -> Iterator[List[ET.Element]]:
    current_group: List[ET.Element] = []
    current_parent: Optional[ET.Element] = None
    for node in _iter_docx_effective_descendants(root):
        if not _is_docx_visible_text_node(node, parent_map=parent_map):
            continue
        if id(node) in covered_nodes:
            if current_group:
                yield current_group
                current_group = []
                current_parent = None
            continue
        paragraph_parent = _nearest_ancestor(node, parent_map=parent_map, tag=PARAGRAPH_TAG)
        group_parent = paragraph_parent if paragraph_parent is not None else parent_map.get(node)
        if current_group and group_parent is not current_parent:
            yield current_group
            current_group = []
        current_group.append(node)
        current_parent = group_parent
    if current_group:
        yield current_group


def _collect_generic_text_units(
    root: ET.Element,
    *,
    state: _DocxPartBuildState,
    container_type: str,
) -> List[DocxVisibleTextUnit]:
    units: List[DocxVisibleTextUnit] = []
    for node in _iter_docx_effective_descendants(root):
        if node.tag not in TEXT_TAGS or not node.text:
            continue
        fragments: List[DocxTextFragment] = []
        _append_text_node_fragment(fragments, node, state=state)
        text = "".join(fragment.text for fragment in fragments)
        if not text.strip():
            continue
        unit_id = state.next_unit_id()
        flags = _container_flags(container_type)
        flags.extend(_fragment_source_flags(fragments))
        units.append(
            DocxVisibleTextUnit(
                unit_id=unit_id,
                part_name=state.part_name,
                unit_type=container_type,
                container_type=container_type,
                text=text,
                order_index=state.order_offset + state.next_unit_index - 1,
                fragments=fragments,
                estimated_page_no=None,
                flags=_unique_strings(flags),
                rewrite_policy=_rewrite_policy_for_container(container_type),
            )
        )
    return units


def _iter_docx_effective_children(node: ET.Element) -> Iterator[ET.Element]:
    if node.tag == ALTERNATE_CONTENT_TAG:
        branch = _select_alternate_content_branch(node)
        if branch is not None:
            yield from _iter_docx_effective_children(branch)
        return
    for child in node:
        if child.tag == ALTERNATE_CONTENT_TAG:
            yield from _iter_docx_effective_children(child)
        else:
            yield child


def _iter_docx_effective_descendants(node: ET.Element) -> Iterator[ET.Element]:
    yield node
    for child in _iter_docx_effective_children(node):
        yield from _iter_docx_effective_descendants(child)


def _select_alternate_content_branch(node: ET.Element) -> Optional[ET.Element]:
    for child in node:
        if child.tag == ALTERNATE_CHOICE_TAG and _element_has_visible_text(child):
            return child
    for child in node:
        if child.tag == ALTERNATE_FALLBACK_TAG and _element_has_visible_text(child):
            return child
    for child in node:
        if _element_has_visible_text(child):
            return child
    return node[0] if len(node) else None


def _element_has_visible_text(node: ET.Element) -> bool:
    return any(
        item.tag in TEXT_TAGS and str(item.text or "").strip()
        for item in node.iter()
    )


def assign_docx_unit_global_offsets(units: Sequence[DocxVisibleTextUnit]) -> str:
    """Assign parser-text offsets to units/fragments and return the joined text."""
    text_parts: List[str] = []
    cursor = 0
    for index, unit in enumerate(units):
        if index:
            text_parts.append("\n")
            cursor += 1
        unit.start = cursor
        unit.end = unit.start + len(unit.text or "")
        for fragment in unit.fragments:
            fragment.start = unit.start + fragment.local_start
            fragment.end = unit.start + fragment.local_end
        text_parts.append(unit.text or "")
        cursor = unit.end
    return "".join(text_parts)


def replace_docx_text_by_ranges(
    file_path: str | Path,
    range_entries: Sequence[Dict[str, Any]],
    *,
    source_text: str | None = None,
) -> bool:
    """Rewrite exact global source ranges in a DOCX package without rebuilding layout."""
    report = replace_docx_text_by_ranges_with_report(
        file_path,
        range_entries,
        source_text=source_text,
    )
    return bool(report.get("applied"))


def normalize_docx_chinese_inline_spaces(
    file_path: str | Path,
) -> Dict[str, Any]:
    """Rewrite a DOCX package so Chinese inline spaces removed from analysis are removed in output too."""

    report: Dict[str, Any] = {
        "applied": False,
        "removed_space_count": 0,
        "modified_part_count": 0,
        "modified_parts": [],
    }
    source_path = Path(file_path)
    with zipfile.ZipFile(source_path) as archive:
        part_roots: Dict[str, ET.Element] = {}
        modified_parts: set[str] = set()
        removed_total = 0
        for part_name in list_docx_visible_text_parts(archive):
            root = ET.fromstring(archive.read(part_name))
            removed = _normalize_docx_root_chinese_inline_spaces(root, part_name=part_name)
            if removed > 0:
                part_roots[part_name] = root
                modified_parts.add(part_name)
                removed_total += removed

        if not modified_parts:
            return report

        temp_fd, temp_name = tempfile.mkstemp(suffix=source_path.suffix, dir=str(source_path.parent))
        os.close(temp_fd)
        try:
            with zipfile.ZipFile(source_path, "r") as source_archive, zipfile.ZipFile(
                temp_name,
                "w",
            ) as target_archive:
                for item in source_archive.infolist():
                    data = source_archive.read(item.filename)
                    if item.filename in modified_parts:
                        data = ET.tostring(
                            part_roots[item.filename],
                            encoding="utf-8",
                            xml_declaration=True,
                        )
                    target_archive.writestr(item, data)
            os.replace(temp_name, source_path)
        except Exception:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise

    report["applied"] = True
    report["removed_space_count"] = removed_total
    report["modified_part_count"] = len(modified_parts)
    report["modified_parts"] = sorted(modified_parts)
    return report


def _normalize_docx_root_chinese_inline_spaces(root: ET.Element, *, part_name: str) -> int:
    units = collect_docx_visible_text_units_from_root(root, part_name=part_name)
    original_nodes: Dict[int, ET.Element] = {}
    original_texts: Dict[int, str] = {}
    for unit in units:
        for fragment in unit.fragments:
            if fragment.node is None or fragment.virtual:
                continue
            node_id = id(fragment.node)
            original_nodes[node_id] = fragment.node
            original_texts[node_id] = fragment.node.text or ""

    removed_count = normalize_docx_unit_virtual_inline_spaces(units)
    if removed_count <= 0:
        return 0

    kept_indexes: Dict[int, set[int]] = {node_id: set() for node_id in original_nodes}
    for unit in units:
        for fragment in unit.fragments:
            if fragment.node is None or fragment.virtual:
                continue
            node_id = id(fragment.node)
            kept_indexes.setdefault(node_id, set()).update(_fragment_source_indexes(fragment))

    actual_removed = 0
    for node_id, node in original_nodes.items():
        original = original_texts.get(node_id, "")
        keep = kept_indexes.get(node_id, set())
        if len(keep) >= len(original):
            continue
        updated = "".join(char for index, char in enumerate(original) if index in keep)
        if updated == original:
            continue
        actual_removed += max(0, len(original) - len(updated))
        node.text = updated
        _sync_space_preserve(node, updated)
    return actual_removed


def replace_docx_text_by_ranges_with_report(
    file_path: str | Path,
    range_entries: Sequence[Dict[str, Any]],
    *,
    source_text: str | None = None,
) -> Dict[str, Any]:
    """Rewrite exact source ranges and return coverage diagnostics."""
    report: Dict[str, Any] = {
        "applied": False,
        "attempted_entry_count": len(range_entries or []),
        "normalized_entry_count": 0,
        "applied_entry_count": 0,
        "unapplied_entry_count": 0,
        "source_text_matched": None,
        "modified_part_count": 0,
        "modified_parts": [],
        "rejection_reason": "",
        "unapplied_ranges": [],
    }
    entries = _normalize_range_entries(range_entries, source_text=source_text)
    report["normalized_entry_count"] = len(entries)
    if not entries:
        report["rejection_reason"] = "no_valid_range_entries"
        return report

    source_path = Path(file_path)
    with zipfile.ZipFile(source_path) as archive:
        text_parts = list_docx_visible_text_parts(archive)
        target_parts = set(text_parts)
        if not target_parts:
            report["rejection_reason"] = "no_text_parts"
            return report
        part_roots: Dict[str, ET.Element] = {}
        units: List[DocxVisibleTextUnit] = []
        for part_name in text_parts:
            root = ET.fromstring(archive.read(part_name))
            part_roots[part_name] = root
            units.extend(
                collect_docx_visible_text_units_from_root(
                    root,
                    part_name=part_name,
                    start_order=len(units),
                )
            )

    normalize_docx_unit_virtual_inline_spaces(units)
    extracted_text = assign_docx_unit_global_offsets(units)
    if source_text is not None and extracted_text != source_text:
        report["source_text_matched"] = False
        report["rejection_reason"] = "source_text_mismatch"
        report["unapplied_entry_count"] = len(entries)
        report["unapplied_ranges"] = _range_entry_diagnostics(entries)
        return report
    report["source_text_matched"] = True

    replacements = _build_docx_fragment_range_replacements(
        units=units,
        entries=entries,
        extracted_text=extracted_text,
    )
    if not replacements:
        report["rejection_reason"] = "no_rewritable_xml_fragments"
        report["unapplied_entry_count"] = len(entries)
        report["unapplied_ranges"] = _range_entry_diagnostics(entries)
        return report

    modified_parts: set[str] = set()
    applied_ranges: set[tuple[int, int, int]] = set()
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        if _apply_docx_fragment_range_replacement(replacement):
            modified_parts.add(str(replacement["part_name"]))
            applied_ranges.add(
                (
                    int(replacement["start"]),
                    int(replacement["end"]),
                    int(replacement.get("input_order", -1)),
                )
            )

    if not modified_parts:
        report["rejection_reason"] = "no_modified_parts"
        report["unapplied_entry_count"] = len(entries)
        report["unapplied_ranges"] = _range_entry_diagnostics(entries)
        return report

    temp_fd, temp_name = tempfile.mkstemp(suffix=source_path.suffix, dir=str(source_path.parent))
    os.close(temp_fd)
    try:
        with zipfile.ZipFile(source_path, "r") as source_archive, zipfile.ZipFile(
            temp_name,
            "w",
        ) as target_archive:
            for item in source_archive.infolist():
                data = source_archive.read(item.filename)
                if item.filename in modified_parts:
                    data = ET.tostring(
                        part_roots[item.filename],
                        encoding="utf-8",
                        xml_declaration=True,
                    )
                target_archive.writestr(item, data)
        os.replace(temp_name, source_path)
        report["applied"] = True
        report["modified_parts"] = sorted(modified_parts)
        report["modified_part_count"] = len(modified_parts)
        report["applied_entry_count"] = len(applied_ranges)
        report["unapplied_entry_count"] = max(0, len(entries) - len(applied_ranges))
        unapplied = [
            entry
            for entry in entries
            if (
                int(entry["start"]),
                int(entry["end"]),
                int(entry.get("input_order", -1)),
            )
            not in applied_ranges
        ]
        report["unapplied_ranges"] = _range_entry_diagnostics(unapplied)
        return report
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def _collect_docx_block_units(
    node: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
    container_override: Optional[str],
) -> List[DocxVisibleTextUnit]:
    if node.tag == PARAGRAPH_TAG:
        return _collect_docx_paragraph_units(
            node,
            state=state,
            parent_map=parent_map,
            table_index=table_index,
            container_override=container_override,
        )
    if node.tag == TABLE_TAG:
        return _collect_docx_table_units(
            node,
            state=state,
            parent_map=parent_map,
            table_index=table_index,
            container_override=container_override,
        )
    if node.tag == TEXTBOX_CONTENT_TAG:
        return _collect_docx_textbox_units(
            node,
            state=state,
            parent_map=parent_map,
            table_index=table_index,
        )

    units: List[DocxVisibleTextUnit] = []
    for child in _iter_docx_effective_children(node):
        units.extend(
            _collect_docx_block_units(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
                container_override=container_override,
            )
        )
    return units


def _collect_docx_paragraph_units(
    paragraph: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
    container_override: Optional[str],
) -> List[DocxVisibleTextUnit]:
    units: List[DocxVisibleTextUnit] = []
    fragments = _paragraph_visible_fragments(paragraph, state=state)
    text = "".join(fragment.text for fragment in fragments)
    if text.strip():
        container_type = container_override or _container_type_for_node(
            state.part_name,
            paragraph,
            parent_map,
        )
        table_meta = _table_position(paragraph, parent_map=parent_map, table_index=table_index)
        unit_id = state.next_unit_id()
        flags = _container_flags(container_type)
        flags.extend(_fragment_source_flags(fragments))
        units.append(
            DocxVisibleTextUnit(
                unit_id=unit_id,
                part_name=state.part_name,
                unit_type="table_cell" if container_type == "table_cell" else container_type,
                container_type=container_type,
                text=text,
                order_index=state.order_offset + state.next_unit_index - 1,
                fragments=fragments,
                estimated_page_no=_page_for_current_unit(state),
                table_index=table_meta.get("table_index"),
                row_index=table_meta.get("row_index"),
                col_index=table_meta.get("col_index"),
                flags=_unique_strings(flags),
                rewrite_policy=_rewrite_policy_for_container(container_type),
            )
        )
    for textbox in paragraph.iter(TEXTBOX_CONTENT_TAG):
        units.extend(
            _collect_docx_textbox_units(
                textbox,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
        )

    _advance_page_for_paragraph(paragraph, state=state)
    return units


def _collect_docx_textbox_units(
    textbox: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> List[DocxVisibleTextUnit]:
    units: List[DocxVisibleTextUnit] = []
    for child in _iter_docx_effective_children(textbox):
        units.extend(
            _collect_docx_block_units(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
                container_override="textbox",
            )
        )
    return units


def _collect_docx_table_units(
    table: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
    container_override: Optional[str],
) -> List[DocxVisibleTextUnit]:
    units: List[DocxVisibleTextUnit] = []
    current_table_index = int(table_index.get(table) or 0)
    rows = [row for row in table if row.tag == TABLE_ROW_TAG]
    for row_index, row in enumerate(rows, start=1):
        row_fragments: List[DocxTextFragment] = []
        cells = [cell for cell in row if cell.tag == TABLE_CELL_TAG]
        non_empty_cell_count = 0
        unit_page_no = _page_for_current_unit(state)
        for cell in cells:
            cell_fragments = _cell_visible_fragments(
                cell,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
            cell_text = "".join(fragment.text for fragment in cell_fragments).strip()
            if not cell_text:
                _advance_page_for_container(cell, state=state)
                continue
            if non_empty_cell_count:
                _append_virtual_fragment(row_fragments, "\t", part_name=state.part_name)
            _extend_unit_fragments(row_fragments, cell_fragments)
            non_empty_cell_count += 1
            _advance_page_for_container(cell, state=state)

        row_text = "".join(fragment.text for fragment in row_fragments)
        if not row_text.strip():
            continue
        container_type = container_override or "table_cell"
        unit_id = state.next_unit_id()
        flags = _container_flags(container_type)
        flags.extend(_fragment_source_flags(row_fragments))
        units.append(
            DocxVisibleTextUnit(
                unit_id=unit_id,
                part_name=state.part_name,
                unit_type="table_row",
                container_type=container_type,
                text=row_text,
                order_index=state.order_offset + state.next_unit_index - 1,
                fragments=row_fragments,
                estimated_page_no=unit_page_no,
                table_index=current_table_index,
                row_index=row_index,
                col_index=None,
                flags=_unique_strings(flags),
                rewrite_policy=_rewrite_policy_for_container(container_type),
            )
        )
    return units


def _cell_visible_fragments(
    cell: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> List[DocxTextFragment]:
    fragments: List[DocxTextFragment] = []
    added = 0
    for child in _iter_docx_effective_children(cell):
        child_fragments: List[DocxTextFragment] = []
        if child.tag == PARAGRAPH_TAG:
            child_fragments = _paragraph_visible_fragments(child, state=state)
        elif child.tag == TABLE_TAG:
            child_fragments = _nested_table_fragments(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
        elif child.tag == TEXTBOX_CONTENT_TAG:
            child_fragments = _textbox_visible_fragments(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
        else:
            for nested in _iter_docx_effective_children(child):
                child_fragments.extend(
                    _cell_visible_fragments(
                        nested,
                        state=state,
                        parent_map=parent_map,
                        table_index=table_index,
                    )
                )
        if not "".join(fragment.text for fragment in child_fragments).strip():
            continue
        if added:
            _append_virtual_fragment(fragments, " ", part_name=state.part_name)
        _extend_unit_fragments(fragments, child_fragments)
        added += 1
    return fragments


def _nested_table_fragments(
    table: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> List[DocxTextFragment]:
    fragments: List[DocxTextFragment] = []
    added_rows = 0
    for row in table.iterfind("./w:tr", NAMESPACES):
        row_fragments: List[DocxTextFragment] = []
        added_cells = 0
        for cell in row.iterfind("./w:tc", NAMESPACES):
            cell_fragments = _cell_visible_fragments(
                cell,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
            if not "".join(fragment.text for fragment in cell_fragments).strip():
                continue
            if added_cells:
                _append_virtual_fragment(row_fragments, "\t", part_name=state.part_name)
            _extend_unit_fragments(row_fragments, cell_fragments)
            added_cells += 1
        if not "".join(fragment.text for fragment in row_fragments).strip():
            continue
        if added_rows:
            _append_virtual_fragment(fragments, "\n", part_name=state.part_name)
        _extend_unit_fragments(fragments, row_fragments)
        added_rows += 1
    return fragments


def _textbox_visible_fragments(
    textbox: ET.Element,
    *,
    state: _DocxPartBuildState,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> List[DocxTextFragment]:
    fragments: List[DocxTextFragment] = []
    added = 0
    for child in _iter_docx_effective_children(textbox):
        child_fragments: List[DocxTextFragment] = []
        if child.tag == PARAGRAPH_TAG:
            child_fragments = _paragraph_visible_fragments(child, state=state)
        elif child.tag == TABLE_TAG:
            child_fragments = _nested_table_fragments(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
        elif child.tag == TEXTBOX_CONTENT_TAG:
            child_fragments = _textbox_visible_fragments(
                child,
                state=state,
                parent_map=parent_map,
                table_index=table_index,
            )
        if not "".join(fragment.text for fragment in child_fragments).strip():
            continue
        if added:
            _append_virtual_fragment(fragments, "\n", part_name=state.part_name)
        _extend_unit_fragments(fragments, child_fragments)
        added += 1
    return fragments


def _paragraph_visible_fragments(
    paragraph: ET.Element,
    *,
    state: _DocxPartBuildState,
) -> List[DocxTextFragment]:
    fragments: List[DocxTextFragment] = []
    _collect_paragraph_visible_fragments(paragraph, state=state, fragments=fragments)
    return fragments


def _collect_paragraph_visible_fragments(
    node: ET.Element,
    *,
    state: _DocxPartBuildState,
    fragments: List[DocxTextFragment],
) -> None:
    for child in _iter_docx_effective_children(node):
        if child.tag == PARAGRAPH_TAG:
            continue
        if child.tag == TEXTBOX_CONTENT_TAG:
            continue
        if child.tag in TEXT_TAGS:
            if child.text:
                _append_text_node_fragment(fragments, child, state=state)
            continue
        if child.tag == TAB_TAG:
            _append_virtual_fragment(fragments, "\t", part_name=state.part_name)
            continue
        if child.tag in BREAK_TAGS:
            _append_virtual_fragment(fragments, "\n", part_name=state.part_name)
            continue
        _collect_paragraph_visible_fragments(child, state=state, fragments=fragments)


def _append_text_node_fragment(
    fragments: List[DocxTextFragment],
    node: ET.Element,
    *,
    state: _DocxPartBuildState,
) -> None:
    text = node.text or ""
    if not text:
        return
    local_start = sum(len(fragment.text) for fragment in fragments)
    fragments.append(
        DocxTextFragment(
            part_name=state.part_name,
            text=text,
            local_start=local_start,
            local_end=local_start + len(text),
            node_index=state.index_text_node(node),
            node=node,
            virtual=False,
        )
    )


def _fragment_source_flags(fragments: Sequence[DocxTextFragment]) -> List[str]:
    flags: List[str] = []
    for fragment in fragments:
        node = fragment.node
        if node is None:
            continue
        if node.tag == f"{{{DRAWINGML_NAMESPACE}}}t":
            flags.append("drawing_text")
        elif node.tag == f"{{{MATH_NAMESPACE}}}t":
            flags.append("math_text")
        elif node.tag == f"{{{CHART_NAMESPACE}}}v":
            flags.append("chart_value_text")
    return flags


def _unique_strings(values: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _append_virtual_fragment(
    fragments: List[DocxTextFragment],
    text: str,
    *,
    part_name: str,
) -> None:
    if not text:
        return
    local_start = sum(len(fragment.text) for fragment in fragments)
    fragments.append(
        DocxTextFragment(
            part_name=part_name,
            text=text,
            local_start=local_start,
            local_end=local_start + len(text),
            virtual=True,
        )
    )


def _extend_unit_fragments(
    target: List[DocxTextFragment],
    source: Sequence[DocxTextFragment],
) -> None:
    for fragment in source:
        local_start = sum(len(item.text) for item in target)
        target.append(
            DocxTextFragment(
                part_name=fragment.part_name,
                text=fragment.text,
                local_start=local_start,
                local_end=local_start + len(fragment.text),
                node_index=fragment.node_index,
                node=fragment.node,
                virtual=fragment.virtual,
                source_indexes=list(fragment.source_indexes) if fragment.source_indexes is not None else None,
            )
        )


def _part_container_type(part_name: str) -> Optional[str]:
    if part_name.startswith("word/header"):
        return "header"
    if part_name.startswith("word/footer"):
        return "footer"
    if part_name.endswith("footnotes.xml"):
        return "footnote"
    if part_name.endswith("endnotes.xml"):
        return "endnote"
    if part_name.startswith("word/comments"):
        return "comment"
    if part_name.startswith("word/charts/"):
        return "chart"
    if part_name.startswith("word/diagrams/"):
        return "diagram"
    if part_name == "word/glossary/document.xml":
        return "glossary"
    return None


def _container_type_for_node(
    part_name: str,
    node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
) -> str:
    part_type = _part_container_type(part_name)
    if part_type:
        return part_type

    current: Optional[ET.Element] = node
    saw_table = False
    while current is not None:
        if current.tag == TEXTBOX_CONTENT_TAG:
            return "textbox"
        if current.tag == TABLE_CELL_TAG:
            saw_table = True
        current = parent_map.get(current)
    return "table_cell" if saw_table else "paragraph"


def _table_position_for_node(
    node: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> Dict[str, int]:
    paragraph = _nearest_ancestor(node, parent_map=parent_map, tag=PARAGRAPH_TAG)
    if paragraph is not None:
        return _table_position(paragraph, parent_map=parent_map, table_index=table_index)
    return {}


def _nearest_ancestor(
    node: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
    tag: str,
) -> Optional[ET.Element]:
    current: Optional[ET.Element] = node
    while current is not None:
        if current.tag == tag:
            return current
        current = parent_map.get(current)
    return None


def _table_position(
    paragraph: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
    table_index: Dict[ET.Element, int],
) -> Dict[str, int]:
    cell: Optional[ET.Element] = None
    row: Optional[ET.Element] = None
    table: Optional[ET.Element] = None
    current: Optional[ET.Element] = paragraph
    while current is not None:
        if current.tag == TABLE_CELL_TAG and cell is None:
            cell = current
        elif current.tag == TABLE_ROW_TAG and row is None:
            row = current
        elif current.tag == TABLE_TAG:
            table = current
            break
        current = parent_map.get(current)
    if cell is None or row is None or table is None:
        return {}
    rows = [item for item in table if item.tag == TABLE_ROW_TAG]
    cells = [item for item in row if item.tag == TABLE_CELL_TAG]
    return {
        "table_index": int(table_index.get(table) or 0),
        "row_index": rows.index(row) + 1 if row in rows else 0,
        "col_index": cells.index(cell) + 1 if cell in cells else 0,
    }


def _container_flags(container_type: str) -> List[str]:
    flags: List[str] = []
    if container_type in {"header", "footer", "footnote", "endnote"}:
        flags.append("low_priority_container")
    if container_type == "textbox":
        flags.append("floating_text")
    if container_type in {"table_cell", "table_row"}:
        flags.append("table_source")
    if container_type in {"comment", "chart", "diagram", "glossary"}:
        flags.append("secondary_story")
    return flags


def _rewrite_policy_for_container(container_type: str) -> str:
    if container_type in {"comment", "chart", "diagram", "glossary"}:
        return "review_required"
    return "exact"


def _classify_docx_text_node(
    *,
    part_name: str,
    node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
) -> str:
    if not _is_docx_supported_text_node(node):
        if part_name.startswith("docProps/"):
            return "metadata"
        if part_name.startswith("customXml/"):
            return "custom_xml"
        if part_name.startswith("_rels/") or "/_rels/" in part_name:
            return "package_metadata"
        if part_name == "[Content_Types].xml":
            return "package_metadata"
        return "non_visible_xml_text"
    if _node_is_hidden_text(node, parent_map=parent_map):
        return "hidden_text"
    if node.tag == FIELD_INSTRUCTION_TAG:
        return "field_instruction"
    if part_name.startswith("docProps/"):
        return "metadata"
    if part_name.startswith("customXml/"):
        return "custom_xml"
    if part_name.startswith("word/embeddings/") or part_name.startswith("word/media/"):
        return "embedded_object"
    container_type = _part_container_type(part_name)
    if container_type in {"comment", "chart", "diagram", "glossary"}:
        return "secondary_visible_story"
    if container_type in {"header", "footer", "footnote", "endnote"}:
        return "secondary_visible_story"
    if part_name == "word/document.xml":
        if _node_in_textbox(node, parent_map=parent_map):
            return "visible_body_text"
        if _node_in_table(node, parent_map=parent_map):
            return "visible_body_text"
        if node.tag in TEXT_TAGS:
            return "visible_body_text"
    if part_name.startswith("word/"):
        if node.tag in TEXT_TAGS:
            return "secondary_visible_story"
        return "word_package_text"
    if part_name.startswith("_rels/") or "/_rels/" in part_name:
        return "package_metadata"
    if part_name == "[Content_Types].xml":
        return "package_metadata"
    return "unknown_xml_text"


def _is_docx_visible_text_node(
    node: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
) -> bool:
    if not _is_docx_supported_text_node(node):
        return False
    if not str(node.text or "").strip():
        return False
    if node.tag == FIELD_INSTRUCTION_TAG:
        return False
    if _node_is_hidden_text(node, parent_map=parent_map):
        return False
    return True


def _is_docx_supported_text_node(node: ET.Element) -> bool:
    return node.tag in TEXT_TAGS or node.tag == FIELD_INSTRUCTION_TAG


def _node_is_hidden_text(
    node: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
) -> bool:
    current: Optional[ET.Element] = node
    while current is not None:
        run_properties = current.find("w:rPr", NAMESPACES)
        if run_properties is not None and run_properties.find("w:vanish", NAMESPACES) is not None:
            return True
        if current.tag == HIDDEN_TEXT_TAG:
            return True
        current = parent_map.get(current)
    return False


def _node_in_textbox(
    node: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
) -> bool:
    current: Optional[ET.Element] = node
    while current is not None:
        if current.tag == TEXTBOX_CONTENT_TAG:
            return True
        current = parent_map.get(current)
    return False


def _node_in_table(
    node: ET.Element,
    *,
    parent_map: Dict[ET.Element, ET.Element],
) -> bool:
    current: Optional[ET.Element] = node
    while current is not None:
        if current.tag == TABLE_CELL_TAG:
            return True
        current = parent_map.get(current)
    return False


def _local_name(tag: str) -> str:
    text = str(tag or "")
    if "}" in text:
        return text.rsplit("}", 1)[-1]
    return text


def _compact_excerpt(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    max_len = max(0, int(limit or 0))
    if max_len <= 0 or len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"


def _increment_counter(counter: Dict[str, int], key: str, amount: int = 1) -> None:
    normalized_key = str(key or "unknown").strip() or "unknown"
    counter[normalized_key] = int(counter.get(normalized_key, 0) or 0) + int(amount or 0)


def _summarize_unit_coverage_by_part(units: Sequence[DocxVisibleTextUnit]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        entry = summary.setdefault(
            unit.part_name,
            {
                "unit_count": 0,
                "text_chars": 0,
                "container_types": [],
                "rewrite_policies": [],
            },
        )
        entry["unit_count"] += 1
        entry["text_chars"] += len(unit.text or "")
        if unit.container_type not in entry["container_types"]:
            entry["container_types"].append(unit.container_type)
        if unit.rewrite_policy not in entry["rewrite_policies"]:
            entry["rewrite_policies"].append(unit.rewrite_policy)
    return summary


def _page_for_current_unit(state: _DocxPartBuildState) -> Optional[int]:
    if state.part_name != "word/document.xml":
        return None
    if state.estimated_page_count <= 0:
        return max(1, state.current_page)
    return max(1, min(state.estimated_page_count, state.current_page))


def _advance_page_for_container(node: ET.Element, *, state: _DocxPartBuildState) -> None:
    for paragraph in node.iter(PARAGRAPH_TAG):
        _advance_page_for_paragraph(paragraph, state=state)


def _advance_page_for_paragraph(paragraph: ET.Element, *, state: _DocxPartBuildState) -> None:
    if state.part_name != "word/document.xml":
        return
    page_breaks = count_paragraph_page_breaks(paragraph)
    page_breaks += count_paragraph_section_page_breaks(paragraph)
    if page_breaks <= 0:
        return
    state.current_page = max(1, state.current_page + page_breaks)
    if state.estimated_page_count > 0:
        state.current_page = min(state.current_page, state.estimated_page_count)


def _normalize_range_entries(
    range_entries: Sequence[Dict[str, Any]],
    *,
    source_text: str | None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    for input_order, raw_entry in enumerate(range_entries):
        if not isinstance(raw_entry, dict):
            continue
        try:
            start = int(raw_entry.get("start"))
            end = int(raw_entry.get("end"))
        except (TypeError, ValueError):
            continue
        source_value = str(raw_entry.get("source_text") or "")
        replacement = str(raw_entry.get("replacement") or "")
        if start < 0 or end <= start or not source_value or not replacement:
            continue
        if source_text is not None and source_text[start:end] != source_value:
            continue
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        entry = dict(raw_entry)
        entry["start"] = start
        entry["end"] = end
        entry["source_text"] = source_value
        entry["replacement"] = replacement
        entry["input_order"] = int(raw_entry.get("input_order", input_order) or input_order)
        entries.append(entry)
    return sorted(entries, key=lambda item: (int(item["start"]), -len(str(item["source_text"]))))


def _build_docx_fragment_range_replacements(
    *,
    units: Sequence[DocxVisibleTextUnit],
    entries: Sequence[Dict[str, Any]],
    extracted_text: str,
) -> List[Dict[str, Any]]:
    replacements: List[Dict[str, Any]] = []
    for entry in entries:
        start = int(entry["start"])
        end = int(entry["end"])
        if extracted_text[start:end] != str(entry.get("source_text") or ""):
            continue
        matching_units = [unit for unit in units if start < unit.end and end > unit.start]
        if len(matching_units) != 1:
            continue
        unit = matching_units[0]
        touched = [
            fragment
            for fragment in unit.fragments
            if start < fragment.end and end > fragment.start
        ]
        if not touched or any(fragment.virtual for fragment in touched):
            continue
        real_fragments = [fragment for fragment in touched if fragment.node is not None]
        if not real_fragments:
            continue
        part_names = {fragment.part_name for fragment in real_fragments}
        if len(part_names) != 1:
            continue
        replacements.append(
            {
                "part_name": real_fragments[0].part_name,
                "start": start,
                "end": end,
                "input_order": int(entry.get("input_order", -1)),
                "replacement": str(entry.get("replacement") or ""),
                "fragments": real_fragments,
            }
        )
    return replacements


def _range_entry_diagnostics(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for entry in entries:
        diagnostics.append(
            {
                "start": int(entry.get("start", -1) or -1),
                "end": int(entry.get("end", -1) or -1),
                "source_text": str(entry.get("source_text") or ""),
                "replacement": str(entry.get("replacement") or ""),
                "entity_type": str(entry.get("entity_type") or ""),
                "input_order": int(entry.get("input_order", -1) or -1),
            }
        )
    return diagnostics


def _apply_docx_fragment_range_replacement(replacement: Dict[str, Any]) -> bool:
    fragments = [
        fragment
        for fragment in replacement.get("fragments", [])
        if isinstance(fragment, DocxTextFragment) and fragment.node is not None
    ]
    if not fragments:
        return False
    start = int(replacement["start"])
    end = int(replacement["end"])
    replacement_text = str(replacement.get("replacement") or "")

    first = fragments[0]
    last = fragments[-1]
    first_node = first.node
    last_node = last.node
    if first_node is None or last_node is None:
        return False

    start_char = _fragment_original_start_index(first, start)
    end_char = _fragment_original_end_index(last, end)
    if start_char > len(first_node.text or "") or end_char > len(last_node.text or ""):
        return False

    if first_node is last_node:
        original = first_node.text or ""
        first_node.text = original[:start_char] + replacement_text + original[end_char:]
        _sync_space_preserve(first_node, first_node.text or "")
        return first_node.text != original

    touched_nodes: List[ET.Element] = []
    for fragment in fragments:
        node = fragment.node
        if node is not None and node not in touched_nodes:
            touched_nodes.append(node)
    if len(touched_nodes) < 2:
        return False

    original_texts = [node.text or "" for node in touched_nodes]
    first_text = original_texts[0]
    last_text = original_texts[-1]
    right_cut = max(0, min(end_char, len(last_text)))
    touched_lengths = [len(first_text) - start_char]
    touched_lengths.extend(len(text) for text in original_texts[1:-1])
    touched_lengths.append(right_cut)
    replacement_segments = _split_replacement_across_fragments(replacement_text, touched_lengths)
    if len(replacement_segments) != len(touched_nodes):
        return False

    touched_nodes[0].text = first_text[:start_char] + replacement_segments[0]
    for node, segment in zip(touched_nodes[1:-1], replacement_segments[1:-1]):
        node.text = segment
    touched_nodes[-1].text = replacement_segments[-1] + last_text[right_cut:]
    for node in touched_nodes:
        _sync_space_preserve(node, node.text or "")
    return [node.text or "" for node in touched_nodes] != original_texts


def _fragment_original_start_index(fragment: DocxTextFragment, global_start: int) -> int:
    local_index = max(0, int(global_start) - int(fragment.start))
    source_indexes = _fragment_source_indexes(fragment)
    if source_indexes and local_index < len(source_indexes):
        return max(0, min(source_indexes[local_index], len(fragment.node.text or "") if fragment.node is not None else source_indexes[local_index]))
    return local_index


def _fragment_original_end_index(fragment: DocxTextFragment, global_end: int) -> int:
    local_exclusive = max(0, int(global_end) - int(fragment.start))
    source_indexes = _fragment_source_indexes(fragment)
    if source_indexes and local_exclusive > 0 and local_exclusive - 1 < len(source_indexes):
        return max(0, source_indexes[local_exclusive - 1] + 1)
    return local_exclusive


def extract_paragraph_text(paragraph: ET.Element) -> str:
    """Collect text content from a paragraph while skipping nested paragraphs."""
    return "".join(_iter_paragraph_text_fragments(paragraph))


def extract_block_texts(root: ET.Element) -> List[str]:
    """Collect visible Word text blocks while preserving table row context."""
    blocks: List[str] = []
    body = root.find(".//w:body", NAMESPACES)
    if body is not None:
        for child in _iter_docx_effective_children(body):
            blocks.extend(_extract_top_level_block_texts(child))
        return blocks

    for paragraph in root.iterfind(".//w:p", NAMESPACES):
        paragraph_text = extract_paragraph_text(paragraph)
        if paragraph_text.strip():
            blocks.append(paragraph_text)
    return blocks


def _extract_top_level_block_texts(node: ET.Element) -> List[str]:
    if node.tag == PARAGRAPH_TAG:
        return _extract_paragraph_block_texts(node)
    if node.tag == TABLE_TAG:
        return _extract_table_row_texts(node)
    if node.tag == TEXTBOX_CONTENT_TAG:
        return _extract_textbox_block_texts(node)

    blocks: List[str] = []
    for child in _iter_docx_effective_children(node):
        if child.tag == PARAGRAPH_TAG:
            blocks.extend(_extract_paragraph_block_texts(child))
        elif child.tag == TABLE_TAG:
            blocks.extend(_extract_table_row_texts(child))
        elif child.tag == TEXTBOX_CONTENT_TAG:
            blocks.extend(_extract_textbox_block_texts(child))
        else:
            blocks.extend(_extract_top_level_block_texts(child))
    return blocks


def _extract_paragraph_block_texts(paragraph: ET.Element) -> List[str]:
    blocks: List[str] = []
    text = extract_paragraph_text(paragraph)
    if text.strip():
        blocks.append(text)
    for textbox in paragraph.iter(TEXTBOX_CONTENT_TAG):
        blocks.extend(_extract_textbox_block_texts(textbox))
    return blocks


def _extract_textbox_block_texts(textbox: ET.Element) -> List[str]:
    blocks: List[str] = []
    for child in _iter_docx_effective_children(textbox):
        if child.tag == PARAGRAPH_TAG:
            text = extract_paragraph_text(child)
            if text.strip():
                blocks.append(text)
        elif child.tag == TABLE_TAG:
            blocks.extend(_extract_table_row_texts(child))
        else:
            blocks.extend(_extract_top_level_block_texts(child))
    return blocks


def _extract_table_row_texts(table: ET.Element) -> List[str]:
    rows: List[str] = []
    for row in table.iterfind("./w:tr", NAMESPACES):
        cells: List[str] = []
        for cell in row.iterfind("./w:tc", NAMESPACES):
            cell_parts: List[str] = []
            for child in _iter_docx_effective_children(cell):
                if child.tag == PARAGRAPH_TAG:
                    for paragraph_text in _extract_paragraph_block_texts(child):
                        if paragraph_text.strip():
                            cell_parts.append(paragraph_text.strip())
                elif child.tag == TABLE_TAG:
                    cell_parts.extend(_extract_table_row_texts(child))
                elif child.tag == TEXTBOX_CONTENT_TAG:
                    cell_parts.extend(_extract_textbox_block_texts(child))
                else:
                    for nested_text in _extract_top_level_block_texts(child):
                        if nested_text.strip():
                            cell_parts.append(nested_text.strip())
            cell_text = " ".join(part for part in cell_parts if part).strip()
            cells.append(cell_text)
        row_text = "\t".join(cell for cell in cells if cell).strip()
        if row_text:
            rows.append(row_text)
    return rows


def count_paragraph_page_breaks(paragraph: ET.Element) -> int:
    """Count explicit page-break markers inside a paragraph."""
    count = 0
    for node in paragraph.iter():
        if node.tag == f"{{{WORD_NAMESPACE}}}br" and node.attrib.get(f"{{{WORD_NAMESPACE}}}type") == "page":
            count += 1
        elif node.tag == f"{{{WORD_NAMESPACE}}}lastRenderedPageBreak":
            count += 1
    return count


def count_paragraph_section_page_breaks(paragraph: ET.Element) -> int:
    """Count non-continuous section breaks attached to a paragraph."""
    count = 0
    for section in paragraph.iter(f"{{{WORD_NAMESPACE}}}sectPr"):
        type_node = section.find("w:type", NAMESPACES)
        section_type = str(type_node.attrib.get(f"{{{WORD_NAMESPACE}}}val") if type_node is not None else "").strip().lower()
        if section_type == "continuous":
            continue
        count += 1
    return count


def _iter_paragraph_text_fragments(node: ET.Element) -> Iterator[str]:
    for child in _iter_docx_effective_children(node):
        if child.tag == PARAGRAPH_TAG:
            continue

        if child.tag in TEXT_TAGS:
            if child.text:
                yield child.text
            continue

        if child.tag == TAB_TAG:
            yield "\t"
            continue

        if child.tag in BREAK_TAGS:
            yield "\n"
            continue

        yield from _iter_paragraph_text_fragments(child)


def replace_text_in_docx(file_path: str | Path, replacements: Sequence[Tuple[str, str]]) -> bool:
    """Apply replacements to relevant DOCX XML parts in-place."""
    normalized = normalize_replacements(replacements)
    if not normalized:
        return False

    source_path = Path(file_path)
    with zipfile.ZipFile(source_path) as archive:
        target_parts = set(list_docx_visible_text_parts(archive))

    if not target_parts:
        return False

    modified = False
    temp_fd, temp_name = tempfile.mkstemp(suffix=source_path.suffix, dir=str(source_path.parent))
    os.close(temp_fd)

    try:
        with zipfile.ZipFile(source_path, "r") as source_archive, zipfile.ZipFile(
            temp_name,
            "w",
        ) as target_archive:
            for item in source_archive.infolist():
                data = source_archive.read(item.filename)
                if item.filename in target_parts:
                    updated = replace_text_in_xml_part(data, normalized, part_name=item.filename)
                    if updated is not None:
                        data = updated
                        modified = True
                target_archive.writestr(item, data)

        if modified:
            os.replace(temp_name, source_path)
        else:
            os.remove(temp_name)
        return modified
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def replace_text_in_xml_part(
    xml_bytes: bytes,
    replacements: Sequence[Tuple[str, str]],
    *,
    part_name: str = "word/document.xml",
) -> bytes | None:
    """Apply replacements to visible text units in a single DOCX XML part."""
    root = ET.fromstring(xml_bytes)
    modified = False

    units = collect_docx_visible_text_units_from_root(root, part_name=part_name)
    assign_docx_unit_global_offsets(units)
    for unit in units:
        real_fragments = [
            fragment
            for fragment in unit.fragments
            if not fragment.virtual and fragment.node is not None
        ]
        if not real_fragments:
            continue
        nodes: List[ET.Element] = []
        for fragment in real_fragments:
            if fragment.node is not None and fragment.node not in nodes:
                nodes.append(fragment.node)
        original_texts = [node.text or "" for node in nodes]
        updated_texts = apply_replacements_to_fragments(original_texts, replacements)
        if updated_texts == original_texts:
            continue
        for node, updated_text in zip(nodes, updated_texts):
            node.text = updated_text
            _sync_space_preserve(node, updated_text)
        modified = True

    if not modified:
        for paragraph in root.iterfind(".//w:p", NAMESPACES):
            if replace_text_in_paragraph_xml(paragraph, replacements):
                modified = True

    if not modified:
        return None

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def replace_text_in_paragraph_xml(
    paragraph: ET.Element,
    replacements: Sequence[Tuple[str, str]],
) -> bool:
    """Apply replacements conservatively inside existing text nodes only."""
    text_nodes = [node for node in _iter_paragraph_text_nodes(paragraph) if node.text]
    if not text_nodes:
        return False

    modified = False
    for node in text_nodes:
        original_text = node.text or ""
        updated_text = original_text
        for source_text, replacement in replacements:
            if source_text and source_text in updated_text:
                updated_text = updated_text.replace(source_text, replacement)
        if updated_text == original_text:
            continue
        node.text = updated_text
        _sync_space_preserve(node, updated_text)
        modified = True

    return modified


def _iter_paragraph_text_nodes(node: ET.Element) -> Iterator[ET.Element]:
    for child in node:
        if child.tag == PARAGRAPH_TAG:
            continue

        if child.tag in TEXT_TAGS:
            yield child
            continue

        yield from _iter_paragraph_text_nodes(child)


def apply_replacements_to_fragments(
    fragments: Sequence[str],
    replacements: Sequence[Tuple[str, str]],
) -> List[str]:
    """Apply ordered replacements across a list of adjacent text fragments."""
    if not fragments:
        return []

    full_text = "".join(fragments)
    if not full_text:
        return list(fragments)

    matches = collect_replacement_matches(full_text, replacements)
    if not matches:
        return list(fragments)

    char_map: List[Tuple[int, int]] = []
    updated_fragments = list(fragments)
    for fragment_index, fragment_text in enumerate(updated_fragments):
        for char_index, _ in enumerate(fragment_text):
            char_map.append((fragment_index, char_index))

    for start, end, replacement in sorted(matches, key=lambda item: item[0], reverse=True):
        start_fragment, start_char = char_map[start]
        end_fragment, end_char = char_map[end - 1]

        if start_fragment == end_fragment:
            text = updated_fragments[start_fragment]
            updated_fragments[start_fragment] = (
                text[:start_char] + replacement + text[end_char + 1 :]
            )
            continue

        start_text = updated_fragments[start_fragment]
        end_text = updated_fragments[end_fragment]
        left_prefix = start_text[:start_char]
        right_suffix = end_text[end_char + 1 :]
        touched_lengths = [len(start_text) - start_char]
        for index in range(start_fragment + 1, end_fragment):
            touched_lengths.append(len(updated_fragments[index]))
        touched_lengths.append(end_char + 1)
        replacement_segments = _split_replacement_across_fragments(replacement, touched_lengths)
        updated_fragments[start_fragment] = left_prefix + replacement_segments[0]

        middle_indexes = range(start_fragment + 1, end_fragment)
        for segment, fragment_index in zip(replacement_segments[1:-1], middle_indexes):
            updated_fragments[fragment_index] = segment

        updated_fragments[end_fragment] = replacement_segments[-1] + right_suffix

    return updated_fragments


def collect_replacement_matches(
    text: str,
    replacements: Sequence[Tuple[str, str]],
) -> List[Tuple[int, int, str]]:
    """Find non-overlapping replacement matches, preferring longer sources first."""
    matches: List[Tuple[int, int, str]] = []
    occupied = [False] * len(text)

    for original, replacement in replacements:
        if not original:
            continue

        start = 0
        while True:
            match_index = text.find(original, start)
            if match_index == -1:
                break

            match_end = match_index + len(original)
            if not any(occupied[match_index:match_end]):
                matches.append((match_index, match_end, replacement))
                for index in range(match_index, match_end):
                    occupied[index] = True

            start = match_index + len(original)

    return matches


def normalize_replacements(
    replacements: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Drop empty replacements and keep longest sources first."""
    unique: List[Tuple[str, str]] = []
    seen = set()
    for source_text, replacement in replacements:
        if not source_text or replacement is None:
            continue
        key = (source_text, replacement)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)

    return sorted(unique, key=lambda item: len(item[0]), reverse=True)


def _split_replacement_across_fragments(replacement: str, fragment_lengths: Sequence[int]) -> List[str]:
    if not fragment_lengths:
        return []
    if len(fragment_lengths) == 1:
        return [replacement]

    total_length = sum(max(0, length) for length in fragment_lengths)
    if total_length <= 0:
        return [replacement] + [""] * (len(fragment_lengths) - 1)

    target_length = len(replacement)
    base_allocations: List[int] = []
    remainders: List[tuple[float, int]] = []
    allocated = 0
    for index, length in enumerate(fragment_lengths):
        exact = target_length * max(0, length) / total_length
        allocation = int(exact)
        base_allocations.append(allocation)
        remainders.append((exact - allocation, index))
        allocated += allocation

    remaining = target_length - allocated
    for _, index in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        base_allocations[index] += 1
        remaining -= 1

    segments: List[str] = []
    cursor = 0
    for allocation in base_allocations[:-1]:
        next_cursor = min(target_length, cursor + allocation)
        segments.append(replacement[cursor:next_cursor])
        cursor = next_cursor
    segments.append(replacement[cursor:])
    return segments


def _sync_space_preserve(node: ET.Element, text: str) -> None:
    if text[:1].isspace() or text[-1:].isspace():
        node.set(XML_SPACE_ATTR, "preserve")
        return

    node.attrib.pop(XML_SPACE_ATTR, None)
