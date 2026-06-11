"""OOXML-level DOCX merge utilities."""

from __future__ import annotations

import copy
import posixpath
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

REL_NS = {"rel": PKG_REL_NS}
W = f"{{{W_NS}}}"
R_ID = f"{{{R_NS}}}id"
W_ID = f"{{{W_NS}}}id"
W_VAL = f"{{{W_NS}}}val"
W_STYLE_ID = f"{{{W_NS}}}styleId"

DOCUMENT_PART = "word/document.xml"
DOCUMENT_RELS_PART = "word/_rels/document.xml.rels"
CONTENT_TYPES_PART = "[Content_Types].xml"

REL_TYPE_STYLES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
REL_TYPE_NUMBERING = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering"
REL_TYPE_COMMENTS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
REL_TYPE_FOOTNOTES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
REL_TYPE_ENDNOTES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"
REL_TYPE_HEADER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"
REL_TYPE_FOOTER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer"
REL_TYPE_OFFICE_DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
REL_TYPE_CORE_PROPERTIES = "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"
REL_TYPE_EXTENDED_PROPERTIES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"
REL_TYPE_CUSTOM_PROPERTIES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties"
REL_TYPE_CUSTOM_XML = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/customXml"

DOCUMENT_REL_TYPES_ALWAYS_KEEP = {
    REL_TYPE_STYLES,
    REL_TYPE_NUMBERING,
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings",
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings",
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable",
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme",
}

ROOT_REL_TYPES_ALWAYS_KEEP = {
    REL_TYPE_OFFICE_DOCUMENT,
    REL_TYPE_CORE_PROPERTIES,
    REL_TYPE_EXTENDED_PROPERTIES,
    REL_TYPE_CUSTOM_PROPERTIES,
}

CT_STYLES = "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"
CT_NUMBERING = "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"
CT_COMMENTS = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"
CT_FOOTNOTES = "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"
CT_ENDNOTES = "application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml"

INDEXED_PARTS = {
    "comments": {
        "part": "word/comments.xml",
        "root": "comments",
        "item": "comment",
        "markers": {"commentRangeStart", "commentRangeEnd", "commentReference"},
        "rel_type": REL_TYPE_COMMENTS,
        "content_type": CT_COMMENTS,
    },
    "footnotes": {
        "part": "word/footnotes.xml",
        "root": "footnotes",
        "item": "footnote",
        "markers": {"footnoteReference"},
        "rel_type": REL_TYPE_FOOTNOTES,
        "content_type": CT_FOOTNOTES,
    },
    "endnotes": {
        "part": "word/endnotes.xml",
        "root": "endnotes",
        "item": "endnote",
        "markers": {"endnoteReference"},
        "rel_type": REL_TYPE_ENDNOTES,
        "content_type": CT_ENDNOTES,
    },
}

def merge_docx_documents(source_paths: Sequence[str | Path], output_path: str | Path) -> None:
    """Merge DOCX packages while preserving body-level OOXML and referenced parts."""

    paths = [Path(path) for path in source_paths if str(path or "").strip()]
    if not paths:
        raise ValueError("merge_docx_documents requires at least one source document")
    if len(paths) == 1:
        Path(output_path).write_bytes(paths[0].read_bytes())
        return

    destination_files = _read_docx_files(paths[0])
    content_types = _xml_root(destination_files, CONTENT_TYPES_PART)
    base_document = _xml_root(destination_files, DOCUMENT_PART)
    base_body = _body(base_document)
    previous_section = _pop_body_section_properties(base_body)

    for document_index, source_path in enumerate(paths[1:], start=1):
        source_files = _read_docx_files(source_path)
        source_document = _xml_root(source_files, DOCUMENT_PART)
        source_body = _body(source_document)
        source_children = [copy.deepcopy(child) for child in source_body]
        source_section = _pop_body_section_properties_from_children(source_children)

        if previous_section is not None:
            _append_section_break(base_body, previous_section)

        _merge_styles(source_files, destination_files, content_types, source_children)
        _merge_numbering(source_files, destination_files, content_types, source_children)
        for config in INDEXED_PARTS.values():
            _merge_indexed_part(
                source_files=source_files,
                destination_files=destination_files,
                content_types=content_types,
                body_elements=source_children,
                config=config,
                document_index=document_index,
            )

        _copy_document_relationships_for_elements(
            source_files=source_files,
            destination_files=destination_files,
            content_types=content_types,
            elements=source_children,
            document_index=document_index,
        )
        if source_section is not None:
            _copy_document_relationships_for_elements(
                source_files=source_files,
                destination_files=destination_files,
                content_types=content_types,
                elements=[source_section],
                document_index=document_index,
            )

        for child in source_children:
            base_body.append(child)
        previous_section = source_section

    if previous_section is not None:
        base_body.append(previous_section)

    destination_files[DOCUMENT_PART] = _to_xml_bytes(base_document)
    destination_files[CONTENT_TYPES_PART] = _to_xml_bytes(content_types)
    _write_docx_files(destination_files, Path(output_path))


def write_docx_body_slice(
    source_path: str | Path,
    output_path: str | Path,
    *,
    group_text: str,
    page_texts: Sequence[str] | None = None,
) -> None:
    """Copy a DOCX package and keep whole body OOXML nodes for one text group."""

    source = Path(source_path)
    destination = Path(output_path)
    files = _read_docx_files(source)
    document = _xml_root(files, DOCUMENT_PART)
    body = _body(document)
    original_children = [copy.deepcopy(child) for child in body]
    selected_children = _select_body_children_for_text_group(
        original_children,
        group_text=group_text,
        page_texts=page_texts or [],
    )
    if not selected_children:
        raise ValueError("DOCX body slice could not match the requested group text")

    body.clear()
    for child in selected_children:
        body.append(child)
    if not any(_local_name(child) == "sectPr" for child in body):
        section = _last_section_properties(original_children)
        if section is not None:
            body.append(copy.deepcopy(section))

    files[DOCUMENT_PART] = _to_xml_bytes(document)
    _prune_docx_slice_package(files, document)
    _write_docx_files(files, destination)


def _prune_docx_slice_package(files: Dict[str, bytes], document_root: etree._Element) -> None:
    """Drop DOCX parts that are not reachable from the sliced document body."""

    _prune_indexed_text_parts(files, document_root)
    _filter_document_relationships_for_slice(files, document_root)
    reachable_parts = _reachable_docx_parts(files)
    _drop_unreachable_parts(files, reachable_parts)
    _filter_relationship_parts_to_existing_targets(files)
    _prune_content_types(files)


def _prune_indexed_text_parts(files: Dict[str, bytes], document_root: etree._Element) -> None:
    for config in INDEXED_PARTS.values():
        part_name = str(config["part"])
        if part_name not in files:
            continue
        used_ids = _used_indexed_ids([document_root], set(config["markers"]))  # type: ignore[arg-type]
        if not used_ids:
            files.pop(part_name, None)
            files.pop(_rels_part_for(part_name), None)
            continue
        root = _optional_xml_root(files, part_name)
        if root is None:
            continue
        item_name = str(config["item"])
        for item in list(root.findall(f"{W}{item_name}")):
            item_id = str(item.attrib.get(W_ID) or "")
            if item_id in used_ids or (item_id.lstrip("-").isdigit() and int(item_id) < 0):
                continue
            root.remove(item)
        files[part_name] = _to_xml_bytes(root)


def _filter_document_relationships_for_slice(files: Dict[str, bytes], document_root: etree._Element) -> None:
    rels_root = _optional_xml_root(files, DOCUMENT_RELS_PART)
    if rels_root is None:
        return
    used_ids = _used_relationship_ids([document_root])
    used_comment_ids = _used_indexed_ids([document_root], set(INDEXED_PARTS["comments"]["markers"]))  # type: ignore[arg-type]
    used_footnote_ids = _used_indexed_ids([document_root], set(INDEXED_PARTS["footnotes"]["markers"]))  # type: ignore[arg-type]
    used_endnote_ids = _used_indexed_ids([document_root], set(INDEXED_PARTS["endnotes"]["markers"]))  # type: ignore[arg-type]

    for relationship in list(_relationship_elements(rels_root)):
        rel_id = str(relationship.attrib.get("Id") or "")
        rel_type = str(relationship.attrib.get("Type") or "")
        target_mode = relationship.attrib.get("TargetMode")
        keep = False
        if rel_type in DOCUMENT_REL_TYPES_ALWAYS_KEEP:
            keep = True
        elif rel_type in {REL_TYPE_HEADER, REL_TYPE_FOOTER}:
            keep = rel_id in used_ids
        elif rel_type == REL_TYPE_COMMENTS:
            keep = bool(used_comment_ids)
        elif rel_type == REL_TYPE_FOOTNOTES:
            keep = bool(used_footnote_ids)
        elif rel_type == REL_TYPE_ENDNOTES:
            keep = bool(used_endnote_ids)
        elif rel_type == REL_TYPE_CUSTOM_XML:
            keep = False
        elif target_mode == "External":
            keep = rel_id in used_ids
        else:
            keep = rel_id in used_ids
        if not keep:
            rels_root.remove(relationship)
    files[DOCUMENT_RELS_PART] = _to_xml_bytes(rels_root)


def _reachable_docx_parts(files: Dict[str, bytes]) -> Set[str]:
    reachable: Set[str] = {CONTENT_TYPES_PART}
    pending: List[str] = []

    if "_rels/.rels" in files:
        reachable.add("_rels/.rels")
        root_rels = _optional_xml_root(files, "_rels/.rels")
        if root_rels is not None:
            for relationship in list(_relationship_elements(root_rels)):
                rel_type = str(relationship.attrib.get("Type") or "")
                target_mode = relationship.attrib.get("TargetMode")
                target = str(relationship.attrib.get("Target") or "")
                if target_mode == "External":
                    continue
                if rel_type not in ROOT_REL_TYPES_ALWAYS_KEEP:
                    root_rels.remove(relationship)
                    continue
                target_part = _resolve_relationship_target("", target)
                if target_part in files and target_part not in reachable:
                    reachable.add(target_part)
                    pending.append(target_part)
            files["_rels/.rels"] = _to_xml_bytes(root_rels)
    else:
        reachable.add(DOCUMENT_PART)
        pending.append(DOCUMENT_PART)

    if DOCUMENT_PART in files and DOCUMENT_PART not in reachable:
        reachable.add(DOCUMENT_PART)
        pending.append(DOCUMENT_PART)

    while pending:
        part_name = pending.pop()
        rels_part = _rels_part_for(part_name)
        if rels_part not in files:
            continue
        reachable.add(rels_part)
        rels_root = _optional_xml_root(files, rels_part)
        if rels_root is None:
            continue
        for relationship in _relationship_elements(rels_root):
            if relationship.attrib.get("TargetMode") == "External":
                continue
            target_part = _resolve_relationship_target(
                part_name,
                str(relationship.attrib.get("Target") or ""),
            )
            if target_part in files and target_part not in reachable:
                reachable.add(target_part)
                pending.append(target_part)
    return reachable


def _drop_unreachable_parts(files: Dict[str, bytes], reachable_parts: Set[str]) -> None:
    for name in list(files):
        if name in reachable_parts:
            continue
        files.pop(name, None)


def _filter_relationship_parts_to_existing_targets(files: Dict[str, bytes]) -> None:
    existing_parts = set(files)
    for rels_part in list(files):
        if not rels_part.endswith(".rels"):
            continue
        root = _optional_xml_root(files, rels_part)
        if root is None:
            continue
        source_part = _source_part_for_rels(rels_part)
        for relationship in list(_relationship_elements(root)):
            if relationship.attrib.get("TargetMode") == "External":
                continue
            target_part = _resolve_relationship_target(
                source_part,
                str(relationship.attrib.get("Target") or ""),
            )
            if target_part not in existing_parts:
                root.remove(relationship)
        if _relationship_elements(root):
            files[rels_part] = _to_xml_bytes(root)
        elif rels_part != "_rels/.rels":
            files.pop(rels_part, None)


def _prune_content_types(files: Dict[str, bytes]) -> None:
    root = _optional_xml_root(files, CONTENT_TYPES_PART)
    if root is None:
        return
    existing_parts = {f"/{name}" for name in files}
    for override in list(root.findall(f"{{{CT_NS}}}Override")):
        if override.attrib.get("PartName") not in existing_parts:
            root.remove(override)
    files[CONTENT_TYPES_PART] = _to_xml_bytes(root)


def _read_docx_files(path: Path) -> Dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _child_visible_text(child: etree._Element) -> str:
    fragments: List[str] = []
    for paragraph in child.iter(f"{W}p"):
        text = "".join(text for _node, text, _kind in _visible_text_units(paragraph))
        if text.strip():
            fragments.append(text)
    if _local_name(child) == "p" and not fragments:
        text = "".join(text for _node, text, _kind in _visible_text_units(child))
        if text.strip():
            fragments.append(text)
    return "\n".join(fragments)


def _body_child_text_spans(children: Sequence[etree._Element]) -> tuple[str, List[tuple[int, int, int]]]:
    parts: List[str] = []
    spans: List[tuple[int, int, int]] = []
    cursor = 0
    for index, child in enumerate(children):
        if _local_name(child) == "sectPr":
            continue
        text = _child_visible_text(child)
        if not text.strip():
            spans.append((index, cursor, cursor))
            continue
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((index, start, cursor))
    return "".join(parts), spans


def _select_body_children_for_text_group(
    children: Sequence[etree._Element],
    *,
    group_text: str,
    page_texts: Sequence[str],
) -> List[etree._Element]:
    body_text, spans = _body_child_text_spans(children)
    ranges = _locate_group_ranges_in_body_text(
        body_text=body_text,
        group_text=group_text,
        page_texts=page_texts,
    )
    if not ranges:
        return []

    selected_indexes: Set[int] = set()
    for index, start, end in spans:
        if end < start:
            continue
        if start == end:
            if any(_range_contains_empty_boundary(start, group_start, group_end) for group_start, group_end in ranges):
                selected_indexes.add(index)
            continue
        if any(start < group_end and end > group_start for group_start, group_end in ranges):
            selected_indexes.add(index)

    selected = [copy.deepcopy(children[index]) for index in sorted(selected_indexes)]
    if selected and _local_name(selected[-1]) == "sectPr" and len(selected) > 1:
        return selected
    section = _section_properties_for_selected_indexes(children, selected_indexes)
    if section is not None and selected and _local_name(selected[-1]) != "sectPr":
        selected.append(copy.deepcopy(section))
    return selected


def _locate_group_ranges_in_body_text(
    *,
    body_text: str,
    group_text: str,
    page_texts: Sequence[str],
) -> List[tuple[int, int]]:
    normalized_group = str(group_text or "").strip()
    if normalized_group:
        found = body_text.find(normalized_group)
        if found >= 0:
            return [(found, found + len(normalized_group))]

    ranges: List[tuple[int, int]] = []
    cursor = 0
    for page_text in page_texts:
        text = str(page_text or "").strip()
        if not text:
            continue
        found = body_text.find(text, cursor)
        if found < 0:
            found = body_text.find(text)
        if found < 0:
            continue
        end = found + len(text)
        ranges.append((found, end))
        cursor = max(cursor, end)
    if not ranges:
        return []
    return ranges


def _range_contains_empty_boundary(point: int, start: int, end: int) -> bool:
    return start <= point <= end


def _visible_text_units(node: etree._Element) -> List[tuple[etree._Element, str, str]]:
    units: List[tuple[etree._Element, str, str]] = []
    for child in node.iter():
        if child is node:
            continue
        local_name = _local_name(child)
        if local_name in {"t", "delText"} and child.text:
            units.append((child, child.text, "text"))
        elif local_name == "tab":
            units.append((child, "\t", "tab"))
        elif local_name in {"br", "cr"}:
            units.append((child, "\n", "break"))
    return units


def _last_section_properties(children: Sequence[etree._Element]) -> Optional[etree._Element]:
    for child in reversed(children):
        if _local_name(child) == "sectPr":
            return child
        if _local_name(child) != "p":
            continue
        paragraph_properties = child.find(f"{W}pPr")
        if paragraph_properties is None:
            continue
        section = paragraph_properties.find(f"{W}sectPr")
        if section is not None:
            return section
    return None


def _section_properties_for_selected_indexes(
    children: Sequence[etree._Element],
    selected_indexes: Set[int],
) -> Optional[etree._Element]:
    if not selected_indexes:
        return _last_section_properties(children)

    last_selected = max(selected_indexes)
    for child in list(children)[last_selected:]:
        section = _section_properties_from_child(child)
        if section is not None:
            return section
    return _last_section_properties(children)


def _section_properties_from_child(child: etree._Element) -> Optional[etree._Element]:
    if _local_name(child) == "sectPr":
        return child
    if _local_name(child) != "p":
        return None
    paragraph_properties = child.find(f"{W}pPr")
    if paragraph_properties is None:
        return None
    return paragraph_properties.find(f"{W}sectPr")


def _write_docx_files(files: Dict[str, bytes], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.writestr(name, files[name])


def _xml_root(files: Dict[str, bytes], part_name: str) -> etree._Element:
    if part_name not in files:
        raise ValueError(f"DOCX package is missing {part_name}")
    return etree.fromstring(files[part_name])


def _optional_xml_root(files: Dict[str, bytes], part_name: str) -> Optional[etree._Element]:
    if part_name not in files:
        return None
    return etree.fromstring(files[part_name])


def _to_xml_bytes(root: etree._Element) -> bytes:
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=False)


def _body(document_root: etree._Element) -> etree._Element:
    body = document_root.find(f".//{W}body")
    if body is None:
        raise ValueError("DOCX document.xml is missing w:body")
    return body


def _local_name(node: etree._Element) -> str:
    return etree.QName(node).localname


def _pop_body_section_properties(body: etree._Element) -> Optional[etree._Element]:
    children = list(body)
    if children and _local_name(children[-1]) == "sectPr":
        section = children[-1]
        body.remove(section)
        return section
    return None


def _pop_body_section_properties_from_children(children: List[etree._Element]) -> Optional[etree._Element]:
    if children and _local_name(children[-1]) == "sectPr":
        return children.pop()
    return None


def _append_section_break(body: etree._Element, section: etree._Element) -> None:
    paragraph = etree.Element(f"{W}p", nsmap=section.nsmap)
    paragraph_properties = etree.SubElement(paragraph, f"{W}pPr")
    section_copy = copy.deepcopy(section)
    section_type = section_copy.find(f"{W}type")
    if section_type is None:
        section_type = etree.Element(f"{W}type")
        section_copy.insert(0, section_type)
    section_type.set(W_VAL, "nextPage")
    paragraph_properties.append(section_copy)
    body.append(paragraph)


def _rels_part_for(part_name: str) -> str:
    directory = posixpath.dirname(part_name)
    filename = posixpath.basename(part_name)
    if directory:
        return f"{directory}/_rels/{filename}.rels"
    return f"_rels/{filename}.rels"


def _part_directory(part_name: str) -> str:
    return posixpath.dirname(part_name)


def _resolve_target(source_part: str, target: str) -> str:
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(_part_directory(source_part), target))


def _resolve_relationship_target(source_part: str, target: str) -> str:
    if not source_part:
        if target.startswith("/"):
            return posixpath.normpath(target.lstrip("/"))
        return posixpath.normpath(target.lstrip("/"))
    return _resolve_target(source_part, target)


def _relative_target(source_part: str, target_part: str) -> str:
    source_dir = _part_directory(source_part)
    if not source_dir:
        return target_part
    return posixpath.relpath(target_part, source_dir)


def _source_part_for_rels(rels_part: str) -> str:
    if rels_part == "_rels/.rels":
        return ""
    if "/_rels/" not in rels_part or not rels_part.endswith(".rels"):
        return ""
    directory, filename = rels_part.rsplit("/_rels/", 1)
    return posixpath.join(directory, filename[:-5])


def _relationship_elements(root: etree._Element) -> List[etree._Element]:
    return root.findall("rel:Relationship", REL_NS)


def _relationships_root(files: Dict[str, bytes], rels_part: str) -> etree._Element:
    root = _optional_xml_root(files, rels_part)
    if root is not None:
        return root
    return etree.Element(f"{{{PKG_REL_NS}}}Relationships", nsmap={None: PKG_REL_NS})


def _next_relationship_id(root: etree._Element) -> str:
    max_id = 0
    for relationship in _relationship_elements(root):
        rel_id = str(relationship.attrib.get("Id") or "")
        if rel_id.startswith("rId") and rel_id[3:].isdigit():
            max_id = max(max_id, int(rel_id[3:]))
    return f"rId{max_id + 1}"


def _add_relationship(
    files: Dict[str, bytes],
    source_part: str,
    rel_type: str,
    target: str,
    *,
    target_mode: Optional[str] = None,
) -> str:
    rels_part = _rels_part_for(source_part)
    root = _relationships_root(files, rels_part)
    rel_id = _next_relationship_id(root)
    attributes = {"Id": rel_id, "Type": rel_type, "Target": target}
    if target_mode:
        attributes["TargetMode"] = target_mode
    etree.SubElement(root, f"{{{PKG_REL_NS}}}Relationship", attributes)
    files[rels_part] = _to_xml_bytes(root)
    return rel_id


def _ensure_document_relationship(
    files: Dict[str, bytes],
    rel_type: str,
    target: str,
) -> None:
    root = _relationships_root(files, DOCUMENT_RELS_PART)
    for relationship in _relationship_elements(root):
        if relationship.attrib.get("Type") == rel_type and relationship.attrib.get("Target") == target:
            files[DOCUMENT_RELS_PART] = _to_xml_bytes(root)
            return
    rel_id = _next_relationship_id(root)
    etree.SubElement(
        root,
        f"{{{PKG_REL_NS}}}Relationship",
        {"Id": rel_id, "Type": rel_type, "Target": target},
    )
    files[DOCUMENT_RELS_PART] = _to_xml_bytes(root)


def _used_relationship_ids(elements: Iterable[etree._Element]) -> Set[str]:
    used: Set[str] = set()
    for element in elements:
        for node in element.iter():
            for attr_name, attr_value in node.attrib.items():
                if attr_name.startswith(f"{{{R_NS}}}") and attr_value:
                    used.add(str(attr_value))
    return used


def _copy_document_relationships_for_elements(
    *,
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    elements: List[etree._Element],
    document_index: int,
) -> None:
    used_ids = _used_relationship_ids(elements)
    if not used_ids:
        return
    source_relationships = _relationships_by_id(source_files, DOCUMENT_RELS_PART)
    relationship_id_map: Dict[str, str] = {}
    for source_id in sorted(used_ids):
        relationship = source_relationships.get(source_id)
        if relationship is None:
            continue
        new_id = _copy_relationship_target(
            source_files=source_files,
            destination_files=destination_files,
            content_types=content_types,
            source_part=DOCUMENT_PART,
            destination_part=DOCUMENT_PART,
            relationship=relationship,
            document_index=document_index,
        )
        relationship_id_map[source_id] = new_id
    _remap_relationship_ids(elements, relationship_id_map)


def _relationships_by_id(files: Dict[str, bytes], rels_part: str) -> Dict[str, etree._Element]:
    root = _optional_xml_root(files, rels_part)
    if root is None:
        return {}
    return {
        str(relationship.attrib.get("Id")): relationship
        for relationship in _relationship_elements(root)
        if relationship.attrib.get("Id")
    }


def _copy_relationship_target(
    *,
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    source_part: str,
    destination_part: str,
    relationship: etree._Element,
    document_index: int,
) -> str:
    rel_type = str(relationship.attrib.get("Type") or "")
    target = str(relationship.attrib.get("Target") or "")
    target_mode = relationship.attrib.get("TargetMode")
    if target_mode == "External":
        return _add_relationship(
            destination_files,
            destination_part,
            rel_type,
            target,
            target_mode=target_mode,
        )

    source_target_part = _resolve_target(source_part, target)
    destination_target_part = _unique_part_name(
        destination_files,
        _prefixed_part_name(source_target_part, document_index),
    )
    _copy_part_recursive(
        source_files=source_files,
        destination_files=destination_files,
        content_types=content_types,
        source_part=source_target_part,
        destination_part=destination_target_part,
        document_index=document_index,
    )
    return _add_relationship(
        destination_files,
        destination_part,
        rel_type,
        _relative_target(destination_part, destination_target_part),
    )


def _prefixed_part_name(part_name: str, document_index: int) -> str:
    directory = posixpath.dirname(part_name)
    filename = posixpath.basename(part_name)
    return posixpath.join(directory, f"merged{document_index}_{filename}") if directory else f"merged{document_index}_{filename}"


def _unique_part_name(files: Dict[str, bytes], desired: str) -> str:
    if desired not in files:
        return desired
    directory = posixpath.dirname(desired)
    filename = posixpath.basename(desired)
    stem, suffix = posixpath.splitext(filename)
    index = 1
    while True:
        candidate_filename = f"{stem}_{index}{suffix}"
        candidate = posixpath.join(directory, candidate_filename) if directory else candidate_filename
        if candidate not in files:
            return candidate
        index += 1


def _copy_part_recursive(
    *,
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    source_part: str,
    destination_part: str,
    document_index: int,
) -> None:
    if source_part not in source_files:
        return
    destination_files[destination_part] = source_files[source_part]
    _copy_content_type(content_types, source_files, source_part, destination_part)

    source_rels_part = _rels_part_for(source_part)
    if source_rels_part not in source_files:
        return

    source_rels_root = _xml_root(source_files, source_rels_part)
    destination_rels_root = copy.deepcopy(source_rels_root)
    for relationship in _relationship_elements(destination_rels_root):
        target_mode = relationship.attrib.get("TargetMode")
        if target_mode == "External":
            continue
        target = str(relationship.attrib.get("Target") or "")
        source_target_part = _resolve_target(source_part, target)
        destination_target_part = _unique_part_name(
            destination_files,
            _prefixed_part_name(source_target_part, document_index),
        )
        _copy_part_recursive(
            source_files=source_files,
            destination_files=destination_files,
            content_types=content_types,
            source_part=source_target_part,
            destination_part=destination_target_part,
            document_index=document_index,
        )
        relationship.attrib["Target"] = _relative_target(destination_part, destination_target_part)
    destination_files[_rels_part_for(destination_part)] = _to_xml_bytes(destination_rels_root)


def _copy_content_type(
    content_types: etree._Element,
    source_files: Dict[str, bytes],
    source_part: str,
    destination_part: str,
) -> None:
    source_content_types = _optional_xml_root(source_files, CONTENT_TYPES_PART)
    if source_content_types is None:
        return
    source_part_name = f"/{source_part}"
    destination_part_name = f"/{destination_part}"
    for override in source_content_types.findall(f"{{{CT_NS}}}Override"):
        if override.attrib.get("PartName") != source_part_name:
            continue
        _ensure_content_type_override(
            content_types,
            destination_part,
            str(override.attrib.get("ContentType") or ""),
        )
        return
    extension = destination_part.rsplit(".", 1)[-1] if "." in destination_part else ""
    if not extension:
        return
    for default in source_content_types.findall(f"{{{CT_NS}}}Default"):
        if default.attrib.get("Extension") == extension:
            _ensure_content_type_default(
                content_types,
                extension,
                str(default.attrib.get("ContentType") or ""),
            )
            return


def _ensure_content_type_override(content_types: etree._Element, part_name: str, content_type: str) -> None:
    if not content_type:
        return
    normalized_part = f"/{part_name.lstrip('/')}"
    for override in content_types.findall(f"{{{CT_NS}}}Override"):
        if override.attrib.get("PartName") == normalized_part:
            override.attrib["ContentType"] = content_type
            return
    etree.SubElement(
        content_types,
        f"{{{CT_NS}}}Override",
        {"PartName": normalized_part, "ContentType": content_type},
    )


def _ensure_content_type_default(content_types: etree._Element, extension: str, content_type: str) -> None:
    if not extension or not content_type:
        return
    for default in content_types.findall(f"{{{CT_NS}}}Default"):
        if default.attrib.get("Extension") == extension:
            return
    etree.SubElement(
        content_types,
        f"{{{CT_NS}}}Default",
        {"Extension": extension, "ContentType": content_type},
    )


def _remap_relationship_ids(elements: Iterable[etree._Element], relationship_id_map: Dict[str, str]) -> None:
    if not relationship_id_map:
        return
    for element in elements:
        for node in element.iter():
            for attr_name, attr_value in list(node.attrib.items()):
                if attr_name.startswith(f"{{{R_NS}}}") and attr_value in relationship_id_map:
                    node.attrib[attr_name] = relationship_id_map[str(attr_value)]


def _merge_styles(
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    body_elements: List[etree._Element],
) -> None:
    source_styles = _optional_xml_root(source_files, "word/styles.xml")
    if source_styles is None:
        return
    destination_styles = _optional_xml_root(destination_files, "word/styles.xml")
    if destination_styles is None:
        destination_styles = etree.Element(f"{W}styles", nsmap={"w": W_NS})

    destination_by_id = {
        str(style.attrib.get(W_STYLE_ID)): style
        for style in destination_styles.findall(f"{W}style")
        if style.attrib.get(W_STYLE_ID)
    }
    style_id_map: Dict[str, str] = {}
    pending_styles: List[etree._Element] = []
    for source_style in source_styles.findall(f"{W}style"):
        old_style_id = str(source_style.attrib.get(W_STYLE_ID) or "")
        if not old_style_id:
            continue
        destination_style = destination_by_id.get(old_style_id)
        if destination_style is not None and _canonical_xml(destination_style) == _canonical_xml(source_style):
            style_id_map[old_style_id] = old_style_id
            continue
        new_style_id = old_style_id if destination_style is None else _unique_style_id(destination_by_id, old_style_id)
        style_id_map[old_style_id] = new_style_id
        copied_style = copy.deepcopy(source_style)
        copied_style.attrib[W_STYLE_ID] = new_style_id
        destination_by_id[new_style_id] = copied_style
        pending_styles.append(copied_style)

    for copied_style in pending_styles:
        _remap_style_references([copied_style], style_id_map)
        destination_styles.append(copied_style)
    _remap_style_references(body_elements, style_id_map)
    destination_files["word/styles.xml"] = _to_xml_bytes(destination_styles)
    _ensure_content_type_override(content_types, "word/styles.xml", CT_STYLES)
    _ensure_document_relationship(destination_files, REL_TYPE_STYLES, "styles.xml")


def _canonical_xml(element: etree._Element) -> bytes:
    return etree.tostring(element, method="c14n")


def _unique_style_id(existing: Dict[str, etree._Element], base: str) -> str:
    index = 1
    while True:
        candidate = f"m{index}_{base}"
        if candidate not in existing:
            return candidate
        index += 1


def _remap_style_references(elements: Iterable[etree._Element], style_id_map: Dict[str, str]) -> None:
    if not style_id_map:
        return
    style_reference_tags = {"pStyle", "rStyle", "tblStyle", "basedOn", "next", "link"}
    for element in elements:
        for node in element.iter():
            if _local_name(node) not in style_reference_tags:
                continue
            value = node.attrib.get(W_VAL)
            if value in style_id_map:
                node.attrib[W_VAL] = style_id_map[str(value)]


def _merge_numbering(
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    body_elements: List[etree._Element],
) -> None:
    source_numbering = _optional_xml_root(source_files, "word/numbering.xml")
    if source_numbering is None:
        return
    destination_numbering = _optional_xml_root(destination_files, "word/numbering.xml")
    if destination_numbering is None:
        destination_numbering = etree.Element(f"{W}numbering", nsmap={"w": W_NS})

    max_abstract_id = _max_child_id(destination_numbering, "abstractNum", "abstractNumId")
    max_num_id = _max_child_id(destination_numbering, "num", "numId")
    abstract_map: Dict[str, str] = {}
    num_map: Dict[str, str] = {}

    for abstract in source_numbering.findall(f"{W}abstractNum"):
        old_id = str(abstract.attrib.get(f"{W}abstractNumId") or "")
        if not old_id:
            continue
        max_abstract_id += 1
        new_id = str(max_abstract_id)
        abstract_map[old_id] = new_id
        copied = copy.deepcopy(abstract)
        copied.attrib[f"{W}abstractNumId"] = new_id
        destination_numbering.append(copied)

    for num in source_numbering.findall(f"{W}num"):
        old_id = str(num.attrib.get(f"{W}numId") or "")
        if not old_id:
            continue
        max_num_id += 1
        new_id = str(max_num_id)
        num_map[old_id] = new_id
        copied = copy.deepcopy(num)
        copied.attrib[f"{W}numId"] = new_id
        abstract_ref = copied.find(f"{W}abstractNumId")
        if abstract_ref is not None:
            old_abstract = str(abstract_ref.attrib.get(W_VAL) or "")
            if old_abstract in abstract_map:
                abstract_ref.attrib[W_VAL] = abstract_map[old_abstract]
        destination_numbering.append(copied)

    _remap_num_ids(body_elements, num_map)
    destination_files["word/numbering.xml"] = _to_xml_bytes(destination_numbering)
    _ensure_content_type_override(content_types, "word/numbering.xml", CT_NUMBERING)
    _ensure_document_relationship(destination_files, REL_TYPE_NUMBERING, "numbering.xml")


def _max_child_id(root: etree._Element, child_name: str, attribute_name: str) -> int:
    max_id = 0
    for child in root.findall(f"{W}{child_name}"):
        value = str(child.attrib.get(f"{W}{attribute_name}") or "")
        if value.isdigit():
            max_id = max(max_id, int(value))
    return max_id


def _remap_num_ids(elements: Iterable[etree._Element], num_map: Dict[str, str]) -> None:
    if not num_map:
        return
    for element in elements:
        for node in element.iter(f"{W}numId"):
            value = node.attrib.get(W_VAL)
            if value in num_map:
                node.attrib[W_VAL] = num_map[str(value)]


def _merge_indexed_part(
    *,
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    body_elements: List[etree._Element],
    config: Dict[str, object],
    document_index: int,
) -> None:
    part_name = str(config["part"])
    source_root = _optional_xml_root(source_files, part_name)
    if source_root is None:
        return
    used_ids = _used_indexed_ids(body_elements, set(config["markers"]))  # type: ignore[arg-type]
    if not used_ids:
        return

    destination_root = _optional_xml_root(destination_files, part_name)
    if destination_root is None:
        destination_root = etree.Element(f"{W}{config['root']}", nsmap={"w": W_NS, "r": R_NS})
    next_id = _max_indexed_id(destination_root, str(config["item"])) + 1
    id_map: Dict[str, str] = {}
    copied_items: List[etree._Element] = []

    for item in source_root.findall(f"{W}{config['item']}"):
        old_id = str(item.attrib.get(W_ID) or "")
        if old_id not in used_ids:
            continue
        while str(next_id) in used_ids:
            next_id += 1
        new_id = str(next_id)
        next_id += 1
        id_map[old_id] = new_id
        copied_item = copy.deepcopy(item)
        copied_item.attrib[W_ID] = new_id
        copied_items.append(copied_item)

    if not copied_items:
        return

    _copy_relationships_for_part_elements(
        source_files=source_files,
        destination_files=destination_files,
        content_types=content_types,
        source_part=part_name,
        destination_part=part_name,
        elements=copied_items,
        document_index=document_index,
    )
    for item in copied_items:
        destination_root.append(item)
    _remap_indexed_ids(body_elements, set(config["markers"]), id_map)  # type: ignore[arg-type]
    destination_files[part_name] = _to_xml_bytes(destination_root)
    _ensure_content_type_override(content_types, part_name, str(config["content_type"]))
    _ensure_document_relationship(
        destination_files,
        str(config["rel_type"]),
        posixpath.basename(part_name),
    )


def _used_indexed_ids(elements: Iterable[etree._Element], marker_names: Set[str]) -> Set[str]:
    used: Set[str] = set()
    for element in elements:
        for node in element.iter():
            if _local_name(node) in marker_names and node.attrib.get(W_ID):
                used.add(str(node.attrib.get(W_ID)))
    return used


def _max_indexed_id(root: etree._Element, item_name: str) -> int:
    max_id = 0
    for item in root.findall(f"{W}{item_name}"):
        value = str(item.attrib.get(W_ID) or "")
        if value.lstrip("-").isdigit():
            max_id = max(max_id, int(value))
    return max_id


def _remap_indexed_ids(
    elements: Iterable[etree._Element],
    marker_names: Set[str],
    id_map: Dict[str, str],
) -> None:
    if not id_map:
        return
    for element in elements:
        for node in element.iter():
            if _local_name(node) in marker_names:
                value = node.attrib.get(W_ID)
                if value in id_map:
                    node.attrib[W_ID] = id_map[str(value)]


def _copy_relationships_for_part_elements(
    *,
    source_files: Dict[str, bytes],
    destination_files: Dict[str, bytes],
    content_types: etree._Element,
    source_part: str,
    destination_part: str,
    elements: List[etree._Element],
    document_index: int,
) -> None:
    used_ids = _used_relationship_ids(elements)
    if not used_ids:
        return
    source_relationships = _relationships_by_id(source_files, _rels_part_for(source_part))
    relationship_id_map: Dict[str, str] = {}
    for source_id in sorted(used_ids):
        relationship = source_relationships.get(source_id)
        if relationship is None:
            continue
        new_id = _copy_relationship_target(
            source_files=source_files,
            destination_files=destination_files,
            content_types=content_types,
            source_part=source_part,
            destination_part=destination_part,
            relationship=relationship,
            document_index=document_index,
        )
        relationship_id_map[source_id] = new_id
    _remap_relationship_ids(elements, relationship_id_map)
