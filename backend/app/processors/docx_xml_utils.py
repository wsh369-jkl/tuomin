"""Utilities for reading and updating DOCX XML content, including tracked changes."""

from __future__ import annotations

import fnmatch
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple
from xml.etree import ElementTree as ET


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
NAMESPACES = {"w": WORD_NAMESPACE}

PARAGRAPH_TAG = f"{{{WORD_NAMESPACE}}}p"
TEXT_TAGS = {
    f"{{{WORD_NAMESPACE}}}t",
    f"{{{WORD_NAMESPACE}}}delText",
}
TAB_TAG = f"{{{WORD_NAMESPACE}}}tab"
BREAK_TAGS = {
    f"{{{WORD_NAMESPACE}}}br",
    f"{{{WORD_NAMESPACE}}}cr",
}
TRACKED_INSERTION_TAG = f"{{{WORD_NAMESPACE}}}ins"
TRACKED_DELETION_TAG = f"{{{WORD_NAMESPACE}}}del"
XML_SPACE_ATTR = f"{{{XML_NAMESPACE}}}space"

DOCX_TEXT_PART_PATTERNS = (
    "word/document.xml",
    "word/header*.xml",
    "word/footer*.xml",
    "word/footnotes.xml",
    "word/endnotes.xml",
)

_REGISTERED_NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "docPropsVTypes": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "dcterms": "http://purl.org/dc/terms/",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
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


def extract_docx_text(file_path: str | Path) -> Tuple[str, Dict[str, int | bool]]:
    """Extract visible text from a DOCX package, including tracked insertions/deletions."""
    text_parts: List[str] = []
    tracked_insertions = 0
    tracked_deletions = 0
    tracked_deleted_text_nodes = 0

    with zipfile.ZipFile(file_path) as archive:
        part_names = list_docx_text_parts(archive.namelist())
        for part_name in part_names:
            root = ET.fromstring(archive.read(part_name))
            tracked_insertions += len(root.findall(".//w:ins", NAMESPACES))
            tracked_deletions += len(root.findall(".//w:del", NAMESPACES))
            tracked_deleted_text_nodes += len(root.findall(".//w:delText", NAMESPACES))

            for paragraph in root.iterfind(".//w:p", NAMESPACES):
                paragraph_text = extract_paragraph_text(paragraph)
                if paragraph_text.strip():
                    text_parts.append(paragraph_text)

    tracked_changes_detected = bool(
        tracked_insertions or tracked_deletions or tracked_deleted_text_nodes
    )
    return "\n".join(text_parts), {
        "tracked_changes_detected": tracked_changes_detected,
        "tracked_insertions": tracked_insertions,
        "tracked_deletions": tracked_deletions,
        "tracked_deleted_text_nodes": tracked_deleted_text_nodes,
    }


def extract_paragraph_text(paragraph: ET.Element) -> str:
    """Collect text content from a paragraph while skipping nested paragraphs."""
    return "".join(_iter_paragraph_text_fragments(paragraph))


def _iter_paragraph_text_fragments(node: ET.Element) -> Iterator[str]:
    for child in node:
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
        target_parts = set(list_docx_text_parts(archive.namelist()))

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
                    updated = replace_text_in_xml_part(data, normalized)
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
) -> bytes | None:
    """Apply replacements to all paragraphs in a single DOCX XML part."""
    root = ET.fromstring(xml_bytes)
    modified = False

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
    """Apply replacements across text-bearing XML nodes inside a paragraph."""
    text_nodes = [node for node in _iter_paragraph_text_nodes(paragraph) if node.text]
    if not text_nodes:
        return False

    fragment_texts = [node.text or "" for node in text_nodes]
    updated_texts = apply_replacements_to_fragments(fragment_texts, replacements)
    if updated_texts == fragment_texts:
        return False

    for node, updated_text in zip(text_nodes, updated_texts):
        node.text = updated_text
        _sync_space_preserve(node, updated_text)

    return True


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
        updated_fragments[start_fragment] = start_text[:start_char] + replacement
        updated_fragments[end_fragment] = end_text[end_char + 1 :]

        for index in range(start_fragment + 1, end_fragment):
            updated_fragments[index] = ""

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


def _sync_space_preserve(node: ET.Element, text: str) -> None:
    if text[:1].isspace() or text[-1:].isspace():
        node.set(XML_SPACE_ATTR, "preserve")
        return

    node.attrib.pop(XML_SPACE_ATTR, None)
