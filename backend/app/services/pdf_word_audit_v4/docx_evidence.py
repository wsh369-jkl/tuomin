from __future__ import annotations

import fnmatch
import math
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
from xml.etree import ElementTree as ET

from app.processors.docx_xml_utils import (
    count_paragraph_page_breaks,
    count_paragraph_section_page_breaks,
)

from .common import NAMESPACES, TEXT_PART_PATTERNS, _w, normalize_text
from .models import DocxEvidenceUnit


class DocxEvidenceExtractor:
    """Extract visible-ish DOCX content units directly from WordprocessingML."""

    def extract(self, *, docx_path: Path, estimated_page_count: int = 0) -> List[DocxEvidenceUnit]:
        units: List[DocxEvidenceUnit] = []
        with zipfile.ZipFile(docx_path) as archive:
            for part_name in self._list_text_parts(archive.namelist()):
                root = ET.fromstring(archive.read(part_name))
                units.extend(
                    self._collect_part_units(
                        root,
                        part_name=part_name,
                        start_order=len(units),
                        estimated_page_count=estimated_page_count,
                    )
                )
        return units

    def _list_text_parts(self, names: Iterable[str]) -> List[str]:
        result: List[str] = []
        for pattern in TEXT_PART_PATTERNS:
            for name in sorted(names):
                if name in result:
                    continue
                if fnmatch.fnmatch(name, pattern):
                    result.append(name)
        return result

    def _collect_part_units(
        self,
        root: ET.Element,
        *,
        part_name: str,
        start_order: int,
        estimated_page_count: int,
    ) -> List[DocxEvidenceUnit]:
        parent_map = {child: parent for parent in root.iter() for child in parent}
        table_index = {table: index for index, table in enumerate(root.iter(_w("tbl")), start=1)}
        paragraph_nodes = list(root.iterfind(".//w:p", NAMESPACES))
        explicit_pages = self._explicit_page_numbers_by_paragraph(paragraph_nodes, estimated_page_count=estimated_page_count)
        text_paragraphs: List[tuple[ET.Element, List[ET.Element], str]] = []
        for paragraph in paragraph_nodes:
            text_nodes = [node for node in self._iter_direct_paragraph_text_nodes(paragraph) if node.text]
            text = "".join(node.text or "" for node in text_nodes)
            if text.strip():
                text_paragraphs.append((paragraph, text_nodes, text))

        total_units = max(1, len(text_paragraphs))
        units: List[DocxEvidenceUnit] = []
        for local_index, (paragraph, _text_nodes, text) in enumerate(text_paragraphs):
            order_index = start_order + local_index
            container_type = self._container_type(part_name, paragraph, parent_map)
            page_no = explicit_pages.get(paragraph)
            if page_no is None:
                page_no = self._estimate_page_no(
                    part_name=part_name,
                    local_index=local_index,
                    total_units=total_units,
                    estimated_page_count=estimated_page_count,
                )
            table_meta = self._table_position(paragraph, parent_map=parent_map, table_index=table_index)
            unit_type = "table_cell" if container_type == "table_cell" else container_type
            flags: List[str] = []
            if container_type in {"header", "footer", "footnote", "endnote"}:
                flags.append("low_priority_container")
            if container_type == "textbox":
                flags.append("floating_text")
            units.append(
                DocxEvidenceUnit(
                    unit_id=f"{part_name}#{local_index}",
                    part_name=part_name,
                    unit_type=unit_type,
                    container_type=container_type,
                    text=text,
                    normalized_text=normalize_text(text),
                    order_index=order_index,
                    estimated_page_no=page_no,
                    table_index=table_meta.get("table_index"),
                    row_index=table_meta.get("row_index"),
                    col_index=table_meta.get("col_index"),
                    flags=flags,
                )
            )
        return units

    def _explicit_page_numbers_by_paragraph(
        self,
        paragraph_nodes: Sequence[ET.Element],
        *,
        estimated_page_count: int,
    ) -> Dict[ET.Element, int]:
        if estimated_page_count <= 0:
            return {}
        page_by_paragraph: Dict[ET.Element, int] = {}
        current_page = 1
        saw_break = False
        for paragraph in paragraph_nodes:
            page_by_paragraph[paragraph] = max(1, min(estimated_page_count, current_page))
            break_count = count_paragraph_page_breaks(paragraph)
            break_count += count_paragraph_section_page_breaks(paragraph)
            if break_count > 0:
                saw_break = True
                current_page = max(1, min(estimated_page_count, current_page + break_count))
        if not saw_break:
            return {}
        explicit_page_count = max(page_by_paragraph.values(), default=1)
        if not self._should_trust_explicit_page_numbers(
            explicit_page_count=explicit_page_count,
            estimated_page_count=estimated_page_count,
        ):
            return {}
        return page_by_paragraph

    def _should_trust_explicit_page_numbers(
        self,
        *,
        explicit_page_count: int,
        estimated_page_count: int,
    ) -> bool:
        if explicit_page_count <= 1 or estimated_page_count <= 1:
            return False
        if explicit_page_count > estimated_page_count:
            return False
        allowed_gap = max(2, int(math.ceil(estimated_page_count * 0.2)))
        if estimated_page_count - explicit_page_count > allowed_gap:
            return False
        return True

    def _iter_direct_paragraph_text_nodes(self, node: ET.Element) -> Iterator[ET.Element]:
        for child in node:
            if child.tag == _w("p"):
                continue
            if child.tag in {_w("t"), _w("delText")}:
                yield child
                continue
            yield from self._iter_direct_paragraph_text_nodes(child)

    def _container_type(self, part_name: str, node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> str:
        if part_name.startswith("word/header"):
            return "header"
        if part_name.startswith("word/footer"):
            return "footer"
        if part_name.endswith("footnotes.xml"):
            return "footnote"
        if part_name.endswith("endnotes.xml"):
            return "endnote"
        current: Optional[ET.Element] = node
        saw_table = False
        while current is not None:
            if current.tag == _w("txbxContent"):
                return "textbox"
            if current.tag == _w("tc"):
                saw_table = True
            current = parent_map.get(current)
        return "table_cell" if saw_table else "paragraph"

    def _table_position(
        self,
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
            if current.tag == _w("tc") and cell is None:
                cell = current
            elif current.tag == _w("tr") and row is None:
                row = current
            elif current.tag == _w("tbl"):
                table = current
                break
            current = parent_map.get(current)
        if cell is None or row is None or table is None:
            return {}
        rows = [item for item in table if item.tag == _w("tr")]
        cells = [item for item in row if item.tag == _w("tc")]
        return {
            "table_index": int(table_index.get(table) or 0),
            "row_index": rows.index(row) + 1 if row in rows else 0,
            "col_index": cells.index(cell) + 1 if cell in cells else 0,
        }

    def _estimate_page_no(
        self,
        *,
        part_name: str,
        local_index: int,
        total_units: int,
        estimated_page_count: int,
    ) -> Optional[int]:
        if estimated_page_count <= 0:
            return None
        if part_name != "word/document.xml":
            return None
        page = int(math.floor((local_index / max(1, total_units)) * estimated_page_count)) + 1
        return max(1, min(estimated_page_count, page))
