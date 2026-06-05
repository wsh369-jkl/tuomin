from __future__ import annotations

import difflib
import fnmatch
import math
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from xml.etree import ElementTree as ET

from app.core.runtime_security import ensure_private_directory, ensure_private_file
from app.processors.docx_xml_utils import (
    count_paragraph_page_breaks,
    count_paragraph_section_page_breaks,
)

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
XML_NS = "http://www.w3.org/XML/1998/namespace"

NAMESPACES = {
    "w": WORD_NS,
    "r": REL_NS,
    "rel": PKG_REL_NS,
    "ct": CONTENT_TYPES_NS,
}

for _prefix, _uri in {
    "w": WORD_NS,
    "r": REL_NS,
    "rel": PKG_REL_NS,
    "ct": CONTENT_TYPES_NS,
    "xml": XML_NS,
}.items():
    ET.register_namespace(_prefix, _uri)

TEXT_PART_PATTERNS = (
    "word/document.xml",
    "word/header*.xml",
    "word/footer*.xml",
    "word/footnotes.xml",
    "word/endnotes.xml",
)

COMMENTS_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"
COMMENTS_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
XML_SPACE_ATTR = f"{{{XML_NS}}}space"
ORGANIZATION_SUFFIXES = (
    "律师事务所",
    "人民法院",
    "有限公司",
    "有限责任公司",
    "股份有限公司",
    "服务所",
    "委员会",
    "管理有限公司",
)

GENERIC_CANONICAL_TABLE_HEADERS = (
    "序号",
    "编号",
    "编码",
    "名称",
    "姓名",
    "项目",
    "内容",
    "日期",
    "时间",
    "金额",
    "数量",
    "单价",
    "单价/m²",
    "总价",
    "小计",
    "合计",
    "总计",
    "税额",
    "税率",
    "单位",
    "规格",
    "型号",
    "账户",
    "账号",
    "备注",
    "摘要",
    "说明",
    "电话",
    "地址",
    "联系人",
    "部门",
    "比例",
    "面积",
    "楼层",
    "房号",
)

GENERIC_TABLE_HEADER_MARKERS = (
    "名称",
    "项目",
    "内容",
    "编号",
    "编码",
    "序号",
    "日期",
    "时间",
    "金额",
    "数量",
    "单价",
    "总价",
    "小计",
    "合计",
    "总计",
    "税额",
    "税率",
    "单位",
    "规格",
    "型号",
    "账户",
    "账号",
    "备注",
    "摘要",
    "说明",
    "名称",
    "电话",
    "地址",
    "联系人",
    "部门",
    "比例",
    "面积",
)

GENERIC_TABLE_TITLE_MARKERS = (
    "表",
    "明细",
    "清单",
    "列表",
    "汇总",
    "统计",
    "台账",
    "目录",
)

GENERIC_TABLE_FRAGMENT_MARKERS = (
    "金额",
    "数量",
    "单价",
    "合计",
    "总计",
    "税额",
    "税率",
    "编号",
    "序号",
    "日期",
    "时间",
    "账户",
    "账号",
    "单位",
    "规格",
    "型号",
    "备注",
)

GENERIC_HIGH_VALUE_FIELD_MARKERS = (
    "身份证",
    "身份号码",
    "证件号码",
    "住址",
    "地址",
    "联系电话",
    "手机号",
    "电话",
    "账号",
    "账户",
    "开户行",
    "金额",
    "价款",
    "总价",
    "合计",
    "总计",
    "日期",
    "时间",
    "编号",
    "编码",
    "案号",
)


def _w(tag: str) -> str:
    return f"{{{WORD_NS}}}{tag}"


def _rel(tag: str) -> str:
    return f"{{{PKG_REL_NS}}}{tag}"


def _ct(tag: str) -> str:
    return f"{{{CONTENT_TYPES_NS}}}{tag}"


def normalize_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\s+", "", value)
    value = value.replace("，", ",").replace("。", ".").replace("：", ":").replace("；", ";")
    value = value.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    return value.strip()


def table_text_artifact_replacement(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if not compact or len(compact) > 32:
        return ""
    replacement = _generic_table_header_replacement(compact)
    if replacement and replacement != compact:
        return replacement
    if compact[:1] in {"!", "！", "|", "丨"}:
        stripped = compact[1:].strip()
        normalized = _generic_table_header_replacement(stripped)
        if normalized and normalized == stripped:
            return stripped
    return ""


def looks_like_table_title(text: Any) -> bool:
    compact = normalize_text(text)
    if not compact or len(compact) < 2 or len(compact) > 64:
        return False
    marker_hits = sum(1 for marker in GENERIC_TABLE_TITLE_MARKERS if marker in compact)
    header_hits = sum(1 for marker in GENERIC_TABLE_HEADER_MARKERS if marker in compact)
    digit_count = sum(ch.isdigit() for ch in compact)
    if marker_hits >= 2:
        return True
    if "表" in compact and (header_hits >= 1 or digit_count >= 2):
        return True
    if marker_hits >= 1 and header_hits >= 1 and len(compact) <= 32:
        return True
    if marker_hits >= 1 and re.search(r"(?:\d{4}[-./年]\d{1,2}|\d+(?:\.\d+)?(?:元|%))", compact):
        return True
    return False


def looks_like_document_title(text: Any) -> bool:
    value = "".join(str(text or "").split())
    if not value or len(value) < 2 or len(value) > 48:
        return False
    if looks_like_table_title(value):
        return True
    if any(value.endswith(suffix) for suffix in ORGANIZATION_SUFFIXES):
        return False
    if any(value.endswith(suffix) for suffix in ("书", "合同", "协议", "证明", "申请书", "声明", "通知", "函")):
        if re.search(r"[，,。；;:：]", value):
            return False
        return True
    return False


def looks_like_organization_name(text: Any) -> bool:
    compact = normalize_text(text)
    if not compact or len(compact) < 4 or len(compact) > 40:
        return False
    return any(compact.endswith(suffix) for suffix in ORGANIZATION_SUFFIXES)


def has_high_value_field_content(text: Any) -> bool:
    value = str(text or "")
    compact = normalize_text(value)
    if not compact:
        return False
    if any(marker in compact for marker in GENERIC_HIGH_VALUE_FIELD_MARKERS):
        return True
    return bool(
        re.search(r"(?<!\d)\d{15,18}[\dXx]?(?!\d)", compact)
        or re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", compact)
        or re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?", value)
        or re.search(r"[¥￥]?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:元|万元|人民币|RMB|%)", value, flags=re.IGNORECASE)
        or re.search(r"(?:编号|编码|案号)\s*[:：]?\s*[\w\-（）()第号]{2,32}", value)
    )


def looks_like_table_header(text: Any) -> bool:
    normalized = _normalize_generic_table_header_text(text)
    return _plausible_generic_table_header(normalized)


def looks_like_paragraphized_table_fragment(text: Any) -> bool:
    """Detect table/header debris that was extracted as a paragraph.

    The check is intentionally shape-based. It relies on short header labels,
    compact numeric columns, and title-like table fragments rather than on any
    document-family-specific vocabulary.
    """
    raw = " ".join(str(text or "").split())
    compact = normalize_text(raw)
    if not compact:
        return False

    if looks_like_table_title(compact):
        return True

    marker_hits = {marker for marker in GENERIC_TABLE_FRAGMENT_MARKERS if marker in compact}
    if len(marker_hits) >= 2 and len(compact) <= 72:
        return True
    if marker_hits and len(compact) <= 36:
        return True

    digit_count = sum(ch.isdigit() for ch in compact)
    cjk_count = sum("\u4e00" <= ch <= "\u9fff" for ch in compact)
    ascii_count = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in compact)
    tokens = [item for item in re.split(r"[\s|丨/\\,:：;；，。()\[\]【】]+", raw) if item]
    short_tokens = [item for item in tokens if len(normalize_text(item)) <= 12]
    numeric_tokens = [
        item
        for item in short_tokens
        if re.search(r"\d", item)
        or re.search(r"(?:\d{4}[-./年]\d{1,2}|\d+(?:\.\d+)?(?:元|%))", item)
    ]
    header_tokens = [
        item
        for item in short_tokens
        if any(marker in normalize_text(item) for marker in GENERIC_TABLE_HEADER_MARKERS)
    ]
    if len(short_tokens) >= 4 and (len(numeric_tokens) >= 2 or len(header_tokens) >= 2):
        return True
    if len(tokens) >= 5 and len(short_tokens) >= max(4, len(tokens) - 1) and digit_count >= 4:
        return True
    if len(compact) <= 28 and ascii_count >= 1 and digit_count >= 1 and cjk_count <= 2:
        return True
    if re.fullmatch(r"[\dA-Za-z./:：\\-]+", compact or "") and digit_count >= 2:
        return True
    if digit_count >= 6 and len(compact) <= 48 and (
        bool(re.search(r"(?:\d{4}[-./年]\d{1,2}|\d+\s*至\s*\d+|\d+\s*-\s*\d+)", compact))
        or bool(re.search(r"\d+(?:\.\d+)?\s*(?:元|%|万|m²)", raw))
    ):
        return True
    return False


def _generic_table_header_replacement(text: str) -> str:
    key = _table_header_key(text)
    if not key:
        return ""
    if key in {"单元", "单元号"}:
        return ""
    category_target = _generic_table_header_category_target(key)
    if category_target:
        return category_target
    normalized = _normalize_generic_table_header_text(text)
    if normalized != text and _plausible_generic_table_header(normalized):
        return normalized
    best_target = ""
    best_score = 0.0
    for target in GENERIC_CANONICAL_TABLE_HEADERS:
        target_key = _table_header_key(target)
        score = max(difflib.SequenceMatcher(None, key, target_key).ratio(), _edit_similarity(key, target_key))
        if _one_edit_away(key, target_key):
            score = max(score, 0.84)
        if score > best_score:
            best_score = score
            best_target = target
    if best_target and best_score >= 0.82 and _header_category_compatible(key, _table_header_key(best_target)):
        return best_target
    return ""


def _generic_table_header_category_target(key: str) -> str:
    if not key:
        return ""
    if _unit_price_header_key(key):
        return "单价/m²"
    if len(key) <= 6 and key.endswith(("姓名", "名")) and any(token in key for token in ("姓", "名")):
        return "姓名"
    if len(key) <= 6 and key.endswith(("面积", "积")) and any(token in key for token in ("面", "积")):
        return "面积"
    if len(key) <= 4 and key.endswith("号") and any(token in key for token in ("楼", "层")):
        return "楼层"
    if len(key) <= 4 and key.endswith("号") and "房" in key:
        return "房号"
    if len(key) <= 4 and key.startswith("编") and any(token in key for token in ("编", "码", "号")):
        return "编号"
    if len(key) <= 3 and key.endswith("注"):
        return "备注"
    return ""


def _table_header_key(text: str) -> str:
    key = "".join(str(text or "").split()).lower()
    key = key.replace("／", "/").replace("㎡", "m²").replace("m2", "m²")
    key = re.sub(r"^[!！|丨:：;；,.，。]+", "", key)
    key = re.sub(r"m[?？3iiln]+", "m²", key)
    key = key.replace("/m²²", "/m²").replace("im²", "/m²")
    return key


def _normalize_generic_table_header_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\s+", "", value)
    value = value.replace("／", "/").replace("㎡", "m²").replace("m2", "m²")
    value = re.sub(r"^[!！|丨:：;；,.，。]+", "", value)
    value = re.sub(r"m[?？3iiln]+", "m²", value, flags=re.IGNORECASE)
    value = value.replace("/m²²", "/m²").replace("im²", "/m²")
    return value


def _plausible_generic_table_header(text: str) -> bool:
    compact = normalize_text(text)
    if not compact or len(compact) > 18:
        return False
    digit_count = sum(ch.isdigit() for ch in compact)
    if digit_count > max(3, len(compact) // 2):
        return False
    if any(marker in compact for marker in GENERIC_TABLE_HEADER_MARKERS):
        return True
    if _unit_price_header_key(_table_header_key(compact)):
        return True
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z/_\-]{2,18}", compact))


def _header_category_compatible(source_key: str, target_key: str) -> bool:
    if not source_key or not target_key:
        return False
    if target_key == "单价/m²":
        return _unit_price_header_key(source_key)
    source_markers = {marker for marker in GENERIC_TABLE_HEADER_MARKERS if marker in source_key}
    target_markers = {marker for marker in GENERIC_TABLE_HEADER_MARKERS if marker in target_key}
    if source_markers and target_markers:
        return bool(source_markers & target_markers)
    if target_key in {"备注", "摘要", "说明"}:
        return source_key.endswith(("注", "要", "明")) or any(token in source_key for token in ("备注", "摘要", "说明"))
    if target_key in {"楼层", "房号"}:
        return any(token in source_key for token in ("楼", "层", "房", "号"))
    if target_key == "姓名":
        return any(token in source_key for token in ("名", "姓", "人"))
    if target_key == "面积":
        return any(token in source_key for token in ("面", "积"))
    return True


def _unit_price_header_key(key: str) -> bool:
    if "m²" not in key and "/m" not in key:
        return False
    if len(key) > 10:
        return False
    return any(marker in key for marker in ("价", "单", "金额", "费用")) or _edit_similarity(key, "单价/m²") >= 0.72


def _edit_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    distance = _levenshtein(left, right)
    return max(0.0, 1.0 - distance / max(len(left), len(right), 1))


def _one_edit_away(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    return _levenshtein(left, right, max_distance=1) <= 1


def _levenshtein(left: str, right: str, *, max_distance: Optional[int] = None) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (0 if left_char == right_char else 1)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_min = min(row_min, value)
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return difflib.SequenceMatcher(None, left, right).ratio()


@dataclass
class CorrectionCandidate:
    id: str
    wps_unit_id: str
    page_no: Optional[int]
    old_text: str
    new_text: str
    action: str
    confidence: float
    alignment_score: float
    reason: str
    comment_text: str
    sensitive_low_priority: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "wps_unit_id": self.wps_unit_id,
            "page_no": self.page_no,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "action": self.action,
            "confidence": round(float(self.confidence), 4),
            "alignment_score": round(float(self.alignment_score), 4),
            "reason": self.reason,
            "comment_text": self.comment_text,
            "sensitive_low_priority": bool(self.sensitive_low_priority),
        }


@dataclass
class WpsTextUnit:
    id: str
    part_name: str
    unit_index: int
    container_type: str
    text: str
    normalized_text: str
    order_index: int
    page_no: Optional[int] = None


@dataclass
class _XmlTextUnit:
    public: WpsTextUnit
    paragraph: ET.Element
    text_nodes: List[ET.Element]


class WpsDocxStructureParser:
    """Extract text units needed for writing Word/WPS comments."""

    def collect_xml_units(
        self,
        root: ET.Element,
        *,
        part_name: str,
        start_order: int = 0,
        estimated_page_count: int = 0,
    ) -> List[_XmlTextUnit]:
        parent_map = {child: parent for parent in root.iter() for child in parent}
        paragraph_nodes = list(root.iterfind(".//w:p", NAMESPACES))
        explicit_page_by_paragraph = self._explicit_page_numbers_by_paragraph(
            paragraph_nodes,
            estimated_page_count=estimated_page_count,
        )
        text_paragraphs: List[tuple[ET.Element, List[ET.Element], str]] = []
        for paragraph in paragraph_nodes:
            text_nodes = [node for node in self._iter_direct_paragraph_text_nodes(paragraph) if node.text]
            text = "".join(node.text or "" for node in text_nodes)
            if text.strip():
                text_paragraphs.append((paragraph, text_nodes, text))

        total_units = max(1, len(text_paragraphs))
        units: List[_XmlTextUnit] = []
        for local_index, (paragraph, text_nodes, text) in enumerate(text_paragraphs):
            order_index = start_order + local_index
            container_type = self._container_type(part_name, paragraph, parent_map)
            page_no = explicit_page_by_paragraph.get(paragraph)
            if page_no is None:
                page_no = self._estimate_page_no(
                    part_name=part_name,
                    local_index=local_index,
                    total_units=total_units,
                    estimated_page_count=estimated_page_count,
                )
            public = WpsTextUnit(
                id=f"{part_name}#{local_index}",
                part_name=part_name,
                unit_index=local_index,
                container_type=container_type,
                text=text,
                normalized_text=normalize_text(text),
                order_index=order_index,
                page_no=page_no,
            )
            units.append(_XmlTextUnit(public=public, paragraph=paragraph, text_nodes=text_nodes))
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
        while current is not None:
            if current.tag == _w("txbxContent"):
                return "textbox"
            if current.tag == _w("tc"):
                return "table_cell"
            current = parent_map.get(current)
        return "paragraph"

    def _estimate_page_no(
        self,
        *,
        part_name: str,
        local_index: int,
        total_units: int,
        estimated_page_count: int,
    ) -> Optional[int]:
        if estimated_page_count <= 0 or part_name != "word/document.xml":
            return None
        page = int(math.floor((local_index / max(1, total_units)) * estimated_page_count)) + 1
        return max(1, min(estimated_page_count, page))


class DocxAuditCommentWriter:
    """Apply review candidates and add Word/WPS comments via OOXML."""

    def write(
        self,
        *,
        template_docx_path: str | Path,
        output_docx_path: str | Path,
        corrections: Sequence[CorrectionCandidate],
    ) -> Dict[str, Any]:
        source = Path(template_docx_path)
        output = Path(output_docx_path)
        ensure_private_directory(output.parent)

        if not corrections:
            shutil.copyfile(source, output)
            ensure_private_file(output)
            return {"comment_count": 0, "auto_corrected": 0, "manual_review": 0}

        temp_fd, temp_name = tempfile.mkstemp(suffix=".docx", dir=str(output.parent))
        os.close(temp_fd)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(source, "r") as archive:
                files = {item.filename: archive.read(item.filename) for item in archive.infolist()}
                infos = {item.filename: item for item in archive.infolist()}

            comments_root, next_comment_id = self._load_or_create_comments(files)
            corrections_by_unit = {item.wps_unit_id: item for item in corrections}
            modified_parts: Dict[str, bytes] = {}
            parser = WpsDocxStructureParser()

            for part_name in self._list_text_parts(files.keys()):
                root = ET.fromstring(files[part_name])
                xml_units = parser.collect_xml_units(root, part_name=part_name)
                part_modified = False
                for xml_unit in xml_units:
                    correction = corrections_by_unit.get(xml_unit.public.id)
                    if correction is None:
                        continue
                    if correction.action in {"replace", "delete"}:
                        self._set_unit_text(xml_unit, correction.new_text)
                        part_modified = True
                    self._append_comment_marker(xml_unit.paragraph, next_comment_id)
                    self._append_comment(comments_root, next_comment_id, correction.comment_text)
                    next_comment_id += 1
                    part_modified = True
                if part_modified:
                    modified_parts[part_name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

            files.update(modified_parts)
            files["word/comments.xml"] = ET.tostring(comments_root, encoding="utf-8", xml_declaration=True)
            files["[Content_Types].xml"] = self._ensure_comments_content_type(files)
            files["word/_rels/document.xml.rels"] = self._ensure_comments_relationship(files)

            with zipfile.ZipFile(temp_path, "w") as target:
                for filename, data in files.items():
                    info = infos.get(filename)
                    if info is not None:
                        target.writestr(info, data)
                    else:
                        target.writestr(filename, data)
            temp_path.replace(output)
            ensure_private_file(output)
            return {
                "comment_count": len(corrections),
                "auto_corrected": sum(1 for item in corrections if item.action in {"replace", "delete"}),
                "manual_review": sum(1 for item in corrections if item.action == "review"),
            }
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _list_text_parts(self, names: Iterable[str]) -> List[str]:
        result: List[str] = []
        for pattern in TEXT_PART_PATTERNS:
            for name in sorted(names):
                if name in result:
                    continue
                if fnmatch.fnmatch(name, pattern):
                    result.append(name)
        return result

    def _set_unit_text(self, unit: _XmlTextUnit, replacement: str) -> None:
        if not unit.text_nodes:
            return
        first = unit.text_nodes[0]
        first.text = replacement
        self._sync_space(first, replacement)
        for node in unit.text_nodes[1:]:
            node.text = ""
            node.attrib.pop(XML_SPACE_ATTR, None)

    def _sync_space(self, node: ET.Element, text: str) -> None:
        if text != text.strip():
            node.set(XML_SPACE_ATTR, "preserve")
        else:
            node.attrib.pop(XML_SPACE_ATTR, None)

    def _append_comment_marker(self, paragraph: ET.Element, comment_id: int) -> None:
        start = ET.Element(_w("commentRangeStart"), {_w("id"): str(comment_id)})
        end = ET.Element(_w("commentRangeEnd"), {_w("id"): str(comment_id)})
        ref_run = ET.Element(_w("r"))
        ET.SubElement(ref_run, _w("commentReference"), {_w("id"): str(comment_id)})
        paragraph.insert(0, start)
        paragraph.append(end)
        paragraph.append(ref_run)

    def _load_or_create_comments(self, files: Dict[str, bytes]) -> tuple[ET.Element, int]:
        if "word/comments.xml" in files:
            root = ET.fromstring(files["word/comments.xml"])
            ids: List[int] = []
            for comment in root.findall("w:comment", NAMESPACES):
                raw = comment.attrib.get(_w("id"))
                if raw is not None and raw.isdigit():
                    ids.append(int(raw))
            return root, (max(ids) + 1 if ids else 0)
        return ET.Element(_w("comments")), 0

    def _append_comment(self, root: ET.Element, comment_id: int, text: str) -> None:
        comment = ET.SubElement(
            root,
            _w("comment"),
            {
                _w("id"): str(comment_id),
                _w("author"): "PDF转Word核查",
                _w("date"): datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            },
        )
        for line in str(text or "").splitlines() or [""]:
            paragraph = ET.SubElement(comment, _w("p"))
            run = ET.SubElement(paragraph, _w("r"))
            text_node = ET.SubElement(run, _w("t"))
            text_node.text = line
            self._sync_space(text_node, line)

    def _ensure_comments_content_type(self, files: Dict[str, bytes]) -> bytes:
        root = ET.fromstring(files["[Content_Types].xml"])
        exists = any(
            override.attrib.get("PartName") == "/word/comments.xml"
            for override in root.findall("ct:Override", NAMESPACES)
        )
        if not exists:
            ET.SubElement(
                root,
                _ct("Override"),
                {"PartName": "/word/comments.xml", "ContentType": COMMENTS_CONTENT_TYPE},
            )
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def _ensure_comments_relationship(self, files: Dict[str, bytes]) -> bytes:
        rels_name = "word/_rels/document.xml.rels"
        if rels_name in files:
            root = ET.fromstring(files[rels_name])
        else:
            root = ET.Element(_rel("Relationships"))
        exists = any(rel.attrib.get("Type") == COMMENTS_REL_TYPE for rel in root.findall("rel:Relationship", NAMESPACES))
        if not exists:
            used_ids = {rel.attrib.get("Id", "") for rel in root.findall("rel:Relationship", NAMESPACES)}
            index = 1
            rel_id = "rIdComments"
            while rel_id in used_ids:
                index += 1
                rel_id = f"rIdComments{index}"
            ET.SubElement(
                root,
                _rel("Relationship"),
                {"Id": rel_id, "Type": COMMENTS_REL_TYPE, "Target": "comments.xml"},
            )
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)
