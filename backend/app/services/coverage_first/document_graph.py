"""Build a passive visible-text document graph from parser structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DocumentUnit:
    unit_id: str
    text: str
    start: int
    end: int
    story_type: str
    xml_part_path: str
    container_type: str
    unit_type: str
    rewrite_policy: str
    fragment_count: int
    metadata: dict[str, Any]

    @property
    def exactly_rewritable(self) -> bool:
        return self.rewrite_policy == "exact" and self.fragment_count > 0


@dataclass(frozen=True)
class DocumentGraph:
    enabled: bool
    units: list[DocumentUnit]
    summary: dict[str, Any]


def build_document_graph(source_structure: dict[str, Any] | None) -> DocumentGraph:
    units = list(_iter_units(source_structure))
    if not units:
        return DocumentGraph(
            enabled=False,
            units=[],
            summary={
                "enabled": False,
                "unit_count": 0,
                "exact_rewrite_unit_count": 0,
                "review_required_unit_count": 0,
                "unknown_part_unit_count": 0,
            },
        )

    deduped: list[DocumentUnit] = []
    seen: set[tuple[str, int, int, str]] = set()
    for unit in units:
        key = (unit.unit_id, unit.start, unit.end, unit.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)

    container_counts: dict[str, int] = {}
    story_counts: dict[str, int] = {}
    part_counts: dict[str, int] = {}
    exact_count = 0
    review_required_count = 0
    unknown_count = 0
    for unit in deduped:
        container_counts[unit.container_type] = container_counts.get(unit.container_type, 0) + 1
        story_counts[unit.story_type] = story_counts.get(unit.story_type, 0) + 1
        part_counts[unit.xml_part_path] = part_counts.get(unit.xml_part_path, 0) + 1
        if unit.exactly_rewritable:
            exact_count += 1
        else:
            review_required_count += 1
        if unit.container_type == "xml_text" or "unknown" in unit.xml_part_path.lower():
            unknown_count += 1

    return DocumentGraph(
        enabled=True,
        units=deduped,
        summary={
            "enabled": True,
            "unit_count": len(deduped),
            "exact_rewrite_unit_count": exact_count,
            "review_required_unit_count": review_required_count,
            "unknown_part_unit_count": unknown_count,
            "container_counts": dict(sorted(container_counts.items())),
            "story_counts": dict(sorted(story_counts.items())),
            "part_counts": dict(sorted(part_counts.items())),
        },
    )


def _iter_units(source_structure: dict[str, Any] | None) -> Iterable[DocumentUnit]:
    if not isinstance(source_structure, dict):
        return []
    raw_units = source_structure.get("docx_text_units")
    if isinstance(raw_units, list):
        return [_coerce_unit(unit, index) for index, unit in enumerate(raw_units) if isinstance(unit, dict)]
    units: list[DocumentUnit] = []
    pages = source_structure.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("units"), list):
                continue
            for unit in page["units"]:
                if isinstance(unit, dict):
                    units.append(_coerce_unit(unit, len(units)))
    return units


def _coerce_unit(unit: dict[str, Any], index: int) -> DocumentUnit:
    text = str(unit.get("text") or "")
    start = _coerce_int(unit.get("start"), -1)
    end = _coerce_int(unit.get("end"), start + len(text) if start >= 0 else -1)
    part_name = str(unit.get("part_name") or unit.get("xml_part_path") or "")
    container_type = str(unit.get("container_type") or unit.get("unit_type") or "unknown").strip() or "unknown"
    story_type = str(unit.get("story_type") or _story_type_from_part(part_name, container_type)).strip() or "body"
    unit_id = str(unit.get("unit_id") or f"{part_name or 'docx'}#{index + 1}").strip()
    return DocumentUnit(
        unit_id=unit_id,
        text=text,
        start=start,
        end=end,
        story_type=story_type,
        xml_part_path=part_name or "unknown",
        container_type=container_type,
        unit_type=str(unit.get("unit_type") or container_type),
        rewrite_policy=str(unit.get("rewrite_policy") or "exact"),
        fragment_count=max(0, _coerce_int(unit.get("fragment_count"), len(unit.get("fragments") or []))),
        metadata=dict(unit),
    )


def _story_type_from_part(part_name: str, container_type: str) -> str:
    lowered = str(part_name or "").lower()
    if "header" in lowered or container_type == "header":
        return "header"
    if "footer" in lowered or container_type == "footer":
        return "footer"
    if "footnote" in lowered or container_type == "footnote":
        return "footnote"
    if "endnote" in lowered or container_type == "endnote":
        return "endnote"
    if "comment" in lowered or container_type == "comment":
        return "comment"
    if "textbox" in container_type:
        return "textbox"
    return "body"


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
