from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .common import normalize_text
from .models import DocxEvidenceUnit


class DocxFragmentMerger:
    """Build page-local merged DOCX fragments for safer mapping review."""

    VERSION = "docx_fragment_merger_v1"

    def __init__(self, *, max_units: int = 6, max_chars: int = 180) -> None:
        self.max_units = max(1, int(max_units or 1))
        self.max_chars = max(20, int(max_chars or 20))

    def build(self, *, docx_units: Sequence[DocxEvidenceUnit]) -> Dict[str, Any]:
        pages: Dict[int, List[DocxEvidenceUnit]] = {}
        for unit in docx_units:
            page_no = int(unit.estimated_page_no or 0)
            if page_no <= 0 or not normalize_text(unit.text):
                continue
            pages.setdefault(page_no, []).append(unit)
        fragments: List[Dict[str, Any]] = []
        for page_no, units in sorted(pages.items()):
            ordered = sorted(units, key=lambda item: int(item.order_index or 0))
            for index, unit in enumerate(ordered):
                texts = []
                unit_ids = []
                for other in ordered[index : index + self.max_units]:
                    candidate = " ".join([*texts, str(other.text or "").strip()]).strip()
                    if len(normalize_text(candidate)) > self.max_chars and texts:
                        break
                    texts.append(str(other.text or "").strip())
                    unit_ids.append(other.unit_id)
                text = " ".join(item for item in texts if item)
                if not text:
                    continue
                fragments.append(
                    {
                        "fragment_id": f"docx_fragment_{len(fragments)+1:04d}",
                        "page_no": page_no,
                        "unit_ids": unit_ids,
                        "anchor_unit_id": unit.unit_id,
                        "text": text[:500],
                        "normalized_text": normalize_text(text),
                        "unit_count": len(unit_ids),
                    }
                )
        return {
            "enabled": True,
            "version": self.VERSION,
            "fragment_count": len(fragments),
            "fragments": fragments,
        }

    def fragments_for_unit(self, *, payload: Dict[str, Any], unit_id: str) -> List[Dict[str, Any]]:
        rows = []
        for fragment in payload.get("fragments") or []:
            if unit_id in set(fragment.get("unit_ids") or []):
                rows.append(dict(fragment))
        rows.sort(key=lambda item: (int(item.get("unit_count") or 0), item.get("fragment_id", "")))
        return rows
