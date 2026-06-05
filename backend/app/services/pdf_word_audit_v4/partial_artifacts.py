from __future__ import annotations

import json
import time
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from app.core.runtime_security import ensure_private_directory, ensure_private_file

T = TypeVar("T")


def write_partial_review_payload(
    *,
    work_dir: Path,
    filename: str,
    version: str,
    reviews: Sequence[Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort incremental artifact writer for expensive serial stages."""

    try:
        raw_dir = Path(work_dir) / "evidence" / "raw"
        ensure_private_directory(raw_dir)
        safe_filename = str(filename or "").strip() or "reviews.partial.json"
        if not safe_filename.endswith(".json"):
            safe_filename = f"{safe_filename}.json"
        target = raw_dir / safe_filename
        payload: Dict[str, Any] = {
            "enabled": True,
            "version": str(version or "partial_reviews_v1"),
            "partial": True,
            "status": "running",
            "review_count": len(reviews),
            "updated_at": time.time(),
            "reviews": [_to_payload(item) for item in reviews],
        }
        if extra:
            payload.update(dict(extra))
        tmp = target.with_name(f"{target.name}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        ensure_private_file(tmp)
        tmp.replace(target)
        ensure_private_file(target)
    except Exception:
        return


def load_partial_review_payload(
    *,
    work_dir: Path,
    filename: str,
    review_type: Type[T],
    expected_resume_key: str = "",
) -> Tuple[List[T], Dict[str, Any]]:
    """Best-effort partial artifact reader with a strict resume key.

    Partial artifacts are expensive-stage checkpoints, not durable cache across
    code revisions.  The resume key prevents old model outputs from being reused
    after prompt/filter semantics change.
    """

    try:
        raw_dir = Path(work_dir) / "evidence" / "raw"
        safe_filename = str(filename or "").strip() or "reviews.partial.json"
        if not safe_filename.endswith(".json"):
            safe_filename = f"{safe_filename}.json"
        target = raw_dir / safe_filename
        if not target.exists():
            return [], {}
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return [], {}
        if expected_resume_key and str(payload.get("resume_key") or "") != expected_resume_key:
            return [], {}
        raw_reviews = payload.get("reviews")
        if not isinstance(raw_reviews, list):
            return [], payload
        rows: List[T] = []
        for item in raw_reviews:
            row = _coerce_review(review_type=review_type, value=item)
            if row is not None:
                rows.append(row)
        return rows, payload
    except Exception:
        return [], {}


def _to_payload(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, dict):
        return dict(value)
    return value


def _coerce_review(*, review_type: Type[T], value: Any) -> Optional[T]:
    if isinstance(value, review_type):
        return value
    if not isinstance(value, dict) or not is_dataclass(review_type):
        return None
    review_fields = fields(review_type)
    kwargs: Dict[str, Any] = {}
    for item in review_fields:
        if item.name in value:
            kwargs[item.name] = value[item.name]
        elif item.default is MISSING and item.default_factory is MISSING:
            return None
    try:
        return review_type(**kwargs)
    except Exception:
        return None
