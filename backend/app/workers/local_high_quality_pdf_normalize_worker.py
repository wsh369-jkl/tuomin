"""One-shot local_high_quality PDF normalization worker."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from app.core.runtime_security import ensure_private_file
from app.services.local_high_quality_pdf_service import (
    LocalHighQualityPdfNormalizationError,
    LocalHighQualityPdfNormalizationService,
)


def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    file_path = str(payload.get("file_path") or "").strip()
    output_dir = str(payload.get("output_dir") or "").strip() or None
    progress_path = str(payload.get("progress_path") or "").strip() or None
    if not file_path:
        raise ValueError("file_path is required")
    service = LocalHighQualityPdfNormalizationService(progress_path=progress_path)
    normalized = service.normalize_pdf(file_path, output_dir=output_dir)
    return {"normalized": normalized}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.local_high_quality_pdf_normalize_worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = _run(payload)
        output_path.write_text(json.dumps({"ok": True, **result}, ensure_ascii=False), encoding="utf-8")
        ensure_private_file(output_path)
        return 0
    except LocalHighQualityPdfNormalizationError as exc:
        error_payload = {
            "ok": False,
            "error_code": exc.code,
            "error": str(exc),
            "metadata": exc.metadata,
            "traceback": traceback.format_exc(),
        }
        try:
            output_path.write_text(json.dumps(error_payload, ensure_ascii=False), encoding="utf-8")
            ensure_private_file(output_path)
        except Exception:
            pass
        return 1
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
