"""One-shot worker for PDF-to-Word OCR audit."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from app.core.runtime_security import ensure_private_file
from app.services.pdf_word_audit_v4 import PdfWordAuditV4Service


def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = str(payload.get("pdf_path") or "").strip()
    wps_docx_path = str(payload.get("wps_docx_path") or "").strip()
    output_dir = str(payload.get("output_dir") or "").strip()
    audit_id = str(payload.get("audit_id") or "").strip() or None
    progress_path = str(payload.get("progress_path") or "").strip() or None
    if not pdf_path:
        raise ValueError("pdf_path is required")
    if not wps_docx_path:
        raise ValueError("wps_docx_path is required")
    if not output_dir:
        raise ValueError("output_dir is required")
    service = PdfWordAuditV4Service(progress_path=progress_path)
    return {"audit": service.audit(pdf_path=pdf_path, wps_docx_path=wps_docx_path, output_dir=output_dir, audit_id=audit_id)}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.pdf_word_audit_worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = _run(payload)
        output_path.write_text(json.dumps({"ok": True, **result}, ensure_ascii=False), encoding="utf-8")
        ensure_private_file(output_path)
        return 0
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
