"""Optional PP-Structure recovery_to_doc sidecar.

PaddleOCR 3.x no longer ships the old ``paddleocr.ppstructure.recovery`` module
in the currently installed package. This sidecar gives the main worker a stable
probe and a forward-compatible execution point if that module is installed in a
separate OCR environment later.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def _ensure_private_file(path: Path) -> None:
    """Keep this sidecar runnable in the isolated OCR venv."""
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _run(payload: dict) -> dict:
    file_path = str(payload.get("file_path") or "").strip()
    if file_path:
        file_path = str(Path(file_path).expanduser().resolve())
    output_docx_path = Path(str(payload.get("output_docx_path") or "").strip()).expanduser().resolve()
    if not file_path or not output_docx_path:
        raise ValueError("file_path and output_docx_path are required")

    try:
        import paddleocr.ppstructure.recovery.recovery_to_doc  # noqa: F401
    except Exception as exc:
        return {
            "ok": True,
            "available": False,
            "warning": "ppstructure_docx_recovery_missing",
            "error": f"recovery_to_doc_unavailable:{type(exc).__name__}:{exc}",
        }

    output_docx_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ppstructure-recovery-") as temp_dir:
        command = [
            str(Path(sys.executable).parent / "paddleocr"),
            "--image_dir",
            file_path,
            "--type=structure",
            "--recovery=true",
            "--lang=ch",
            "--use_gpu=false",
            "--show_log=false",
            "--output",
            temp_dir,
            "--table=true" if bool(payload.get("table_enabled")) else "--table=false",
            "--formula=true" if bool(payload.get("formula_enabled")) else "--formula=false",
        ]
        page_num = int(payload.get("page_num") or 0)
        if page_num > 0:
            command.append(f"--page_num={page_num}")
        executable = Path(command[0])
        if not executable.exists():
            return {
                "ok": True,
                "available": False,
                "warning": "ppstructure_docx_recovery_missing",
                "error": f"paddleocr_cli_missing:{executable}",
            }
        env = os.environ.copy()
        env.setdefault("FLAGS_enable_pir_api", "0")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        timeout = max(30, int(payload.get("timeout") or 600))
        try:
            result = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": True,
                "available": False,
                "warning": "ppstructure_docx_recovery_failed",
                "error": f"recovery_to_doc_timeout:{timeout}:{str(exc)[-600:]}",
            }
        if result.returncode != 0:
            return {
                "ok": True,
                "available": False,
                "warning": "ppstructure_docx_recovery_failed",
                "error": (result.stderr or result.stdout or "paddleocr_recovery_failed")[-1200:],
            }
        docx_files = sorted(Path(temp_dir).rglob("*.docx"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not docx_files:
            return {
                "ok": True,
                "available": False,
                "warning": "ppstructure_docx_recovery_failed",
                "error": "recovery_to_doc_no_docx_output",
            }
        shutil.copyfile(docx_files[0], output_docx_path)
        _ensure_private_file(output_docx_path)
        return {
            "ok": True,
            "available": True,
            "output_docx_path": str(output_docx_path),
            "warning": "ppstructure_docx_recovery_used",
            "stdout_tail": (result.stdout or "")[-1000:],
        }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.local_pdf_recovery_sidecar INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = _run(payload)
        output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        _ensure_private_file(output_path)
        return 0
    except Exception as exc:
        output_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "available": False,
                    "warning": "ppstructure_docx_recovery_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _ensure_private_file(output_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
