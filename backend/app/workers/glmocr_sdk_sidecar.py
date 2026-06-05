"""Isolated GLM-OCR SDK sidecar.

The main app process does not import glmocr/transformers/torch.  This worker
uses the official GLM-OCR SDK pipeline when available and returns structured
page evidence.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List


def _ensure_private_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _to_builtin(value.to_dict())
        except Exception:
            pass
    return str(value)


def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    pages = payload.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return {"available": False, "warnings": ["glmocr_pages_missing"], "pages": []}

    from glmocr import GlmOcr

    base_url = str(payload.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
    host = base_url.replace("http://", "").replace("https://", "").split("/", 1)[0].split(":", 1)[0]
    port = 11434
    if ":" in base_url.replace("http://", "").replace("https://", "").split("/", 1)[0]:
        try:
            port = int(base_url.rsplit(":", 1)[-1])
        except Exception:
            port = 11434
    timeout = max(60, int(payload.get("timeout") or 900))
    max_tokens = max(1024, int(payload.get("max_tokens") or 4096))
    model = str(payload.get("model") or "glm-ocr:latest")
    layout_device = str(payload.get("layout_device") or "cpu")
    layout_batch_size = max(1, int(payload.get("layout_batch_size") or 1))

    dotted = {
        "pipeline.maas.enabled": False,
        "pipeline.ocr_api.api_host": host,
        "pipeline.ocr_api.api_port": port,
        "pipeline.ocr_api.api_path": "/api/generate",
        "pipeline.ocr_api.api_mode": "ollama_generate",
        "pipeline.ocr_api.model": model,
        "pipeline.ocr_api.request_timeout": timeout,
        "pipeline.ocr_api.connect_timeout": 10,
        "pipeline.ocr_api.retry_max_attempts": 0,
        "pipeline.max_workers": 1,
        "pipeline.page_loader.max_tokens": max_tokens,
        "pipeline.page_loader.temperature": 0.0,
        "pipeline.result_formatter.output_format": str(payload.get("result_format") or "both"),
        "pipeline.layout.device": layout_device,
        "pipeline.layout.batch_size": layout_batch_size,
    }
    layout_model_dir = str(payload.get("layout_model_dir") or "").strip()
    if layout_model_dir:
        dotted["pipeline.layout.model_dir"] = layout_model_dir
    page_results: List[Dict[str, Any]] = []
    with GlmOcr(mode="selfhosted", model=model, timeout=timeout, layout_device=layout_device, _dotted=dotted) as parser:
        for page in pages:
            page_no = int(page.get("page") or len(page_results) + 1)
            image_path = Path(str(page.get("image_path") or "")).expanduser()
            if not image_path.exists():
                page_results.append(
                    {"page": page_no, "ok": False, "warning": "glmocr_page_image_missing", "image_path": str(image_path)}
                )
                continue
            try:
                parsed = parser.parse(str(image_path), save_layout_visualization=False, preserve_order=True)
            except TypeError:
                parsed = parser.parse(str(image_path), save_layout_visualization=False)
            public = _to_builtin(parsed.to_dict())
            page_results.append(
                {
                    "page": page_no,
                    "ok": True,
                    "image_path": str(image_path),
                    "json_result": public.get("json_result"),
                    "markdown": public.get("markdown_result") or "",
                    "raw": public,
                }
            )
    return {
        "available": True,
        "engine": "glmocr_sdk_selfhosted_ollama",
        "model_settings": {
            "model": model,
            "layout_model_dir": layout_model_dir,
            "layout_device": layout_device,
            "layout_batch_size": layout_batch_size,
            "max_tokens": max_tokens,
            "result_format": str(payload.get("result_format") or "both"),
        },
        "pages": page_results,
        "warnings": [],
    }


def main() -> int:
    import sys

    if len(sys.argv) != 3:
        print("usage: python -m app.workers.glmocr_sdk_sidecar INPUT_JSON OUTPUT_JSON", file=sys.stderr)
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
                    "available": False,
                    "warnings": ["glmocr_sdk_failed"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "pages": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _ensure_private_file(output_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
