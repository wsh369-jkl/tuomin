"""PaddleOCR-VL one-shot sidecar for local_high_quality PDF fallback pages."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import traceback
import time
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from pathlib import Path
from typing import Any


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
        return {
            str(key): _to_builtin(item)
            for key, item in value.items()
            if str(key) not in {"output_img", "input_img"}
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _to_builtin(value.to_dict())
        except Exception:
            pass
    public: dict[str, Any] = {}
    for name in ("markdown", "res", "data", "page_id", "input_path"):
        if hasattr(value, name):
            try:
                public[name] = _to_builtin(getattr(value, name))
            except Exception:
                pass
    return public or str(value)


TEXT_KEYS = {"text", "content", "markdown", "block_content", "rec_text", "html", "parsing_res_list"}


def _clean_text_value(value: str, *, key_hint: str) -> list[str]:
    stripped = str(value or "").strip()
    if not stripped:
        return []
    if key_hint == "parsing_res_list":
        matches = re.findall(r"content:\s*(.*?)(?:\n#{3,}|$)", stripped, flags=re.S)
        cleaned = [match.strip() for match in matches if match.strip()]
        if cleaned:
            return cleaned
    return [stripped]


def _model_name_from_inference_config(model_dir: Path) -> str:
    config_path = model_dir / "inference.yml"
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("model_name:"):
                return stripped.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return "PaddleOCR-VL-1.5"


def _server_ready(server_url: str) -> bool:
    url = str(server_url or "").rstrip("/")
    if not url:
        return False
    try:
        request = Request(f"{url}/v1/models", method="GET")
        with urlopen(request, timeout=1.5) as response:
            if not (200 <= int(getattr(response, "status", 200)) < 300):
                return False
            payload = json.loads(response.read().decode("utf-8") or "{}")
            return bool(payload.get("data"))
    except Exception:
        return False
    return False


def _server_host_port(server_url: str) -> tuple[str, int]:
    parsed = urlparse(str(server_url or "http://127.0.0.1:8111/"))
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 8111)
    return host, port


def _default_mlx_server_executable() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / "mlx_vlm_venv" / "bin" / "mlx_vlm.server"


def _start_mlx_server(payload: dict[str, Any], *, server_url: str, image_path: Path) -> subprocess.Popen | None:
    if str(payload.get("auto_start_mlx_server", "true")).strip().lower() in {"0", "false", "no", "off"}:
        return None
    executable = Path(str(payload.get("mlx_server_executable") or _default_mlx_server_executable())).expanduser()
    if not executable.exists():
        return None
    host, port = _server_host_port(server_url)
    log_path = Path(str(payload.get("mlx_server_log_path") or image_path.parent / "mlx_vlm_server.log")).expanduser()
    model_dir = Path(str(payload.get("model_dir") or "")).expanduser()
    if not model_dir.exists():
        return None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(executable),
                "--model",
                str(model_dir),
                "--trust-remote-code",
                "--host",
                host,
                "--port",
                str(port),
                "--prefill-step-size",
                str(int(payload.get("mlx_prefill_step_size") or 256)),
                "--max-kv-size",
                str(int(payload.get("mlx_max_kv_size") or 8192)),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return process
    except Exception:
        return None


def _stop_started_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _wait_server_ready(server_url: str, process: subprocess.Popen | None, *, timeout: float = 20.0) -> bool:
    deadline = time.time() + max(1.0, timeout)
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if _server_ready(server_url):
            return True
        time.sleep(0.35)
    return _server_ready(server_url)


def _collect_text(value: Any, *, key_hint: str = "") -> list[str]:
    parts: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and key_hint in TEXT_KEYS:
            parts.extend(_clean_text_value(stripped, key_hint=key_hint))
    elif isinstance(value, dict):
        for key, item in value.items():
            parts.extend(_collect_text(item, key_hint=str(key)))
    elif isinstance(value, list):
        for item in value:
            parts.extend(_collect_text(item, key_hint=key_hint))
    return parts


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(payload.get("image_path") or "")).expanduser()
    model_dir = Path(str(payload.get("model_dir") or "")).expanduser()
    requested_backend = str(payload.get("vl_rec_backend") or "auto").strip() or "auto"
    backend = requested_backend
    server_url = str(payload.get("vl_rec_server_url") or "http://localhost:8111/").strip()
    api_model_name = str(payload.get("vl_rec_api_model_name") or "").strip() or "PaddlePaddle/PaddleOCR-VL-1.5"
    native_model_name = str(payload.get("vl_rec_model_name") or "").strip() or _model_name_from_inference_config(model_dir)
    warnings: list[str] = []
    started_mlx_server: subprocess.Popen | None = None
    if backend == "auto":
        if _server_ready(server_url):
            backend = "mlx-vlm-server"
        else:
            started_mlx_server = _start_mlx_server(payload, server_url=server_url, image_path=image_path)
            if started_mlx_server is not None and _wait_server_ready(server_url, started_mlx_server, timeout=float(payload.get("mlx_server_start_timeout") or 20.0)):
                backend = "mlx-vlm-server"
                warnings.append("vl_mlx_server_autostarted")
            else:
                _stop_started_server(started_mlx_server)
                started_mlx_server = None
                backend = "native"
                warnings.append("vl_mlx_unavailable")
    elif backend == "mlx-vlm-server" and not _server_ready(server_url):
        started_mlx_server = _start_mlx_server(payload, server_url=server_url, image_path=image_path)
        if started_mlx_server is None or not _wait_server_ready(server_url, started_mlx_server, timeout=float(payload.get("mlx_server_start_timeout") or 20.0)):
            _stop_started_server(started_mlx_server)
            return {"ok": False, "warning": "paddleocr_vl_missing", "error": "mlx_vlm_server_unavailable"}
    if not image_path.exists():
        return {"ok": False, "warning": "paddleocr_vl_missing", "error": f"image_not_found:{image_path}"}
    if backend != "mlx-vlm-server" and not model_dir.exists():
        return {"ok": False, "warning": "paddleocr_vl_missing", "error": f"model_dir_not_found:{model_dir}"}
    threads = max(1, int(payload.get("threads") or 2))

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

    from paddleocr import PaddleOCRVL

    init_kwargs: dict[str, Any] = {
        "pipeline_version": "v1.5",
        "vl_rec_backend": backend,
        "use_doc_orientation_classify": bool(payload.get("use_doc_orientation_classify", True)),
        "use_doc_unwarping": bool(payload.get("use_doc_unwarping", False)),
        "use_layout_detection": bool(payload.get("use_layout_detection", True)),
        "use_chart_recognition": bool(payload.get("use_chart_recognition", False)),
        "use_seal_recognition": bool(payload.get("use_seal_recognition", False)),
        "use_ocr_for_image_block": bool(payload.get("use_ocr_for_image_block", True)),
        "format_block_content": True,
        "merge_layout_blocks": True,
        "use_queues": False,
        "device": "cpu",
        "cpu_threads": threads,
        "enable_mkldnn": False,
    }
    for key in (
        "layout_detection_model_dir",
        "doc_orientation_classify_model_dir",
        "doc_unwarping_model_dir",
    ):
        configured = str(payload.get(key) or "").strip()
        if configured and Path(configured).expanduser().exists():
            init_kwargs[key] = str(Path(configured).expanduser())
    if backend == "mlx-vlm-server":
        init_kwargs.update(
            {
                "vl_rec_server_url": server_url,
                "vl_rec_api_model_name": api_model_name,
            }
        )
    else:
        init_kwargs.update(
            {
                "vl_rec_model_name": native_model_name,
                "vl_rec_model_dir": str(model_dir),
            }
        )

    pipeline = PaddleOCRVL(**init_kwargs)
    predict_kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": bool(payload.get("use_doc_orientation_classify", True)),
        "use_doc_unwarping": bool(payload.get("use_doc_unwarping", False)),
        "use_layout_detection": bool(payload.get("use_layout_detection", True)),
        "use_chart_recognition": bool(payload.get("use_chart_recognition", False)),
        "use_seal_recognition": bool(payload.get("use_seal_recognition", False)),
        "use_ocr_for_image_block": bool(payload.get("use_ocr_for_image_block", True)),
        "format_block_content": True,
        "merge_layout_blocks": True,
        "min_pixels": int(payload.get("min_pixels") or 112896),
        "max_pixels": int(payload.get("max_pixels") or 1605632),
        "max_new_tokens": int(payload.get("max_new_tokens") or 8192),
        "temperature": 0.0,
        "top_p": float(payload.get("top_p") or 0.001),
        "repetition_penalty": float(payload.get("repetition_penalty") or 1.05),
    }
    try:
        result = pipeline.predict(str(image_path), **predict_kwargs)
    finally:
        try:
            pipeline.close()
        except Exception:
            pass
        _stop_started_server(started_mlx_server)

    raw = _to_builtin(result)
    text = "\n".join(dict.fromkeys(_collect_text(raw))).strip()
    model_settings = {
        "vl_rec_backend": backend,
        "vl_rec_backend_requested": requested_backend,
        "vl_rec_server_url": server_url if backend == "mlx-vlm-server" else "",
        "vl_rec_api_model_name": api_model_name if backend == "mlx-vlm-server" else "",
        "vl_rec_model_name": native_model_name if backend != "mlx-vlm-server" else "",
        "use_layout_detection": predict_kwargs["use_layout_detection"],
        "use_doc_orientation_classify": predict_kwargs["use_doc_orientation_classify"],
        "use_doc_unwarping": predict_kwargs["use_doc_unwarping"],
        "use_ocr_for_image_block": predict_kwargs["use_ocr_for_image_block"],
        "use_seal_recognition": predict_kwargs["use_seal_recognition"],
        "use_chart_recognition": predict_kwargs["use_chart_recognition"],
        "max_pixels": predict_kwargs["max_pixels"],
        "max_new_tokens": predict_kwargs["max_new_tokens"],
        "threads": threads,
        "mlx_server_autostarted": bool(started_mlx_server is not None),
        "layout_detection_model_dir": str(init_kwargs.get("layout_detection_model_dir") or ""),
        "doc_orientation_classify_model_dir": str(init_kwargs.get("doc_orientation_classify_model_dir") or ""),
        "doc_unwarping_model_dir": str(init_kwargs.get("doc_unwarping_model_dir") or ""),
    }
    if not text:
        return {
            "ok": True,
            "text": "",
            "quality": "failed",
            "lines": [],
            "raw": raw,
            "model_settings": model_settings,
            "warnings": [*warnings, "paddleocr_vl_empty"],
        }
    return {
        "ok": True,
        "text": text,
        "quality": "medium",
        "lines": [{"text": line, "bbox": [0.08, 0.0, 0.92, 0.02], "confidence": 0.82} for line in text.splitlines() if line.strip()],
        "raw": raw,
        "model_settings": model_settings,
        "warnings": [*warnings, "paddleocr_vl_local_used"],
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.local_paddleocr_vl_sidecar INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = _run(payload)
        output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        _ensure_private_file(output_path)
        return 0 if result.get("ok") else 1
    except Exception as exc:
        output_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "warning": "paddleocr_vl_missing",
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
