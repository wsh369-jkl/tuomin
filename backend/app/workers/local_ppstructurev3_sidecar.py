"""Optional PP-StructureV3 layout sidecar for local_high_quality PDF pages."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


PIXEL_ARRAY_KEYS = {
    "input_img",
    "output_img",
    "src_img",
    "orig_img",
    "resized_img",
    "rot_img",
    "image",
    "img",
    "doc_img",
    "table_img",
}
MAX_BUILTIN_LIST_ITEMS = 500


def _ensure_private_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _array_summary(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    size = getattr(value, "size", None)
    return {
        "omitted": True,
        "reason": "pixel_array_removed_from_ppstructure_payload",
        "type": type(value).__name__,
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else "",
        "size": int(size) if isinstance(size, (int, float)) else None,
    }


def _looks_like_large_matrix(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 64:
        return False
    first = value[0] if value else None
    if isinstance(first, (list, tuple)):
        return True
    return len(value) > MAX_BUILTIN_LIST_ITEMS


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in PIXEL_ARRAY_KEYS:
                result[key_text] = _array_summary(item)
            else:
                result[key_text] = _to_builtin(item)
        return result
    if isinstance(value, (list, tuple, set)):
        if _looks_like_large_matrix(value):
            return _array_summary(value)
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        size = getattr(value, "size", None)
        if isinstance(size, (int, float)) and int(size) > MAX_BUILTIN_LIST_ITEMS:
            return _array_summary(value)
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _to_builtin(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "json"):
        try:
            return _to_builtin(getattr(value, "json"))
        except Exception:
            pass
    public = {}
    for name in ("markdown", "res", "data", "page_id", "input_path"):
        if hasattr(value, name):
            try:
                public[name] = _to_builtin(getattr(value, name))
            except Exception:
                pass
    if public:
        return public
    return str(value)


def _validate_paddlex_config(config_path: Path) -> tuple[bool, str]:
    if not config_path.exists():
        return False, f"ppstructurev3_config_not_found:{config_path}"
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception as exc:
        return False, f"ppstructurev3_config_invalid:{type(exc).__name__}:{exc}"

    missing: list[str] = []

    disabled_prefixes: list[str] = []
    if not bool(payload.get("use_doc_preprocessor")):
        disabled_prefixes.append("SubPipelines.DocPreprocessor")
    if not bool(payload.get("use_chart_recognition")):
        disabled_prefixes.append("SubModules.ChartRecognition")
    if not bool(payload.get("use_region_detection")):
        disabled_prefixes.append("SubModules.RegionDetection")
    if not bool(payload.get("use_table_recognition")):
        disabled_prefixes.append("SubPipelines.TableRecognition")
    if not bool(payload.get("use_seal_recognition")):
        disabled_prefixes.append("SubPipelines.SealRecognition")
    if not bool(payload.get("use_formula_recognition")):
        disabled_prefixes.append("SubPipelines.FormulaRecognition")

    def walk(node: Any, prefix: str = "") -> None:
        if prefix and any(prefix == disabled or prefix.startswith(f"{disabled}.") for disabled in disabled_prefixes):
            return
        if isinstance(node, dict):
            if (
                prefix.endswith("TableRecognition.SubPipelines.GeneralOCR")
                and not bool(payload.get("SubPipelines", {})
                    .get("TableRecognition", {})
                    .get("use_ocr_model"))
            ):
                return
            model_name = node.get("model_name")
            if model_name and "model_dir" in node:
                model_dir = str(node.get("model_dir") or "").strip()
                if not model_dir or not Path(model_dir).expanduser().exists():
                    missing.append(f"{prefix}.model_dir:{model_name}")
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{prefix}[{index}]")

    walk(payload)
    if missing:
        return False, "ppstructurev3_model_dir_missing:" + ",".join(missing[:12])
    return True, ""


def _model_name(model_dir: str | None) -> str | None:
    value = str(model_dir or "").strip()
    if not value:
        return None
    return Path(value).expanduser().name


def _pipeline_kwargs(payload: dict, *, config_path: Path | None, threads: int) -> dict:
    base = {
        "device": str(payload.get("device") or "cpu"),
        "cpu_threads": threads,
        "enable_mkldnn": bool(payload.get("enable_mkldnn", False)),
        "use_doc_orientation_classify": bool(payload.get("use_doc_orientation_classify", True)),
        "use_doc_unwarping": bool(payload.get("use_doc_unwarping", False)),
        "use_textline_orientation": bool(payload.get("use_textline_orientation", True)),
        "use_table_recognition": bool(payload.get("use_table_recognition", True)),
        "use_region_detection": bool(payload.get("use_region_detection", True)),
        "use_seal_recognition": bool(payload.get("use_seal_recognition", False)),
        "use_formula_recognition": bool(payload.get("use_formula_recognition", False)),
        "use_chart_recognition": bool(payload.get("use_chart_recognition", False)),
        "format_block_content": bool(payload.get("format_block_content", True)),
    }
    if config_path is not None:
        return {"paddlex_config": str(config_path), **base}

    model_fields = {
        "layout_detection_model_dir": "layout_detection_model_name",
        "doc_orientation_classify_model_dir": "doc_orientation_classify_model_name",
        "doc_unwarping_model_dir": "doc_unwarping_model_name",
        "text_detection_model_dir": "text_detection_model_name",
        "textline_orientation_model_dir": "textline_orientation_model_name",
        "text_recognition_model_dir": "text_recognition_model_name",
        "table_classification_model_dir": "table_classification_model_name",
        "wired_table_structure_recognition_model_dir": "wired_table_structure_recognition_model_name",
        "wireless_table_structure_recognition_model_dir": "wireless_table_structure_recognition_model_name",
        "wired_table_cells_detection_model_dir": "wired_table_cells_detection_model_name",
        "wireless_table_cells_detection_model_dir": "wireless_table_cells_detection_model_name",
        "table_orientation_classify_model_dir": "table_orientation_classify_model_name",
        "seal_text_detection_model_dir": "seal_text_detection_model_name",
        "seal_text_recognition_model_dir": "seal_text_recognition_model_name",
        "formula_recognition_model_dir": "formula_recognition_model_name",
    }
    kwargs = dict(base)
    for dir_key, name_key in model_fields.items():
        model_dir = str(payload.get(dir_key) or "").strip()
        if not model_dir:
            continue
        expanded = str(Path(model_dir).expanduser())
        if not Path(expanded).exists():
            continue
        kwargs[dir_key] = expanded
        kwargs[name_key] = str(payload.get(name_key) or _model_name(expanded) or "")
    for key in (
        "text_det_limit_side_len",
        "text_det_limit_type",
        "text_det_thresh",
        "text_det_box_thresh",
        "text_det_unclip_ratio",
        "text_recognition_batch_size",
        "text_rec_score_thresh",
        "layout_threshold",
        "layout_nms",
        "layout_unclip_ratio",
    ):
        if payload.get(key) is not None:
            kwargs[key] = payload.get(key)
    return kwargs


def _run_command(payload: dict) -> dict:
    image_path = str(payload.get("image_path") or "").strip()
    command = str(payload.get("command") or "").strip()
    timeout = max(30, int(payload.get("timeout") or 900))
    if not image_path or not command:
        return {"ok": True, "available": False, "warning": "ppstructurev3_layout_missing", "error": "command_or_image_missing"}
    formatted = command.format(image=image_path)
    result = subprocess.run(
        shlex.split(formatted),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return {
            "ok": True,
            "available": False,
            "warning": "ppstructurev3_layout_failed",
            "error": (result.stderr or result.stdout)[-1200:],
        }
    raw = str(result.stdout or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"markdown": raw}
    return {"ok": True, "available": True, "engine": "ppstructurev3_command", "raw": parsed}


def _run_command_batch(payload: dict) -> dict:
    pages = payload.get("pages") or []
    page_results: list[dict] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_payload = {**payload, "image_path": page.get("image_path")}
        page_result = _run_command(page_payload)
        page_result["page"] = int(page.get("page") or len(page_results) + 1)
        page_results.append(page_result)
    return {"ok": True, "available": bool(page_results), "engine": "ppstructurev3_command_batch", "pages": page_results}


def _run_python(payload: dict) -> dict:
    image_path = str(payload.get("image_path") or "").strip()
    config_path = Path(str(payload.get("config_path") or "").strip()).expanduser()
    timeout = max(30, int(payload.get("timeout") or 900))
    threads = max(1, int(payload.get("threads") or 2))
    if not image_path:
        return {"ok": True, "available": False, "warning": "ppstructurev3_layout_missing", "error": "image_path_missing"}
    has_direct_models = bool(str(payload.get("layout_detection_model_dir") or "").strip())
    valid, reason = _validate_paddlex_config(config_path) if str(config_path) != "." else (False, "ppstructurev3_config_not_set")
    if not valid and not has_direct_models:
        return {"ok": True, "available": False, "warning": "ppstructurev3_layout_missing", "error": reason}

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(**_pipeline_kwargs(payload, config_path=config_path if valid else None, threads=threads))
    try:
        result = pipeline.predict(
            image_path,
            use_doc_orientation_classify=bool(payload.get("use_doc_orientation_classify", True)),
            use_doc_unwarping=bool(payload.get("use_doc_unwarping", False)),
            use_textline_orientation=bool(payload.get("use_textline_orientation", True)),
            use_seal_recognition=bool(payload.get("use_seal_recognition", False)),
            use_table_recognition=bool(payload.get("use_table_recognition", True)),
            use_formula_recognition=bool(payload.get("use_formula_recognition", False)),
            use_chart_recognition=bool(payload.get("use_chart_recognition", False)),
            use_region_detection=bool(payload.get("use_region_detection", True)),
            format_block_content=bool(payload.get("format_block_content", True)),
        )
    finally:
        try:
            pipeline.close()
        except Exception:
            pass
    return {
        "ok": True,
        "available": True,
        "engine": "ppstructurev3_python",
        "model_settings": _model_settings(payload),
        "raw": _to_builtin(result),
    }


def _run_python_batch(payload: dict) -> dict:
    pages = payload.get("pages") or []
    config_path = Path(str(payload.get("config_path") or "").strip()).expanduser()
    threads = max(1, int(payload.get("threads") or 2))
    if not isinstance(pages, list) or not pages:
        return {"ok": True, "available": False, "warning": "ppstructurev3_layout_missing", "error": "pages_missing", "pages": []}
    has_direct_models = bool(str(payload.get("layout_detection_model_dir") or "").strip())
    valid, reason = _validate_paddlex_config(config_path) if str(config_path) != "." else (False, "ppstructurev3_config_not_set")
    if not valid and not has_direct_models:
        return {
            "ok": True,
            "available": False,
            "warning": "ppstructurev3_layout_missing",
            "error": reason,
            "pages": [],
        }

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(**_pipeline_kwargs(payload, config_path=config_path if valid else None, threads=threads))
    page_results: list[dict] = []
    try:
        for index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            image_path = str(page.get("image_path") or "").strip()
            page_number = int(page.get("page") or index + 1)
            if not image_path:
                page_results.append(
                    {
                        "ok": True,
                        "available": False,
                        "page": page_number,
                        "warning": "ppstructurev3_layout_missing",
                        "error": "image_path_missing",
                    }
                )
                continue
            try:
                result = pipeline.predict(
                    image_path,
                    use_doc_orientation_classify=bool(payload.get("use_doc_orientation_classify", True)),
                    use_doc_unwarping=bool(payload.get("use_doc_unwarping", False)),
                    use_textline_orientation=bool(payload.get("use_textline_orientation", True)),
                    use_seal_recognition=bool(payload.get("use_seal_recognition", False)),
                    use_table_recognition=bool(payload.get("use_table_recognition", True)),
                    use_formula_recognition=bool(payload.get("use_formula_recognition", False)),
                    use_chart_recognition=bool(payload.get("use_chart_recognition", False)),
                    use_region_detection=bool(payload.get("use_region_detection", True)),
                    format_block_content=bool(payload.get("format_block_content", True)),
                )
                page_results.append(
                    {
                        "ok": True,
                        "available": True,
                        "page": page_number,
                        "engine": "ppstructurev3_python",
                        "model_settings": _model_settings(payload),
                        "raw": _to_builtin(result),
                    }
                )
            except Exception as exc:
                page_results.append(
                    {
                        "ok": True,
                        "available": False,
                        "page": page_number,
                        "warning": "ppstructurev3_layout_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        try:
            pipeline.close()
        except Exception:
            pass
    return {
        "ok": True,
        "available": True,
        "engine": "ppstructurev3_python_batch",
        "model_settings": _model_settings(payload),
        "pages": page_results,
    }


def _model_settings(payload: dict) -> dict:
    return {
        "use_doc_orientation_classify": bool(payload.get("use_doc_orientation_classify", True)),
        "use_doc_unwarping": bool(payload.get("use_doc_unwarping", False)),
        "use_textline_orientation": bool(payload.get("use_textline_orientation", True)),
        "use_table_recognition": bool(payload.get("use_table_recognition", True)),
        "use_region_detection": bool(payload.get("use_region_detection", True)),
        "use_seal_recognition": bool(payload.get("use_seal_recognition", False)),
        "use_formula_recognition": bool(payload.get("use_formula_recognition", False)),
        "use_chart_recognition": bool(payload.get("use_chart_recognition", False)),
        "threads": max(1, int(payload.get("threads") or 2)),
        "layout_detection_model_dir": str(payload.get("layout_detection_model_dir") or ""),
        "text_detection_model_dir": str(payload.get("text_detection_model_dir") or ""),
        "text_recognition_model_dir": str(payload.get("text_recognition_model_dir") or ""),
        "table_classification_model_dir": str(payload.get("table_classification_model_dir") or ""),
        "wired_table_structure_recognition_model_dir": str(payload.get("wired_table_structure_recognition_model_dir") or ""),
        "wired_table_cells_detection_model_dir": str(payload.get("wired_table_cells_detection_model_dir") or ""),
        "table_orientation_classify_model_dir": str(payload.get("table_orientation_classify_model_dir") or ""),
        "text_det_limit_type": str(payload.get("text_det_limit_type") or ""),
        "text_det_limit_side_len": payload.get("text_det_limit_side_len"),
        "device": str(payload.get("device") or "cpu"),
    }


def _run(payload: dict) -> dict:
    mode = str(payload.get("mode") or "auto").strip().lower()
    if isinstance(payload.get("pages"), list):
        if mode == "command":
            return _run_command_batch(payload)
        if mode == "python":
            return _run_python_batch(payload)
        if str(payload.get("command") or "").strip():
            return _run_command_batch(payload)
        if str(payload.get("config_path") or "").strip() or str(payload.get("layout_detection_model_dir") or "").strip():
            return _run_python_batch(payload)
        return {
            "ok": True,
            "available": False,
            "warning": "ppstructurev3_layout_missing",
            "error": "ppstructurev3_config_or_command_missing",
            "pages": [],
        }
    if mode == "command":
        return _run_command(payload)
    if mode == "python":
        return _run_python(payload)
    if str(payload.get("command") or "").strip():
        return _run_command(payload)
    if str(payload.get("config_path") or "").strip() or str(payload.get("layout_detection_model_dir") or "").strip():
        return _run_python(payload)
    return {
        "ok": True,
        "available": False,
        "warning": "ppstructurev3_layout_missing",
        "error": "ppstructurev3_config_or_command_missing",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.local_ppstructurev3_sidecar INPUT_JSON OUTPUT_JSON", file=sys.stderr)
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
                    "warning": "ppstructurev3_layout_failed",
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
