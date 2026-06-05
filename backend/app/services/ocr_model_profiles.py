from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OcrAssetPaths:
    paddlex_root: Path = Path.home() / ".paddlex" / "official_models"
    paddleocr_root: Path = Path.home() / ".paddleocr" / "whl"
    hf_root: Path = Path.home() / ".cache" / "huggingface" / "hub"

    def official(self, name: str) -> str:
        path = self.paddlex_root / name
        return str(path) if path.exists() else ""

    def paddleocr_whl(self, *parts: str) -> str:
        path = self.paddleocr_root.joinpath(*parts)
        return str(path) if path.exists() else ""

    def glm_layout_snapshot(self) -> str:
        root = self.hf_root / "models--PaddlePaddle--PP-DocLayoutV3_safetensors"
        snapshots = root / "snapshots"
        if not snapshots.exists():
            return ""
        for item in sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if (item / "config.json").exists() and any(p.exists() and p.stat().st_size >= 50 * 1024 * 1024 for p in item.glob("*.safetensors")):
                return str(item)
        return ""

    def paddlex_model(self, name: str) -> str:
        path = self.paddlex_root / name
        return str(path) if (path / "inference.yml").exists() else ""


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


def ppocrv5_server_profile(*, mode: str = "document_clear", threads: int = 2) -> dict[str, Any]:
    assets = OcrAssetPaths()
    hard = str(mode or "").lower() in {"hard", "blur", "photo", "table", "document_hard"}
    return {
        "profile_name": f"ppocrv5_server_{'hard_min_1600' if hard else 'clear_min_1280'}",
        "det_model_dir": assets.official("PP-OCRv5_server_det"),
        "rec_model_dir": assets.official("PP-OCRv5_server_rec"),
        "textline_orientation_model_dir": assets.official("PP-LCNet_x1_0_textline_ori"),
        "doc_orientation_model_dir": assets.official("PP-LCNet_x1_0_doc_ori"),
        "doc_unwarping_model_dir": assets.official("UVDoc"),
        "threads": max(1, int(threads or 2)),
        "det_limit_type": "min",
        "det_limit_side_len": 1600 if hard else 1280,
        "text_det_thresh": 0.3,
        "text_det_box_thresh": 0.5 if hard else 0.6,
        "text_det_unclip_ratio": 1.5,
        "doc_orientation": bool(hard),
        "doc_unwarping": str(mode or "").lower() in {"photo", "warped", "photo_warped"},
        "use_textline_orientation": True,
        "return_word_box": True,
        "rec_batch_size": 1,
        "device": "cpu",
        "enable_mkldnn": False,
    }


def ppstructurev3_profile(*, mode: str = "document_table", threads: int = 2) -> dict[str, Any]:
    assets = OcrAssetPaths()
    normalized_mode = str(mode or "").lower()
    photo = normalized_mode in {"photo", "warped", "photo_warped"}
    formula = "formula" in normalized_mode
    chart = "chart" in normalized_mode
    return {
        "profile_name": f"ppstructurev3_{mode}",
        "layout_detection_model_dir": assets.official("PP-DocLayout_plus-L") or assets.official("PP-DocBlockLayout"),
        "doc_orientation_classify_model_dir": assets.official("PP-LCNet_x1_0_doc_ori"),
        "doc_unwarping_model_dir": assets.official("UVDoc"),
        "text_detection_model_dir": assets.official("PP-OCRv5_server_det"),
        "text_recognition_model_dir": assets.official("PP-OCRv5_server_rec"),
        "textline_orientation_model_dir": assets.official("PP-LCNet_x1_0_textline_ori"),
        "table_classification_model_dir": assets.official("PP-LCNet_x1_0_table_cls"),
        "wired_table_structure_recognition_model_dir": assets.official("SLANeXt_wired") or assets.official("SLANet_plus"),
        "wireless_table_structure_recognition_model_dir": assets.official("SLANet_plus"),
        "wired_table_cells_detection_model_dir": assets.official("RT-DETR-L_wired_table_cell_det"),
        "wireless_table_cells_detection_model_dir": assets.official("RT-DETR-L_wireless_table_cell_det"),
        "table_orientation_classify_model_dir": assets.official("PP-LCNet_x1_0_doc_ori"),
        "formula_recognition_model_dir": assets.paddleocr_whl("formula", "rec_latex_ocr_infer"),
        "threads": max(1, int(threads or 2)),
        "text_det_limit_type": "min",
        "text_det_limit_side_len": 1600 if photo else 1280,
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": bool(photo),
        "use_textline_orientation": True,
        "use_table_recognition": True,
        "use_region_detection": True,
        "use_seal_recognition": False,
        "use_formula_recognition": bool(formula and assets.paddleocr_whl("formula", "rec_latex_ocr_infer")),
        "use_chart_recognition": bool(chart),
        "format_block_content": True,
        "device": "cpu",
        "enable_mkldnn": False,
    }


def glmocr_sdk_profile(*, base_url: str, model: str, max_tokens: int = 8192, timeout: int = 900) -> dict[str, Any]:
    assets = OcrAssetPaths()
    return {
        "base_url": str(base_url or "http://127.0.0.1:11434"),
        "model": str(model or "glm-ocr:latest"),
        "timeout": max(60, int(timeout or 900)),
        "max_tokens": max(1024, min(int(max_tokens or 8192), 8192)),
        "layout_model_dir": assets.glm_layout_snapshot(),
        "layout_device": "cpu",
        "layout_batch_size": 1,
        "result_format": "both",
    }


def paddleocr_vl_profile(*, mode: str = "hard_document", server_url: str = "http://localhost:8111/", threads: int = 2) -> dict[str, Any]:
    assets = OcrAssetPaths()
    backend_root = Path(__file__).resolve().parents[2]
    mlx_server_executable = backend_root / "mlx_vlm_venv" / "bin" / "mlx_vlm.server"
    backend = "mlx-vlm-server" if _server_ready(server_url) else "auto"
    warnings: list[str] = []
    if backend == "auto" and mlx_server_executable.exists():
        warnings.append("vl_mlx_autostart_pending")
    elif backend == "auto":
        warnings.append("vl_mlx_unavailable")
    photo = str(mode or "").lower() in {"photo", "warped", "photo_warped"}
    model_dir = assets.official("PaddleOCR-VL-1.5")
    return {
        "profile_name": f"paddleocr_vl_1_5_{backend}_{mode}",
        "model_dir": model_dir,
        "layout_detection_model_dir": assets.paddlex_model("PP-DocLayoutV3"),
        "doc_orientation_classify_model_dir": assets.official("PP-LCNet_x1_0_doc_ori"),
        "doc_unwarping_model_dir": assets.official("UVDoc"),
        "vl_rec_backend": backend,
        "vl_rec_server_url": str(server_url or "http://localhost:8111/"),
        "auto_start_mlx_server": True,
        "mlx_server_executable": str(mlx_server_executable),
        "mlx_prefill_step_size": 256,
        "mlx_max_kv_size": 8192,
        "vl_rec_api_model_name": "PaddlePaddle/PaddleOCR-VL-1.5",
        "vl_rec_model_name": "PaddleOCR-VL-1.5-0.9B",
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": bool(photo),
        "use_layout_detection": True,
        "use_chart_recognition": True,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": True,
        "min_pixels": 112896,
        "max_pixels": 3211264 if photo else 1605632,
        "max_new_tokens": 8192,
        "temperature": 0.0,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "threads": max(1, int(threads or 2)),
        "device": "cpu",
        "enable_mkldnn": False,
        "warnings": warnings,
    }


def model_profile_report() -> dict[str, Any]:
    assets = OcrAssetPaths()
    required = {
        "ppocr_det": assets.official("PP-OCRv5_server_det"),
        "ppocr_rec": assets.official("PP-OCRv5_server_rec"),
        "textline_orientation": assets.official("PP-LCNet_x1_0_textline_ori"),
        "doc_orientation": assets.official("PP-LCNet_x1_0_doc_ori"),
        "uvdoc": assets.official("UVDoc"),
        "glm_layout": assets.glm_layout_snapshot(),
        "ppstructure_layout": assets.official("PP-DocLayout_plus-L") or assets.official("PP-DocBlockLayout"),
        "table_cls": assets.official("PP-LCNet_x1_0_table_cls"),
        "wired_table_structure": assets.official("SLANeXt_wired") or assets.official("SLANet_plus"),
        "wired_table_cell": assets.official("RT-DETR-L_wired_table_cell_det"),
        "wireless_table_cell": assets.official("RT-DETR-L_wireless_table_cell_det"),
        "paddleocr_vl": assets.official("PaddleOCR-VL-1.5"),
        "formula": assets.paddleocr_whl("formula", "rec_latex_ocr_infer"),
    }
    return {
        "assets": {key: {"path": value, "available": bool(value)} for key, value in required.items()},
        "profiles": {
            "ppocrv5_clear": ppocrv5_server_profile(mode="document_clear"),
            "ppocrv5_hard": ppocrv5_server_profile(mode="document_hard"),
            "ppstructurev3_table": ppstructurev3_profile(mode="document_table"),
            "glmocr_sdk": glmocr_sdk_profile(base_url="http://127.0.0.1:11434", model="glm-ocr:latest"),
            "paddleocr_vl": paddleocr_vl_profile(mode="hard_document"),
        },
    }
