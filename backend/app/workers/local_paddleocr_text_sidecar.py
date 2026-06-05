"""High-precision PaddleOCR text sidecar for PDF/WPS audit.

This worker is intentionally isolated from the API process because PaddleOCR
loads Paddle/PaddleX native libraries and model weights. It returns only text
evidence, not DOCX layout reconstruction.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
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
        return {str(key): _to_builtin(item) for key, item in value.items()}
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
    for name in ("json", "res", "data", "input_path"):
        if hasattr(value, name):
            try:
                public[name] = _to_builtin(getattr(value, name))
            except Exception:
                pass
    return public or str(value)


def _model_dir(name: str, configured: str | None = None) -> str | None:
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)
    candidates = [
        Path.home() / ".paddlex" / "official_models" / name,
        Path.home() / ".paddleocr" / "whl" / name,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _extract_result_dict(value: Any) -> dict[str, Any]:
    raw = _to_builtin(value)
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return {"raw": raw}
    if isinstance(raw.get("res"), dict):
        return raw["res"]
    if isinstance(raw.get("json"), dict):
        nested = raw["json"]
        if isinstance(nested.get("res"), dict):
            return nested["res"]
        return nested
    return raw


def _bbox_from_poly(poly: Any) -> list[float]:
    try:
        points = poly.tolist() if hasattr(poly, "tolist") else poly
        xs = [float(point[0]) for point in points if isinstance(point, (list, tuple)) and len(point) >= 2]
        ys = [float(point[1]) for point in points if isinstance(point, (list, tuple)) and len(point) >= 2]
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        pass
    return []


def _lines_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    texts = result.get("rec_texts") or result.get("texts") or result.get("text") or []
    if isinstance(texts, str):
        texts = [line for line in texts.splitlines() if line.strip()]
    scores = result.get("rec_scores") or result.get("scores") or []
    polys = result.get("dt_polys") or result.get("rec_polys") or result.get("boxes") or []
    word_boxes = (
        result.get("rec_word_info")
        or result.get("word_boxes")
        or result.get("rec_word_boxes")
        or result.get("word_info")
        or []
    )
    lines: list[dict[str, Any]] = []
    if isinstance(texts, list):
        for index, text in enumerate(texts):
            value = str(text or "").strip()
            if not value:
                continue
            score = 0.0
            try:
                score = float(scores[index]) if isinstance(scores, list) and index < len(scores) else 0.0
            except Exception:
                score = 0.0
            bbox = _bbox_from_poly(polys[index]) if isinstance(polys, list) and index < len(polys) else []
            word_box = word_boxes[index] if isinstance(word_boxes, list) and index < len(word_boxes) else None
            lines.append(
                {
                    "text": value,
                    "confidence": round(score if score > 0 else 0.82, 4),
                    "bbox": bbox,
                    "word_box": _to_builtin(word_box) if word_box is not None else [],
                    "source": "ppocrv5_server",
                }
            )
    return lines


def _quality(lines: list[dict[str, Any]]) -> tuple[str, float]:
    if not lines:
        return "failed", 0.0
    confidence = sum(float(line.get("confidence") or 0.0) for line in lines) / max(1, len(lines))
    text_chars = sum(len(str(line.get("text") or "").strip()) for line in lines)
    if confidence >= 0.92 and text_chars >= 80:
        return "high", confidence
    if confidence >= 0.82 and text_chars >= 20:
        return "medium", confidence
    return "low", confidence


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    pages = payload.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return {"ok": False, "warning": "paddleocr_text_missing", "error": "pages_missing", "pages": []}

    det_dir = _model_dir("PP-OCRv5_server_det", payload.get("det_model_dir"))
    rec_dir = _model_dir("PP-OCRv5_server_rec", payload.get("rec_model_dir"))
    ori_dir = _model_dir("PP-LCNet_x1_0_textline_ori", payload.get("textline_orientation_model_dir"))
    doc_ori_dir = _model_dir("PP-LCNet_x1_0_doc_ori", payload.get("doc_orientation_model_dir"))
    unwarp_dir = _model_dir("UVDoc", payload.get("doc_unwarping_model_dir"))
    if not det_dir or not rec_dir:
        return {
            "ok": False,
            "warning": "paddleocr_text_model_missing",
            "error": "PP-OCRv5_server_det_or_rec_missing",
            "pages": [],
        }

    threads = str(max(1, int(payload.get("threads") or 2)))
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads

    from paddleocr import PaddleOCR

    det_limit_type = str(payload.get("det_limit_type") or "min").strip().lower()
    if det_limit_type not in {"min", "max"}:
        det_limit_type = "min"
    use_textline_orientation = bool(payload.get("use_textline_orientation", True)) and bool(ori_dir)
    device = str(payload.get("device") or "cpu")

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=det_dir,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=rec_dir,
        doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori" if doc_ori_dir else None,
        doc_orientation_classify_model_dir=doc_ori_dir,
        doc_unwarping_model_name="UVDoc" if unwarp_dir else None,
        doc_unwarping_model_dir=unwarp_dir,
        textline_orientation_model_name="PP-LCNet_x1_0_textline_ori" if ori_dir else None,
        textline_orientation_model_dir=ori_dir,
        use_doc_orientation_classify=bool(payload.get("doc_orientation")),
        use_doc_unwarping=bool(payload.get("doc_unwarping")),
        use_textline_orientation=use_textline_orientation,
        text_recognition_batch_size=max(1, int(payload.get("rec_batch_size") or 1)),
        text_det_limit_side_len=max(1280, int(payload.get("det_limit_side_len") or 1280)),
        text_det_limit_type=det_limit_type,
        text_det_thresh=payload.get("text_det_thresh"),
        text_det_box_thresh=payload.get("text_det_box_thresh"),
        text_det_unclip_ratio=payload.get("text_det_unclip_ratio"),
        text_rec_score_thresh=float(payload.get("text_rec_score_thresh") or 0.0),
        return_word_box=bool(payload.get("return_word_box", True)),
        device=device,
        cpu_threads=max(1, int(payload.get("threads") or 2)),
        enable_mkldnn=bool(payload.get("enable_mkldnn", False)),
    )
    page_results: list[dict[str, Any]] = []
    try:
        for index, page in enumerate(pages):
            page_number = int(page.get("page") or index + 1)
            image_path = Path(str(page.get("image_path") or "")).expanduser()
            if not image_path.exists():
                page_results.append(
                    {
                        "ok": False,
                        "page": page_number,
                        "warning": "paddleocr_text_image_missing",
                        "error": f"image_not_found:{image_path}",
                    }
                )
                continue
            try:
                raw_result = ocr.predict(str(image_path))
                result = _extract_result_dict(raw_result)
                lines = _lines_from_result(result)
                quality, confidence = _quality(lines)
                text = "\n".join(str(line.get("text") or "") for line in lines if str(line.get("text") or "").strip())
                page_results.append(
                    {
                        "ok": True,
                        "page": page_number,
                        "text": text,
                        "quality": quality,
                        "confidence": round(confidence, 4),
                        "lines": lines,
                        "engine": "ppocrv5_server",
                        "model_settings": {
                            "profile_name": str(payload.get("profile_name") or "base"),
                            "det_limit_type": det_limit_type,
                            "det_limit_side_len": max(1280, int(payload.get("det_limit_side_len") or 1280)),
                            "doc_orientation": bool(payload.get("doc_orientation")),
                            "doc_unwarping": bool(payload.get("doc_unwarping")),
                            "doc_orientation_model_dir": bool(doc_ori_dir),
                            "doc_unwarping_model_dir": bool(unwarp_dir),
                            "use_textline_orientation": use_textline_orientation,
                            "text_det_thresh": payload.get("text_det_thresh"),
                            "text_det_box_thresh": payload.get("text_det_box_thresh"),
                            "text_det_unclip_ratio": payload.get("text_det_unclip_ratio"),
                            "return_word_box": bool(payload.get("return_word_box", True)),
                            "rec_batch_size": max(1, int(payload.get("rec_batch_size") or 1)),
                            "device": device,
                        },
                        "warnings": [] if text.strip() else ["paddleocr_text_empty"],
                    }
                )
            except Exception as exc:
                page_results.append(
                    {
                        "ok": False,
                        "page": page_number,
                        "warning": "paddleocr_text_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        try:
            ocr.close()
        except Exception:
            pass

    return {
        "ok": True,
        "available": True,
        "engine": "ppocrv5_server",
        "model_settings": {
            "profile_name": str(payload.get("profile_name") or "base"),
            "det_limit_type": det_limit_type,
            "det_limit_side_len": max(1280, int(payload.get("det_limit_side_len") or 1280)),
            "doc_orientation": bool(payload.get("doc_orientation")),
            "doc_unwarping": bool(payload.get("doc_unwarping")),
            "doc_orientation_model_dir": bool(doc_ori_dir),
            "doc_unwarping_model_dir": bool(unwarp_dir),
            "use_textline_orientation": use_textline_orientation,
            "text_det_thresh": payload.get("text_det_thresh"),
            "text_det_box_thresh": payload.get("text_det_box_thresh"),
            "text_det_unclip_ratio": payload.get("text_det_unclip_ratio"),
            "return_word_box": bool(payload.get("return_word_box", True)),
            "rec_batch_size": max(1, int(payload.get("rec_batch_size") or 1)),
            "device": device,
        },
        "pages": page_results,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.workers.local_paddleocr_text_sidecar INPUT_JSON OUTPUT_JSON", file=sys.stderr)
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
                    "available": False,
                    "warning": "paddleocr_text_failed",
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
