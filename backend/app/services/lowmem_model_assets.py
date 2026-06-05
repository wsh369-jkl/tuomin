"""Local model path resolution for the high-quality low-memory workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app.core.config import settings


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    path: Optional[Path]
    installed: bool
    role: str
    backend: str
    memory_tier: str = "lowmem"


OCR_MODEL_PATTERNS = {
    "det": [
        "RapidAI/RapidOCR/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx",
        "rapidocr/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx",
        "PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx",
        "PP-OCRv5_mobile_det/inference.onnx",
        "**/ch_PP-OCRv5_mobile_det.onnx",
        "**/*PP-OCRv5*mobile*det*.onnx",
    ],
    "rec": [
        "RapidAI/RapidOCR/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx",
        "rapidocr/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx",
        "PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx",
        "RapidAI/RapidOCR/onnx/PP-OCRv5/rec/ch_PP-OCRv5_mobile_rec.onnx",
        "rapidocr/onnx/PP-OCRv5/rec/ch_PP-OCRv5_mobile_rec.onnx",
        "PP-OCRv5/rec/ch_PP-OCRv5_mobile_rec.onnx",
        "PP-OCRv5_mobile_rec/inference.onnx",
        "**/ch_PP-OCRv5_rec_mobile_infer.onnx",
        "**/ch_PP-OCRv5_mobile_rec.onnx",
        "**/*PP-OCRv5*mobile*rec*.onnx",
        "**/*PP-OCRv5*rec*mobile*.onnx",
    ],
    "cls": [
        "RapidAI/RapidOCR/onnx/PP-OCRv5/cls/ch_PP-OCRv5_mobile_cls.onnx",
        "rapidocr/onnx/PP-OCRv5/cls/ch_PP-OCRv5_mobile_cls.onnx",
        "PP-OCRv5/cls/ch_PP-OCRv5_mobile_cls.onnx",
        "**/*PP-OCRv5*mobile*cls*.onnx",
        "**/*ppocr*mobile*cls*.onnx",
        "**/ch_ppocr_mobile_v2.0_cls_infer.onnx",
    ],
}


MODEL_ALIASES = {
    "uie-base": [
        "uer/roberta-base-finetuned-cluener2020-chinese",
        "damo/nlp_structbert_siamese-uie_chinese-base",
        "lvyufeng/uie-base",
    ],
    "mlx-community/Qwen3-1.7B-4bit": [
        "mlx-community/Qwen3-1.7B-4bit",
        "Qwen/Qwen3-1.7B-MLX-4bit",
    ],
    "Qwen3.5-0.8B-Q4_K_M-GGUF": [
        "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "bartowski/Qwen_Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
    ],
    "damo/nlp_raner_named-entity-recognition_chinese-base-news": [
        "p988744/eland-ner-zh",
    ],
}


def model_root() -> Path:
    return Path(settings.LOWMEM_MODEL_ROOT).expanduser().resolve()


def _candidate_ids(model_id: str) -> list[str]:
    normalized = str(model_id or "").strip()
    if not normalized:
        return []
    return [normalized, *MODEL_ALIASES.get(normalized, [])]


def _has_model_file(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in {".gguf", ".safetensors", ".bin", ".onnx"}
    if not path.exists() or not path.is_dir():
        return False
    model_file_names = {
        "pytorch_model.bin",
        "model.safetensors",
        "model.onnx",
    }
    if any((path / item).exists() for item in model_file_names):
        return True
    return any(path.glob("*.gguf"))


def resolve_model_path(model_id: str) -> Optional[Path]:
    root = model_root()
    for candidate in _candidate_ids(model_id):
        candidate_path = Path(candidate).expanduser()
        paths: list[Path] = []
        if candidate_path.is_absolute():
            paths.append(candidate_path)
        else:
            paths.append(root / candidate)

        for path in paths:
            if _has_model_file(path):
                return path
    return None


def _configured_ocr_model_path(kind: str) -> Optional[Path]:
    attr_name = f"PDF_OCR_{kind.upper()}_MODEL_PATH"
    raw_path = str(getattr(settings, attr_name, "") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.exists() and path.is_file():
        return path.resolve()
    return None


def resolve_ocr_model_path(kind: str) -> Optional[Path]:
    """Resolve a local RapidOCR model file without triggering runtime downloads."""
    normalized_kind = str(kind or "").strip().lower()
    configured = _configured_ocr_model_path(normalized_kind)
    if configured is not None:
        return configured

    root_candidates = [
        model_root(),
        Path(settings.RESOURCE_ROOT).expanduser().resolve() / "models",
        Path(settings.RUNTIME_ROOT).expanduser().resolve() / "models",
    ]
    patterns = OCR_MODEL_PATTERNS.get(normalized_kind, [])
    for root in root_candidates:
        if not root.exists():
            continue
        for pattern in patterns:
            if pattern.startswith("**/"):
                matches = sorted(path for path in root.glob(pattern) if path.is_file())
                if matches:
                    return matches[0].resolve()
                continue
            candidate = root / pattern
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
    return None


def ocr_model_paths() -> dict[str, Optional[Path]]:
    return {
        "det": resolve_ocr_model_path("det"),
        "rec": resolve_ocr_model_path("rec"),
        "cls": resolve_ocr_model_path("cls"),
    }


def ocr_models_ready(*, require_cls: bool = False) -> bool:
    paths = ocr_model_paths()
    if not paths.get("det") or not paths.get("rec"):
        return False
    if require_cls and not paths.get("cls"):
        return False
    return True


def ocr_assets() -> list[ModelAsset]:
    paths = ocr_model_paths()
    specs = [
        ("RapidAI/RapidOCR/ch_PP-OCRv5_mobile_det", "pdf_ocr_det", "det", True),
        ("RapidAI/RapidOCR/ch_PP-OCRv5_rec_mobile", "pdf_ocr_rec", "rec", True),
        ("RapidAI/RapidOCR/ch_ppocr_mobile_v2.0_cls", "pdf_ocr_cls", "cls", False),
    ]
    return [
        ModelAsset(
            model_id=model_id,
            path=paths.get(kind),
            installed=paths.get(kind) is not None,
            role=role,
            backend="rapidocr_onnxruntime",
            memory_tier="ocr_ppocrv5_mobile_required" if required else "ocr_angle_cls_optional",
        )
        for model_id, role, kind, required in specs
    ]


def model_installed(model_id: str) -> bool:
    return resolve_model_path(model_id) is not None


def build_model_asset(
    model_id: str,
    *,
    role: str,
    backend: str,
    memory_tier: str = "lowmem",
) -> ModelAsset:
    path = resolve_model_path(model_id)
    return ModelAsset(
        model_id=model_id,
        path=path,
        installed=path is not None,
        role=role,
        backend=backend,
        memory_tier=memory_tier,
    )


def all_assets() -> list[ModelAsset]:
    return [
        build_model_asset(
            settings.PRIMARY_IE_MODEL,
            role="primary_ie",
            backend=settings.PRIMARY_IE_BACKEND,
            memory_tier="primary_lowmem",
        ),
        build_model_asset(
            settings.PRIMARY_NER_MODEL,
            role="primary_ner",
            backend=settings.PRIMARY_NER_BACKEND,
            memory_tier="primary_lowmem",
        ),
        build_model_asset(
            settings.SECONDARY_NER_MODEL,
            role="secondary_ner",
            backend=settings.SECONDARY_NER_BACKEND,
            memory_tier="primary_lowmem",
        ),
        build_model_asset(
            settings.REVIEW_MODEL,
            role="review",
            backend=settings.REVIEW_BACKEND,
            memory_tier="review_1_7b_4bit",
        ),
        build_model_asset(
            settings.REVIEW_MODEL_FALLBACK,
            role="review_fallback",
            backend=settings.REVIEW_MODEL_FALLBACK_BACKEND,
            memory_tier="review_0_8b_q4",
        ),
    ]


def primary_models_ready() -> bool:
    required = [
        settings.PRIMARY_IE_MODEL if settings.ENABLE_PRIMARY_UIE else "",
        settings.PRIMARY_NER_MODEL if settings.ENABLE_PRIMARY_NER else "",
        settings.SECONDARY_NER_MODEL if settings.ENABLE_SECONDARY_NER else "",
    ]
    return all(model_installed(item) for item in required if item)


def review_model_installed() -> bool:
    if not settings.ENABLE_QWEN_REVIEW:
        return False
    return model_installed(settings.REVIEW_MODEL) or model_installed(settings.REVIEW_MODEL_FALLBACK)


def installed_model_names(model_ids: Iterable[str]) -> list[str]:
    return [model_id for model_id in model_ids if model_installed(model_id)]
