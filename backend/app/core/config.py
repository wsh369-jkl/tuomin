"""Application settings and runtime path resolution."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.runtime_security import (
    ensure_private_directory,
    ensure_private_file,
    packaged_runtime_data_dir,
)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUNNING_FROZEN = bool(getattr(sys, "frozen", False))
SOURCE_BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_BACKEND_ROOT.parent
BUNDLED_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", SOURCE_BACKEND_ROOT))
APP_DATA_DIR_NAME = "ContractDesensitize"
STABLE_OLLAMA_MODEL = "qwen3.5:4b"
FAST_REVIEW_OLLAMA_MODEL = "qwen3:8b"
MID_REVIEW_OLLAMA_MODEL = "qwen3.5:9b"
MID_REVIEW_FALLBACK_OLLAMA_MODEL = "qwen3:14b"
DEFAULT_REVIEW_OLLAMA_MODEL = "qwen3.5:27b-q4_K_M"
HEAVY_ARBITRATION_OLLAMA_MODEL = "qwen3.6:27b"
REVIEW_OLLAMA_MODEL_PREFIX = "qwen3.5:27b"
HIGH_QUALITY_LOWMEM_MODE = "high_quality_lowmem"
LOCAL_HIGH_QUALITY_MODE = "local_high_quality"
LEGACY_DESENSITIZE_MODE = "legacy"
SUPPORTED_DESENSITIZE_MODES = {
    HIGH_QUALITY_LOWMEM_MODE,
    LOCAL_HIGH_QUALITY_MODE,
    LEGACY_DESENSITIZE_MODE,
}
DESENSITIZE_MODE_ALIASES = {
    "lowmem": HIGH_QUALITY_LOWMEM_MODE,
    "high_quality": HIGH_QUALITY_LOWMEM_MODE,
    "high_quality_low_memory": HIGH_QUALITY_LOWMEM_MODE,
    "local": LOCAL_HIGH_QUALITY_MODE,
    "local_high": LOCAL_HIGH_QUALITY_MODE,
    "local_quality": LOCAL_HIGH_QUALITY_MODE,
    "baseline": LEGACY_DESENSITIZE_MODE,
    "standard": LEGACY_DESENSITIZE_MODE,
    "ollama": LEGACY_DESENSITIZE_MODE,
}
_DESENSITIZE_MODE_OVERRIDE: ContextVar[Optional[str]] = ContextVar(
    "desensitize_mode_override",
    default=None,
)
DEFAULT_LOW_MEM_PRIMARY_IE_MODEL = "uer/roberta-base-finetuned-cluener2020-chinese"
DEFAULT_LOW_MEM_PRIMARY_NER_MODEL = "p988744/eland-ner-zh"
DEFAULT_LOW_MEM_SECONDARY_NER_MODEL = "shibing624/bert4ner-base-chinese"
DEFAULT_LOW_MEM_REVIEW_MODEL = "Qwen/Qwen3-1.7B-MLX-4bit"
DEFAULT_LOW_MEM_REVIEW_FALLBACK_MODEL = "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
LOCAL_CORS_ORIGIN_REGEX = r"^(null|https?://(localhost|127\.0\.0\.1)(:\d+)?)$"


def normalize_desensitize_mode(value: Optional[str]) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    return DESENSITIZE_MODE_ALIASES.get(normalized, normalized)


def is_supported_desensitize_mode(value: Optional[str]) -> bool:
    normalized = normalize_desensitize_mode(value)
    return bool(normalized and normalized in SUPPORTED_DESENSITIZE_MODES)


@contextmanager
def desensitize_mode_context(value: Optional[str]) -> Iterator[None]:
    normalized = normalize_desensitize_mode(value)
    if not normalized:
        yield
        return

    token = _DESENSITIZE_MODE_OVERRIDE.set(normalized)
    try:
        yield
    finally:
        _DESENSITIZE_MODE_OVERRIDE.reset(token)


def _default_runtime_root() -> Path:
    if RUNNING_FROZEN:
        return packaged_runtime_data_dir(APP_DATA_DIR_NAME)
    return SOURCE_BACKEND_ROOT


def _default_lowmem_model_root() -> Path:
    if RUNNING_FROZEN:
        return _default_runtime_root() / "models"
    return Path.home() / ".contract-desensitize" / "models"


def _default_frontend_dist_dir() -> Path:
    if RUNNING_FROZEN:
        return BUNDLED_RESOURCE_ROOT / "frontend_dist"
    return PROJECT_ROOT / "frontend" / "dist"


def _default_custom_config_path() -> Path:
    if RUNNING_FROZEN:
        return _default_runtime_root() / "config" / "custom_entities.json"
    return SOURCE_BACKEND_ROOT / "config" / "custom_entities.json"


def _default_seed_custom_config_path() -> Path:
    return BUNDLED_RESOURCE_ROOT / "config" / "custom_entities.json"


def _resolve_path(value: str | Path, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(SOURCE_BACKEND_ROOT / ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "合同脱敏系统"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = not RUNNING_FROZEN
    APP_HOST: str = "127.0.0.1"
    API_V1_PREFIX: str = "/api/v1"
    APP_PORT: int = 8000
    CORS_ALLOW_ORIGIN_REGEX: str = LOCAL_CORS_ORIGIN_REGEX

    LLM_BACKEND: str = "ollama"
    DESENSITIZE_MODE: str = HIGH_QUALITY_LOWMEM_MODE
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = STABLE_OLLAMA_MODEL
    OLLAMA_MODEL_OPTIONS: str = (
        f"{MID_REVIEW_OLLAMA_MODEL},{MID_REVIEW_FALLBACK_OLLAMA_MODEL},{FAST_REVIEW_OLLAMA_MODEL},"
        f"{STABLE_OLLAMA_MODEL},{HEAVY_ARBITRATION_OLLAMA_MODEL},{DEFAULT_REVIEW_OLLAMA_MODEL}"
    )
    OLLAMA_TIMEOUT: int = 600
    OLLAMA_NUM_CTX: int = 4096
    PDF_OCR_ENABLED: bool = True
    PDF_OCR_TIMEOUT: int = 90
    PDF_OCR_TEXT_THRESHOLD: int = 80
    PDF_OCR_MIN_READABLE_RATIO: float = 0.45
    PDF_OCR_RENDER_SCALE: float = 2.0
    PDF_OCR_IMAGE_MAX_EDGE: int = 2200
    PDF_OCR_JPEG_QUALITY: int = 88
    PDF_NORMALIZE_TO_DOCX: bool = True
    PDF_NORMALIZE_MAX_IMAGE_EDGE: int = 1800
    PDF_NORMALIZE_PAGE_CONCURRENCY: int = 1
    PDF_NORMALIZE_WORKER_PROCESS: bool = True
    PDF_NORMALIZE_WORKER_TIMEOUT: int = 900
    PDF_OCR_ENGINE: str = "rapidocr"
    PDF_OCR_MODEL: str = "ppocrv5_mobile"
    PDF_OCR_ACCURACY_PROFILE: str = "wps_challenger"
    PDF_OCR_DET_MODEL_PATH: Optional[str] = None
    PDF_OCR_REC_MODEL_PATH: Optional[str] = None
    PDF_OCR_CLS_MODEL_PATH: Optional[str] = None
    PDF_OCR_ENABLE_ANGLE_CLS: bool = False
    PDF_OCR_REC_BATCH_NUM: int = 1
    PDF_OCR_MAX_CANDIDATES: int = 3
    PDF_OCR_LOW_QUALITY_MAX_CANDIDATES: int = 5
    PDF_OCR_LOCAL_RETRY_MAX_REGIONS: int = 12
    PDF_OCR_LINE_RETRY_UPSCALE: float = 1.6
    PDF_OCR_LOW_CONFIDENCE_THRESHOLD: float = 0.65
    PDF_OCR_MIN_AVG_CONFIDENCE: float = 0.72
    PDF_OCR_TARGET_AVG_CONFIDENCE: float = 0.86
    PDF_OCR_HIGH_RES_RETRY_ENABLED: bool = True
    PDF_OCR_HIGH_RES_RETRY_SCALE: float = 3.0
    PDF_OCR_HIGH_RES_RETRY_MAX_IMAGE_EDGE: int = 2400
    PDF_OCR_HIGH_RES_RETRY_MAX_PAGES: int = 4
    PDF_OCR_HIGH_RES_RETRY_MIN_SCORE_GAIN: float = 4.0
    PDF_OCR_QUALITY_GATE_MIN_CONFIDENCE: float = 0.78
    PDF_OCR_QUALITY_GATE_MAX_LOW_RATIO: float = 0.25
    PDF_OCR_QUALITY_GATE_MAX_ABNORMAL_RATIO: float = 0.10
    PDF_OCR_PAGE_RISK_THRESHOLD: float = 0.45
    PDF_OCR_UNPAPER_MODE: str = "off"
    LOCAL_PDF_FRONTLINE_ENABLED: bool = True
    LOCAL_PDF_FRONTLINE_PROFILE: str = "quality_first_adaptive"
    LOCAL_PDF_GLM_MODEL: str = "glm-ocr:latest"
    LOCAL_PDF_GLM_API_MODE: str = "ollama_generate"
    LOCAL_PDF_GLM_POLICY: str = "quality_gate"
    LOCAL_PDF_GLM_SCOPE: str = "page_or_block"
    LOCAL_PDF_GLM_MAX_WORKERS: int = 1
    LOCAL_PDF_GLM_MAX_TOKENS: int = 8192
    LOCAL_PDF_GLM_RENDER_DPI: int = 160
    LOCAL_PDF_GLM_HIGH_RES_DPI: int = 180
    LOCAL_PDF_AUDIT_GLM_MAX_PAGES: int = 3
    LOCAL_PDF_AUDIT_GLM_MAX_PAGE_RATIO: float = 0.25
    PDF_WORD_AUDIT_REVIEW_MODEL: str = ""
    PDF_WORD_AUDIT_REVIEW_PROMPT_PROFILE: str = "evidence_arbitration_v2"
    PDF_WORD_AUDIT_V4_WORKER_THREADS: int = 1
    PDF_WORD_AUDIT_V4_RENDER_DPI: int = 180
    PDF_WORD_AUDIT_V4_RENDER_MAX_EDGE: int = 1900
    PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_NORMALIZE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MAX_EDGE: int = 1000
    PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_TIMEOUT: int = 30
    PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MAX_OCR_PAGES: int = 30
    PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MAX_OCR_CANDIDATES: int = 4
    PDF_WORD_AUDIT_V4_PAGE_ORIENTATION_MIN_CONFIDENCE: float = 0.62
    PDF_WORD_AUDIT_V4_QWEN_GATE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_QWEN_GATE_MAX_CANDIDATES: int = 9999
    PDF_WORD_AUDIT_V4_QWEN_GATE_TIMEOUT: int = 120
    PDF_WORD_AUDIT_V4_QWEN_VL_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_QWEN_VL_MODEL: str = "qwen3-vl:8b"
    PDF_WORD_AUDIT_V4_QWEN_VL_MAX_CANDIDATES: int = 9999
    PDF_WORD_AUDIT_V4_QWEN_VL_TIMEOUT: int = 240
    PDF_WORD_AUDIT_V4_QWEN_VL_GATE_MAX_TOTAL_SECONDS: int = 0
    PDF_WORD_AUDIT_V4_QWEN_VL_GATE_MAX_FAILED_REQUESTS: int = 0
    PDF_WORD_AUDIT_V4_QWEN_VL_MIN_CONFIDENCE: float = 0.72
    PDF_WORD_AUDIT_V4_QWEN_VL_STRICT_SELECTION_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_QWEN_VL_SHARED_SESSION_ENABLED: bool = False
    PDF_WORD_AUDIT_V4_QWEN_VL_FAST_OCR_GATE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_QWEN_VL_GENERATE_FALLBACK_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_QWEN_VL_KEEP_ALIVE: str = "5m"
    PDF_WORD_AUDIT_V4_CONFIRMED_COMMENTS_ONLY: bool = True
    PDF_WORD_AUDIT_V4_TABLE_VL_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_TABLE_VL_MAX_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_TABLE_VL_MAX_QWEN_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_TABLE_VL_TIMEOUT: int = 240
    PDF_WORD_AUDIT_V4_TABLE_VL_MAX_SAMPLES: int = 36
    PDF_WORD_AUDIT_V4_TABLE_VL_MAX_TOTAL_SECONDS: int = 0
    PDF_WORD_AUDIT_V4_TABLE_VL_PROMPT_MAX_CHARS: int = 6000
    PDF_WORD_AUDIT_V4_TABLE_VL_NUM_PREDICT: int = 1600
    PDF_WORD_AUDIT_V4_TABLE_VL_ORIENTATION_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_TABLE_VL_FAST_OCR_BYPASS_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_TABLE_VL_FAST_OCR_MIN_VALUES: int = 1
    PDF_WORD_AUDIT_V4_TABLE_VL_DEEP_MIN_UNRESOLVED: int = 6
    PDF_WORD_AUDIT_V4_TABLE_CELL_EVIDENCE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_TABLE_CELL_EVIDENCE_MAX_REVIEWS: int = 9999
    PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_RESULTS: int = 9999
    PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_RESULTS_PER_PAGE: int = 9999
    PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_COMMENTS: int = 120
    PDF_WORD_AUDIT_V4_TABLE_SPECIALIST_MAX_COMMENTS_PER_PAGE: int = 40
    PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_RESULTS: int = 9999
    PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_RESULTS_PER_PAGE: int = 9999
    PDF_WORD_AUDIT_V4_IMAGE_SPECIALIST_MAX_COMMENTS: int = 8
    PDF_WORD_AUDIT_V4_FRAGMENT_ANOMALY_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_FRAGMENT_ANOMALY_MAX_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_FRAGMENT_ANOMALY_MAX_COMMENTS: int = 4
    PDF_WORD_AUDIT_V4_IMAGE_PAGE_REVIEW_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_IMAGE_PAGE_REVIEW_MAX_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_IMAGE_PAGE_REVIEW_MAX_COMMENTS: int = 8
    PDF_WORD_AUDIT_V4_IMAGE_PAGE_COMMENTS_ENABLED: bool = False
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_SAMPLES: int = 10
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_TIMEOUT: int = 180
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_TOTAL_SECONDS: int = 0
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_UNPRODUCTIVE: int = 0
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_MAX_COMMENTS_PER_PAGE: int = 8
    PDF_WORD_AUDIT_V4_IMAGE_TEXT_VL_SKIP_BACKFILLED_PAGES: bool = False
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_SAMPLES: int = 16
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_TIMEOUT: int = 120
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_TOTAL_SECONDS: int = 0
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_PDF_CHARS: int = 3200
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MIN_PDF_CHARS: int = 16
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_NUM_PREDICT: int = 520
    PDF_WORD_AUDIT_V4_PAGE_TEXT_QWEN_MAX_COMMENTS_PER_PAGE: int = 6
    PDF_WORD_AUDIT_V4_MODEL_RECALL_MAX_CANDIDATES: int = 9999
    PDF_WORD_AUDIT_V4_MODEL_RECALL_MAX_PER_PAGE: int = 9999
    PDF_WORD_AUDIT_V4_PAGE_OCR_TEXT_EVIDENCE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_PAGE_OCR_TEXT_EVIDENCE_MAX_REVIEWS: int = 9999
    PDF_WORD_AUDIT_V4_PAGE_OCR_TEXT_EVIDENCE_MAX_COMMENTS_PER_PAGE: int = 8
    PDF_WORD_AUDIT_V4_PAGE_TEXT_COVERAGE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_PAGE_TEXT_COVERAGE_MAX_GAP_SAMPLES: int = 24
    PDF_WORD_AUDIT_V4_PAGE_TEXT_GAP_MAX_RESULTS: int = 9999
    PDF_WORD_AUDIT_V4_PAGE_TEXT_GAP_MAX_PER_PAGE: int = 9999
    PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_WINDOW: int = 4
    PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_MIN_TABLE_UNITS: int = 20
    PDF_WORD_AUDIT_V4_DOCX_PAGE_REMAP_MAX_SAMPLES: int = 9999
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MAX_PAGES: int = 9999
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_UNRESOLVED: int = 12
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_TABLE_UNRESOLVED: int = 8
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_VISUAL_UNRESOLVED: int = 6
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MIN_MAPPING_UNCERTAIN: int = 6
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MAX_SAMPLES: int = 10
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_COMMENTS_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_HIGH_RISK_PAGE_COVERAGE_MAX_COMMENTS: int = 6
    PDF_WORD_AUDIT_V4_FULL_CONTENT_COVERAGE_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_FULL_CONTENT_MAX_COMMENTS: int = 18
    PDF_WORD_AUDIT_V4_FULL_CONTENT_BACKFILL_MAX_COMMENTS: int = 12
    PDF_WORD_AUDIT_V4_FULL_CONTENT_MIN_TEXT_CHARS: int = 4
    PDF_WORD_AUDIT_V4_PAGE_COVERAGE_MAX_COMMENTS: int = 8
    PDF_WORD_AUDIT_V4_TABLE_GAP_GROUP_COMMENTS_ENABLED: bool = False
    PDF_WORD_AUDIT_V4_PAGE_COVERAGE_GROUP_COMMENTS_ENABLED: bool = False
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MAX_REGIONS: int = 9999
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MAX_PER_PAGE: int = 9999
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MIN_TEXT_CHARS: int = 4
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_MAX_TEXT_CHARS: int = 6000
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_TIMEOUT: int = 60
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_VL_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_VL_MAX_REGIONS: int = 9999
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_FULL_PAGE_TIMEOUT: int = 300
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_QWEN_FULL_PAGE_NUM_PREDICT: int = 1800
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_RESOLVE_MIN_PAGE_TEXT_CHARS: int = 20
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_RESOLVE_MIN_DOCX_CHARS: int = 4
    PDF_WORD_AUDIT_V4_COVERAGE_BACKFILL_RESOLVE_FUZZY_THRESHOLD: float = 0.9
    PDF_WORD_AUDIT_V4_COVERAGE_SCHEDULER_ENABLED: bool = True
    PDF_WORD_AUDIT_V4_COVERAGE_SCHEDULER_MAX_TASKS: int = 9999
    LOCAL_PDF_DOCX_RECOVERY_ENGINE: str = "layout_ir_docx_renderer"
    LOCAL_PDF_DOCX_RECOVERY_TIMEOUT: int = 180
    LOCAL_PDF_DOCX_RECOVERY_PYTHON: Optional[str] = None
    LOCAL_PDF_DOCX_RECOVERY_TABLE: bool = False
    LOCAL_PDF_DOCX_RECOVERY_FORMULA: bool = False
    LOCAL_PDF_OCR_PYTHON: Optional[str] = None
    LOCAL_PDF_PPOCRV5_ENABLED: bool = True
    LOCAL_PDF_PPOCRV5_POLICY: str = "always"
    LOCAL_PDF_FAST_ANCHOR_MIN_CONFIDENCE: float = 0.93
    LOCAL_PDF_FAST_ANCHOR_MIN_CHARS: int = 120
    LOCAL_PDF_FAST_ANCHOR_MAX_LOW_CONF_RATIO: float = 0.10
    LOCAL_PDF_FAST_ANCHOR_MAX_ABNORMAL_RATIO: float = 0.025
    LOCAL_PDF_PPSTRUCTUREV3_ENABLED: bool = True
    LOCAL_PDF_PPSTRUCTUREV3_API_MODE: str = "auto"
    LOCAL_PDF_PPSTRUCTUREV3_POLICY: str = "always"
    LOCAL_PDF_PPSTRUCTUREV3_CONFIG: Optional[str] = "models/local_pdf/PP-StructureV3.local.yaml"
    LOCAL_PDF_PPSTRUCTUREV3_COMMAND: Optional[str] = None
    LOCAL_PDF_PPSTRUCTUREV3_TIMEOUT: int = 900
    LOCAL_PDF_PPSTRUCTUREV3_THREADS: int = 1
    LOCAL_PDF_PPSTRUCTUREV3_PAGE_BATCH_SIZE: int = 1
    LOCAL_PDF_PPSTRUCTUREV3_DOC_ORIENTATION: bool = True
    LOCAL_PDF_PPSTRUCTUREV3_DOC_UNWARPING: Any = "auto"
    LOCAL_PDF_PPSTRUCTUREV3_TEXTLINE_ORIENTATION: bool = True
    LOCAL_PDF_PPSTRUCTUREV3_TABLE_RECOGNITION: bool = True
    LOCAL_PDF_PPSTRUCTUREV3_REGION_DETECTION: bool = True
    LOCAL_PDF_PPSTRUCTUREV3_SEAL_RECOGNITION: bool = False
    LOCAL_PDF_PPSTRUCTUREV3_FORMULA_RECOGNITION: bool = False
    LOCAL_PDF_PPSTRUCTUREV3_CHART_RECOGNITION: bool = False
    LOCAL_PDF_VL_FALLBACK_ENABLED: bool = True
    LOCAL_PDF_VL_FALLBACK_POLICY: str = "quality_gate"
    LOCAL_PDF_VL_MAX_PAGES: int = 2
    LOCAL_PDF_VL_MAX_PAGE_RATIO: float = 0.15
    LOCAL_PDF_VL_API_MODE: str = "paddleocr_python"
    LOCAL_PDF_VL_COMMAND: Optional[str] = None
    LOCAL_PDF_VL_MODEL: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    LOCAL_PDF_VL_MODEL_DIR: Optional[str] = None
    LOCAL_PDF_VL_REC_BACKEND: str = "auto"
    LOCAL_PDF_VL_REC_SERVER_URL: str = "http://localhost:8111/"
    LOCAL_PDF_VL_LAYOUT_DETECTION: bool = True
    LOCAL_PDF_VL_DOC_ORIENTATION: bool = True
    LOCAL_PDF_VL_DOC_UNWARPING: Any = "auto"
    LOCAL_PDF_VL_OCR_FOR_IMAGE_BLOCK: bool = True
    LOCAL_PDF_VL_SEAL_RECOGNITION: bool = False
    LOCAL_PDF_VL_CHART_RECOGNITION: bool = False
    LOCAL_PDF_VL_MIN_PIXELS: int = 112896
    LOCAL_PDF_VL_MAX_PIXELS: int = 1605632
    LOCAL_PDF_VL_TIMEOUT: int = 600
    LOCAL_PDF_VL_THREADS: int = 1
    LOCAL_PDF_PAGE_CHECKPOINT_ENABLED: bool = True
    LOCAL_PDF_CHECKPOINT_TTL_HOURS: int = 24
    LOCAL_PDF_RENDER_DPI: int = 200
    LOCAL_PDF_HIGH_RES_DPI: int = 260
    LOCAL_PDF_FRONTLINE_WORKER_TIMEOUT: int = 7200
    PDF_REVIEW_OCR_RENDER_SCALE: float = 1.45
    PDF_REVIEW_OCR_IMAGE_MAX_EDGE: int = 1700
    PAGE_SESSION_HEARTBEAT_GRACE_SECONDS: int = 15
    PAGE_SESSION_WATCH_INTERVAL_SECONDS: int = 3

    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
    LLM_MODEL_PATH: Optional[str] = None
    LLM_MAX_LENGTH: int = 4096
    LLM_GPU_MEMORY_UTILIZATION: float = 0.85
    LLM_TENSOR_PARALLEL_SIZE: int = 1
    LLM_DTYPE: str = "float16"
    USE_VLLM_SERVER: bool = False
    VLLM_API_BASE: Optional[str] = None

    PRIMARY_IE_MODEL: str = DEFAULT_LOW_MEM_PRIMARY_IE_MODEL
    PRIMARY_NER_MODEL: str = DEFAULT_LOW_MEM_PRIMARY_NER_MODEL
    SECONDARY_NER_MODEL: str = DEFAULT_LOW_MEM_SECONDARY_NER_MODEL
    REVIEW_MODEL: str = DEFAULT_LOW_MEM_REVIEW_MODEL
    REVIEW_MODEL_FALLBACK: str = DEFAULT_LOW_MEM_REVIEW_FALLBACK_MODEL
    PRIMARY_IE_BACKEND: str = "transformers_token_classification"
    PRIMARY_NER_BACKEND: str = "transformers"
    SECONDARY_NER_BACKEND: str = "transformers"
    ALLOW_UNSAFE_MODELSCOPE_UIE_RUNTIME: bool = False
    REVIEW_BACKEND: str = "mlx"
    REVIEW_MODEL_FALLBACK_BACKEND: str = "llama_cpp"
    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    REVIEW_LAZY_LOAD: bool = True
    REVIEW_UNLOAD_AFTER_TASK: bool = True
    REVIEW_NUM_CTX: int = 1536
    REVIEW_MAX_TOKENS: int = 384
    REVIEW_THINKING_MAX_TOKENS: int = 1024
    REVIEW_TEMPERATURE: float = 0.0
    REVIEW_MAX_SNIPPETS: int = 10
    REVIEW_MAX_CHARS_PER_SNIPPET: int = 1200
    MID_REVIEW_MODEL: str = MID_REVIEW_OLLAMA_MODEL
    MID_REVIEW_FALLBACK_MODEL: str = MID_REVIEW_FALLBACK_OLLAMA_MODEL
    FAST_REVIEW_MODEL: str = FAST_REVIEW_OLLAMA_MODEL
    LOWMEM_ALLOW_MID_REVIEW_MODEL: bool = False
    LOWMEM_ENABLE_LOCAL_REVIEW_FALLBACK: bool = False
    LOWMEM_ENABLE_HEAVY_ARBITRATION: bool = False
    REVIEW_WORKER_TIMEOUT: int = 180
    REVIEW_OLLAMA_TIMEOUT: int = 60
    LOWMEM_REVIEW_OLLAMA_TIMEOUT: int = 45
    HEAVY_ARBITRATION_MODEL: str = HEAVY_ARBITRATION_OLLAMA_MODEL
    ENABLE_HEAVY_ARBITRATION: bool = True
    HEAVY_ARBITRATION_MAX_SNIPPETS: int = 3
    MID_REVIEW_MAX_SNIPPETS: int = 12
    REVIEW_THINKING_MODE: str = "mid_review"
    ENABLE_PRIMARY_UIE: bool = True
    ENABLE_PRIMARY_NER: bool = True
    ENABLE_SECONDARY_NER: bool = False
    ENABLE_QWEN_REVIEW: bool = True
    LOWMEM_UNLOAD_PRIMARY_AFTER_STAGE: bool = True
    ANALYSIS_WORKER_PROCESS: bool = True
    ANALYSIS_STAGE_ISOLATION: bool = True
    ANALYSIS_WORKER_TIMEOUT: int = 900
    PROCESS_WORKER_TIMEOUT: int = 1800
    ANALYSIS_WORKER_PYTHON: Optional[str] = None
    ANALYSIS_WORKER_PYTHON_CHECK_TIMEOUT: int = 45
    QUALITY_POLICY: str = "recall_first"
    LOW_CONFIDENCE_ACTION: str = "keep_and_review"
    ALIAS_PROPAGATION: bool = True
    STRUCTURED_BACKFILL: bool = True
    RULE_RESULTS_ARE_AUTHORITATIVE: bool = True
    MODEL_RESULTS_CAN_ONLY_ADD: bool = True
    MODEL_RESULTS_CANNOT_DELETE_RULE_RESULTS: bool = True
    LOWMEM_MODEL_ROOT: str = str(_default_lowmem_model_root())

    IS_PACKAGED: bool = RUNNING_FROZEN
    RESOURCE_ROOT: str = str(BUNDLED_RESOURCE_ROOT)
    RUNTIME_ROOT: str = str(_default_runtime_root())
    FRONTEND_DIST_DIR: str = str(_default_frontend_dist_dir())
    CUSTOM_CONFIG_PATH: str = str(_default_custom_config_path())
    DEFAULT_CUSTOM_CONFIG_PATH: str = str(_default_seed_custom_config_path())

    UPLOAD_DIR: str = str(_default_runtime_root() / "uploads")
    OUTPUT_DIR: str = str(_default_runtime_root() / "outputs")
    DATABASE_PATH: str = str(_default_runtime_root() / "desensitize.db")
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = Field(default_factory=lambda: [".pdf", ".docx", ".txt"])
    DATABASE_URL: Optional[str] = None
    DATABASE_ECHO: bool = False

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(_default_runtime_root() / "logs" / "app.log")

    TASK_RETENTION_HOURS: int = 24
    MAX_PRESERVED_TASKS: int = 24

    def get_effective_ollama_model(self) -> str:
        return STABLE_OLLAMA_MODEL

    def get_effective_desensitize_mode(self) -> str:
        return (
            normalize_desensitize_mode(_DESENSITIZE_MODE_OVERRIDE.get())
            or normalize_desensitize_mode(self.DESENSITIZE_MODE)
            or HIGH_QUALITY_LOWMEM_MODE
        )

    def is_high_quality_lowmem_mode(self) -> bool:
        return self.get_effective_desensitize_mode() == HIGH_QUALITY_LOWMEM_MODE

    def is_local_high_quality_mode(self) -> bool:
        return self.get_effective_desensitize_mode() == LOCAL_HIGH_QUALITY_MODE

    def is_high_quality_desensitize_mode(self) -> bool:
        return self.is_high_quality_lowmem_mode() or self.is_local_high_quality_mode()

    def get_high_quality_profile_key(self) -> str:
        if self.is_local_high_quality_mode():
            return LOCAL_HIGH_QUALITY_MODE
        return HIGH_QUALITY_LOWMEM_MODE

    def get_high_quality_profile_label(self) -> str:
        if self.is_local_high_quality_mode():
            return "本机高质量脱敏模式"
        return "高质量低内存模式"

    def get_high_quality_memory_tier(self) -> str:
        if self.is_local_high_quality_mode():
            return "local_high_quality"
        return "lowmem_high_quality"

    def _parse_configured_ollama_model_options(self) -> List[str]:
        configured = [
            item.strip()
            for item in str(self.OLLAMA_MODEL_OPTIONS or "").split(",")
            if item.strip()
        ]
        if not configured:
            configured = [
                self.MID_REVIEW_MODEL,
                self.MID_REVIEW_FALLBACK_MODEL,
                self.FAST_REVIEW_MODEL,
                self.get_effective_ollama_model(),
                self.HEAVY_ARBITRATION_MODEL,
                DEFAULT_REVIEW_OLLAMA_MODEL,
            ]
        return configured

    @staticmethod
    def _parse_model_size_in_b(model_name: str | None) -> float:
        import re

        normalized = str(model_name or "").strip().lower()
        match = re.search(r":(\d+(?:\.\d+)?)b", normalized)
        return float(match.group(1)) if match else 0.0

    def is_fast_primary_ollama_model(self, model_name: str | None) -> bool:
        normalized = str(model_name or "").strip().lower()
        size_in_b = self._parse_model_size_in_b(normalized)
        return normalized in {self.get_effective_ollama_model().lower(), self.FAST_REVIEW_MODEL.lower()} or (
            0 < size_in_b < 9 and normalized.startswith(("qwen", "qwen3"))
        )

    def is_mid_review_ollama_model(self, model_name: str | None) -> bool:
        normalized = str(model_name or "").strip().lower()
        size_in_b = self._parse_model_size_in_b(normalized)
        if normalized in {
            self.MID_REVIEW_MODEL.lower(),
            self.MID_REVIEW_FALLBACK_MODEL.lower(),
            self.FAST_REVIEW_MODEL.lower(),
            "qwen3.5:9b",
        }:
            return True
        return normalized.startswith(("qwen", "qwen3")) and 8 <= size_in_b < 20

    def is_heavy_arbitration_ollama_model(self, model_name: str | None) -> bool:
        normalized = str(model_name or "").strip().lower()
        size_in_b = self._parse_model_size_in_b(normalized)
        return (
            normalized in {self.HEAVY_ARBITRATION_MODEL.lower(), DEFAULT_REVIEW_OLLAMA_MODEL.lower()}
            or normalized.startswith(REVIEW_OLLAMA_MODEL_PREFIX)
            or size_in_b >= 20
            or ":27b" in normalized
        )

    def is_review_capable_ollama_model(self, model_name: str | None) -> bool:
        return self.is_mid_review_ollama_model(model_name) or self.is_heavy_arbitration_ollama_model(model_name)

    def get_lowmem_ollama_review_candidates(self, available_models: Optional[List[str]] = None) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()

        def _append(model_name: str | None) -> None:
            normalized = str(model_name or "").strip()
            if not normalized or normalized in seen:
                return
            if self.is_fast_primary_ollama_model(normalized):
                seen.add(normalized)
                ordered.append(normalized)

        _append(self.FAST_REVIEW_MODEL)
        _append(self.get_effective_ollama_model())
        for item in self._parse_configured_ollama_model_options():
            _append(item)
        for item in available_models or []:
            _append(item)
        return ordered

    def get_route_review_model_candidates(self, available_models: Optional[List[str]] = None) -> List[str]:
        if self.is_high_quality_lowmem_mode():
            return [
                self.REVIEW_MODEL,
                self.REVIEW_MODEL_FALLBACK,
                *self.get_lowmem_ollama_review_candidates(available_models=available_models),
            ]
        return [
            self.MID_REVIEW_MODEL,
            self.MID_REVIEW_FALLBACK_MODEL,
            self.FAST_REVIEW_MODEL,
            self.get_effective_ollama_model(),
        ]

    def get_ollama_model_options(self, available_models: Optional[List[str]] = None) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()

        def _append(model_name: str | None) -> None:
            normalized = str(model_name or "").strip()
            if not normalized or normalized in seen:
                return
            if (
                normalized == self.get_effective_ollama_model()
                or self.is_fast_primary_ollama_model(normalized)
                or self.is_review_capable_ollama_model(normalized)
            ):
                seen.add(normalized)
                ordered.append(normalized)

        _append(self.MID_REVIEW_MODEL)
        _append(self.MID_REVIEW_FALLBACK_MODEL)
        _append(self.FAST_REVIEW_MODEL)
        _append(self.get_effective_ollama_model())
        _append(self.HEAVY_ARBITRATION_MODEL)
        _append(DEFAULT_REVIEW_OLLAMA_MODEL)
        for item in self._parse_configured_ollama_model_options():
            _append(item)
        for item in available_models or []:
            _append(item)
        return ordered

    def get_preferred_ollama_review_model(self, available_models: Optional[List[str]] = None) -> Optional[str]:
        available = {str(item).strip() for item in available_models or [] if str(item).strip()}
        if self.is_high_quality_lowmem_mode() and not self.LOWMEM_ALLOW_MID_REVIEW_MODEL:
            for candidate in self.get_lowmem_ollama_review_candidates(available_models=available_models):
                if not available or candidate in available:
                    return candidate
            return None

        options = self.get_ollama_model_options(available_models=available_models)

        def _available_or_unchecked(model_name: str) -> bool:
            return not available or model_name in available

        for candidate in (
            self.MID_REVIEW_MODEL,
            self.MID_REVIEW_FALLBACK_MODEL,
            self.FAST_REVIEW_MODEL,
            self.get_effective_ollama_model(),
        ):
            if candidate in options and _available_or_unchecked(candidate):
                return candidate
        for model_name in options:
            if self.is_mid_review_ollama_model(model_name) and _available_or_unchecked(model_name):
                return model_name
        for model_name in options:
            if self.is_heavy_arbitration_ollama_model(model_name) and _available_or_unchecked(model_name):
                return model_name
        return None

    def get_default_review_llm_model(self, available_models: Optional[List[str]] = None) -> Optional[str]:
        if self.is_high_quality_lowmem_mode():
            available = {str(item).strip() for item in available_models or [] if str(item).strip()}
            for candidate in (self.REVIEW_MODEL, self.REVIEW_MODEL_FALLBACK):
                if not available or candidate in available:
                    return candidate
            return self.get_preferred_ollama_review_model(available_models=available_models)
        return self.get_preferred_ollama_review_model(available_models=available_models)

    def get_default_llm_model(self) -> str:
        if self.LLM_BACKEND.lower() == "ollama":
            return self.get_effective_ollama_model()
        return self.LLM_MODEL_NAME

    def get_frontend_dist_path(self) -> Path:
        return Path(self.FRONTEND_DIST_DIR)

    def frontend_dist_available(self) -> bool:
        return (self.get_frontend_dist_path() / "index.html").exists()

    def model_post_init(self, __context) -> None:
        runtime_root = Path(_resolve_path(self.RUNTIME_ROOT, SOURCE_BACKEND_ROOT))
        resource_base = BUNDLED_RESOURCE_ROOT if self.IS_PACKAGED else SOURCE_BACKEND_ROOT
        resource_root = Path(_resolve_path(self.RESOURCE_ROOT, resource_base))
        frontend_base = BUNDLED_RESOURCE_ROOT if self.IS_PACKAGED else PROJECT_ROOT

        if self.LLM_BACKEND.lower() == "ollama":
            self.OLLAMA_MODEL = self.get_effective_ollama_model()
            self.OLLAMA_MODEL_OPTIONS = ",".join(self.get_ollama_model_options())

        self.RUNTIME_ROOT = str(runtime_root)
        self.RESOURCE_ROOT = str(resource_root)
        self.UPLOAD_DIR = _resolve_path(self.UPLOAD_DIR, runtime_root)
        self.OUTPUT_DIR = _resolve_path(self.OUTPUT_DIR, runtime_root)
        self.DATABASE_PATH = _resolve_path(self.DATABASE_PATH, runtime_root)
        self.LOG_FILE = _resolve_path(self.LOG_FILE, runtime_root)
        self.LOWMEM_MODEL_ROOT = _resolve_path(self.LOWMEM_MODEL_ROOT, runtime_root)
        self.CUSTOM_CONFIG_PATH = _resolve_path(self.CUSTOM_CONFIG_PATH, runtime_root)
        self.DEFAULT_CUSTOM_CONFIG_PATH = _resolve_path(
            self.DEFAULT_CUSTOM_CONFIG_PATH,
            resource_root,
        )
        self.FRONTEND_DIST_DIR = _resolve_path(self.FRONTEND_DIST_DIR, frontend_base)

        ensure_private_directory(runtime_root)
        ensure_private_directory(self.UPLOAD_DIR)
        ensure_private_directory(self.OUTPUT_DIR)
        ensure_private_directory(self.LOWMEM_MODEL_ROOT)
        ensure_private_directory(Path(self.LOG_FILE).parent)
        ensure_private_directory(Path(self.DATABASE_PATH).parent)
        ensure_private_directory(Path(self.CUSTOM_CONFIG_PATH).parent)

        ensure_private_file(self.LOG_FILE)
        ensure_private_file(self.DATABASE_PATH)
        ensure_private_file(self.CUSTOM_CONFIG_PATH)


settings = Settings()
