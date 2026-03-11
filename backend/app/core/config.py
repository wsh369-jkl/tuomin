"""Application settings and runtime path resolution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.runtime_security import (
    ensure_private_directory,
    ensure_private_file,
    packaged_runtime_data_dir,
)


RUNNING_FROZEN = bool(getattr(sys, "frozen", False))
SOURCE_BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_BACKEND_ROOT.parent
BUNDLED_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", SOURCE_BACKEND_ROOT))
APP_DATA_DIR_NAME = "ContractDesensitize"
STABLE_OLLAMA_MODEL = "qwen3.5:4b"
LOCAL_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def _default_runtime_root() -> Path:
    if RUNNING_FROZEN:
        return packaged_runtime_data_dir(APP_DATA_DIR_NAME)
    return SOURCE_BACKEND_ROOT


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
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = STABLE_OLLAMA_MODEL
    OLLAMA_MODEL_OPTIONS: str = STABLE_OLLAMA_MODEL
    OLLAMA_TIMEOUT: int = 600
    OLLAMA_NUM_CTX: int = 4096
    PDF_OCR_ENABLED: bool = True
    PDF_OCR_TEXT_THRESHOLD: int = 80
    PDF_OCR_MIN_READABLE_RATIO: float = 0.45
    PDF_OCR_RENDER_SCALE: float = 2.0
    PDF_OCR_IMAGE_MAX_EDGE: int = 2200
    PDF_OCR_JPEG_QUALITY: int = 88

    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
    LLM_MODEL_PATH: Optional[str] = None
    LLM_MAX_LENGTH: int = 4096
    LLM_GPU_MEMORY_UTILIZATION: float = 0.85
    LLM_TENSOR_PARALLEL_SIZE: int = 1
    LLM_DTYPE: str = "float16"
    USE_VLLM_SERVER: bool = False
    VLLM_API_BASE: Optional[str] = None

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

    def get_ollama_model_options(self) -> List[str]:
        return [self.get_effective_ollama_model()]

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
            self.OLLAMA_MODEL_OPTIONS = self.OLLAMA_MODEL

        self.RUNTIME_ROOT = str(runtime_root)
        self.RESOURCE_ROOT = str(resource_root)
        self.UPLOAD_DIR = _resolve_path(self.UPLOAD_DIR, runtime_root)
        self.OUTPUT_DIR = _resolve_path(self.OUTPUT_DIR, runtime_root)
        self.DATABASE_PATH = _resolve_path(self.DATABASE_PATH, runtime_root)
        self.LOG_FILE = _resolve_path(self.LOG_FILE, runtime_root)
        self.CUSTOM_CONFIG_PATH = _resolve_path(self.CUSTOM_CONFIG_PATH, runtime_root)
        self.DEFAULT_CUSTOM_CONFIG_PATH = _resolve_path(
            self.DEFAULT_CUSTOM_CONFIG_PATH,
            resource_root,
        )
        self.FRONTEND_DIST_DIR = _resolve_path(self.FRONTEND_DIST_DIR, frontend_base)

        ensure_private_directory(runtime_root)
        ensure_private_directory(self.UPLOAD_DIR)
        ensure_private_directory(self.OUTPUT_DIR)
        ensure_private_directory(Path(self.LOG_FILE).parent)
        ensure_private_directory(Path(self.DATABASE_PATH).parent)
        ensure_private_directory(Path(self.CUSTOM_CONFIG_PATH).parent)

        ensure_private_file(self.LOG_FILE)
        ensure_private_file(self.DATABASE_PATH)
        ensure_private_file(self.CUSTOM_CONFIG_PATH)


settings = Settings()
