"""Minimal packaged API entry point for the high-quality low-memory workflow."""

from __future__ import annotations

import logging
import os
import runpy
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

os.environ.setdefault("DESENSITIZE_MODE", "high_quality_lowmem")
os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("ENABLE_PRIMARY_UIE", "True")
os.environ.setdefault("ENABLE_PRIMARY_NER", "True")
os.environ.setdefault("ENABLE_SECONDARY_NER", "False")
os.environ.setdefault("ENABLE_QWEN_REVIEW", "True")
os.environ.setdefault("ANALYSIS_WORKER_PROCESS", "True")
os.environ.setdefault("ANALYSIS_STAGE_ISOLATION", "True")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


def _maybe_run_packaged_worker() -> bool:
    if len(sys.argv) < 4 or sys.argv[1] != "--worker-module":
        return False

    module_name = sys.argv[2]
    sys.argv = [module_name, *sys.argv[3:]]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    return True


if _maybe_run_packaged_worker():
    raise SystemExit(0)


from app.api import desensitize
from app.core.config import settings
from app.core.runtime_security import ensure_private_file


LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 3


file_handler = RotatingFileHandler(
    settings.LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        file_handler,
        logging.StreamHandler(),
    ],
)

ensure_private_file(settings.LOG_FILE)

logger = logging.getLogger(__name__)


def _frontend_dist_dir() -> Path:
    return Path(settings.FRONTEND_DIST_DIR)


def _frontend_index_path() -> Path:
    return _frontend_dist_dir() / "index.html"


def _frontend_available() -> bool:
    return _frontend_index_path().exists()


def _is_reserved_path(full_path: str) -> bool:
    reserved_paths = {"health", "docs", "redoc", "openapi.json"}
    return (
        full_path in reserved_paths
        or full_path.startswith("api/")
        or full_path.startswith("docs/")
        or full_path.startswith("redoc/")
    )


app = FastAPI(
    title=settings.APP_NAME,
    description="High-quality low-memory contract desensitization service",
    version=settings.APP_VERSION,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(desensitize.router, prefix=settings.API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    if _frontend_available():
        return FileResponse(_frontend_index_path())

    return JSONResponse(
        {
            "message": f"{settings.APP_NAME} API",
            "version": settings.APP_VERSION,
            "health": "/health",
            "client_available": False,
            "profile": "high_quality_lowmem",
        }
    )


@app.get("/health", include_in_schema=False)
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "host": settings.APP_HOST,
        "port": settings.APP_PORT,
        "llm_backend": settings.LLM_BACKEND,
        "llm_model": settings.get_default_llm_model(),
        "client_available": _frontend_available(),
        "packaged": settings.IS_PACKAGED,
        "profile": "high_quality_lowmem",
    }


@app.get("/{full_path:path}", include_in_schema=False)
async def frontend_spa(full_path: str):
    if not full_path or _is_reserved_path(full_path):
        raise HTTPException(status_code=404, detail="Not found")

    if not _frontend_available():
        raise HTTPException(status_code=404, detail="Client assets are not available.")

    dist_dir = _frontend_dist_dir().resolve()
    requested_path = (dist_dir / full_path).resolve()

    try:
        requested_path.relative_to(dist_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc

    if requested_path.is_file():
        return FileResponse(requested_path)

    return FileResponse(_frontend_index_path())


@app.on_event("startup")
async def startup_event():
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Profile: high_quality_lowmem")
    logger.info("Binding host: %s", settings.APP_HOST)
    logger.info("LLM backend: %s", settings.LLM_BACKEND)
    logger.info("Runtime root: %s", settings.RUNTIME_ROOT)
    logger.info("Model root: %s", settings.LOWMEM_MODEL_ROOT)
    logger.info("Client assets: %s", settings.FRONTEND_DIST_DIR)
    logger.info("Client available: %s", _frontend_available())

    from app.core.database import init_db

    init_db()
    ensure_private_file(settings.DATABASE_PATH)
    ensure_private_file(settings.LOG_FILE)
    logger.info("Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down %s", settings.APP_NAME)


if __name__ == "__main__":
    uvicorn.run(
        app if settings.IS_PACKAGED else "main_high_quality:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
