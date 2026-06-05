"""Independent lawyer assistant workflow APIs."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api import desensitize as desensitize_api
from app.core.config import settings
from app.core.runtime_security import ensure_private_file
from app.engine.desensitization_engine import get_engine
from app.processors.document_parser import DocumentParser
from app.schemas.desensitize import (
    AssistantResult,
    AssistantTaskStatus,
)
from app.services.assistant_service import AssistantWorkflowService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])

document_parser = DocumentParser()
assistant_service = AssistantWorkflowService()
assistant_tasks: Dict[str, Dict] = {}
ASSISTANT_FAILURE_DETAIL = "律师辅助分析失败，请稍后重试。"

STAGE_LABELS = {
    "intake": "文书接入与 OCR",
    "classification": "文书分类与范围判断",
    "extract": "案件要素提取",
    "request": "请求事项拆解",
    "finalize": "程序核对与缺口整理",
}


def _task_created_at(task: Dict) -> datetime:
    created_at = task.get("created_at")
    if isinstance(created_at, datetime):
        return created_at
    return datetime.min


def _purge_assistant_task(assistant_id: str) -> None:
    task = assistant_tasks.pop(assistant_id, None)
    if not isinstance(task, dict):
        return
    file_path = task.get("file_path")
    if not file_path:
        return
    try:
        path = Path(file_path).resolve()
        allowed_parent = Path(settings.UPLOAD_DIR).resolve()
        path.relative_to(allowed_parent)
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to remove assistant upload: %s", file_path, exc_info=True)


def _purge_terminal_assistant_tasks(now: Optional[datetime] = None) -> None:
    if not assistant_tasks:
        return
    current_time = now or datetime.now()
    retention = timedelta(hours=max(settings.TASK_RETENTION_HOURS, 1))
    finished: list[tuple[datetime, str]] = []
    expired: list[str] = []
    for assistant_id, task in list(assistant_tasks.items()):
        status = str(task.get("status") or "").strip().lower()
        if status not in {"completed", "failed"}:
            continue
        created_at = _task_created_at(task)
        finished.append((created_at, assistant_id))
        if current_time - created_at > retention:
            expired.append(assistant_id)
    for assistant_id in expired:
        _purge_assistant_task(assistant_id)
    remaining = [(created_at, assistant_id) for created_at, assistant_id in finished if assistant_id in assistant_tasks]
    overflow = len(remaining) - max(settings.MAX_PRESERVED_TASKS, 1)
    if overflow <= 0:
        return
    for _, assistant_id in sorted(remaining, key=lambda item: item[0])[:overflow]:
        _purge_assistant_task(assistant_id)


def _save_upload(assistant_id: str, filename: str, content: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum size is {settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )
    file_path = Path(settings.UPLOAD_DIR) / f"assistant_{assistant_id}{suffix}"
    file_path.write_bytes(content)
    ensure_private_file(file_path)
    return str(file_path), suffix


def _resolve_assistant_model(llm_model: Optional[str]) -> str:
    if settings.is_high_quality_lowmem_mode():
        return llm_model or settings.REVIEW_MODEL
    if settings.LLM_BACKEND.lower() != "ollama":
        return llm_model or settings.get_default_llm_model()

    catalog = desensitize_api._get_model_catalog()
    installed_review_models = [item.name for item in catalog.models if item.installed and item.supports_precision_review]
    selected_model = (llm_model or settings.get_default_review_llm_model(installed_review_models) or "").strip()
    if not selected_model:
        raise HTTPException(
            status_code=400,
            detail="当前未检测到可用于律师辅助的 27B 模型，请先安装 qwen3.5:27b 系列模型。",
        )
    if not settings.is_review_capable_ollama_model(selected_model):
        raise HTTPException(
            status_code=400,
            detail=f"模型 {selected_model} 不属于 27B 协助路线，请选择 qwen3.5:27b 系列模型。",
        )
    selected = next((item for item in catalog.models if item.name == selected_model), None)
    if selected is not None and catalog.service_available and not selected.installed:
        raise HTTPException(status_code=400, detail=f"模型 {selected_model} 尚未安装。")
    return selected_model


def _set_assistant_task_state(
    task: Dict,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    stage_key: Optional[str] = None,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    if status is not None:
        task["status"] = status
    if progress is not None:
        task["progress"] = max(0, min(100, int(progress)))
    if stage_key is not None:
        task["stage_key"] = stage_key
        task["stage_label"] = STAGE_LABELS.get(stage_key, stage_key)
    if message is not None:
        task["message"] = message
    if error_message is not None:
        task["error_message"] = error_message
    task["updated_at"] = datetime.now()

    if stage_key:
        stage_entry = {
            "stage_key": stage_key,
            "stage_label": task.get("stage_label"),
            "progress": task.get("progress"),
            "message": task.get("message"),
            "updated_at": task["updated_at"].isoformat(),
        }
        history = task.setdefault("stage_trace", [])
        if not history or history[-1].get("stage_key") != stage_key or history[-1].get("message") != task.get("message"):
            history.append(stage_entry)


def _assistant_status_response(assistant_id: str, task: Dict) -> AssistantTaskStatus:
    return AssistantTaskStatus(
        assistant_id=assistant_id,
        filename=task.get("filename"),
        status=str(task.get("status") or "queued"),
        progress=max(0, min(100, int(task.get("progress") or 0))),
        stage_key=task.get("stage_key"),
        stage_label=task.get("stage_label"),
        message=task.get("message"),
        error_message=task.get("error_message"),
        created_at=_task_created_at(task),
    )


def _assistant_progress_callback(assistant_id: str):
    progress_bands = {
        "prepare": ("classification", 28, 34),
        "base": ("extract", 34, 58),
        "recall": ("extract", 58, 70),
        "specialized": ("request", 70, 82),
        "finalize": ("finalize", 82, 90),
        "review": ("finalize", 90, 94),
    }

    def _callback(payload: Dict[str, object]) -> None:
        task = assistant_tasks.get(assistant_id)
        if task is None:
            return
        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        mapped = progress_bands.get(stage)
        stage_key = task.get("stage_key") or "extract"
        progress = int(task.get("progress") or 28)
        if mapped is not None:
            stage_key, start, end = mapped
            if total > 0:
                ratio = min(max(current / total, 0.0), 1.0)
                progress = start + int((end - start) * ratio)
            else:
                progress = max(progress, start)
        _set_assistant_task_state(
            task,
            status="processing",
            progress=progress,
            stage_key=stage_key,
            message=str(payload.get("message") or "").strip() or task.get("message") or "正在执行律师辅助分析...",
        )

    return _callback


async def _run_assistant_task(assistant_id: str) -> None:
    task = assistant_tasks.get(assistant_id)
    if task is None:
        return
    try:
        llm_model = str(task["llm_model"])
        _set_assistant_task_state(task, status="processing", progress=10, stage_key="intake", message="正在解析文书并准备 OCR...")
        doc_result = await document_parser.parse(
            task["file_path"],
            use_llm=True,
            llm_model=llm_model,
        )
        text = doc_result["text"]
        structure = doc_result.get("structure")
        metadata = {
            **dict(doc_result.get("metadata") or {}),
            "file_type": str(task.get("suffix") or "").lstrip("."),
        }

        _set_assistant_task_state(task, status="processing", progress=28, stage_key="classification", message="正在执行文书分类与范围判断...")
        engine = get_engine()
        entities = await engine.analyze(
            text,
            use_llm=True,
            use_custom=False,
            llm_model=llm_model,
            progress_callback=_assistant_progress_callback(assistant_id),
        )
        quality_metadata = engine.get_last_quality_metadata()
        metadata.update(desensitize_api._get_llm_analysis_metadata(engine, llm_model))

        _set_assistant_task_state(task, status="processing", progress=94, stage_key="finalize", message="正在整理案件首页与核对清单...")
        result = assistant_service.generate_assistant_result(
            assistant_id=assistant_id,
            filename=str(task["filename"]),
            text=text,
            entities=entities,
            metadata=metadata,
            structure=structure,
            quality_metadata=quality_metadata,
            llm_model=llm_model,
            stage_trace=list(task.get("stage_trace") or []),
        )
        task["result"] = result
        _set_assistant_task_state(task, status="completed", progress=100, stage_key="finalize", message="律师辅助分析完成。")
    except Exception as exc:
        logger.error("Assistant task failed: %s", exc, exc_info=True)
        _set_assistant_task_state(
            task,
            status="failed",
            progress=100,
            stage_key="finalize",
            message=ASSISTANT_FAILURE_DETAIL,
            error_message=ASSISTANT_FAILURE_DETAIL,
        )


@router.post("/upload", response_model=AssistantTaskStatus)
async def upload_for_assistant(
    file: UploadFile = File(...),
    llm_model: Optional[str] = None,
):
    _purge_terminal_assistant_tasks()
    content = await file.read()
    filename = file.filename or "unknown.txt"
    assistant_id = str(uuid.uuid4())
    selected_model = _resolve_assistant_model(llm_model)
    file_path, suffix = _save_upload(assistant_id, filename, content)
    task = {
        "assistant_id": assistant_id,
        "filename": filename,
        "file_path": file_path,
        "suffix": suffix,
        "llm_model": selected_model,
        "status": "queued",
        "progress": 5,
        "stage_key": "intake",
        "stage_label": STAGE_LABELS["intake"],
        "message": "已加入律师辅助队列，正在准备分析任务...",
        "created_at": datetime.now(),
        "stage_trace": [],
    }
    assistant_tasks[assistant_id] = task
    _set_assistant_task_state(task, status="queued", progress=5, stage_key="intake", message=task["message"])
    asyncio.create_task(_run_assistant_task(assistant_id))
    return _assistant_status_response(assistant_id, task)


@router.get("/status/{assistant_id}", response_model=AssistantTaskStatus)
async def get_assistant_status(assistant_id: str):
    task = assistant_tasks.get(assistant_id)
    if task is None:
        raise HTTPException(status_code=404, detail="律师辅助任务不存在。")
    return _assistant_status_response(assistant_id, task)


@router.get("/result/{assistant_id}", response_model=AssistantResult)
async def get_assistant_result(assistant_id: str):
    task = assistant_tasks.get(assistant_id)
    if task is None:
        raise HTTPException(status_code=404, detail="律师辅助任务不存在。")
    status = str(task.get("status") or "").strip().lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or ASSISTANT_FAILURE_DETAIL)
    if status != "completed" or not task.get("result"):
        raise HTTPException(status_code=409, detail="律师辅助结果尚未准备完成，请稍后重试。")
    return AssistantResult(**task["result"])
