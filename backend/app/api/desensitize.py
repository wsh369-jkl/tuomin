"""Desensitization API routes."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.anonymization_strategy import (
    DEFAULT_ANONYMIZATION_STRATEGY,
    get_anonymization_strategy_profile,
)
from app.core.config import settings
from app.core.llm_strategy import get_llm_strategy_profile
from app.core.runtime_probe import (
    detected_ollama_path,
    download_hint,
    installer_hint,
    platform_label,
)
from app.core.runtime_security import ensure_private_file
from app.engine.desensitization_engine import get_engine
from app.processors.document_exporter import DOCX_MIME_TYPE, TEXT_MIME_TYPE, DocumentExporter
from app.processors.document_parser import DocumentParser
from app.schemas.desensitize import (
    AnalyzeResponse,
    DesensitizeRequest,
    DesensitizeResponse,
    Entity,
    LLMModelListResponse,
    LLMModelOption,
    RuntimeStatusResponse,
    TaskStatus,
)
from app.services.ollama_service import OllamaLLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/desensitize", tags=["desensitize"])

document_parser = DocumentParser()
document_exporter = DocumentExporter()
tasks: Dict[str, Dict] = {}
ANALYZE_FAILURE_DETAIL = "文件解析或识别失败，请检查文件内容后重试。"
ANONYMIZE_FAILURE_DETAIL = "脱敏处理失败，请稍后重试或检查当前配置。"
ENGINE_INFO_FAILURE_DETAIL = "运行时信息读取失败，请稍后重试。"


def _task_created_at(task: Dict) -> datetime:
    created_at = task.get("created_at")
    if isinstance(created_at, datetime):
        return created_at
    return datetime.min


def _delete_if_managed(path_value: Optional[str], *, allowed_parent: str) -> None:
    if not path_value:
        return

    path = Path(path_value)
    try:
        resolved_path = path.resolve()
        resolved_parent = Path(allowed_parent).resolve()
        resolved_path.relative_to(resolved_parent)
    except Exception:
        return

    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to remove stale runtime file: %s", path, exc_info=True)


def _purge_task(task_id: str) -> None:
    task = tasks.pop(task_id, None)
    if not isinstance(task, dict):
        return

    _delete_if_managed(task.get("output_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_if_managed(task.get("file_path"), allowed_parent=settings.UPLOAD_DIR)


def _purge_terminal_tasks(now: Optional[datetime] = None) -> None:
    if not tasks:
        return

    current_time = now or datetime.now()
    retention = timedelta(hours=max(settings.TASK_RETENTION_HOURS, 1))
    finished_tasks: list[tuple[datetime, str]] = []
    expired_task_ids: list[str] = []

    for task_id, task in list(tasks.items()):
        if not isinstance(task, dict):
            continue

        status = str(task.get("status") or "").strip().lower()
        if status not in {"completed", "failed"}:
            continue

        created_at = _task_created_at(task)
        finished_tasks.append((created_at, task_id))
        if current_time - created_at > retention:
            expired_task_ids.append(task_id)

    for task_id in expired_task_ids:
        _purge_task(task_id)

    remaining_finished = [
        (created_at, task_id)
        for created_at, task_id in finished_tasks
        if task_id in tasks
    ]
    overflow = len(remaining_finished) - max(settings.MAX_PRESERVED_TASKS, 1)
    if overflow <= 0:
        return

    for _, task_id in sorted(remaining_finished, key=lambda item: item[0])[:overflow]:
        _purge_task(task_id)


def _get_model_catalog() -> LLMModelListResponse:
    default_model = settings.get_default_llm_model()
    default_strategy = get_llm_strategy_profile(default_model)
    if settings.LLM_BACKEND.lower() != "ollama":
        return LLMModelListResponse(
            backend=settings.LLM_BACKEND,
            default_model=default_model,
            service_available=True,
            models=[
                LLMModelOption(
                    name=default_model,
                    installed=True,
                    is_default=True,
                    strategy_key=default_strategy.key,
                    strategy_label=default_strategy.label,
                    strategy_description=default_strategy.description,
                )
            ],
        )

    ollama = OllamaLLMService(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        timeout=settings.OLLAMA_TIMEOUT,
        num_ctx=settings.OLLAMA_NUM_CTX,
    )
    installed_models = set(ollama.list_models()) if ollama.available else set()
    return LLMModelListResponse(
        backend=settings.LLM_BACKEND,
        default_model=default_model,
        service_available=ollama.available,
        models=[
            LLMModelOption(
                name=model_name,
                installed=model_name in installed_models,
                is_default=model_name == default_model,
                strategy_key=get_llm_strategy_profile(model_name).key,
                strategy_label=get_llm_strategy_profile(model_name).label,
                strategy_description=get_llm_strategy_profile(model_name).description,
            )
            for model_name in settings.get_ollama_model_options()
        ],
    )


def _get_runtime_status() -> RuntimeStatusResponse:
    default_model = settings.get_default_llm_model()
    current_platform = platform_label()
    current_backend = settings.LLM_BACKEND.lower()

    if current_backend != "ollama":
        return RuntimeStatusResponse(
            backend=settings.LLM_BACKEND,
            platform=current_platform,
            ready=True,
            ollama_install_detected=False,
            ollama_path=None,
            service_available=True,
            required_model=default_model,
            required_model_installed=True,
            default_model=default_model,
            installer_hint=installer_hint(),
            download_hint=download_hint(default_model),
            recommended_action="当前运行环境已就绪，可直接开始处理文档。",
        )

    catalog = _get_model_catalog()
    ollama_path = detected_ollama_path()
    install_detected = bool(ollama_path)
    selected_model = next((item for item in catalog.models if item.name == default_model), None)
    required_model_installed = bool(selected_model and selected_model.installed)
    ready = catalog.service_available and required_model_installed

    if ready:
        recommended_action = "运行环境已就绪，可直接开始高质量 4B 脱敏处理。"
    elif not install_detected:
        recommended_action = "请先安装 Ollama，再回到客户端完成模型检查。"
    elif not catalog.service_available:
        recommended_action = "已检测到 Ollama，但服务暂不可用。请先打开 Ollama，然后点击重新检测。"
    elif not required_model_installed:
        recommended_action = f"请先下载固定模型 {default_model}，完成后再开始正式处理。"
    else:
        recommended_action = "请先完成运行环境检查，确认 Ollama 和固定模型都已就绪。"

    return RuntimeStatusResponse(
        backend=settings.LLM_BACKEND,
        platform=current_platform,
        ready=ready,
        ollama_install_detected=install_detected,
        ollama_path=ollama_path,
        service_available=catalog.service_available,
        required_model=default_model,
        required_model_installed=required_model_installed,
        default_model=default_model,
        installer_hint=installer_hint(),
        download_hint=download_hint(default_model),
        recommended_action=recommended_action,
    )


def _get_strategy_payload(llm_model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not llm_model:
        return None, None
    strategy = get_llm_strategy_profile(llm_model)
    return strategy.key, strategy.label


def _get_anonymization_strategy_payload(
    strategy_key: Optional[str],
) -> tuple[str, str]:
    strategy = get_anonymization_strategy_profile(strategy_key)
    return strategy.key, strategy.label


def _get_llm_analysis_metadata(engine, llm_model: Optional[str]) -> Dict[str, object]:
    if not llm_model:
        return {}

    recognizer = engine.recognizer_registry.get_recognizer("llm")
    if recognizer is None or not hasattr(recognizer, "get_last_run_metadata"):
        return {}

    try:
        metadata = recognizer.get_last_run_metadata(llm_model)
    except Exception:
        return {}

    document_type = str(metadata.get("document_type", "")).strip()
    if not document_type:
        return {}

    payload: Dict[str, object] = {
        "llm_document_type": document_type,
        "llm_document_type_label": metadata.get("document_type_label") or document_type,
    }
    if metadata.get("document_type_reason"):
        payload["llm_document_type_reason"] = metadata["document_type_reason"]
    if metadata.get("document_type_confidence"):
        payload["llm_document_type_confidence"] = metadata["document_type_confidence"]
    if metadata.get("document_type_source"):
        payload["llm_document_type_source"] = metadata["document_type_source"]
    if metadata.get("engine_strategy"):
        payload["engine_strategy"] = metadata["engine_strategy"]
    if metadata.get("focus_plan"):
        payload["llm_focus_plan"] = metadata["focus_plan"]
    if metadata.get("recall_passes"):
        payload["recall_passes"] = metadata["recall_passes"]
        payload["llm_recall_passes"] = metadata["recall_passes"]
    if metadata.get("specialized_passes"):
        payload["llm_specialized_passes"] = metadata["specialized_passes"]
    if metadata.get("high_risk_blocks"):
        payload["high_risk_blocks"] = metadata["high_risk_blocks"]
    if metadata.get("definition_hints"):
        payload["definition_hints"] = metadata["definition_hints"]
    return payload


def _resolve_requested_llm_model(llm_model: Optional[str]) -> str:
    requested_model = (llm_model or settings.get_default_llm_model()).strip()
    if settings.LLM_BACKEND.lower() != "ollama":
        return requested_model or settings.LLM_MODEL_NAME

    default_model = settings.get_default_llm_model()
    if requested_model and requested_model != default_model:
        logger.info(
            "Ignoring requested Ollama model %s and using stable default %s.",
            requested_model,
            default_model,
        )
    return default_model


def _ensure_model_ready(llm_model: Optional[str]) -> str:
    model_name = _resolve_requested_llm_model(llm_model)
    if settings.LLM_BACKEND.lower() != "ollama":
        return model_name

    catalog = _get_model_catalog()
    if not catalog.service_available:
        return model_name

    selected = next((item for item in catalog.models if item.name == model_name), None)
    if selected is not None and not selected.installed:
        raise HTTPException(
            status_code=400,
            detail=f"模型 {model_name} 尚未下载，请先执行 ollama pull {model_name}",
        )
    return model_name


def _compact_terminal_task(task: Dict) -> None:
    if not isinstance(task, dict):
        return

    text = task.get("text")
    if isinstance(text, str):
        task["text_length"] = len(text)
        task.pop("text", None)

    structure = task.get("structure")
    if isinstance(structure, dict):
        pages = structure.get("pages")
        if isinstance(pages, list):
            task["page_count"] = len(pages)
        task.pop("structure", None)

    entities = task.get("entities")
    if isinstance(entities, list):
        task["entities_count"] = len(entities)
        task.pop("entities", None)

    anonymized_text = task.get("anonymized_text")
    if isinstance(anonymized_text, str):
        task["anonymized_text_length"] = len(anonymized_text)
        task.pop("anonymized_text", None)


async def _ensure_task_document_loaded(task: Dict) -> None:
    if task.get("text") and task.get("structure") is not None:
        return

    doc_result = await document_parser.parse(
        task["file_path"],
        use_llm=bool(task.get("config", {}).get("use_llm", False)),
        llm_model=task.get("config", {}).get("llm_model"),
    )
    task["text"] = doc_result["text"]
    task["structure"] = doc_result.get("structure")
    if not task.get("metadata"):
        task["metadata"] = doc_result.get("metadata", {})


def _serialize_entities(entities) -> list:
    serialized = []
    for entity in entities:
        if hasattr(entity, "model_dump"):
            serialized.append(entity.model_dump())
        elif hasattr(entity, "dict"):
            serialized.append(entity.dict())
        else:
            serialized.append(entity)
    return serialized


def _save_upload(task_id: str, filename: str, content: bytes) -> tuple[str, str]:
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

    file_path = Path(settings.UPLOAD_DIR) / f"{task_id}{suffix}"
    file_path.write_bytes(content)
    ensure_private_file(file_path)
    return str(file_path), suffix


def _media_type_for_path(output_path: str) -> str:
    suffix = Path(output_path).suffix.lower()
    if suffix == ".docx":
        return DOCX_MIME_TYPE
    return TEXT_MIME_TYPE


def _download_name(filename: str, output_path: str) -> str:
    return f"{Path(filename).stem}_anonymized{Path(output_path).suffix}"


@router.post("/upload", response_model=AnalyzeResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    use_llm: bool = True,
    use_custom: bool = True,
    llm_model: Optional[str] = None,
    anonymization_strategy: Optional[str] = None,
):
    logger.info("Received upload request")
    _purge_terminal_tasks()

    task_id = str(uuid.uuid4())
    start_time = perf_counter()
    content = await file.read()
    filename = file.filename or "unknown.txt"
    selected_llm_model = _ensure_model_ready(llm_model) if use_llm else None
    strategy_key, strategy_label = _get_strategy_payload(selected_llm_model)
    anonymization_strategy_key, anonymization_strategy_label = _get_anonymization_strategy_payload(
        anonymization_strategy or DEFAULT_ANONYMIZATION_STRATEGY
    )
    saved_file_path: Optional[str] = None

    try:
        file_path, suffix = _save_upload(task_id, filename, content)
        saved_file_path = file_path
        doc_result = await document_parser.parse(
            file_path,
            use_llm=use_llm,
            llm_model=selected_llm_model,
        )
        metadata = {**doc_result["metadata"], "file_type": suffix.lstrip(".")}
        text = doc_result["text"]
        structure = doc_result.get("structure")

        engine = get_engine()
        entities = await engine.analyze(
            text,
            use_llm=use_llm,
            use_custom=use_custom,
            llm_model=selected_llm_model,
        )
        metadata.update(_get_llm_analysis_metadata(engine, selected_llm_model))
        statistics = engine.get_entity_statistics(entities)
        analysis_time = round(perf_counter() - start_time, 4)

        task_data = {
            "task_id": task_id,
            "filename": filename,
            "file_path": file_path,
            "text": text,
            "entities": entities,
            "metadata": metadata,
            "structure": structure,
            "status": "processing",
            "created_at": datetime.now(),
            "analysis_time": analysis_time,
            "config": {
                "use_llm": use_llm,
                "use_custom": use_custom,
                "llm_model": selected_llm_model,
                "llm_strategy": strategy_key,
                "anonymization_strategy": anonymization_strategy_key,
            },
        }
        tasks[task_id] = task_data

        return AnalyzeResponse(
            task_id=task_id,
            filename=filename,
            text=text,
            entities=[Entity(**entity) for entity in entities],
            statistics=statistics,
            metadata=metadata,
            llm_model=selected_llm_model,
            llm_strategy=strategy_key,
            llm_strategy_label=strategy_label,
            anonymization_strategy=anonymization_strategy_key,
            anonymization_strategy_label=anonymization_strategy_label,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _delete_if_managed(saved_file_path, allowed_parent=settings.UPLOAD_DIR)
        logger.error("File analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=ANALYZE_FAILURE_DETAIL)


@router.post("/process", response_model=DesensitizeResponse)
@router.post("/anonymize", response_model=DesensitizeResponse)
async def anonymize_text(request: DesensitizeRequest):
    logger.info("Received process request: %s", request.task_id)

    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task was not found. Please upload the file again.")

    task = tasks[request.task_id]
    await _ensure_task_document_loaded(task)
    text = task["text"]
    start_time = perf_counter()
    use_llm = bool(task.get("config", {}).get("use_llm", False))
    selected_llm_model = (
        _ensure_model_ready(task.get("config", {}).get("llm_model") or request.llm_model)
        if use_llm
        else None
    )
    strategy_key, strategy_label = _get_strategy_payload(selected_llm_model)
    anonymization_strategy_key, anonymization_strategy_label = _get_anonymization_strategy_payload(
        request.anonymization_strategy
        or task.get("config", {}).get("anonymization_strategy")
        or DEFAULT_ANONYMIZATION_STRATEGY
    )

    try:
        entities = _serialize_entities(request.entities)
        engine = get_engine()
        entities = await engine.prepare_entities_for_anonymization(
            text=text,
            entities=entities,
            use_llm=use_llm,
            operator_config=request.config,
            llm_model=selected_llm_model,
            anonymization_strategy=anonymization_strategy_key,
        )
        quality_metadata = engine.get_last_quality_metadata()
        anonymized_text = await engine.anonymize(
            text=text,
            entities=entities,
            operator_config=request.config,
        )

        export_result = document_exporter.export(
            task_id=request.task_id,
            source_path=task["file_path"],
            original_filename=task["filename"],
            source_text=text,
            source_metadata=task.get("metadata"),
            source_structure=task.get("structure"),
            entities=entities,
            anonymized_text=anonymized_text,
            operator_config=request.config,
        )

        total_processing_time = round(
            task.get("analysis_time", 0) + (perf_counter() - start_time), 4
        )

        task["status"] = "completed"
        task["entities"] = entities
        task["anonymized_text"] = anonymized_text
        task["output_path"] = export_result["output_path"]
        task["output_filename"] = export_result["download_name"]
        task["output_file_type"] = export_result["file_type"]
        task["output_media_type"] = export_result["media_type"]
        task["preserves_format"] = export_result["preserves_format"]
        quality_warning = None
        if quality_metadata and not quality_metadata.get("quality_gate_passed", True):
            quality_warning = (
                "质量闸检测到仍有残留命中或一致性问题"
                + (
                    f"：{quality_metadata.get('quality_gate_reason')}"
                    if quality_metadata.get("quality_gate_reason")
                    else ""
                )
            )
        warning_message = export_result["warning"]
        if quality_warning:
            warning_message = f"{warning_message} | {quality_warning}" if warning_message else quality_warning
        task["export_warning"] = warning_message
        task["processing_time"] = total_processing_time
        task["config"]["anonymization_strategy"] = anonymization_strategy_key
        task["quality_metadata"] = quality_metadata

        response_metadata = {
            **(task.get("metadata") or {}),
            **(quality_metadata or {}),
        }

        _compact_terminal_task(task)

        return DesensitizeResponse(
            task_id=request.task_id,
            status="completed",
            anonymized_text=anonymized_text,
            entities=[Entity(**entity) for entity in entities],
            metadata=response_metadata,
            download_url=f"/api/v1/desensitize/download/{request.task_id}",
            output_filename=export_result["download_name"],
            output_file_type=export_result["file_type"],
            preserves_format=bool(export_result["preserves_format"]),
            llm_assisted=use_llm,
            llm_model=selected_llm_model,
            llm_strategy=strategy_key,
            llm_strategy_label=strategy_label,
            anonymization_strategy=anonymization_strategy_key,
            anonymization_strategy_label=anonymization_strategy_label,
            warning=warning_message,
            message="Desensitization completed.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Desensitization failed: %s", exc, exc_info=True)
        task["status"] = "failed"
        task["error_message"] = ANONYMIZE_FAILURE_DETAIL
        raise HTTPException(status_code=500, detail=ANONYMIZE_FAILURE_DETAIL)


@router.get("/download/{task_id}")
async def download_result(task_id: str):
    logger.info("Received download request: %s", task_id)
    _purge_terminal_tasks()

    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")

    filename = task["filename"]
    output_path = task.get("output_path")
    download_name = task.get("output_filename")
    media_type = task.get("output_media_type")

    if not output_path or not filename:
        raise HTTPException(status_code=400, detail="The task has not been exported yet.")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="The exported file no longer exists.")

    return FileResponse(
        output_path,
        filename=download_name or _download_name(filename, output_path),
        media_type=media_type or _media_type_for_path(output_path),
    )


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    _purge_terminal_tasks()
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")

    status = task["status"]
    progress = 100 if status == "completed" else 50
    return TaskStatus(
        task_id=task_id,
        status=status,
        progress=progress,
        message=f"Task status: {status}",
        created_at=task["created_at"],
    )


@router.get("/health")
async def health_check():
    try:
        _purge_terminal_tasks()
        engine = get_engine()
        return {
            "status": "healthy",
            "engine": engine.get_engine_info(),
            "tasks_count": len(tasks),
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc, exc_info=True)
        return {
            "status": "unhealthy",
            "error": "health_check_failed",
        }


@router.get("/models", response_model=LLMModelListResponse)
async def get_available_models():
    return _get_model_catalog()


@router.get("/runtime-status", response_model=RuntimeStatusResponse)
async def get_runtime_status():
    return _get_runtime_status()


@router.get("/info")
async def get_engine_info():
    try:
        engine = get_engine()
        return engine.get_engine_info()
    except Exception as exc:
        logger.error("Failed to get engine info: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=ENGINE_INFO_FAILURE_DETAIL)

