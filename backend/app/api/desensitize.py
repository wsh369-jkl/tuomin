"""Desensitization API routes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.anonymization_strategy import (
    DEFAULT_ANONYMIZATION_STRATEGY,
    get_anonymization_strategy_profile,
)
from app.core.config import (
    SUPPORTED_DESENSITIZE_MODES,
    desensitize_mode_context,
    is_supported_desensitize_mode,
    normalize_desensitize_mode,
    settings,
)
from app.core.document_workflow import classify_document_workflow, resolve_large_document_page_count
from app.core.runtime_probe import (
    installer_hint,
    platform_label,
)
from app.core.runtime_security import ensure_private_directory, ensure_private_file
from app.engine.desensitization_engine import get_engine
from app.processors.document_exporter import (
    DOCX_MIME_TYPE,
    PDF_MIME_TYPE,
    TEXT_MIME_TYPE,
    DocumentExporter,
)
from app.processors.document_parser import DocumentParser
from app.schemas.desensitize import (
    AnalyzeResponse,
    BatchFileItem,
    BatchResult,
    BatchTaskStatus,
    DesensitizeRequest,
    DesensitizeResponse,
    Entity,
    LLMModelListResponse,
    LLMModelOption,
    PageSessionRequest,
    PageSessionResponse,
    RuntimeStatusResponse,
    TaskStatus,
)
from app.services.lowmem_model_assets import (
    all_assets,
    model_installed,
    primary_models_ready,
)
from app.services.review_input_compactor import compact_review_worker_payload_entities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/desensitize", tags=["desensitize"])

document_parser = DocumentParser()
document_exporter = DocumentExporter()
tasks: Dict[str, Dict] = {}
batch_tasks: Dict[str, Dict] = {}
analysis_task_runners: Dict[str, asyncio.Task] = {}
process_task_runners: Dict[str, asyncio.Task] = {}
batch_task_runners: Dict[str, asyncio.Task] = {}
active_request_tasks: Dict[str, asyncio.Task] = {}
page_sessions: Dict[str, Dict[str, Any]] = {}
page_session_watchdogs: Dict[str, asyncio.Task] = {}


def _query_bool_override(request: Request, name: str, current: bool) -> bool:
    if name not in request.query_params:
        return current
    normalized = str(request.query_params.get(name) or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return current


def _query_str_override(request: Request, name: str, current: Optional[str]) -> Optional[str]:
    if name not in request.query_params:
        return current
    value = str(request.query_params.get(name) or "").strip()
    return value or None
analysis_worker_python_cache: Optional[str] = None
analysis_worker_semaphore = asyncio.Semaphore(1)
process_worker_semaphore = asyncio.Semaphore(1)
pdf_normalize_worker_semaphore = asyncio.Semaphore(1)
ANALYZE_FAILURE_DETAIL = "文件解析或识别失败，请检查文件内容后重试。"
ANONYMIZE_FAILURE_DETAIL = "脱敏处理失败，请稍后重试或检查当前配置。"
BATCH_FAILURE_DETAIL = "批量脱敏失败，请稍后重试或缩小文件夹范围。"
ENGINE_INFO_FAILURE_DETAIL = "运行时信息读取失败，请稍后重试。"
PAGE_SESSION_STOP_DETAIL = "前端页面已关闭，后台任务已自动停止。"
PAGE_SESSION_TIMEOUT_DETAIL = "前端页面已断开连接，后台任务继续运行，可稍后查看任务状态。"
TASK_STATE_DIR = Path(settings.RUNTIME_ROOT) / "task_state"
TASK_STATE_EXCLUDED_KEYS = {
    "process_cancel_event",
}


def _resolve_desensitize_mode(desensitize_mode: Optional[str]) -> str:
    normalized = normalize_desensitize_mode(desensitize_mode) or settings.get_effective_desensitize_mode()
    if not is_supported_desensitize_mode(normalized):
        supported = ", ".join(sorted(SUPPORTED_DESENSITIZE_MODES))
        raise HTTPException(status_code=400, detail=f"不支持的处理线路：{desensitize_mode}。可选值：{supported}")
    return normalized


def _task_desensitize_mode(task: Dict[str, Any]) -> str:
    config = task.get("config") if isinstance(task, dict) else {}
    if isinstance(config, dict):
        return _resolve_desensitize_mode(config.get("desensitize_mode"))
    return _resolve_desensitize_mode(None)


def _should_use_analysis_worker(use_llm: bool) -> bool:
    return bool(
        use_llm
        and settings.is_high_quality_desensitize_mode()
        and bool(settings.ANALYSIS_WORKER_PROCESS)
    )


def _resolve_analysis_workflow(
    *,
    text: str,
    use_llm: bool,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    document_workflow = classify_document_workflow(
        text=text,
        source_metadata=source_metadata,
        source_structure=source_structure,
    )
    uses_analysis_worker = _should_use_analysis_worker(use_llm)
    stage_isolation = bool(uses_analysis_worker and settings.ANALYSIS_STAGE_ISOLATION)
    if stage_isolation and document_workflow["enabled"]:
        mode = "large_document_pre_routed"
    elif stage_isolation:
        mode = "stage_isolated_review"
    elif uses_analysis_worker:
        mode = "single_worker_full_analysis"
    else:
        mode = "in_process_analysis"
    return {
        "mode": mode,
        "uses_analysis_worker": uses_analysis_worker,
        "stage_isolation": stage_isolation,
        "document_workflow": document_workflow,
    }


def _analysis_workflow_metadata(analysis_workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    workflow = analysis_workflow or {}
    document_workflow = workflow.get("document_workflow") if isinstance(workflow, dict) else {}
    if not isinstance(document_workflow, dict):
        document_workflow = {}
    return {
        "analysis_workflow_mode": str(workflow.get("mode") or "default"),
        "analysis_large_document_mode": bool(document_workflow.get("enabled")),
        "analysis_large_document_policy": {
            "enabled": bool(document_workflow.get("enabled")),
            "page_count": int(document_workflow.get("page_count") or 0),
            "text_length": int(document_workflow.get("text_length") or 0),
            "page_threshold": int(document_workflow.get("page_threshold") or 0),
            "text_threshold": int(document_workflow.get("text_threshold") or 0),
            "triggered_by_page_count": bool(document_workflow.get("triggered_by_page_count")),
            "triggered_by_text_length": bool(document_workflow.get("triggered_by_text_length")),
        },
    }


def _large_document_parent_process_deferred_analysis_result(
    analysis_workflow: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = _analysis_workflow_metadata(analysis_workflow)
    metadata.update(
        {
            "large_document_mode": True,
            "execution_mode": "large_document_parent_process_deferred",
            "large_document_execution_mode": "defer_to_process_worker_grouped_default_line",
            "analysis_stage_review_skipped": "large_document_parent_process",
            "review_configured": False,
            "review_dispatched": False,
            "review_started": False,
            "review_completed": False,
            "_large_document_pre_routed": True,
        }
    )
    return {
        "entities": [],
        "statistics": {},
        "analysis_metadata": metadata,
    }


def _sync_task_page_count_hint(task: Dict[str, Any], page_count_hint: int) -> None:
    if not isinstance(task, dict):
        return
    try:
        page_count = int(page_count_hint or 0)
    except (TypeError, ValueError):
        return
    if page_count <= 0:
        return

    existing_task_page_count = 0
    try:
        existing_task_page_count = int(task.get("page_count", 0) or 0)
    except (TypeError, ValueError):
        existing_task_page_count = 0
    task["page_count"] = max(page_count, existing_task_page_count)

    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        task["metadata"] = metadata

    for key in ("pages", "page_count"):
        existing = 0
        try:
            existing = int(metadata.get(key, 0) or 0)
        except (TypeError, ValueError):
            existing = 0
        metadata[key] = max(page_count, existing)

    structure = task.get("structure")
    if isinstance(structure, dict):
        structure_page_count = resolve_large_document_page_count(
            source_metadata=metadata,
            source_structure=structure,
        )
        structure["page_count"] = max(page_count, int(structure_page_count or 0), 1)


def _merge_task_document_metadata(task: Dict[str, Any], *, doc_result: Dict[str, Any]) -> None:
    if not isinstance(task, dict):
        return
    doc_metadata = dict(doc_result.get("metadata") or {})
    existing_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    merged_metadata = {
        **existing_metadata,
        **doc_metadata,
    }
    task["metadata"] = merged_metadata

    normalized_file_path = str(merged_metadata.get("normalized_file_path") or "").strip()
    if normalized_file_path:
        task["normalized_file_path"] = normalized_file_path

    page_count_hint = resolve_large_document_page_count(
        source_metadata=merged_metadata,
        source_structure=task.get("structure") if isinstance(task.get("structure"), dict) else None,
    )
    if page_count_hint > 0:
        _sync_task_page_count_hint(task, page_count_hint)


def _entity_statistics(entities: List[Dict]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for entity in entities:
        entity_type = str(entity.get("type") or "")
        if not entity_type:
            continue
        bucket = stats.setdefault(entity_type, {"count": 0, "examples": []})
        bucket["count"] = int(bucket["count"]) + 1
        examples = bucket.setdefault("examples", [])
        if isinstance(examples, list) and len(examples) < 3:
            examples.append(entity.get("text"))
    return stats


def _ensure_task_state_dir() -> Path:
    TASK_STATE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_private_directory(TASK_STATE_DIR)
    return TASK_STATE_DIR


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        safe_dict: Dict[str, Any] = {}
        for key, val in value.items():
            key_text = str(key)
            if key_text in TASK_STATE_EXCLUDED_KEYS:
                continue
            safe_dict[key_text] = _json_safe(val)
        return safe_dict
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value)]
    return repr(value)


def _task_state_path(task_id: str) -> Path:
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "").strip()) or "unknown-task"
    return _ensure_task_state_dir() / f"{safe_task_id}.json"


def _persist_task_state(task_id: str, task: Dict[str, Any]) -> None:
    if not isinstance(task, dict):
        return
    snapshot = {
        "task_id": task_id,
        "task": _json_safe(task),
    }
    path = _task_state_path(task_id)
    try:
        path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        ensure_private_file(path)
    except Exception:
        logger.warning("Failed to persist task state for %s", task_id, exc_info=True)


def _hydrate_task(task: Dict[str, Any]) -> Dict[str, Any]:
    hydrated = dict(task)
    for key in ("created_at", "updated_at", "page_session_detached_at"):
        value = hydrated.get(key)
        if isinstance(value, str):
            try:
                hydrated[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
    return hydrated


def _restore_task_state(task_id: str) -> Optional[Dict[str, Any]]:
    path = _task_state_path(task_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read persisted task state: %s", path, exc_info=True)
        return None
    task = payload.get("task")
    if not isinstance(task, dict):
        return None
    restored = _hydrate_task(task)
    _sync_task_page_count_hint(
        restored,
        resolve_large_document_page_count(
            source_metadata=restored.get("metadata") if isinstance(restored.get("metadata"), dict) else None,
            source_structure=restored.get("structure") if isinstance(restored.get("structure"), dict) else None,
        ),
    )
    tasks[task_id] = restored
    return restored


def _get_task(task_id: str) -> Optional[Dict[str, Any]]:
    task = tasks.get(task_id)
    if task is not None:
        return task
    return _restore_task_state(task_id)


def _delete_task_state(task_id: str) -> None:
    try:
        _task_state_path(task_id).unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to delete persisted task state for %s", task_id, exc_info=True)


def _process_rss_kib(pid: int) -> int:
    if pid <= 0:
        return 0
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return 0
    try:
        return int(result.stdout.strip() or "0")
    except ValueError:
        return 0


def _process_group_rss_kib(pgid: int) -> int:
    if pgid <= 0:
        return 0
    try:
        result = subprocess.run(
            ["ps", "-axo", "pgid=,rss="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return 0

    total = 0
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            row_pgid = int(parts[0])
            row_rss = int(parts[1])
        except ValueError:
            continue
        if row_pgid == pgid:
            total += row_rss
    return total


def _worker_rss_kib(process: subprocess.Popen[str]) -> int:
    if os.name == "posix":
        group_rss = _process_group_rss_kib(process.pid)
        if group_rss:
            return group_rss
    return _process_rss_kib(process.pid)


def _terminate_worker_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    def _signal(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            return
        except Exception:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    _signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    _signal(signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def _memory_value_to_mib(value: str, unit: str) -> Optional[float]:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    normalized_unit = str(unit or "").strip().upper()
    factors = {
        "B": 1 / (1024 * 1024),
        "KB": 1 / 1024,
        "KIB": 1 / 1024,
        "MB": 1.0,
        "MIB": 1.0,
        "GB": 1024.0,
        "GIB": 1024.0,
    }
    factor = factors.get(normalized_unit)
    if factor is None:
        return None
    return numeric_value * factor


def _parse_footprint_mib(output: str) -> Optional[float]:
    if not output:
        return None

    preferred = re.search(
        r"phys_footprint(?:_peak)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)",
        output,
        re.IGNORECASE,
    )
    fallback = re.search(
        r"\bFootprint\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)",
        output,
        re.IGNORECASE,
    )
    match = preferred or fallback
    if not match:
        return None
    return _memory_value_to_mib(match.group(1), match.group(2))


def _process_footprint_mib(pid: int) -> Optional[float]:
    """Return macOS physical footprint in MiB when available."""
    if pid <= 0 or sys.platform != "darwin":
        return None

    footprint_bin = "/usr/bin/footprint" if Path("/usr/bin/footprint").exists() else shutil.which("footprint")
    if not footprint_bin:
        return None

    try:
        result = subprocess.run(
            [footprint_bin, "-summary", "-pid", str(pid)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=4,
            check=False,
        )
    except Exception:
        return None

    return _parse_footprint_mib(f"{result.stdout}\n{result.stderr}")


def _numeric_memory_values(*values: object) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def _peak_memory_mib_from_worker(worker: Dict[str, Any]) -> tuple[Optional[float], str]:
    footprint_peak = worker.get("peak_footprint_mib")
    if isinstance(footprint_peak, (int, float)):
        return float(footprint_peak), "macos_footprint"

    rss_peak = worker.get("peak_rss_mib")
    if isinstance(rss_peak, (int, float)):
        return float(rss_peak), "rss"

    return None, "unavailable"


def _peak_memory_mib_from_workers(*workers: Dict[str, Any]) -> tuple[Optional[float], str]:
    footprint_peaks = _numeric_memory_values(*(worker.get("peak_footprint_mib") for worker in workers))
    if footprint_peaks:
        return round(max(footprint_peaks), 1), "macos_footprint"

    rss_peaks = _numeric_memory_values(*(worker.get("peak_rss_mib") for worker in workers))
    if rss_peaks:
        return round(max(rss_peaks), 1), "rss"

    return None, "unavailable"


def _analysis_worker_temp_file(prefix: str) -> Path:
    runtime_root = Path(settings.RUNTIME_ROOT)
    runtime_root.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=prefix,
        suffix=".json",
        dir=runtime_root,
        delete=False,
    )
    handle.close()
    path = Path(handle.name)
    ensure_private_file(path)
    return path


def _analysis_worker_probe_code() -> str:
    lines = [
        "import transformers",
        "from packaging.version import Version",
        "assert Version(transformers.__version__) >= Version('5.0.0'), transformers.__version__",
    ]
    if settings.is_high_quality_desensitize_mode() and settings.ENABLE_QWEN_REVIEW:
        review_backend = str(settings.REVIEW_BACKEND or "").strip().lower()
        fallback_backend = str(settings.REVIEW_MODEL_FALLBACK_BACKEND or "").strip().lower()
        if model_installed(settings.REVIEW_MODEL) and review_backend == "mlx":
            lines.append("import mlx_lm")
        elif model_installed(settings.REVIEW_MODEL_FALLBACK) and fallback_backend in {"llama_cpp", "llamacpp", "gguf"}:
            lines.append("import llama_cpp")
    lines.append("print('ok')")
    return "\n".join(lines)


def _python_can_run_analysis_worker(python_path: str) -> bool:
    try:
        result = subprocess.run(
            [
                python_path,
                "-c",
                _analysis_worker_probe_code(),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=max(10, int(settings.ANALYSIS_WORKER_PYTHON_CHECK_TIMEOUT)),
            check=False,
        )
    except Exception as exc:
        logger.warning("Analysis worker Python probe failed for %s: %s", python_path, exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "Analysis worker Python probe failed for %s: rc=%s stderr=%s",
            python_path,
            result.returncode,
            result.stderr[-1000:],
        )
        return False
    return result.returncode == 0


def _analysis_worker_python() -> str:
    global analysis_worker_python_cache
    if analysis_worker_python_cache and _python_can_run_analysis_worker(analysis_worker_python_cache):
        return analysis_worker_python_cache

    candidates: list[str] = []
    configured = str(settings.ANALYSIS_WORKER_PYTHON or os.getenv("ANALYSIS_WORKER_PYTHON") or "").strip()
    if configured:
        candidates.append(configured)
    candidates.append(sys.executable)
    found_python3 = shutil.which("python3")
    if found_python3:
        candidates.append(found_python3)
    candidates.append("/Library/Frameworks/Python.framework/Versions/3.14/bin/python3")

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists() and _python_can_run_analysis_worker(candidate):
            analysis_worker_python_cache = candidate
            return candidate

    raise RuntimeError(
        "analysis_worker_python_missing_dependencies: "
        "未找到同时安装 transformers 和当前精审后端依赖的 Python。"
    )


def _run_worker_process_blocking(
    *,
    module_name: str,
    payload: Dict[str, Any],
    input_prefix: str,
    output_prefix: str,
    env_overrides: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
    progress_callback=None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    input_path = _analysis_worker_temp_file(input_prefix)
    output_path = _analysis_worker_temp_file(output_prefix)
    progress_path = _analysis_worker_temp_file(f"{input_prefix}progress-")
    worker_payload = dict(payload)
    worker_payload["progress_path"] = str(progress_path)
    input_path.write_text(json.dumps(worker_payload, ensure_ascii=False), encoding="utf-8")
    ensure_private_file(input_path)

    if settings.IS_PACKAGED:
        command = [
            sys.executable,
            "--worker-module",
            module_name,
            str(input_path),
            str(output_path),
        ]
    else:
        worker_python = _analysis_worker_python()
        command = [
            worker_python,
            "-m",
            module_name,
            str(input_path),
            str(output_path),
        ]
    child_env = os.environ.copy()
    child_env["DESENSITIZE_MODE"] = settings.get_effective_desensitize_mode()
    for key, value in (env_overrides or {}).items():
        child_env[str(key)] = str(value)
    existing_pythonpath = str(child_env.get("PYTHONPATH") or "").strip()
    child_env["PYTHONPATH"] = (
        str(Path(settings.RUNTIME_ROOT))
        if not existing_pythonpath
        else f"{settings.RUNTIME_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    start_time = perf_counter()
    process: subprocess.Popen[str] | None = None
    peak_rss_kib = 0
    peak_footprint_mib: Optional[float] = None
    next_footprint_sample_at = 0.0
    last_progress_mtime = 0.0

    try:
        logger.info("Starting one-shot worker: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            cwd=settings.RUNTIME_ROOT,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        timeout_value = int(timeout_seconds) if timeout_seconds is not None else int(settings.ANALYSIS_WORKER_TIMEOUT)
        deadline = None if timeout_value <= 0 else time.time() + max(30, timeout_value)
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                _terminate_worker_process_tree(process)
                stdout, stderr = process.communicate(timeout=5)
                raise RuntimeError(
                    f"{module_name}_cancelled pid={process.pid} stdout={stdout[-500:]} stderr={stderr[-1000:]}"
                )
            peak_rss_kib = max(peak_rss_kib, _worker_rss_kib(process))
            now = time.monotonic()
            if now >= next_footprint_sample_at:
                footprint_mib = _process_footprint_mib(process.pid)
                if isinstance(footprint_mib, (int, float)):
                    peak_footprint_mib = max(peak_footprint_mib or 0, float(footprint_mib))
                next_footprint_sample_at = now + 6.0
            if progress_callback is not None and progress_path.exists():
                try:
                    mtime = progress_path.stat().st_mtime
                    if mtime > last_progress_mtime:
                        last_progress_mtime = mtime
                        progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
                        if isinstance(progress_payload, dict):
                            progress_callback(progress_payload)
                except Exception:
                    logger.debug("Failed to read worker progress file: %s", progress_path, exc_info=True)
            if deadline is not None and time.time() > deadline:
                _terminate_worker_process_tree(process)
                stdout, stderr = process.communicate(timeout=5)
                raise TimeoutError(
                    f"{module_name}_timeout pid={process.pid} stdout={stdout[-500:]} stderr={stderr[-1000:]}"
                )
            time.sleep(0.25)

        peak_rss_kib = max(peak_rss_kib, _worker_rss_kib(process))
        stdout, stderr = process.communicate(timeout=5)
        if process.returncode != 0:
            details = ""
            if output_path.exists():
                try:
                    details = output_path.read_text(encoding="utf-8")[-2000:]
                except Exception:
                    details = ""
            raise RuntimeError(
                f"{module_name}_failed rc={process.returncode} stdout={stdout[-500:]} "
                f"stderr={stderr[-1000:]} details={details}"
            )

        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError(f"{module_name}_invalid_result:{result!r}")

        result["_worker_process"] = {
            "module": module_name,
            "pid": process.pid,
            "seconds": round(perf_counter() - start_time, 4),
            "peak_rss_mib": round(peak_rss_kib / 1024, 1) if peak_rss_kib else None,
            "peak_footprint_mib": round(peak_footprint_mib, 1) if peak_footprint_mib else None,
        }
        return result
    finally:
        if process is not None and process.poll() is None:
            _terminate_worker_process_tree(process)
        for path in (input_path, output_path, progress_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove analysis worker temp file: %s", path, exc_info=True)


def _run_single_analysis_worker_blocking(
    *,
    text: str,
    entities: Optional[List[str]],
    use_llm: bool,
    use_custom: bool,
    llm_model: Optional[str],
    anonymization_strategy: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
    analysis_workflow: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "base",
                "current": 0,
                "total": 1,
                "message": "正在执行规则识别与低内存主识别...",
            }
        )
    result = _run_worker_process_blocking(
        module_name="app.workers.analysis_worker",
        payload={
            "text": text,
            "entities": entities,
            "use_llm": use_llm,
            "use_custom": use_custom,
            "llm_model": llm_model,
            "anonymization_strategy": anonymization_strategy,
            "source_metadata": source_metadata or {},
            "source_structure": source_structure,
            "desensitize_mode": settings.get_effective_desensitize_mode(),
        },
        input_prefix="analysis-worker-in-",
        output_prefix="analysis-worker-out-",
    )
    worker = dict(result.get("_worker_process") or {})
    worker_memory_peak_mib, worker_memory_peak_source = _peak_memory_mib_from_worker(worker)

    metadata = dict(result.get("analysis_metadata") or {})
    metadata.update(
        {
            **_analysis_workflow_metadata(analysis_workflow),
            "analysis_worker_process": True,
            "analysis_stage_isolation": False,
            "analysis_worker_pid": worker.get("pid"),
            "analysis_worker_seconds": worker.get("seconds"),
            "analysis_worker_peak_rss_mib": worker.get("peak_rss_mib"),
            "analysis_worker_peak_footprint_mib": worker.get("peak_footprint_mib"),
            "analysis_worker_memory_peak_mib": round(worker_memory_peak_mib, 1)
            if worker_memory_peak_mib is not None
            else None,
            "analysis_worker_memory_peak_source": worker_memory_peak_source,
        }
    )
    return {
        "entities": result.get("entities") or [],
        "statistics": result.get("statistics") or {},
        "analysis_metadata": metadata,
    }


def _run_stage_isolated_analysis_worker_blocking(
    *,
    text: str,
    entities: Optional[List[str]],
    use_llm: bool,
    use_custom: bool,
    llm_model: Optional[str],
    anonymization_strategy: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
    analysis_workflow: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    start_time = perf_counter()
    workflow_metadata = _analysis_workflow_metadata(analysis_workflow)
    if str((analysis_workflow or {}).get("mode") or "").strip() == "large_document_pre_routed":
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "large_document_parent_process",
                    "current": 0,
                    "total": 1,
                    "message": "已识别为大文件模式，跳过旧分析线，等待父流程裁切分组处理...",
                }
            )
        return _large_document_parent_process_deferred_analysis_result(analysis_workflow)

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "base",
                "current": 0,
                "total": 1,
                "message": "正在执行规则识别与低内存主识别...",
            }
        )
    primary_result = _run_worker_process_blocking(
        module_name="app.workers.analysis_worker",
        payload={
            "text": text,
            "entities": entities,
            "use_llm": use_llm,
            "use_custom": use_custom,
            "llm_model": llm_model,
            "anonymization_strategy": anonymization_strategy,
            "source_metadata": source_metadata or {},
            "source_structure": source_structure,
            "stage_mode": "primary_review_surface",
            "analysis_workflow_mode": str((analysis_workflow or {}).get("mode") or "").strip(),
            "desensitize_mode": settings.get_effective_desensitize_mode(),
        },
        input_prefix="analysis-primary-in-",
        output_prefix="analysis-primary-out-",
        env_overrides={"ENABLE_QWEN_REVIEW": "False"},
    )
    primary_worker = dict(primary_result.get("_worker_process") or {})
    primary_memory_peak_mib, primary_memory_peak_source = _peak_memory_mib_from_worker(primary_worker)
    primary_entities = primary_result.get("entities") or []
    primary_review_entities = primary_result.get("review_entities") or []
    primary_metadata = dict(primary_result.get("analysis_metadata") or {})
    large_document_pre_routed = str((analysis_workflow or {}).get("mode") or "").strip() == "large_document_pre_routed"

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "deep_review",
                "current": 0,
                "total": 1,
                "message": (
                    "已识别为大文件模式，跳过前置片段审查，后续进入分页归并..."
                    if large_document_pre_routed
                    else "正在调度高风险片段审查..."
                ),
            }
        )

    if large_document_pre_routed or not settings.ENABLE_QWEN_REVIEW:
        metadata = dict(primary_metadata)
        metadata.update(
            {
                **workflow_metadata,
                "analysis_worker_process": True,
                "analysis_stage_isolation": True,
                "review_configured": bool(settings.ENABLE_QWEN_REVIEW) if large_document_pre_routed else False,
                "review_dispatched": False,
                "review_started": False,
                "review_completed": False,
                "analysis_stage_review_skipped": (
                    "large_document_pre_routed" if large_document_pre_routed else "review_disabled"
                ),
                "primary_worker_pid": primary_worker.get("pid"),
                "primary_worker_seconds": primary_worker.get("seconds"),
                "primary_worker_peak_rss_mib": primary_worker.get("peak_rss_mib"),
                "primary_worker_peak_footprint_mib": primary_worker.get("peak_footprint_mib"),
                "review_worker_pid": None,
                "review_worker_seconds": None,
                "review_worker_peak_rss_mib": None,
                "review_worker_peak_footprint_mib": None,
                "analysis_worker_pid": primary_worker.get("pid"),
                "analysis_worker_seconds": round(perf_counter() - start_time, 4),
                "analysis_worker_peak_rss_mib": primary_worker.get("peak_rss_mib"),
                "analysis_worker_peak_footprint_mib": primary_worker.get("peak_footprint_mib"),
                "analysis_worker_memory_peak_mib": round(primary_memory_peak_mib, 1)
                if primary_memory_peak_mib is not None
                else None,
                "analysis_worker_memory_peak_source": primary_memory_peak_source,
            }
        )
        return {
            "entities": primary_entities,
            "statistics": primary_result.get("statistics") or {},
            "analysis_metadata": metadata,
        }

    review_compaction = compact_review_worker_payload_entities(
        primary_entities=primary_entities,
        review_entities=primary_review_entities,
    )
    compacted_primary_entities = review_compaction["entities"]
    compacted_review_entities = review_compaction["review_entities"]
    compacted_primary_metadata = dict(primary_metadata)
    compacted_primary_metadata["review_input_compaction"] = dict(review_compaction.get("summary") or {})
    compacted_stage_counts = dict(compacted_primary_metadata.get("stage_counts") or {})
    compacted_stage_counts["review_input_primary_compacted"] = len(compacted_primary_entities)
    compacted_stage_counts["review_input_surface_compacted"] = len(compacted_review_entities)
    compacted_primary_metadata["stage_counts"] = compacted_stage_counts

    try:
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "quality_gate",
                    "current": 0,
                    "total": 1,
                    "message": "正在执行片段审查、质量复扫与必要时疑难仲裁...",
                }
            )
        review_result = _run_worker_process_blocking(
            module_name="app.workers.qwen_review_worker",
            payload={
                "text": text,
                "entities": compacted_primary_entities,
                "review_entities": compacted_review_entities,
                "entities_filter": entities,
                "analysis_metadata": compacted_primary_metadata,
                "source_metadata": source_metadata or {},
                "source_structure": source_structure,
                "desensitize_mode": settings.get_effective_desensitize_mode(),
            },
            input_prefix="analysis-review-in-",
            output_prefix="analysis-review-out-",
            env_overrides={
                "ENABLE_QWEN_REVIEW": "True",
                "ENABLE_PRIMARY_UIE": "False",
                "ENABLE_PRIMARY_NER": "False",
                "ENABLE_SECONDARY_NER": "False",
            },
            timeout_seconds=0,
        )
    except Exception as exc:
        logger.error("Qwen final review worker failed; blocking export", exc_info=True)
        raise RuntimeError(f"final_review_worker_failed:{exc}") from exc

    review_worker = dict(review_result.get("_worker_process") or {})
    primary_peak = primary_worker.get("peak_rss_mib")
    review_peak = review_worker.get("peak_rss_mib")
    primary_footprint_peak = primary_worker.get("peak_footprint_mib")
    review_footprint_peak = review_worker.get("peak_footprint_mib")
    memory_peak_mib, memory_peak_source = _peak_memory_mib_from_workers(primary_worker, review_worker)
    numeric_peaks = [
        float(value)
        for value in (primary_peak, review_peak)
        if isinstance(value, (int, float))
    ]
    metadata = dict(review_result.get("analysis_metadata") or {})
    review_worker_completed = bool(metadata.get("review_completed", metadata.get("review_model_used", False)))
    review_worker_started = bool(metadata.get("review_started", metadata.get("review_model_used", False)))
    review_worker_incomplete = bool(metadata.get("ledger_conflict_adjudication_incomplete"))
    if review_worker_incomplete:
        metadata["requires_manual_review"] = True
        metadata["quality_gate_passed"] = False
        quality_flags = list(metadata.get("quality_flags") or [])
        quality_flags.append("ledger_conflict_adjudication_incomplete")
        metadata["quality_flags"] = sorted(set(str(flag) for flag in quality_flags if str(flag).strip()))
    metadata.update(
        {
            **workflow_metadata,
            "analysis_worker_process": True,
            "analysis_stage_isolation": True,
            "review_configured": True,
            "review_dispatched": True,
            "review_started": review_worker_started,
            "review_completed": review_worker_completed and not review_worker_incomplete,
            "analysis_stage_review_skipped": None,
            "primary_worker_pid": primary_worker.get("pid"),
            "primary_worker_seconds": primary_worker.get("seconds"),
            "primary_worker_peak_rss_mib": primary_peak,
            "primary_worker_peak_footprint_mib": primary_footprint_peak,
            "review_worker_pid": review_worker.get("pid"),
            "review_worker_seconds": review_worker.get("seconds"),
            "review_worker_peak_rss_mib": review_peak,
            "review_worker_peak_footprint_mib": review_footprint_peak,
            "analysis_worker_pid": review_worker.get("pid"),
            "analysis_worker_seconds": round(perf_counter() - start_time, 4),
            "analysis_worker_peak_rss_mib": round(max(numeric_peaks), 1) if numeric_peaks else None,
            "analysis_worker_peak_footprint_mib": memory_peak_mib if memory_peak_source == "macos_footprint" else None,
            "analysis_worker_memory_peak_mib": memory_peak_mib,
            "analysis_worker_memory_peak_source": memory_peak_source,
            "analysis_worker_peak_strategy": "max_sequential_worker_rss",
        }
    )
    return {
        "entities": review_result.get("entities") or [],
        "statistics": review_result.get("statistics") or {},
        "analysis_metadata": metadata,
    }


def _run_analysis_worker_blocking(
    *,
    text: str,
    entities: Optional[List[str]],
    use_llm: bool,
    use_custom: bool,
    llm_model: Optional[str],
    anonymization_strategy: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
    analysis_workflow: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    if analysis_workflow is None:
        analysis_workflow = _resolve_analysis_workflow(
            text=text,
            use_llm=use_llm,
            source_metadata=source_metadata,
            source_structure=source_structure,
        )
    if str((analysis_workflow or {}).get("mode") or "").strip() == "large_document_pre_routed":
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "large_document_parent_process",
                    "current": 0,
                    "total": 1,
                    "message": "已识别为大文件模式，跳过旧分析线，等待父流程裁切分组处理...",
                }
            )
        return _large_document_parent_process_deferred_analysis_result(analysis_workflow)

    if settings.is_high_quality_desensitize_mode() and bool(settings.ANALYSIS_STAGE_ISOLATION):
        return _run_stage_isolated_analysis_worker_blocking(
            text=text,
            entities=entities,
            use_llm=use_llm,
            use_custom=use_custom,
            llm_model=llm_model,
            anonymization_strategy=anonymization_strategy,
            source_metadata=source_metadata,
            source_structure=source_structure,
            analysis_workflow=analysis_workflow,
            progress_callback=progress_callback,
        )
    return _run_single_analysis_worker_blocking(
        text=text,
        entities=entities,
        use_llm=use_llm,
        use_custom=use_custom,
        llm_model=llm_model,
        anonymization_strategy=anonymization_strategy,
        source_metadata=source_metadata,
        source_structure=source_structure,
        analysis_workflow=analysis_workflow,
        progress_callback=progress_callback,
    )


async def _analyze_entities(
    *,
    text: str,
    entities: Optional[List[str]] = None,
    use_llm: bool,
    use_custom: bool,
    llm_model: Optional[str],
    anonymization_strategy: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    source_structure: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    analysis_workflow = _resolve_analysis_workflow(
        text=text,
        use_llm=use_llm,
        source_metadata=source_metadata,
        source_structure=source_structure,
    )
    if str(analysis_workflow.get("mode") or "").strip() == "large_document_pre_routed":
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "large_document_parent_process",
                    "current": 0,
                    "total": 1,
                    "message": "已识别为大文件模式，跳过旧分析线，等待父流程裁切分组处理...",
                }
            )
        return _large_document_parent_process_deferred_analysis_result(analysis_workflow)
    if _should_use_analysis_worker(use_llm):
        if progress_callback is not None:
            if analysis_worker_semaphore.locked():
                progress_callback(
                    {
                        "stage": "deep_review",
                        "current": 0,
                        "total": 1,
                        "message": "正在等待上一轮本地脱敏工作流结束...",
                    }
                )
        async with analysis_worker_semaphore:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "deep_review",
                        "current": 0,
                        "total": 1,
                        "message": "正在启动顺序处理进程...",
                    }
                )
            return await asyncio.to_thread(
                _run_analysis_worker_blocking,
                text=text,
                entities=entities,
                use_llm=use_llm,
                use_custom=use_custom,
                llm_model=llm_model,
                anonymization_strategy=anonymization_strategy,
                source_metadata=source_metadata,
                source_structure=source_structure,
                analysis_workflow=analysis_workflow,
                progress_callback=progress_callback,
            )

    engine = get_engine()
    analyze_kwargs = {
        "use_llm": use_llm,
        "use_custom": use_custom,
        "llm_model": llm_model,
        "source_metadata": source_metadata,
        "source_structure": source_structure,
        "progress_callback": progress_callback,
    }
    if entities is not None:
        analyze_kwargs["entities"] = entities
    detected_entities = await engine.analyze(text, **analyze_kwargs)
    if hasattr(engine, "get_entity_statistics"):
        statistics = engine.get_entity_statistics(detected_entities)
    else:
        statistics = _entity_statistics(detected_entities)
    metadata = _get_llm_analysis_metadata(engine, llm_model)
    metadata.update(_analysis_workflow_metadata(analysis_workflow))
    return {
        "entities": detected_entities,
        "statistics": statistics,
        "analysis_metadata": metadata,
    }


def _task_created_at(task: Dict) -> datetime:
    created_at = task.get("created_at")
    if isinstance(created_at, datetime):
        return created_at
    return datetime.min


def _clamp_progress(value: int) -> int:
    return max(0, min(100, int(value)))


def _set_task_state(
    task: Dict,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    if status is not None:
        task["status"] = status
    if progress is not None:
        task["progress"] = _clamp_progress(progress)
    if message is not None:
        task["message"] = message
    if error_message is not None:
        task["error_message"] = error_message
    task["updated_at"] = datetime.now()
    task_id = str(task.get("task_id") or "").strip()
    if task_id:
        _persist_task_state(task_id, task)


def _analysis_completion_state(metadata: Dict[str, Any]) -> tuple[str, str]:
    metadata = metadata if isinstance(metadata, dict) else {}
    quality_gate_passed = bool(metadata.get("quality_gate_passed", True))
    requires_manual_review = bool(metadata.get("requires_manual_review", False))
    flags = [str(flag) for flag in (metadata.get("quality_flags") or []) if str(flag).strip()]
    docx_blocking_metrics = (
        "docx_unhandled_text_part_count",
        "docx_unknown_text_part_count",
        "docx_hidden_text_node_count",
        "docx_field_instruction_node_count",
        "docx_review_required_text_node_count",
        "docx_entity_unrewritable_count",
        "docx_entity_range_crosses_virtual_fragment_count",
        "docx_entity_without_unit_count",
        "docx_entity_missing_unit_count",
        "docx_post_rewrite_residual_count",
        "final_directory_missing_replacement_count",
        "final_directory_subject_multi_replacement_count",
        "final_directory_replacement_reused_by_multi_subject_count",
    )
    for metric in docx_blocking_metrics:
        try:
            if int(metadata.get(metric) or 0) > 0:
                return "ready", "识别完成，请确认实体后生成脱敏文件。"
        except Exception:
            continue
    if not quality_gate_passed or requires_manual_review or flags:
        return "ready", "识别完成，请确认实体后生成脱敏文件。"
    return "ready", "识别完成，请确认实体后生成脱敏文件。"


def _task_status_response(task_id: str, task: Dict) -> TaskStatus:
    return TaskStatus(
        task_id=task_id,
        filename=task.get("filename"),
        status=str(task.get("status") or "pending"),
        progress=_clamp_progress(int(task.get("progress") or 0)),
        message=task.get("message"),
        error_message=task.get("error_message"),
        created_at=_task_created_at(task),
    )


def _build_processing_response_metadata(task: Dict[str, Any]) -> Dict[str, Any]:
    quality_metadata = task.get("quality_metadata") or {}
    entities = task.get("entities") or []
    return {
        **(task.get("metadata") or {}),
        **quality_metadata,
        "canonical_groups": _build_canonical_groups_payload(entities, quality_metadata),
        "suspected_misses": _build_suspected_misses_payload(quality_metadata),
    }


def _serialize_desensitize_result(task_id: str, task: Dict) -> DesensitizeResponse:
    config = task.get("config") or {}
    llm_model = config.get("llm_model")
    llm_strategy = config.get("llm_strategy")
    anonymization_strategy = config.get("anonymization_strategy")
    with desensitize_mode_context(config.get("desensitize_mode")):
        _, computed_llm_strategy_label = _get_strategy_payload(llm_model)
    _, computed_anonymization_strategy_label = _get_anonymization_strategy_payload(anonymization_strategy)
    llm_strategy_label = config.get("llm_strategy_label") or computed_llm_strategy_label
    anonymization_strategy_label = (
        config.get("anonymization_strategy_label") or computed_anonymization_strategy_label
    )
    download_url = (
        f"/api/v1/desensitize/download/{task_id}"
        if str(task.get("output_path") or "").strip()
        else None
    )
    mapping_download_url = (
        f"/api/v1/desensitize/download/mapping/{task_id}"
        if str(task.get("mapping_output_path") or "").strip()
        else None
    )
    return DesensitizeResponse(
        task_id=task_id,
        status=str(task.get("status") or "completed"),
        anonymized_text=task.get("anonymized_text"),
        entities=[Entity(**entity) for entity in task.get("entities") or []],
        metadata=_build_processing_response_metadata(task),
        download_url=download_url,
        mapping_download_url=mapping_download_url,
        output_filename=task.get("output_filename"),
        mapping_output_filename=task.get("mapping_output_filename"),
        output_file_type=task.get("output_file_type"),
        mapping_output_file_type=task.get("mapping_output_file_type"),
        preserves_format=bool(task.get("preserves_format")),
        llm_assisted=bool(config.get("use_llm", False)),
        llm_model=llm_model,
        llm_strategy=llm_strategy,
        llm_strategy_label=llm_strategy_label,
        anonymization_strategy=anonymization_strategy,
        anonymization_strategy_label=anonymization_strategy_label,
        warning=task.get("export_warning"),
        message=task.get("message"),
    )


def _count_batch_items(task: Dict) -> dict[str, int]:
    items = task.get("items") if isinstance(task, dict) else []
    if not isinstance(items, list):
        return {
            "file_count": 0,
            "completed_count": 0,
            "succeeded_count": 0,
            "failed_count": 0,
        }

    file_count = len(items)
    completed_count = sum(
        1 for item in items if str(item.get("status") or "").strip().lower() in {"completed", "failed"}
    )
    succeeded_count = sum(1 for item in items if str(item.get("status") or "").strip().lower() == "completed")
    failed_count = sum(1 for item in items if str(item.get("status") or "").strip().lower() == "failed")
    return {
        "file_count": file_count,
        "completed_count": completed_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
    }


def _serialize_batch_item(batch_id: str, item: Dict) -> BatchFileItem:
    output_path = str(item.get("output_path") or "").strip()
    download_url = item.get("download_url")
    if not download_url and output_path:
        download_url = f"/api/v1/desensitize/batch/download/{batch_id}/{item.get('item_id')}"
    mapping_output_path = str(item.get("mapping_output_path") or "").strip()
    mapping_download_url = item.get("mapping_download_url")
    if not mapping_download_url and mapping_output_path:
        mapping_download_url = f"/api/v1/desensitize/batch/download/{batch_id}/{item.get('item_id')}/mapping"

    return BatchFileItem(
        item_id=str(item.get("item_id") or ""),
        filename=str(item.get("filename") or ""),
        relative_path=str(item.get("relative_path") or item.get("filename") or ""),
        status=str(item.get("status") or "pending"),
        progress=_clamp_progress(int(item.get("progress") or 0)),
        message=item.get("message"),
        error_message=item.get("error_message"),
        entities_count=max(0, int(item.get("entities_count") or 0)),
        output_filename=item.get("output_filename"),
        mapping_output_filename=item.get("mapping_output_filename"),
        output_file_type=item.get("output_file_type"),
        mapping_output_file_type=item.get("mapping_output_file_type"),
        preserves_format=bool(item.get("preserves_format")),
        warning=item.get("warning"),
        download_url=download_url,
        mapping_download_url=mapping_download_url,
        metadata=item.get("metadata") or {},
    )


def _batch_task_status_response(batch_id: str, task: Dict) -> BatchTaskStatus:
    counts = _count_batch_items(task)
    items = task.get("items") if isinstance(task.get("items"), list) else []
    return BatchTaskStatus(
        batch_id=batch_id,
        folder_name=str(task.get("folder_name") or "selected-folder"),
        output_folder_name=(
            Path(str(task.get("output_dir") or "")).name
            if str(task.get("output_dir") or "").strip()
            else None
        ),
        status=str(task.get("status") or "pending"),
        progress=_clamp_progress(int(task.get("progress") or 0)),
        message=task.get("message"),
        error_message=task.get("error_message"),
        file_count=counts["file_count"],
        completed_count=counts["completed_count"],
        succeeded_count=counts["succeeded_count"],
        failed_count=counts["failed_count"],
        created_at=_task_created_at(task),
        items=[_serialize_batch_item(batch_id, item) for item in items if isinstance(item, dict)],
    )


def _serialize_batch_result(batch_id: str, task: Dict) -> BatchResult:
    status_payload = _batch_task_status_response(batch_id, task)
    return BatchResult(
        batch_id=status_payload.batch_id,
        folder_name=status_payload.folder_name,
        output_folder_name=status_payload.output_folder_name,
        status=status_payload.status,
        progress=status_payload.progress,
        message=status_payload.message,
        error_message=status_payload.error_message,
        file_count=status_payload.file_count,
        completed_count=status_payload.completed_count,
        succeeded_count=status_payload.succeeded_count,
        failed_count=status_payload.failed_count,
        archive_download_url=(
            f"/api/v1/desensitize/batch/download/{batch_id}"
            if str(task.get("archive_path") or "").strip()
            else None
        ),
        archive_filename=task.get("archive_filename"),
        items=status_payload.items,
    )


def _serialize_analysis_result(task_id: str, task: Dict) -> AnalyzeResponse:
    statistics = task.get("statistics") or {}
    metadata = dict(task.get("metadata") or {})
    entities = _annotate_response_entities(task.get("entities") or [], metadata)
    metadata.update(
        {
            "canonical_groups": _build_canonical_groups_payload(entities, metadata),
            "suspected_misses": _build_suspected_misses_payload(metadata),
        }
    )
    config = task.get("config") or {}
    llm_model = config.get("llm_model")
    llm_strategy = config.get("llm_strategy")
    anonymization_strategy = config.get("anonymization_strategy")
    with desensitize_mode_context(config.get("desensitize_mode")):
        _, computed_llm_strategy_label = _get_strategy_payload(llm_model)
    _, computed_anonymization_strategy_label = _get_anonymization_strategy_payload(anonymization_strategy)
    llm_strategy_label = config.get("llm_strategy_label") or computed_llm_strategy_label
    anonymization_strategy_label = (
        config.get("anonymization_strategy_label") or computed_anonymization_strategy_label
    )

    return AnalyzeResponse(
        task_id=task_id,
        filename=task["filename"],
        text=task["text"],
        entities=[Entity(**entity) for entity in entities],
        statistics=statistics,
        metadata=metadata,
        llm_model=llm_model,
        llm_strategy=llm_strategy,
        llm_strategy_label=llm_strategy_label,
        anonymization_strategy=anonymization_strategy,
        anonymization_strategy_label=anonymization_strategy_label,
    )


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


def _delete_directory_if_managed(path_value: Optional[str], *, allowed_parent: str) -> None:
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
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        logger.warning("Failed to remove stale runtime directory: %s", path, exc_info=True)


def _normalize_page_session_id(page_session_id: Optional[str]) -> Optional[str]:
    normalized = str(page_session_id or "").strip()
    if not normalized:
        return None
    return normalized[:128]


def _page_session_has_bindings(session: Dict[str, Any]) -> bool:
    return bool(session.get("task_ids") or session.get("batch_ids") or session.get("request_keys"))


def _remove_page_session_if_idle(page_session_id: Optional[str]) -> None:
    session_id = _normalize_page_session_id(page_session_id)
    if not session_id:
        return

    session = page_sessions.get(session_id)
    if session is None:
        return

    if _page_session_has_bindings(session):
        return

    watchdog = page_session_watchdogs.get(session_id)
    if watchdog is not None and watchdog.done():
        page_session_watchdogs.pop(session_id, None)

    if session.get("closed") or not page_session_watchdogs.get(session_id):
        page_sessions.pop(session_id, None)


def _ensure_page_session(page_session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    session_id = _normalize_page_session_id(page_session_id)
    if not session_id:
        return None

    now = datetime.now()
    session = page_sessions.get(session_id)
    if session is None:
        session = {
            "page_session_id": session_id,
            "created_at": now,
            "last_seen_at": now,
            "closed": False,
            "task_ids": set(),
            "batch_ids": set(),
            "request_keys": set(),
        }
        page_sessions[session_id] = session
    else:
        session["last_seen_at"] = now
        session.setdefault("task_ids", set())
        session.setdefault("batch_ids", set())
        session.setdefault("request_keys", set())
    return session


def _touch_page_session(page_session_id: Optional[str]) -> Optional[str]:
    session = _ensure_page_session(page_session_id)
    if session is None:
        return None
    return str(session["page_session_id"])


def _register_page_session_binding(
    page_session_id: Optional[str],
    *,
    task_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    request_key: Optional[str] = None,
) -> Optional[str]:
    session = _ensure_page_session(page_session_id)
    if session is None:
        return None

    if task_id:
        session["task_ids"].add(task_id)
    if batch_id:
        session["batch_ids"].add(batch_id)
    if request_key:
        session["request_keys"].add(request_key)

    _ensure_page_session_watchdog(str(session["page_session_id"]))
    return str(session["page_session_id"])


def _unregister_page_session_binding(
    page_session_id: Optional[str],
    *,
    task_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    request_key: Optional[str] = None,
) -> None:
    session_id = _normalize_page_session_id(page_session_id)
    if not session_id:
        return

    session = page_sessions.get(session_id)
    if session is None:
        return

    if task_id:
        session.setdefault("task_ids", set()).discard(task_id)
    if batch_id:
        session.setdefault("batch_ids", set()).discard(batch_id)
    if request_key:
        session.setdefault("request_keys", set()).discard(request_key)

    _remove_page_session_if_idle(session_id)


def _mark_task_stopped(task: Dict[str, Any], reason: str) -> None:
    status = str(task.get("status") or "").strip().lower()
    if status in {"completed", "failed"}:
        return
    _set_task_state(
        task,
        status="failed",
        progress=100,
        message=reason,
        error_message=reason,
    )


def _mark_batch_task_stopped(task: Dict[str, Any], reason: str) -> None:
    status = str(task.get("status") or "").strip().lower()
    if status in {"completed", "failed"}:
        return

    items = task.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_status = str(item.get("status") or "").strip().lower()
            if item_status in {"completed", "failed"}:
                continue
            _set_batch_item_state(
                item,
                status="failed",
                progress=100,
                message=reason,
                error_message=reason,
            )

    _refresh_batch_task_progress(task, reason)
    task["status"] = "failed"
    task["progress"] = 100
    task["message"] = reason
    task["error_message"] = reason
    task["updated_at"] = datetime.now()


def _cancel_page_session_tasks(page_session_id: Optional[str], *, reason: str) -> None:
    session_id = _normalize_page_session_id(page_session_id)
    if not session_id:
        return

    session = page_sessions.get(session_id)
    if session is None:
        return

    session["closed"] = True
    session["last_seen_at"] = datetime.now()

    for task_id in list(session.get("task_ids") or []):
        task = tasks.get(task_id)
        if isinstance(task, dict):
            _mark_task_stopped(task, reason)
            cancel_event = task.get("process_cancel_event")
            if isinstance(cancel_event, threading.Event):
                cancel_event.set()
        runner = analysis_task_runners.get(task_id)
        if runner is not None and not runner.done():
            runner.cancel()
        process_runner = process_task_runners.get(task_id)
        if process_runner is not None and not process_runner.done():
            process_runner.cancel()

    for batch_id in list(session.get("batch_ids") or []):
        task = batch_tasks.get(batch_id)
        if isinstance(task, dict):
            _mark_batch_task_stopped(task, reason)
        runner = batch_task_runners.get(batch_id)
        if runner is not None and not runner.done():
            runner.cancel()

    for request_key in list(session.get("request_keys") or []):
        request_task = active_request_tasks.get(request_key)
        if request_task is not None and not request_task.done():
            request_task.cancel()

    _remove_page_session_if_idle(session_id)


def _detach_page_session_tasks(page_session_id: Optional[str], *, reason: str) -> None:
    session_id = _normalize_page_session_id(page_session_id)
    if not session_id:
        return

    session = page_sessions.get(session_id)
    if session is None:
        return

    logger.info("Page session %s detached from background tasks: %s", session_id, reason)
    for task_id in list(session.get("task_ids") or []):
        task = tasks.get(task_id)
        if isinstance(task, dict) and task.get("page_session_id") == session_id:
            task["page_session_id"] = None
            task["page_session_detached_at"] = datetime.now()
            task["page_session_detached_reason"] = reason

    for batch_id in list(session.get("batch_ids") or []):
        task = batch_tasks.get(batch_id)
        if isinstance(task, dict) and task.get("page_session_id") == session_id:
            task["page_session_id"] = None
            task["page_session_detached_at"] = datetime.now()
            task["page_session_detached_reason"] = reason

    session.setdefault("task_ids", set()).clear()
    session.setdefault("batch_ids", set()).clear()
    session.setdefault("request_keys", set()).clear()
    session["closed"] = True
    session["last_seen_at"] = datetime.now()
    _remove_page_session_if_idle(session_id)


async def _watch_page_session(page_session_id: str) -> None:
    watch_interval = max(1, int(settings.PAGE_SESSION_WATCH_INTERVAL_SECONDS))
    heartbeat_grace = max(watch_interval, int(settings.PAGE_SESSION_HEARTBEAT_GRACE_SECONDS))

    try:
        while True:
            await asyncio.sleep(watch_interval)
            session = page_sessions.get(page_session_id)
            if session is None:
                return

            if session.get("closed"):
                _cancel_page_session_tasks(page_session_id, reason=PAGE_SESSION_STOP_DETAIL)
                return

            if not _page_session_has_bindings(session):
                _remove_page_session_if_idle(page_session_id)
                return

            last_seen = session.get("last_seen_at")
            if isinstance(last_seen, datetime) and (datetime.now() - last_seen).total_seconds() > heartbeat_grace:
                logger.info("Page session %s expired, detaching bound background tasks.", page_session_id)
                _detach_page_session_tasks(page_session_id, reason=PAGE_SESSION_TIMEOUT_DETAIL)
                return
    finally:
        current_task = asyncio.current_task()
        if page_session_watchdogs.get(page_session_id) is current_task:
            page_session_watchdogs.pop(page_session_id, None)
        _remove_page_session_if_idle(page_session_id)


def _ensure_page_session_watchdog(page_session_id: Optional[str]) -> None:
    session_id = _normalize_page_session_id(page_session_id)
    if not session_id:
        return

    existing = page_session_watchdogs.get(session_id)
    if existing is not None and not existing.done():
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    page_session_watchdogs[session_id] = loop.create_task(_watch_page_session(session_id))


def _purge_task(task_id: str) -> None:
    task = tasks.pop(task_id, None)
    if not isinstance(task, dict):
        _delete_task_state(task_id)
        return

    analysis_task_runners.pop(task_id, None)
    process_task_runners.pop(task_id, None)
    _unregister_page_session_binding(task.get("page_session_id"), task_id=task_id)
    _delete_if_managed(task.get("output_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_if_managed(task.get("mapping_output_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_if_managed(task.get("processed_result_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_if_managed(task.get("file_path"), allowed_parent=settings.UPLOAD_DIR)
    _delete_task_state(task_id)


def _purge_batch_task(batch_id: str) -> None:
    task = batch_tasks.pop(batch_id, None)
    if not isinstance(task, dict):
        return

    batch_task_runners.pop(batch_id, None)
    _unregister_page_session_binding(task.get("page_session_id"), batch_id=batch_id)
    for item in task.get("items") or []:
        if not isinstance(item, dict):
            continue
        _delete_if_managed(item.get("file_path"), allowed_parent=settings.UPLOAD_DIR)

    _delete_if_managed(task.get("archive_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_directory_if_managed(task.get("output_dir"), allowed_parent=settings.OUTPUT_DIR)


def _purge_terminal_tasks(now: Optional[datetime] = None) -> None:
    if not tasks and not batch_tasks:
        return

    current_time = now or datetime.now()
    retention = timedelta(hours=max(settings.TASK_RETENTION_HOURS, 1))
    finished_tasks: list[tuple[datetime, str]] = []
    expired_task_ids: list[str] = []
    finished_batch_tasks: list[tuple[datetime, str]] = []
    expired_batch_task_ids: list[str] = []

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

    for batch_id, task in list(batch_tasks.items()):
        if not isinstance(task, dict):
            continue

        status = str(task.get("status") or "").strip().lower()
        if status not in {"completed", "failed"}:
            continue

        created_at = _task_created_at(task)
        finished_batch_tasks.append((created_at, batch_id))
        if current_time - created_at > retention:
            expired_batch_task_ids.append(batch_id)

    for task_id in expired_task_ids:
        _purge_task(task_id)

    for batch_id in expired_batch_task_ids:
        _purge_batch_task(batch_id)

    remaining_finished = [
        (created_at, task_id)
        for created_at, task_id in finished_tasks
        if task_id in tasks
    ]
    overflow = len(remaining_finished) - max(settings.MAX_PRESERVED_TASKS, 1)
    if overflow > 0:
        for _, task_id in sorted(remaining_finished, key=lambda item: item[0])[:overflow]:
            _purge_task(task_id)

    remaining_finished_batches = [
        (created_at, batch_id)
        for created_at, batch_id in finished_batch_tasks
        if batch_id in batch_tasks
    ]
    batch_overflow = len(remaining_finished_batches) - max(settings.MAX_PRESERVED_TASKS, 1)
    if batch_overflow > 0:
        for _, batch_id in sorted(remaining_finished_batches, key=lambda item: item[0])[:batch_overflow]:
            _purge_batch_task(batch_id)


def _get_model_catalog() -> LLMModelListResponse:
    assets = all_assets()
    primary_models = []
    if settings.ENABLE_PRIMARY_UIE:
        primary_models.append(settings.PRIMARY_IE_MODEL)
    if settings.ENABLE_PRIMARY_NER:
        primary_models.append(settings.PRIMARY_NER_MODEL)
    if settings.ENABLE_SECONDARY_NER:
        primary_models.append(settings.SECONDARY_NER_MODEL)
    review_models = list(
        dict.fromkeys(
            [
                settings.REVIEW_MODEL,
                settings.REVIEW_MODEL_FALLBACK,
            ]
        )
    )
    local_options = [
        LLMModelOption(
            name=asset.model_id,
            installed=asset.installed,
            is_default=False,
            strategy_key=settings.get_high_quality_profile_key(),
            strategy_label=settings.get_high_quality_profile_label(),
            strategy_description="中文规则、外挂规则、UIE/NER 与片段审查模型组合的本地工作流。",
            tier=asset.role,
            supports_precision_review=asset.role in {"review", "review_fallback"},
            supports_vision=False,
            recommended_for=(
                ["本地小模型片段精审备用"]
                if asset.role in {"review", "review_fallback"}
                else ["中文主识别", "实体召回"]
            ),
            role=asset.role,
            memory_tier=asset.memory_tier,
            local_path=str(asset.path) if asset.path else None,
        )
        for asset in assets
    ]
    installed_local_models = [asset.model_id for asset in assets if asset.installed]
    default_review_model = (
        settings.get_default_review_llm_model(available_models=installed_local_models)
        or settings.REVIEW_MODEL
    )
    return LLMModelListResponse(
        backend=settings.get_high_quality_profile_key(),
        default_model=default_review_model,
        service_available=True,
        profile=settings.get_high_quality_profile_key(),
        primary_models=primary_models,
        review_models=review_models,
        models=local_options,
    )


def _get_runtime_status() -> RuntimeStatusResponse:
    current_platform = platform_label()
    catalog = _get_model_catalog()
    available_processing_models = [item.name for item in catalog.models if item.installed]
    review_candidates = settings.get_route_review_model_candidates(
        available_models=available_processing_models
    )
    preferred_review_model = next(
        (model_name for model_name in review_candidates if model_name in available_processing_models),
        None,
    )
    review_ready = bool(preferred_review_model)
    required_model = settings.REVIEW_MODEL
    default_review_model = settings.get_default_review_llm_model(
        available_models=available_processing_models
    ) or required_model
    required_review_installed = required_model in available_processing_models
    primary_ready = primary_models_ready()
    ready = primary_ready
    if primary_ready and review_ready:
        recommended_action = (
            f"高质量低内存工作流已就绪：主识别后会用 {preferred_review_model} "
            "进行高风险片段审查；最终质量以规则层硬闸和小模型兜底为主。"
        )
    elif primary_ready:
        recommended_action = "中文主识别模型已就绪，但未检测到低内存片段审查模型；系统会继续识别并标记人工复核。"
    else:
        recommended_action = f"请先下载中文 UIE/NER 主识别模型，再开始{settings.get_high_quality_profile_label()}。"

    return RuntimeStatusResponse(
        backend=settings.get_high_quality_profile_key(),
        platform=current_platform,
        ready=ready,
        ollama_install_detected=False,
        ollama_path=None,
        service_available=True,
        required_model=required_model,
        required_model_installed=required_review_installed,
        available_processing_models=available_processing_models,
        preferred_processing_model=preferred_review_model,
        default_model=default_review_model,
        installer_hint=installer_hint(),
        download_hint=(
            f"主识别模型目录：{settings.LOWMEM_MODEL_ROOT}；低内存片段审查模型："
            f"{settings.REVIEW_MODEL} / {settings.REVIEW_MODEL_FALLBACK}"
        ),
        recommended_action=recommended_action,
        desensitize_mode=settings.get_effective_desensitize_mode(),
        primary_models_ready=primary_ready,
        review_model_installed=review_ready,
        review_model_loaded=False,
        estimated_memory_tier=settings.get_high_quality_memory_tier(),
        analysis_worker_process=bool(settings.ANALYSIS_WORKER_PROCESS),
        analysis_stage_isolation=bool(settings.ANALYSIS_STAGE_ISOLATION),
        analysis_worker_timeout=int(settings.ANALYSIS_WORKER_TIMEOUT),
    )


def _get_strategy_payload(llm_model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    return settings.get_high_quality_profile_key(), settings.get_high_quality_profile_label()


def _get_anonymization_strategy_payload(
    strategy_key: Optional[str],
) -> tuple[str, str]:
    strategy = get_anonymization_strategy_profile(strategy_key)
    return strategy.key, strategy.label


def _get_llm_analysis_metadata(engine, llm_model: Optional[str]) -> Dict[str, object]:
    recognizer = engine.recognizer_registry.get_recognizer("high_quality_lowmem")
    if recognizer is None or not hasattr(recognizer, "get_last_run_metadata"):
        return {}
    try:
        return dict(recognizer.get_last_run_metadata(llm_model))
    except Exception:
        return {}


def _resolve_requested_llm_model(llm_model: Optional[str]) -> str:
    default_review_model = settings.get_default_review_llm_model()
    fallback = settings.REVIEW_MODEL
    return (llm_model or default_review_model or fallback).strip() or fallback


def _ensure_model_ready(llm_model: Optional[str]) -> str:
    return _resolve_requested_llm_model(llm_model)


def _resolve_document_parse_models(
    file_path: str,
    *,
    use_llm: bool,
    llm_model: Optional[str],
) -> dict[str, object]:
    requested_model = (llm_model or "").strip() or None
    parse_models: dict[str, object] = {
        "llm_model": requested_model,
        "ocr_llm_model": None,
    }
    if not use_llm or Path(file_path).suffix.lower() != ".pdf":
        return parse_models
    return parse_models


def _apply_document_parse_metadata(
    doc_result: Dict,
    *,
    requested_llm_model: Optional[str],
    parse_models: dict[str, object],
) -> Dict:
    metadata = dict(doc_result.get("metadata") or {})
    requested_model = (requested_llm_model or "").strip()
    ocr_pages = max(0, int(metadata.get("ocr_pages") or 0))
    raw_ocr_model = str(metadata.get("ocr_model") or "").strip()
    effective_ocr_model = str(metadata.get("ocr_review_model") or raw_ocr_model).strip()
    rapidocr_normalized = str(metadata.get("ocr_engine") or "").strip() == "rapidocr_onnxruntime"

    if requested_model:
        metadata["requested_llm_model"] = requested_model
    if rapidocr_normalized:
        if ocr_pages > 0:
            metadata["effective_ocr_model"] = effective_ocr_model or "ppocrv5_mobile"
            metadata["effective_ocr_engine"] = "rapidocr_onnxruntime"
            metadata["ocr_upgrade_applied"] = False
        doc_result["metadata"] = metadata
        return doc_result
    if ocr_pages > 0 and effective_ocr_model:
        metadata["effective_ocr_model"] = effective_ocr_model
    if ocr_pages > 0 and raw_ocr_model:
        metadata["effective_ocr_engine"] = str(metadata.get("ocr_engine") or raw_ocr_model).strip()

    if ocr_pages > 0 and requested_model and effective_ocr_model and effective_ocr_model != requested_model:
        metadata["ocr_upgrade_applied"] = True
        metadata["ocr_upgrade_reason"] = (
            "scan_pdf_local_ocr_review_model"
            if str(metadata.get("ocr_engine") or "").strip() == "macos_vision"
            else "scan_pdf_ocr_uses_review_model"
        )
        metadata["ocr_upgrade_from_model"] = requested_model
        metadata["ocr_upgrade_to_model"] = effective_ocr_model
        metadata["recommended_llm_model"] = effective_ocr_model
        metadata["recommended_llm_reason"] = (
            "扫描页 OCR 已自动切到 27B。若文档里还有简称、别名或低清晰度片段，建议后续整篇精查也使用该模型。"
        )
    elif ocr_pages > 0 and effective_ocr_model:
        metadata["ocr_upgrade_applied"] = False

    requested_ocr_model = str(parse_models.get("ocr_llm_model") or "").strip()
    if requested_ocr_model:
        metadata["ocr_requested_model"] = requested_ocr_model

    doc_result["metadata"] = metadata
    return doc_result


def _should_normalize_pdf_for_lowmem(file_path: str) -> bool:
    return bool(
        settings.is_high_quality_lowmem_mode()
        and bool(settings.PDF_NORMALIZE_TO_DOCX)
        and Path(file_path).suffix.lower() == ".pdf"
    )


def _pdf_normalization_output_dir() -> str:
    output_dir = Path(settings.RUNTIME_ROOT) / "pdf_normalized"
    ensure_private_directory(output_dir)
    return str(output_dir)


def _run_pdf_normalization_worker_blocking(file_path: str) -> Dict[str, Any]:
    source_path = str(Path(file_path).expanduser().resolve())
    result = _run_worker_process_blocking(
        module_name="app.workers.pdf_normalize_worker",
        payload={
            "file_path": source_path,
            "output_dir": _pdf_normalization_output_dir(),
        },
        input_prefix="pdf-normalize-in-",
        output_prefix="pdf-normalize-out-",
        timeout_seconds=int(settings.PDF_NORMALIZE_WORKER_TIMEOUT),
        env_overrides={
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    )
    normalized = dict(result.get("normalized") or {})
    metadata = dict(normalized.get("metadata") or {})
    worker = dict(result.get("_worker_process") or {})
    worker_memory_peak_mib, worker_memory_peak_source = _peak_memory_mib_from_worker(worker)
    metadata.update(
        {
            "normalization_worker_process": True,
            "normalization_worker_pid": worker.get("pid"),
            "normalization_worker_seconds": worker.get("seconds"),
            "normalization_worker_peak_rss_mib": worker.get("peak_rss_mib"),
            "normalization_worker_peak_footprint_mib": worker.get("peak_footprint_mib"),
            "normalization_worker_peak_mib": round(worker_memory_peak_mib, 1)
            if worker_memory_peak_mib is not None
            else None,
            "normalization_worker_peak_source": worker_memory_peak_source,
        }
    )
    normalized["metadata"] = metadata
    return normalized


async def _normalize_pdf_for_parsing(file_path: str) -> Dict[str, Any]:
    async with pdf_normalize_worker_semaphore:
        return await asyncio.to_thread(_run_pdf_normalization_worker_blocking, file_path)


def _callable_accepts_keyword(func: object, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


async def _parse_normalized_pdf_document(
    file_path: str,
    *,
    use_llm: bool,
    llm_model: Optional[str],
) -> Dict:
    normalized = await _normalize_pdf_for_parsing(file_path)
    normalized_path = str(normalized.get("normalized_file_path") or "").strip()
    if not normalized_path:
        raise RuntimeError("pdf_normalization_missing_output_path")
    if not Path(normalized_path).exists():
        raise RuntimeError(f"pdf_normalization_output_not_found:{normalized_path}")

    docx_result = await document_parser.parse(
        normalized_path,
        use_llm=False,
        llm_model=None,
        ocr_llm_model=None,
    )
    normalized_metadata = dict(normalized.get("metadata") or {})
    metadata = {
        **(docx_result.get("metadata") or {}),
        **normalized_metadata,
        "original_pdf_file_path": file_path,
        "normalized_file_path": normalized_path,
    }
    structure = normalized.get("structure") or docx_result.get("structure")
    doc_result = {
        "text": str(docx_result.get("text") or ""),
        "metadata": metadata,
        "structure": structure,
        "normalized_file_path": normalized_path,
    }
    if not doc_result["text"] and isinstance(structure, dict):
        doc_result["text"] = "\n\n".join(
            str(page.get("text") or "")
            for page in structure.get("pages", [])
            if isinstance(page, dict) and str(page.get("text") or "").strip()
        )
    return doc_result


def _task_normalized_file_path(container: Dict[str, Any]) -> Optional[str]:
    direct = str(container.get("normalized_file_path") or "").strip()
    if direct:
        return direct
    metadata = container.get("metadata")
    if isinstance(metadata, dict):
        normalized = str(metadata.get("normalized_file_path") or "").strip()
        if normalized:
            return normalized
    return None


def _source_path_for_export(container: Dict[str, Any]) -> str:
    return _task_normalized_file_path(container) or str(container["file_path"])


def _analysis_failure_detail(exc: Exception, default: str) -> str:
    error_text = str(exc)
    if "ocr_model_missing" in error_text:
        return "PDF OCR 模型缺失，请先下载 RapidOCR PP-OCRv5 mobile det/rec 模型后重试。"
    if "ocr_quality_gate_failed" in error_text:
        return "PDF OCR 质量门禁未通过，存在识别失败或低质量页面，请检查原 PDF 清晰度和本机 OCR/VL 组件配置后重试。"
    if "pdf_normalization" in error_text or "pdf-normalize" in error_text:
        return "PDF 前置识别或 DOCX 规范化失败，请检查 PDF 内容或 OCR 组件配置后重试。"
    return default


def _processing_failure_detail(exc: Exception, default: str) -> str:
    error_text = str(exc)
    if "process_worker_timeout" in error_text:
        return "超大文档在当前时限内仍未完成脱敏与导出，后台已安全停止本次任务。请稍后重试或调整配置后继续。"
    if "process_worker_cancelled" in error_text:
        return PAGE_SESSION_STOP_DETAIL
    return _analysis_failure_detail(exc, default)


async def _parse_document(
    file_path: str,
    *,
    use_llm: bool,
    llm_model: Optional[str],
    progress_callback=None,
) -> Dict:
    parse_models = _resolve_document_parse_models(
        file_path,
        use_llm=use_llm,
        llm_model=llm_model,
    )
    if _should_normalize_pdf_for_lowmem(file_path):
        doc_result = await _parse_normalized_pdf_document(
            file_path,
            use_llm=use_llm,
            llm_model=llm_model,
        )
    else:
        doc_result = await document_parser.parse(
            file_path,
            use_llm=use_llm,
            llm_model=llm_model,
            ocr_llm_model=parse_models.get("ocr_llm_model"),
        )
    return _apply_document_parse_metadata(
        doc_result,
        requested_llm_model=llm_model,
        parse_models=parse_models,
    )


async def _parse_document_with_optional_progress(
    file_path: str,
    *,
    use_llm: bool,
    llm_model: Optional[str],
    progress_callback=None,
) -> Dict:
    kwargs: Dict[str, Any] = {
        "use_llm": use_llm,
        "llm_model": llm_model,
    }
    if progress_callback is not None and _callable_accepts_keyword(_parse_document, "progress_callback"):
        kwargs["progress_callback"] = progress_callback
    return await _parse_document(file_path, **kwargs)


def _set_batch_item_state(
    item: Dict,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    if status is not None:
        item["status"] = status
    if progress is not None:
        item["progress"] = _clamp_progress(progress)
    if message is not None:
        item["message"] = message
    if error_message is not None:
        item["error_message"] = error_message
    item["updated_at"] = datetime.now()


def _refresh_batch_task_progress(task: Dict, message: Optional[str] = None) -> None:
    items = task.get("items")
    if not isinstance(items, list) or not items:
        task["progress"] = 0
        if message is not None:
            task["message"] = message
        return

    completed_progress = sum(_clamp_progress(int(item.get("progress") or 0)) for item in items if isinstance(item, dict))
    task["progress"] = _clamp_progress(int(round(completed_progress / len(items))))
    if message is not None:
        task["message"] = message
    task["updated_at"] = datetime.now()


def _normalize_relative_upload_path(raw_path: Optional[str], fallback_name: str) -> str:
    fallback = Path(fallback_name or "document.txt").name or "document.txt"
    normalized = str(raw_path or "").strip().replace("\\", "/")
    if not normalized:
        return fallback

    try:
        path = PurePosixPath(normalized)
    except Exception:
        return fallback

    parts: List[str] = []
    for part in path.parts:
        token = str(part).strip()
        if not token or token in {".", "/"}:
            continue
        if token == "..":
            raise HTTPException(status_code=400, detail="目录路径不合法。")
        parts.append(token)

    if not parts:
        return fallback

    parts[-1] = Path(parts[-1]).name or fallback
    return str(PurePosixPath(*parts))


def _resolve_folder_name(folder_name: Optional[str], relative_paths: List[str]) -> str:
    candidate = Path(str(folder_name or "").strip()).name.strip()
    if candidate:
        return candidate

    for relative_path in relative_paths:
        normalized = str(relative_path or "").strip().replace("\\", "/")
        if not normalized:
            continue
        parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "."}]
        if len(parts) > 1:
            return parts[0]

    return "selected-folder"


def _sanitize_output_folder_name(folder_name: str) -> str:
    normalized = re.sub(r"[^\w\-.]+", "_", str(folder_name or "").strip(), flags=re.UNICODE)
    normalized = normalized.strip("._")
    return normalized or "selected-folder"


def _build_unique_output_folder(folder_name: str, batch_id: str) -> Path:
    base_name = f"{_sanitize_output_folder_name(folder_name)}_anonymized"
    preferred = Path(settings.OUTPUT_DIR) / base_name
    if not preferred.exists():
        return preferred
    return Path(settings.OUTPUT_DIR) / f"{base_name}_{batch_id[:8]}"


def _batch_output_root(batch_id: str, folder_name: str) -> Path:
    output_root = _build_unique_output_folder(folder_name, batch_id)
    ensure_private_directory(output_root)
    return output_root


def _relative_batch_output_path(relative_path: str, folder_name: str, output_filename: str) -> PurePosixPath:
    relative = PurePosixPath(relative_path)
    parts = [part for part in relative.parts if part not in {"", "."}]
    normalized_folder_name = _sanitize_output_folder_name(folder_name)
    if parts and parts[0] == folder_name:
        parts = parts[1:]
    elif parts and parts[0] == normalized_folder_name:
        parts = parts[1:]
    if not parts:
        return PurePosixPath(output_filename)
    parts[-1] = output_filename
    return PurePosixPath(*parts)


def _build_batch_output_path(output_root: Path, relative_path: str, folder_name: str, output_filename: str) -> Path:
    relative_output_path = _relative_batch_output_path(relative_path, folder_name, output_filename)
    parent = relative_output_path.parent if str(relative_output_path.parent) != "." else PurePosixPath()
    target_path = output_root
    if parent.parts:
        target_path = output_root.joinpath(*parent.parts)
        ensure_private_directory(target_path)
    return output_root.joinpath(*relative_output_path.parts)


def _move_batch_output(export_result: Dict[str, object], target_path: Path) -> str:
    source_path = Path(str(export_result.get("output_path") or ""))
    if not source_path.exists():
        raise FileNotFoundError(f"Batch export output was not found: {source_path}")

    ensure_private_directory(target_path.parent)
    shutil.move(str(source_path), str(target_path))
    ensure_private_file(target_path)
    return str(target_path)


def _create_batch_archive(batch_id: str, folder_name: str, output_root: Path) -> tuple[Optional[str], Optional[str]]:
    if not output_root.exists():
        return None, None

    archive_filename = f"{output_root.name}.zip"
    archive_path = Path(settings.OUTPUT_DIR) / f"{batch_id}_bundle.zip"

    has_files = False
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_root.rglob("*"):
            if not file_path.is_file():
                continue
            has_files = True
            archive.write(file_path, arcname=str(file_path.relative_to(output_root.parent)))

    if not has_files:
        archive_path.unlink(missing_ok=True)
        return None, None

    ensure_private_file(archive_path)
    return str(archive_path), archive_filename


def _compact_terminal_task(task: Dict, *, preserve_anonymized_text: bool = False) -> None:
    if not isinstance(task, dict):
        return

    text = task.get("text")
    if isinstance(text, str):
        task["text_length"] = len(text)
        task.pop("text", None)

    structure = task.get("structure")
    if isinstance(structure, dict):
        page_count = 0
        for key in ("page_count", "pages_count"):
            try:
                page_count = max(page_count, int(structure.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue
        pages = structure.get("pages")
        if page_count <= 0 and isinstance(pages, list):
            page_count = sum(1 for item in pages if isinstance(item, dict))
        if page_count > 0:
            _sync_task_page_count_hint(task, page_count)
        task.pop("structure", None)

    entities = task.get("entities")
    if isinstance(entities, list):
        task["entities_count"] = len(entities)

    anonymized_text = task.get("anonymized_text")
    if isinstance(anonymized_text, str):
        task["anonymized_text_length"] = len(anonymized_text)
        if not preserve_anonymized_text:
            task.pop("anonymized_text", None)


async def _ensure_task_document_loaded(task: Dict) -> None:
    if task.get("text") and task.get("structure") is not None:
        return

    reload_path = _task_normalized_file_path(task)
    if not reload_path or not Path(reload_path).exists():
        reload_path = str(task["file_path"])

    with desensitize_mode_context(_task_desensitize_mode(task)):
        doc_result = await _parse_document(
            reload_path,
            use_llm=bool(task.get("config", {}).get("use_llm", False)),
            llm_model=task.get("config", {}).get("llm_model"),
        )
    task["text"] = doc_result["text"]
    task["structure"] = doc_result.get("structure")
    _merge_task_document_metadata(task, doc_result=doc_result)


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


def _clear_task_processing_artifacts(task: Dict[str, Any]) -> None:
    _delete_if_managed(task.get("output_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_if_managed(task.get("mapping_output_path"), allowed_parent=settings.OUTPUT_DIR)
    _delete_if_managed(task.get("processed_result_path"), allowed_parent=settings.OUTPUT_DIR)
    for key in (
        "output_path",
        "output_filename",
        "output_file_type",
        "output_media_type",
        "mapping_output_path",
        "mapping_output_filename",
        "mapping_output_file_type",
        "mapping_output_media_type",
        "preserves_format",
        "export_warning",
        "anonymized_text",
        "quality_metadata",
        "processing_time",
        "processed_result_path",
    ):
        task.pop(key, None)


def _build_process_worker_payload(
    task_id: str,
    task: Dict[str, Any],
    request_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "source_path": _source_path_for_export(task),
        "original_filename": str(task.get("filename") or ""),
        "source_text": str(task.get("text") or ""),
        "source_metadata": task.get("metadata") or {},
        "source_structure": task.get("structure"),
        "entities": _serialize_entities(request_payload.get("entities") or []),
        "operator_config": request_payload.get("operator_config"),
        "use_llm": bool(task.get("config", {}).get("use_llm", False)),
        "llm_model": request_payload.get("llm_model"),
        "anonymization_strategy": request_payload.get("anonymization_strategy"),
        "desensitize_mode": request_payload.get("desensitize_mode") or _task_desensitize_mode(task),
    }


async def _run_process_worker_async(
    *,
    worker_payload: Dict[str, Any],
    progress_callback=None,
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    if progress_callback is not None and process_worker_semaphore.locked():
        progress_callback(
            {
                "stage": "waiting",
                "current": 0,
                "total": 1,
                "message": "正在等待上一轮后台脱敏导出结束...",
            }
        )
    async with process_worker_semaphore:
        return await asyncio.to_thread(
            _run_worker_process_blocking,
            module_name="app.workers.process_worker",
            payload=worker_payload,
            input_prefix="process-worker-in-",
            output_prefix="process-worker-out-",
            timeout_seconds=_process_worker_timeout_seconds(worker_payload),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )


def _is_large_document_parent_process_payload(worker_payload: Dict[str, Any]) -> bool:
    source_metadata = worker_payload.get("source_metadata")
    if not isinstance(source_metadata, dict):
        return False
    return (
        source_metadata.get("_large_document_pre_routed") is True
        or str(source_metadata.get("analysis_workflow_mode") or "").strip() == "large_document_pre_routed"
        or str(source_metadata.get("large_document_execution_mode") or "").strip()
        == "defer_to_process_worker_grouped_default_line"
    )


def _process_worker_timeout_seconds(worker_payload: Dict[str, Any]) -> int:
    if _is_large_document_parent_process_payload(worker_payload):
        return 0
    return int(settings.PROCESS_WORKER_TIMEOUT)


def _apply_completed_process_result(
    *,
    task_id: str,
    task: Dict[str, Any],
    process_result: Dict[str, Any],
    total_processing_time: float,
) -> None:
    worker = dict(process_result.get("_worker_process") or {})
    worker_memory_peak_mib, worker_memory_peak_source = _peak_memory_mib_from_worker(worker)
    quality_metadata = dict(process_result.get("quality_metadata") or {})
    quality_metadata.update(
        {
            "process_worker_process": True,
            "process_worker_pid": worker.get("pid"),
            "process_worker_seconds": worker.get("seconds"),
            "process_worker_peak_rss_mib": worker.get("peak_rss_mib"),
            "process_worker_peak_footprint_mib": worker.get("peak_footprint_mib"),
            "process_worker_memory_peak_mib": round(worker_memory_peak_mib, 1)
            if worker_memory_peak_mib is not None
            else None,
            "process_worker_memory_peak_source": worker_memory_peak_source,
        }
    )

    entities = _annotate_response_entities(process_result.get("entities") or [], quality_metadata)
    anonymized_text = str(process_result.get("anonymized_text") or "")
    export_result = dict(process_result.get("export_result") or {})

    quality_warning = None
    if quality_metadata and not quality_metadata.get("quality_gate_passed", True):
        quality_warning = (
            "质量检查提示：仍有残留命中或一致性问题"
            + (
                f"：{quality_metadata.get('quality_gate_reason')}"
                if quality_metadata.get("quality_gate_reason")
                else ""
            )
        )
    warning_message = export_result.get("warning")
    if quality_warning:
        warning_message = f"{warning_message} | {quality_warning}" if warning_message else quality_warning

    task["entities"] = entities
    task["anonymized_text"] = anonymized_text
    task["output_path"] = export_result.get("output_path")
    task["output_filename"] = export_result.get("download_name")
    task["output_file_type"] = export_result.get("file_type")
    task["output_media_type"] = export_result.get("media_type")
    task["mapping_output_path"] = export_result.get("mapping_output_path")
    task["mapping_output_filename"] = export_result.get("mapping_download_name")
    task["mapping_output_file_type"] = export_result.get("mapping_file_type")
    task["mapping_output_media_type"] = export_result.get("mapping_media_type")
    task["preserves_format"] = bool(export_result.get("preserves_format"))
    task["export_warning"] = warning_message
    task["processing_time"] = round(total_processing_time, 4)
    task["quality_metadata"] = quality_metadata
    task["progress"] = 100
    task["status"] = "completed"
    task["message"] = "脱敏完成，可下载结果文件。"
    task["error_message"] = None
    task["updated_at"] = datetime.now()
    _compact_terminal_task(task, preserve_anonymized_text=True)
    _persist_task_state(task_id, task)


async def _run_process_task(task_id: str) -> None:
    task = tasks.get(task_id)
    if task is None:
        return

    request_payload = task.get("pending_process_request")
    if not isinstance(request_payload, dict):
        _set_task_state(
            task,
            status="failed",
            progress=100,
            message=ANONYMIZE_FAILURE_DETAIL,
            error_message=ANONYMIZE_FAILURE_DETAIL,
        )
        return

    started_at = perf_counter()
    cancel_event = threading.Event()
    task["process_cancel_event"] = cancel_event
    try:
        await _ensure_task_document_loaded(task)
        worker_payload = _build_process_worker_payload(task_id, task, request_payload)
        result = await _run_process_worker_async(
            worker_payload=worker_payload,
            progress_callback=_process_progress_callback(task_id),
            cancel_event=cancel_event,
        )
        _apply_completed_process_result(
            task_id=task_id,
            task=task,
            process_result=result,
            total_processing_time=task.get("analysis_time", 0) + (perf_counter() - started_at),
        )
    except asyncio.CancelledError:
        cancel_event.set()
        logger.info("Background desensitization cancelled: %s", task_id)
        _mark_task_stopped(task, PAGE_SESSION_STOP_DETAIL)
        raise
    except Exception as exc:
        cancel_event.set()
        logger.error("Background desensitization failed: %s", exc, exc_info=True)
        failure_detail = _processing_failure_detail(exc, ANONYMIZE_FAILURE_DETAIL)
        _set_task_state(
            task,
            status="failed",
            progress=100,
            message=failure_detail,
            error_message=failure_detail,
        )
        _persist_task_state(task_id, task)
    finally:
        task.pop("process_cancel_event", None)
        task.pop("pending_process_request", None)
        process_task_runners.pop(task_id, None)
        _unregister_page_session_binding(task.get("page_session_id"), task_id=task_id)


def _annotate_response_entities(entities: list[Dict], quality_metadata: Dict) -> list[Dict]:
    evidence_summary = quality_metadata.get("evidence_summary")
    group_labels: Dict[str, str] = {}
    if isinstance(evidence_summary, list):
        for item in evidence_summary:
            if not isinstance(item, dict):
                continue
            canonical_key = str(item.get("canonical_key", "")).strip()
            if not canonical_key:
                continue
            group_labels[canonical_key] = str(item.get("primary_text", "")).strip() or canonical_key
    _merge_subject_ledger_group_labels(group_labels, quality_metadata)

    residual_keys = set()
    residual_hits = quality_metadata.get("residual_hits")
    if isinstance(residual_hits, list):
        for item in residual_hits:
            if isinstance(item, dict) and item.get("canonical_key"):
                residual_keys.add(str(item["canonical_key"]).strip())

    annotated: list[Dict] = []
    for entity in entities:
        item = dict(entity)
        canonical_key = str(item.get("canonical_key") or "").strip()
        if not canonical_key:
            metadata = item.get("metadata") or {}
            canonical_key = str(metadata.get("canonical_key") or "").strip()
        if canonical_key:
            item["group_id"] = canonical_key
            item["group_label"] = group_labels.get(canonical_key) or str(
                item.get("context_label") or item.get("text") or ""
            )
            if canonical_key in residual_keys:
                item["needs_review"] = True
                item["review_reason"] = item.get("review_reason") or "该主体仍存在残留命中，需要人工核对。"
        annotated.append(item)
    return annotated


def _build_canonical_groups_payload(entities: list[Dict], quality_metadata: Dict) -> list[Dict]:
    groups: Dict[str, Dict[str, object]] = {}
    ledger_group_labels: Dict[str, str] = {}
    _merge_subject_ledger_group_labels(ledger_group_labels, quality_metadata)
    for entity in entities:
        canonical_key = str(entity.get("group_id") or entity.get("canonical_key") or "").strip()
        if not canonical_key:
            continue
        group = groups.setdefault(
            canonical_key,
            {
                "group_id": canonical_key,
                "group_label": str(
                    ledger_group_labels.get(canonical_key)
                    or entity.get("group_label")
                    or entity.get("context_label")
                    or entity.get("text")
                    or ""
                ).strip(),
                "primary_text": str(entity.get("text") or "").strip(),
                "canonical_role": str(entity.get("canonical_role") or "").strip() or None,
                "aliases": set(),
                "mentions": 0,
            },
        )
        text = str(entity.get("text") or "").strip()
        if text:
            group["aliases"].add(text)
            if len(text) > len(str(group["primary_text"] or "")):
                group["primary_text"] = text
        group["mentions"] = int(group["mentions"] or 0) + 1
        if not group.get("group_label") and entity.get("group_label"):
            group["group_label"] = str(entity.get("group_label"))
        if not group.get("canonical_role") and entity.get("canonical_role"):
            group["canonical_role"] = str(entity.get("canonical_role"))

    evidence_summary = quality_metadata.get("evidence_summary")
    if isinstance(evidence_summary, list):
        for item in evidence_summary:
            if not isinstance(item, dict):
                continue
            canonical_key = str(item.get("canonical_key", "")).strip()
            if canonical_key and canonical_key in groups:
                groups[canonical_key]["confirmed"] = not bool(item.get("conflict"))

    result = []
    for group in groups.values():
        result.append(
            {
                "group_id": group["group_id"],
                "group_label": group["group_label"] or group["primary_text"],
                "primary_text": group["primary_text"],
                "canonical_role": group["canonical_role"],
                "aliases": sorted(group["aliases"], key=len, reverse=True),
                "mentions": group["mentions"],
                "confirmed": bool(group.get("confirmed", True)),
            }
        )
    result.sort(key=lambda item: (-int(item["mentions"]), -len(str(item["primary_text"])), str(item["group_id"])))
    return result


def _merge_subject_ledger_group_labels(group_labels: Dict[str, str], quality_metadata: Dict) -> None:
    metadata = quality_metadata if isinstance(quality_metadata, dict) else {}
    ledger = metadata.get("resolved_subject_ledger")
    if not isinstance(ledger, dict):
        ledger = metadata.get("rule_first_subject_ledger")
    if not isinstance(ledger, dict):
        return
    for subject in ledger.get("subjects") or []:
        if not isinstance(subject, dict):
            continue
        subject_id = str(subject.get("subject_id") or "").strip()
        if not subject_id:
            continue
        group_key = f"LEDGER_SUBJECT_{re.sub(r'[^A-Za-z0-9_]+', '_', subject_id).strip('_')}"
        label = str(subject.get("canonical_text") or "").strip()
        if group_key and label:
            group_labels.setdefault(group_key, label)


def _build_suspected_misses_payload(quality_metadata: Dict) -> list[Dict]:
    results: list[Dict] = []
    residual_hits = quality_metadata.get("residual_hits")
    if isinstance(residual_hits, list):
        for item in residual_hits[:8]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": f"疑似遗漏的 {item.get('type') or '实体'} 提及",
                    "severity": "medium",
                    "reason": str(item.get("line") or item.get("window_text") or "当前文本中仍存在未覆盖变体。").strip(),
                    "evidence_refs": [],
                    "action_hint": "建议人工确认该片段是否应补充进当前实体列表。",
                }
            )

    consistency_issues = quality_metadata.get("consistency_issues")
    if isinstance(consistency_issues, list) and consistency_issues:
        results.append(
            {
                "title": "替换或主体归并一致性待确认",
                "severity": "medium",
                "reason": "系统检测到同一主体可能存在多个替换表达，建议人工核对。",
                "evidence_refs": [],
                "action_hint": "建议检查同一主体、简称和角色称谓是否已统一。",
            }
        )

    return results


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


def _analysis_progress_bands() -> dict[str, tuple[int, int]]:
    return {
        "prepare": (32, 36),
        "base": (36, 56),
        "recall": (56, 68),
        "specialized": (68, 76),
        "deep_review": (76, 86),
        "quality_gate": (86, 93),
        "arbitration": (93, 96),
        "finalize": (96, 98),
        "review": (98, 99),
    }


def _parse_progress_bands() -> dict[str, tuple[int, int]]:
    return {
        "workflow_preflight": (10, 11),
        "audit_prepare": (11, 12),
        "wps_semantic_parse": (12, 14),
        "workflow_plan": (14, 15),
        "page_render": (15, 18),
        "ppocr_anchor": (18, 25),
        "ppstructure_layout": (25, 29),
        "glm_ocr_verify": (29, 34),
        "glmocr_sdk_parse": (29, 34),
        "paddleocr_vl_fallback": (34, 36),
        "evidence_align": (36, 37),
        "audit_write_docx": (37, 38),
        "workflow_postcheck": (38, 39),
        "audit_done": (39, 40),
        "anchor_preflight": (12, 18),
        "layout_analysis": (18, 21),
        "glm_page": (21, 28),
        "glm_block": (21, 28),
        "vl_fallback": (28, 30),
        "docx_render": (30, 31),
        "audit_parse_template": (31, 33),
        "audit_align": (33, 35),
    }


def _parse_progress_callback(task_id: str):
    progress_bands = _parse_progress_bands()

    def _callback(payload: Dict[str, object]) -> None:
        task = tasks.get(task_id)
        if task is None:
            return
        if str(task.get("status") or "").strip().lower() in {"ready", "completed", "failed"}:
            return

        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        message = str(payload.get("message") or "").strip() or task.get("message") or "正在解析文本..."
        progress = int(task.get("progress") or 10)
        band = progress_bands.get(stage)
        if band is not None:
            start, end = band
            if total > 0:
                ratio = min(max(current / total, 0.0), 1.0)
                progress = start + int((end - start) * ratio)
            else:
                progress = max(progress, start)
        _set_task_state(task, status="parsing", progress=progress, message=message)

    return _callback


def _batch_parse_progress_callback(task: Dict, item: Dict, progress_prefix: str):
    progress_bands = _parse_progress_bands()

    def _callback(payload: Dict[str, object]) -> None:
        if str(item.get("status") or "").strip().lower() in {"completed", "failed"}:
            return
        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        item_message = str(payload.get("message") or "").strip() or str(item.get("message") or "正在解析文本...")
        progress = int(item.get("progress") or 10)
        band = progress_bands.get(stage)
        if band is not None:
            start, end = band
            if total > 0:
                ratio = min(max(current / total, 0.0), 1.0)
                progress = start + int((end - start) * ratio)
            else:
                progress = max(progress, start)
        _set_batch_item_state(item, status="parsing", progress=progress, message=item_message)
        _refresh_batch_task_progress(task, f"{progress_prefix}，{item_message}")

    return _callback


def _analysis_progress_callback(task_id: str):
    progress_bands = _analysis_progress_bands()

    def _callback(payload: Dict[str, object]) -> None:
        task = tasks.get(task_id)
        if task is None:
            return
        if str(task.get("status") or "").strip().lower() in {"ready", "completed", "failed"}:
            return

        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        message = str(payload.get("message") or "").strip() or task.get("message") or "正在执行主识别..."
        progress = int(task.get("progress") or 32)

        band = progress_bands.get(stage)
        if band is not None:
            start, end = band
            if total > 0:
                ratio = min(max(current / total, 0.0), 1.0)
                progress = start + int((end - start) * ratio)
            else:
                progress = max(progress, start)

        _set_task_state(
            task,
            status="analyzing",
            progress=progress,
            message=message,
        )

    return _callback


def _process_progress_bands() -> dict[str, tuple[int, int]]:
    return {
        "waiting": (70, 72),
        "prepare_entities": (72, 90),
        "anonymize_text": (90, 96),
        "export_file": (96, 99),
        "finalize": (99, 100),
    }


def _process_progress_callback(task_id: str):
    progress_bands = _process_progress_bands()

    def _callback(payload: Dict[str, object]) -> None:
        task = tasks.get(task_id)
        if task is None:
            return
        if str(task.get("status") or "").strip().lower() in {"completed", "failed"}:
            return

        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        message = str(payload.get("message") or "").strip() or task.get("message") or "正在生成脱敏结果..."
        progress = int(task.get("progress") or 72)
        band = progress_bands.get(stage)
        if band is not None:
            start, end = band
            if total > 0:
                ratio = min(max(current / total, 0.0), 1.0)
                progress = start + int((end - start) * ratio)
            else:
                progress = max(progress, start)
        status = "anonymizing" if stage in {"anonymize_text", "export_file", "finalize"} else "processing"
        _set_task_state(task, status=status, progress=progress, message=message)

    return _callback


def _batch_analysis_progress_callback(task: Dict, item: Dict, progress_prefix: str):
    progress_bands = _analysis_progress_bands()

    def _callback(payload: Dict[str, object]) -> None:
        if str(item.get("status") or "").strip().lower() in {"completed", "failed"}:
            return

        stage = str(payload.get("stage") or "").strip().lower()
        current = max(0, int(payload.get("current") or 0))
        total = max(0, int(payload.get("total") or 0))
        item_message = str(payload.get("message") or "").strip() or str(item.get("message") or "正在执行主识别...")
        progress = int(item.get("progress") or 38)

        band = progress_bands.get(stage)
        if band is not None:
            start, end = band
            if total > 0:
                ratio = min(max(current / total, 0.0), 1.0)
                progress = start + int((end - start) * ratio)
            else:
                progress = max(progress, start)

        _set_batch_item_state(
            item,
            status="analyzing",
            progress=progress,
            message=item_message,
        )
        _refresh_batch_task_progress(task, f"{progress_prefix}，{item_message}")

    return _callback


async def _run_batch_task(batch_id: str) -> None:
    task = batch_tasks.get(batch_id)
    if task is None:
        return

    items = task.get("items")
    if not isinstance(items, list) or not items:
        _set_task_state(
            task,
            status="failed",
            progress=100,
            message="批量任务中没有可处理文件。",
            error_message="批量任务中没有可处理文件。",
        )
        return

    use_llm = bool(task.get("config", {}).get("use_llm", False))
    use_custom = bool(task.get("config", {}).get("use_custom", False))
    selected_llm_model = task.get("config", {}).get("llm_model")
    selected_desensitize_mode = _task_desensitize_mode(task)
    anonymization_strategy_key = task.get("config", {}).get("anonymization_strategy") or DEFAULT_ANONYMIZATION_STRATEGY
    operator_config = task.get("config", {}).get("operator_config")
    folder_name = str(task.get("folder_name") or "selected-folder")
    output_root = _batch_output_root(batch_id, folder_name)
    task["output_dir"] = str(output_root)

    task["status"] = "processing"
    task["progress"] = 2
    task["message"] = f"已进入批量处理，共 {len(items)} 个文件。"
    task["updated_at"] = datetime.now()

    try:
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            display_name = str(item.get("relative_path") or item.get("filename") or f"文件 {index + 1}")
            progress_prefix = f"正在处理第 {index + 1}/{len(items)} 个文件：{display_name}"

            try:
                _set_batch_item_state(item, status="parsing", progress=10, message="正在解析文本...")
                _refresh_batch_task_progress(task, f"{progress_prefix}，解析文本...")

                with desensitize_mode_context(selected_desensitize_mode):
                    doc_result = await _parse_document_with_optional_progress(
                        str(item["file_path"]),
                        use_llm=use_llm,
                        llm_model=selected_llm_model,
                        progress_callback=_batch_parse_progress_callback(task, item, progress_prefix),
                    )
                metadata = {
                    **(doc_result.get("metadata") or {}),
                    "file_type": str(item.get("suffix") or "").lstrip("."),
                    "desensitize_mode": selected_desensitize_mode,
                }
                normalized_file_path = str(metadata.get("normalized_file_path") or "").strip()
                if normalized_file_path:
                    item["normalized_file_path"] = normalized_file_path
                text = str(doc_result.get("text") or "")
                structure = doc_result.get("structure")

                _set_batch_item_state(item, status="analyzing", progress=38, message="正在执行规则识别与主识别...")
                _refresh_batch_task_progress(task, f"{progress_prefix}，规则识别与主识别...")

                with desensitize_mode_context(selected_desensitize_mode):
                    analysis_payload = await _analyze_entities(
                        text=text,
                        use_llm=use_llm,
                        use_custom=use_custom,
                        llm_model=selected_llm_model,
                        source_metadata=metadata,
                        source_structure=structure,
                        progress_callback=_batch_analysis_progress_callback(task, item, progress_prefix),
                    )
                entities = analysis_payload["entities"]
                metadata.update(analysis_payload.get("analysis_metadata") or {})

                _set_batch_item_state(item, status="anonymizing", progress=72, message="正在生成文本结果...")
                _refresh_batch_task_progress(task, f"{progress_prefix}，生成文本结果...")

                with desensitize_mode_context(selected_desensitize_mode):
                    engine = get_engine()
                    prepare_kwargs: Dict[str, Any] = {
                        "text": text,
                        "entities": _serialize_entities(entities),
                        "use_llm": use_llm,
                        "operator_config": operator_config,
                        "llm_model": selected_llm_model,
                        "anonymization_strategy": anonymization_strategy_key,
                    }
                    if _callable_accepts_keyword(engine.prepare_entities_for_anonymization, "source_metadata"):
                        prepare_kwargs["source_metadata"] = metadata
                    if _callable_accepts_keyword(engine.prepare_entities_for_anonymization, "source_structure"):
                        prepare_kwargs["source_structure"] = structure
                    entities = await engine.prepare_entities_for_anonymization(**prepare_kwargs)
                    quality_metadata = engine.get_last_quality_metadata()
                    entities = _annotate_response_entities(entities, quality_metadata)
                    anonymized_text = await engine.anonymize(
                        text=text,
                        entities=entities,
                        operator_config=operator_config,
                    )
                export_result = document_exporter.export(
                    task_id=str(item.get("item_id") or uuid.uuid4()),
                    source_path=_source_path_for_export(item),
                    original_filename=str(item["filename"]),
                    source_text=text,
                    source_metadata=metadata,
                    source_structure=structure,
                    entities=entities,
                    anonymized_text=anonymized_text,
                    operator_config=operator_config,
                )

                target_path = _build_batch_output_path(
                    output_root,
                    str(item.get("relative_path") or item.get("filename") or ""),
                    folder_name,
                    str(export_result.get("download_name") or ""),
                )
                final_output_path = _move_batch_output(export_result, target_path)
                mapping_final_output_path = None
                if str(export_result.get("mapping_output_path") or "").strip():
                    mapping_target_path = _build_batch_output_path(
                        output_root,
                        str(item.get("relative_path") or item.get("filename") or ""),
                        folder_name,
                        str(export_result.get("mapping_download_name") or ""),
                    )
                    mapping_final_output_path = _move_batch_output(
                        {
                            "output_path": export_result.get("mapping_output_path"),
                        },
                        mapping_target_path,
                    )
                item_metadata = {
                    **metadata,
                    **(quality_metadata or {}),
                    "canonical_groups": _build_canonical_groups_payload(entities, quality_metadata or {}),
                    "suspected_misses": _build_suspected_misses_payload(quality_metadata or {}),
                }
                item.update(
                    {
                        "entities_count": len(entities),
                        "output_filename": export_result.get("download_name"),
                        "mapping_output_filename": export_result.get("mapping_download_name"),
                        "output_file_type": export_result.get("file_type"),
                        "mapping_output_file_type": export_result.get("mapping_file_type"),
                        "preserves_format": bool(export_result.get("preserves_format")),
                        "warning": export_result.get("warning"),
                        "output_path": final_output_path,
                        "mapping_output_path": mapping_final_output_path,
                        "metadata": item_metadata,
                    }
                )
                _set_batch_item_state(item, status="completed", progress=100, message="处理完成。")
                _refresh_batch_task_progress(task, f"{progress_prefix}，处理完成。")
            except Exception as exc:
                logger.error("Batch item processing failed: %s", exc, exc_info=True)
                item_failure_detail = _analysis_failure_detail(exc, "文件处理失败，请检查文档内容或当前配置。")
                _set_batch_item_state(
                    item,
                    status="failed",
                    progress=100,
                    message="文件处理失败。",
                    error_message=item_failure_detail,
                )
                _refresh_batch_task_progress(task, f"{progress_prefix}，处理失败。")

        counts = _count_batch_items(task)
        archive_path, archive_filename = _create_batch_archive(batch_id, str(task.get("folder_name") or ""), output_root)
        task["archive_path"] = archive_path
        task["archive_filename"] = archive_filename

        if counts["succeeded_count"] > 0:
            task["status"] = "completed"
            task["progress"] = 100
            task["message"] = (
                f"批量处理完成，成功 {counts['succeeded_count']} 个，失败 {counts['failed_count']} 个。"
            )
        else:
            task["status"] = "failed"
            task["progress"] = 100
            task["message"] = "批量处理失败，未生成任何脱敏文件。"
            task["error_message"] = BATCH_FAILURE_DETAIL
        task["updated_at"] = datetime.now()
    except asyncio.CancelledError:
        logger.info("Batch processing cancelled: %s", batch_id)
        _mark_batch_task_stopped(task, PAGE_SESSION_STOP_DETAIL)
        raise
    except Exception as exc:
        logger.error("Batch processing failed: %s", exc, exc_info=True)
        task["status"] = "failed"
        task["progress"] = 100
        task["message"] = BATCH_FAILURE_DETAIL
        task["error_message"] = BATCH_FAILURE_DETAIL
        task["updated_at"] = datetime.now()
    finally:
        batch_task_runners.pop(batch_id, None)
        _unregister_page_session_binding(task.get("page_session_id"), batch_id=batch_id)


async def _run_analysis_task(task_id: str) -> None:
    task = tasks.get(task_id)
    if task is None:
        return

    start_time = perf_counter()
    try:
        use_llm = bool(task.get("config", {}).get("use_llm", False))
        use_custom = bool(task.get("config", {}).get("use_custom", False))
        selected_llm_model = task.get("config", {}).get("llm_model")
        selected_desensitize_mode = _task_desensitize_mode(task)

        _set_task_state(task, status="parsing", progress=10, message="正在解析文本...")
        with desensitize_mode_context(selected_desensitize_mode):
            doc_result = await _parse_document_with_optional_progress(
                task["file_path"],
                use_llm=use_llm,
                llm_model=selected_llm_model,
                progress_callback=_parse_progress_callback(task_id),
            )
        task["text"] = doc_result["text"]
        task["structure"] = doc_result.get("structure")
        task["metadata"] = {
            **doc_result.get("metadata", {}),
            "file_type": str(task.get("suffix", "")).lstrip("."),
            "desensitize_mode": selected_desensitize_mode,
        }
        _merge_task_document_metadata(task, doc_result={"metadata": task["metadata"]})

        _set_task_state(task, status="analyzing", progress=32, message="正在执行规则识别与主识别...")
        with desensitize_mode_context(selected_desensitize_mode):
            analysis_payload = await _analyze_entities(
                text=task["text"],
                use_llm=use_llm,
                use_custom=use_custom,
                llm_model=selected_llm_model,
                source_metadata=task.get("metadata"),
                source_structure=task.get("structure"),
                progress_callback=_analysis_progress_callback(task_id),
            )
        entities = analysis_payload["entities"]
        task["metadata"].update(analysis_payload.get("analysis_metadata") or {})
        entities = _annotate_response_entities(entities, task["metadata"])
        task["entities"] = entities
        task["statistics"] = analysis_payload.get("statistics") or {}
        task["analysis_time"] = round(perf_counter() - start_time, 4)
        completion_status, completion_message = _analysis_completion_state(task.get("metadata") or {})

        _set_task_state(
            task,
            status=completion_status,
            progress=100,
            message=completion_message,
        )
    except asyncio.CancelledError:
        logger.info("Analysis task cancelled: %s", task_id)
        _mark_task_stopped(task, PAGE_SESSION_STOP_DETAIL)
        raise
    except Exception as exc:
        logger.error("File analysis failed: %s", exc, exc_info=True)
        failure_detail = _analysis_failure_detail(exc, ANALYZE_FAILURE_DETAIL)
        _set_task_state(
            task,
            status="failed",
            progress=100,
            message=failure_detail,
            error_message=failure_detail,
        )
    finally:
        analysis_task_runners.pop(task_id, None)
        _unregister_page_session_binding(task.get("page_session_id"), task_id=task_id)


def _media_type_for_path(output_path: str) -> str:
    suffix = Path(output_path).suffix.lower()
    if suffix == ".docx":
        return DOCX_MIME_TYPE
    if suffix == ".pdf":
        return PDF_MIME_TYPE
    return TEXT_MIME_TYPE


def _download_name(filename: str, output_path: str) -> str:
    return f"{Path(filename).stem}_anonymized{Path(output_path).suffix}"


@router.post("/upload", response_model=Union[AnalyzeResponse, TaskStatus])
async def upload_and_analyze(
    request: Request,
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
    use_custom: bool = Form(True),
    llm_model: Optional[str] = Form(None),
    anonymization_strategy: Optional[str] = Form(None),
    desensitize_mode: Optional[str] = Form(None),
    async_mode: bool = Form(False),
    page_session_id: Optional[str] = Form(None),
):
    logger.info("Received upload request")
    _purge_terminal_tasks()
    use_llm = _query_bool_override(request, "use_llm", use_llm)
    use_custom = _query_bool_override(request, "use_custom", use_custom)
    async_mode = _query_bool_override(request, "async_mode", async_mode)
    llm_model = _query_str_override(request, "llm_model", llm_model)
    anonymization_strategy = _query_str_override(request, "anonymization_strategy", anonymization_strategy)
    desensitize_mode = _query_str_override(request, "desensitize_mode", desensitize_mode)
    page_session_id = _query_str_override(request, "page_session_id", page_session_id)

    task_id = str(uuid.uuid4())
    content = await file.read()
    filename = file.filename or "unknown.txt"
    selected_desensitize_mode = _resolve_desensitize_mode(desensitize_mode)
    with desensitize_mode_context(selected_desensitize_mode):
        selected_llm_model = _ensure_model_ready(llm_model) if use_llm else None
        strategy_key, strategy_label = _get_strategy_payload(selected_llm_model)
        anonymization_strategy_key, anonymization_strategy_label = _get_anonymization_strategy_payload(
            anonymization_strategy or DEFAULT_ANONYMIZATION_STRATEGY
        )

    saved_file_path: Optional[str] = None
    try:
        file_path, suffix = _save_upload(task_id, filename, content)
        saved_file_path = file_path

        if not async_mode:
            start_time = perf_counter()
            with desensitize_mode_context(selected_desensitize_mode):
                doc_result = await _parse_document(
                    file_path,
                    use_llm=use_llm,
                    llm_model=selected_llm_model,
                )
            metadata = {
                **doc_result["metadata"],
                "file_type": suffix.lstrip("."),
                "desensitize_mode": selected_desensitize_mode,
            }
            text = doc_result["text"]
            structure = doc_result.get("structure")

            with desensitize_mode_context(selected_desensitize_mode):
                analysis_payload = await _analyze_entities(
                    text=text,
                    use_llm=use_llm,
                    use_custom=use_custom,
                    llm_model=selected_llm_model,
                    source_metadata=metadata,
                    source_structure=structure,
                )
            entities = analysis_payload["entities"]
            metadata.update(analysis_payload.get("analysis_metadata") or {})
            entities = _annotate_response_entities(entities, metadata)
            statistics = analysis_payload.get("statistics") or {}
            analysis_time = round(perf_counter() - start_time, 4)
            completion_status, completion_message = _analysis_completion_state(metadata)

            task_data = {
                "task_id": task_id,
                "filename": filename,
                "file_path": file_path,
                "normalized_file_path": str(metadata.get("normalized_file_path") or "").strip() or None,
                "text": text,
                "entities": entities,
                "statistics": statistics,
                "metadata": metadata,
                "structure": structure,
                "status": completion_status,
                "progress": 100,
                "message": completion_message,
                "created_at": datetime.now(),
                "analysis_time": analysis_time,
                "suffix": suffix,
                "config": {
                    "use_llm": use_llm,
                    "use_custom": use_custom,
                    "llm_model": selected_llm_model,
                    "llm_strategy": strategy_key,
                    "anonymization_strategy": anonymization_strategy_key,
                    "llm_strategy_label": strategy_label,
                    "anonymization_strategy_label": anonymization_strategy_label,
                    "desensitize_mode": selected_desensitize_mode,
                },
            }
            _sync_task_page_count_hint(
                task_data,
                resolve_large_document_page_count(
                    source_metadata=metadata,
                    source_structure=structure,
                ),
            )
            tasks[task_id] = task_data
            _persist_task_state(task_id, task_data)

            return AnalyzeResponse(
                task_id=task_id,
                filename=filename,
                text=text,
                entities=[Entity(**entity) for entity in entities],
                statistics=statistics,
                metadata={
                    **metadata,
                    "canonical_groups": _build_canonical_groups_payload(entities, metadata),
                    "suspected_misses": _build_suspected_misses_payload(metadata),
                },
                llm_model=selected_llm_model,
                llm_strategy=strategy_key,
                llm_strategy_label=strategy_label,
                anonymization_strategy=anonymization_strategy_key,
                anonymization_strategy_label=anonymization_strategy_label,
            )

        task_data = {
            "task_id": task_id,
            "filename": filename,
            "file_path": file_path,
            "created_at": datetime.now(),
            "suffix": suffix,
            "status": "queued",
            "progress": 5,
            "message": "文件已上传，正在准备识别任务...",
            "config": {
                "use_llm": use_llm,
                "use_custom": use_custom,
                "llm_model": selected_llm_model,
                "llm_strategy": strategy_key,
                "anonymization_strategy": anonymization_strategy_key,
                "llm_strategy_label": strategy_label,
                "anonymization_strategy_label": anonymization_strategy_label,
                "desensitize_mode": selected_desensitize_mode,
            },
        }
        task_data["page_session_id"] = _register_page_session_binding(page_session_id, task_id=task_id)
        tasks[task_id] = task_data
        _persist_task_state(task_id, task_data)
        with desensitize_mode_context(selected_desensitize_mode):
            analysis_task_runners[task_id] = asyncio.create_task(_run_analysis_task(task_id))

        return _task_status_response(task_id, task_data)
    except HTTPException:
        raise
    except Exception as exc:
        _delete_if_managed(saved_file_path, allowed_parent=settings.UPLOAD_DIR)
        logger.error("File analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=_analysis_failure_detail(exc, ANALYZE_FAILURE_DETAIL))


@router.post("/batch/upload", response_model=BatchTaskStatus)
async def upload_and_process_folder(
    request: Request,
    files: List[UploadFile] = File(...),
    relative_paths: Optional[List[str]] = Form(None),
    folder_name: Optional[str] = Form(None),
    operator_config_json: Optional[str] = Form(None),
    use_llm: bool = Form(True),
    use_custom: bool = Form(True),
    llm_model: Optional[str] = Form(None),
    anonymization_strategy: Optional[str] = Form(None),
    desensitize_mode: Optional[str] = Form(None),
    page_session_id: Optional[str] = Form(None),
):
    logger.info("Received batch upload request")
    _purge_terminal_tasks()
    use_llm = _query_bool_override(request, "use_llm", use_llm)
    use_custom = _query_bool_override(request, "use_custom", use_custom)
    llm_model = _query_str_override(request, "llm_model", llm_model)
    anonymization_strategy = _query_str_override(request, "anonymization_strategy", anonymization_strategy)
    desensitize_mode = _query_str_override(request, "desensitize_mode", desensitize_mode)
    page_session_id = _query_str_override(request, "page_session_id", page_session_id)

    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一个文件。")

    batch_id = str(uuid.uuid4())
    selected_desensitize_mode = _resolve_desensitize_mode(desensitize_mode)
    with desensitize_mode_context(selected_desensitize_mode):
        selected_llm_model = _ensure_model_ready(llm_model) if use_llm else None
        strategy_key, strategy_label = _get_strategy_payload(selected_llm_model)
        anonymization_strategy_key, anonymization_strategy_label = _get_anonymization_strategy_payload(
            anonymization_strategy or DEFAULT_ANONYMIZATION_STRATEGY
        )
    operator_config: Optional[Dict[str, Any]] = None
    if operator_config_json:
        try:
            parsed_config = json.loads(operator_config_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="批量处理配置不是合法 JSON。") from exc
        if parsed_config is not None and not isinstance(parsed_config, dict):
            raise HTTPException(status_code=400, detail="批量处理配置必须是对象。")
        operator_config = parsed_config

    saved_items: List[Dict[str, Any]] = []
    normalized_relative_paths: List[str] = []
    try:
        for index, upload_file in enumerate(files):
            raw_filename = upload_file.filename or f"document-{index + 1}.txt"
            relative_candidate = (
                relative_paths[index]
                if isinstance(relative_paths, list) and index < len(relative_paths)
                else raw_filename
            )
            normalized_relative_path = _normalize_relative_upload_path(relative_candidate, raw_filename)
            normalized_relative_paths.append(normalized_relative_path)

            content = await upload_file.read()
            item_id = f"{batch_id}-{index + 1}-{uuid.uuid4().hex[:8]}"
            file_path, suffix = _save_upload(item_id, raw_filename, content)
            saved_items.append(
                {
                    "item_id": item_id,
                    "filename": Path(raw_filename).name,
                    "relative_path": normalized_relative_path,
                    "file_path": file_path,
                    "suffix": suffix,
                    "status": "queued",
                    "progress": 0,
                    "message": "等待批量处理开始...",
                    "created_at": datetime.now(),
                }
            )

        resolved_folder_name = _resolve_folder_name(folder_name, normalized_relative_paths)
        batch_task = {
            "batch_id": batch_id,
            "folder_name": resolved_folder_name,
            "status": "queued",
            "progress": 2,
            "message": f"文件夹已接收，共 {len(saved_items)} 个文件，正在准备批量处理...",
            "created_at": datetime.now(),
            "items": saved_items,
            "config": {
                "use_llm": use_llm,
                "use_custom": use_custom,
                "llm_model": selected_llm_model,
                "llm_strategy": strategy_key,
                "llm_strategy_label": strategy_label,
                "anonymization_strategy": anonymization_strategy_key,
                "anonymization_strategy_label": anonymization_strategy_label,
                "operator_config": operator_config,
                "desensitize_mode": selected_desensitize_mode,
            },
        }
        batch_task["page_session_id"] = _register_page_session_binding(page_session_id, batch_id=batch_id)
        batch_tasks[batch_id] = batch_task
        with desensitize_mode_context(selected_desensitize_mode):
            batch_task_runners[batch_id] = asyncio.create_task(_run_batch_task(batch_id))
        return _batch_task_status_response(batch_id, batch_task)
    except HTTPException:
        for item in saved_items:
            _delete_if_managed(item.get("file_path"), allowed_parent=settings.UPLOAD_DIR)
        raise
    except Exception as exc:
        for item in saved_items:
            _delete_if_managed(item.get("file_path"), allowed_parent=settings.UPLOAD_DIR)
        logger.error("Batch upload failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=BATCH_FAILURE_DETAIL)


@router.get("/result/{task_id}", response_model=AnalyzeResponse)
async def get_analysis_result(task_id: str):
    _purge_terminal_tasks()
    task = _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")

    status = str(task.get("status") or "").strip().lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or ANALYZE_FAILURE_DETAIL)
    if status not in {"ready", "review_required", "review_failed", "processing", "anonymizing", "completed", "completed_with_warnings"}:
        raise HTTPException(status_code=409, detail="Analysis is still running. Please retry shortly.")

    if not task.get("text"):
        await _ensure_task_document_loaded(task)

    if not task.get("text") or task.get("entities") is None:
        raise HTTPException(status_code=409, detail="Analysis result is not ready yet.")

    return _serialize_analysis_result(task_id, task)


@router.get("/batch/status/{batch_id}", response_model=BatchTaskStatus)
async def get_batch_task_status(batch_id: str):
    _purge_terminal_tasks()
    task = batch_tasks.get(batch_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Batch task was not found.")
    return _batch_task_status_response(batch_id, task)


@router.get("/batch/result/{batch_id}", response_model=BatchResult)
async def get_batch_result(batch_id: str):
    _purge_terminal_tasks()
    task = batch_tasks.get(batch_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Batch task was not found.")

    status = str(task.get("status") or "").strip().lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or BATCH_FAILURE_DETAIL)
    if status != "completed":
        raise HTTPException(status_code=409, detail="Batch processing is still running. Please retry shortly.")

    return _serialize_batch_result(batch_id, task)


@router.post("/session/heartbeat", response_model=PageSessionResponse)
async def heartbeat_page_session(request: PageSessionRequest):
    page_session_id = _touch_page_session(request.page_session_id)
    if not page_session_id:
        raise HTTPException(status_code=400, detail="页面会话无效。")
    return PageSessionResponse(ok=True)


@router.post("/session/close", response_model=PageSessionResponse)
async def close_page_session(request: PageSessionRequest):
    page_session_id = _normalize_page_session_id(request.page_session_id)
    if not page_session_id:
        raise HTTPException(status_code=400, detail="页面会话无效。")

    _cancel_page_session_tasks(page_session_id, reason=PAGE_SESSION_STOP_DETAIL)
    session = page_sessions.get(page_session_id)
    if session is not None:
        session["closed"] = True
        session["last_seen_at"] = datetime.now()
    _remove_page_session_if_idle(page_session_id)
    return PageSessionResponse(ok=True)


@router.post("/process", response_model=Union[DesensitizeResponse, TaskStatus])
@router.post("/anonymize", response_model=Union[DesensitizeResponse, TaskStatus])
async def anonymize_text(request: DesensitizeRequest):
    logger.info("Received process request: %s", request.task_id)

    task = _get_task(request.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found. Please upload the file again.")
    task_status = str(task.get("status") or "").strip().lower()
    has_analysis_payload = bool(task.get("text")) and task.get("entities") is not None
    if task_status not in {"ready", "review_required", "review_failed", "processing", "anonymizing", "completed", "completed_with_warnings"}:
        if not (task_status == "failed" and has_analysis_payload):
            raise HTTPException(status_code=409, detail="识别尚未完成，请先等待识别结果准备完成。")

    task_config = task.get("config") if isinstance(task.get("config"), dict) else {}
    selected_desensitize_mode = _resolve_desensitize_mode(
        task_config.get("desensitize_mode") or request.desensitize_mode
    )
    if not isinstance(task.get("config"), dict):
        task["config"] = {}
        task_config = task["config"]
    task_config["desensitize_mode"] = selected_desensitize_mode
    use_llm = bool(task_config.get("use_llm", False))
    with desensitize_mode_context(selected_desensitize_mode):
        selected_llm_model = (
            _ensure_model_ready(task_config.get("llm_model") or request.llm_model)
            if use_llm
            else None
        )
        strategy_key, strategy_label = _get_strategy_payload(selected_llm_model)
        anonymization_strategy_key, anonymization_strategy_label = _get_anonymization_strategy_payload(
            request.anonymization_strategy
            or task_config.get("anonymization_strategy")
            or DEFAULT_ANONYMIZATION_STRATEGY
        )
    existing_page_session_id = _normalize_page_session_id(task.get("page_session_id"))
    current_page_session_id = _touch_page_session(request.page_session_id) or existing_page_session_id
    if current_page_session_id and current_page_session_id != existing_page_session_id:
        _unregister_page_session_binding(existing_page_session_id, task_id=request.task_id)
    task["page_session_id"] = current_page_session_id
    current_request = asyncio.current_task()
    request_key: Optional[str] = None
    if current_page_session_id and current_request is not None:
        request_key = f"process:{request.task_id}:{id(current_request)}"
        active_request_tasks[request_key] = current_request
        _register_page_session_binding(
            current_page_session_id,
            task_id=request.task_id,
            request_key=request_key,
        )

    await _ensure_task_document_loaded(task)
    task["config"]["llm_model"] = selected_llm_model
    task["config"]["llm_strategy"] = strategy_key
    task["config"]["llm_strategy_label"] = strategy_label
    task["config"]["anonymization_strategy"] = anonymization_strategy_key
    task["config"]["anonymization_strategy_label"] = anonymization_strategy_label
    task["config"]["desensitize_mode"] = selected_desensitize_mode
    request_payload = request.model_dump()
    request_payload["llm_model"] = selected_llm_model
    request_payload["anonymization_strategy"] = anonymization_strategy_key
    request_payload["desensitize_mode"] = selected_desensitize_mode

    try:
        _clear_task_processing_artifacts(task)
        task["pending_process_request"] = request_payload
        task["error_message"] = None
        _set_task_state(task, status="anonymizing", progress=96, message="正在后台生成脱敏结果并导出...")

        if request.async_mode:
            process_runner = process_task_runners.get(request.task_id)
            if process_runner is None or process_runner.done():
                process_task_runners[request.task_id] = asyncio.create_task(_run_process_task(request.task_id))
            return _task_status_response(request.task_id, task)

        await _run_process_task(request.task_id)
        task = _get_task(request.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task was not found. Please upload the file again.")
        if str(task.get("status") or "").strip().lower() == "failed":
            raise HTTPException(
                status_code=500,
                detail=task.get("error_message") or ANONYMIZE_FAILURE_DETAIL,
            )
        return _serialize_desensitize_result(request.task_id, task)
    except asyncio.CancelledError:
        logger.info("Desensitization request cancelled: %s", request.task_id)
        _mark_task_stopped(task, PAGE_SESSION_STOP_DETAIL)
        raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Desensitization failed: %s", exc, exc_info=True)
        _set_task_state(
            task,
            status="failed",
            progress=100,
            message=ANONYMIZE_FAILURE_DETAIL,
            error_message=ANONYMIZE_FAILURE_DETAIL,
        )
        raise HTTPException(status_code=500, detail=ANONYMIZE_FAILURE_DETAIL)
    finally:
        if request_key:
            active_request_tasks.pop(request_key, None)
        _unregister_page_session_binding(
            current_page_session_id,
            task_id=request.task_id,
            request_key=request_key,
        )


@router.get("/processed-result/{task_id}", response_model=DesensitizeResponse)
async def get_processed_result(task_id: str):
    _purge_terminal_tasks()
    task = _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")

    status = str(task.get("status") or "").strip().lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or ANONYMIZE_FAILURE_DETAIL)
    if status not in {"completed", "completed_with_warnings", "review_failed"}:
        raise HTTPException(status_code=409, detail="Desensitization is still running. Please retry shortly.")
    return _serialize_desensitize_result(task_id, task)


@router.get("/download/mapping/{task_id}")
async def download_mapping_result(task_id: str):
    logger.info("Received mapping download request: %s", task_id)
    _purge_terminal_tasks()

    task = _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")
    status = str(task.get("status") or "").strip().lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or ANONYMIZE_FAILURE_DETAIL)
    if status not in {"completed", "completed_with_warnings", "review_failed"}:
        raise HTTPException(status_code=409, detail="Desensitization is still running. Please retry shortly.")

    filename = task["filename"]
    output_path = task.get("mapping_output_path")
    download_name = task.get("mapping_output_filename")
    media_type = task.get("mapping_output_media_type")

    if not output_path or not filename:
        raise HTTPException(status_code=400, detail="The mapping document has not been exported yet.")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="The mapping document no longer exists.")

    return FileResponse(
        output_path,
        filename=download_name or _download_name(filename, output_path),
        media_type=media_type or _media_type_for_path(output_path),
    )


@router.get("/download/{task_id}")
async def download_result(task_id: str):
    logger.info("Received download request: %s", task_id)
    _purge_terminal_tasks()

    task = _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")
    status = str(task.get("status") or "").strip().lower()
    if status == "failed":
        raise HTTPException(status_code=500, detail=task.get("error_message") or ANONYMIZE_FAILURE_DETAIL)
    if status not in {"completed", "completed_with_warnings", "review_failed"}:
        raise HTTPException(status_code=409, detail="Desensitization is still running. Please retry shortly.")

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


@router.get("/batch/download/{batch_id}")
async def download_batch_archive(batch_id: str):
    logger.info("Received batch archive download request: %s", batch_id)
    _purge_terminal_tasks()

    task = batch_tasks.get(batch_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Batch task was not found.")

    archive_path = task.get("archive_path")
    archive_filename = task.get("archive_filename")
    if not archive_path:
        raise HTTPException(status_code=400, detail="The batch archive is not available.")
    if not os.path.exists(archive_path):
        raise HTTPException(status_code=404, detail="The batch archive no longer exists.")

    return FileResponse(
        archive_path,
        filename=archive_filename or f"{batch_id}_anonymized.zip",
        media_type="application/zip",
    )


@router.get("/batch/download/{batch_id}/{item_id}/mapping")
async def download_batch_item_mapping(batch_id: str, item_id: str):
    logger.info("Received batch mapping download request: %s/%s", batch_id, item_id)
    _purge_terminal_tasks()

    task = batch_tasks.get(batch_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Batch task was not found.")

    items = task.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=404, detail="Batch item was not found.")
    item = next(
        (
            candidate
            for candidate in items
            if isinstance(candidate, dict) and str(candidate.get("item_id") or "") == item_id
        ),
        None,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Batch item was not found.")

    output_path = item.get("mapping_output_path")
    output_filename = item.get("mapping_output_filename")
    if not output_path:
        raise HTTPException(status_code=400, detail="The batch mapping document has not been exported yet.")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="The exported batch mapping document no longer exists.")

    return FileResponse(
        output_path,
        filename=output_filename or Path(output_path).name,
        media_type=_media_type_for_path(output_path),
    )


@router.get("/batch/download/{batch_id}/{item_id}")
async def download_batch_item(batch_id: str, item_id: str):
    logger.info("Received batch item download request: %s/%s", batch_id, item_id)
    _purge_terminal_tasks()

    task = batch_tasks.get(batch_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Batch task was not found.")

    items = task.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=404, detail="Batch item was not found.")
    item = next(
        (
            candidate
            for candidate in items
            if isinstance(candidate, dict) and str(candidate.get("item_id") or "") == item_id
        ),
        None,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Batch item was not found.")

    output_path = item.get("output_path")
    output_filename = item.get("output_filename")
    if not output_path:
        raise HTTPException(status_code=400, detail="The batch item has not been exported yet.")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="The exported batch item no longer exists.")

    return FileResponse(
        output_path,
        filename=output_filename or Path(output_path).name,
        media_type=_media_type_for_path(output_path),
    )


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    _purge_terminal_tasks()
    task = _get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")

    return _task_status_response(task_id, task)


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
async def get_available_models(desensitize_mode: Optional[str] = None):
    with desensitize_mode_context(_resolve_desensitize_mode(desensitize_mode)):
        return _get_model_catalog()


@router.get("/runtime-status", response_model=RuntimeStatusResponse)
async def get_runtime_status(desensitize_mode: Optional[str] = None):
    with desensitize_mode_context(_resolve_desensitize_mode(desensitize_mode)):
        return _get_runtime_status()


@router.get("/info")
async def get_engine_info(desensitize_mode: Optional[str] = None):
    try:
        with desensitize_mode_context(_resolve_desensitize_mode(desensitize_mode)):
            engine = get_engine()
            return engine.get_engine_info()
    except Exception as exc:
        logger.error("Failed to get engine info: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=ENGINE_INFO_FAILURE_DETAIL)
