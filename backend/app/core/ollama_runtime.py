"""Best-effort local Ollama service startup helpers."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import requests

from app.core.runtime_probe import find_ollama_app_bundle, find_ollama_command

logger = logging.getLogger(__name__)

_START_ATTEMPTED = False
_STARTED_PROCESS: subprocess.Popen | None = None


def _tags_url(base_url: str) -> str:
    return f"{str(base_url or '').rstrip('/')}/api/tags"


def is_ollama_service_ready(base_url: str, *, timeout: float = 1.0) -> bool:
    try:
        response = requests.get(_tags_url(base_url), timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _session_kwargs() -> dict[str, object]:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def _start_candidates() -> list[tuple[str, Sequence[str], Path | None, bool]]:
    candidates: list[tuple[str, Sequence[str], Path | None, bool]] = []
    if sys.platform == "darwin":
        app_bundle = find_ollama_app_bundle()
        if app_bundle is not None:
            candidates.append(("Ollama app bundle", ["open", str(app_bundle)], None, False))

    command = find_ollama_command()
    if command:
        command_path = Path(command).resolve()
        candidates.append(("Ollama service", [str(command_path), "serve"], command_path.parent, True))
    return candidates


def ensure_ollama_service_running(
    base_url: str,
    *,
    startup_timeout: float = 20.0,
    poll_interval: float = 0.5,
    force_retry: bool = False,
) -> bool:
    """Start a locally installed Ollama service if it is not already reachable."""

    global _START_ATTEMPTED, _STARTED_PROCESS

    if is_ollama_service_ready(base_url):
        return True
    if _START_ATTEMPTED and not force_retry:
        return is_ollama_service_ready(base_url)

    _START_ATTEMPTED = True
    candidates = _start_candidates()
    if not candidates:
        logger.warning("Ollama is not reachable and no local Ollama installation was detected.")
        return False

    for label, command, cwd, keep_process in candidates:
        try:
            logger.info("Starting %s for backend model review: %s", label, " ".join(command))
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                **_session_kwargs(),
            )
        except Exception as exc:
            logger.warning("Failed to start %s: %s", label, exc, exc_info=True)
            continue

        if keep_process:
            _STARTED_PROCESS = process

        deadline = time.monotonic() + max(1.0, float(startup_timeout or 1.0))
        while time.monotonic() < deadline:
            if is_ollama_service_ready(base_url):
                logger.info("%s is ready for backend model review.", label)
                return True
            if keep_process and process.poll() is not None:
                logger.warning("%s exited before Ollama became reachable.", label)
                break
            time.sleep(max(0.1, float(poll_interval or 0.5)))

        logger.warning("%s did not become reachable before startup timeout.", label)

    return is_ollama_service_ready(base_url)
