"""Runtime environment probing helpers for desktop onboarding."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _extract_ollama_app_bundle(raw_path: str) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.name == "Ollama.app":
        return candidate

    if "Ollama.app" not in candidate.parts:
        return None

    app_index = candidate.parts.index("Ollama.app")
    return Path(*candidate.parts[: app_index + 1])


def find_ollama_app_bundle() -> Path | None:
    if sys.platform != "darwin":
        return None

    raw_candidates = [
        os.getenv("OLLAMA_APP_PATH", "").strip(),
        os.getenv("OLLAMA_PATH", "").strip(),
        "/Applications/Ollama.app",
        str(Path.home() / "Applications" / "Ollama.app"),
    ]
    for raw_candidate in raw_candidates:
        if not raw_candidate:
            continue

        app_bundle = _extract_ollama_app_bundle(raw_candidate)
        if app_bundle is not None and app_bundle.exists():
            return app_bundle

    return None


def find_ollama_command() -> str | None:
    resolved = shutil.which("ollama")
    if resolved:
        return resolved

    candidates: list[Path] = []
    custom_path = os.getenv("OLLAMA_PATH", "").strip()
    if custom_path:
        custom_app_bundle = _extract_ollama_app_bundle(custom_path)
        if custom_app_bundle is not None:
            candidates.append(custom_app_bundle / "Contents" / "Resources" / "ollama")
        else:
            candidates.append(Path(custom_path))

    if sys.platform == "win32":
        candidates.extend(
            [
                Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                Path(os.getenv("ProgramFiles", "")) / "Ollama" / "ollama.exe",
                Path(os.getenv("ProgramFiles(x86)", "")) / "Ollama" / "ollama.exe",
            ]
        )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path("/Applications/Ollama.app/Contents/Resources/ollama"),
                Path.home() / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
                Path("/opt/homebrew/bin/ollama"),
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
                Path("/opt/homebrew/bin/ollama"),
            ]
        )

    seen_candidates: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        if candidate.is_file():
            return str(candidate)

    return None


def platform_label() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def installer_hint() -> str:
    current_platform = platform_label()
    if current_platform == "windows":
        return "请先运行安装包，然后从桌面快捷方式或开始菜单启动客户端。"
    if current_platform == "macos":
        return "首次建议运行 start.command 完成授权，之后可直接打开 contract-desensitize.app。"
    return "请先运行打包目录中的启动脚本或主程序。"


def download_hint(model_name: str) -> str:
    current_platform = platform_label()
    if current_platform == "windows":
        return f"可运行 download_ollama_model.bat，或手动执行：ollama pull {model_name}"
    if current_platform == "macos":
        return f"可运行 download_ollama_model.command，或手动执行：ollama pull {model_name}"
    return f"请手动执行：ollama pull {model_name}"


def detected_ollama_path() -> str | None:
    app_bundle = find_ollama_app_bundle()
    if app_bundle is not None:
        return str(app_bundle)
    return find_ollama_command()
