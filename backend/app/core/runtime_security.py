"""Helpers for local runtime paths and file permissions."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def packaged_runtime_data_dir(app_data_dir_name: str) -> Path:
    if sys.platform == "win32":
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / app_data_dir_name
        return Path.home() / app_data_dir_name

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_data_dir_name

    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / app_data_dir_name
    return Path.home() / ".local" / "share" / app_data_dir_name


def ensure_private_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    _set_private_mode(directory, 0o700)
    return directory


def ensure_private_file(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.exists():
        _set_private_mode(file_path, 0o600)
    return file_path


def _set_private_mode(path: Path, mode: int) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except (FileNotFoundError, PermissionError, OSError):
        return
