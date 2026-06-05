"""Local macOS Vision OCR helper for scanned PDF pages."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.runtime_security import ensure_private_directory, ensure_private_file


class MacOSVisionOCRService:
    """Use the macOS Vision framework through a small Swift helper."""

    model = "macos_vision"

    def __init__(self, *, timeout: int) -> None:
        self.timeout = max(1, int(timeout or 30))
        self.available = self.is_supported()

    @classmethod
    def is_supported(cls) -> bool:
        if platform.system().lower() != "darwin":
            return False
        return cls._bundled_binary_path() is not None or (
            cls._source_script_path().exists() and bool(cls._find_xcrun())
        )

    @staticmethod
    def _source_script_path() -> Path:
        return Path(__file__).resolve().parent.parent / "resources" / "macos_vision_ocr.swift"

    @staticmethod
    def _bundled_binary_path() -> Optional[Path]:
        resource_root = Path(settings.RESOURCE_ROOT)
        candidates = [
            resource_root / "bin" / "macos_vision_ocr",
            Path(getattr(sys, "_MEIPASS", resource_root)) / "bin" / "macos_vision_ocr",
            Path(__file__).resolve().parent.parent.parent / "bin" / "macos_vision_ocr",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _helper_binary_path() -> Path:
        runtime_root = Path(settings.RUNTIME_ROOT)
        bin_dir = runtime_root / "bin"
        ensure_private_directory(bin_dir)
        return bin_dir / "macos_vision_ocr"

    @staticmethod
    def _find_xcrun() -> Optional[str]:
        return subprocess.run(
            ["bash", "-lc", "command -v xcrun"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or None

    @staticmethod
    def _infer_image_suffix(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        return ".img"

    @classmethod
    def _compile_helper_binary(cls) -> Path:
        binary_path = cls._helper_binary_path()
        bundled_binary = cls._bundled_binary_path()
        if bundled_binary is not None:
            if (
                not binary_path.exists()
                or binary_path.stat().st_mtime < bundled_binary.stat().st_mtime
                or binary_path.stat().st_size != bundled_binary.stat().st_size
            ):
                ensure_private_directory(binary_path.parent)
                shutil.copy2(bundled_binary, binary_path)
            binary_path.chmod(0o700)
            return binary_path

        source_path = cls._source_script_path()
        if binary_path.exists() and binary_path.stat().st_mtime >= source_path.stat().st_mtime:
            binary_path.chmod(0o700)
            return binary_path

        xcrun = cls._find_xcrun()
        if not xcrun:
            raise RuntimeError("xcrun is unavailable for macOS Vision OCR")

        ensure_private_directory(binary_path.parent)
        command = [xcrun, "swiftc", str(source_path), "-O", "-o", str(binary_path)]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "failed_to_compile_macos_vision_ocr")

        binary_path.chmod(0o700)
        return binary_path

    async def extract_document_text_from_image_async(
        self,
        image_bytes: bytes,
        *,
        page_number: int | None = None,
        total_pages: int | None = None,
    ) -> Dict[str, Any]:
        if not image_bytes:
            return {"text": "", "quality": "low", "warnings": ["empty_image_input"]}
        if not self.available:
            raise RuntimeError("macOS Vision OCR is unavailable.")

        binary_path = await asyncio.to_thread(self._compile_helper_binary)
        suffix = self._infer_image_suffix(image_bytes)
        temp_fd, temp_name = tempfile.mkstemp(prefix="vision-ocr-", suffix=suffix)
        os.close(temp_fd)
        temp_path = Path(temp_name)
        try:
            temp_path.write_bytes(image_bytes)
            ensure_private_file(temp_path)
            process = await asyncio.create_subprocess_exec(
                str(binary_path),
                str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise TimeoutError("macOS Vision OCR request timed out")
            except asyncio.CancelledError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise

            if process.returncode != 0:
                raise RuntimeError((stderr or b"").decode("utf-8", errors="ignore").strip() or "macOS Vision OCR failed")

            parsed = json.loads((stdout or b"{}").decode("utf-8", errors="ignore") or "{}")
            if not isinstance(parsed, dict):
                raise RuntimeError("invalid_macos_vision_ocr_payload")

            warnings = parsed.get("warnings")
            if not isinstance(warnings, list):
                warnings = []

            return {
                "text": str(parsed.get("text", "") or "").strip(),
                "quality": str(parsed.get("quality", "medium") or "medium").strip().lower(),
                "layout": str(parsed.get("layout", "plain_text") or "plain_text").strip().lower(),
                "lines": parsed.get("lines") if isinstance(parsed.get("lines"), list) else [],
                "blocks": [],
                "warnings": [str(item).strip() for item in warnings if str(item).strip()],
                "engine": self.model,
                "page_number": page_number,
                "total_pages": total_pages,
            }
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
