"""Best-effort runtime memory release helpers for local low-memory inference."""

from __future__ import annotations

import gc
import sys
from typing import Any


def _call_noarg(func: Any) -> None:
    if not callable(func):
        return
    try:
        func()
    except Exception:
        return


def release_runtime_memory() -> None:
    """Release Python, PyTorch and MLX caches without importing new runtimes.

    Importing torch or mlx only to clear memory can itself increase RSS. This
    helper only touches runtimes that are already loaded in the current process.
    """
    gc.collect()

    torch = sys.modules.get("torch")
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            is_available = getattr(cuda, "is_available", None)
            try:
                if callable(is_available) and is_available():
                    _call_noarg(getattr(cuda, "empty_cache", None))
            except Exception:
                pass

        mps = getattr(torch, "mps", None)
        if mps is not None:
            _call_noarg(getattr(mps, "empty_cache", None))

    mlx_core = sys.modules.get("mlx.core")
    if mlx_core is not None:
        _call_noarg(getattr(mlx_core, "clear_cache", None))
        metal = getattr(mlx_core, "metal", None)
        _call_noarg(getattr(metal, "clear_cache", None))

    gc.collect()
