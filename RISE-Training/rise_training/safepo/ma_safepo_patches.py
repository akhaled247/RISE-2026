"""Lightweight SafePO MA bootstrap: farama filter + spawn before CUDA.

Vec-env spawn/IPC fixes live in Safe-Policy-Optimization/safepo/common/wrappers.py
(module-level shareworker — required for pickling under spawn).
"""

from __future__ import annotations

import multiprocessing as mp
import sys

_PATCHED = False


def _install_farama_filter() -> None:
    from rise_training.safepo import farama_filter

    sys.modules["safepo.common.farama_filter"] = farama_filter  # type: ignore[assignment]
    farama_filter.silence_farama_adroit_spam()


def _ensure_spawn_start_method() -> None:
    if mp.get_start_method(allow_none=True) is not None:
        return
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def apply_ma_safepo_patches() -> None:
    """Idempotent: farama filter + spawn start method before SafePO MA train."""
    global _PATCHED
    if _PATCHED:
        return
    _install_farama_filter()
    _ensure_spawn_start_method()
    _PATCHED = True
