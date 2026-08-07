"""Spawn-safe PYTHONPATH bootstrap for GenZ multiprocess env workers."""

from __future__ import annotations

from rise_training.paths import ensure_genz_paths as _ensure_genz_paths


def ensure_genz_paths() -> None:
    """Ensure worker interpreters can import GenZ `src` and SpecRLBench."""
    _ensure_genz_paths()
