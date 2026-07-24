"""Ensure SpecRLBench Safety-Gymnasium fork is importable before safepo."""

from __future__ import annotations

import sys
from pathlib import Path

_SPECRL_ROOT = Path(__file__).resolve().parents[2]
_SG_ROOT = _SPECRL_ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"


def ensure_specrlbench_paths() -> None:
    """Prepend SpecRLBench root + vendored safety-gymnasium to sys.path."""
    for p in (str(_SPECRL_ROOT), str(_SG_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
