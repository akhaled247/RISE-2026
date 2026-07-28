"""Ensure SpecRLBench Safety-Gymnasium fork is importable before safepo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_RISE_ROOT = Path(__file__).resolve().parents[2]
_SPECRL_ROOT = _RISE_ROOT / "SpecRLBench"
_SG_ROOT = _SPECRL_ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"
_SAFEPO_ROOT = _RISE_ROOT / "Safe-Policy-Optimization"
_TRAINING_ROOT = _RISE_ROOT / "RISE-Training"


def specrl_path_roots() -> tuple[str, str, str]:
    """Absolute SpecRLBench, safety-gymnasium fork, and SafePO roots."""
    return str(_SPECRL_ROOT), str(_SG_ROOT), str(_SAFEPO_ROOT)


def default_log_dir() -> str:
    """Default SafePO run log root under RISE-Training."""
    env = os.environ.get("RISE_LOG_ROOT")
    if env:
        return env
    return str(_TRAINING_ROOT / "_training_logs" / "safepo")


def ensure_specrlbench_paths() -> None:
    """Prepend SpecRLBench root + vendored safety-gymnasium (+ SafePO) to sys.path.

    Also mirrors into ``PYTHONPATH`` so spawn/fork workers inherit the same roots
    (Linux ``ShareSubprocVecEnv`` uses spawn; children do not copy parent ``sys.path``).
    """
    roots = (str(_SPECRL_ROOT), str(_SG_ROOT), str(_SAFEPO_ROOT))
    for p in roots:
        if p not in sys.path:
            sys.path.insert(0, p)

    cur = os.environ.get("PYTHONPATH", "")
    parts = [x for x in cur.split(os.pathsep) if x]
    changed = False
    for p in reversed(roots):
        if p not in parts:
            parts.insert(0, p)
            changed = True
    if changed:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
