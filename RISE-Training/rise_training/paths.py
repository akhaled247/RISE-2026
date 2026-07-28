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


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _prepend_sys_path(path: str) -> None:
    """Move ``path`` to the front of ``sys.path`` (spawn workers see fork first)."""
    path = os.path.abspath(path)
    sys.path[:] = [p for p in sys.path if _norm_path(p) != _norm_path(path)]
    sys.path.insert(0, path)


def _purge_stale_safety_gymnasium() -> None:
    """Drop a cached PyPI ``safety_gymnasium`` so the vendored fork can load."""
    mod = sys.modules.get("safety_gymnasium")
    if mod is None:
        return
    expected = _norm_path(str(_SG_ROOT))
    mod_file = getattr(mod, "__file__", None)
    if mod_file and _norm_path(mod_file).startswith(expected):
        return
    mod_paths = getattr(mod, "__path__", None)
    if mod_paths is not None:
        for entry in mod_paths:
            if _norm_path(str(entry)).startswith(expected):
                return
    stale = [
        name
        for name in list(sys.modules)
        if name == "safety_gymnasium" or name.startswith("safety_gymnasium.")
    ]
    for name in stale:
        sys.modules.pop(name, None)


def ensure_specrlbench_paths() -> None:
    """Prepend SpecRLBench root + vendored safety-gymnasium (+ SafePO) to sys.path.

    Also mirrors into ``PYTHONPATH`` so spawn/fork workers inherit the same roots
    (Linux ``ShareSubprocVecEnv`` uses spawn; children do not copy parent ``sys.path``).
    """
    roots = (str(_SPECRL_ROOT), str(_SG_ROOT), str(_SAFEPO_ROOT))
    for p in reversed(roots):
        _prepend_sys_path(p)
    _purge_stale_safety_gymnasium()

    cur = os.environ.get("PYTHONPATH", "")
    parts = [x for x in cur.split(os.pathsep) if x]
    changed = False
    for p in reversed(roots):
        if p not in parts:
            parts.insert(0, p)
            changed = True
    if changed:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
