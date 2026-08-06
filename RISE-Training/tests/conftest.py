"""Pytest bootstrap: GenZ src + SpecRLBench on sys.path for genz_deploy tests."""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parents[1]
_RISE_ROOT = _TRAINING_ROOT.parent

if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from rise_training.paths import ensure_genz_paths

ensure_genz_paths()
