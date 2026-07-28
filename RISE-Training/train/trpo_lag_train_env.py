"""Train TRPO-Lagrangian via installed SafePO."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rise_training.safepo.cli import main

if __name__ == "__main__":
    if not any(a.startswith("--algo") for a in sys.argv[1:]):
        sys.argv.extend(["--algo", "trpo_lag"])
    main("trpo_lag")
