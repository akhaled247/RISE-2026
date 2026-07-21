"""Train PPO via installed SafePO (pip install safepo)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.safepo.cli import main

if __name__ == "__main__":
    # Override argv default algo by injecting if --algo absent
    if not any(a.startswith("--algo") for a in sys.argv[1:]):
        sys.argv.extend(["--algo", "ppo"])
    main("ppo")
