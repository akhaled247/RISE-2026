"""Evaluate SafePO SpecRLBench checkpoints (post-train).

Examples:
  python SpecRLBench/eval_safepo_env.py --run-dir SpecRLBench/_training_logs/safepo/specrlbench/PointLTL4MASAR1WC-v0/ppo/seed-000-2026-07-21-19-59-19 --eval-episodes 50
  python SpecRLBench/eval_safepo_env.py --benchmark-dir SpecRLBench/_training_logs/safepo/specrlbench --eval-episodes 50
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backends.safepo.evaluate import main

if __name__ == "__main__":
    main()
