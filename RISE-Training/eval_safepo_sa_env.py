"""Evaluate SafePO SpecRLBench single-agent checkpoints (post-train).

Examples:
  cd RISE-Training
  python eval_safepo_sa_env.py --run-dir ./_training_logs/safepo/PointLTL2MASAR1WC-v0/ppo/seed-000-... --eval-episodes 50

Writes eval_summary.json next to the run dir and prints EVAL_PATH + metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rise_training.paths import ensure_specrlbench_paths

ensure_specrlbench_paths()

from rise_training.safepo.evaluate import main

if __name__ == "__main__":
    main()
