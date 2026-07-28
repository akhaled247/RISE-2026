"""Evaluate SafePO SpecRLBench multi-agent checkpoints (post-train).

Examples:
  cd RISE-Training
  python eval_safepo_ma_env.py --run-dir ./_training_logs/safepo/PointLTL0MASAR2-v0/mappo/seed-000-... --eval-episodes 50 --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rise_training.paths import ensure_specrlbench_paths

ensure_specrlbench_paths()

from rise_training.safepo.evaluate_ma import main

if __name__ == "__main__":
    main()
