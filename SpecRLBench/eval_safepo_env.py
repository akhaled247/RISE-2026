"""Evaluate SafePO SpecRLBench checkpoints (post-train).

Examples:
  python SpecRLBench/eval_safepo_env.py --run-dir SpecRLBench/_training_logs/safepo/PointLTL5MASAR1WC-v0/ppo/seed-000-2026-07-22-07-59-09 --eval-episodes 50
  python SpecRLBench/eval_safepo_env.py --benchmark-dir SpecRLBench/_training_logs/safepo --eval-episodes 50

Stdout (Q33.D — pasteable; no sidecar JSON):

  EVAL_PATH=/abs/.../seed-000-2026-07-22-11-47-36
  After 50 episodes evaluation, the ppo in PointLTL5MASAR1WC-v0 reward: 0.68±0.47, cost: 0.32±0.47, ep_len: 227.84±87.76, rescue: 68.0%
"""
from __future__ import annotations

import sys
from pathlib import Path

"""
    python ~/RISE-2026/SpecRLBench/eval_safepo_env.py --run-dir \
    ~/RISE-2026/SpecRLBench/_training_logs/safepo/PointLTL5MASAR1WC-v0/ppo_lag/seed-000-2026-07-24-09-32-50 \
    --eval-episodes 50
"""

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backends.safepo.evaluate import main

if __name__ == "__main__":
    main()
