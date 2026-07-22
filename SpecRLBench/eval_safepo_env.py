"""Evaluate SafePO SpecRLBench checkpoints (post-train).

Examples:
  python SpecRLBench/eval_safepo_env.py --run-dir SpecRLBench/_training_logs/safepo/PointLTL5MASAR1WC-v0/ppo/seed-000-2026-07-22-07-59-09 --eval-episodes 50
  python SpecRLBench/eval_safepo_env.py --benchmark-dir SpecRLBench/_training_logs/safepo --eval-episodes 50
"""
from __future__ import annotations

import sys
from pathlib import Path

"""
    python ~/RISE-2026/SpecRLBench/eval_safepo_env.py --run-dir \
    ~/RISE-2026/SpecRLBench/_training_logs/safepo/PointLTL5MASAR1WC-v0/ppo/seed-000-2026-07-22-07-52-51 \
    --eval-episodes 50
"""


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backends.safepo.evaluate import main

if __name__ == "__main__":
    main()
