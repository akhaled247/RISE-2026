"""Evaluate SafePO SpecRLBench single-agent checkpoints (post-train).

Examples:
  python SpecRLBench/eval_safepo_sa_env.py --run-dir SpecRLBench/_training_logs/safepo/PointLTL2MASAR1WC-v0/ppo/seed-000-... --eval-episodes 50

Writes eval_summary.json next to the run dir and prints EVAL_PATH + metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

"""
  python ~/RISE-2026/SpecRLBench/eval_safepo_sa_env.py --run-dir \
  ~/RISE-2026/SpecRLBench/_training_logs/safepo/PointLTL4MASAR1-v0/ppo/seed-000-... \
  --eval-episodes 50
"""

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backends.safepo.evaluate import main

if __name__ == "__main__":
    main()
