"""Evaluate SafePO SpecRLBench multi-agent checkpoints (post-train).

Examples:
  python SpecRLBench/eval_safepo_ma_env.py --run-dir SpecRLBench/_training_logs/safepo/PointLTL0MASAR2-v0/mappo/seed-000-... --eval-episodes 50 --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

"""
  python ~/RISE-2026/SpecRLBench/eval_safepo_ma_env.py --run-dir \
  ~/RISE-2026/SpecRLBench/_training_logs/safepo/PointLTL0MASAR2-v0/mappo/seed-000-2026-07-24-23-46-46 \
  --eval-episodes 50 --device cuda
"""

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backends.safepo.evaluate_ma import main

if __name__ == "__main__":
    main()
