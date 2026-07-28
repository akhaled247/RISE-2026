"""Train IPPO-Lag via SafePO multi_agent on SpecRLBench MASAR tasks.

  python train/ippo_lag_train_env.py --task PointLTL0MASAR2-v0 --seed 0 \\
      --total-steps 400000 --num-envs 8 --device cuda --cost-limit 0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rise_training.safepo.ma_cli import main

if __name__ == "__main__":
    if not any(a.startswith("--algo") for a in sys.argv[1:]):
        sys.argv.extend(["--algo", "ippo_lag"])
    main("ippo_lag")
