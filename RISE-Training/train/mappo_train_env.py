"""Train MAPPO via SafePO multi_agent on SpecRLBench MASAR tasks.

  python train/mappo_train_env.py --task PointLTL0MASAR2-v0 --seed 0 \\
      --total-steps 4000000 --num-envs 8 --device cuda --use-tensorboard True
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rise_training.safepo.ma_cli import main

if __name__ == "__main__":
    if not any(a.startswith("--algo") for a in sys.argv[1:]):
        sys.argv.extend(["--algo", "mappo"])
    main("mappo")
