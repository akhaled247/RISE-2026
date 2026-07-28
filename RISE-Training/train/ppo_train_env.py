"""Train PPO via installed SafePO (editable in-repo Safe-Policy-Optimization).

Pass flags on one line (or with ``\\`` continuations):
  python train/ppo_train_env.py --task PointLTL1MASAR1WC-v0 --seed 0 \\
      --total-steps 40000 --num-envs 1 --steps-per-epoch 2000 --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rise_training.safepo.cli import main

if __name__ == "__main__":
    # Override argv default algo by injecting if --algo absent
    if not any(a.startswith("--algo") for a in sys.argv[1:]):
        sys.argv.extend(["--algo", "ppo"])
    main("ppo")
