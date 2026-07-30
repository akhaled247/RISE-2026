"""Evaluate SafePO SA checkpoints on a multi-agent SAR env (paper deploy).

Examples:
  cd RISE-Training
  python eval_safepo_sa_on_ma_env.py \\
    --run-dir ./_training_logs/safepo/PointLTL0MASAR1WC-v0/ppo/seed-000-... \\
    --eval-env PointLTL0MASAR2WC-v0 --eval-episodes 50
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rise_training.paths import ensure_specrlbench_paths

ensure_specrlbench_paths()

from rise_training.safepo.evaluate_sa_on_ma import eval_single_run


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Deploy SA SafePO policy on MA SAR env")
    p.add_argument("--run-dir", required=True, help="SA training run directory")
    p.add_argument(
        "--eval-env",
        default="PointLTL0MASAR2WC-v0",
        help="MA eval env (default: paper protocol MASAR2WC)",
    )
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render-mode", default=None)
    p.add_argument(
        "--sar-ltl-ordering",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override sar_ltl_ordering (default: from train config.json)",
    )
    args = p.parse_args(argv)

    out_path = eval_single_run(
        args.run_dir,
        eval_env=args.eval_env,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        sar_ltl_ordering=args.sar_ltl_ordering,
        render_mode=args.render_mode
    )
    print(f"EVAL_MA_DEPLOY_PATH={out_path}")


if __name__ == "__main__":
    main()
