"""Train scripts → installed SafePO algorithms + SpecRLBench env hook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# SpecRLBench root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.safepo.runners import train_with_safepo


def build_parser(default_algo: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"SpecRLBench + SafePO ({default_algo})")
    p.add_argument("--algo", type=str, default=default_algo)
    p.add_argument("--task", "--env-id", dest="task", type=str, default="PointLTL4MASAR1WC-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=20000)
    p.add_argument("--cost-limit", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="./_training_logs/safepo")
    p.add_argument("--experiment", type=str, default="specrlbench")
    p.add_argument("--lagrangian-multiplier-init", type=float, default=0.001)
    p.add_argument("--lagrangian-multiplier-lr", type=float, default=0.035)
    return p


def main(default_algo: str = "ppo") -> None:
    args = build_parser(default_algo).parse_args()
    train_with_safepo(
        args.algo,
        args.task,
        seed=args.seed,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        steps_per_epoch=args.steps_per_epoch,
        cost_limit=args.cost_limit,
        device=args.device,
        device_id=args.device_id,
        log_dir=args.log_dir,
        experiment=args.experiment,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
    )


if __name__ == "__main__":
    main("ppo")
