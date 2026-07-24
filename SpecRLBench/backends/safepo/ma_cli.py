"""CLI for SpecRLBench + SafePO multi-agent algorithms."""

from __future__ import annotations

import argparse
import sys
from distutils.util import strtobool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.safepo.ma_runners import train_with_safepo_ma


def _str2bool(v: str) -> bool:
    return bool(strtobool(v))


def build_ma_parser(default_algo: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"SpecRLBench + SafePO MA ({default_algo})")
    p.add_argument("--algo", type=str, default=default_algo)
    p.add_argument("--task", "--env-id", dest="task", type=str, default="PointLTL0MASAR2-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--cost-limit", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="./_training_logs/safepo")
    p.add_argument("--experiment", type=str, default="specrlbench")
    p.add_argument("--write-terminal", type=_str2bool, default=True)
    p.add_argument("--use-eval", type=_str2bool, default=False)
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Override YAML entropy_coef (e.g. 0.02 for explore)",
    )
    p.add_argument(
        "--share-policy",
        type=_str2bool,
        default=None,
        help="IPPO shared policy (default True for ippo*)",
    )
    p.add_argument("--model-dir", type=str, default="")
    return p


def main(default_algo: str = "mappo") -> None:
    args = build_ma_parser(default_algo).parse_args()
    train_with_safepo_ma(
        args.algo,
        args.task,
        seed=args.seed,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        cost_limit=args.cost_limit,
        device=args.device,
        device_id=args.device_id,
        log_dir=args.log_dir,
        experiment=args.experiment,
        write_terminal=args.write_terminal,
        use_eval=args.use_eval,
        entropy_coef=args.entropy_coef,
        share_policy=args.share_policy,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
