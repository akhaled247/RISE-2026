"""CLI for SpecRLBench + SafePO multi-agent algorithms."""

from __future__ import annotations

import argparse
import sys
from distutils.util import strtobool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.safepo.config import MA_IPPO_RECIPE_B, MA_SPECRL_RECIPE_B, SafePOTrainConfig
from backends.safepo.ma_runners import train_with_safepo_ma

_CFG = SafePOTrainConfig()


def _str2bool(v: str) -> bool:
    return bool(strtobool(v))


def build_ma_parser(default_algo: str) -> argparse.ArgumentParser:
    default_ent = (
        MA_IPPO_RECIPE_B["entropy_coef"]
        if default_algo.startswith("ippo")
        else MA_SPECRL_RECIPE_B["entropy_coef"]
    )
    p = argparse.ArgumentParser(description=f"SpecRLBench + SafePO MA ({default_algo})")
    p.add_argument("--algo", type=str, default=default_algo)
    p.add_argument("--task", "--env-id", dest="task", type=str, default="PointLTL0MASAR2-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=4_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--cost-limit", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="./_training_logs/safepo")
    p.add_argument("--experiment", type=str, default="specrlbench")
    p.add_argument("--write-terminal", type=_str2bool, default=False)
    p.add_argument(
        "--use-tensorboard",
        type=_str2bool,
        default=True,
        help="Toggles TensorBoard SummaryWriter in EpochLogger (writes log_dir/tb)",
    )
    p.add_argument("--use-eval", type=_str2bool, default=False)
    p.add_argument(
        "--entropy-coef",
        "--ent-coef",
        dest="entropy_coef",
        type=float,
        default=default_ent,
        help="Entropy bonus (default 0.0 for IPPO, 0.02 for MAPPO recipe-B)",
    )
    p.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Rollout horizon per PPO epoch (default 4096 for SpecRL MASAR; env max still 2500)",
    )
    p.add_argument(
        "--learning-iters",
        type=int,
        default=None,
        help="PPO epochs per rollout (default 10 for SpecRL MASAR)",
    )
    p.add_argument(
        "--share-policy",
        type=_str2bool,
        default=None,
        help="IPPO shared policy (default True for ippo*)",
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Run deterministic inline eval every N epochs (IPPO default 25 on SpecRL; 0=off)",
    )
    p.add_argument("--model-dir", type=str, default="")
    p.add_argument(
        "--save-model-freq",
        type=int,
        default=_CFG.save_model_freq,
        help="Checkpoint every N training epochs (default 10); always also saves epoch 0 and last",
    )
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
        use_tensorboard=args.use_tensorboard,
        use_eval=args.use_eval,
        entropy_coef=args.entropy_coef,
        episode_length=args.episode_length,
        learning_iters=args.learning_iters,
        share_policy=args.share_policy,
        model_dir=args.model_dir,
        save_model_freq=args.save_model_freq,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main()
