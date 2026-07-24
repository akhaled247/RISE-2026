"""Train scripts → installed SafePO algorithms + SpecRLBench env hook.

CLI tip: pass all flags on one line, or continue lines with ``\\``. A bare
newline before ``--device`` makes the shell run ``--device`` as a command.
"""

from __future__ import annotations

import argparse
import sys
from distutils.util import strtobool
from pathlib import Path

# SpecRLBench root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.safepo.config import ALGO_DEFAULTS, SafePOTrainConfig
from backends.safepo.runners import train_with_safepo

_CFG = SafePOTrainConfig()


def _str2bool(v: str) -> bool:
    """SafePO-style bool: ``True``/``False``/``1``/``0``/``yes``/``no``."""
    return bool(strtobool(v))


def build_parser(default_algo: str) -> argparse.ArgumentParser:
    # Per-algo overrides (e.g. TRPO learning_iters / target_kl)
    algo_over = dict(ALGO_DEFAULTS.get(default_algo, {}))
    target_kl = float(algo_over.get("target_kl", _CFG.target_kl))
    learning_iters = int(algo_over.get("learning_iters", _CFG.learning_iters))

    p = argparse.ArgumentParser(
        description=f"SpecRLBench + SafePO ({default_algo})",
        epilog=(
            "Example (one line): python train/ppo_train_env.py "
            "--task PointLTL1MASAR1WC-v0 --seed 0 --total-steps 40000 "
            "--num-envs 8 --steps-per-epoch 16384 --device cpu "
            "--write-terminal False --use-tensorboard True"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--algo", type=str, default=default_algo)
    p.add_argument("--task", "--env-id", dest="task", type=str, default=_CFG.env_id)
    p.add_argument("--seed", type=int, default=_CFG.seed)
    p.add_argument("--total-steps", type=int, default=_CFG.total_steps)
    p.add_argument("--num-envs", type=int, default=_CFG.num_envs)
    p.add_argument("--steps-per-epoch", type=int, default=_CFG.steps_per_epoch)
    p.add_argument("--cost-limit", type=float, default=_CFG.cost_limit)
    p.add_argument("--device", type=str, default=_CFG.device)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--log-dir", type=str, default=_CFG.log_dir)
    p.add_argument("--experiment", type=str, default=_CFG.experiment)

    # PPO / shared update knobs (SafePO default_cfg + Linux Adam args)
    p.add_argument("--actor-lr", type=float, default=_CFG.actor_lr)
    p.add_argument("--critic-lr", type=float, default=_CFG.critic_lr)
    p.add_argument("--batch-size", type=int, default=_CFG.batch_size)
    p.add_argument("--learning-iters", type=int, default=learning_iters)
    p.add_argument("--target-kl", type=float, default=target_kl)
    p.add_argument("--gamma", type=float, default=_CFG.gamma)
    p.add_argument("--lam", type=float, default=_CFG.lam)
    p.add_argument("--lam-c", type=float, default=_CFG.lam_c)
    p.add_argument("--clip-ratio", type=float, default=_CFG.clip_ratio)
    p.add_argument(
        "--ent-coef",
        type=float,
        default=_CFG.ent_coef,
        help="Entropy bonus coefficient (SB3-style; 0.0 = SafePO default)",
    )
    p.add_argument("--max-grad-norm", type=float, default=_CFG.max_grad_norm)
    p.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=list(_CFG.hidden_sizes),
        help="MLP hidden sizes (default: 64 64)",
    )
    p.add_argument("--lr_end_factor", type=float, default=_CFG.lr_end_factor)
    p.add_argument(
        "--save-model-freq",
        type=int,
        default=_CFG.save_model_freq,
        help="Checkpoint every N epochs (default 10); always also saves epoch 0 and last",
    )

    # Lag
    p.add_argument(
        "--lagrangian-multiplier-init",
        type=float,
        default=_CFG.lagrangian_multiplier_init,
    )
    p.add_argument(
        "--lagrangian-multiplier-lr",
        type=float,
        default=_CFG.lagrangian_multiplier_lr,
    )

    p.add_argument(
        "--write-terminal",
        type=_str2bool,
        default=True,
        help="Toggles terminal logging (False → seed*_terminal.log / error.log)",
    )
    p.add_argument(
        "--use-tensorboard",
        type=_str2bool,
        default=True,
        help="Toggles TensorBoard SummaryWriter in EpochLogger",
    )
    p.add_argument(
        "--parallel",
        type=_str2bool,
        default=_CFG.parallel,
        help="True → SafetyAsyncVectorEnv when num-envs>1 (SB3 Subproc-like); "
        "False → SyncVectorSafetyEnv (serial)",
    )
    return p


def main(default_algo: str = "ppo") -> None:
    args = build_parser(default_algo).parse_args()
    # If user passed --algo different from script default, re-apply ALGO_DEFAULTS
    # only when they did not override learning-iters/target-kl explicitly is hard;
    # runners merge ALGO_DEFAULTS for unspecified keys via train_with_safepo.
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
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        learning_iters=args.learning_iters,
        target_kl=args.target_kl,
        gamma=args.gamma,
        lam=args.lam,
        lam_c=args.lam_c,
        clip_ratio=args.clip_ratio,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_sizes=list(args.hidden_sizes),
        lr_end_factor=args.lr_end_factor,
        save_model_freq=args.save_model_freq,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
        write_terminal=args.write_terminal,
        use_tensorboard=args.use_tensorboard,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main("ppo")
