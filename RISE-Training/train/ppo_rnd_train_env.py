"""Train RND-PPO via stock SafePO PPO + RISE-RND env wrapper.

Intrinsic reward is added inside ``rise_rnd.SafetyRNDWrapper``; SafePO ``ppo.main``
is unchanged. Pass flags on one line (or with ``\\`` continuations):

  python train/ppo_rnd_train_env.py --task PointLTL1MASAR1-v0 --seed 0 \\
      --total-steps 40000 --num-envs 8 --steps-per-epoch 16384 --device cpu \\
      --rnd-coef 0.5 --ent-coef 0.005 --sweep S2
"""

from __future__ import annotations

import argparse
import sys
from distutils.util import strtobool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rise_training.safepo.cli import build_parser
from rise_training.safepo.config import SafePOTrainConfig
from rise_training.safepo.env_hook import disable_rnd_wrapper, enable_rnd_wrapper
from rise_training.safepo.runners import train_with_safepo
from rise_rnd import RNDConfig

_CFG = SafePOTrainConfig()

# S0–S4: (intrinsic_reward_coef, ent_coef, use_rnd)
SWEEP_TABLE: dict[str, tuple[float, float, bool]] = {
    "S0": (0.0, 0.02, False),
    "S1": (0.1, 0.01, True),
    "S2": (0.5, 0.005, True),
    "S3": (1.0, 0.0, True),
    "S4": (0.5, 0.0, True),
}


def _str2bool(v: str) -> bool:
    return bool(strtobool(v))


def build_rnd_parser() -> argparse.ArgumentParser:
    p = build_parser("ppo")
    p.description = "SpecRLBench RND-PPO (SafePO stock PPO + RISE-RND wrapper)"
    p.add_argument(
        "--sweep",
        type=str,
        default="S2",
        choices=list(SWEEP_TABLE),
        help="Preset sweep id (beta, ent_coef, use_rnd)",
    )
    p.add_argument(
        "--rnd-coef",
        type=float,
        default=None,
        help="Override intrinsic reward coef (beta); default from --sweep",
    )
    p.add_argument(
        "--use-rnd",
        type=_str2bool,
        default=None,
        help="Override use_rnd; default from --sweep",
    )
    return p


def main() -> None:
    args = build_rnd_parser().parse_args()
    beta, ent, use_rnd = SWEEP_TABLE[args.sweep]
    if args.rnd_coef is not None:
        beta = args.rnd_coef
    if args.use_rnd is not None:
        use_rnd = args.use_rnd
    if args.ent_coef == _CFG.ent_coef and args.sweep in SWEEP_TABLE:
        ent_coef = ent
    else:
        ent_coef = args.ent_coef

    rnd_config = RNDConfig(
        use_rnd=use_rnd,
        intrinsic_reward_coef=beta,
        feature_dim=256,
        predictor_learning_rate=1e-4,
    )
    enable_rnd_wrapper(
        rnd_config,
        args.device,
        args.steps_per_epoch,
        args.num_envs,
    )
    try:
        train_with_safepo(
            "ppo",
            args.task,
            seed=args.seed,
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            steps_per_epoch=args.steps_per_epoch,
            cost_limit=args.cost_limit,
            device=args.device,
            device_id=args.device_id,
            log_dir=args.log_dir.replace("/safepo", "/safepo_rnd"),
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
            ent_coef=ent_coef,
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
    finally:
        disable_rnd_wrapper()


if __name__ == "__main__":
    main()
