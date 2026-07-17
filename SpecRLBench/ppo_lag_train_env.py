"""Train PPOLagrangian on SpecRLBench WC (wall-terminate) sparse SAR.

Modeled after ``ppo_train_env.py``. Saves under ``_models/ppo_lag_*`` so
``ppo_load_env.py`` routes to ``PPOLagrangian.load``.

Edit knobs under ``--- edit these ---`` before each run.

Sweep (lag-only; run L4WC S0→S3 first, then best 1–2 on L5WC / L6WC):

| Run | cost_lim | penalty_lr | penalty_init | Purpose                          |
|-----|----------|------------|--------------|----------------------------------|
| S0  | 0.0      | 1e-2       | 1.0          | Recommended WC default (strict)  |
| S1  | 0.0      | 5e-2       | 1.0          | OpenAI-hot penalty LR            |
| S2  | 0.0      | 5e-3       | 1.0          | Gentle λ (less choke explore)    |
| S3  | 1.0      | 1e-2       | 1.0          | Soft limit (~1 hit/ep mean OK)   |

Lag coeff ranges (freeze PPO knobs; only sweep Lag):
  cost_lim     default 0.0   range {0.0, 0.25, 1.0}
  penalty_lr   default 1e-2  range {5e-3, 1e-2, 5e-2}
  penalty_init default 1.0   range {0.1, 1.0}  (frozen in S0–S3)

Frozen PPO knobs (match ppo_train_env): lr=5e-5, n_steps=2048, n_envs=8,
  batch_size=256, n_epochs=10, clip=0.2, target_kl=0.05, ent_coef=0.02,
  cost_gamma=0.99, cost_gae_lambda=0.97, vf_lr=1e-3, max_ep_len=1000.

Eval: ppo_load_env rescue /50. Also watch train EpCost / Penalty — want EpCost→~0
  without rescue collapsing.

EpCost caveat: WC wrapper does not set top-level info['cost']. default_cost_extractor
  may still see nested cost_walls. If EpCost stays ~0 while agents hit walls,
  constraint is inactive — fix extractor later if needed.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_lagrangian import PPOLagrangian
from ppo_load_env import eval_model
from utils.env_utils import make_vec

# --- edit these before each run ---
# Levels: PointLTL4MASAR1WC-v0 | PointLTL5MASAR1WC-v0 | PointLTL6MASAR1WC-v0
env_name = "PointLTL4MASAR1WC-v0"
# Sweep id: "S0" | "S1" | "S2" | "S3"
SWEEP_RUN = "S0"

name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/ppo_lag_{env_name}_tensorboard/"

# S0–S3: (cost_lim, penalty_lr, penalty_init)
SWEEP_TABLE: dict[str, tuple[float, float, float]] = {
    "S0": (0.0, 1e-2, 1.0),   # recommended WC default
    "S1": (0.0, 5e-2, 1.0),   # OpenAI-hot penalty LR
    "S2": (0.0, 5e-3, 1.0),   # gentle λ
    "S3": (1.0, 1e-2, 1.0),   # soft limit
}


def _level_tag(env_id: str) -> str:
    if "LTL4" in env_id:
        return "L4"
    if "LTL5" in env_id:
        return "L5"
    if "LTL6" in env_id:
        return "L6"
    return "LX"


def train(
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    n_envs: int = 8,
    ent_coef: float = 0.02,
    learning_rate: float = 5e-5,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    target_kl: float = 0.05,
    # Lag knobs overridden by SWEEP_TABLE when sweep_run is set
    cost_lim: float | None = None,
    penalty_init: float | None = None,
    penalty_lr: float | None = None,
    cost_gamma: float = 0.99,
    cost_gae_lambda: float = 0.97,
    vf_lr: float = 1e-3,
    max_ep_len: int = 1000,
    sweep_run: str = SWEEP_RUN,
    startup_log: bool = True,
) -> tuple[str, str]:
    if sweep_run not in SWEEP_TABLE:
        raise KeyError(f"Unknown SWEEP_RUN={sweep_run!r}; choose from {list(SWEEP_TABLE)}")

    cl, plr, pinit = SWEEP_TABLE[sweep_run]
    if cost_lim is not None:
        cl = cost_lim
    if penalty_lr is not None:
        plr = penalty_lr
    if penalty_init is not None:
        pinit = penalty_init

    level = _level_tag(env_name)
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print(
            f"<<<{rollout_steps / batch_size}>>> buffer/batch ratio (OpenAI uses full buffer)"
            f"\n <<<{total_timesteps // rollout_steps}>>> policy updates total"
        )
        print("=" * 40)
        print(
            f"train env={env_name} level={level} sweep={sweep_run} "
            f"device={device} steps={total_timesteps}"
        )
        print(
            f"PPOLag iter = {rollout_steps} env steps collect + "
            f"pi_iters={n_epochs} vf_iters={n_epochs} "
            f"cost_lim={cl} penalty_lr={plr} penalty_init={pinit}"
        )

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)  # constant env seed to reduce variation between master seeds
    env.reset()

    model = PPOLagrangian(
        "MultiInputPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        target_kl=target_kl,
        device=device,
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
        clip_range=clip_range,
        cost_lim=cl,
        penalty_init=pinit,
        penalty_lr=plr,
        cost_gamma=cost_gamma,
        cost_gae_lambda=cost_gae_lambda,
        vf_lr=vf_lr,
        max_ep_len=max_ep_len,
    )

    env.seed(seed=0)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        tb_log_name=(
            f"PPOLag_{level}_{sweep_run}_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps / 1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_ep{n_epochs}"
            f"_cr{clip_range}"
            f"_kl{target_kl}"
            f"_cl{cl}"
            f"_plr{plr}"
            f"_pi{pinit}"
            f"_s{seed}"
        ),
    )

    # ``ppo_lag`` tag required for ppo_load_env.py routing
    model_path = f"_models/ppo_lag_{level}_{sweep_run}_{name_time}_{env_name}_{seed}"
    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    for i in range(1):
        model_path, _ = train(
            seed=int(i),
            startup_log=True,
            total_timesteps=5_000_000,
            sweep_run=SWEEP_RUN,
        )
        eval_model(
            env_name=env_name,
            render_mode=None,
            m_path=model_path,
        )
