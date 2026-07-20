"""Train PPOLagrangian (SB3 Tier-2) on SpecRLBench WC sparse SAR.

Modeled after ``ppo_rnd_train_env.py``. Saves under ``_models/ppo_lag_*`` so
``ppo_load_env.py`` routes to ``PPOLagrangian.load`` (.zip).

Edit knobs under ``--- edit these ---`` before each run.

Sweep (lag-only; run L4WC S0→S3 first, then best 1–2 on L5WC / L6WC):

| Run | cost_lim | penalty_lr | penalty_init | Purpose                          |
|-----|----------|------------|--------------|----------------------------------|
| S0  | 0.0      | 1e-2       | 1.0          | Recommended WC default (strict)  |
| S1  | 0.0      | 5e-2       | 1.0          | OpenAI-hot penalty LR            |
| S2  | 0.0      | 5e-3       | 1.0          | Gentle λ (less choke explore)    |
| S3  | 1.0      | 1e-2       | 1.0          | Soft limit (~1 hit/ep mean OK)   |

Frozen PPO knobs: lr=5e-5, n_steps=2048, n_envs=8, batch_size=256, n_epochs=10,
  clip=0.2, target_kl=0.05, ent_coef=0.02, gae_lambda=0.97, lag_mode=openai.

Eval: ppo_load_env rescue /50. Watch EpCost / Penalty — want EpCost→~0
  without rescue collapsing.

EpCost: WC wrapper should set info['cost'] from cost_walls (see #28).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from stable_baselines3.common.logger import configure

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_lagrangian import PPOLagrangian
from ppo_load_env import eval_model
from utils.env_utils import make_vec

# --- edit these before each run ---
# Levels: PointLTL4MASAR1WC-v0 | PointLTL5MASAR1WC-v0 | PointLTL6MASAR1WC-v0
env_name = "PointLTL5MASAR1WC-v0"
# Sweep id: "S0" | "S1" | "S2" | "S3"
SWEEP_RUN = "S1"

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
    cost_lim: float | None = None,
    penalty_init: float | None = None,
    penalty_lr: float | None = None,
    cost_gamma: float = 0.99,
    cost_gae_lambda: float = 0.97,
    vf_lr: float = 1e-3,
    lag_mode: str = "openai",
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
    model_path = f"_models/ppo_lag_{level}_{sweep_run}_{name_time}_{env_name}_{seed}"
    tb_log_name = (
            f"lag_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps / 1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_s{seed}"
            f"_{sweep_run}")
    log_dir = f"{TRAINING_LOG_PATH}{tb_log_name}"

    if startup_log:
        print("=" * 40)
        print(
            f"train env={env_name} level={level} sweep={sweep_run} "
            f"device={device} steps={total_timesteps} lag_mode={lag_mode}"
        )
        print(
            f"PPOLag iter = {rollout_steps} env steps + "
            f"n_epochs={n_epochs} batch_size={batch_size} "
            f"cost_lim={cl} penalty_lr={plr} penalty_init={pinit}"
        )
        print(f"Logging to {TRAINING_LOG_PATH}...")

    env = make_vec(
        env_name,
        n_envs=n_envs,
        render_mode=None,
        sb3=True,
        normalize=True,
        # Avoid CUDA-before-fork deadlocks with SubprocVecEnv on Linux.
        vec_env_kwargs={"start_method": "forkserver"} if sys.platform != "win32" else None,
    )
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)
    env.reset()

    model = PPOLagrangian(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.97,
        ent_coef=ent_coef,
        target_kl=target_kl,
        clip_range=clip_range,
        normalize_advantage=False,
        device=device,
        seed=seed,
        tensorboard_log=TRAINING_LOG_PATH,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.Tanh,
            log_std_init=-0.5,
        ),
        cost_lim=cl,
        penalty_init=pinit,
        penalty_lr=plr,
        cost_gamma=cost_gamma,
        cost_gae_lambda=cost_gae_lambda,
        vf_lr=vf_lr,
        lag_mode=lag_mode,  # type: ignore[arg-type]
        verbose=0,
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))

    env.seed(seed=0)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name=tb_log_name
    )

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
