"""Train PPOLagrangian (SB3 Tier-2) on SpecRLBench WC sparse SAR.

Modeled after ``train/ppo_rnd_train_env.py``. Saves under ``_models/ppo_lag_*`` so
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

# Allow `python train/ppo_lag_train_env.py` from SpecRLBench root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_lagrangian import PPOLag
from ppo_load_env import eval_model
from utils.env_utils import make_vec

# --- edit these before each run ---
env_names = [
    "PointLTL4MASAR1WC-v0",
    "PointLTL5MASAR1WC-v0",
    "PointLTL6MASAR1WC-v0",
]
envs_timesteps = [
    3_000_000,
    4_000_000,
    5_000_000,
]
env_name = "PointLTL5MASAR1WC-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")


def train(
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    e_name: str = env_name,
    n_envs: int = 8,
    ent_coef: float = 0.02,
    learning_rate: float = 5e-5,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    target_kl: float = 0.05,
    cost_lim: float = 0.0,
    penalty_init: float = 0.25,
    penalty_lr: float = 1e-2,  # [1e-2, 5e-2]
    cost_gamma: float = 0.99,
    cost_gae_lambda: float = 0.97,
    vf_lr: float = 1e-3,
    lag_mode: str = "openai",
    startup_log: bool = True,
) -> tuple[str, str]:
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    training_log_path = f"./_training_logs/{e_name}_tensorboard/"
    model_path = f"_models/ppo_lag_{name_time}_{e_name}_{seed}"
    tb_log_name = (
        f"PPO_Lag_t{name_time}"
        f"_st{n_steps}"
        f"_bs{batch_size}"
        f"_tt{total_timesteps / 1_000_000:.1f}M"
        f"_ec{ent_coef}"
        f"_lr{learning_rate}"
        f"_s{seed}"
        f"_plr{penalty_lr}"
    )
    log_dir = f"{training_log_path}{tb_log_name}"

    if startup_log:
        print("=" * 40)
        print(
            f"train env={e_name} "
            f"device={device} steps={total_timesteps} lag_mode={lag_mode}"
        )
        print(
            f"PPOLag iter = {rollout_steps} env steps + "
            f"n_epochs={n_epochs} batch_size={batch_size} "
            f"cost_lim={cost_lim} penalty_lr={penalty_lr} penalty_init={penalty_init}"
        )
        print(f"Logging to {training_log_path}...")

    env = make_vec(
        e_name,
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

    model = PPOLag(
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
        tensorboard_log=training_log_path,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.Tanh,
            log_std_init=-0.5,
        ),
        cost_lim=cost_lim,
        penalty_init=penalty_init,
        penalty_lr=penalty_lr,
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
        tb_log_name=tb_log_name,
    )

    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    for i, (e_name, total_timesteps) in enumerate(zip(env_names, envs_timesteps)):
        print(total_timesteps)
        model_path, _ = train(
            seed=0,  # i when training multiple of same run
            startup_log=True,
            e_name=e_name,
            total_timesteps=total_timesteps,
        )
        eval_model(env_name=e_name, render_mode=None, m_path=model_path)
