"""Train SACLagrangian on SpecRLBench WC sparse SAR.

Modeled after ``train/sac_train_env.py`` + Lag knobs from ``ppo_lag_train_env.py``.
Saves under ``_models/sac_lag_*`` so ``ppo_load_env.py`` routes to ``SACLag.load``.

EpCost dual (consistent with PPO/TRPO); actor uses ``+ penalty * Qc``.
EpCost: WC wrapper should set info['cost'] from cost_walls (see #28).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from stable_baselines3.common.logger import configure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_load_env import eval_model
from sac_lagrangian import SACLag
from utils.env_utils import make_vec

# --- edit these before each run ---
env_name = "PointLTL5MASAR1WC-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/{env_name}_tensorboard/"


def train(
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    learning_starts: int = 10_000,
    batch_size: int = 256,
    train_freq: int = 8,
    gradient_steps: int = 8,
    tau: float = 0.005,
    gamma: float = 0.995,
    ent_coef: str | float = "auto",
    cost_lim: float = 0.0,
    penalty_init: float = 1.0,
    penalty_lr: float = 2.5e-2,
    startup_log: bool = True,
) -> tuple[str, str]:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = f"_models/sac_lag_{name_time}_{env_name}_{seed}"
    tb_log_name = (
        f"SAC_Lag_t{name_time}"
        f"_bs{batch_size}"
        f"_tt{total_timesteps / 1_000_000:.1f}M"
        f"_lr{learning_rate}"
        f"_g{gamma}"
        f"_s{seed}"
        f"_plr{penalty_lr}"
    )
    log_dir = f"{TRAINING_LOG_PATH}{tb_log_name}"

    if startup_log:
        print("=" * 40)
        print(f"train env={env_name} device={device} steps={total_timesteps}")
        print(
            f"SACLag buffer={buffer_size} cost_lim={cost_lim} "
            f"penalty_lr={penalty_lr} penalty_init={penalty_init}"
        )
        print(f"Logging to {TRAINING_LOG_PATH}...")

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)
    env.reset()

    model = SACLag(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        tau=tau,
        gamma=gamma,
        ent_coef=ent_coef,
        device=device,
        seed=seed,
        tensorboard_log=TRAINING_LOG_PATH,
        policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.ReLU),
        cost_lim=cost_lim,
        penalty_init=penalty_init,
        penalty_lr=penalty_lr,
        verbose=0,
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    env.seed(seed=0)
    model.learn(total_timesteps=total_timesteps, log_interval=1, progress_bar=True, tb_log_name=tb_log_name)

    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    model_path, _ = train(seed=0, startup_log=True, total_timesteps=1_000_000)
    eval_model(env_name=env_name, render_mode=None, m_path=model_path)
