"""Train RNDPPO on SpecRLBench SAR L4 / L5 (sparse-exploration defaults).

Edit the knobs under ``--- edit these ---`` before each run.

Sweep (run per level; recommended order L4 S0→S2, then L5 S0→S2):

| Run | beta | ent_coef | Purpose                          |
|-----|------|----------|----------------------------------|
| S0  | 0.0  | 0.02     | PPO control                      |
| S1  | 0.1  | 0.01     | Weak intrinsic                   |
| S2  | 0.5  | 0.005    | Recommended default              |
| S3  | 1.0  | 0.0      | Strong intrinsic, no entropy     |
| S4  | 0.5  | 0.0      | Default beta, entropy off        |

L5 success bar: beat plain-PPO ~12/50 rescue. L4: beat that level's own S0.
Assumes true-sparse extrinsic reward (you own env sparse switch).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Allow `python train/ppo_rnd_train_env.py` from SpecRLBench root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_load_env import eval_model
from rnd import RNDConfig, RND, resolve_rnd_obs_keys
from utils.env_utils import make_vec

# --- edit these before each run ---
# Levels: PointLTL4MASAR1-v0 (walls only) | PointLTL5MASAR1-v0 (walls + buildings)
env_name = "PointLTL4MASAR1-v0"
# Sweep id: "S0" | "S1" | "S2" | "S3" | "S4"  (see table in module docstring)
SWEEP_RUN = "S2"

name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/ppo_rnd_{env_name}_tensorboard/"

# Level → RND obs key substrings (buildings/walls focus; casualties excluded)
LEVEL_RND_PREFIXES: dict[str, list[str]] = {
    "PointLTL4MASAR1-v0": ["walls", "wall_sensor"],
    "PointLTL4MASAR1WC-v0": ["walls", "wall_sensor"],
    "PointLTL5MASAR1-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
    "PointLTL5MASAR1WC-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
    "PointLTL6MASAR1-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
    "PointLTL6MASAR1WC-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
}

# S0–S4: (intrinsic_reward_coef, ent_coef, use_rnd)
SWEEP_TABLE: dict[str, tuple[float, float, bool]] = {
    "S0": (0.0, 0.02, False),   # pure PPO control
    "S1": (0.1, 0.01, True),
    "S2": (0.5, 0.005, True),  # recommended default
    "S3": (1.0, 0.0, True),
    "S4": (0.5, 0.0, True),
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
    learning_rate: float = 5e-5,
    n_steps: int = 4096,
    batch_size: int = 256,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    target_kl: float = 0.05,
    sweep_run: str = SWEEP_RUN,
    startup_log: bool = True,
) -> tuple[str, str]:
    if sweep_run not in SWEEP_TABLE:
        raise KeyError(f"Unknown SWEEP_RUN={sweep_run!r}; choose from {list(SWEEP_TABLE)}")
    if env_name not in LEVEL_RND_PREFIXES:
        raise KeyError(
            f"env_name={env_name!r} has no RND profile; "
            f"known={list(LEVEL_RND_PREFIXES)}"
        )

    intrinsic_reward_coef, ent_coef, use_rnd = SWEEP_TABLE[sweep_run]
    # beta=0 with use_rnd still builds RND; S0 uses pure PPO for clean control
    level = _level_tag(env_name)
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    tb_log_name = (
            f"RND_{sweep_run}_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps / 1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_beta{intrinsic_reward_coef}"
            f"_s{seed}"
        )
    log_dir = f"{TRAINING_LOG_PATH}{tb_log_name}"

    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print(
            f"<<<{rollout_steps / batch_size}>>> minibatches per rollout"
            f"\n <<<{total_timesteps // rollout_steps}>>> policy updates total"
        )
        print("=" * 40)
        print(
            f"train env={env_name} level={level} sweep={sweep_run} "
            f"device={device} steps={total_timesteps}"
        )
        print(
            f"use_rnd={use_rnd} beta={intrinsic_reward_coef} ent_coef={ent_coef} "
            f"lr={learning_rate}"
        )

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)
    env.reset()

    rnd_obs_keys: list[str] | None = None
    if use_rnd and isinstance(env.observation_space, spaces.Dict):
        rnd_obs_keys = resolve_rnd_obs_keys(
            env.observation_space,
            include_substrings=LEVEL_RND_PREFIXES[env_name],
        )
        if startup_log:
            print(f"RND obs_keys ({len(rnd_obs_keys)}): {rnd_obs_keys}")

    rnd_config = RNDConfig(
        use_rnd=use_rnd,
        intrinsic_reward_coef=intrinsic_reward_coef,
        feature_dim=256,
        predictor_learning_rate=1e-4,
        obs_keys=rnd_obs_keys,
    )

    model = RND(
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
        rnd_config=rnd_config,
    )

    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    env.seed(seed=0)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    model_path = (
        f"_models/ppo_rnd_{level}_{sweep_run}_{name_time}_{env_name}_{seed}"
        if use_rnd
        else f"_models/ppo_{level}_{sweep_run}_{name_time}_{env_name}_{seed}"
    )
    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    for i in range(1):
        # print(f"{i}/5")    
        model_path, _ = train(
            seed=0,
            startup_log=True,
            total_timesteps=5_000_000,
            sweep_run=SWEEP_RUN,
        )
        eval_model(
        env_name=env_name,
        render_mode=None,
        m_path=model_path,
    )
