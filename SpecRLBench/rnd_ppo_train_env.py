"""Train RNDPPO on SpecRLBench environments (drop-in parallel to ppo_train_env)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_load_env import eval_model
from rnd import RNDConfig, RNDPPO
from utils.env_utils import make_vec

# --- edit these before each run ---
env_name = "PointLTL5MASAR1-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/rnd_ppo_{env_name}_tensorboard/"

USE_RND = True
INTRINSIC_REWARD_COEF = 0.01


def train(
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    n_envs: int = 8,
    ent_coef: float = 0.02,
    learning_rate: float = 5e-5,
    n_steps: int = 4096,
    batch_size: int = 256,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    target_kl: float = 0.05,
    use_rnd: bool = USE_RND,
    intrinsic_reward_coef: float = INTRINSIC_REWARD_COEF,
    startup_log: bool = True,
) -> tuple[str, str]:
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print(
            f"<<<{rollout_steps / batch_size}>>> minibatches per rollout"
            f"\n <<<{total_timesteps // rollout_steps}>>> policy updates total"
        )
        print("=" * 40)
        print(f"train env={env_name} device={device} steps={total_timesteps} use_rnd={use_rnd}")
        print(f"intrinsic_reward_coef={intrinsic_reward_coef}")

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)
    env.reset()

    rnd_config = RNDConfig(
        use_rnd=use_rnd,
        intrinsic_reward_coef=intrinsic_reward_coef,
        feature_dim=128,
        predictor_learning_rate=1e-4,
    )

    if use_rnd:
        model = RNDPPO(
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
        algo_tag = "RNDPPO"
    else:
        # Pure PPO baseline (unchanged path)
        model = PPO(
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
        )
        algo_tag = "PPO"

    env.seed(seed=0)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        tb_log_name=(
            f"{algo_tag}_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps / 1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_beta{intrinsic_reward_coef}"
            f"_s{seed}"
        ),
    )

    model_path = f"_models/rnd_ppo_{name_time}_{env_name}_{seed}"
    if not use_rnd:
        model_path = f"_models/ppo_{name_time}_{env_name}_{seed}"
    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    model_path, _ = train(
        seed=0,
        startup_log=True,
        total_timesteps=5_000_000,
    )
    eval_model(
        env_name=env_name,
        render_mode=None,
        m_path=model_path,
    )
