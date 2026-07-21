import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Allow `python train/ppo_train_env.py` from SpecRLBench root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_vec
from ppo_load_env import eval_model

# --- edit these before each run ---
env_names = [
    "PointLTL0MASAR1-v0",
    "PointLTL4MASAR1-v0",
    "PointLTL4MASAR1WC-v0",
    "PointLTL5MASAR1-v0",
    "PointLTL6MASAR1-v0",
    "PointLTL6MASAR1WC-v0",
]
envs_timesteps = [
    1_000_000,
    2_500_000,
    4_000_000,
    3_000_000,
    5_000_000,
    5_000_000,
]
env_name = "PointLTL6MASAR1-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")


def train(
    total_timesteps=5_000_000,
    seed=0,
    e_name=env_name,
    n_envs=8,
    ent_coef=0.02,
    learning_rate=5e-5,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    clip_range=0.2,
    target_kl=0.05,  # 0.08
    gamma=0.995,
    gae_lambda=0.98,
    startup_log=True,
) -> tuple[str, str]:
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    training_log_path = f"./_training_logs/{e_name}_tensorboard/"
    tb_log_name = (
        f"PPO_t{name_time}"
        f"_st{n_steps}"
        f"_bs{batch_size}"
        f"_tt{total_timesteps / 1_000_000:.1f}M"
        f"_ec{ent_coef}"
        f"_lr{learning_rate}"
        f"_ep{n_epochs}"
        f"_cr{clip_range}"
        f"_kl{target_kl}"
        f"_γ{gamma}"
        f"_λ{gae_lambda}"
        f"_s{seed}"
    )
    log_dir = f"{training_log_path}{tb_log_name}"

    if startup_log:
        print(f"Logging to {training_log_path}...")
        print(
            f"<<<{rollout_steps / batch_size}>>> minibatches per rollout"
            f"\n <<<{total_timesteps // rollout_steps}>>> policy updates total"
        )
        print("=" * 40)
        print(f"train env={e_name} device={device} steps={total_timesteps}")
        print(
            f"PPO iter = {rollout_steps} env steps collect + {n_epochs} epochs x "
            f"{rollout_steps // batch_size} minibatches "
            f"- SB3 iters/s scales ~1/n_steps"
        )

    env = make_vec(e_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)  # constant env seed across master seeds
    env.reset()

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
        tensorboard_log=training_log_path,
        seed=seed,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    env.seed(seed=0)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    model_path = f"_models/ppo_{name_time}_{e_name}_{seed}"
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
            seed=i,
            startup_log=True,
            e_name=e_name,
            total_timesteps=total_timesteps,
        )
        eval_model(env_name=e_name, render_mode=None, m_path=model_path)

