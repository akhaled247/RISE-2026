import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

# Allow `python train/ppo_train_env.py` from SpecRLBench root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from utils.env_utils_sb3 import ThroughputCallback, make_vec
from load_env import eval_model

# --- edit these before each run ---
env_names = [
    "PointLTL4MASAR1-v0",
]
envs_timesteps = [
    2_500_000,
]
env_name = "PointLTL4MASAR1-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/{env_name}_tensorboard/"


def train(
        total_timesteps=2_500_000,
        e_name=env_name,
        seed=0,
        n_envs=8,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        tau=0.005,
        gamma=0.995,
        ent_coef="auto",
        startup_log=True,
) -> tuple[str, str]:

    TRAINING_LOG_PATH = f"./_training_logs/{e_name}_tensorboard/"

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    tb_log_name = (
        f"SAC_t{name_time}"
        f"_bs{batch_size}"
        f"_tt{total_timesteps / 1_000_000:.1f}M"
        f"_lr{learning_rate}"
        f"_g{gamma}"
        f"_s{seed}"
    )

    log_dir = f"{TRAINING_LOG_PATH}{tb_log_name}"

    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print("=" * 40)
        print(f"train env={e_name} device={device} steps={total_timesteps}")
        print(
            f"SAC replay buffer={buffer_size} "
            f"learning_starts={learning_starts} "
            f"batch_size={batch_size} "
            f"train_freq={train_freq} "
            f"gradient_steps={gradient_steps}"
        )

    env = make_vec(
        e_name,
        n_envs=n_envs,
        render_mode=None,
        sb3=True,
        normalize=True,
    )

    if startup_log:
        print("Warming up vector envs...")

    # Constants env seed to reduce variation between master seeds
    env.seed(seed=0)
    env.reset()

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=0,
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
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
    )

    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))

    # Constants env seed to reduce variation between master seeds
    env.seed(seed=0)

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        # callback=ThroughputCallback(total_timesteps),
        tb_log_name=tb_log_name,
    )

    model_path = f"_models/sac_{name_time}_{e_name}_{seed}"
    vec_norm_path = f"{model_path}_vecnormalize.pkl"

    model.save(model_path)
    env.save(vec_norm_path)

    print(f"saved model: {model_path}.zip")
    # print(f"saved vecnorm: {vec_norm_path}")

    env.close()

    return model_path, vec_norm_path


if __name__ == "__main__":
    for i, _ in enumerate(env_names):
        print(envs_timesteps[i])

        model_path, vec_norm_path = train(
            seed=0nt(i),
            startup_log=True,
            e_name=env_names[i],
            total_timesteps=envs_timesteps[i],
        )

        eval_model(
            env_name=env_names[i],
            render_mode=None,
            m_path=model_path,
        )