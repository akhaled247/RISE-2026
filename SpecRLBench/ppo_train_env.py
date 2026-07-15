import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import ThroughputCallback, make_vec
from ppo_load_env import eval_model
import time
from datetime import datetime

# --- edit these before each run ---
env_name = "PointLTL4MASAR1-v0"
run_num = 9  # INCREMENT EACH TIME <<Level4 = 9, Level0 = 13>>
name_time = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f"_models/ppo_{name_time}_{env_name}_run{run_num}"
VEC_NORM_PATH = f"{MODEL_PATH}_vecnormalize.pkl"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"
TOTAL_TIMESTEPS = 1_000_000
seed = 0
n_envs = 8
ent_coef = 0.02
learning_rate = 5e-5
n_steps = 2048  # 512 Level0, 2048 Level4
batch_size = 256
n_epochs = 10
clip_range = 0.2


def train() -> tuple[str, str]:
    print(f"Logging to {TRAINING_LOG_PATH}...")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"train env={env_name} device={device} steps={TOTAL_TIMESTEPS}")
    rollout_steps = n_steps * n_envs
    print(
        f"PPO iter = {rollout_steps} env steps collect + {n_epochs} epochs x "
        f"{rollout_steps // batch_size} minibatches — SB3 iters/s scales ~1/n_steps"
    )

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    print("Warming up vector envs...")
    env.seed(seed=seed)
    env.reset()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        target_kl=0.03,
        device=device,
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
        clip_range=clip_range,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=ThroughputCallback(TOTAL_TIMESTEPS),
        tb_log_name=(
            f"PPO_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{TOTAL_TIMESTEPS/1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_ep{n_epochs}"
            f"_cr{clip_range}"
        ),
    )
    model.save(MODEL_PATH)
    env.save(VEC_NORM_PATH)
    print(f"saved model: {MODEL_PATH}.zip")
    print(f"saved vecnorm: {VEC_NORM_PATH}")
    env.close()
    return MODEL_PATH, VEC_NORM_PATH


if __name__ == "__main__":
    print(f"Make sure <<<run_num = {run_num}>>> is correct before continuing! Will continue in 7 seconds.")
    print(f"<<<{(n_steps*n_envs)/batch_size}>>> minibatches per rollout")
    print(f"Policy will update <<<{TOTAL_TIMESTEPS//(n_steps*n_envs)}>>> times.")
    time.sleep(7.0)
    train()

    eval_model(
        env_name=env_name,
        run_num=run_num,
        render_mode="human",
        m_path=MODEL_PATH,
    )
