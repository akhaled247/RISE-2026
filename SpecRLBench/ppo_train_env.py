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
env_name = "PointLTL5MASAR1-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")
model_path = f"_models/ppo_{name_time}_{env_name}"
vec_norm_path = f"{model_path}_vecnormalize.pkl"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"


def train(
        total_timesteps = 1_000_000,
        seed = 0,
        n_envs = 8,
        ent_coef = 0.02,
        learning_rate = 5e-5,
        n_steps = 2048,
        batch_size = 256,
        n_epochs = 10,
        clip_range = 0.2,
        target_kl = 0.05, #0.08
        startup_log = True
) -> tuple[str, str]:
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print(f"<<<{rollout_steps/batch_size}>>> minibatches per rollout"
            f"\n <<<{total_timesteps//rollout_steps}>>> policy updates total")
        print("=" * 40)
        print(f"train env={env_name} device={device} steps={total_timesteps}")
        print(
            f"PPO iter = {rollout_steps} env steps collect + {n_epochs} epochs x "
            f"{rollout_steps // batch_size} minibatches "
            f"- SB3 iters/s scales ~1/n_steps"
        )

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log: print("Warming up vector envs...")
    env.seed(seed=0) #Constants env seed to reduce variation between master seeds
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
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
        clip_range=clip_range,
    )
    
    env.seed(seed=0) #Constants env seed to reduce variation between master seeds
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        # callback=ThroughputCallback(total_timesteps),
        tb_log_name=(
            f"PPO_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps/1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_ep{n_epochs}"
            f"_cr{clip_range}"
            f"_kl{target_kl}"
            f"_s{seed}"
        ),
    )
    model_path=f"_models/ppo_{name_time}_{env_name}_{seed}"
    vec_norm_path=f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    # print(f"saved vecnorm: {VEC_NORM_PATH}")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    for i in range(1):
        train(
            seed=int(i), #Tested up to and including env 3 at home
            startup_log=True,
            total_timesteps=2_500_000)
        eval_model(
            env_name=env_name,
            render_mode=None,
            m_path=model_path,
        )
