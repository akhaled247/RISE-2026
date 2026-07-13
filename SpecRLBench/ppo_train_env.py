import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import ThroughputCallback, make_vec
from ppo_load_env import eval_model

# --- edit these before each run ---
env_name = "PointLTL4MASAR1-v0"
'''
PointLTL0MASAR1-v0 >> run_num = 8
PointLTL4MASAR1-v0 >> run_num = 4
'''
run_num = 3  # INCREMENT EACH TIME;
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
VEC_NORM_PATH = f"{MODEL_PATH}_vecnormalize.pkl"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"
TOTAL_TIMESTEPS = 500_000
seed = 0
n_envs = 8


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"train env={env_name} device={device}")

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    print("Warming up vector envs...")
    env.reset()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        ent_coef=0.01,
        target_kl=0.02,
        device=device,
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        # callback=ThroughputCallback(TOTAL_TIMESTEPS),
    )
    model.save(MODEL_PATH)
    env.save(VEC_NORM_PATH)
    print(f"saved model: {MODEL_PATH}.zip")
    print(f"saved vecnorm: {VEC_NORM_PATH}")

    env.close()


if __name__ == "__main__":
    print(f'Did you change the run number? Current run_num = {run_num}')
    breakpoint()
    train()
    eval_model(render_mode="human")
