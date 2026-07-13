import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env

# --- must match the train run ---
env_name = "PointLTL0MASAR1-v0"
run_num = 1
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
VEC_NORM_PATH = f"{MODEL_PATH}_vecnormalize.pkl"
eval_episodes = 10
seed = 0


def eval_model(render_mode=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"eval env={env_name} device={device}")
    print(f"loading {MODEL_PATH}.zip")

    base_env = make_env(env_name, sb3=True, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: Monitor(base_env)])
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=vec_env, device=device)

    for episode in range(eval_episodes):
        obs = vec_env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += float(reward[0])
            done = bool(done[0])
        print(f"Episode {episode + 1}: {episode_reward:.3f}")

    vec_env.close()


if __name__ == "__main__":
    eval_model()
