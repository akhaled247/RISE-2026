import torch
from stable_baselines3 import PPO
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env, make_vec

# 1. Initialize the standard Gymnasium environment
env_name = 'PointLTL0MASAR1-v0'
run_num = 1  # INCREMENT EACH TIME
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"
TOTAL_TIMESTEPS = 500_000
SMOKE_TIMESTEPS = 50_000
seed = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 40)
print(f"env={env_name} device={device}")

n_envs = 8
env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)

# 2. Instantiate the PPO Agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    ent_coef=0.01,
    target_kl=0.02,
    device=device,
    tensorboard_log=TRAINING_LOG_PATH,
    seed=seed,
)

# 3.1. Train the agent
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
model.save(MODEL_PATH)
env.save(f"{MODEL_PATH}_vecnormalize.pkl")

# 3.2. Load saved agent (comment out for fresh training)
# model = PPO.load(MODEL_PATH, env=env, device=device)

# 4. Evaluate the trained agent
eval_env = make_env(env_name, sb3=True, render_mode=None)
obs, info = eval_env.reset(seed=seed)
episodes = 10
for episode in range(episodes):
    obs, info = eval_env.reset()
    episode_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: {episode_reward:.3f}")

eval_env.close()
env.close()
