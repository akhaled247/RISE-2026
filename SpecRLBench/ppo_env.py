from stable_baselines3 import PPO
import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from utils.env_utils import *

# 1. Initialize the standard Gymnasium environment
env_name = 'PointLTL0MASAR2-v0'
run_num = 7 #INCREMENT EACH TIME
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"
steps = 10000
seed = 0
print(f"="*40)
render_mode = "human" if 'Vision' not in env_name else None
env = make_vec(env_name, 
               n_envs=8,
               render_mode=None,
               sb3=True)

# 2. Instantiate the PPO Agent 
# "MlpPolicy" is used for feature vectors (like positions and velocities)
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    device="cuda:1",
    n_steps=2048//8,
    tensorboard_log=TRAINING_LOG_PATH
)
# 3.1. Train the agent
# model.learn(total_timesteps=500_000, progress_bar=True)
# model.save(MODEL_PATH) #BE SURE TO INCREMENT EACH TIME
# 3.2. Load saved agent
model = PPO.load(MODEL_PATH, env=env, device="cuda")

# 4. Evaluate the trained agent
env = make_env(env_name, sb3=True, render_mode="human")
obs, info = env.reset(seed=seed)
total_reward = 0
episodes = 10
for episode in range(episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    i = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        done = terminated or truncated

    print(f"Episode {episode+1}: {episode_reward}")

env.close()
