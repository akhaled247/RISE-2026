from stable_baselines3 import PPO
import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from utils.env_utils import *

seed = 0

# 1. Initialize the standard Gymnasium environment
env_name = 'PointLTL0MASAR2-v0'
steps = 750

print(f"="*40)
render_mode = "human" if 'Vision' not in env_name else None
env = make_env(env_name, render_mode=None, sb3=True)

# 2. Instantiate the PPO Agent 
# "MlpPolicy" is used for feature vectors (like positions and velocities)
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    device="cuda",
)
# 3. Train the agent
model.learn(total_timesteps=1000, progress_bar=True)

# 4. Evaluate the trained agent
env = make_env(env_name, render_mode="human", sb3=True)
obs, info = env.reset(seed=seed)
total_reward = 0
episodes = 1
for episode in range(episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    i = 0
    while not done or i>steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        done = terminated or truncated
        i+=1

    print(f"Episode {episode+1}: {episode_reward}")

env.close()
