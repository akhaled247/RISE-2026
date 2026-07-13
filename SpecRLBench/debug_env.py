import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from utils.env_utils import *

seed = 0
env_name = 'PointLTL4MASAR1Debug-v0'
steps = 2500

print(f"="*40)
print(f"environment: {env_name}")
render_mode = "human" if 'Vision' not in env_name else None
env = make_env(env_name, render_mode=render_mode)
# env = FlattenObservation(gym.make(env_name, render_mode="human"))
obs, info = env.reset(seed=seed)
for i in range(steps):
    try:
        action = env.action_space.sample()
    except:
        action = {a: env.action_space(a).sample() for a in env.unwrapped.possible_agents}
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()

    # if any(terminated.values()):
    #     print('Terminated')
    #     break