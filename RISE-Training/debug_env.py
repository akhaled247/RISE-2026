import gymnasium as gym 
from numpy import uint8

from rise_training.paths import ensure_specrlbench_paths

ensure_specrlbench_paths()

import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from rise_training.env_utils import *

seed = 0
env_name = 'PointLTL2MASAR1WCDebug-v0'
steps = 2500

print(f"="*40)
print(f"environment: {env_name}")
render_mode = "human" if 'Vision' not in env_name else None
env = make_env(env_name, render_mode=render_mode)
# env = FlattenObservation(gym.make(env_name, render_mode="human"))
obs, info = env.reset(seed=seed)
done = False
while not done:
    try:
        action = env.action_space.sample()
    except:
        action = {a: env.action_space(a).sample() for a in env.unwrapped.possible_agents}
    obs, reward, terminated, truncated, info = env.step(action)
    done = any(list(terminated.values())) or any(list(truncated.values()))
    # print(
    #     f'obs = {obs} \n'
    #     f'reward = {reward} \n'
    #     f'terminated = {terminated} \n'
    #     f'truncated = {truncated} \n'
    #     f'info = {info} \n'
    #     )
    # if info['agent_0']['cost_walls']>0: print(info['agent_0']['cost_walls'])
    if reward['agent_0'] != 0: print(f"[debug_env] reward = {reward['agent_0']}")
    # print(f'terminated {any(list(terminated.values()))}')
    # print(f'truncated {any(list(truncated.values()))}')
    # env.render()

    # if any(terminated.values()):
    #     print('Terminated')
    #     break