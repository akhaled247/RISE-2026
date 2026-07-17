import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from utils.env_utils import *

seed = 0
env_name = 'PointLTL4MASAR1DebugWC-v0'
steps = 2500
print('_models/rnd_ppo_L5_S2_20260717_1109_PointLTL5MASAR1WC-v0_0'.split('_'))
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
    print(obs['agent_0'])
    # if info['agent_0']['cost_walls']>0: print(info['agent_0']['cost_walls'])
    if reward['agent_0'] != 0: print(f"[debug_env] reward = {reward['agent_0']}")
    # print(f'terminated {any(list(terminated.values()))}')
    # print(f'truncated {any(list(truncated.values()))}')
    # env.render()

    # if any(terminated.values()):
    #     print('Terminated')
    #     break