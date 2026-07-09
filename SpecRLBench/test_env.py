import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from utils.env_utils import *

seed = 0
env_names = [

    # letter envs (full/partial observability)
    'LetterLTL0-v0',
    'LetterLTL0-v0.partial',

    # zone envs, multi-agent
    'PointLTL0MA3-v0',

    # zone envs, single-agent (full/partial observability, overlap/no overlap, static/dynamic zones)
    'PointLTL0-v0',
    'PointLTL0-v0.partial',
    'PointLTL0-v0.overlap',
    'PointLTL0-v0.partial_overlap',

    'PointLTL1-v0',
    'PointLTL1-v0.partial',
    'PointLTL1-v0.overlap',
    'PointLTL1-v0.partial_overlap',

    'PointLTL2-v0',
    'PointLTL2-v0.partial',
    'PointLTL2-v0.overlap',
    'PointLTL2-v0.partial_overlap',

    'CarLTL0-v0',
    'CarLTL0-v0.partial',
    'CarLTL0-v0.overlap',
    'CarLTL0-v0.partial_overlap',

    'CarLTL1-v0',
    'CarLTL1-v0.partial',
    'CarLTL1-v0.overlap',
    'CarLTL1-v0.partial_overlap',

    'AntLTL0-v0',
    'AntLTL0-v0.partial',
    'AntLTL0-v0.overlap',
    'AntLTL0-v0.partial_overlap',

    'AntLTL1-v0',
    'AntLTL1-v0.partial',
    'AntLTL1-v0.overlap',
    'AntLTL1-v0.partial_overlap',

    'AntLTL2-v0',
    'AntLTL2-v0.partial',
    'AntLTL2-v0.overlap',
    'AntLTL2-v0.partial_overlap',    
    
    'PointLTL0Vision-v0',
    'PointLTL0Vision-v0.overlap',

    'PointLTL1Vision-v0',
    'PointLTL1Vision-v0.overlap',

    'PointLTL2Vision-v0',
    'PointLTL2Vision-v0.overlap',

    'CarLTL0Vision-v0',
    'CarLTL0Vision-v0.overlap',

    'CarLTL1Vision-v0',
    'CarLTL1Vision-v0.overlap',

    'CarLTL2Vision-v0',
    'CarLTL2Vision-v0.overlap',

    'AntLTL0Vision-v0',
    'AntLTL0Vision-v0.overlap',

    'AntLTL1Vision-v0',
    'AntLTL1Vision-v0.overlap',

    'AntLTL2Vision-v0',
    'AntLTL2Vision-v0.overlap',

    # Arm envs (full/partial observability, grippers-only/grippers and arm)
    'PandaLTLReach0Joints-v0',
    'PandaLTLReach0Joints-v0.partial',

    'PandaLTLReach1Joints-v0',
    'PandaLTLReach1Joints-v0.partial',

    # safety-gymnasium defaults that could be useful for real-world applications"
    "SafetyPointBuildingGoal0-v0",
]

env_name = 'PointLTLMASAR2Debug-v0'
steps = 750

for env_name in env_names:
    print(f"="*40)
    env = make_env(env_name, render_mode=None)
    obs, info = env.reset(seed=seed)
    for i in range(2):
        try:
            action = env.action_space.sample()
        except:
            action = {a: env.action_space(a).sample() for a in env.unwrapped.possible_agents}
        obs, reward, terminated, truncated, info = env.step(action)
    print(f"checked env: {env_name}")