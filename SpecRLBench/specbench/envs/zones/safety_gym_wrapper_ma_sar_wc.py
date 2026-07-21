from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from specbench.utils.ltl.logic import Assignment
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
    agent_has_entrapped_at_building,
    agent_inside_building_idx,
)

class SafetyGymWrapperMASARWC(SafetyGymWrapperMASAR):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cost"] = 0
        for a in self.env.unwrapped.possible_agents:
            if info[a].get("cost_walls", 0) > 0:
                info["cost"] += 1
                if self.sb3:
                    terminated = True  # parent already collapsed to bool
                else:
                    terminated[a] = True
        return obs, reward, terminated, truncated, info
        
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["cost"] = 0
        return obs, info