from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box
from specbench.envs.zones.safety_gym_wrapper_ma_sar import SafetyGymWrapperMASAR
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
            try:
                if info[a].get("cost_walls", 0) > 0:
                    info["cost"] += 1
                    terminated[a] = True
            except Exception as e:
                  pass
            try:
                if info[a].get("cost_collision") > 0:
                    info["cost"] += 1
                    terminated[a] = True
            except Exception as e:
                pass
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["cost"] = 0
        return obs, info