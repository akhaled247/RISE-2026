from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from specbench.utils.ltl.logic import Assignment

class SafetyGymWrapperMASAR(gymnasium.Wrapper):
    """
    A wrapper from safety gymnasium LTL environments to the gymnasium API.
    """

    def __init__(self, env: Any, wall_sensor=True):
        super().__init__(env)
        self.render_parameters.camera_name = 'track'
        self.render_parameters.width = 256
        self.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins

        # Robustly handle both property and method for observation_space
        obs_space = env.observation_space
        if callable(obs_space):
            # If it's a method, call with None (or agent name if needed)
            obs_space = obs_space(None)
        obs_keys = obs_space.spaces.keys()
        # obs_keys = env.observation_space["agent_0"].spaces.keys()
        # print(f"DEBUG: obs_keys = {obs_keys}")
        self.colors = set()
        self.atomic_propositions = set()
        self.num_agents = env.num_agents
        # self.num_agents = 1
        for key in obs_keys:
            # if key.endswith('zones_lidar'):
            if "zones" in key.split('_'):
                color = key.split('_')[0]
                self.colors.add(color)
                for i in range(self.num_agents):
                    self.atomic_propositions.add(color + '_' + str(i))
        # print(f"DEBUG: self.colors = {self.colors}")
        # print(f"DEBUG: self.atomic_propositions = {self.atomic_propositions}")

        # Robustly handle both property and method for observation_space
        obs_space = env.observation_space
        if callable(obs_space):
            obs_space = obs_space(None)
        # If it's already a Dict, use it directly; otherwise, wrap if needed
        if isinstance(obs_space, spaces.Dict):
            self.observation_space = obs_space
        else:
            self.observation_space = spaces.Dict(obs_space)
        # self.observation_space = spaces.Dict(env.observation_space["agent_0"])  # copy the observation space

        if wall_sensor:
            for i, a in enumerate(self.env.possible_agents):
                self.observation_space[f'wall_sensor_{i}'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
            # self.observation_space['wall_sensor'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
            # self.observation_space['wall_sensor1'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        # print(f"DEBUG: self.observation_space = {self.observation_space}")
        self.last_dist = None

    _reward_inside_building = 0.05
    _reward_find_casualty = 0.5
    _reward_collision = -1.0
    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        # print(f"DEBUG: info = {info}")
        # print(f"DEBUG: terminated = {terminated}, truncated = {truncated}")

        # print(f"DEBUG: obs = {obs}")
        # print(f"DEBUG: info = {info}")
        # update env boundary wall sensor info
        if 'wall_sensor' in info["agent_0"]:
            # obs["agent_0"]['wall_sensor']  = info["agent_0"]['wall_sensor']
            # obs["agent_1"]['wall_sensor1'] = info["agent_1"]['wall_sensor']
            # for i, a in enumerate(self.env.agents):
            #     suffix = '' if i == 0 else str(i)
            #     obs[a][f'wall_sensor{suffix}'] = info[a]['wall_sensor']
            for i, agent in enumerate(self.env.possible_agents):
                obs[agent][f'wall_sensor_{i}'] = info[agent]['wall_sensor']
            # print(f"DEBUG: obs wrapper = {obs}")

        # self.task.original_obs = obs
        self.env.unwrapped.task.original_obs = obs
        # print(f"DEBUG: original_obs SafetyGymWrapper step = {obs}")

        # update termination based on wall collision
        # TODO: may need to have seperate termination for each agent, 
        # one agent may violate its own subgoal such that the whole spec cannot be satisfied 
        # (the episode should terminate), but it does not necessarily mean the other agent's action is not valid. 
        if 'cost_ltl_walls' in info["agent_0"]:

            # terminated["agent_0"] = terminated["agent_0"] or \
            #     info["agent_0"]['cost_ltl_walls'] > 0 or \
            #     info["agent_0"]['cost_collision'] > 0
            # terminated["agent_1"] = terminated["agent_1"] or \
            #     info["agent_1"]['cost_ltl_walls'] > 0 or \
            #     info["agent_1"]['cost_collision'] > 0
            for i, a in enumerate(self.env.possible_agents):
                terminated[a] = terminated[a] or \
                    info[a]['cost_ltl_walls'] > 0
                if info[a]['cost_ltl_walls'] > 0:
                    print(f"DEBUG: wall collision detected for {a}!")
                if info[a]['cost_collision'] > 0:
                    reward[a] += self._reward_collision
                    # print(f"DEBUG: agent collision detected for {a}!")
            
            # if any(terminated.values()):
            #     print(f"DEBUG: collision detected!")
                # info['violation'] = True

        info['propositions'] = []
        
        for i, a in enumerate(self.env.possible_agents):
            # suffix = '' if i == 0 else f"_{i}"
            agent_info: dict = info[a]
            # print(zone_info) if i==0 else print('')
            # active_props = [c + '_' + str(i) for c in self.colors if zone_info[f'cost_zone_{c}'] > 0]
            # active_props = [cost for cost in zone_info.values() if cost > 0]
            # info['propositions'].extend(active_props)
            active_props = {}
            for k, v in agent_info.items():
                if (isinstance(v, (int, float))):
                    if v > 0 and "cost_sum" not in k:
                        # print((k, v))
                        active_props.update({f"{k}_{i}": v})
            # ap = {k: v for k, v in zone_info.items() if v > 0}
            # print(active_props) if i==0 else print('')
            info['propositions'].extend(active_props)
            if f'cost_buildings_terracotta_{i}' in info['propositions']:
                # print('Agent '+str(i)+' in building')
                # terminated[a] = True
                reward[f"agent_{i}"] += self._reward_inside_building
            else:
                obs[a][f'entrapped_casualtys_lidar_{i}'] = np.zeros(obs[a][f'entrapped_casualtys_lidar_{i}'].size)
                pass
            if f'cost_casualtys_surface_{i}' in info['propositions']:
                # print('Agent '+str(i)+' found entrapped casualty')
                # terminated[a] = True
                reward[f"agent_{i}"] += self._reward_find_casualty
            if f'cost_casualtys_entrapped_{i}' in info['propositions']:
                # print('Agent '+str(i)+' found entrapped casualty')
                # terminated[a] = True
                reward[f"agent_{i}"] += (self._reward_find_casualty * 2.0)
            
            lidar_keys = [k for k in obs[a] if "lidar" in k]
            arr = np.stack([obs[a][k] for k in lidar_keys])
            # if i == 0: print(lidar_keys)
            # if i == 0: print(arr)
            # if i == 0: print(obs[a])

        # print(obs["agent_0"]['surface_casualtys_lidar_0'])
        # print(reward)
        # print(info['propositions'])
        # for i, a in enumerate(self.env.possible_agents):
        #     if 'yellow_'+str(i) in info['propositions']:
        #         print('Agent '+str(i)+' in yellow env')
        #         terminated[a] = terminated[a] or \
        #                 info[a]['cost_zones_yellow'] > 0

        # zone_info = info["agent_0"]
        # active_props = [c + '_0' for c in self.colors if zone_info[f'cost_zones_{c}'] > 0]
        # info['propositions'].extend(active_props)

        # zone_info = info["agent_1"]
        # active_props = [c + '_1' for c in self.colors if zone_info[f'cost_zones_{c}'] > 0]
        # info['propositions'].extend(active_props)

        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info['propositions'] = []
        # obs["agent_0"]['wall_sensor'] = np.array([0, 0, 0, 0])
        # obs["agent_1"]['wall_sensor1'] = np.array([0, 0, 0, 0])
        for i, a in enumerate(self.env.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
            # print(obs[a])
        # Ensure original_obs is set at reset
        self.env.unwrapped.task.original_obs = obs
        # print(f"DEBUG: original_obs SafetyGymWrapper reset= {obs}")
        return obs, info

    def get_propositions(self) -> list[str]:
        return sorted(self.atomic_propositions)

    def get_possible_assignments(self) -> list[Assignment]:
        # For multi-agent: allow at most one proposition per agent to be true, but allow different agents' props to be true simultaneously
        assignments = []
        agent_props = {}
        # Group props by agent index (e.g., color0, color1)
        for prop in self.atomic_propositions:
            agent_idx = prop[-1]
            agent_props.setdefault(agent_idx, set()).add(prop)
        # For each agent, get zero_or_one assignments
        per_agent_assignments = []
        for props in agent_props.values():
            per_agent_assignments.append(Assignment.zero_or_one_propositions(props))
        # Cartesian product of per-agent assignments
        import itertools
        for combo in itertools.product(*per_agent_assignments):
            merged = Assignment()
            for a in combo:
                merged.update(a)
            assignments.append(merged)
        assert len(assignments) == (len(self.colors) + 1) ** self.num_agents, \
            f"Expected {(len(self.colors) + 1) ** self.num_agents} assignments, got {len(assignments)}"
        # print(f"DEBUG: possible assignments: {[a.get_true_propositions() for a in assignments]}")
        return assignments

    def get_all_possible_assignments(self) -> list[Assignment]:
        return Assignment.all_possible_assignments(tuple(self.get_propositions()))
