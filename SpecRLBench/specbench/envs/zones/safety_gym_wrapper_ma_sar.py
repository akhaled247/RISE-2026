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
    sb3 = False
    action_dim = 2
    def __init__(self, env: Any, wall_sensor=True, sb3=False):
        super().__init__(env)
        self.unwrapped.render_parameters.camera_name = 'track'
        self.unwrapped.render_parameters.width = 256
        self.unwrapped.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins
        self.sb3 = sb3

        # Robustly handle both property and method for observation_space
        obs_space = env.observation_space
        if callable(obs_space):
            # If it's a method, call with None (or agent name if needed)
            obs_space = obs_space(None)
        obs_keys = obs_space.spaces.keys()
        # print(f"DEBUG: obs_keys = {obs_keys}")
        self.colors = set()
        self.atomic_propositions = set()
        self.num_agents = env.unwrapped.num_agents
        # self.num_agents = 1
        for key in obs_keys:
            # if key.endswith('zones_lidar'):
            if "zones" in key.split('_'):
                color = key.split('_')[0]
                self.colors.add(color)
                for i in range(self.num_agents*2):
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

        if self.sb3:
            act_space = env.action_space
            if callable(act_space):
                act_space = Box(low=-1.0, high=1.0, shape=(self.num_agents*self.action_dim,))
            if isinstance(act_space, spaces.Box):
                self.action_space = act_space
            else:
                print(type(act_space))
                self.observation_space = spaces.Box(act_space)
        if wall_sensor:
            for i, a in enumerate(self.env.unwrapped.possible_agents):
                self.observation_space[f'wall_sensor_{i}'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        # print(f"DEBUG: self.observation_space = {self.observation_space}")
        self.last_dist = None

    # PPO Notes
    # Note: Keep reward scales between [-1, 1]
    # Dense rewards are better for PPO >> Better critic
    # _reward_inside_building = 1
    _reward_find_casualty = 1.0
    _reward_agent_collision = -0.001
    _reward_casualty_scalar = 0.00001 # * 1000 = 1.0 == _reward_find_casualty
    _reward_wall_collision = -0.01
    def step(self, action: ActType):
        # print(action)
        if self.sb3: action = self.dictify_action(action)
        obs, reward, cost, terminated, truncated, info = super().step(action)
        # print(f"DEBUG: info = {info}")
        # print(f"DEBUG: terminated = {terminated}, truncated = {truncated}")
        # print(f"DEBUG: obs = {obs}")
        # update env boundary wall sensor info
        if 'wall_sensor' in info["agent_0"]:
            for i, agent in enumerate(self.env.unwrapped.possible_agents):
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
            for i, a in enumerate(self.env.unwrapped.possible_agents):
                # terminated[a] = terminated[a] or \
                #     info[a]['cost_ltl_walls'] > 0
                if info[a]['cost_ltl_walls'] > 0:
                    # print(f"DEBUG: wall collision detected for {a}!")
                    reward[a] += self._reward_wall_collision
                if info[a]['cost_collision'] > 0:
                    reward[a] += self._reward_agent_collision
                    pass
                    # print(f"DEBUG: agent collision detected for {a}!")
        
        # if any(terminated.values()):
        #     print(f"DEBUG: collision detected!")
            # info['violation'] = True

        info['propositions'] = []
        # print(f"DEBUG: action = {action}")
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            agent_info: dict = info[a]
            # print(zone_info) if i==0 else print('')
            active_props = {}
            for k, v in agent_info.items():
                if (isinstance(v, (int, float))):
                    if v > 0 and "cost_sum" not in k:
                        # print((k, v))
                        active_props.update({f"{k}_{i}": v})
            # print(active_props) if i==0 else print('')
            
            info['propositions'].extend(active_props)
            if f'cost_buildings_terracotta_{i}' in info['propositions']:
                # print('Agent '+str(i)+' in building')
                # terminated[a] = True
                reward[f"agent_{i}"] += self._reward_inside_building
            else:
                try:
                    obs[a][f'entrapped_casualtys_lidar_{i}'] = np.zeros(obs[a][f'entrapped_casualtys_lidar_{i}'].size)
                except KeyError as e:
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
            # if i == 0: print(obs[a]['surface_casualtys_lidar_ids_0'])
            try:
                reward[f"agent_{i}"]+=(max(obs[a][f'surface_casualtys_lidar_{i}'])*self._reward_casualty_scalar)
                pass
            except KeyError as e:
                print(f"ERROR: {e} \n No surface casualtys") 
            # if i == 0: print(obs[a])
        if self.sb3:
            obs = self.flatten_obs(obs)
            reward = sum(list(reward.values())) / (len(reward)/self.num_agents) #avg
            truncated = any(list(truncated.values()))
            terminated = any(list(terminated.values()))
        # print(f"DEBUG: truncated = {truncated}")
        # print(f"DEBUG: terminated = {terminated}")
        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # print("DEBUG: Environment Reset!")
        info['propositions'] = []
        # obs["agent_0"]['wall_sensor'] = np.array([0, 0, 0, 0])
        # obs["agent_1"]['wall_sensor1'] = np.array([0, 0, 0, 0])
        for i, a in enumerate(self.env.unwrapped.possible_agents):
            obs[a][f'wall_sensor_{i}'] = np.array([0, 0, 0, 0])
            # print(obs[a])
        # Ensure original_obs is set at reset
        self.env.unwrapped.task.original_obs = obs
        if self.sb3: obs = self.flatten_obs(obs)
        # print(f"DEBUG: obs reset= {obs}")
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

    def flatten_obs(self, obs):
            flat = {
                k: v
                for agent_obs in obs.values()
                for k, v in agent_obs.items()
            }
            # print(f"DEBUG: flatten_obs flat = {flat}")
            # if isinstance(flat, spaces.Dict):
            return flat

    def dictify_action(self, action) -> dict:
        actions = {
            f"agent_{i}": action[i * self.action_dim:(i + 1) * self.action_dim]
            for i in range(self.num_agents)
        }
        return actions

            
