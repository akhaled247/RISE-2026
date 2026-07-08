from typing import Any, SupportsFloat, Callable

import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            # 'features': env.observation_space,
            # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
            'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        
        cost = 1.0 if reward == -1. else 0.0
        reach, avoid = self.goal_seq[self.num_reached] \
            if self.num_reached < len(self.goal_seq) else self.goal_seq[-1]
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation(obs, info)
        
        self.obs = obs
        self.info = info
        # obs = self.complete_observation(obs, info)
        return obs, (reward, cost), terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        
        reach, avoid = self.goal_seq[self.num_reached]
        obs = self.pre_process_obs(reach, avoid)
        
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }
        
    def pre_process_obs(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        pre-process the observation
        """
        original_obs = self.task.original_obs
        # print(f"original_obs = {original_obs}")
        
        reach_zones = [r.to_string()[0]+"_zones_lidar" for r in list(reach)]
        avoid_zones = [a.to_string()[0]+"_zones_lidar" for a in list(avoid)]
        
        agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]        
        agent_obs = np.concatenate([original_obs[key].flatten() if original_obs[key].ndim > 1 else original_obs[key] for key in agent_obs_keys])
        lidar_dim = 16
        
        reach_obs = np.vstack([original_obs[color] for color in reach_zones]) # len(reach_zones) x lidar_dim
        reach_obs = np.max(reach_obs, axis=0) # lidar_dim
        if len(avoid_zones):
            avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
            avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
        else:
            avoid_obs = np.zeros(lidar_dim)
            
        assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        return np.concatenate([agent_obs, reach_obs, avoid_obs])


class SequenceSafetyWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)

        self.region_order = env.get_propositions()
        ap_dim = len(self.region_order)
        obs_dim = env.observation_space.shape[0]

        if "Point" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
            })
            self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]
        elif "Car" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 28 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (60,), dtype=np.float32)
            })
            self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", 
                                   "ballangvel_rear", "ballquat_rear",
                                   "wall_sensor"]
        elif "Ant" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 44 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (76,), dtype=np.float32)
            })
            self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", # 12
                                   'hip_1', 'hip_1_vel', 'hip_2', 'hip_2_vel', 'hip_3', 'hip_3_vel', 'hip_4', 'hip_4_vel', 
                                   'ankle_1', 'ankle_1_vel', 'ankle_2','ankle_2_vel', 'ankle_3', 'ankle_3_vel', 'ankle_4', 'ankle_4_vel', 
                                   'agent_pos', 'agent_qvel',
                                   "wall_sensor"]
        elif "Letter" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })

        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.reward_scale = 0.0 # 1.0 for dense rewards
        self.cost_scale = 1.0
        self.num_reached = 0
        
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # if (action == LDBASequence.EPSILON).all():
        #     obs, _, terminated, truncated, info = self.apply_epsilon_action()
        #     reward = 0.
        # else:
        #     assert not (action == LDBASequence.EPSILON).any()
        obs, reward, terminated, truncated, info = super().step(action)
        # assert self.prev_dist is not None
        reach, avoid = self.goal_seq[self.num_reached]
        
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        
        # # dense reward
        # curr_dist = self.calculate_dist_to_goal(reach); 
        # reward = (self.prev_dist - curr_dist) * self.reward_scale
        # self.prev_dist = curr_dist
        reward = 0.0; cost = 0.0; agent_alive = True # terminated = False
        if terminated:
            agent_alive = False
        
        # reach the avoid area
        if assignment in avoid:
            # reward = -1.0 # for ppo test
            info['violation'] = True
            cost = 1.0; terminated = True
            # cost = 1.0; terminated = False # for visualization
            # self.violated = True

        # reach goal
        elif reach != LDBASequence.EPSILON and assignment in reach:
            info['success'] = True
            # info['success'] = True if not self.violated else False # for visualization
            # # hard_case
            # reward += 1.0; cost = 0.0; terminated = True
            self.goal_seq = self.sample_sequence(assignment)
            
            reward = 1.0; cost = 0.0; terminated = False
            # reward = 1.0; cost = 0.0; terminated = True # letter env
            # reward += 1.0; cost = 0.0; terminated = True # for visualization v1/v3
            # reward = 1.0 if not self.violated else 0.0; terminated = True # for visualization v2
            reach, avoid = self.goal_seq[self.num_reached]
            # curr_dist = self.calculate_dist_to_goal(reach)
            # self.prev_dist = curr_dist
        
        # reach the boundary of the environment
        elif 'cost_ltl_walls' in info and info['cost_ltl_walls'] > 0:
            # reward = -1.0 # for ppo test
            cost = 1.0; terminated = True
            # cost = 1.0; terminated = False # for visualization
            # self.violated = True
        
        # nothing happens    
        # else:
        #     assert reward == 0.0 and cost == 0.0 and not terminated
        
        if not agent_alive:
            cost = 1.0
            terminated = True

        cost *= self.cost_scale
        
        obs = self.pre_process_obs(reach, avoid)
        # obs = self.pre_process_obs_zones(reach, avoid)
        # obs = self.pre_process_obs_letter(reach, avoid)
        # obs = self.pre_process_obs_flatworld(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        self.obs = obs
        self.info = info
        # obs = self.complete_observation(obs, info)
        return obs, (reward, cost), terminated, truncated, info
    
    # def calculate_dist_to_goal(self, reach: frozenset[FrozenAssignment]):
    #     """
    #     calculate reward based on the distance to the reach zone 
    #     and also set current distance and previous distance
    #     """
    #     agent_pos = self.task.agent.pos
    #     zones = [r.to_string()[0] + "_zones" for r in list(reach)]
    #     dists = []
    #     for zone in zones:
    #         zone_poses = self.task._geoms[zone].pos
    #         dist = [np.linalg.norm((agent_pos - zone_pos)[:2]) for zone_pos in zone_poses]
    #         dists.extend(dist)
    #     curr_dist = min(dists)
    #     return curr_dist

    # def apply_epsilon_action(self):
    #     assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
    #     self.num_reached += 1
    #     return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        # self.violated = False
        
        # dense reward
        reach, avoid = self.goal_seq[self.num_reached]
        # curr_dist = self.calculate_dist_to_goal(reach)
        # self.prev_dist = curr_dist
        # print(f"reset goal: {reach}, curr_dist: {curr_dist}")
        
        # obs = self.complete_observation(obs, info)
        # obs = self.pre_process_obs_zones_ant(reach, avoid)
        obs = self.pre_process_obs(reach, avoid)
        # obs = self.pre_process_obs_letter(reach, avoid)
        # obs = self.pre_process_obs_flatworld(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        self.obs = obs
        self.info = info
        return obs, info


    # def pre_process_obs_zones(self,
    #                     reach: frozenset[FrozenAssignment], 
    #                     avoid: frozenset[FrozenAssignment]) -> np.ndarray:
    #     """
    #     pre-process the observation
    #     """
    #     original_obs = self.task.original_obs
    #     # print(f"original_obs = {original_obs}")
        
    #     reach_zones = [r.to_string()[0] + "_zones_lidar" for r in list(reach)]
    #     avoid_zones = [a.to_string()[0] + "_zones_lidar" for a in list(avoid) if a.to_string()]
        
    #     # agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]
    #     agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])
    #     lidar_dim = 16
        
    #     reach_obs = np.vstack([original_obs[color] for color in reach_zones]) # len(reach_zones) x lidar_dim
    #     reach_obs = np.max(reach_obs, axis=0) # lidar_dim
    #     if len(avoid_zones):
    #         avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
    #         avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
    #     else:
    #         avoid_obs = np.zeros(lidar_dim)
            
    #     assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
    #     return np.concatenate([agent_obs, reach_obs, avoid_obs])


    # def pre_process_obs_zones(self,
    #                     reach: frozenset[FrozenAssignment], 
    #                     avoid: frozenset[FrozenAssignment]) -> np.ndarray:
    #     """
    #     re-arrange the order of observations and add LTL vector
    #     """
    #     original_obs = self.task.original_obs
    #     agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])
    #     lidar_obs = np.concatenate([original_obs[region+"_zones_lidar"] for region in self.region_order])
    #     return np.concatenate([agent_obs, lidar_obs])
    
    #     reach_list = [r.to_string()[0] for r in list(reach)]
    #     avoid_list = [a.to_string()[0] for a in list(avoid)]
    #     reach_vec = np.isin(np.array(self.region_order), reach_list).astype(float)
    #     avoid_vec = np.isin(np.array(self.region_order), avoid_list).astype(float)
    #     ltl_emb = np.concatenate([reach_vec, avoid_vec])

    #     return np.concatenate([agent_obs, lidar_obs, ltl_emb])

    def pre_process_obs(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        if "Letter" in self.env.spec.id:
            return self.pre_process_obs_letter(reach, avoid)
        return self.pre_process_obs_zones(reach, avoid)

    def pre_process_obs_zones(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        observation reduction
        """
        original_obs = self.task.original_obs
        # print(f"original_obs: {original_obs.keys()}")
        lidar_dim = self.task.lidar_conf.num_bins
        agent_obs = np.concatenate(
            [original_obs[key].flatten() if original_obs[key].ndim > 1 
             else original_obs[key] for key in self.agent_obs_keys])
        # print(f"agent_obs: {agent_obs.shape}, {agent_obs}")

        reach_zones = [r.to_string()[0] + "_zones_lidar" for r in list(reach)]
        avoid_zones = [a.to_string()[0] + "_zones_lidar" for a in list(avoid) if a.to_string()]
        
        reach_obs = np.vstack([original_obs[color] for color in reach_zones])
        reach_obs = np.max(reach_obs, axis=0) # lidar_dim
        if len(avoid_zones):
            avoid_obs = np.vstack([original_obs[color] for color in avoid_zones])
            avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
        else:
            avoid_obs = np.zeros(lidar_dim)
        # print(agent_obs)
        # assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        # print(f"obs.shape = {np.concatenate([agent_obs, reach_obs, avoid_obs]).shape}")
        obs = np.concatenate([agent_obs, reach_obs, avoid_obs])
        assert obs.shape == self.observation_space['features'].shape, \
            f"obs.shape = {obs.shape}, expected {self.observation_space['features'].shape}"
        return obs

    def pre_process_obs_letter(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        # return self.env.original_obs # emb

        obs = self.env.original_obs
        new_obs = np.zeros((obs.shape[0], obs.shape[1]), dtype=obs.dtype)
        letter_to_index = {letter: i for i, letter in enumerate(self.region_order)}
        
        reach_indices = [letter_to_index[r.to_string()[0]] for r in list(reach)]
        avoid_indices = [letter_to_index[a.to_string()[0]] for a in list(avoid)]
        
        reach_mask = np.any(obs[:, :, reach_indices] > 0, axis=2)
        avoid_mask = np.any(obs[:, :, avoid_indices] > 0, axis=2)
        agent_mask = obs[:, :, -1] > 0
        
        new_obs = np.zeros(obs.shape[:2], dtype=np.float32)
        new_obs[avoid_mask] = 0.5
        new_obs[reach_mask] = 1.0
        new_obs[agent_mask] = 0.2
        # print(f"original_obs = {obs}")
        # print(f"reach = {reach}, avoid = {avoid}, AP = {self.region_order}")
        # print(f"reach_indices = {reach_indices}, avoid_indices = {avoid_indices}")
        # print(f"processed_obs = {new_obs}")
        return new_obs[..., None]
    

    def pre_process_obs_flatworld(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        return self.env.original_obs



    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }
        
    def complete_observation_current(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq,
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }

