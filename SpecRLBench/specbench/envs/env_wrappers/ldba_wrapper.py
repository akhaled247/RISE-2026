import functools
from typing import Any, SupportsFloat
import numpy as np
import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr
from ltl.automata import ltl2ldba, LDBA, LDBASequence
from ltl.logic import Assignment, FrozenAssignment


class LDBAWrapper(gymnasium.Wrapper):
    """
    Wrapper that keeps track of LTL goal satisfaction using an LDBA, which is added to the observation space.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        # if not isinstance(env.observation_space, gymnasium.spaces.Dict):
        #     raise ValueError('LDBA wrapper requires dict observations')
        # if 'goal' not in env.observation_space.spaces:
        #     raise ValueError('LDBA wrapper requires goal in observation space')
        # no_ltl_emb
        # self.observation_space = spaces.Dict({
        #     # 'features': env.observation_space,
        #     # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
        #     'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
        # })
        
        self.region_order = env.get_propositions()
        ap_dim = len(self.region_order)
        # print(f"LDBAWrapper, {env.observation_space}")
        
        
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
            obs_dim = env.observation_space["features"].shape[0]
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })
        
        self.terminate_on_acceptance = False
        self.ldba = None
        self.ldba_state = None
        self.num_accepting_visits = 0
        # self.obs = None
        # self.info = None
        self.traj = []

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # if (action == LDBASequence.EPSILON).all():
        #     obs, reward, terminated, truncated, info = self.obs, 0.0, False, False, self.info
        #     take_epsilon = True
        # else:
        # assert not (action == LDBASequence.EPSILON).any()
        obs, reward, terminated, truncated, info = super().step(action)
        take_epsilon = False
            # self.obs = obs
            # self.info = info
        # if terminated:
        #     print(f"agent touches the ground")
        if len(info['propositions']):
            if len(self.traj):
                if info['propositions'] != self.traj[-1]:
                    self.traj.append(info['propositions'])
            else:
                self.traj.append(info['propositions'])
        new_ldba_state, accepting = self.ldba.get_next_state(self.ldba_state, info['propositions'], take_epsilon)
        # print(f"current = {info['propositions']}, previous state = {self.ldba_state}, current state = {new_ldba_state}")
        if new_ldba_state != self.ldba_state:
            # print(f"current = {info['propositions']}, previous state = {self.ldba_state}, current state = {new_ldba_state}")
            self.ldba_state = new_ldba_state
            info['ldba_state_changed'] = True
        self.complete_observation(obs, info)
        if self.terminate_on_acceptance and accepting:
            terminated = True
            info['success'] = True
            # print(f"success, traj = {self.traj}")
        if accepting:
            self.num_accepting_visits += 1
        scc = self.ldba.state_to_scc[self.ldba_state]
        if scc.bottom and not scc.accepting:
            terminated = True
            info['violation'] = True
            # print(f"violation, traj = {self.traj}")
        info['accepting'] = accepting
        info['num_accepting_visits'] = self.num_accepting_visits
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # self.obs = obs
        # self.info= info
        # print(f"LDBAWrapper, formula = {obs['goal']}")
        self.ldba = self.construct_ldba(obs['goal'])
        self.terminate_on_acceptance = self.ldba.is_finite_specification()
        self.ldba_state = self.ldba.initial_state
        self.num_accepting_visits = 0
        self.complete_observation(obs, info)
        info['ldba_state_changed'] = True
        self.traj = []
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None):
        obs['ldba'] = self.ldba
        obs['ldba_state'] = self.ldba_state
        obs['propositions'] = info['propositions']

    @functools.cache
    def construct_ldba(self, formula: str) -> LDBA:
        # print(f"formula: {formula}")
        propositions = get_env_attr(self.env, 'get_propositions')()
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        # print(ldba.state_to_transitions)
        possible_assignments = get_env_attr(self.env, 'get_possible_assignments')()
        ldba.prune(possible_assignments)
        ldba.complete_sink_state()
        ldba.compute_sccs()
        # print(ldba.state_to_transitions)
        # for state, transitions in ldba.state_to_transitions.items():
        #     for t in transitions:
        #         print(f"state: {state}, transitions: {t}, label: {t.label}")
        initial_scc = ldba.state_to_scc[ldba.initial_state]
        if initial_scc.bottom and not initial_scc.accepting:
            raise ValueError(f'The language of the LDBA for {formula} is empty.')
        return ldba

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
        pre-process the observation
        """
        original_obs = self.task.original_obs
        lidar_dim = self.task.lidar_conf.num_bins
        # print(f"original_obs = {original_obs}")

        assert len(reach) == 1
        reach_zones = [r.to_string()[0]+"_zones_lidar" for r in list(reach)]
        reach_obs = np.vstack([original_obs[color] for color in reach_zones]) # len(reach_zones) x lidar_dim
        reach_obs = np.max(reach_obs, axis=0) # lidar_dim
        
        ### avoid
        avoid_zones = [a.to_string()[0] + "_zones_lidar" for a in list(avoid) if a.to_string()]
        if len(avoid_zones):
            avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
            avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
        else:
            avoid_obs = np.zeros(lidar_dim)
            
        agent_obs = np.concatenate(
            [original_obs[key].flatten() if original_obs[key].ndim > 1 
             else original_obs[key] for key in self.agent_obs_keys])
        # assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        obs = np.concatenate([agent_obs, reach_obs, avoid_obs])
        assert obs.shape == self.observation_space['features'].shape, \
            f"obs.shape = {obs.shape}, expected {self.observation_space['features'].shape}"
        return obs
    
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
        
    # #     reach_list = [r.to_string()[0] for r in list(reach)]
    # #     avoid_list = [a.to_string()[0] for a in list(avoid)]
    # #     reach_vec = np.isin(np.array(self.region_order), reach_list).astype(float)
    # #     avoid_vec = np.isin(np.array(self.region_order), avoid_list).astype(float)
    # #     ltl_emb = np.concatenate([reach_vec, avoid_vec])

    # #     return np.concatenate([agent_obs, lidar_obs, ltl_emb])
    
    def pre_process_obs_letter(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        
        # return self.env.original_obs
        # self.region_order
        obs = self.env.original_obs
        new_obs = np.zeros((obs.shape[0], obs.shape[1]), dtype=obs.dtype)
        letter_to_index = {letter: i for i, letter in enumerate(self.region_order)}
        
        reach_indices = [letter_to_index[r.to_string()[0]] for r in list(reach)]
        avoid_indices = [letter_to_index[a.to_string()[0]] for a in list(avoid) if a.to_string()]
        
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
    
    # def pre_process_obs(self,
    #                     reach: frozenset[FrozenAssignment], 
    #                     avoid: frozenset[FrozenAssignment]) -> np.ndarray:
    #     """
    #     pre-process the observation
    #     """
    #     original_obs = self.task.original_obs
    #     lidar_dim = self.task.lidar_conf.num_bins
    #     # print(f"original_obs = {original_obs}")
    #     # print(f"reach = {reach}, avoid = {avoid}")
        
        
    #     agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]        
    #     agent_obs = np.concatenate([original_obs[key] for key in agent_obs_keys])
        
    #     if reach == LDBASequence.EPSILON:
    #         reach_obs = np.zeros(lidar_dim)
    #     else:
    #         reach_zones = [r.to_string()[0]+"_zones_lidar" for r in list(reach)]
    #         reach_obs = np.vstack([original_obs[color] for color in reach_zones]) # len(reach_zones) x lidar_dim
    #         reach_obs = np.max(reach_obs, axis=0) # lidar_dim

    #     avoid_zones = []
    #     for a in list(avoid):
    #         if len(a.to_string()) == 0:
    #             continue
    #         # print(a)
    #         avoid_zones.append(a.to_string()[0]+"_zones_lidar")
    #     # avoid_zones = [a.to_string()[0]+"_zones_lidar" for a in list(avoid)]
    #     if len(avoid_zones):
    #         avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
    #         avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
    #     else:
    #         avoid_obs = np.zeros(lidar_dim)
            
    #     assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
    #     return np.concatenate([agent_obs, reach_obs, avoid_obs])


from gymnasium import spaces
class LDBAWrapperFixedSequence(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sampler, partial_reward=False):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            # 'features': env.observation_space['features'],
            # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
            'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
        })
        # print(f"env.observation_space['features'].shape = {env.observation_space['features'].shape}")
        # self.formula = formula
        formula = sampler()
        print(f"LDBAWrapperFixedSequence, formula = {formula}")
        self.lookuptable = self.construct_lookuptable()
        self.goal_seq = self.lookuptable[formula]
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None
        self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]
        self.region_order = env.get_propositions()
        # self.stay = ''

    def construct_lookuptable(self) -> LDBASequence:
        propositions = get_env_attr(self.env, 'get_propositions')()
        lookuptable = {
            "!blue U yellow": None,
            "!green U magenta": None,
            "!magenta U blue": None,
            "!yellow U green": None,
            
            "(!blue & !green) U yellow": None,
            "(!blue & !green) U magenta": None,
            "(!yellow & !magenta) U green": None,
            "(!green & !magenta) U blue": None,
            "(!yellow & !blue) U magenta": None,
            
            # "(!blue & !green) U (yellow | magenta)": None,
            # "(!yellow & !magenta) U (blue | green)": None,
            # "(!green & !magenta) U (yellow | blue)": None,
            # "(!yellow & !blue) U (green | magenta)": None,
            
            "(!blue & !magenta & !green) U yellow": None,
            "(!green & !blue & !yellow) U magenta": None,
            "(!magenta & !green & !yellow) U blue": None,
            "(!yellow & !blue & !magenta) U green": None,
            
            'F (green & (!blue U yellow)) & F magenta': None, # ((frozenset({green}), frozenset()), (frozenset({magenta}), frozenset({blue})), (frozenset({yellow}), frozenset({blue})))
            '(F blue) & (!blue U (green & F yellow))': None, # ((frozenset({green}), frozenset({blue})), (frozenset({yellow}), frozenset()), (frozenset({blue}), frozenset()))
            'F (blue | green) & F yellow & F magenta': None, # ((frozenset({blue, green}), frozenset()), (frozenset({yellow}), frozenset()), (frozenset({magenta}), frozenset()))
            '!(magenta | yellow) U (blue & F green)': None, # ((frozenset({blue}), frozenset({magenta, yellow})), (frozenset({green}), frozenset()))
            '!green U ((blue | magenta) & (!green U yellow))': None, # ((frozenset({magenta, blue}), frozenset({green})), (frozenset({yellow}), frozenset({green})))
            '((green | blue) => (!yellow U magenta)) U yellow': None, # ((frozenset({yellow}), frozenset({green, blue})),)
            
            
            
            '!(yellow | magenta) U green': None,
            '!(blue | green) U magenta': None,
            '!(green | blue) U magenta': None,
            '!yellow U magenta': None,
            '!magenta U yellow': None,

            'FG blue': None,
            
            # PointLtlSafety3-v0
            "!cyan U red": None,
            "!red U cyan": None,
            "!red U magenta": None,
            "!cyan U blue": None,
            
            "(!blue & !green) U red": None,
            "(!yellow & !magenta) U cyan": None,
            "(!green & !red) U blue": None,
            "(!cyan & !blue) U magenta": None,
            
            "(!blue & !magenta & !green) U red": None,
            "(!green & !blue & !yellow) U cyan": None,
            "(!magenta & !cyan & !yellow) U blue": None,
            "(!red & !cyan & !magenta) U green": None,
            
            "(!blue & !magenta & !green & !yellow) U red": None,
            "(!green & !blue & !yellow & !red) U cyan": None,
            "(!blue & !cyan & !yellow & !red) U magenta": None,
            "(!red & !blue & !magenta & !cyan) U green": None,
            
            "(!blue & !magenta & !green & !yellow & !cyan) U red": None,
            "(!green & !blue & !yellow & !red & !magenta) U cyan": None,
            "(!blue & !cyan & !yellow & !red & !green) U magenta": None,
            "(!red & !blue & !magenta & !cyan & !yellow) U green": None,
            
            # PointLtlSafety4-v0
            "!cyan U orange": None,
            "!red U purple": None,
            "!blue U cyan": None,
            "!magenta U red": None,
            
            # PointLtlSafety5-v0
            "!cyan U orange": None,
            "!lime U purple": None,
            "!teal U cyan": None,
            "!red U teal": None,
        }
        for formula in lookuptable.keys():
            if formula == '!blue U yellow':
                reach = ['yellow']
                avoid = ['blue']
            elif formula == '!green U magenta':
                reach = ['magenta']
                avoid = ['green']
            elif formula == '!magenta U blue':
                reach = ['blue']
                avoid = ['magenta']
            elif formula == '!yellow U green':
                reach = ['green']
                avoid = ['yellow']
            # elif formula == '(!blue & !green) U (yellow | magenta)':
            #     reach = [['yellow', 'magenta']]
            #     avoid = [['blue', 'green']]
            # elif formula == '(!yellow & !magenta) U (blue | green)':
            #     reach = [['blue', 'green']]
            #     avoid = [['yellow', 'magenta']]
            # elif formula == '(!green & !magenta) U (yellow | blue)':
            #     reach = [['yellow', 'blue']]
            #     avoid = [['green', 'magenta']]
            # elif formula == '(!yellow & !blue) U (green | magenta)':
            #     reach = [['green', 'magenta']]
            #     avoid = [['yellow', 'blue']]
            elif formula == "(!blue & !green) U yellow":
                reach = ['yellow']
                avoid = [['blue', 'green']]
            elif formula == "(!blue & !green) U magenta":
                reach = ['magenta']
                avoid = [['blue', 'green']]
            elif formula == "(!yellow & !magenta) U green":
                reach = ['green']
                avoid = [['yellow', 'magenta']]
            elif formula == "(!green & !magenta) U blue":
                reach = ['blue']
                avoid = [['green', 'magenta']]
            elif formula == "(!yellow & !blue) U magenta":
                reach = ['magenta']
                avoid = [['yellow', 'blue']]
            elif formula == '(!blue & !magenta & !green) U yellow':
                reach = ['yellow']
                avoid = [['blue', 'magenta', 'green']]
            elif formula == '(!green & !blue & !yellow) U magenta':
                reach = ['magenta']
                avoid = [['green', 'blue', 'yellow']]
            elif formula == '(!magenta & !green & !yellow) U blue':
                reach = ['blue']
                avoid = [['magenta', 'green', 'yellow']]
            elif formula == '(!yellow & !blue & !magenta) U green':
                reach = ['green']
                avoid = [['yellow', 'blue', 'magenta']]
            elif formula == 'F (green & (!blue U yellow)) & F magenta':
                reach = ['green', 'magenta', 'yellow']
                avoid = ['blue', 'blue', 'blue']
            elif formula == '(F blue) & (!blue U (green & F yellow))':
                reach = ['green', 'yellow', 'blue']
                avoid = ['blue', 'blue', '']
            elif formula == 'F (blue | green) & F yellow & F magenta':
                reach = [['blue', 'green'], 'yellow', 'magenta']
                avoid = ['', '', '']
            elif formula == '!(magenta | yellow) U (blue & F green)':
                reach = ['blue', 'green']
                avoid = [['magenta', 'yellow'], ['magenta', 'yellow']]
            # elif formula == '!(magenta | yellow) U (blue & F green)':
            #     reach = ['green', 'blue']
            #     avoid = [['magenta', 'yellow'], ['magenta', 'yellow']]
            elif formula == '!green U ((blue | magenta) & (!green U yellow))':
                reach = [['magenta', 'blue'], 'yellow']
                avoid = ['green', 'green']
            elif formula == '((green | blue) => (!yellow U magenta)) U yellow':
                reach = ['yellow']
                avoid = [['green', 'blue'], '']
            elif formula == '!(yellow | magenta) U green':
                reach = ['green']
                avoid = [['yellow', 'magenta']]
            elif formula == '!(blue | green) U magenta':
                reach = ['magenta']
                avoid = [['blue', 'green']]
            elif formula == '!(green | blue) U magenta':
                reach = ['magenta']
                avoid = [['green', 'blue']]
            elif formula == '!yellow U magenta':
                reach = ['magenta']
                avoid = ['yellow']
            elif formula == 'FG blue':
                reach = ['blue']
                avoid = ['']
            elif formula == '!magenta U yellow':
                reach = ['yellow']
                avoid = ['magenta']
                
            # PointLtlSafety3-v0
            # until1
            elif formula == "!cyan U red":
                reach = ['red']
                avoid = ['cyan']
            elif formula == "!red U cyan":
                reach = ['cyan']
                avoid = ['red']
            elif formula == "!red U magenta":
                reach = ['magenta']
                avoid = ['red']
            elif formula == "!cyan U blue":
                reach = ['blue']
                avoid = ['cyan']
            # until2
            elif formula == "(!blue & !green) U red":
                reach = ['red']
                avoid = [['blue', 'green']]
            elif formula == "(!yellow & !magenta) U cyan":
                reach = ['cyan']
                avoid = [['yellow', 'magenta']]
            elif formula == "(!green & !red) U blue":
                reach = ['blue']
                avoid = [['green', 'red']]
            elif formula == "(!cyan & !blue) U magenta":
                reach = ['magenta']
                avoid = [['cyan', 'blue']]
            # until3
            elif formula == "(!blue & !magenta & !green) U red":
                reach = ['red']
                avoid = [['blue', 'magenta', 'green']]
            elif formula == "(!green & !blue & !yellow) U cyan":
                reach = ['cyan']
                avoid = [['green', 'blue', 'yellow']]
            elif formula == "(!magenta & !cyan & !yellow) U blue":
                reach = ['blue']
                avoid = [['magenta', 'cyan', 'yellow']]
            elif formula == "(!red & !cyan & !magenta) U green":
                reach = ['green']
                avoid = [['red', 'cyan', 'magenta']]
            # until4
            elif formula == "(!blue & !magenta & !green & !yellow) U red":
                reach = ['red']
                avoid = [['blue', 'magenta', 'green', 'yellow']]
            elif formula == "(!green & !blue & !yellow & !red) U cyan":
                reach = ['cyan']
                avoid = [['green', 'blue', 'yellow', 'red']]
            elif formula == "(!blue & !cyan & !yellow & !red) U magenta":
                reach = ['magenta']
                avoid = [['blue', 'cyan', 'yellow', 'red']]
            elif formula == "(!red & !blue & !magenta & !cyan) U green":
                reach = ['green']
                avoid = [['red', 'blue', 'magenta', 'cyan']]
            # until5
            elif formula == "(!blue & !magenta & !green & !yellow & !cyan) U red":
                reach = ['red']
                avoid = [['blue', 'magenta', 'green', 'yellow', 'cyan']]
            elif formula == "(!green & !blue & !yellow & !red & !magenta) U cyan":
                reach = ['cyan']
                avoid = [['green', 'blue', 'yellow', 'red', 'magenta']]
            elif formula == "(!blue & !cyan & !yellow & !red & !green) U magenta":
                reach = ['magenta']
                avoid = [['blue', 'cyan', 'yellow', 'red', 'green']]
            elif formula == "(!red & !blue & !magenta & !cyan & !yellow) U green":
                reach = ['green']
                avoid = [['red', 'blue', 'magenta', 'cyan', 'yellow']]
                
            # PointLtlSafety4-v0
            # until1
            elif formula == "!cyan U orange":
                reach = ['orange']
                avoid = ['cyan']
            elif formula == "!red U purple":
                reach = ['purple']
                avoid = ['red']
            elif formula == "!blue U cyan":
                reach = ['cyan']
                avoid = ['blue']
            elif formula == "!magenta U red":
                reach = ['red']
                avoid = ['magenta']
                
            # PointLtlSafety5-v0
            # until1
            elif formula == "!cyan U orange":
                reach = ['orange']
                avoid = ['cyan']
            elif formula == "!lime U purple":
                reach = ['purple']
                avoid = ['lime']
            elif formula == "!teal U cyan":
                reach = ['cyan']
                avoid = ['teal']
            elif formula == "!red U teal":
                reach = ['teal']
                avoid = ['red']
            
            seq = []
            for r, a in zip(reach, avoid):
                if type(r) == list:
                    reach_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in r])
                else:
                    reach_assignments = frozenset([Assignment.single_proposition(r, propositions).to_frozen()])
                if type(a) == list:
                    avoid_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in a])
                else:
                    avoid_assignments = frozenset([Assignment.single_proposition(a, propositions).to_frozen()]) if len(a) > 0 else frozenset()
                seq.append((reach_assignments, avoid_assignments))
                
            lookuptable[formula] = LDBASequence(seq)
        return lookuptable
            
    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info['propositions']
        # print(f"reached = {active_props}")
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            # self.stay += '1'
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        # else:
        #     self.stay += '0'
        reach, avoid = self.goal_seq[self.num_reached] \
            if self.num_reached < len(self.goal_seq) else self.goal_seq[-1]
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation(obs, info)
        
        self.obs = obs
        self.info = info
        
        # print(obs)
        return obs, reward, terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        # self.stay = ''
        # print(obs)
        # print(f"self.goal_seq = {self.goal_seq}")
        reach, avoid = self.goal_seq[self.num_reached]
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation(obs, info)
        self.obs = obs
        # print(obs)
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs["features"],
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }
        
    # def pre_process_obs(self,
    #                     reach: frozenset[FrozenAssignment], 
    #                     avoid: frozenset[FrozenAssignment]) -> np.ndarray:
    #     """
    #     pre-process the observation
    #     """
    #     original_obs = self.task.original_obs
    #     # print(f"original_obs = {original_obs}")
        
    #     reach_zones = [r.to_string()[0]+"_zones_lidar" for r in list(reach)]
    #     avoid_zones = [a.to_string()[0]+"_zones_lidar" for a in list(avoid)]
        
    #     agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]        
    #     agent_obs = np.concatenate([original_obs[key] for key in agent_obs_keys])
    #     lidar_dim = 16
        
    #     reach_obs = np.vstack([original_obs[color] for color in reach_zones]) # len(reach_zones) x lidar_dim
    #     reach_obs = np.max(reach_obs, axis=0) # lidar_dim
    #     if len(avoid_zones):
    #         avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
    #         avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
    #     else:
    #         avoid_obs = np.zeros(lidar_dim)
            
    #     assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
    #     return {'features': np.concatenate([agent_obs, reach_obs, avoid_obs])}

    def pre_process_obs(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        re-arrange the order of observations
        """
        original_obs = self.task.original_obs
        # print(f"original_obs = {original_obs}")
        agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])
        lidar_obs = np.concatenate([original_obs[region+"_zones_lidar"] for region in self.region_order])
        return {'features': np.concatenate([agent_obs, lidar_obs])}