# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi Goal with a SAR environment."""

import json
import time
from copy import deepcopy
from pathlib import Path

import gymnasium
import mujoco
import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Walls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.zones import Zones
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.tasks.safe_multi_agent.utils.common_utils import rot2quat
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import *
from safety_gymnasium.tasks.safe_multi_agent import agents
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


CASUALTY_KEEPOUT = 0.2
class MultiGoalSARLevel0(BaseTask):
    """Multi-agent zone navigation with optional ring-placed interior walls."""
    _cached_wall_half_sizes = None
    _cached_building_locations = None
    _cached_building_rots = None

    wall_ring_radius = 2.0
    wall_base_half_sizes = [0.1, 0.3, 0.2]
    wall_count = 20
    wall_margin = 1.0
    building_keepout = 0.3
    building_border_side_length = 4.5
    building_margin = 0.8
    casualty_keepout = 0.2
    agent_keepout = 0.25
    max_dist = None
    reward_distance = 1.0
    reward_goal = 1.0

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-3.5, -3.5, 3.5, 3.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = self.max_dist
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.lidar_conf.type = 'pseudo_occluded'  # choices: 'pseudo' 'natural' 'pseudo_occluded'
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False
        self.render_conf.lidar_markers = False
        self.mechanism_conf.continue_goal = False
        self.last_dist_casualty = None

        # Spawn agents in a specified area
        self._build_agent(self.agent_name, keepout=self.agent_keepout, placements=[(-0.67, -0.67, 0.67, 0.67)])

        casualty_num = 1 if self.agent_num == 1 else self.agent_num
        self._add_geoms(
            LtlWalls(contype=1),
            Casualtys(
                category=list(Casualtys.CATEGORIES)[-2],
                size=0.05,
                num=casualty_num,
                keepout=self.casualty_keepout,
            ),
        )

        if self.agent_num > 1:
            self._add_mocaps(
                Gremlins(num=config['agent_num'], size=0.15, dist_threshold=0.10, keepout=0.0)
            )

    def _dist_to_casualty(self, agent_idx: int) -> float:
        if not hasattr(self, 'surface_casualtys'):
            return 0.0
        casualty_pos = self.surface_casualtys.pos[0]
        return self.agent.dist_xy(agent_idx, casualty_pos)

    def calculate_reward(self):
        rewards = {}
        touch_threshold = 0.0
        if hasattr(self, 'surface_casualtys'):
            touch_threshold = self.surface_casualtys.size + 0.15
        for i in range(self.agent_num):
            reward = 0.0
            dist = self._dist_to_casualty(i)
            if self.last_dist_casualty is not None:
                reward += (self.last_dist_casualty[i] - dist) * self.reward_distance
            self.last_dist_casualty[i] = dist
            if dist <= touch_threshold:
                reward += self.reward_goal
            rewards[f'agent_{i}'] = reward
        return rewards

    def specific_reset(self):
        self.last_dist_casualty = [self._dist_to_casualty(i) for i in range(self.agent_num)]
        return super().specific_reset()

    # #region agent log
    def _dbg_reset(self, hypothesis_id, message, data):
        try:
            root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
            payload = {
                'sessionId': 'b1323e',
                'hypothesisId': hypothesis_id,
                'location': 'multi_sar_level0.py:reset',
                'message': message,
                'data': data,
                'timestamp': int(time.time() * 1000),
            }
            with open(root / 'debug-b1323e.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload) + '\n')
        except Exception:
            pass
    # #endregion

    def _apply_layout_from_config(self) -> None:
        """Update MuJoCo body poses without rebuilding the model from XML."""
        config = self.world_info.world_config_dict
        agent_xy = config['agent_xy']
        if isinstance(agent_xy, list):
            agent_positions = [np.asarray(pos, dtype=float) for pos in agent_xy]
        else:
            agent_positions = [np.asarray(agent_xy, dtype=float)]

        agent_rot = config['agent_rot']
        if np.isscalar(agent_rot):
            agent_rots = [float(agent_rot)] * self.agent_num
        else:
            agent_rots = [float(r) for r in agent_rot]

        mujoco.mj_resetData(self.model, self.data)  # pylint: disable=no-member
        z = self.agent.z_height
        for i in range(self.agent_num):
            body_name = f'agent_{i}'
            xy = agent_positions[i][:2]
            self.model.body(body_name).pos[:2] = xy
            self.model.body(body_name).pos[2] = z
            self.model.body(body_name).quat[:] = rot2quat(agent_rots[i])

        for geom_name, geom_cfg in config.get('geoms', {}).items():
            pos = np.asarray(geom_cfg['pos'], dtype=float)
            self._set_goal(geom_name, pos[:2])
            if pos.shape[0] >= 3:
                self.model.body(geom_name).pos[2] = pos[2]

        self.data.qvel[:] = 0
        if self.model.na:
            self.data.act[:] = 0
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def reset(self) -> None:
        """Reset task state; reuse MuJoCo model after the first build."""
        if self.world is None:
            t0 = time.perf_counter()
            super().reset()
            self._dbg_reset('H6', 'full reset completed', {
                'path': 'full',
                'elapsed_sec': round(time.perf_counter() - t0, 4),
            })
            return

        t0 = time.perf_counter()
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        if self.random_generator.agent_num is None:
            self.random_generator.agent_num = self.agent.agent_num

        self.world_info.layout = self.random_generator.build_layout()
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)

        if hasattr(self, 'surface_casualtys'):
            self.surface_casualtys.rescued = [False] * self.surface_casualtys.num
        self.last_dist_casualty = None

        self._apply_layout_from_config()
        self.world_info.reset_layout = deepcopy(self.world_info.layout)
        self._dbg_reset('H6', 'fast reset completed', {
            'path': 'fast',
            'elapsed_sec': round(time.perf_counter() - t0, 4),
        })

    def specific_step(self):
        return super().specific_step()

    def update_world(self):
        pass

    def _replace_geom(self, geom) -> None:
        """Update _geoms like _add_geoms but without duplicate registration checks."""
        self._geoms[geom.name] = geom
        setattr(self, geom.name, geom)
        geom.set_agent(self.agent)

    def _build(self):
        return super()._build()

    def try_lidar_ids(self, obstacle, obs, i):
        want_ids = getattr(obstacle, 'is_lidar_ids_observed', False)
        if want_ids and self.lidar_conf.type == 'pseudo_occluded':
            lidar, lidar_ids = self._obs_lidar_pseudo_occluded_new(
                i, obstacle, return_ids=True,
            )
            obs[f"{obstacle.name}_lidar_{i}"] = lidar
            obs[f"{obstacle.name}_lidar_ids_{i}"] = lidar_ids
        else:
            obs[f"{obstacle.name}_lidar_{i}"] = self._obs_lidar_new(
                i, obstacle.pos, obstacle.group, obstacle=obstacle,
            )  

    def obs(self) -> dict | np.ndarray:
            """Return the observation of our agent."""
            # pylint: disable-next=no-member
            mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
            obs = {}
    
            obs.update(self.agent.obs_sensor())
    
            # observations of obstacles
            inside_building = False
            for obstacle in self._obstacles:
                if "terracotta" in obstacle.name and "building" in obstacle.name and any(obstacle.cal_cost())>0:
                    inside_building = True
                # print(f"obstacle.name: {obstacle.name}, obstacle.pos: {obstacle.pos}, obstacle.group: {obstacle.group}")
                if obstacle.is_lidar_observed:
                    if 'gremlins' in obstacle.name:
                        for i in range(self.agent_num):
                            name = f"{obstacle.name}_lidar_{i}"
                            poses = obstacle.pos.copy()
                            del poses[i]
                            obs[name] = self._obs_lidar_new(
                                i, poses, obstacle.group, obstacle=obstacle,
                            )
                    elif inside_building and ("entrapped" in obstacle.name or obstacle.name == "walls"):
                        for i in range(self.agent_num):
                            name = f"{obstacle.name}_lidar_{i}"
                            obs[name] = self._obs_lidar_pseudo_new(i, obstacle.pos)
                        # print(f"DEBUG: obstacle names: {str(obstacle.name)}")
                    else:
                        for i in range(self.agent_num):
                            self.try_lidar_ids(obstacle, obs, i)            
    
                    
                if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                    obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)
    
            if self.observe_vision:
                for i in range(self.agent_num):
                    name = f'vision_{i}'
                    obs[name] = self._obs_vision(camera_name=name)
            # print(f"DEBUG: obs before flatten: {obs}")
            # assert self.obs_info.obs_space_dict.contains(
            #     obs,
            # ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'
            # print(f"obs: {obs}")
            # self.original_obs = obs
            if self.observation_flatten:
                obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
            return obs

    @property
    def goal_achieved(self):
        if not hasattr(self, 'surface_casualtys'):
            return tuple(False for _ in range(self.agent_num))
        rescued = self.surface_casualtys.rescued
        if self.agent_num == 1:
            return (rescued[0],)
        return tuple(
            rescued[i] if i < len(rescued) else False
            for i in range(self.agent_num)
        )
