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

import gymnasium
import mujoco

from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import Walls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.zones import Zones
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
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

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-3.5, -3.5, 3.5, 3.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = 2.0
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.lidar_conf.type = 'pseudo_occluded'  # choices: 'pseudo' 'natural' 'pseudo_occluded'
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False
        self.render_conf.lidar_markers = False
        # self.render_conf.lidar_size = 1.0

        # Spawn agents in a specified area
        self._build_agent(self.agent_name, keepout=self.agent_keepout, placements=[(-0.67, -0.67, 0.67, 0.67)])

        self._add_geoms(
            LtlWalls(),

            #Surface Casualties
            Casualtys(
                category=list(Casualtys.CATEGORIES)[-2],
                size=0.05,
                num=self.agent_num,
                keepout=self.casualty_keepout,
            ),
        )

        self._add_mocaps(
            Gremlins(num=config['agent_num'], size=0.15, dist_threshold=0.10, keepout=0.0)
        )

    def calculate_reward(self):
        return {f'agent_{i}': 0.0 for i in range(self.agent_num)}

    def specific_reset(self):
        # print(f"GEOM KEYS: {(self._geoms.keys())}")
        # print(f"BUILDING SIZE: {self._geoms.get('building_ltl_walls_0').size}")
        return super().specific_reset()

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
                            name = f"{obstacle.name}_lidar_{i}"
                            obs[name] = self._obs_lidar_new(
                                i, obstacle.pos, obstacle.group, obstacle=obstacle,
                            )                
    
                    
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
        return tuple(False for _ in range(self.agent_num))
