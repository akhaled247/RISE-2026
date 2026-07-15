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
import numpy as np

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
    time_alive_decay = 0.0
    surface_casualtys_frac: float = 1.0
    entrapped_casualtys_frac: float = 0.0

    def __init__(self, config) -> None:
        self._cached_wall_half_sizes = None
        self._cached_building_locations = None
        self._cached_building_rots = None
        super().__init__(config=config)

        self.placements_conf.extents = [-3.5, -3.5, 3.5, 3.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = self.max_dist
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.lidar_conf.type = 'pseudo'  # choices: 'pseudo' 'natural' 'pseudo_occluded'
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False
        self.render_conf.lidar_markers = False
        self.mechanism_conf.continue_goal = False
        self.last_dist_casualty = None

        # Spawn agents in a specified area
        self._build_agent(self.agent_name, keepout=self.agent_keepout, placements=[(-0.67, -0.67, 0.67, 0.67)])
        surface_casualtys_int = int(self.agent_num * self.surface_casualtys_frac)
        # One surface casualty for solo training; otherwise one per agent.
        self.casualty_num = self.agent_num
        self._add_geoms(
            LtlWalls(contype=1),
        )

        if surface_casualtys_int>0: 
            self._add_geoms(
                Casualtys(
                    category=list(Casualtys.CATEGORIES)[-2],
                    size=0.05,
                    num=surface_casualtys_int,
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

    def _dist_to_casualtys(self, agent_idx: int) -> list[float]:
        if hasattr(self, 'surface_casualtys'):
            casualty_poses = (self.surface_casualtys.pos[i] for i in range(self.casualty_num))
            return [self.agent.dist_xy(agent_idx, pos) for pos in casualty_poses]
        elif hasattr(self, 'entrapped_casualtys'):
            casualty_poses = (self.entrapped_casualtys.pos[i] for i in range(self.casualty_num))
            return [self.agent.dist_xy(agent_idx, pos) for pos in casualty_poses]
        return []
            

    def build_observation_space(self) -> gymnasium.spaces.Dict:
        super().build_observation_space()
        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict
        return self.observation_space

    def calculate_reward(self):
        """Distance delta toward visible casualty and touch bonus."""
        rewards = {}
        touch_threshold = 0.0
        if hasattr(self, 'surface_casualtys'):
            touch_threshold = self.surface_casualtys.size + 0.15
        for i in range(self.agent_num):
            a = f'agent_{i}'
            reward = self.time_alive_decay

            # Distance-based reward shaping
            dists = self._dist_to_casualtys(i)
            min_dist = min(dists) if dists else 0.0
            if min_dist <= touch_threshold:
                print("uhoh")
                reward += self.reward_goal
            # else:
            #     reward += self.last_dist_casualty[i]-min_dist
            self.last_dist_casualty[i] = min_dist

            rewards[a] = reward
        return rewards

    def specific_reset(self):
        """Reset SAR-specific episode state after layout resample."""
        if hasattr(self, 'surface_casualtys'):
            self.surface_casualtys.rescued = [False] * self.surface_casualtys.num
        if hasattr(self, 'entrapped_casualtys'):
            self.entrapped_casualtys.rescued = [False] * self.entrapped_casualtys.num
        self.last_dist_casualty = [self._dist_to_casualty(i) for i in range(self.agent_num)]
        return super().specific_reset()

    def specific_step(self):
        return super().specific_step()

    def update_world(self):
        pass

    def _building_geom(self):
        for name in self._geoms:
            if name.endswith('_buildings'):
                return getattr(self, name)
        return None

    def _update_building_ltl_wall_site(self, wall, center_xy, rot) -> None:
        wall.d_x, wall.d_y = center_xy[0], center_xy[1]
        wall.theta = rot
        wall.locations = [
            (wall.locate_factor + wall.d_x, wall.d_y),
            (-wall.locate_factor + wall.d_x, wall.d_y),
            (wall.d_x, wall.locate_factor + wall.d_y),
            (wall.d_x, -wall.locate_factor + wall.d_y),
        ]
        cos_t, sin_t = np.cos(wall.theta), np.sin(wall.theta)
        wall.locations = [
            (
                (x - wall.d_x) * cos_t - (y - wall.d_y) * sin_t + wall.d_x,
                (x - wall.d_x) * sin_t + (y - wall.d_y) * cos_t + wall.d_y,
            )
            for x, y in wall.locations
        ]

    def _resample_building_sites(self) -> None:
        buildings = self._building_geom()
        if buildings is None:
            return
        self._cached_building_locations = [
            draw_border_placement_from_loop(
                self.building_border_side_length,
                self.building_margin,
                self.building_keepout,
                i,
                self.random_generator,
            )
            for i in range(self.agent_num)
        ]
        self._cached_building_rots = self.random_generator.generate_rots(self.agent_num)
        buildings.locations = list(self._cached_building_locations)
        buildings.rots = list(self._cached_building_rots)
        if hasattr(self, 'entrapped_casualtys'):
            self.entrapped_casualtys.locations = list(self._cached_building_locations)
        for i in range(self.agent_num):
            wall_name = f'building{i}_ltl_walls'
            if hasattr(self, wall_name):
                self._update_building_ltl_wall_site(
                    getattr(self, wall_name),
                    self._cached_building_locations[i],
                    self._cached_building_rots[i],
                )

    def reset(self) -> None:
        self._resample_building_sites()
        super().reset()

    def _replace_geom(self, geom) -> None:
        """Update _geoms like _add_geoms but without duplicate registration checks."""
        self._geoms[geom.name] = geom
        setattr(self, geom.name, geom)
        geom.set_agent(self.agent)

    def _build(self):
        return super()._build()

    def try_lidar_ids(self, obstacle, obs, i):
        """pseudo_occluded lidar with per-instance line-of-sight (walls block view)."""
        want_ids = getattr(obstacle, 'is_lidar_ids_observed', False)
        is_occluded = getattr(obstacle, 'is_occluded', True)
        if want_ids and self.lidar_conf.type == 'pseudo_occluded':
            lidar, lidar_ids = self._obs_lidar_pseudo_occluded_new(
                i, obstacle, return_ids=True,
            )
            obs[f"{obstacle.name}_lidar_{i}"] = lidar
            obs[f"{obstacle.name}_lidar_ids_{i}"] = lidar_ids
        elif not is_occluded:
            for i in range(self.agent.agent_num):
                name = f"{obstacle.name}_lidar_{i}"
                obs[name] = self._obs_lidar_pseudo_new(i, obstacle.pos)
        else:
            obs[f"{obstacle.name}_lidar_{i}"] = self._obs_lidar_pseudo_occluded_new(
                i, obstacle,
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
                        self.try_lidar_ids(obstacle, obs, i)
                else:
                    for i in range(self.agent_num):
                        self.try_lidar_ids(obstacle, obs, i)

            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)

        if self.observe_vision:
            for i in range(self.agent_num):
                name = f'vision_{i}'
                obs[name] = self._obs_vision(camera_name=name)
        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    @property
    def goal_achieved(self):
        return mission_goal_achieved(self)
