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


# Ring layout for interior walls
WALL_RING_RADIUS = 2.0
WALL_BASE_HALF_SIZES = [0.1, 0.3, 0.2]
WALL_COUNT = 20
BUILDING_KEEPOUT = 0.3
BUILDING_BORDER_SIDE_LENGTH = 4.5
BUILDING_MARGIN = 0.8
HUMAN_KEEPOUT = 0.2
WALL_MARGIN = 1.0

class MultiGoalSAR(BaseTask):
    """Multi-agent zone navigation with optional ring-placed interior walls."""
    _cached_wall_half_sizes = None
    _cached_building_locations = None
    _cached_building_rots = None

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

        self._build_agent(self.agent_name, keepout=0.2, placements=[(-0.67, -0.67, 0.67, 0.67)])

        for i in range(self.agent_num):
            self._add_geoms(LtlWalls(name=f"building{i}_ltl_walls"))

        self._add_geoms(
            LtlWalls(),
            # Zones(color='green', size=0.3, num=2, keepout=0.5),
            Buildings(
                color=list(Buildings.COLORS)[0],
                size=BUILDING_KEEPOUT*0.75,
                num=self.agent_num),
            Casualtys(
                category=list(Casualtys.CATEGORIES)[-2],
                size=0.05,
                num=self.agent_num-(self.agent_num//2),
                keepout=HUMAN_KEEPOUT,
            ),
            Casualtys(
                num=self.agent_num//2,
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05),
            Walls(num=WALL_COUNT),
        )

        self._add_mocaps(
            Gremlins(num=config['agent_num'], size=0.15, dist_threshold=0.15, keepout=0.0)
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

    # def _build_agent(self, agent_name, locations: list = None):
    #     """Build the agent in the world."""
    #     assert hasattr(agents, agent_name), 'agent not found'
    #     agent_cls = getattr(agents, agent_name)
    #     self.agent = agent_cls(agent_num=self.agent_num, random_generator=self.random_generator,
    #                            locations=locations)

    def _build(self):
        # randomized wall sizes that persist between runs
        self._cached_wall_half_sizes = size_randomization(
            WALL_BASE_HALF_SIZES, 
            WALL_COUNT, 
            margins=(np.array(WALL_BASE_HALF_SIZES)/2).tolist(), #make tolerance half of size 
            random_generator=self.random_generator
            ) if self._cached_wall_half_sizes is None else self._cached_wall_half_sizes
        self._replace_geom(Walls(
                num=WALL_COUNT,
                placements=ring_placements(
                    WALL_RING_RADIUS, WALL_COUNT, margin=WALL_MARGIN
                ),
                half_sizes=self._cached_wall_half_sizes,
                keepout=0.4,
            ))
        
        # randomized building/entrapped casualty locations that persist between runs
        self._cached_building_locations = [draw_border_placement_from_loop(
            BUILDING_BORDER_SIDE_LENGTH,
            BUILDING_MARGIN,
            BUILDING_KEEPOUT,
            i,
            self.random_generator
            ) for i in range(self.agent_num)] if self._cached_building_locations is None else self._cached_building_locations
        self._cached_building_rots=self.random_generator.generate_rots(self.agent_num)
        self._replace_geom(Buildings(
                color=list(Buildings.COLORS)[0],
                size=BUILDING_KEEPOUT*0.75,
                num=self.agent_num,
                keepout=0.0,
                locations = self._cached_building_locations,
                debug=False,
                rots = self._cached_building_rots
                    ))
        self._replace_geom(Casualtys(
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05,
                num=self.agent_num//2,
                keepout=0.0,
                locations = self._cached_building_locations))
        
        # set ltl walls to surround building geoms
        for i in range(self.agent_num):     
            self._replace_geom(LtlWalls(
                name=f'building{i}_ltl_walls',
                locate_factor=BUILDING_KEEPOUT*0.75,
                size=BUILDING_KEEPOUT*0.75,
                height=0.75,
                locations=self._cached_building_locations[i],
                rots = self._cached_building_rots,
                collision_threshold=8.0))
            
        return super()._build()

    def obs(self) -> dict | np.ndarray:
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        # observations of obstacles
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
