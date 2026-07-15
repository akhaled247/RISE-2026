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
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level0 import MultiGoalSARLevel0

BUILDING_KEEPOUT = 0.3
BUILDING_BORDER_SIDE_LENGTH = 4.5
BUILDING_MARGIN = 0.8

CASUALTY_KEEPOUT = 0.2

class MultiGoalSARLevel1(MultiGoalSARLevel0):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        for i in range(self.agent_num):
            self._add_geoms(LtlWalls(name=f"building{i}_ltl_walls"))

        self._add_geoms(
            Buildings(
                color=list(Buildings.COLORS)[0],
                size=self.building_keepout*0.75,
                num=self.agent_num),
            Casualtys(
                            category=list(Casualtys.CATEGORIES)[-2],
                            size=0.05,
                            num=self.agent_num-(self.agent_num//2),
                            keepout=self.casualty_keepout,),
            Casualtys(
                num=self.agent_num//2,
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05),
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
        self._replace_geom(Buildings(
                color=list(Buildings.COLORS)[0],
                size=self.building_keepout*0.75,
                num=self.agent_num,
                keepout=0.0,
                locations=self._cached_building_locations,
                debug=False,
                rots=self._cached_building_rots,
                    ))
        self._replace_geom(Casualtys(
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05,
                num=self.agent_num//2,
                keepout=0.0,
                locations=self._cached_building_locations))
        
        # set ltl walls to surround building geoms
        for i in range(self.agent_num):     
            self._replace_geom(LtlWalls(
                name=f'building{i}_ltl_walls',
                locate_factor=self.building_keepout*0.75,
                size=self.building_keepout*0.75,
                height=0.75,
                locations=self._cached_building_locations[i],
                rots=self._cached_building_rots,
                collision_threshold=8.0))
            
        return super()._build()

