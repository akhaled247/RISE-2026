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

class MultiGoalSARLevel4(MultiGoalSARLevel0):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    wall_count = 20
    
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self._add_geoms(
            Walls(num=self.wall_count)
        )

    def calculate_reward(self):
        return super().calculate_reward()

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
        # randomized wall sizes that persist between runs
        self._cached_wall_half_sizes = size_randomization(
            self.wall_base_half_sizes, 
            self.wall_count, 
            margins=(np.array(self.wall_base_half_sizes)/2).tolist(), #make tolerance half of size 
            random_generator=self.random_generator
            ) if self._cached_wall_half_sizes is None else self._cached_wall_half_sizes
        self._replace_geom(Walls(
                num=self.wall_count,
                placements=ring_placements(
                    self.wall_ring_radius, self.wall_count, margin=self.wall_margin
                ),
                half_sizes=self._cached_wall_half_sizes,
                keepout=0.4,
            ))       
        return super()._build()

