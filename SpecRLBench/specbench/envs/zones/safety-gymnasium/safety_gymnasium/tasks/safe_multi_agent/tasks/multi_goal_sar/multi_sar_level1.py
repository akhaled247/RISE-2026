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

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms import LtlWalls
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.buildings import Buildings
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.casualtys import Casualtys
from safety_gymnasium.tasks.safe_multi_agent.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import border_placements
from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level0 import MultiGoalSARLevel0


class MultiGoalSARLevel1(MultiGoalSARLevel0):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        for i in range(self.agent_num):
            self._add_geoms(LtlWalls(name=f'building{i}_ltl_walls'))

        self._add_geoms(
            Buildings(
                color=list(Buildings.COLORS)[0],
                size=self.building_keepout * 0.75,
                num=self.agent_num,
                keepout=self.building_keepout,
                placements=border_placements(
                    self.building_border_side_length,
                    self.building_margin,
                ),
            ),
            Casualtys(
                category=list(Casualtys.CATEGORIES)[-2],
                size=0.05,
                num=self.agent_num - (self.agent_num // 2),
                keepout=self.casualty_keepout,
            ),
            Casualtys(
                num=self.agent_num // 2,
                category=list(Casualtys.CATEGORIES)[-1],
                size=0.05,
            ),
        )

        self._add_mocaps(
            Gremlins(num=config['agent_num'], size=0.15, dist_threshold=0.15, keepout=0.0),
        )

    def calculate_reward(self):
        return {f'agent_{i}': 0.0 for i in range(self.agent_num)}

    def _build(self):
        self._replace_border_buildings()
        self._replace_geom(Casualtys(
            category=list(Casualtys.CATEGORIES)[-1],
            size=0.05,
            num=self.agent_num // 2,
            keepout=0.0,
        ))
        self._replace_building_perimeter_walls()
        return super()._build()
