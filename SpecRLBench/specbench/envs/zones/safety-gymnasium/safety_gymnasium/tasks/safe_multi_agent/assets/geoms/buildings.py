# Copyright 2024 anonymous-elephant. All Rights Reserved.
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

from dataclasses import field

import numpy as np
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


class Buildings(Geom):  # pylint: disable=too-many-instance-attributes
    """Colored buildings."""

    COLORS = {
        "terracotta": np.array([226, 125, 91, 255])/255,
        "tan": np.array([0.8, 0.75, 0.6, 1.0]),
        "green": np.array([0, 1, 0, 1]),
        "red": np.array([1, 0, 0, 1]),
        "yellow": np.array([1, 1, 0, 1]),
        "light_gray": np.array([0.75, 0.75, 0.75, 1.0]),
    }

    def __init__(self, color: str, size: float, num: int, locations=None, placements=None, keepout=0.55, debug: bool=False, rots=None):
        self.color_name = color
        self.name = f'{color}_buildings'
        self.num = num 
        self.size: float = size
        self.placements: list = placements  # Placements list for hazards (defaults to full extents)
        self.locations: list = locations if locations else []  # Fixed locations to override placements
        self.rots: list = rots
        self.keepout: float = keepout  # Radius of hazard keepout for placement
        self.alpha: float = 1e-2 if not debug else 0.25
        
        # if self.color_name not in self.COLORS:
        #     self.color = self.COLORS['black']
        # else:
        self.color: np.array = self.COLORS[self.color_name]
        self.group: int = GROUP['wall']
        self.is_lidar_observed: bool = True
        self.is_constrained: bool = True
        self.is_meshed: bool = False

    def process_config(self, config, layout, rots):
        return super().process_config(config, layout, np.zeros(self.num) if self.rots is None else self.rots )
    
    def calculate_group(self) -> int:
        # return GROUP['goal']
        max_predefined_group = max(GROUP.values())
        return max_predefined_group + sorted(self.COLORS.keys()).index(self.color_name) + 1

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        # Return a flat geom config (single geom), compatible with World.build
        geom = {
            'name': 'self.name',
            'pos': np.r_[xy_pos, self.size*3],
            'rot': rot,
            'size': [self.size, self.size, self.size*3],
            'type': 'box',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * np.array([1.0, 1.0, 1.0, self.alpha]),
        }
        if self.is_meshed:
            geom.update(
                {
                    'type': 'mesh',
                    'mesh': 'bush',
                    'material': 'bush',
                    'euler': [np.pi / 2, 0, 0],
                },
            )
        return geom

    def cal_cost(self):
        # cost = {f'cost_buildings_{self.color}': 0}
        # cost = {'agent_0': {f'cost_buildings_{self.color}': 0}, 
        #         'agent_1': {f'cost_buildings_{self.color}': 0}}
        cost = {agent: {f'cost_buildings_{self.color_name}': 0} for agent in self.agent.possible_agents}
        # print(f"self.pos: {self.pos}")
        for h_pos in self.pos:
            for i in range(self.agent.agent_num):
                agent_h_dist = self.agent.dist_xy(i, h_pos)
                if agent_h_dist <= self.size:
                    cost[f'agent_{i}'][f'cost_buildings_{self.color_name}'] = 1.0
            # agent0_h_dist = self.agent.dist_xy(0, h_pos)
            # agent1_h_dist = self.agent.dist_xy(1, h_pos)
            # if agent0_h_dist <= self.size:
            #     cost['agent_0'][f'cost_buildings_{self.color}'] = 1
            # if agent1_h_dist <= self.size:
            #     cost['agent_1'][f'cost_buildings_{self.color}'] = 1
        
        return cost

    @property
    def pos(self):
        """Helper to get the hazards positions from layout."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
