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
"""Wall."""

from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.tasks.safe_multi_agent.assets.color import COLOR
from safety_gymnasium.tasks.safe_multi_agent.assets.group import GROUP
from safety_gymnasium.tasks.safe_multi_agent.bases.base_object import Geom


@dataclass
class Walls(Geom):  # pylint: disable=too-many-instance-attributes
    """Interior box walls (immovable). Supports per-wall size via index in get_config."""

    name: str = 'walls'
    num: int = 0
    # Single [thickness, half_length, half_height] for all walls, or one triple per wall.
    half_sizes: list = field(default_factory=lambda: [0.1, 0.3, 0.2])
    locations: list = field(default_factory=list)
    placements: list | None = None
    keepout: float = 0.25
    tangent: bool = False  # If True, wall runs tangent to ring when locations set
    random_size: bool = False

    color: np.array = COLOR['wall']
    group: np.array = GROUP['wall']
    is_lidar_observed: bool = True
    is_constrained: bool = False

    def __post_init__(self) -> None:
        self._index = 0
        if self.locations and len(self.locations) >= self.num:
            self._angles = [float(np.arctan2(y, x)) for x, y in self.locations[: self.num]]
        else:
            self._angles = [0.0] * max(self.num, 1)

    def _size_for_index(self, idx: int) -> np.ndarray:
        # print(self.half_sizes)
        if (
            self.num > 0
            and isinstance(self.half_sizes[0], (list, tuple, np.ndarray))
            and len(self.half_sizes) >= self.num
        ):
            return np.asarray(self.half_sizes[idx], dtype=float)
        return np.asarray(self.half_sizes, dtype=float)

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """Build MuJoCo box geom for the current wall index."""
        idx = self._index
        size = self._size_for_index(idx)
        wall_rot = self._angles[idx] + (np.pi / 2 if self.tangent else 0.0)
        self._index = (self._index + 1) % max(self.num, 1)
        return {
            'name': self.name,
            'size': size,
            'pos': np.r_[xy_pos, size[-1] + 1e-5],
            'rot': wall_rot+rot,
            'type': 'box',
            'contype': 1,
            'conaffinity': 1,
            'group': self.group,
            'rgba': self.color,
        }

    @property
    def pos(self):
        """Helper to get list of wall positions."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
