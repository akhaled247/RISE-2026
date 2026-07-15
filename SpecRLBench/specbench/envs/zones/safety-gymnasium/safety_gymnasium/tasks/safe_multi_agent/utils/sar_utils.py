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
"""Utils for wall placement on rings and arcs."""

from __future__ import annotations

import json
import os
import time

import numpy as np
from typing import TYPE_CHECKING


def _debug_log34211f(location, message, data, hypothesis_id, run_id='pre-fix'):
    # #region agent log
    root = os.path.abspath(os.getcwd())
    for _ in range(15):
        if os.path.isdir(os.path.join(root, '.git')):
            break
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    path = os.path.join(root, 'debug-34211f.log')
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': '34211f',
                'runId': run_id,
                'hypothesisId': hypothesis_id,
                'location': location,
                'message': message,
                'data': data,
                'timestamp': int(time.time() * 1000),
            }) + '\n')
    except OSError:
        pass
    # #endregion



def ring_locations(radius: float, n: int) -> list[tuple[float, float]]:
    """Fixed (x, y) centers evenly spaced on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(radius * np.cos(theta)), float(radius * np.sin(theta))) for theta in angles]

def ring_placements(
    radius: float,
    n: int,
    margin: float | None = None,
    keepout: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Axis-aligned sampling boxes centered on ring points (for random placement).

    Sampler shrinks each box by ``keepout`` on all sides; need ``margin > keepout``
  or ``draw_placement`` asserts with no valid rectangles.
    """
    if margin is None:
        margin = keepout + 0.05
    margin = max(margin, keepout + 1e-3)
    boxes = []
    for x, y in ring_locations(radius, n):
        boxes.append((x - margin, y - margin, x + margin, y + margin))
    return boxes

def border_placements(side_length, margin):
    """
    Generates 4 non-overlapping border boxes around a central square of side length N.
    The center of the system is at (0, 0).
    Output format: (x_min, y_min, x_max, y_max)
    """
    # Inner boundaries (the edges of the central square)
    half_n = side_length / 2.0
    outer = half_n + margin

    boxes = [
        (-outer,  half_n,  outer,  outer),        
        ( half_n, -half_n,  outer,  half_n),
        (-outer, -outer,  outer, -half_n),
        (-outer, -half_n, -half_n,  half_n)
    ]
    return boxes

def border_placement_keepout(margin: float, keepout: float) -> float:
    """Clamp keepout so border strips (thickness ``margin``) stay sampleable."""
    return min(keepout, margin / 2.0 - 1e-3)

def size_randomization(
    base_half_sizes: list,
    n: int,
    margin: float | None = None,
    margins: list[float] = [0, 0, 0],
    random_generator: RandomGenerator | None = None,
) -> list[list[float, float, float]]:
    """Axis-aligned sampling boxes centered on ring points (for random placement).
    Sampler shrinks each box by ``keepout`` on all sides; need ``margin > keepout``
  or ``draw_placement`` asserts with no valid rectangles.
    """
    if margin != None: margins = [margin, margin, margin] if margins == [0, 0, 0] else margins
    x, y, z = base_half_sizes
    x_margin, y_margin, z_margin = margins

    assert any(np.array(base_half_sizes)-np.array(margins) > 0), "Margins should ensure non-negative values" 
    
    return np.array([random_generator.uniform(
        x-x_margin, x+x_margin, n),
        random_generator.uniform(
        y-y_margin, y+y_margin, n),
        random_generator.uniform(
        z-z_margin, z+z_margin, n)]).transpose()

if TYPE_CHECKING:
    from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
    from safety_gymnasium.tasks.safe_multi_agent.utils.random_generator import RandomGenerator


def is_building_ltl_wall(name: str) -> bool:
    """True for per-building perimeter walls, not the arena ``ltl_walls``."""
    return name.startswith('building') and name.endswith('_ltl_walls')


_CASUALTY_GEOM_NAMES = ('surface_casualtys', 'entrapped_casualtys')


def all_casualties_rescued(task: BaseTask) -> bool:
    """Return True when every surface and entrapped casualty (if present) is rescued."""
    found_any = False
    for attr in _CASUALTY_GEOM_NAMES:
        if not hasattr(task, attr):
            continue
        geom = getattr(task, attr)
        if geom.num <= 0:
            continue
        found_any = True
        if not all(geom.rescued):
            return False
    return found_any


def mission_goal_achieved(task: BaseTask) -> tuple[bool, ...]:
    """Shared goal_achieved tuple: same team mission flag for each agent."""
    mission_complete = all_casualties_rescued(task)
    return tuple(mission_complete for _ in range(task.agent_num))

