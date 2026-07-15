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

from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level2 import MultiGoalSARLevel2


class MultiGoalSARLevel3(MultiGoalSARLevel2):
    """Multi-agent zone navigation with optional ring-placed interior walls."""

    wall_count = 20

    def calculate_reward(self):
        return {f'agent_{i}': 0.0 for i in range(self.agent_num)}

    def specific_reset(self):
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
