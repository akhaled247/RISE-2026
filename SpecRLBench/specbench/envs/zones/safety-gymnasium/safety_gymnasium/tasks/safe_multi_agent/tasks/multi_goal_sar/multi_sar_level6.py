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
"""Multi Goal SAR level 6: same as level 5 with multiple buildings."""

from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal_sar.multi_sar_level5 import (
    MultiGoalSARLevel5,
)


class MultiGoalSARLevel6(MultiGoalSARLevel5):
    """L5 setup with two buildings."""

    building_num = 2
