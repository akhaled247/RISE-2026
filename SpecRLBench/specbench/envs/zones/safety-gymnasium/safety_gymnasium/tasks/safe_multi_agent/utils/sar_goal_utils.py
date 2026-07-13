"""Shared SAR mission-complete helpers for task goal checks.

Per-agent LTL subtasks will replace the team-level rule later; IPPO and
centralized PPO can both import these helpers until then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask

_CASUALTY_GEOM_NAMES = ('surface_casualtys', 'entrapped_casualtys')


def all_casualties_rescued(task: BaseTask) -> bool:
    """Return True when every surface and entrapped casualty (if present) is rescued."""
    found_any = False
    for attr in _CASUALTY_GEOM_NAMES:
        if not hasattr(task, attr):
            continue
        geom = getattr(task, attr)
        found_any = True
        if not all(geom.rescued):
            return False
    return found_any


def mission_goal_achieved(task: BaseTask) -> tuple[bool, ...]:
    """Shared goal_achieved tuple: same team mission flag for each agent."""
    mission_complete = all_casualties_rescued(task)
    return tuple(mission_complete for _ in range(task.agent_num))
