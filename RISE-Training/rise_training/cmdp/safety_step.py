"""Bridge Gymnasium 5-tuple (+ info['cost']) to Safety-Gymnasium 6-tuple."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ActType, ObsType


class GymnasiumToSafetyStep(gymnasium.Wrapper):
    """``step`` → ``(obs, reward, cost, terminated, truncated, info)``.

    Cost source (OpenAI / SpecRLBench WC)::

        cost = float(info.get("cost", 0))
    """

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(info.get("cost", 0.0))
        return obs, reward, cost, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info.setdefault("cost", 0.0)
        return obs, info
