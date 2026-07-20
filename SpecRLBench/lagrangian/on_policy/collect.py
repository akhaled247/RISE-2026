"""Episode cost tracker shared by on-policy collect and SAC step hooks."""

from __future__ import annotations

import numpy as np


class EpCostTracker:
    """Accumulate per-env episode costs; report mean of finished eps in a window."""

    def __init__(self) -> None:
        self._ep_cost = np.zeros(1, dtype=np.float64)
        self._finished: list[float] = []
        self.last_ep_cost_mean = 0.0

    def reset_envs(self, n_envs: int) -> None:
        self._ep_cost = np.zeros(n_envs, dtype=np.float64)

    def begin_window(self) -> None:
        """Clear finished-ep list (start of on-policy rollout or after dual consume)."""
        self._finished = []

    def on_step(self, costs: np.ndarray, dones: np.ndarray) -> None:
        """Add step costs; on done, record finished ep cost and reset that env."""
        costs_f = np.asarray(costs, dtype=np.float64).reshape(-1)
        dones_b = np.asarray(dones).reshape(-1)
        if costs_f.shape[0] != self._ep_cost.shape[0]:
            self.reset_envs(int(costs_f.shape[0]))
        self._ep_cost += costs_f
        for idx, done in enumerate(dones_b):
            if done:
                self._finished.append(float(self._ep_cost[idx]))
                self._ep_cost[idx] = 0.0

    def finalize_mean(self) -> float:
        """Mean of finished eps in current window (0 if none); keep list."""
        self.last_ep_cost_mean = float(np.mean(self._finished)) if self._finished else 0.0
        return self.last_ep_cost_mean

    def consume_mean(self) -> float:
        """Finalize mean then clear finished list (SAC dual window)."""
        mean = self.finalize_mean()
        self._finished = []
        return mean
