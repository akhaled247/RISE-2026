"""Utility helpers matching OpenAI safety-starter-agents ``safe_rl/pg/utils.py``."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.signal

EPS = 1e-8


def combined_shape(length: int, shape: int | tuple[int, ...] | None = None) -> tuple[int, ...]:
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(d: dict[str, Any]) -> list[str]:
    return sorted(list(d.keys()))


def values_as_sorted_list(d: dict[str, Any]) -> list[Any]:
    return [d[k] for k in keys_as_sorted_list(d)]


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Discounted cumulative sum (rllab / Spinning Up / Safety Starter Agents).

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
