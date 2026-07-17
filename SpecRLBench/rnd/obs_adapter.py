"""Observation adapters for RND input tensors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch as th
from gymnasium import spaces


class RNDObsAdapter:
    """Convert Box or Dict observations into a flat float32 batch for RND.

    For Dict spaces (priority order):
      1. ``obs_keys`` — explicit multi-key list
      2. ``obs_key`` — single key
      3. otherwise flatten and concatenate all Box keys (sorted)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        obs_key: str | None = None,
        obs_keys: Sequence[str] | None = None,
    ) -> None:
        self.observation_space = observation_space
        self.obs_key = obs_key
        self.obs_keys = list(obs_keys) if obs_keys else None
        self.keys: list[str] = []
        self.input_dim = self._infer_input_dim()

    def _infer_input_dim(self) -> int:
        space = self.observation_space
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, spaces.Dict):
            if self.obs_keys is not None:
                missing = [k for k in self.obs_keys if k not in space.spaces]
                if missing:
                    raise KeyError(
                        f"rnd obs_keys missing from observation_space: {missing}. "
                        f"available={list(space.spaces.keys())}"
                    )
                for k in self.obs_keys:
                    if not isinstance(space.spaces[k], spaces.Box):
                        raise TypeError(f"RND obs key {k!r} must be Box, got {type(space.spaces[k])}")
                self.keys = list(self.obs_keys)
                return int(sum(np.prod(space.spaces[k].shape) for k in self.keys))
            if self.obs_key is not None:
                if self.obs_key not in space.spaces:
                    raise KeyError(
                        f"rnd_obs_key={self.obs_key!r} not in observation_space "
                        f"keys={list(space.spaces.keys())}"
                    )
                sub = space.spaces[self.obs_key]
                if not isinstance(sub, spaces.Box):
                    raise TypeError(f"RND obs_key={self.obs_key!r} must be Box, got {type(sub)}")
                self.keys = [self.obs_key]
                return int(np.prod(sub.shape))
            self.keys = sorted(
                k for k, v in space.spaces.items() if isinstance(v, spaces.Box)
            )
            if not self.keys:
                raise TypeError("Dict observation_space has no Box subspaces for RND")
            return int(sum(np.prod(space.spaces[k].shape) for k in self.keys))
        raise TypeError(
            f"RNDObsAdapter supports Box and Dict spaces, got {type(space)}"
        )

    def to_numpy(self, obs: Any) -> np.ndarray:
        """Return float32 array of shape ``(n_envs, input_dim)``."""
        if isinstance(self.observation_space, spaces.Box):
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == len(self.observation_space.shape):
                arr = arr[None, ...]
            return arr.reshape(arr.shape[0], -1)

        assert isinstance(obs, dict)
        parts: list[np.ndarray] = []
        for key in self.keys:
            arr = np.asarray(obs[key], dtype=np.float32)
            # VecEnv: (n_envs, *shape) or single env (*shape)
            sub_shape = self.observation_space.spaces[key].shape  # type: ignore[index]
            if arr.ndim == len(sub_shape):
                arr = arr[None, ...]
            parts.append(arr.reshape(arr.shape[0], -1))
        return np.concatenate(parts, axis=1)

    def to_torch(self, obs: Any, device: th.device | str) -> th.Tensor:
        return th.as_tensor(self.to_numpy(obs), device=device, dtype=th.float32)


def resolve_rnd_obs_keys(
    observation_space: spaces.Space,
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str] = ("casualty", "casualtys"),
) -> list[str]:
    """Pick Dict Box keys matching any include substring, excluding casualty-like keys.

    Matching is case-insensitive substring on the key name.
    """
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError(
            f"resolve_rnd_obs_keys expects Dict observation_space, got {type(observation_space)}"
        )
    include = [s.lower() for s in include_substrings]
    exclude = [s.lower() for s in exclude_substrings]
    keys: list[str] = []
    for key, sub in observation_space.spaces.items():
        if not isinstance(sub, spaces.Box):
            continue
        k = key.lower()
        if any(ex in k for ex in exclude):
            continue
        if any(inc in k for inc in include):
            keys.append(key)
    keys = sorted(keys)
    if not keys:
        raise ValueError(
            f"No RND obs keys matched include={list(include_substrings)} "
            f"exclude={list(exclude_substrings)}. "
            f"available={list(observation_space.spaces.keys())}"
        )
    return keys
