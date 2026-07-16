"""Observation adapters for RND input tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces


class RNDObsAdapter:
    """Convert Box or Dict observations into a flat float32 batch for RND.

    For Dict spaces:
      - if ``obs_key`` is set, use that key only
      - otherwise flatten and concatenate all Box keys (sorted) along feature dim
    """

    def __init__(self, observation_space: spaces.Space, obs_key: str | None = None) -> None:
        self.observation_space = observation_space
        self.obs_key = obs_key
        self.keys: list[str] = []
        self.input_dim = self._infer_input_dim()

    def _infer_input_dim(self) -> int:
        space = self.observation_space
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, spaces.Dict):
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
