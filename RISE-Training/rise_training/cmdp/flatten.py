"""Deterministic Dict → Box flatten (sorted keys) for SafePO ActorVCritic."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


class DictFlattenWrapper(gymnasium.Wrapper):
    """Flatten ``Dict`` obs to 1-D ``Box`` with stable sorted key order.

    Supports Gymnasium 5-tuple and Safety 6-tuple ``step`` returns.
    Expects flat Dict of arrays (``flat=True`` SpecRLBench wrappers).
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Dict):
            if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
                self._keys: list[str] = []
                self.observation_space = obs_space
                return
            raise TypeError(
                f"DictFlattenWrapper expects Dict or 1-D Box, got {type(obs_space)}"
            )
        self._keys = sorted(obs_space.spaces.keys())
        for k in self._keys:
            sp = obs_space.spaces[k]
            if not isinstance(sp, spaces.Box):
                raise TypeError(f"Key {k!r} must be Box, got {type(sp)}")
        low = np.concatenate(
            [np.ravel(obs_space.spaces[k].low).astype(np.float32) for k in self._keys]
        )
        high = np.concatenate(
            [np.ravel(obs_space.spaces[k].high).astype(np.float32) for k in self._keys]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def flatten_keys(self) -> list[str]:
        return list(self._keys)

    def _flat(self, observation: Any) -> np.ndarray:
        if not self._keys:
            return np.asarray(observation, dtype=np.float32).ravel()
        parts = [np.ravel(np.asarray(observation[k], dtype=np.float32)) for k in self._keys]
        return np.concatenate(parts, axis=0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flat(obs), info

    def step(self, action: ActType):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._flat(obs), reward, terminated, truncated, info
        obs, reward, cost, terminated, truncated, info = result
        if "final_observation" in info and info["final_observation"] is not None:
            info = dict(info)
            fo = info["final_observation"]
            if isinstance(fo, dict):
                info["final_observation"] = self._flat(fo)
        return self._flat(obs), reward, cost, terminated, truncated, info


class AutoResetSafetyWrapper(gymnasium.Wrapper):
    """Auto-reset on terminated/truncated; stash final obs like SafePO expects."""

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        done = bool(terminated) or bool(truncated)
        if done:
            info = dict(info)
            info["final_observation"] = obs
            if truncated and not terminated:
                info["TimeLimit.truncated"] = True
            obs, reset_info = self.env.reset()
            info["reset_info"] = reset_info
        return obs, reward, cost, terminated, truncated, info
