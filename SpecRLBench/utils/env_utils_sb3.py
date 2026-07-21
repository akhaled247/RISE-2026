"""SB3-only env helpers (vectorize + throughput callback).

Import this module only from SB3 train/resume paths. SafePO must use
``utils.env_utils.make_env`` instead — it must not pull ``stable_baselines3``.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from utils.env_utils import make_env


class ThroughputCallback(BaseCallback):
    """Log real rollout FPS and ETA (SB3 progress bar rate can lie early on)."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._last_time = None
        self._last_steps = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            dsteps = self.num_timesteps - self._last_steps
            rollout_fps = dsteps / dt if dt > 0 else 0.0
            remaining = max(self.total_timesteps - self.num_timesteps, 0)
            eta_min = (remaining / rollout_fps / 60.0) if rollout_fps > 0 else None
            ep_len_mean = None
            if len(self.model.ep_info_buffer) > 0:
                ep_len_mean = float(np.mean([e["l"] for e in self.model.ep_info_buffer]))
            print(
                f"[throughput] steps={self.num_timesteps} "
                f"fps={rollout_fps:.1f} "
                f"eta_min={eta_min:.1f} "
                f"ep_len={ep_len_mean}"
            )
        self._last_time = now
        self._last_steps = self.num_timesteps


def make_vec(
    env_name,
    n_envs,
    render_mode=None,
    sb3=False,
    normalize=True,
    parallel=True,
    vec_env_kwargs=None,
):
    """Vectorized env factory. Uses SubprocVecEnv on Linux with fork for real parallelism."""
    vec_env_kwargs = dict(vec_env_kwargs or {})
    if parallel and n_envs > 1:
        vec_env_cls = SubprocVecEnv
        if sys.platform != "win32" and "start_method" not in vec_env_kwargs:
            vec_env_kwargs["start_method"] = "fork"
    else:
        vec_env_cls = DummyVecEnv
        vec_env_kwargs = {}
    vec_env = make_vec_env(
        lambda: Monitor(make_env(env_name, render_mode, sb3)),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs if vec_env_kwargs else None,
    )
    if sb3 and normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    return vec_env
