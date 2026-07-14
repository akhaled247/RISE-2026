import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


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
            # #region agent log
            try:
                from debug.debug_log import agent_log
                agent_log(
                    "env_utils.py:ThroughputCallback",
                    "rollout_end",
                    {
                        "num_timesteps": int(self.num_timesteps),
                        "rollout_fps": round(rollout_fps, 1),
                        "rollout_seconds": round(dt, 2),
                        "ep_len_mean": ep_len_mean,
                    },
                    "H4",
                    "train",
                )
            except Exception:
                pass
            # #endregion
        self._last_time = now
        self._last_steps = self.num_timesteps


def make_env(env_name, render_mode=None, sb3=False):
    if env_name.startswith("Letter"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Panda"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
        from specbench.envs.zones.safety_gym_wrapper_ma_sar import SafetyGymWrapperMASAR
        from specbench.envs.zones.safety_gym_wrapper import SafetyGymWrapper
        import safety_gymnasium
        env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        if "SAR" in env_name:
            env = SafetyGymWrapperMASAR(env, sb3=sb3)
        elif "MA" in env_name:
            env = SafetyGymWrapperMA(env)
        else:
            env = SafetyGymWrapper(env)
    else:
        try:
            import safety_gymnasium
            env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        except Exception:
            raise ValueError(f"Unknown environment name: {env_name}")
    return env


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
    # #region agent log
    try:
        from debug.debug_log import agent_log
        agent_log(
            "env_utils.py:make_vec",
            "vec_env_config",
            {
                "env_name": env_name,
                "n_envs": n_envs,
                "vec_env_cls": vec_env_cls.__name__,
                "vec_env_kwargs": vec_env_kwargs,
                "platform": sys.platform,
            },
            "H2",
            "train",
        )
    except Exception:
        pass
    # #endregion
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
