"""Env constructors shared by SafePO and SB3 paths.

``make_env`` has no ``stable_baselines3`` dependency so SafePO CMDP factory
can import it in an SB3-free venv. SB3 helpers live in ``env_utils_sb3``.
"""

from __future__ import annotations

import gymnasium as gym


def make_env(env_name, render_mode=None, sb3=False):
    if env_name.startswith("Letter"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Panda"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
        from specbench.envs.zones.safety_gym_wrapper_ma_sar import SafetyGymWrapperMASAR
        from specbench.envs.zones.safety_gym_wrapper_ma_sar_wc import SafetyGymWrapperMASARWC
        from specbench.envs.zones.safety_gym_wrapper import SafetyGymWrapper
        import safety_gymnasium
        if 'WC' in env_name:
            env = safety_gymnasium.make(env_name.replace("WC", ""), disable_env_checker=True, render_mode=render_mode)
        else:
            env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        if "SAR" in env_name:
            if 'WC' in env_name:
                env = SafetyGymWrapperMASARWC(env, sb3=sb3)
            else:
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


def __getattr__(name: str):
    """Lazy re-export SB3 helpers so old ``from utils.env_utils import make_vec`` still works."""
    if name in ("make_vec", "ThroughputCallback"):
        from utils import env_utils_sb3 as _sb3

        return getattr(_sb3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
