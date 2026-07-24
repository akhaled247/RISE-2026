"""Env constructors shared by SafePO CMDP factory."""

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
