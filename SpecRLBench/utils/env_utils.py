import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env

def make_env(env_name, render_mode=None, sb3=False):
    if env_name.startswith("Letter"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Panda"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
        from specbench.envs.zones.safety_gym_wrapper_ma_sro import SafetyGymWrapperMASAR
        from specbench.envs.zones.safety_gym_wrapper import SafetyGymWrapper
        import safety_gymnasium
        env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        if "SAR" in env_name: env = SafetyGymWrapperMASAR(env, sb3=sb3)
        elif "MA" in env_name: env = SafetyGymWrapperMA(env)
        else: env = SafetyGymWrapper(env)
    else:
        # env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
        try:
            import safety_gymnasium
            env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        except Exception as e:
            raise ValueError(f"Unknown environment name: {env_name}")
    return env

def make_vec(env_name, n_envs, render_mode=None, sb3=False):
    return make_vec_env(
        lambda: make_env(env_name, render_mode, sb3),
        n_envs=n_envs
    )