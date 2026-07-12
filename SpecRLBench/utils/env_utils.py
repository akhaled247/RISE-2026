import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


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


def make_vec(env_name, n_envs, render_mode=None, sb3=False, normalize=True, parallel=True):
    vec_env_cls = SubprocVecEnv if parallel and n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(
        lambda: Monitor(make_env(env_name, render_mode, sb3)),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )
    if sb3 and normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    return vec_env
