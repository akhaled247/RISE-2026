import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env

# --- must match the train run ---
env_name = "PointLTL4MASAR1-v0"
run_num = 5
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
VEC_NORM_PATH = f"{MODEL_PATH}_vecnormalize.pkl"
eval_episodes = 20
seed = 0


def _get_task(vec_env):
    base = vec_env.venv.envs[0]
    while hasattr(base, "env"):
        base = base.env
    return base.unwrapped.task


def eval_model(
    env_name: str = env_name,
    run_num: int = run_num,
    render_mode=None,
    eval_episodes: int = eval_episodes,
    seed: int = seed,
    deterministic: bool = True,
):
    model_path = f"_models/ppo_{env_name}_run{run_num}"
    vec_norm_path = f"{model_path}_vecnormalize.pkl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"eval env={env_name} device={device}")
    print(f"loading {model_path}.zip")

    base_env = make_env(env_name, sb3=True, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: Monitor(base_env)])
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=device)
    task = _get_task(vec_env)

    episode_rewards = []
    rescue_count = 0
    visible_step_frac = []

    for episode in range(eval_episodes):
        obs = vec_env.reset()
        episode_reward = 0.0
        visible_steps = 0
        total_steps = 0
        rescued = False
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += float(reward[0])
            total_steps += 1

            if getattr(task, "_casualty_visible_sticky", None):
                if task._casualty_visible_sticky[0]:
                    visible_steps += 1

            prop_keys = info[0].get("propositions", [])
            if any("cost_casualtys_surface" in k for k in prop_keys):
                rescued = True

            done = bool(done[0])

        episode_rewards.append(episode_reward)
        if rescued:
            rescue_count += 1
        frac = visible_steps / max(total_steps, 1)
        visible_step_frac.append(frac)
        print(
            f"Episode {episode + 1}: reward={episode_reward:.3f} "
            f"rescued={rescued} visible_frac={frac:.2%}"
        )

    vec_env.close()

    print("-" * 40)
    print(f"Mean reward:      {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"Rescue rate:      {rescue_count}/{eval_episodes} "
          f"({100 * rescue_count / eval_episodes:.1f}%)")
    print(f"Mean visible %:   {100 * np.mean(visible_step_frac):.1f}%")
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "rescue_rate": rescue_count / eval_episodes,
        "mean_visible_frac": float(np.mean(visible_step_frac)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval PPO SAR model")
    parser.add_argument("--env", default=env_name)
    parser.add_argument("--run-num", type=int, default=run_num)
    parser.add_argument("--episodes", type=int, default=eval_episodes)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    eval_model(
        env_name=args.env,
        run_num=args.run_num,
        render_mode="human" if args.render else None,
        eval_episodes=args.episodes,
    )
