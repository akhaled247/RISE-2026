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
from datetime import datetime

# --- must match the train run ---
env_name = "PointLTL4MASAR1-v0"
run_num = 9
name_time = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f"_models/ppo_{name_time}_{env_name}_run{run_num}"
VEC_NORM_PATH = f"{MODEL_PATH}_vecnormalize.pkl"
eval_episodes = 50
s = 0
render_mode='human'

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
    seed: int = s,
    deterministic: bool = True,
    m_path: str = None
):
    model_path = m_path if m_path is not None else MODEL_PATH
    vec_norm_path = f"{model_path}_vecnormalize.pkl"

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    totals_steps = []
    casualty_visible_step_0s = []
    rescue_count = 0
    rescues = []

    for episode in range(eval_episodes):
        vec_env.seed(seed=seed)
        obs = vec_env.reset()
        episode_reward = 0.0
        total_steps = 0
        rescued = False
        casualty_visible_step_0 = False
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)

            prop_keys = info[0].get("propositions", [])
            casualty_visible_step_0 = (
                (total_steps == 0) 
                * (info[0].get("casualty_visible", False)) 
                + casualty_visible_step_0)
            
            if any("cost_casualtys_surface" in k for k in prop_keys):
                rescued = True

            episode_reward += float(reward[0])
            total_steps += 1
            done = bool(done[0])

        episode_rewards.append(episode_reward)
        totals_steps.append(total_steps)
        casualty_visible_step_0s.append(casualty_visible_step_0)
        rescues.append(int(rescued))
        seed+=1
        if rescued:
            rescue_count += 1
        print(
            f"Episode {episode + 1}: reward={episode_reward:.3f} "
            f"rescued={rescued} "
            f"total_steps={total_steps} "
            f"casualty_visible_step_0={casualty_visible_step_0} "
        )

    vec_env.close()

    visible_step_0_rescues = [
        rescues[i]
        for i, v in enumerate(casualty_visible_step_0s)
        if v == 1
        ]
    invisible_step_0_rescues = [
        rescues[i]
        for i, v in enumerate(casualty_visible_step_0s)
        if v == 0
        ]

    print("-" * 40)
    print(f"Mean reward:        {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"Rescue rate:        {rescue_count}/{eval_episodes} "
          f"({100 * rescue_count / eval_episodes:.1f}%)")
    print(f"s0-Vis rescue %:    {sum(visible_step_0_rescues)}/{len(visible_step_0_rescues)} "
    f"({100 * sum(visible_step_0_rescues) / len(visible_step_0_rescues):.1f}%)"
    )
    print(f"s0-Invis rescue %:  {sum(invisible_step_0_rescues)}/{len(invisible_step_0_rescues)} "
        f"({100 * sum(invisible_step_0_rescues) / len(invisible_step_0_rescues):.1f}%)"
        )
    print(f"Mean ep_len:        {np.mean(totals_steps):.3f} +/- {np.std(totals_steps):.3f} ")
    print(f"Mean reward:        {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "rescue_rate": rescue_count / eval_episodes,
    }


if __name__ == "__main__":
    eval_model(
        env_name=env_name,
        run_num=run_num,
        render_mode=render_mode,
        eval_episodes=eval_episodes,
        m_path="_models/ppo_20260714_1623_PointLTL4MASAR1-v0_run9"
    )
