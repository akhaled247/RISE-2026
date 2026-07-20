import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import trange

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env


# =============================================================================
# Configuration
# =============================================================================

ENV_NAME = "PointLTL5MASAR1WC-v0"

MODELS = [
    "_models/ppo_lag_L5_S0_20260719_1510_PointLTL5MASAR1WC-v0_0",
]

EVAL_EPISODES = 50
SEED = 0

RENDER_MODE = 'human'
DETERMINISTIC = True

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

NAME_TIME = datetime.now().strftime("%Y%m%d_%H%M")
DEFAULT_MODEL_PATH = f"_models/ppo_{NAME_TIME}_{ENV_NAME}"


def eval_model(
    env_name: str = ENV_NAME,
    render_mode=RENDER_MODE,
    eval_episodes: int = EVAL_EPISODES,
    seed: int = SEED,
    deterministic: bool = DETERMINISTIC,
    m_path: str | None = None,
):
    model_path = m_path if m_path is not None else DEFAULT_MODEL_PATH
    vec_norm_path = f"{model_path}_vecnormalize.pkl"

    print("=" * 40)
    print(f"eval env={env_name} device={DEVICE} render_mode={render_mode}")
    print(f"loading {model_path}")

    base_env = make_env(env_name, sb3=True, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: Monitor(base_env)])
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Route by filename tag.
    name_l = Path(model_path).name.lower()
    if "ppo_lag" in name_l or "ppolagrangian" in name_l:
        from ppo_lagrangian import PPOLagrangian

        # SB3 Tier-2: .zip only (legacy .pt raises ValueError)
        model = PPOLagrangian.load(model_path, env=vec_env, device=DEVICE)
    elif "rnd_ppo" in name_l:
        from rnd import RNDPPO

        model = RNDPPO.load(model_path, env=vec_env, device=DEVICE)
    else:
        model = PPO.load(model_path, env=vec_env, device=DEVICE)

    episode_rewards = []
    totals_steps = []
    casualty_visible_step_0s = []
    rescue_count = 0
    rescues = []

    for episode in trange(eval_episodes):
        vec_env.seed(seed)
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
                * info[0].get("casualty_visible", False)
                + casualty_visible_step_0
            )

            if any("cost_casualtys_surface" in k for k in prop_keys):
                rescued = True
            if any("cost_casualtys_entrapped" in k for k in prop_keys):
                rescued = True

            episode_reward += float(reward[0])
            total_steps += 1
            done = bool(done[0])

        episode_rewards.append(episode_reward)
        totals_steps.append(total_steps)
        casualty_visible_step_0s.append(casualty_visible_step_0)
        rescues.append(int(rescued))

        seed += 1
        if rescued:
            rescue_count += 1

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

    print("=" * 40)
    print(
        f"Mean reward:        {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}"
    )
    print(
        f"Rescue rate:        {rescue_count}/{eval_episodes} "
        f"({100 * rescue_count / eval_episodes:.1f}%)"
    )
    # print(f"s0-Vis rescue %:    {sum(visible_step_0_rescues)}/{len(visible_step_0_rescues)} "
    #       f"({100 * sum(visible_step_0_rescues) / len(visible_step_0_rescues):.1f}%)")
    # print(f"s0-Invis rescue %:  {sum(invisible_step_0_rescues)}/{len(invisible_step_0_rescues)} "
    #       f"({100 * sum(invisible_step_0_rescues) / len(invisible_step_0_rescues):.1f}%)")
    print(
        f"Mean ep_len:        {np.mean(totals_steps):.3f} +/- {np.std(totals_steps):.3f}"
    )

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "rescue_rate": rescue_count / eval_episodes,
    }


if __name__ == "__main__":
    for model in MODELS:
        eval_model(m_path=model)