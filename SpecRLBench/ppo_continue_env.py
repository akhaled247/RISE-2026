import sys
from pathlib import Path
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_vec
from ppo_load_env import eval_model

# ------------------------------------------------------------------

env_name = "PointLTL5MASAR1-v0"

MODEL_PATH = "_models/ppo_20260716_0925_PointLTL5MASAR1-v0_0"
VECNORM_PATH = MODEL_PATH + "_vecnormalize.pkl"

TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"

name_time = datetime.now().strftime("%Y%m%d_%H%M")

ADDITIONAL_TIMESTEMPS = 5_000_000

# ------------------------------------------------------------------


def continue_training(
    total_timesteps=4_000_000,
    seed=0,
):

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    env = make_vec(
        env_name,
        n_envs=8,
        render_mode=None,
        sb3=True,
        normalize=True,
    )

    # Restore normalization statistics
    env = VecNormalize.load(VECNORM_PATH, env)

    env.seed(seed=0)
    env.reset()

    # Restore model
    model = PPO.load(
        MODEL_PATH,
        env=env,
        device=device,
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        log_interval=1,
        progress_bar=True,
        tb_log_name=f"PPO_t{name_time}_continue",
    )

    save_path = f"_models/ppo_{name_time}_{env_name}_{seed}_continued"

    model.save(save_path)
    env.save(save_path + "_vecnormalize.pkl")

    env.close()

    return save_path


if __name__ == "__main__":

    model_path = continue_training(
        total_timesteps=ADDITIONAL_TIMESTEMPS,
    )

    eval_model(
        env_name=env_name,
        render_mode=None,
        m_path=model_path,
    )