"""Continue training a saved PPO checkpoint, then eval (like ppo_train_env.py).

Edit the constants below before each run. Loads ``{MODEL_PATH}.zip`` and
``{MODEL_PATH}_vecnormalize.pkl``. ``ADDITIONAL_TIMESTEPS`` is additive on top of
the loaded ``model.num_timesteps``. Uses ``reset_num_timesteps=False`` so
TensorBoard and LR schedules continue from the checkpoint step count.
"""
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from load_env import eval_model
from utils.env_utils import make_vec

# --- edit these before each run ---
env_name = "PointLTL5MASAR1-v0"
MODEL_PATH = "_models/ppo_20260716_1342_PointLTL5MASAR1-v0_0"
ADDITIONAL_TIMESTEPS = 2_000_000
n_envs = 8
seed = 0
name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"


def resume_train(
    model_path: str = MODEL_PATH,
    additional_timesteps: int = ADDITIONAL_TIMESTEPS,
    run_seed: int = 0,
    n_envs: int = 8,
    startup_log: bool = True,
) -> tuple[str, str]:
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    vec_norm_path = f"{model_path}_vecnormalize.pkl"

    if not Path(f"{model_path}.zip").exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}.zip")
    if not Path(vec_norm_path).exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vec_norm_path}\n"
            "Resume requires the pickle saved alongside the model.",
        )

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print("=" * 40)
        print(f"resume env={env_name} device={device}")
        print(f"loading model: {model_path}.zip")
        print(f"loading vecnorm: {vec_norm_path}")
        print(f"additional timesteps: {additional_timesteps:,}")

    vec_env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = True
    vec_env.norm_reward = False

    vec_env.seed(seed=0)
    vec_env.reset()

    model = PPO.load(model_path, env=vec_env, device=device)

    start_timesteps = model.num_timesteps
    if startup_log:
        print(f"num_timesteps before learn: {start_timesteps:,}")

    vec_env.seed(seed=0)
    model.learn(
        total_timesteps=additional_timesteps,
        reset_num_timesteps=False,
        log_interval=1,
        progress_bar=True,
        tb_log_name=f"PPO_t{name_time}_resume_from_{start_timesteps}",
    )

    end_timesteps = model.num_timesteps
    if startup_log:
        print(f"num_timesteps after learn:  {end_timesteps:,} (+{end_timesteps - start_timesteps:,})")

    out_path = f"_models/ppo_{name_time}_{env_name}_{run_seed}_resumed"
    vec_out_path = f"{out_path}_vecnormalize.pkl"
    model.save(out_path)
    vec_env.save(vec_out_path)
    print(f"saved model: {out_path}.zip")
    vec_env.close()
    return out_path, vec_out_path


if __name__ == "__main__":
    model_path, _ = resume_train(run_seed=seed, startup_log=True)
    eval_model(
        env_name=env_name,
        render_mode=None,
        m_path=model_path,
    )
