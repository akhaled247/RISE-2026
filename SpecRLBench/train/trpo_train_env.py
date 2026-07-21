import sys
from datetime import datetime
from pathlib import Path

import torch
from sb3_contrib import TRPO
from stable_baselines3.common.logger import configure

# Allow `python train/ppo_train_env.py` from SpecRLBench root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from utils.env_utils import ThroughputCallback, make_vec
from ppo_load_env import eval_model

# --- edit these before each run ---
env_names = ["PointLTL6MASAR1WC-v0"]
envs_timesteps = [5_000_000]
env_name = "PointLTL6MASAR1WC-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/{env_name}_tensorboard/"

def train(
        total_timesteps = 2_500_000,
        e_name = env_name,
        seed = 0,
        n_envs = 8,
        learning_rate = 5e-5,
        n_steps = 2048,
        batch_size = 256,
        target_kl = 0.02, #0.08
        gamma=0.995,
        gae_lambda=0.98,
        startup_log = True
) -> tuple[str, str]:
    TRAINING_LOG_PATH = f"./_training_logs/{e_name}_tensorboard/"
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    tb_log_name = (
            f"TRPO_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps/1_000_000:.1f}M"
            f"_lr{learning_rate}"
            f"_kl{target_kl}"
            f"_s{seed}"
        )
    log_dir = f"{TRAINING_LOG_PATH}{tb_log_name}"
    

    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print(f"<<<{rollout_steps/batch_size}>>> minibatches per rollout"
            f"\n <<<{total_timesteps//rollout_steps}>>> policy updates total")
        print("=" * 40)
        print(f"train env={e_name} device={device} steps={total_timesteps}")
        print(
            f"TRPO iter = {rollout_steps} env steps collect "
            f"{rollout_steps // batch_size} minibatches "
            f"- SB3 iters/s scales ~1/n_steps"
        )

    env = make_vec(e_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log: print("Warming up vector envs...")
    env.seed(seed=0) #Constants env seed to reduce variation between master seeds
    env.reset()

    model = TRPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        target_kl=target_kl,
        device=device,
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
        )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    env.seed(seed=0) #Constants env seed to reduce variation between master seeds
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        # callback=ThroughputCallback(total_timesteps),
        tb_log_name=tb_log_name,
    )

    model_path=f"_models/trpo_{name_time}_{e_name}_{seed}"
    vec_norm_path=f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    # print(f"saved vecnorm: {vec_norm_path}")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    for i, _ in enumerate(env_names):
        print(envs_timesteps[i])
        model_path, vec_norm_path = train(
            seed=int(i), #Tested up to and including env 3 at home
            startup_log=True,
            e_name=env_names[i],
            total_timesteps=envs_timesteps[i])
        eval_model(
            env_name=env_names[i],
            render_mode=None,
            m_path=model_path,
        )

