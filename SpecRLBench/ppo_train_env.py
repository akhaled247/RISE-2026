import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import ThroughputCallback, make_vec
from ppo_load_env import eval_model

# --- curriculum: L0 open arena then L4 with interior walls ---
CURRICULUM = [
    {
        "env_name": "PointLTL0MASAR1-v0",
        "total_timesteps": 300_000,
        "run_num": 10,
        "learning_rate": 3e-4,
        "n_steps": 512,
        "n_epochs": 4,
        "target_kl": 0.05,
    },
    {
        "env_name": "PointLTL4MASAR1-v0",
        "total_timesteps": 700_000,
        "run_num": 5,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "n_epochs": 10,
        "target_kl": 0.05,
    },
]

seed = 0
n_envs = 8
ent_coef = 0.01


def train_stage(stage: dict) -> tuple[str, str]:
    env_name = stage["env_name"]
    run_num = stage["run_num"]
    total_timesteps = stage["total_timesteps"]
    model_path = f"_models/ppo_{env_name}_run{run_num}"
    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    log_path = f"./_training_logs/ppo_{env_name}_tensorboard/"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"train env={env_name} device={device} steps={total_timesteps}")
    # #region agent log
    try:
        from debug.debug_log import agent_log
        agent_log(
            "ppo_train_env.py:train_stage",
            "train_start",
            {
                "env_name": env_name,
                "device": device,
                "n_envs": n_envs,
                "ent_coef": ent_coef,
                "n_steps": stage["n_steps"],
                "n_epochs": stage["n_epochs"],
                "total_timesteps": total_timesteps,
                "run_num": run_num,
            },
            "H4",
            "train",
        )
    except Exception:
        pass
    # #endregion

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    print("Warming up vector envs...")
    env.reset()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=stage["learning_rate"],
        n_steps=stage["n_steps"],
        batch_size=256,
        n_epochs=stage["n_epochs"],
        ent_coef=ent_coef,
        target_kl=stage["target_kl"],
        device=device,
        tensorboard_log=log_path,
        seed=seed,
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=ThroughputCallback(total_timesteps),
    )
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    print(f"saved vecnorm: {vec_norm_path}")
    env.close()
    return model_path, vec_norm_path


def train_curriculum():
    for i, stage in enumerate(CURRICULUM):
        print(f"\n>>> Curriculum stage {i + 1}/{len(CURRICULUM)}: {stage['env_name']}")
        train_stage(stage)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO SAR curriculum training")
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Run single curriculum stage index (0=L0, 1=L4); default runs full curriculum",
    )
    parser.add_argument("--eval", action="store_true", help="Eval L4 model after training")
    args = parser.parse_args()

    if args.stage is not None:
        train_stage(CURRICULUM[args.stage])
    else:
        train_curriculum()

    if args.eval:
        eval_model(
            env_name=CURRICULUM[-1]["env_name"],
            run_num=CURRICULUM[-1]["run_num"],
            render_mode=None,
        )
