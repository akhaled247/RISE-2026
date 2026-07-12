import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env, make_vec


class ThroughputCallback(BaseCallback):
    """Log real rollout FPS and ETA (progress bar rate can lie early on)."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._last_time = None
        self._last_steps = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            dsteps = self.num_timesteps - self._last_steps
            rollout_fps = dsteps / dt if dt > 0 else 0.0
            remaining = max(self.total_timesteps - self.num_timesteps, 0)
            eta_min = (remaining / rollout_fps / 60.0) if rollout_fps > 0 else None
            ep_len_mean = None
            if len(self.model.ep_info_buffer) > 0:
                ep_len_mean = float(np.mean([e["l"] for e in self.model.ep_info_buffer]))
            print(
                f"[throughput] steps={self.num_timesteps} "
                f"fps={rollout_fps:.1f} "
                f"eta_min={eta_min:.1f} "
                f"ep_len={ep_len_mean}"
            )
        self._last_time = now
        self._last_steps = self.num_timesteps


# Config
env_name = "PointLTL0MASAR1-v0"
run_num = 1  # INCREMENT EACH TIME
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
VEC_NORM_PATH = f"{MODEL_PATH}_vecnormalize.pkl"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"
TOTAL_TIMESTEPS = 500_000
seed = 0
n_envs = 8
eval_episodes = 10


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"train env={env_name} device={device}")

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    print("Warming up vector envs (one-time MuJoCo build per worker)...")
    env.reset()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        ent_coef=0.01,
        target_kl=0.02,
        device=device,
        tensorboard_log=TRAINING_LOG_PATH,
        seed=seed,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False,
        callback=ThroughputCallback(TOTAL_TIMESTEPS),
    )
    model.save(MODEL_PATH)
    env.save(VEC_NORM_PATH)
    print(f"saved model: {MODEL_PATH}.zip")
    print(f"saved vecnorm: {VEC_NORM_PATH}")

    env.close()


def eval_model(render_mode=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"eval env={env_name} device={device}")
    print(f"loading {MODEL_PATH}.zip")

    base_env = make_env(env_name, sb3=True, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: Monitor(base_env)])
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=vec_env, device=device)

    for episode in range(eval_episodes):
        obs = vec_env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += float(reward[0])
            done = bool(done[0])
        print(f"Episode {episode + 1}: {episode_reward:.3f}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate PPO on SAR0 env.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=["train", "eval"],
        help="train (default) or eval (load saved model)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        eval_model()
