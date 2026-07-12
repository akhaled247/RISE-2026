import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))

import safety_gymnasium  # noqa: F401
from utils.env_utils import make_env, make_vec


# #region agent log
def _dbg_train(hypothesis_id, message, data):
    try:
        root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
        payload = {
            "sessionId": "b1323e",
            "hypothesisId": hypothesis_id,
            "location": "ppo_env.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(root / "debug-b1323e.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


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
            row = {
                "timesteps": int(self.num_timesteps),
                "rollout_fps": round(rollout_fps, 1),
                "eta_minutes": round(eta_min, 1) if eta_min is not None else None,
                "ep_len_mean": ep_len_mean,
            }
            _dbg_train("H7", "ppo rollout throughput", row)
            print(
                f"[throughput] steps={row['timesteps']} "
                f"fps={row['rollout_fps']} "
                f"eta_min={row['eta_minutes']} "
                f"ep_len={row['ep_len_mean']}"
            )
        self._last_time = now
        self._last_steps = self.num_timesteps

# #endregion

# Config
env_name = 'PointLTL0MASAR1-v0'
run_num = 1  # INCREMENT EACH TIME
MODEL_PATH = f"_models/ppo_{env_name}_run{run_num}"
TRAINING_LOG_PATH = f"./_training_logs/ppo_{env_name}_tensorboard/"
TOTAL_TIMESTEPS = 500_000
SMOKE_TIMESTEPS = 50_000
seed = 0
n_envs = 8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"env={env_name} device={device}")

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
    env.save(f"{MODEL_PATH}_vecnormalize.pkl")

    eval_env = make_env(env_name, sb3=True, render_mode=None)
    obs, info = eval_env.reset(seed=seed)
    episodes = 10
    for episode in range(episodes):
        obs, info = eval_env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        print(f"Episode {episode + 1}: {episode_reward:.3f}")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
