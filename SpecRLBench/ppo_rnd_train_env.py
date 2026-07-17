import sys
from datetime import datetime
from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "specbench" / "envs" / "zones" / "safety-gymnasium"))
sys.path.insert(0, str(ROOT))

import safety_gymnasium  # noqa: F401
from ppo_load_env import eval_model
from rnd import RNDConfig, PPORND, resolve_rnd_obs_keys
from utils.env_utils import make_vec

# --- edit these before each run ---
# Levels: PointLTL4MASAR1-v0 | PointLTL4MASAR1WC-v0 | PointLTL5MASAR1-v0 | PointLTL5MASAR1WC-v0 | …
env_name = "PointLTL4MASAR1WC-v0"
name_time = datetime.now().strftime("%Y%m%d_%H%M")
TRAINING_LOG_PATH = f"./_training_logs/rnd_ppo_{env_name}_tensorboard/"

# RND knobs (OpenAI-faithful PPORND: int_coeff scales intrinsic advantage; ext_coeff=1 so beta alone scales)
# use_rnd=False → plain PPO control (same hparams otherwise)
use_rnd = True
int_coeff = 0.5   # legacy "beta"; OpenAI default int_coeff=1.0 — start lower on sparse SAR
ext_coeff = 1.0   # OpenAI Atari used 2.0; keep 1.0 so sweeps only touch int_coeff

# --- tuning suggestions (comments only; not applied unless you edit knobs above / train() defaults) ---
# Sweep table (run per level; try L4 then L5):
#   S0  int_coeff=0.0  ent_coef=0.02   → PPO control (set use_rnd=False)
#   S1  int_coeff=0.1  ent_coef=0.01   → weak intrinsic
#   S2  int_coeff=0.5  ent_coef=0.005  → recommended default for sparse SAR
#   S3  int_coeff=1.0  ent_coef=0.0    → strong intrinsic, no entropy
#   S4  int_coeff=0.5  ent_coef=0.0    → default beta, entropy off
# OpenAI Atari defaults (reference, not SAR): n_steps=128, n_epochs=4, clip=0.1, ent=0.001,
#   lr=1e-4, int_coeff=1, ext_coeff=2, gamma=0.99, gamma_ext=0.99, use_news=False.
# Match plain PPO first (n_steps=2048, ent_coef=0.02, lr=5e-5, clip=0.2) then lower ent when
#   int_coeff > 0 so entropy + intrinsic do not both dominate.
# L5 success bar: beat plain-PPO ~12/50 rescue. L4: beat that level's own S0 (use_rnd=False).
# Assumes true-sparse extrinsic reward (you own env sparse switch).
# RND obs: L4 walls/wall_sensor; L5+ buildings/walls/ltl_walls/wall_sensor; casualties excluded.
# feature_dim / rnd_rep_size=256 is a SAR compromise; OpenAI Atari uses rep_size=512.

# Level → RND obs key substrings (buildings/walls focus; casualties excluded)
LEVEL_RND_PREFIXES: dict[str, list[str]] = {
    "PointLTL4MASAR1-v0": ["walls", "wall_sensor"],
    "PointLTL4MASAR1WC-v0": ["walls", "wall_sensor"],
    "PointLTL5MASAR1-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
    "PointLTL5MASAR1WC-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
    "PointLTL6MASAR1-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
    "PointLTL6MASAR1WC-v0": ["buildings", "walls", "ltl_walls", "wall_sensor"],
}


def train(
        total_timesteps=1_000_000,
        seed=0,
        n_envs=8,
        ent_coef=0.005,  # S2 default; use 0.02 for PPO-parity control (use_rnd=False)
        learning_rate=5e-5,
        n_steps=2048,  # match ppo_train_env; was 4096 in older RND sweeps (slower iters/s)
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        target_kl=0.05,  # 0.08
        startup_log=True,
) -> tuple[str, str]:
    rollout_steps = n_steps * n_envs
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    if startup_log:
        print(f"Logging to {TRAINING_LOG_PATH}...")
        print(f"<<<{rollout_steps / batch_size}>>> minibatches per rollout"
              f"\n <<<{total_timesteps // rollout_steps}>>> policy updates total")
        print("=" * 40)
        print(f"train env={env_name} device={device} steps={total_timesteps}")
        print(
            f"use_rnd={use_rnd} int_coeff={int_coeff} ext_coeff={ext_coeff} "
            f"ent_coef={ent_coef} lr={learning_rate}"
        )
        print(
            f"PPO iter = {rollout_steps} env steps collect + {n_epochs} epochs x "
            f"{rollout_steps // batch_size} minibatches "
            f"- SB3 iters/s scales ~1/n_steps"
        )

    env = make_vec(env_name, n_envs=n_envs, render_mode=None, sb3=True, normalize=True)
    if startup_log:
        print("Warming up vector envs...")
    env.seed(seed=0)  # Constants env seed to reduce variation between master seeds
    env.reset()

    if use_rnd:
        if env_name not in LEVEL_RND_PREFIXES:
            raise KeyError(
                f"env_name={env_name!r} has no RND obs profile; "
                f"known={list(LEVEL_RND_PREFIXES)}"
            )
        rnd_obs_keys = None
        if isinstance(env.observation_space, spaces.Dict):
            rnd_obs_keys = resolve_rnd_obs_keys(
                env.observation_space,
                include_substrings=LEVEL_RND_PREFIXES[env_name],
            )
            if startup_log:
                print(f"RND obs_keys ({len(rnd_obs_keys)}): {rnd_obs_keys}")

        rnd_config = RNDConfig(
            use_rnd=True,
            intrinsic_reward_coef=int_coeff,
            feature_dim=256,
            obs_keys=rnd_obs_keys,
            update_ob_stats_from_random_agent=False,
            random_obs_init_steps=0,
            ext_coeff=ext_coeff,
            use_news=False,
        )
        model = PPORND(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            target_kl=target_kl,
            device=device,
            tensorboard_log=TRAINING_LOG_PATH,
            seed=seed,
            clip_range=clip_range,
            rnd_config=rnd_config,
            int_coeff=int_coeff,
            ext_coeff=ext_coeff,
        )
        algo_tag = "PPORND"
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            target_kl=target_kl,
            device=device,
            tensorboard_log=TRAINING_LOG_PATH,
            seed=seed,
            clip_range=clip_range,
        )
        algo_tag = "PPO"

    env.seed(seed=0)  # Constants env seed to reduce variation between master seeds
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        progress_bar=True,
        # callback=ThroughputCallback(total_timesteps),
        tb_log_name=(
            f"{algo_tag}_t{name_time}"
            f"_st{n_steps}"
            f"_bs{batch_size}"
            f"_tt{total_timesteps / 1_000_000:.1f}M"
            f"_ec{ent_coef}"
            f"_lr{learning_rate}"
            f"_ep{n_epochs}"
            f"_cr{clip_range}"
            f"_kl{target_kl}"
            f"_beta{int_coeff}"
            f"_s{seed}"
        ),
    )

    model_path = (
        f"_models/rnd_ppo_{name_time}_{env_name}_{seed}"
        if use_rnd
        else f"_models/ppo_{name_time}_{env_name}_{seed}"
    )
    vec_norm_path = f"{model_path}_vecnormalize.pkl"
    model.save(model_path)
    env.save(vec_norm_path)
    print(f"saved model: {model_path}.zip")
    # print(f"saved vecnorm: {vec_norm_path}")
    env.close()
    return model_path, vec_norm_path


if __name__ == "__main__":
    for i in range(1):
        model_path, vec_norm_path = train(
            seed=int(i),  # Tested up to and including env 3 at home
            startup_log=True,
            total_timesteps=5_000_000,
        )
        eval_model(
            env_name=env_name,
            render_mode=None,
            m_path=model_path,
        )
