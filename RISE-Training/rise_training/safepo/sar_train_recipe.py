"""SAR MASAR1WC training recipes (paper protocol, portable — no FileTransfer paths)."""

from __future__ import annotations

from typing import Any

TRAIN_ENV = "PointLTL0MASAR1WC-v0"
EVAL_ENV_SA = "PointLTL0MASAR1WC-v0"
EVAL_ENV_MA = "PointLTL0MASAR2WC-v0"

# Shared SafePO recipe from successful L0 MASAR1WC runs (10M steps, rescue_rate >= 0.94).
SHARED_TRAIN_KWARGS: dict[str, Any] = {
    "task": TRAIN_ENV,
    "total_steps": 10_000_000,
    "num_envs": 16,
    "steps_per_epoch": 65536,
    "gamma": 0.995,
    "lam": 0.98,
    "lam_c": 0.98,
    "actor_lr": 5e-5,
    "critic_lr": 1e-3,
    "batch_size": 256,
    "target_kl": 0.05,
    "clip_ratio": 0.2,
    "max_grad_norm": 40.0,
    "hidden_sizes": [64, 64],
    "parallel": True,
    "write_terminal": False,
    "use_tensorboard": True,
    "lr_end_factor": 1.0,
    "sar_ltl_ordering": True,
}

ALGO_RECIPES: dict[str, dict[str, Any]] = {
    "ppo": {
        **SHARED_TRAIN_KWARGS,
        "learning_iters": 10,
        "ent_coef": 0.02,
        "cost_limit": 0.0,
        "experiment": "ppo_sar_masaru1wc",
    },
    "ppo_lag": {
        **SHARED_TRAIN_KWARGS,
        "learning_iters": 10,
        "ent_coef": 0.02,
        "cost_limit": 0.25,
        "lagrangian_multiplier_init": 0.0,
        "lagrangian_multiplier_lr": 0.01,
        "experiment": "ppo_lag_sar_masaru1wc",
    },
    "trpo": {
        **SHARED_TRAIN_KWARGS,
        "learning_iters": 1,
        "ent_coef": 0.0,
        "cost_limit": 0.0,
        "experiment": "trpo_sar_masaru1wc",
    },
    "trpo_lag": {
        **SHARED_TRAIN_KWARGS,
        "learning_iters": 1,
        "ent_coef": 0.0,
        "cost_limit": 0.25,
        "lagrangian_multiplier_init": 0.0,
        "lagrangian_multiplier_lr": 0.01,
        "experiment": "trpo_lag_sar_masaru1wc",
    },
}

GENZ_RCO_RECIPE: dict[str, Any] = {
    "env": TRAIN_ENV,
    "curriculum": TRAIN_ENV,
    "model_config": TRAIN_ENV,
    "num_steps": 10_000_000,
    "num_procs": 24,
    "steps_per_process": 4096,
    "batch_size": 2048,
    "epochs": 10,
    "discount": 0.998,
    "lr": 3e-4,
    "entropy_coef": 0.003,
    "vec_backend": "safety_async",
    "sar_env_backend": "specrl",
    "fast_action_bridge": True,
    "save_interval": 10,
}
