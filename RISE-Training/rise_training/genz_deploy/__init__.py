"""SA→MA GenZ SAR deploy protocol (paper §5.3).

Multi-agent rollout, Büchi coordinator, checkpoint loading, and parallel eval.
Feature packing lives in GenZ ``envs.sar_features``; import submodules here, e.g.:

  from rise_training.genz_deploy.ma_rollout import simulate_ma_sar
  from rise_training.genz_deploy.coordinator import MultiAgentSARCoordinator
"""

__all__ = [
    "checkpoint_kind",
    "coordinator",
    "env_check",
    "eval_stack",
    "loading",
    "ma_phase_gating",
    "ma_rollout",
    "ma_rollout_ppo",
    "parallel_eval",
    "ppo_loading",
    "sar_debug",
]
