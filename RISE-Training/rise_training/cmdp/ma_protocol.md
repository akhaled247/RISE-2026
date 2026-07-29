# SpecRLBench multi-agent SafePO adapter protocol
#
# SafePO MultiGoalEnv / ShareEnv expect:
#   reset → (obs_n, share_obs_n, avail_actions)
#   step(actions) → (obs, share_obs, rewards, costs, dones, infos, avail_actions)
#
# RISE-Training path: `rise_training.safepo.ma_factory.SpecRLMultiGoalEnv` wraps
# `rise_training.env_utils.make_env(..., flat=False)` → SafetyGymWrapperMASAR / WC / AC.
#
# Wiring:
# 1. `env_hook.patch_safepo_ma_env_factory()` redirects
#    `safepo.common.env.make_ma_multi_goal_env` for Point/Car/AntLTL* tasks to
#    ShareDummyVecEnv / ShareSubprocVecEnv over SpecRLMultiGoalEnv.
# 2. Per-agent costs from `cost_walls` / `cost_collision` (gremlins = agent stand-ins).
# 3. Do not use `flat=True` flatten for true MA training.
#
# See: safepo.common.wrappers.MultiGoalEnv
# See: rise_training/safepo/ma_factory.py
