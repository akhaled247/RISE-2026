# Future multi-agent SafePO adapter protocol (stub — no training in this migration).
#
# SafePO MultiGoalEnv / ShareEnv expect:
#   reset → (obs_n, share_obs_n, avail_actions)
#   step(actions) → (obs, share_obs, rewards, costs, dones, infos, avail_actions)
#
# SpecRLBench native path: make_env(..., sb3=False) keeps per-agent Dict
# obs/reward/cost from SafetyGymWrapperMASAR(WC).
#
# When implementing MAPPO-Lag / MACPO:
# 1. Build MultiGoalEnv-shaped wrapper over SafetyGymWrapperMASARWC(sb3=False).
# 2. Map per-agent cost_walls (or info cost) into SafePO costs[agent].
# 3. Do not flatten to a single SB3 agent.
#
# See: safepo.common.wrappers.MultiGoalEnv
# See: backends/safepo/ma_factory.py
