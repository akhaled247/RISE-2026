# SpecRLBench SAR paper protocol (GenZ-LTL + SafePO)

Aligned with SpecRLBench paper §5.3: **train single-agent → deploy shared policy on multi-agent eval**. Inter-agent collision is **not** modeled (no gremlin cost/termination on SAR WC/LTL wrappers).

## Unified pathway

| Step | GenZ-LTL | SafePO |
|------|----------|--------|
| Train | `PointLTL0MASAR1WC-v0`, RCO + LTL curriculum | `PointLTL0MASAR1WC-v0`, PPO / PPO-Lag |
| Eval | `PointLTL0MASAR2WC-v0`, `simulate_ma_sar.py` + Büchi | `PointLTL0MASAR2WC-v0`, `eval_safepo_sa_on_ma_env.py` |

Team episode end on MA deploy: **`any(terminated/truncated)`**.

## Constraints

| Signal | WC (`sar_ltl_ordering=False`) | LTL (`sar_ltl_ordering=True`) |
|--------|------------------------------|-------------------------------|
| Walls | `cost_walls` → `info['cost']`, terminate | same |
| Entrapped-first | — | surface rescue before all entrapped rescued → cost |
| Collision | **ignored** | **ignored** |

Raw `cost_collision` may still appear in per-agent info for diagnostics; wrappers do not propagate it.

## SafePO train

Recipe constant: `rise_training.safepo.config.SAR_PAPER_PROTOCOL` (auto-applied for `*MASAR1*WC*` in `runners.py`).

```bash
cd RISE-Training
python train/ppo_train_env.py --task PointLTL0MASAR1WC-v0 --seed 0
python train/ppo_lag_train_env.py --task PointLTL0MASAR1WC-v0 --seed 0 --cost-limit 1.0
```

## SafePO MA deploy eval

```bash
python eval_safepo_sa_on_ma_env.py \
  --run-dir ./_training_logs/safepo/PointLTL0MASAR1WC-v0/ppo/seed-000-... \
  --eval-env PointLTL0MASAR2WC-v0 --eval-episodes 50
```

Uses `rise_training/cmdp/obs_spec.py` flatten keys from SA training (not `ma_factory._pack_obs`).

Writes `eval_summary_ma_deploy.json` beside the run dir.

## GenZ-LTL

```bash
PYTHONPATH=src/ python src/train/train_rco.py \
  --env PointLTL0MASAR1WC-v0 --curriculum PointLTL0MASAR1WC-v0 \
  --model_config zones_safety --name GenZ-SAR --seed 0

PYTHONPATH=src/ python src/evaluation/simulate_ma_sar.py --exp GenZ-SAR --seed 0
```

## Optional MARL baseline (not paper protocol)

IPPO / MAPPO training on `PointLTL0MASAR2-v0` remains in `RISE-Training` for comparison; see `ma_factory.py` and `eval_safepo_ma_env.py`.

## Multi-agent SafePO adapter (reference)

SafePO MultiGoalEnv / ShareEnv expect:

- `reset` → `(obs_n, share_obs_n, avail_actions)`
- `step(actions)` → `(obs, share_obs, rewards, costs, dones, infos, avail_actions)`

`rise_training.safepo.ma_factory.SpecRLMultiGoalEnv` wraps `make_env(..., flat=False)`.

Do not use `flat=True` for true MA training.

See: `safepo.common.wrappers.MultiGoalEnv` · `rise_training/safepo/ma_factory.py`
