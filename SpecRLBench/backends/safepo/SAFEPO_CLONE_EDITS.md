# SafePO clone edits (Phase 8) — SpecRLBenchMods fork

**Canonical remote:** [`akhaled247/Safe-Policy-Optimization`](https://github.com/akhaled247/Safe-Policy-Optimization) branch **`SpecRLBenchMods`**
(https://github.com/akhaled247/Safe-Policy-Optimization/tree/SpecRLBenchMods).

Phase 8 Adam / GAE λ / clip-from-args changes live on that branch. Prefer:

```bash
git clone -b SpecRLBenchMods \
  https://github.com/akhaled247/Safe-Policy-Optimization.git \
  ~/RISE-2026/Safe-Policy-Optimization
# or: cd existing clone && git remote add fork … && git fetch && git checkout SpecRLBenchMods
touch ~/RISE-2026/Safe-Policy-Optimization/safepo/single_agent/__init__.py
pip install -e ~/RISE-2026/Safe-Policy-Optimization --no-deps
```

SpecRL [`runners.py`](runners.py) already patches module ``default_cfg``
(``batch_size``, ``learning_iters``, ``gamma``, ``target_kl``, …) and sets
``args.actor_lr`` / ``args.critic_lr`` / ``args.lam`` / ``args.lam_c`` /
``args.clip_ratio``.

Stock PKU SafePO **ignores** those args for Adam LRs and buffer GAE λ.
The checklist below is what ``SpecRLBenchMods`` should contain (re-apply if
rebasing onto newer upstream).

## Files

```text
safepo/single_agent/
  ppo.py  ppo_lag.py  trpo.py  trpo_lag.py  cpo.py
```

## 1. Adam learning rates

Replace hardcoded ``lr=3e-4`` (three places: actor, reward_critic, cost_critic)
with:

```python
actor_lr = float(getattr(args, "actor_lr", 3e-4))
critic_lr = float(getattr(args, "critic_lr", 3e-4))
actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=actor_lr)
reward_critic_optimizer = torch.optim.Adam(
    policy.reward_critic.parameters(), lr=critic_lr
)
cost_critic_optimizer = torch.optim.Adam(
    policy.cost_critic.parameters(), lr=critic_lr
)
```

## 2. Buffer GAE λ / γ

Where ``VectorizedOnPolicyBuffer(...)`` is constructed, pass:

```python
buffer = VectorizedOnPolicyBuffer(
    obs_space=obs_space,
    act_space=act_space,
    size=local_steps_per_epoch,
    device=device,
    num_envs=args.num_envs,
    gamma=config["gamma"],
    lam=float(getattr(args, "lam", 0.95)),
    lam_c=float(getattr(args, "lam_c", 0.95)),
)
```

## 3. Clip ratio (PPO / PPO-Lag only)

Replace ``torch.clamp(ratio, 0.8, 1.2)`` with:

```python
clip = float(getattr(args, "clip_ratio", 0.2))
ratio_cliped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
```

## 4. Reinstall / import check

```bash
cd ~/RISE-2026
source .venv-safepo/bin/activate
touch Safe-Policy-Optimization/safepo/single_agent/__init__.py
pip install -e ./Safe-Policy-Optimization --no-deps
python -c "from safepo.single_agent import ppo; print('ok', ppo.default_cfg)"
```

Operator note: AI Vault [[SpecRLBench SafePO Phases 7-9 Linux]] (#31).

## 5. SpecRL CMDP vec must use SafetyAsync (not Sync)

Stock SafePO ``make_sa_mujoco_env`` for ``num_envs > 1`` wraps
``SafetyAsyncVectorEnv`` (true multi-process). SpecRLBench env hook must do
the same via ``envs.cmdp.factory.make_cmdp_vec(..., parallel=True)`` →
``AsyncVectorSafetyEnv``.

**Do not** reintroduce serial-only ``SyncVectorSafetyEnv`` as the default for
``num_envs > 1`` — that made 1M MuJoCo steps ~6–8× slower than SB3 Subproc.

- CLI: ``--parallel True`` (default) / ``--parallel False`` (debug Sync)
- No ``SpecRLBenchMods`` Python diffs required for this speed fix
- Worker envs: ``autoreset=False`` (async worker already autoresets +
  ``final_observation``); Linux ``fork``, Windows ``spawn``

## 6. Checkpoint cadence

Stock / early SpecRLBenchMods only saved when ``(epoch+1) % 100 == 0 or epoch == 0``.
Runs with ``epochs < 100`` (e.g. L5WC 5M / T65536 → **76** epochs) wrote **only**
``model0.pt`` / ``state0.pkl``. Eval then loaded epoch-0 weights+RMS while
``progress.csv`` showed live EpRet from **unsaved** later epochs → train↔eval gap.

In ``ppo.py``, ``ppo_lag.py``, ``trpo.py``, ``trpo_lag.py``, ``cpo.py`` replace
the ``% 100`` gate with:

```python
save_freq = int(getattr(args, "save_model_freq", 10))  # epochs
if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % save_freq == 0:
    logger.torch_save(itr=epoch)
    if args.task not in isaac_gym_map.keys():
        logger.save_state({"Normalizer": env.obs_rms}, itr=epoch)
```

Also write a final ``torch_save`` / ``save_state`` for ``epochs - 1`` after the
training loop (belt-and-suspenders).

- Defaults: ``save_model_freq=10``, **always** save last epoch
- SpecRL CLI: ``--save-model-freq`` → ``args.save_model_freq``
- Smoke: ``--total-steps 196608 --steps-per-epoch 65536`` (3 epochs) → expect
  ``model0`` + ``model2`` and last ``state*.pkl`` with RMS ``count`` ≫ one-epoch
  worker steps (~8k for 8 envs × T65536)
