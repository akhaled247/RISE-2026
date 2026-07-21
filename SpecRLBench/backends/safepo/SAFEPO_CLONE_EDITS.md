# SafePO clone edits (Phase 8) — apply on Linux `depend`

SpecRL [`runners.py`](runners.py) already patches module ``default_cfg``
(``batch_size``, ``learning_iters``, ``gamma``, ``target_kl``, …) and sets
``args.actor_lr`` / ``args.critic_lr`` / ``args.lam`` / ``args.lam_c`` /
``args.clip_ratio``.

Stock SafePO **ignores** those args for Adam LRs and buffer GAE λ. Edit the
editable clone:

```text
~/RISE-2026/Safe-Policy-Optimization/safepo/single_agent/
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

Operator checklist with smoke trains: AI Vault [[SpecRLBench SafePO Phases 7-9 Linux]] (#31).
