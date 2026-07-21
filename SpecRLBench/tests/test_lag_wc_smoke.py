"""Optional WC tiny-env smoke for all three Lag algos (skip if env unavailable)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_wc_tiny_smoke_or_skip():
    try:
        import safety_gymnasium  # noqa: F401
        from utils.env_utils_sb3 import make_vec
    except Exception as e:
        print(f"SKIP WC smoke (import): {e}")
        return

    env_id = "PointLTL4MASAR1WC-v0"
    try:
        env = make_vec(env_id, n_envs=1, render_mode=None, sb3=True, normalize=False)
        env.reset()
    except Exception as e:
        print(f"SKIP WC smoke (env make): {e}")
        return

    from ppo_lagrangian import PPOLag
    from sac_lagrangian import SACLag
    from trpo_lagrangian import TRPOLag

    common_kw = dict(
        cost_lim=0.0,
        penalty_init=1.0,
        penalty_lr=5e-2,
        device="cpu",
        seed=0,
        verbose=0,
        policy_kwargs=dict(net_arch=[32, 32]),
    )

    ppo = PPOLag("MultiInputPolicy", env, n_steps=32, batch_size=16, n_epochs=1, **common_kw)
    ppo.learn(total_timesteps=64)
    assert float(ppo.penalty.item()) >= 0.0
    print("WC PPOLag smoke OK")

    trpo = TRPOLag(
        "MultiInputPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_critic_updates=1,
        **common_kw,
    )
    trpo.learn(total_timesteps=64)
    assert float(trpo.penalty.item()) >= 0.0
    print("WC TRPOLag smoke OK")

    sac = SACLag(
        "MultiInputPolicy",
        env,
        learning_starts=16,
        buffer_size=2000,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        **common_kw,
    )
    sac.learn(total_timesteps=96)
    assert float(sac.penalty.item()) >= 0.0
    print("WC SACLag smoke OK")

    env.close()
    print("ALL WC TINY SMOKE PASSED")


if __name__ == "__main__":
    test_wc_tiny_smoke_or_skip()
