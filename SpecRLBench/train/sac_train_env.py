"""SAC training — deferred (not in SafePO single-agent set).

SafePO migration (Phase 8): SAC / SAC-Lag are out of scope until a later
off-policy vendor. Use PPO / PPO-Lag / TRPO / CPO via ``train/ppo_train_env.py``
etc. Legacy SB3 SAC code lives in git history until Phase 10 cleanup.
"""

from __future__ import annotations

import sys

_MSG = (
    "SAC is deferred post-SafePO migration (SafePO SA has no SAC). "
    "Use train/ppo_train_env.py or train/ppo_lag_train_env.py. "
)

if __name__ == "__main__":
    print(_MSG, file=sys.stderr)
    raise SystemExit(_MSG)
