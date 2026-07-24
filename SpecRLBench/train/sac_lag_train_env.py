"""SAC-Lagrangian training — deferred (not in SafePO single-agent set).

SafePO migration: SAC / SAC-Lag are out of scope until a later off-policy vendor.
Use ``train/ppo_lag_train_env.py`` for constrained WC.
"""

from __future__ import annotations

import sys

_MSG = (
    "SAC-Lag is deferred post-SafePO migration (SafePO SA has no SAC). "
    "Use train/ppo_lag_train_env.py. "
    "See AI Vault [[SpecRLBench SafePO Migration Plan]] Phase 8."
)

if __name__ == "__main__":
    print(_MSG, file=sys.stderr)
    raise SystemExit(_MSG)
