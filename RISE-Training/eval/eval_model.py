"""Re-export SafePO post-train eval (plan path ``eval/eval_model.py``).

Implementation lives in ``rise_training.safepo.evaluate``. User CLI:
``python eval_safepo_env.py --run-dir ...``.
"""

from __future__ import annotations

from rise_training.safepo.evaluate import (
    benchmark_eval,
    build_parser,
    eval_single_run,
    main,
)

__all__ = ["benchmark_eval", "build_parser", "eval_single_run", "main"]
