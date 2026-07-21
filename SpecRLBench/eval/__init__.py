"""SafePO / framework-agnostic eval package (Phase 7).

SB3 zip eval remains in ``load_env.py`` until Phase 10.
"""

from .eval_model import eval_single_run, main

__all__ = ["eval_single_run", "main"]
