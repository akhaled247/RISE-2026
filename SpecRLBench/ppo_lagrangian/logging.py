"""Lightweight epoch logger (stdout + optional TensorBoard)."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class EpochLogger:
    """Accumulate per-epoch stats and dump tabular / TensorBoard summaries."""

    def __init__(self, tensorboard_log: str | None = None, tb_log_name: str = "PPOLagrangian", verbose: int = 1) -> None:
        self.verbose = verbose
        self.epoch_dict: dict[str, list[Any]] = defaultdict(list)
        self._writer = None
        self._tb_step = 0
        if tensorboard_log is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = Path(tensorboard_log) / f"{tb_log_name}_{int(time.time())}"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(str(log_dir))
            except ImportError:
                if verbose:
                    print("tensorboard not installed; logging to stdout only")

    def store(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.epoch_dict[k].append(v)

    def get_stats(self, key: str) -> tuple[float, float]:
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(np.asarray(v[0]).shape) > 0 else np.asarray(v, dtype=np.float64)
        return float(np.mean(vals)), float(np.std(vals))

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def dump_tabular(self, step: int | None = None) -> dict[str, float]:
        """Average stored values, print, write TB, clear epoch store."""
        results: dict[str, float] = {}
        for k, v in self.epoch_dict.items():
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(np.asarray(v[0]).shape) > 0 else np.asarray(v, dtype=np.float64)
            mean = float(np.mean(vals))
            results[k] = mean
            if self._writer is not None:
                tb_step = step if step is not None else self._tb_step
                self._writer.add_scalar(k, mean, tb_step)
        if self.verbose:
            keys = sorted(results.keys())
            width = max((len(k) for k in keys), default=10)
            print("-" * (width + 20))
            for k in keys:
                print(f"| {k:<{width}} | {results[k]:12.5g} |")
            print("-" * (width + 20), flush=True)
        self.epoch_dict.clear()
        if step is not None:
            self._tb_step = step
        else:
            self._tb_step += 1
        if self._writer is not None:
            self._writer.flush()
        return results

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
