"""Torch API shims for SafePO on newer PyTorch.

SafePO passes ``verbose=`` to ``LinearLR``. That kwarg was removed in torch>=2.7.
"""

from __future__ import annotations

import inspect
import types
from typing import Any


def patch_linear_lr_verbose(mod: types.ModuleType | None = None) -> dict[str, Any]:
    """Drop ``verbose`` from ``LinearLR`` when torch no longer accepts it.

    Patches ``torch.optim.lr_scheduler.LinearLR`` and, if given, ``mod.LinearLR``
    (SafePO does ``from torch.optim.lr_scheduler import LinearLR`` at import time).
    """
    import torch.optim.lr_scheduler as lrs

    orig = lrs.LinearLR
    accepts_verbose = "verbose" in inspect.signature(orig.__init__).parameters
    info: dict[str, Any] = {
        "accepts_verbose": accepts_verbose,
        "patched": False,
        "mod_patched": False,
    }
    if accepts_verbose:
        return info

    if not getattr(lrs, "_specrlbench_linearlr_patched", False):

        class LinearLR(orig):  # type: ignore[valid-type,misc]
            def __init__(self, *args: Any, verbose: bool = False, **kwargs: Any) -> None:
                # ``verbose`` accepted then ignored — SafePO still passes it.
                super().__init__(*args, **kwargs)

        lrs.LinearLR = LinearLR  # type: ignore[misc,assignment]
        lrs._specrlbench_linearlr_patched = True
        info["patched"] = True
    else:
        info["patched"] = True  # already patched earlier in process

    if mod is not None and hasattr(mod, "LinearLR"):
        mod.LinearLR = lrs.LinearLR  # type: ignore[attr-defined]
        info["mod_patched"] = True
    return info
