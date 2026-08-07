"""GenZ async vector env + OOM guardrails for multiprocess SAR training.

Spawn-safe ``GenZSafetyAsyncEnv``, worker factory, and action bridge for torch_ac.
"""

__all__ = [
    "action_bridge",
    "genz_async_vec",
    "genz_env_runner",
    "oom_guard",
    "paths_bootstrap",
    "worker_factory",
]
