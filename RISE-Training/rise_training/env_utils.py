"""Training-side env constructor (delegates to SpecRLBench)."""

from __future__ import annotations

from specbench.envs.zones import make_zone_env


def make_env(env_name, render_mode=None, sb3=False):
    return make_zone_env(env_name, render_mode=render_mode, sb3=sb3)
