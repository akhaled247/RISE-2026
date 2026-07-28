"""Training-side env constructor (delegates to SpecRLBench zone_env)."""

from __future__ import annotations

from rise_training.zones_factory import make_zone_env


def make_env(env_name, render_mode=None, sb3=False):
    return make_zone_env(env_name, render_mode=render_mode, sb3=sb3)
