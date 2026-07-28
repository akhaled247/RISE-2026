# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Load and apply shared SAR YAML constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SAR_CONFIG_PATH = Path(__file__).with_name('sar_config.yaml')
_CACHED_CONFIG: dict[str, Any] | None = None

# Registration-only keys that must not reach BaseTask._parse.
REGISTRATION_KEYS = frozenset({'env_id'})

# Nested YAML blocks that map onto task / conf objects.
_NESTED_CONF_ROOTS = frozenset({
    'lidar_conf',
    'cost_conf',
    'mechanism_conf',
    'placements_conf',
    'render_conf',
})


def load_sar_config(*, force_reload: bool = False) -> dict[str, Any]:
    """Load and cache ``sar_config.yaml``."""
    global _CACHED_CONFIG  # pylint: disable=global-statement
    if _CACHED_CONFIG is None or force_reload:
        with _SAR_CONFIG_PATH.open(encoding='utf-8') as handle:
            _CACHED_CONFIG = yaml.safe_load(handle) or {}
    return _CACHED_CONFIG


def _as_tuple_placements(value: Any) -> list[tuple[float, float, float, float]]:
    return [tuple(float(x) for x in row) for row in value]


def _apply_mapping(task: Any, mapping: dict[str, Any], *, prefix: str = '') -> None:
    """Apply a nested dict onto ``task`` or nested conf objects."""
    for key, value in mapping.items():
        path = f'{prefix}.{key}' if prefix else key

        if key == 'agent' and isinstance(value, dict):
            if 'keepout' in value:
                task.agent_keepout = float(value['keepout'])
            if 'placements' in value:
                task.agent_placements = _as_tuple_placements(value['placements'])
            continue

        if key == 'gremlins' and isinstance(value, dict):
            task.gremlin_size = float(value['size'])
            task.gremlin_dist_threshold = float(value['dist_threshold'])
            task.gremlin_keepout = float(value['keepout'])
            continue

        if key == 'building_perimeter_walls' and isinstance(value, dict):
            task.building_perimeter_wall_height = float(value['height'])
            task.building_perimeter_wall_collision_threshold = float(
                value['collision_threshold'],
            )
            continue

        if key in _NESTED_CONF_ROOTS and isinstance(value, dict):
            conf_obj = getattr(task, key)
            for nested_key, nested_value in value.items():
                setattr(conf_obj, nested_key, nested_value)
            continue

        if isinstance(value, dict):
            _apply_mapping(task, value, prefix=path)
            continue

        setattr(task, key, value)


def apply_sar_constants(
    task: Any,
    sections: tuple[str, ...] = ('must_be_constant', 'configurable'),
    skip_keys: frozenset[str] | None = None,
) -> None:
    """Set must-be-constant and configurable values from YAML onto ``task``.

    ``skip_keys`` preserves explicit CustomizedSAR overrides already applied via
    ``BaseTask._parse`` (top-level names or dotted ``lidar_conf.num_bins``).
    """
    skip = skip_keys or frozenset()
    cfg = load_sar_config()
    for section in sections:
        section_data = cfg.get(section) or {}
        if not isinstance(section_data, dict):
            raise TypeError(f'sar_config.yaml section {section!r} must be a mapping')
        filtered = _filter_mapping(section_data, skip)
        _apply_mapping(task, filtered)


def _filter_mapping(mapping: dict[str, Any], skip: frozenset[str], prefix: str = '') -> dict[str, Any]:
    """Drop keys (and nested leaves) listed in ``skip``."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f'{prefix}.{key}' if prefix else key
        if path in skip or key in skip:
            continue
        if isinstance(value, dict) and key not in {'agent', 'gremlins', 'building_perimeter_walls'}:
            nested = _filter_mapping(value, skip, prefix=path)
            if nested:
                out[key] = nested
            continue
        if isinstance(value, dict):
            # Treat special blocks as atomic unless a nested path is skipped.
            nested = _filter_mapping(value, skip, prefix=path)
            if nested:
                out[key] = nested
            continue
        out[key] = value
    return out


def merge_easy_sar_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML ``easy`` defaults into a CustomizedSAR env config.

    Leaves registration keys stripped for BaseTask._parse. Nested
    ``lidar_conf.num_bins`` is returned under dotted key so callers can apply
    it after ``lidar_conf`` exists.
    """
    merged = dict(config)
    for key in REGISTRATION_KEYS:
        merged.pop(key, None)

    easy = dict(load_sar_config().get('easy') or {})
    lidar_easy = dict(easy.pop('lidar_conf', None) or {})

    for key, value in easy.items():
        if key not in merged:
            merged[key] = value

    if 'num_bins' in lidar_easy and 'lidar_conf.num_bins' not in merged:
        # Prefer explicit dotted override already in config.
        if not (
            isinstance(merged.get('lidar_conf'), dict)
            and 'num_bins' in merged['lidar_conf']
        ):
            merged['lidar_conf.num_bins'] = lidar_easy['num_bins']

    # building_num null means "use agent_num" — drop so L2 fills it.
    if merged.get('building_num') is None:
        merged.pop('building_num', None)

    # Flatten accidental nested lidar_conf from user config into dotted form
    # after Underlying._parse (lidar_conf object not created yet).
    nested_lidar = merged.pop('lidar_conf', None)
    if isinstance(nested_lidar, dict):
        for nested_key, nested_value in nested_lidar.items():
            dotted = f'lidar_conf.{nested_key}'
            if dotted not in merged:
                merged[dotted] = nested_value

    return merged


def pop_post_init_lidar_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Pull ``lidar_conf.*`` keys that must be applied after BaseTask creates confs."""
    overrides = {}
    for key in list(config.keys()):
        if key.startswith('lidar_conf.'):
            overrides[key.split('.', 1)[1]] = config.pop(key)
    return overrides
