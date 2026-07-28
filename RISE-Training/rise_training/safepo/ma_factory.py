"""MultiGoalEnv-shaped adapter for SpecRLBench SAR envs (SafePO MA)."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box


def cmdp_cost_channels(task_id: str) -> tuple[bool, bool]:
    """Return (count_walls, count_collision) per SAR env ID suffix (WC / AC)."""
    return ("WC" in task_id, "AC" in task_id)


def make_ma_cmdp_env(task: str, seed: int, cfg_train: dict | None = None):
    """Build one SpecRL MultiGoal-compatible env (used inside Share* vec workers)."""
    del cfg_train  # SafePO passes cfg; unused for construction parity with MultiGoalEnv
    return SpecRLMultiGoalEnv(task=task, seed=seed)


class SpecRLMultiGoalEnv:
    """Bridge ``make_env(..., flat=False)`` to SafePO MultiGoalEnv step/reset contract.

    reset → (obs_n, share_obs_n, avail_actions)
    step(actions) → (obs, share_obs, rewards, costs, dones, infos, avail_actions)
    """

    def __init__(self, task: str, seed: int, render_mode = None):
        from rise_training.env_utils import make_env

        self.task_id = task
        self.env = make_env(task, flat=False, render_mode=render_mode)
        self._seed = int(seed)
        obs0, info0 = self.env.reset(seed=self._seed)
        self._last_info = info0

        unwrapped = self.env.unwrapped
        self.possible_agents = list(unwrapped.possible_agents)
        self.num_agents = int(getattr(unwrapped, "num_agents", len(self.possible_agents)))
        assert self.num_agents == len(self.possible_agents)

        self.action_spaces = {
            agent: self.env.action_space(agent) for agent in self.possible_agents
        }
        self.single_action_space = self.action_spaces[self.possible_agents[0]]
        self.n_actions = int(np.prod(self.single_action_space.shape))

        # Probe obs sizes from flattened per-agent dicts (keys differ per agent)
        self._obs_keys_by_agent = {
            agent: sorted(obs0[agent].keys()) for agent in self.possible_agents
        }
        flats0 = [
            self._flatten_agent_obs(obs0[agent], agent) for agent in self.possible_agents
        ]
        self._agent_obs_dim = max(int(f.shape[0]) for f in flats0)
        self.obs_size = self._agent_obs_dim + self.num_agents
        self.share_obs_size = self._agent_obs_dim * self.num_agents

        self.observation_spaces = {}
        self.share_observation_spaces = {}
        for i in range(self.num_agents):
            self.observation_spaces[f"agent_{i}"] = Box(
                low=-10.0, high=10.0, shape=(self.obs_size,), dtype=np.float32
            )
            self.share_observation_spaces[f"agent_{i}"] = Box(
                low=-10.0, high=10.0, shape=(self.share_obs_size,), dtype=np.float32
            )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _flatten_agent_obs(self, agent_obs: dict, agent: str | None = None) -> np.ndarray:
        if agent is not None and hasattr(self, "_obs_keys_by_agent"):
            keys = self._obs_keys_by_agent[agent]
        else:
            keys = sorted(agent_obs.keys())
        parts = []
        for k in keys:
            if k not in agent_obs:
                continue
            v = agent_obs[k]
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        flat = np.concatenate(parts, axis=0).astype(np.float32)
        dim = getattr(self, "_agent_obs_dim", flat.shape[0])
        if flat.shape[0] < dim:
            flat = np.pad(flat, (0, dim - flat.shape[0]))
        elif flat.shape[0] > dim:
            flat = flat[:dim]
        return flat

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        std = float(np.std(x))
        if std < 1e-8:
            return x - np.mean(x)
        return (x - np.mean(x)) / (std + 1e-8)

    def _pack_obs(self, obs_dict: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
        flats = [
            self._flatten_agent_obs(obs_dict[agent], agent)
            for agent in self.possible_agents
        ]
        share_raw = np.concatenate(flats, axis=0).astype(np.float32)
        share_norm = self._normalize(share_raw)
        obs_n: list[np.ndarray] = []
        share_n: list[np.ndarray] = []
        for i, flat in enumerate(flats):
            agent_id = np.zeros(self.num_agents, dtype=np.float32)
            agent_id[i] = 1.0
            local = np.concatenate([self._normalize(flat), agent_id], axis=0)
            obs_n.append(local.astype(np.float32))
            share_n.append(share_norm.copy())
        return obs_n, share_n

    def _get_avail_actions(self) -> np.ndarray:
        return np.ones((self.num_agents, self.n_actions), dtype=np.float32)

    def _agent_cost(self, agent: str, info: dict) -> float:
        agent_info = info.get(agent, {}) if isinstance(info, dict) else {}
        if not isinstance(agent_info, dict):
            agent_info = {}
        use_walls, use_collision = cmdp_cost_channels(self.task_id)
        walls = float(agent_info.get("cost_walls", 0) or 0) if use_walls else 0.0
        coll = float(agent_info.get("cost_collision", 0) or 0) if use_collision else 0.0
        if walls or coll:
            return walls + coll
        if (use_walls or use_collision) and isinstance(info, dict) and "cost" in info:
            return float(info.get("cost", 0) or 0) / float(self.num_agents)
        return 0.0

    def reset(self, seed: int | None = None):
        """Reset env; only pin layout seed when ``seed`` is explicit.

        Vec-env autoreset calls ``reset()`` with no seed. Passing a fixed
        ``self._seed`` every time blocked :class:`SafetyGymWrapperMASAR`
        ``_layout_seed`` cycling (SA parity) and locked each worker to one layout.
        """
        if seed is not None:
            self._seed = int(seed)
            obs, info = self.env.reset(seed=self._seed)
        else:
            # Let SAR wrapper advance ``_layout_seed`` (same as SA path).
            obs, info = self.env.reset()
        self._last_info = info
        obs_n, share_n = self._pack_obs(obs)
        return obs_n, share_n, self._get_avail_actions()

    def step(self, actions):
        dict_actions = {}
        for agent_id, agent in enumerate(self.possible_agents):
            act = actions[agent_id]
            # Torch tensor → numpy; plain ndarray passes through
            if type(act).__module__.startswith("torch") and hasattr(act, "detach"):
                act = act.detach().cpu().numpy()
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            dict_actions[agent] = act

        obs, rewards, terminated, truncated, info = self.env.step(dict_actions)
        self._last_info = info

        dones = []
        rew_list = []
        cost_list = []
        info_list = []
        for agent in self.possible_agents:
            term = terminated[agent] if isinstance(terminated, dict) else bool(terminated)
            trunc = truncated[agent] if isinstance(truncated, dict) else bool(truncated)
            dones.append(bool(term or trunc))
            r = rewards[agent] if isinstance(rewards, dict) else float(rewards)
            rew_list.append([float(r)])
            cost_list.append([float(self._agent_cost(agent, info))])
            info_list.append(info.get(agent, {}) if isinstance(info, dict) else {})

        # Attach team-level fields for eval + term/trunc for MA logging
        for i, agent in enumerate(self.possible_agents):
            merged = dict(info_list[i]) if isinstance(info_list[i], dict) else {}
            if isinstance(info, dict):
                for k in ("propositions", "cost", "casualty_visible"):
                    if k in info:
                        merged[k] = info[k]
            term = terminated[agent] if isinstance(terminated, dict) else bool(terminated)
            trunc = truncated[agent] if isinstance(truncated, dict) else bool(truncated)
            merged["terminated"] = bool(term)
            merged["truncated"] = bool(trunc)
            info_list[i] = merged

        obs_n, share_n = self._pack_obs(obs)
        return obs_n, share_n, rew_list, cost_list, dones, info_list, self._get_avail_actions()

    def close(self):
        return self.env.close()
