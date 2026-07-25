"""Runtime patches for vendored SafePO MA vec-env (spawn + worker errors + farama)."""

from __future__ import annotations

import multiprocessing as mp
import sys

_PATCHED = False


def _install_farama_filter() -> None:
    from backends.safepo import farama_filter

    sys.modules["safepo.common.farama_filter"] = farama_filter  # type: ignore[assignment]
    farama_filter.silence_farama_adroit_spam()


def _ensure_spawn_start_method() -> None:
    if mp.get_start_method(allow_none=True) is not None:
        return
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def _patch_wrappers() -> None:
    import numpy as np
    import torch
    from safepo.common.wrappers import CloudpickleWrapper, ShareSubprocVecEnv, ShareVecEnv

    def _cloudpickle_setstate(self, ob):
        import cloudpickle

        self.x = cloudpickle.loads(ob)

    CloudpickleWrapper.__setstate__ = _cloudpickle_setstate  # type: ignore[method-assign]

    def shareworker(remote, parent_remote, env_fn_wrapper):
        parent_remote.close()
        try:
            env = env_fn_wrapper.x()
        except BaseException:
            import traceback

            try:
                remote.send(("init_error", traceback.format_exc()))
            except Exception:
                pass
            raise
        remote.send(("ready", None))
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                if torch.is_tensor(data):
                    data = data.detach().cpu()
                ob, s_ob, reward, cost, done, info, available_actions = env.step(data)
                if "bool" in done.__class__.__name__:
                    if done:
                        ob, s_ob, available_actions = env.reset()
                else:
                    if np.all(done):
                        ob, s_ob, available_actions = env.reset()
                remote.send((ob, s_ob, reward, cost, done, info, available_actions))
            elif cmd == "reset":
                ob, s_ob, available_actions = env.reset()
                remote.send((ob, s_ob, available_actions))
            elif cmd == "reset_task":
                ob = env.reset_task()
                remote.send(ob)
            elif cmd == "render":
                if data == "rgb_array":
                    fr = env.render(mode=data)
                    remote.send(fr)
                elif data == "human":
                    env.render(mode=data)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send(
                    (env.observation_spaces, env.share_observation_spaces, env.action_spaces)
                )
            elif cmd == "render_vulnerability":
                fr = env.render_vulnerability(data)
                remote.send(fr)
            elif cmd == "get_num_agents":
                remote.send(env.num_agents)
            else:
                raise NotImplementedError

    import safepo.common.wrappers as wrappers_mod

    wrappers_mod.shareworker = shareworker

    def _share_subproc_init(self, env_fns, device=torch.device("cpu")):
        self.waiting = False
        self.closed = False
        self.device = device
        nenvs = len(env_fns)
        _ensure_spawn_start_method()
        ctx = torch.multiprocessing.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [
            ctx.Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        for i, remote in enumerate(self.remotes):
            try:
                msg = remote.recv()
            except EOFError as exc:
                exit_codes = [p.exitcode for p in self.ps]
                raise RuntimeError(
                    f"Subproc env worker {i} died during init (exit codes={exit_codes}). "
                    "Re-run with --num-envs 1 to surface the traceback, or check stderr."
                ) from exc
            if isinstance(msg, tuple) and msg[0] == "init_error":
                for p in self.ps:
                    p.join(timeout=1)
                raise RuntimeError(f"Subproc env worker {i} failed during init:\n{msg[1]}")
        self.remotes[0].send(("get_num_agents", None))
        self.num_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    ShareSubprocVecEnv.__init__ = _share_subproc_init  # type: ignore[method-assign]

    def _step_async(self, actions):
        env_actions = torch.transpose(torch.stack(actions), 1, 0)
        env_actions = env_actions.detach().cpu()
        for remote, action in zip(self.remotes, env_actions):
            remote.send(("step", action))
        self.waiting = True

    ShareSubprocVecEnv.step_async = _step_async  # type: ignore[method-assign]


def apply_ma_safepo_patches() -> None:
    """Idempotent: farama filter + spawn vec-env patches before SafePO MA train."""
    global _PATCHED
    if _PATCHED:
        return
    _install_farama_filter()
    _ensure_spawn_start_method()
    _patch_wrappers()
    _PATCHED = True
