"""Encode/decode for SpecRLBench SafePO single-agent train hyperparam comments."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from ..common.fmt import (
    fmt_num,
    fmt_steps,
    parse_steps,
    normalize_shell,
    scalar_to_flag_str,
)

SCRIPT_ALGO: dict[str, str] = {
    "ppo_train_env.py": "ppo",
    "trpo_train_env.py": "trpo",
    "ppo_lag_train_env.py": "ppo_lag",
    "trpo_lag_train_env.py": "trpo_lag",
    "cpo_train_env.py": "cpo",
}

ALGO_SCRIPT: dict[str, str] = {v: k for k, v in SCRIPT_ALGO.items()}

ALGOS_LONGEST_FIRST = ("ppo_lag", "trpo_lag", "ppo", "trpo", "cpo")

PPO_FAMILY = frozenset({"ppo", "ppo_lag"})
LAG_ALGOS = frozenset({"ppo_lag", "trpo_lag"})
CPO_ALGOS = frozenset({"cpo"})

CANONICAL_FLAGS = {
    "task",
    "seed",
    "total-steps",
    "num-envs",
    "steps-per-epoch",
    "actor-lr",
    "critic-lr",
    "batch-size",
    "learning-iters",
    "target-kl",
    "gamma",
    "lam",
    "lam-c",
    "clip-ratio",
    "max-grad-norm",
    "hidden-sizes",
    "lr_end_factor",
    "cost-limit",
    "lagrangian-multiplier-init",
    "lagrangian-multiplier-lr",
    "ent-coef",
    "algo",
}

REQUIRED_CORE = (
    "task",
    "total-steps",
    "steps-per-epoch",
    "num-envs",
    "actor-lr",
    "critic-lr",
    "batch-size",
    "learning-iters",
    "target-kl",
    "gamma",
    "lam",
    "lam-c",
    "max-grad-norm",
    "hidden-sizes",
)


@dataclass
class DecodeExtras:
    seed: int = 0
    device: str = "cuda"
    device_id: int = 1
    write_terminal: bool = False
    use_tensorboard: bool = True
    parallel: bool = True
    experiment: str | None = None
    task_override: str | None = None


def _canonical_flag(token: str) -> str | None:
    if not token.startswith("--"):
        return None
    name = token[2:]
    aliases = {
        "env-id": "task",
        "lr-end-factor": "lr_end_factor",
        "lr_end_factor": "lr_end_factor",
    }
    if name in aliases:
        return aliases[name]
    if name in CANONICAL_FLAGS:
        return name
    return None


def parse_flags(text: str) -> dict[str, str | list[str]]:
    """Parse ``python train/...py --flag value ...`` into a flag dict."""
    normalized = normalize_shell(text)
    if not normalized:
        raise ValueError("empty command after normalization")

    tokens = shlex.split(normalized)
    flags: dict[str, str | list[str]] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue

        key = _canonical_flag(tok)
        if key is None:
            i += 1
            continue

        if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
            raise ValueError(f"missing value for {tok}")
        i += 1

        if key == "hidden-sizes":
            sizes: list[str] = []
            while i < len(tokens) and not tokens[i].startswith("--"):
                sizes.append(tokens[i])
                i += 1
            flags[key] = sizes
            continue

        flags[key] = tokens[i]
        i += 1

    return flags


def detect_algo(text: str, flags: dict[str, str | list[str]]) -> str:
    """Detect algorithm from script basename or --algo."""
    normalized = normalize_shell(text)
    for script, algo in SCRIPT_ALGO.items():
        if script in normalized:
            return algo

    algo_flag = flags.get("algo")
    if isinstance(algo_flag, str) and algo_flag in SCRIPT_ALGO.values():
        return algo_flag

    raise ValueError(
        "unknown or missing algorithm; expected train/{ppo,trpo,...}_train_env.py or --algo"
    )


def _require(flags: dict[str, str | list[str]], key: str) -> str:
    val = flags.get(key)
    if val is None or isinstance(val, list):
        raise ValueError(f"missing required flag --{key}")
    return val


CONFIG_FIELD_MAP: dict[str, str] = {
    "task": "task",
    "total_steps": "total-steps",
    "steps_per_epoch": "steps-per-epoch",
    "num_envs": "num-envs",
    "actor_lr": "actor-lr",
    "critic_lr": "critic-lr",
    "batch_size": "batch-size",
    "learning_iters": "learning-iters",
    "target_kl": "target-kl",
    "gamma": "gamma",
    "lam": "lam",
    "lam_c": "lam-c",
    "clip_ratio": "clip-ratio",
    "max_grad_norm": "max-grad-norm",
    "hidden_sizes": "hidden-sizes",
    "lr_end_factor": "lr_end_factor",
    "cost_limit": "cost-limit",
    "lagrangian_multiplier_init": "lagrangian-multiplier-init",
    "lagrangian_multiplier_lr": "lagrangian-multiplier-lr",
    "ent_coef": "ent-coef",
}


def detect_algo_from_config(config: dict) -> str:
    """Detect algorithm from SafePO ``config.json`` ``log_dir`` or ``exp_name``."""
    log_dir = str(config.get("log_dir", "")).replace("\\", "/")
    for algo in ALGOS_LONGEST_FIRST:
        if f"/{algo}/" in log_dir:
            return algo

    exp_name = str(config.get("exp_name", ""))
    for algo in ALGOS_LONGEST_FIRST:
        needle = f"-{algo}-"
        if needle in exp_name:
            return algo

    raise ValueError(
        "cannot detect algorithm; expected /ppo_lag/ or /ppo/ etc. in log_dir, "
        "or -ppo_lag- in exp_name"
    )


def config_to_flags(config: dict) -> tuple[str, dict[str, str | list[str]]]:
    """Map SafePO SA config.json fields to encoder flag dict."""
    algo = detect_algo_from_config(config)
    flags: dict[str, str | list[str]] = {}

    for cfg_key, flag_key in CONFIG_FIELD_MAP.items():
        if cfg_key not in config:
            continue
        raw = config[cfg_key]
        if flag_key == "hidden-sizes":
            if not isinstance(raw, list) or not raw:
                raise ValueError("hidden_sizes must be a non-empty list")
            flags[flag_key] = [str(int(x)) for x in raw]
        elif isinstance(raw, (int, float, bool)):
            flags[flag_key] = scalar_to_flag_str(flag_key, raw)
        else:
            flags[flag_key] = str(raw)

    return algo, flags


def encode_from_flags(algo: str, flags: dict[str, str | list[str]]) -> str:
    """Build quoted hyperparam comment from SA algorithm + flag dict."""
    for key in REQUIRED_CORE:
        if key not in flags:
            raise ValueError(f"config missing required field for --{key}")

    hidden = flags["hidden-sizes"]
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("config missing hidden_sizes")

    parts: list[str] = [
        algo,
        f"env{_require(flags, 'task')}",
        fmt_steps(_require(flags, "total-steps")),
        f"T{fmt_num(_require(flags, 'steps-per-epoch'))}",
        f"N{fmt_num(_require(flags, 'num-envs'))}",
        f"αa{fmt_num(_require(flags, 'actor-lr'), style='scientific')}",
        f"αc{fmt_num(_require(flags, 'critic-lr'), style='scientific')}",
        f"B{fmt_num(_require(flags, 'batch-size'))}",
        f"I{fmt_num(_require(flags, 'learning-iters'))}",
        f"DKL{fmt_num(_require(flags, 'target-kl'), style='decimal')}",
        f"γ{fmt_num(_require(flags, 'gamma'), style='decimal')}",
        f"λ{fmt_num(_require(flags, 'lam'), style='decimal')}",
        f"λc{fmt_num(_require(flags, 'lam-c'), style='decimal')}",
    ]

    if algo in CPO_ALGOS:
        parts.append(f"Cl{fmt_num(_require(flags, 'cost-limit'), style='decimal')}")

    if algo in PPO_FAMILY:
        parts.append(f"ε{fmt_num(_require(flags, 'clip-ratio'), style='decimal')}")

    parts.append(f"g{fmt_num(_require(flags, 'max-grad-norm'))}")
    parts.append("h" + "×".join(fmt_num(x) for x in hidden))

    if algo in PPO_FAMILY:
        lre = flags.get("lr_end_factor")
        if lre is None:
            raise ValueError("config missing lr_end_factor for PPO-family algos")
        parts.append(f"lre{fmt_num(str(lre), style='decimal')}")

    if algo in LAG_ALGOS:
        parts.extend(
            [
                f"cl{fmt_num(_require(flags, 'cost-limit'), style='decimal')}",
                f"λi{fmt_num(_require(flags, 'lagrangian-multiplier-init'), style='decimal')}",
                f"λlr{fmt_num(_require(flags, 'lagrangian-multiplier-lr'), style='scientific')}",
            ]
        )

    ent_coef = flags.get("ent-coef")
    if isinstance(ent_coef, str) and float(ent_coef) != 0.0:
        parts.append(f"ec{fmt_num(ent_coef, style='decimal')}")

    return "'" + "_".join(parts) + "'"


def encode_config(config: dict) -> str:
    """Build hyperparam comment from a SafePO SA ``config.json`` object."""
    algo, flags = config_to_flags(config)
    return encode_from_flags(algo, flags)


def encode_command(text: str) -> str:
    """Build hyperparam comment from a train shell command."""
    flags = parse_flags(text)
    algo = detect_algo(text, flags)
    return encode_from_flags(algo, flags)


def _strip_comment_prefix(comment: str) -> str:
    line = comment.strip()
    if line.startswith("#"):
        return line[1:].strip()
    if len(line) >= 2 and line[0] == line[-1] == "'":
        return line[1:-1]
    return line


def _detect_algo_from_tag(tag: str) -> tuple[str, str]:
    for algo in ALGOS_LONGEST_FIRST:
        prefix = f"{algo}_"
        if tag.startswith(prefix):
            return algo, tag[len(prefix) :]
    raise ValueError(f"unknown algorithm in comment: {tag!r}")


def _take(tokens: list[str], idx: list[int]) -> str:
    if idx[0] >= len(tokens):
        raise ValueError("unexpected end of comment")
    tok = tokens[idx[0]]
    idx[0] += 1
    return tok


def _take_prefixed(tokens: list[str], idx: list[int], prefix: str, key: str) -> str:
    tok = _take(tokens, idx)
    if not tok.startswith(prefix):
        raise ValueError(f"expected {prefix}* token for {key}, got {tok!r}")
    return tok[len(prefix) :]


def _apply_tail_token(tok: str, parsed: dict[str, str | list[str]]) -> None:
    """Parse one SA suffix token (order-flexible; supports legacy + vault aliases)."""
    rules: list[tuple[str, tuple[str, ...], str]] = [
        ("lagrangian-multiplier-lr", ("λlr", "αμ"), "value"),
        ("lagrangian-multiplier-init", ("λi", "μ0"), "value"),
        ("cost-limit", ("cl", "Cl"), "value"),
        ("ent-coef", ("ec",), "value"),
        ("lr_end_factor", ("lre",), "value"),
        ("clip-ratio", ("ε",), "value"),
        ("max-grad-norm", ("g",), "value"),
        ("hidden-sizes", ("h",), "hidden"),
    ]
    for key, prefixes, kind in rules:
        for prefix in sorted(prefixes, key=len, reverse=True):
            if not tok.startswith(prefix):
                continue
            if key in parsed:
                raise ValueError(f"duplicate token for --{key}: {tok!r}")
            val = tok[len(prefix) :]
            if kind == "hidden":
                parsed[key] = val.replace("x", "×").split("×")
            else:
                parsed[key] = val
            return
    raise ValueError(f"unrecognized comment token: {tok!r}")


def _validate_parsed(algo: str, parsed: dict[str, str | list[str]]) -> None:
    missing = [k for k in REQUIRED_CORE if k not in parsed]
    if missing:
        raise ValueError(f"comment missing encoded fields: {', '.join(missing)}")

    if algo in LAG_ALGOS:
        for key in ("cost-limit", "lagrangian-multiplier-init", "lagrangian-multiplier-lr"):
            if key not in parsed:
                raise ValueError(f"comment missing --{key} for {algo}")
    elif algo in CPO_ALGOS and "cost-limit" not in parsed:
        raise ValueError(f"comment missing --cost-limit for {algo}")

    if algo in PPO_FAMILY:
        for key in ("clip-ratio", "lr_end_factor"):
            if key not in parsed:
                raise ValueError(f"comment missing --{key} for {algo}")
    else:
        for key in ("clip-ratio", "lr_end_factor"):
            if key in parsed:
                raise ValueError(f"unexpected --{key} token for {algo}")

    hidden = parsed.get("hidden-sizes")
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("comment missing --hidden-sizes")


def parse_comment(comment: str) -> tuple[str, dict[str, str | list[str]]]:
    """Parse comment tag into algorithm name and flag dict."""
    tag = _strip_comment_prefix(comment)
    algo, rest = _detect_algo_from_tag(tag)
    return _parse_sa_comment_body(algo, rest)


def _parse_sa_comment_body(algo: str, rest: str) -> tuple[str, dict[str, str | list[str]]]:
    tokens = [t for t in rest.split("_") if t]
    idx = [0]
    parsed: dict[str, str | list[str]] = {}

    env_tok = _take(tokens, idx)
    if not env_tok.startswith("env"):
        raise ValueError(
            f"expected env* task token, got {env_tok!r}; "
            "old comments without env require --task on decode"
        )
    parsed["task"] = env_tok[3:]

    parsed["total-steps"] = parse_steps(_take(tokens, idx))
    parsed["steps-per-epoch"] = _take_prefixed(tokens, idx, "T", "steps-per-epoch")
    parsed["num-envs"] = _take_prefixed(tokens, idx, "N", "num-envs")
    parsed["actor-lr"] = _take_prefixed(tokens, idx, "αa", "actor-lr")
    parsed["critic-lr"] = _take_prefixed(tokens, idx, "αc", "critic-lr")
    parsed["batch-size"] = _take_prefixed(tokens, idx, "B", "batch-size")
    parsed["learning-iters"] = _take_prefixed(tokens, idx, "I", "learning-iters")
    parsed["target-kl"] = _take_prefixed(tokens, idx, "DKL", "target-kl")
    parsed["gamma"] = _take_prefixed(tokens, idx, "γ", "gamma")
    parsed["lam"] = _take_prefixed(tokens, idx, "λ", "lam")
    parsed["lam-c"] = _take_prefixed(tokens, idx, "λc", "lam-c")

    while idx[0] < len(tokens):
        _apply_tail_token(tokens[idx[0]], parsed)
        idx[0] += 1

    _validate_parsed(algo, parsed)
    return algo, parsed


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


def format_shell(
    algo: str,
    flags: dict[str, str | list[str]],
    extras: DecodeExtras,
) -> str:
    """Format decoded flags as a multiline SafePO train shell command."""
    script = ALGO_SCRIPT.get(algo)
    if script is None:
        raise ValueError(f"unsupported algorithm: {algo}")

    task = flags.get("task")
    if isinstance(task, list) or not task:
        if extras.task_override:
            task = extras.task_override
        else:
            raise ValueError(
                "comment has no env* token; pass --task to supply --task"
            )

    hidden = flags["hidden-sizes"]
    assert isinstance(hidden, list)

    lines: list[str] = [
        f"python train/{script} \\",
        f"    --task {task} --seed {extras.seed} \\",
        (
            f"    --total-steps {flags['total-steps']} --num-envs {flags['num-envs']} "
            f"--steps-per-epoch {flags['steps-per-epoch']} \\"
        ),
    ]

    if algo in LAG_ALGOS:
        lines.append(f"    --cost-limit {flags['cost-limit']} \\")
        lines.append(
            "    --lagrangian-multiplier-init "
            f"{flags['lagrangian-multiplier-init']} "
            f"--lagrangian-multiplier-lr {flags['lagrangian-multiplier-lr']} \\"
        )
    elif algo in CPO_ALGOS:
        lines.append(f"    --cost-limit {flags['cost-limit']} \\")

    lines.extend(
        [
            (
                f"    --actor-lr {flags['actor-lr']} --critic-lr {flags['critic-lr']} \\"
            ),
            (
                f"    --batch-size {flags['batch-size']} "
                f"--learning-iters {flags['learning-iters']} \\"
            ),
            (
                f"    --target-kl {flags['target-kl']} --gamma {flags['gamma']} "
                f"--lam {flags['lam']} --lam-c {flags['lam-c']} \\"
            ),
        ]
    )

    if algo in PPO_FAMILY:
        clip_line = f"    --clip-ratio {flags['clip-ratio']} --max-grad-norm {flags['max-grad-norm']}"
        if len(hidden) == 2:
            clip_line += f" --hidden-sizes {hidden[0]} {hidden[1]} \\"
        else:
            clip_line += f" --hidden-sizes {' '.join(hidden)} \\"
        lines.append(clip_line)
    else:
        grad_line = f"    --max-grad-norm {flags['max-grad-norm']}"
        if len(hidden) == 2:
            grad_line += f" --hidden-sizes {hidden[0]} {hidden[1]} \\"
        else:
            grad_line += f" --hidden-sizes {' '.join(hidden)} \\"
        lines.append(grad_line)

    lines.append(f"    --device {extras.device} --device-id {extras.device_id} \\")

    lines.append(
        f"    --write-terminal {_bool_str(extras.write_terminal)} "
        f"--use-tensorboard {_bool_str(extras.use_tensorboard)} \\"
    )

    if extras.experiment:
        parallel_prefix = (
            f"    --experiment {extras.experiment} "
            f"--parallel {_bool_str(extras.parallel)}"
        )
    else:
        parallel_prefix = f"    --parallel {_bool_str(extras.parallel)}"

    if algo in PPO_FAMILY:
        tail = f"{parallel_prefix} --lr_end_factor {flags['lr_end_factor']}"
    else:
        tail = parallel_prefix

    ent_coef = flags.get("ent-coef")
    if isinstance(ent_coef, str):
        tail += f" --ent-coef {ent_coef}"
    lines.append(tail)

    return "\n".join(lines)


def decode_comment(comment: str, extras: DecodeExtras | None = None) -> str:
    """Parse hyperparam comment and return multiline train shell command."""
    extras = extras or DecodeExtras()
    algo, parsed = parse_comment(comment)
    return format_shell(algo, parsed, extras)
