"""Encode/decode for SpecRLBench SafePO multi-agent train hyperparam comments.

Generated ``python train/...`` commands assume cwd is ``RISE-Training/``.
"""

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
from .yaml_defaults import (
    cli_flags_to_config_overrides,
    flatten_ma_yaml,
    load_ma_yaml,
    merge_ma_config,
    merge_yaml_gaps,
)

MA_SCRIPT_ALGO: dict[str, str] = {
    "mappo_lag_train_env.py": "mappo_lag",
    "ippo_lag_train_env.py": "ippo_lag",
    "mappo_train_env.py": "mappo",
    "happo_train_env.py": "happo",
    "ippo_train_env.py": "ippo",
}

MA_ALGO_SCRIPT: dict[str, str] = {v: k for k, v in MA_SCRIPT_ALGO.items()}

MA_ALGOS_LONGEST_FIRST = ("mappo_lag", "ippo_lag", "mappo", "happo", "ippo")

MA_ALGOS = frozenset({"mappo", "happo", "mappo_lag", "ippo", "ippo_lag"})
MA_LAG_ALGOS = frozenset({"mappo_lag", "ippo_lag"})

MA_CANONICAL_FLAGS = {
    "task",
    "seed",
    "total-steps",
    "num-envs",
    "cost-limit",
    "entropy-coef",
    "share-policy",
    "algo",
    "device",
    "device-id",
    "write-terminal",
    "use-tensorboard",
    "use-eval",
    "experiment",
    "log-dir",
    "model-dir",
}

MA_REQUIRED_CORE = (
    "task",
    "total-steps",
    "episode-length",
    "num-envs",
    "actor-lr",
    "critic-lr",
    "batch-size",
    "learning-iters",
    "target-kl",
    "gamma",
    "gae-lambda",
    "clip-ratio",
    "max-grad-norm",
    "data-chunk-length",
    "recurrent-N",
    "cost-limit",
)

MA_CONFIG_FIELD_MAP: dict[str, str] = {
    "env_name": "task",
    "task": "task",
    "num_env_steps": "total-steps",
    "n_rollout_threads": "num-envs",
    "episode_length": "episode-length",
    "actor_lr": "actor-lr",
    "critic_lr": "critic-lr",
    "num_mini_batch": "batch-size",
    "learning_iters": "learning-iters",
    "target_kl": "target-kl",
    "gamma": "gamma",
    "gae_lambda": "gae-lambda",
    "lam_c": "lam-c",
    "clip_param": "clip-ratio",
    "max_grad_norm": "max-grad-norm",
    "data_chunk_length": "data-chunk-length",
    "recurrent_N": "recurrent-N",
    "cost_limit": "cost-limit",
    "entropy_coef": "entropy-coef",
    "lr_end_factor": "lr_end_factor",
    "lagrangian_multiplier_init": "lagrangian-multiplier-init",
    "lagrangian_multiplier_lr": "lagrangian-multiplier-lr",
}


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


def normalize_ma_algo(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key == "mappolag":
        return "mappo_lag"
    if key not in MA_ALGOS:
        raise ValueError(f"unknown MA algorithm: {name!r}")
    return key


def _canonical_flag(token: str) -> str | None:
    if not token.startswith("--"):
        return None
    name = token[2:]
    aliases = {"env-id": "task"}
    if name in aliases:
        return aliases[name]
    if name in MA_CANONICAL_FLAGS:
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
        flags[key] = tokens[i]
        i += 1

    return flags


def detect_algo(text: str, flags: dict[str, str | list[str]]) -> str:
    """Detect algorithm from script basename or --algo."""
    normalized = normalize_shell(text)
    for script, algo in MA_SCRIPT_ALGO.items():
        if script in normalized:
            return algo

    algo_flag = flags.get("algo")
    if isinstance(algo_flag, str):
        return normalize_ma_algo(algo_flag)

    raise ValueError(
        "unknown or missing MA algorithm; expected train/{mappo,ippo,...}_train_env.py or --algo"
    )


def _require(flags: dict[str, str | list[str]], key: str) -> str:
    val = flags.get(key)
    if val is None or isinstance(val, list):
        raise ValueError(f"missing required flag --{key}")
    return val


def detect_ma_algo_from_config(config: dict) -> str:
    """Detect MA algorithm from ``algorithm_name``, ``log_dir``, or ``exp_name``."""
    raw = config.get("algorithm_name")
    if raw is not None:
        return normalize_ma_algo(str(raw))

    log_dir = str(config.get("log_dir", "")).replace("\\", "/")
    for algo in MA_ALGOS_LONGEST_FIRST:
        if f"/{algo}/" in log_dir:
            return algo
        if algo == "mappo_lag" and "/mappolag/" in log_dir:
            return algo

    exp_name = str(config.get("exp_name", ""))
    for algo in MA_ALGOS_LONGEST_FIRST:
        if f"-{algo}-" in exp_name:
            return algo

    raise ValueError(
        "cannot detect MA algorithm; expected algorithm_name, /mappo/ in log_dir, "
        "or -mappo- in exp_name"
    )


def ma_config_to_flags(config: dict) -> tuple[str, dict[str, str | list[str]]]:
    """Map SafePO MA config fields to encoder flag dict."""
    algo = detect_ma_algo_from_config(config)
    flags: dict[str, str | list[str]] = {}

    for cfg_key, flag_key in MA_CONFIG_FIELD_MAP.items():
        if cfg_key not in config:
            continue
        raw = config[cfg_key]
        if raw is None or raw == "":
            continue
        if isinstance(raw, (int, float, bool)):
            flags[flag_key] = scalar_to_flag_str(flag_key, raw)
        else:
            flags[flag_key] = str(raw)

    if "task" not in flags:
        raise ValueError("MA config missing env_name/task")

    return algo, flags


def _merged_config_for_algo(algo: str, overrides: dict) -> dict:
    yaml_base = load_ma_yaml(algo)
    yaml_flat = flatten_ma_yaml(yaml_base)
    return merge_ma_config(yaml_flat, overrides)


def encode_from_flags(algo: str, flags: dict[str, str | list[str]]) -> str:
    """Build quoted hyperparam comment from MA algorithm + flag dict."""
    for key in MA_REQUIRED_CORE:
        if key not in flags:
            raise ValueError(f"MA config missing required field for {key}")

    parts: list[str] = [
        algo,
        f"env{_require(flags, 'task')}",
        fmt_steps(_require(flags, "total-steps")),
        f"E{fmt_num(_require(flags, 'episode-length'))}",
        f"N{fmt_num(_require(flags, 'num-envs'))}",
        f"αa{fmt_num(_require(flags, 'actor-lr'), style='scientific')}",
        f"αc{fmt_num(_require(flags, 'critic-lr'), style='scientific')}",
        f"B{fmt_num(_require(flags, 'batch-size'))}",
        f"I{fmt_num(_require(flags, 'learning-iters'))}",
        f"DKL{fmt_num(_require(flags, 'target-kl'), style='decimal')}",
        f"γ{fmt_num(_require(flags, 'gamma'), style='decimal')}",
        f"λ{fmt_num(_require(flags, 'gae-lambda'), style='decimal')}",
    ]

    lam_c = flags.get("lam-c")
    if isinstance(lam_c, str):
        parts.append(f"λc{fmt_num(lam_c, style='decimal')}")

    parts.extend(
        [
            f"ε{fmt_num(_require(flags, 'clip-ratio'), style='decimal')}",
            f"g{fmt_num(_require(flags, 'max-grad-norm'))}",
            f"D{fmt_num(_require(flags, 'data-chunk-length'))}",
            f"R{fmt_num(_require(flags, 'recurrent-N'))}",
            f"cl{fmt_num(_require(flags, 'cost-limit'), style='decimal')}",
        ]
    )

    if algo in MA_LAG_ALGOS:
        parts.extend(
            [
                f"λi{fmt_num(_require(flags, 'lagrangian-multiplier-init'), style='decimal')}",
                f"λlr{fmt_num(_require(flags, 'lagrangian-multiplier-lr'), style='scientific')}",
            ]
        )

    entropy_coef = flags.get("entropy-coef")
    if isinstance(entropy_coef, str) and float(entropy_coef) != 0.0:
        parts.append(f"ec{fmt_num(entropy_coef, style='decimal')}")

    lre = flags.get("lr_end_factor")
    if isinstance(lre, str):
        parts.append(f"lre{fmt_num(lre, style='decimal')}")

    return "'" + "_".join(parts) + "'"


def encode_ma_from_cli_flags(algo: str, flags: dict[str, str | list[str]]) -> str:
    """Build partial MA comment from thin ``ma_cli`` shell flags (legacy)."""
    task = flags.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("MA command missing --task")

    parts: list[str] = [algo, f"env{task}"]

    total_steps = flags.get("total-steps")
    if isinstance(total_steps, str):
        parts.append(fmt_steps(total_steps))

    num_envs = flags.get("num-envs")
    if isinstance(num_envs, str):
        parts.append(f"N{fmt_num(num_envs)}")

    cost_limit = flags.get("cost-limit")
    if isinstance(cost_limit, str):
        parts.append(f"cl{fmt_num(cost_limit, style='decimal')}")

    entropy_coef = flags.get("entropy-coef")
    if isinstance(entropy_coef, str) and float(entropy_coef) != 0.0:
        parts.append(f"ec{fmt_num(entropy_coef, style='decimal')}")

    if len(parts) < 3:
        raise ValueError(
            "MA command needs at least --task and one of --total-steps, --num-envs, "
            "--cost-limit, --entropy-coef"
        )

    return "'" + "_".join(parts) + "'"


def encode_config(config: dict) -> str:
    """Build hyperparam comment from a SafePO MA ``config.json`` object."""
    algo = detect_ma_algo_from_config(config)
    yaml_base = load_ma_yaml(algo)
    yaml_flat = flatten_ma_yaml(yaml_base)
    merged = merge_yaml_gaps(yaml_flat, config)
    algo, flags = ma_config_to_flags(merged)
    return encode_from_flags(algo, flags)


def encode_command(text: str) -> str:
    """Build full hyperparam comment from a thin MA train shell command + YAML defaults."""
    flags = parse_flags(text)
    algo = detect_algo(text, flags)
    overrides = cli_flags_to_config_overrides(flags)
    merged = _merged_config_for_algo(algo, overrides)
    algo, encoded_flags = ma_config_to_flags(merged)
    return encode_from_flags(algo, encoded_flags)


def _strip_comment_prefix(comment: str) -> str:
    line = comment.strip()
    if line.startswith("#"):
        return line[1:].strip()
    if len(line) >= 2 and line[0] == line[-1] == "'":
        return line[1:-1]
    return line


def _detect_algo_from_tag(tag: str) -> tuple[str, str]:
    for algo in MA_ALGOS_LONGEST_FIRST:
        prefix = f"{algo}_"
        if tag.startswith(prefix):
            return algo, tag[len(prefix) :]
    raise ValueError(f"unknown MA algorithm in comment: {tag!r}")


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


def _apply_ma_tail_token(tok: str, parsed: dict[str, str | list[str]]) -> None:
    """Parse one MA suffix token (order-flexible)."""
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("lagrangian-multiplier-lr", ("λlr",)),
        ("lagrangian-multiplier-init", ("λi",)),
        ("lam-c", ("λc",)),
        ("entropy-coef", ("ec",)),
        ("lr_end_factor", ("lre",)),
        ("cost-limit", ("cl",)),
        ("recurrent-N", ("R",)),
        ("data-chunk-length", ("D",)),
        ("clip-ratio", ("ε",)),
        ("max-grad-norm", ("g",)),
    ]
    for key, prefixes in rules:
        for prefix in sorted(prefixes, key=len, reverse=True):
            if not tok.startswith(prefix):
                continue
            if key in parsed:
                raise ValueError(f"duplicate token for {key}: {tok!r}")
            parsed[key] = tok[len(prefix) :]
            return
    raise ValueError(f"unrecognized MA comment token: {tok!r}")


def _validate_ma_parsed(algo: str, parsed: dict[str, str | list[str]]) -> None:
    missing = [k for k in MA_REQUIRED_CORE if k not in parsed]
    if missing:
        raise ValueError(f"MA comment missing encoded fields: {', '.join(missing)}")

    if algo in MA_LAG_ALGOS:
        for key in ("lagrangian-multiplier-init", "lagrangian-multiplier-lr"):
            if key not in parsed:
                raise ValueError(f"MA comment missing {key} for {algo}")


def parse_comment(comment: str) -> tuple[str, dict[str, str | list[str]]]:
    """Parse comment tag into algorithm name and flag dict."""
    tag = _strip_comment_prefix(comment)
    algo, rest = _detect_algo_from_tag(tag)
    return _parse_ma_comment_body(algo, rest)


def _parse_ma_comment_body(algo: str, rest: str) -> tuple[str, dict[str, str | list[str]]]:
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
    parsed["episode-length"] = _take_prefixed(tokens, idx, "E", "episode-length")
    parsed["num-envs"] = _take_prefixed(tokens, idx, "N", "num-envs")
    parsed["actor-lr"] = _take_prefixed(tokens, idx, "αa", "actor-lr")
    parsed["critic-lr"] = _take_prefixed(tokens, idx, "αc", "critic-lr")
    parsed["batch-size"] = _take_prefixed(tokens, idx, "B", "batch-size")
    parsed["learning-iters"] = _take_prefixed(tokens, idx, "I", "learning-iters")
    parsed["target-kl"] = _take_prefixed(tokens, idx, "DKL", "target-kl")
    parsed["gamma"] = _take_prefixed(tokens, idx, "γ", "gamma")
    parsed["gae-lambda"] = _take_prefixed(tokens, idx, "λ", "gae-lambda")

    while idx[0] < len(tokens):
        _apply_ma_tail_token(tokens[idx[0]], parsed)
        idx[0] += 1

    _validate_ma_parsed(algo, parsed)
    return algo, parsed


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


def format_shell(
    algo: str,
    flags: dict[str, str | list[str]],
    extras: DecodeExtras,
) -> str:
    """Format decoded MA flags as a thin multiline ``ma_cli`` shell command."""
    script = MA_ALGO_SCRIPT.get(algo)
    if script is None:
        raise ValueError(f"unsupported MA algorithm: {algo}")

    task = flags.get("task")
    if isinstance(task, list) or not task:
        if extras.task_override:
            task = extras.task_override
        else:
            raise ValueError(
                "comment has no env* token; pass --task to supply --task"
            )

    lines: list[str] = [
        f"python train/{script} \\",
        (
            f"    --task {task} --seed {extras.seed} "
            f"--total-steps {flags['total-steps']} --num-envs {flags['num-envs']} \\"
        ),
        f"    --cost-limit {flags['cost-limit']} \\",
        f"    --device {extras.device} --device-id {extras.device_id} \\",
    ]

    write_line = (
        f"    --write-terminal {_bool_str(extras.write_terminal)} "
        f"--use-tensorboard {_bool_str(extras.use_tensorboard)}"
    )
    tail_parts: list[str] = []
    if extras.experiment:
        tail_parts.append(f"--experiment {extras.experiment}")
    entropy_coef = flags.get("entropy-coef")
    if isinstance(entropy_coef, str) and float(entropy_coef) != 0.0:
        tail_parts.append(f"--entropy-coef {entropy_coef}")

    if tail_parts:
        lines.append(write_line + " \\")
        lines.append("    " + " ".join(tail_parts))
    else:
        lines.append(write_line)
    return "\n".join(lines)


def decode_comment(comment: str, extras: DecodeExtras | None = None) -> str:
    """Parse hyperparam comment and return multiline train shell command."""
    extras = extras or DecodeExtras()
    algo, parsed = parse_comment(comment)
    return format_shell(algo, parsed, extras)
