"""Numeric and shell formatting helpers."""

from __future__ import annotations

import re

STEPS_SUFFIX_RE = re.compile(r"^(\d+)([MK])?$")


def normalize_shell(text: str) -> str:
    """Join line continuations and drop standalone comment lines."""
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and not stripped.startswith("#!"):
            continue
        lines.append(stripped)

    joined = " ".join(lines)
    joined = re.sub(r"\\\s+", " ", joined)
    return joined.strip()


def _format_scientific(f: float) -> str:
    mantissa, exp = f"{f:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".") or "0"
    exp_i = int(exp)
    return f"{mantissa}e{exp_i}"


def fmt_num(value: str, *, style: str = "auto") -> str:
    """Compact numeric string for hyperparam tags."""
    s = value.strip()
    if not s:
        raise ValueError("empty numeric value")

    if "e" in s.lower():
        return _format_scientific(float(s))

    f = float(s)
    if f == 0.0:
        return "0"

    if f == int(f):
        if "." in s and s.endswith(".0"):
            return s
        return str(int(f))

    if style == "scientific" or (style == "auto" and abs(f) < 0.001):
        return _format_scientific(f)

    out = f"{f:f}".rstrip("0").rstrip(".")
    return out if out else "0"


def fmt_steps(value: str) -> str:
    """Format total-steps as 5M / 400K / raw."""
    n = int(float(value))
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def parse_steps(token: str) -> str:
    """Inverse of ``fmt_steps``."""
    m = STEPS_SUFFIX_RE.match(token)
    if not m:
        raise ValueError(f"invalid total-steps token: {token!r}")
    n = int(m.group(1))
    suffix = m.group(2)
    if suffix == "M":
        return str(n * 1_000_000)
    if suffix == "K":
        return str(n * 1_000)
    return str(n)


SCIENTIFIC_FLAG_KEYS = frozenset(
    {"actor-lr", "critic-lr", "lagrangian-multiplier-lr"}
)

WHOLE_NUMBER_FLAGS = frozenset(
    {
        "batch-size",
        "learning-iters",
        "num-envs",
        "max-grad-norm",
        "steps-per-epoch",
        "episode-length",
        "data-chunk-length",
        "recurrent-N",
    }
)


def scalar_to_flag_str(flag_key: str, value: int | float | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if (
        isinstance(value, float)
        and value == int(value)
        and flag_key in WHOLE_NUMBER_FLAGS
    ):
        value = int(value)
    if flag_key in SCIENTIFIC_FLAG_KEYS:
        return fmt_num(str(value), style="scientific")
    return fmt_num(str(value), style="decimal")
