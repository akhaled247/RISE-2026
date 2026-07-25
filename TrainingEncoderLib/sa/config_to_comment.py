#!/usr/bin/env python3
"""Convert SafePO SA run config.json files to compact hyperparam comments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from TrainingEncoderLib.sa.train_comment_lib import encode_config
else:
    from .train_comment_lib import encode_config

SCRIPT_DIR = Path(__file__).resolve().parent
SMOKE_CONFIGS = (
    SCRIPT_DIR / "smoke_configs" / "ppo_l4wc_seed0.json",
    SCRIPT_DIR / "smoke_configs" / "ppo_lag_l4wc_seed0.json",
)

_JSON_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_JSON_LITERAL = re.compile(r"true|false|null", re.IGNORECASE)


def _fix_powershell_json(text: str) -> str:
    """Re-quote keys/values stripped by PowerShell native argument parsing."""
    s = text.strip()
    s = re.sub(
        r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:",
        r'\1"\2":',
        s,
    )

    def _quote_value(match: re.Match[str]) -> str:
        prefix, val, suffix = match.group(1), match.group(2).strip(), match.group(3)
        if _JSON_LITERAL.fullmatch(val):
            return match.group(0)
        if _JSON_NUMBER.fullmatch(val):
            return match.group(0)
        if val.startswith("[") or val.startswith("{"):
            return match.group(0)
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'{prefix}"{escaped}"{suffix}'

    s = re.sub(
        r"(:\s*)([^,\[\]{}\n\r]+?)(\s*[,}\]])",
        _quote_value,
        s,
        flags=re.MULTILINE,
    )
    return s


def _loads_config_json(text: str) -> dict:
    """Parse strict JSON; fall back to PowerShell-mangled pseudo-JSON."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty JSON input")

    errors: list[str] = []
    for candidate in (stripped, _fix_powershell_json(stripped)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(data, dict):
            raise ValueError("expected JSON object at top level")
        return data

    raise json.JSONDecodeError(
        f"invalid JSON (tried strict + PowerShell repair): {errors[-1]}",
        stripped,
        0,
    )


def _load_config(path: Path) -> dict:
    return _loads_config_json(path.read_text(encoding="utf-8"))


def _parse_config_input(arg: str) -> dict:
    """Load config from inline JSON text or a file path."""
    stripped = arg.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return _loads_config_json(stripped)

    p = Path(arg)
    if p.is_file():
        return _load_config(p)

    try:
        return _loads_config_json(arg)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a file and not valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Convert SafePO SA config.json files to hyperparam comments."
    )
    parser.add_argument(
        "config",
        nargs="*",
        help="Path(s) to config.json, inline JSON, '-' for stdin, or omit with --smoke",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run bundled smoke configs in sa/smoke_configs/",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        inputs = [str(p) for p in SMOKE_CONFIGS]
    elif args.config == ["-"]:
        try:
            print(encode_config(_loads_config_json(sys.stdin.read())))
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    elif args.config:
        inputs = list(args.config)
    else:
        if sys.stdin.isatty():
            print(
                "Provide config.json path(s), inline JSON, pipe JSON to stdin, "
                "or use --smoke",
                file=sys.stderr,
            )
            return 1
        try:
            print(encode_config(_loads_config_json(sys.stdin.read())))
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    exit_code = 0
    for i, raw in enumerate(inputs):
        if i:
            print()
        label = raw.strip()[:40] + "..." if raw.strip().startswith("{") else Path(raw).name
        try:
            comment = encode_config(_parse_config_input(raw))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"error: {label}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        if len(inputs) > 1:
            print(f"# {label}")
        print(comment)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
