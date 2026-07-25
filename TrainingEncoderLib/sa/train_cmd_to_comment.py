#!/usr/bin/env python3
"""Convert SpecRLBench SafePO SA train shell commands to compact hyperparam comments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from TrainingEncoderLib.sa.train_comment_lib import encode_command
else:
    from .train_comment_lib import encode_command


def _read_input(path: str | None) -> str:
    if path is None:
        if sys.stdin.isatty():
            print("Paste train command, then EOF (Ctrl-D / Ctrl-Z):", file=sys.stderr)
        return sys.stdin.read()
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Convert SpecRLBench SafePO SA train shell commands to hyperparam comments."
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Train command string, file path, or omit to read stdin",
    )
    args = parser.parse_args(argv)

    try:
        raw = _read_input(args.command)
        comment = encode_command(raw)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
