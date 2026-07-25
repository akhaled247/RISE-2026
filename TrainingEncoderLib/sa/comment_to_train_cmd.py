#!/usr/bin/env python3
"""Convert SpecRLBench SafePO SA hyperparam comments back to train shell commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from TrainingEncoderLib.sa.train_comment_lib import DecodeExtras, decode_comment
else:
    from .train_comment_lib import DecodeExtras, decode_comment


def _str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _read_input(path: str | None) -> str:
    if path is None:
        if sys.stdin.isatty():
            print("Paste comment line, then EOF (Ctrl-D / Ctrl-Z):", file=sys.stderr)
        return sys.stdin.read()
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Convert SpecRLBench SafePO SA hyperparam comments to train shell commands."
    )
    parser.add_argument(
        "comment",
        nargs="?",
        help="Comment line, file path, or omit to read stdin",
    )
    parser.add_argument("--task", dest="task_override", help="Fallback when comment lacks env* token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--write-terminal", type=_str2bool, default=False)
    parser.add_argument("--use-tensorboard", type=_str2bool, default=True)
    parser.add_argument("--parallel", type=_str2bool, default=True)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args(argv)

    extras = DecodeExtras(
        seed=args.seed,
        device=args.device,
        device_id=args.device_id,
        write_terminal=args.write_terminal,
        use_tensorboard=args.use_tensorboard,
        parallel=args.parallel,
        experiment=args.experiment,
        task_override=args.task_override,
    )

    try:
        raw = _read_input(args.comment)
        shell = decode_comment(raw, extras)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(shell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
