#!/usr/bin/env python3
"""Repo-root wrapper for GenZ log analysis skill script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / ".cursor"
    / "skills"
    / "genz-log-analysis"
    / "scripts"
    / "analyze_genz_logs.py"
)


def main() -> int:
    if not SCRIPT.is_file():
        print(f"Missing analyzer: {SCRIPT}", file=sys.stderr)
        return 1
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
