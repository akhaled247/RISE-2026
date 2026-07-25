"""NDJSON debug logger for agent debug session 5d8186."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_SESSION = "5d8186"
_LOG_NAME = "debug-5d8186.log"


def _log_path() -> Path:
    env = os.environ.get("RISE_DEBUG_LOG")
    if env:
        return Path(env)
    # SpecRLBench/backends/safepo/debug_log.py -> RISE-2026/
    return Path(__file__).resolve().parents[3] / _LOG_NAME


def agent_dbg(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": _SESSION,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
        }
        with _log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion
