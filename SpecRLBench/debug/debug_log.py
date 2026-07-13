"""Compact NDJSON logger for debug session 3376cb."""
import json
import time
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parents[1].parent / "debug-3376cb.log"
_SESSION = "3376cb"


def agent_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    # #region agent log
    entry = {
        "sessionId": _SESSION,
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": run_id,
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # #endregion
