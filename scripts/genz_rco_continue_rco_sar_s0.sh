#!/usr/bin/env bash
# Resume GenZ-RCO-SAR-s0 (lr=3e-4, max_lag=5) from FileTransfer settings.
# Optional restore from FileTransfer copy:
#   RESTORE_FROM=~/RISE-2026/FileTransfer/singleagent_runs/GenZ-2300-Runs/GenZ-RCO-SAR-s0/0 \
#     ./scripts/genz_rco_continue_rco_sar_s0.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export NAME="${NAME:-GenZ-RCO-SAR-s0}"
export LR="${LR:-0.0003}"
export MAX_LAG="${MAX_LAG:-5}"

exec "${SCRIPT_DIR}/genz_rco_continue.sh" "$@"
