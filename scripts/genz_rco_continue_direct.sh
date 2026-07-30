#!/usr/bin/env bash
# Resume GenZ-LTL-RCO-Direct (lr=1e-4, max_lag=10) from FileTransfer settings.
# Target: experiments/rco/PointLTL0MASAR1WC-v0/GenZ-LTL-RCO-Direct/0
#
# Optional restore from FileTransfer copy:
#   RESTORE_FROM=~/RISE-2026/FileTransfer/singleagent_runs/GenZ-2300-Runs/GenZ-LTL-RCO-Direct/0 \
#     ./scripts/genz_rco_continue_direct.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export NAME="${NAME:-GenZ-LTL-RCO-Direct}"
export LR="${LR:-0.0001}"
export MAX_LAG="${MAX_LAG:-10}"

exec "${SCRIPT_DIR}/genz_rco_continue.sh" "$@"
