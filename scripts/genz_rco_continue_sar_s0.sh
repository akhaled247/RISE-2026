#!/usr/bin/env bash
# Resume GenZ-SAR-s0 (successful RCO run; lr=3e-4, max_lag=5).
# Target: experiments/rco/PointLTL0MASAR1WC-v0/GenZ-SAR-s0/0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export NAME="${NAME:-GenZ-SAR-s0}"
export LR="${LR:-0.0003}"
export MAX_LAG="${MAX_LAG:-5}"

exec "${SCRIPT_DIR}/genz_rco_continue.sh" "$@"
