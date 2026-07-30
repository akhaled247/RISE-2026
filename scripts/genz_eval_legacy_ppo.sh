#!/usr/bin/env bash
# Legacy PPO training (GenZ-SAR-s0): SA gate + optional MA deploy.
#
# Single-agent (MASAR1WC):
#   bash scripts/genz_eval_legacy_ppo.sh
#
# Multi-agent deploy (MASAR2WC):
#   MODE=ma bash scripts/genz_eval_legacy_ppo.sh
#
# First run — copy checkpoint from FileTransfer (adjust path if needed):
#   RESTORE_FROM=FileTransfer/singleagent_runs/GenZ-Maybe-PointLTL0MASAR1WC-v0/GenZ-SAR-s0/0 \
#     bash scripts/genz_eval_legacy_ppo.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NAME="${NAME:-GenZ-SAR-s0}"
export SEED="${SEED:-0}"
export RESTORE_FROM="${RESTORE_FROM:-$REPO_ROOT/FileTransfer/singleagent_runs/GenZ-Maybe-PointLTL0MASAR1WC-v0/GenZ-SAR-s0/0}"
exec bash "$REPO_ROOT/scripts/genz_ppo_eval.sh"
