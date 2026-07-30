#!/usr/bin/env bash
# OG PPO training (og-GenZ-LTL-L0-PPO-s0): SA gate + optional MA deploy.
#
# Single-agent (MASAR1WC):
#   bash scripts/genz_eval_og_ppo.sh
#
# Multi-agent deploy (MASAR2WC):
#   MODE=ma bash scripts/genz_eval_og_ppo.sh
#
# First run — copy checkpoint from FileTransfer:
#   RESTORE_FROM=FileTransfer/singleagent_runs/GenZ-Maybe-PointLTL0MASAR1WC-v0/og-GenZ-LTL-L0-PPO-s0/0 \
#     bash scripts/genz_eval_og_ppo.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NAME="${NAME:-og-GenZ-LTL-L0-PPO-s0}"
export SEED="${SEED:-0}"
export RESTORE_FROM="${RESTORE_FROM:-$REPO_ROOT/FileTransfer/singleagent_runs/GenZ-Maybe-PointLTL0MASAR1WC-v0/og-GenZ-LTL-L0-PPO-s0/0}"
exec bash "$REPO_ROOT/scripts/genz_ppo_eval.sh"
