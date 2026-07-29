#!/usr/bin/env bash
# Eval latest L0/L1 runs for PPO (plain + WC).
# Example run dir:
#   ./_training_logs/safepo/PointLTL0MASAR1-v0/ppo/seed-000-2026-07-28-21-13-08
# Run:  bash scripts/ladder_algo_ltl/eval_ppo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${RISE_TRAINING_ROOT}"

ALGO="ppo"
RESULTS="$(results_file_for "$ALGO")"
mkdir -p "$LOG_ROOT"
: > "$RESULTS"

while IFS= read -r row; do
  # shellcheck disable=SC2086
  eval_latest_for_job "$ALGO" $row
done < <(filtered_eval_jobs)

echo "PPO L0/L1 eval done. Results → $RESULTS"
