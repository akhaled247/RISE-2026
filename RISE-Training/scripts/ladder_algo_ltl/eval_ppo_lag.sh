#!/usr/bin/env bash
# Eval latest L0/L1 runs for PPO-Lag (plain + WC).
# Run:  bash scripts/ladder_algo_ltl/eval_ppo_lag.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${RISE_TRAINING_ROOT}"

ALGO="ppo_lag"
RESULTS="$(results_file_for "$ALGO")"
mkdir -p "$LOG_ROOT"
: > "$RESULTS"

while IFS= read -r row; do
  # shellcheck disable=SC2086
  eval_latest_for_job "$ALGO" $row
done < <(filtered_eval_jobs)

echo "PPO-Lag L0/L1 eval done. Results → $RESULTS"
