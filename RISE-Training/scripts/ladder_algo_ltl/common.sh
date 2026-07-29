#!/usr/bin/env bash
# Shared settings for ladder_algo_ltl training scripts.
# Six envs: PointLTL{0,1,3}MASAR1[-WC]-v0 (plain + WC each).

RISE_TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"

# LEVEL VARIANT  (override with LEVEL=3 VARIANT=wc to run a single job)
JOBS=(
  "0 plain"
  "0 wc"
  "1 plain"
  "1 wc"
  "3 plain"
  "3 wc"
)

task_for() {
  local level="$1" variant="$2"
  if [[ "$variant" == "wc" ]]; then
    echo "PointLTL${level}MASAR1WC-v0"
  else
    echo "PointLTL${level}MASAR1-v0"
  fi
}

experiment_for() {
  local algo="$1" level="$2" variant="$3"
  if [[ "$variant" == "wc" ]]; then
    echo "${algo}_ltl_l${level}wc_s${SEED}"
  else
    echo "${algo}_ltl_l${level}_s${SEED}"
  fi
}

filtered_jobs() {
  if [[ -n "${LEVEL:-}" && -n "${VARIANT:-}" ]]; then
    echo "${LEVEL} ${VARIANT}"
    return
  fi
  printf '%s\n' "${JOBS[@]}"
}
