#!/usr/bin/env bash
# SA + MA deploy eval for every seed under ppo/ and ppo_lag/ for MASAR1WC.
#
# Usage:
#   ./scripts/sar_safepo_eval_all_seeds.sh
#   EVAL_EPISODES=50 DEVICE=cuda ./scripts/sar_safepo_eval_all_seeds.sh
#   ALGOS=ppo_lag ./scripts/sar_safepo_eval_all_seeds.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/RISE-Training"

TASK="${TASK:-PointLTL0MASAR1WC-v0}"
EVAL_ENV="${EVAL_ENV:-PointLTL0MASAR2WC-v0}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
ALGOS="${ALGOS:-ppo ppo_lag}"

TASK_DIR="${LOG_ROOT}/${TASK}"
if [[ ! -d "${TASK_DIR}" ]]; then
  echo "ERROR: missing ${TASK_DIR}" >&2
  exit 1
fi

found_any=0
for algo in ${ALGOS}; do
  algo_dir="${TASK_DIR}/${algo}"
  if [[ ! -d "${algo_dir}" ]]; then
    echo "Skip missing algo dir: ${algo_dir}"
    continue
  fi

  shopt -s nullglob
  seed_dirs=("${algo_dir}"/seed-*)
  shopt -u nullglob

  if [[ ${#seed_dirs[@]} -eq 0 ]]; then
    echo "Skip ${algo}: no seed-* dirs"
    continue
  fi

  for run_dir in "${seed_dirs[@]}"; do
    [[ -f "${run_dir}/config.json" ]] || continue
    found_any=1
    echo ""
    echo "======== ${algo} SA eval seed_dir=${run_dir} ========"
    python eval_safepo_sa_env.py \
      --run-dir "$run_dir" \
      --eval-episodes "$EVAL_EPISODES" \
      --seed "$SEED" \
      --device "$DEVICE"

    echo "======== ${algo} MA deploy eval seed_dir=${run_dir} ========"
    python eval_safepo_sa_on_ma_env.py \
      --run-dir "$run_dir" \
      --eval-env "$EVAL_ENV" \
      --eval-episodes "$EVAL_EPISODES" \
      --seed "$SEED" \
      --device "$DEVICE"
  done
done

if [[ "${found_any}" -eq 0 ]]; then
  echo "ERROR: no seed run dirs with config.json under ${TASK_DIR}/{${ALGOS}}" >&2
  exit 1
fi

echo ""
echo "Done. Summaries written next to each run dir (eval_summary.json, eval_summary_ma_deploy.json)."
