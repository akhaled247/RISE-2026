#!/usr/bin/env bash
# SafePO TRPO-Lag — PointLTL{0,1,2}MASAR1[-WC]-v0 (train + eval).
# Lag rematch: λ_init=0, per-level recipe from common.sh.
# Run in its own terminal:  bash scripts/ladder_algo_ltl/train_trpo_lag.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${RISE_TRAINING_ROOT}"

ALGO="trpo_lag"
RESULTS="$(results_file_for "$ALGO")"
mkdir -p "$LOG_ROOT"
: > "$RESULTS"

run_one() {
  local level="$1" variant="$2"
  local task experiment run_dir steps spe gamma lam tkl lam_i lam_lr cost_lim ltl_flag
  task="$(task_for "$level" "$variant")"
  experiment="$(experiment_for "$ALGO" "$level" "$variant")"
  read -r steps spe gamma lam tkl < <(recipe_for "$level" "$variant")
  lam_i="$(lag_lambda_init)"
  lam_lr="$(lag_lambda_lr)"
  cost_lim="$(lag_cost_limit)"
  ltl_flag="$(sar_ltl_ordering_flag "$variant")"

  echo "======== TRAIN $ALGO $task  exp=$experiment  steps=$steps T=$spe λi=$lam_i  gpu=$DEVICE_ID ========"
  python train/trpo_lag_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps "$steps" --num-envs 8 --steps-per-epoch "$spe" \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 1 \
    --target-kl "$tkl" --gamma "$gamma" --lam "$lam" --lam-c "$lam" \
    --max-grad-norm 40 --hidden-sizes 64 64 \
    --cost-limit "$cost_lim" \
    --lagrangian-multiplier-init "$lam_i" \
    --lagrangian-multiplier-lr "$lam_lr" \
    --save-model-freq 10 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True \
    $ltl_flag

  run_dir="$(latest_run_dir "$task" "$ALGO")"
  eval_run_dir "$ALGO" "$task" "$experiment" "$run_dir"
}

while IFS= read -r row; do
  # shellcheck disable=SC2086
  run_one $row
done < <(filtered_train_jobs)

echo "TRPO-Lag ladder done (seed=$SEED). Eval results → $RESULTS"
