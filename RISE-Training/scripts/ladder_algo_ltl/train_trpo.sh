#!/usr/bin/env bash
# SafePO TRPO — PointLTL{0,1,3}MASAR1[-WC]-v0 (6 envs, fixed hyperparams).
# Run in its own terminal:  bash scripts/ladder_algo_ltl/train_trpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${RISE_TRAINING_ROOT}"

run_one() {
  local level="$1" variant="$2"
  local task experiment
  task="$(task_for "$level" "$variant")"
  experiment="$(experiment_for "trpo" "$level" "$variant")"

  echo "======== TRAIN trpo $task  exp=$experiment  gpu=$DEVICE_ID ========"
  python train/trpo_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps 5000000 --num-envs 8 --steps-per-epoch 65536 \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 1 \
    --target-kl 0.05 --gamma 0.995 --lam 0.98 --lam-c 0.98 \
    --max-grad-norm 40 --hidden-sizes 64 64 \
    --cost-limit 0.0 \
    --lagrangian-multiplier-init 1.0 \
    --lagrangian-multiplier-lr 0.01 \
    --save-model-freq 10 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True
}

while IFS= read -r row; do
  # shellcheck disable=SC2086
  run_one $row
done < <(filtered_jobs)

echo "TRPO ladder done (seed=$SEED, log_root=$LOG_ROOT)."
