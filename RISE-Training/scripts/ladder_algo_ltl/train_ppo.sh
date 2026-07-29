#!/usr/bin/env bash
# SafePO PPO — PointLTL2MASAR1[-WC]-v0 (train + eval).
# Run in its own terminal:  bash scripts/ladder_algo_ltl/train_ppo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${RISE_TRAINING_ROOT}"

ALGO="ppo"
RESULTS="$(results_file_for "$ALGO")"
mkdir -p "$LOG_ROOT"
: > "$RESULTS"

run_one() {
  local level="$1" variant="$2"
  local task experiment run_dir
  task="$(task_for "$level" "$variant")"
  experiment="$(experiment_for "$ALGO" "$level" "$variant")"

  echo "======== TRAIN $ALGO $task  exp=$experiment  gpu=$DEVICE_ID ========"
  python train/ppo_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps 5000000 --num-envs 8 --steps-per-epoch 65536 \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 10 \
    --target-kl 0.05 --gamma 0.995 --lam 0.98 --lam-c 0.98 \
    --clip-ratio 0.2 --max-grad-norm 40 --hidden-sizes 64 64 \
    --cost-limit 0.0 \
    --lagrangian-multiplier-init 1.0 \
    --lagrangian-multiplier-lr 0.01 \
    --save-model-freq 10 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True --lr_end_factor 1.0 --ent-coef 0.0

  run_dir="$(latest_run_dir "$task" "$ALGO")"
  eval_run_dir "$ALGO" "$task" "$experiment" "$run_dir"
}

while IFS= read -r row; do
  # shellcheck disable=SC2086
  run_one $row
done < <(filtered_train_jobs)

echo "PPO ladder done (seed=$SEED). Eval results → $RESULTS"
