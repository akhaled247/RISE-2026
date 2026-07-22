#!/usr/bin/env bash
# SafePO PPO ladder: PointLTL{4,5,6}MASAR1[-WC]-v0 — train then 50-ep eval (one GPU).
# Recipe = L4WC win (kl0.05, 32k/epoch, lr_end_factor 1.0 → ~78% rescue).
set -euo pipefail
cd "$(dirname "$0")/.."   # SpecRLBench root (script lives in scripts/)

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
RESULTS="${RESULTS:-${LOG_ROOT}/ladder_ppo_results.txt}"

mkdir -p "$LOG_ROOT"
: > "$RESULTS"

# LEVEL VARIANT TOTAL_STEPS
JOBS=(
  "4 plain 4000000"
  "4 wc    4000000"
  "5 plain 4000000"
  "5 wc    4000000"
  "6 plain 5000000"
  "6 wc    5000000"
)

latest_run_dir() {
  local task="$1"
  local base="${LOG_ROOT}/${task}/ppo"
  ls -td "${base}"/seed-* 2>/dev/null | head -1
}

run_one() {
  local level="$1" variant="$2" steps="$3"
  local task experiment
  if [[ "$variant" == "wc" ]]; then
    task="PointLTL${level}MASAR1WC-v0"
    experiment="l${level}wc_B_kl05_32k_s${SEED}"
  else
    task="PointLTL${level}MASAR1-v0"
    experiment="l${level}_B_kl05_32k_s${SEED}"
  fi

  echo "======== TRAIN $task  experiment=$experiment  steps=$steps  gpu=$DEVICE_ID ========"
  python train/ppo_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps "$steps" --num-envs 8 --steps-per-epoch 32768 \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 10 \
    --target-kl 0.05 --gamma 0.99 --lam 0.95 --lam-c 0.95 \
    --clip-ratio 0.2 --max-grad-norm 40 --hidden-sizes 64 64 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True --lr_end_factor 1.0

  local run_dir
  run_dir="$(latest_run_dir "$task")"
  if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
    echo "ERROR: no run dir under ${LOG_ROOT}/${task}/ppo/" | tee -a "$RESULTS"
    return 1
  fi

  echo "======== EVAL  $run_dir  episodes=$EVAL_EPISODES ========"
  {
    echo "---- $task / $experiment ----"
    echo "run_dir=$run_dir"
    python eval_safepo_env.py \
      --run-dir "$run_dir" \
      --eval-episodes "$EVAL_EPISODES" \
      --device "$DEVICE" \
      --seed "$SEED"
    echo
  } | tee -a "$RESULTS"
}

for row in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run_one $row
done

echo "All done. Aggregated eval → $RESULTS"
