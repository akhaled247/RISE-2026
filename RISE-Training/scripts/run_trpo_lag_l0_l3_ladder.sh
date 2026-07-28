#!/usr/bin/env bash
# SafePO TRPO-Lag ladder: PointLTL{0,1,2,3}MASAR1[-WC]-v0 — train then 50-ep eval.
# L0←L4 WC, L1←L5 WC (plain inherits WC). L2–L3: synthesize from
# trpo_lag L5 TR/Lag knobs + ppo_lag L6 budget (no L6 trpo_lag win).
set -euo pipefail
cd "$(dirname "$0")/.."   # RISE-Training root (script lives in scripts/)

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
RESULTS="${RESULTS:-${LOG_ROOT}/ladder_trpo_lag_l0_l3_results.txt}"

mkdir -p "$LOG_ROOT"
: > "$RESULTS"

# LEVEL VARIANT
JOBS=(
  "0 plain"
  "0 wc"
  "1 plain"
  "1 wc"
  "2 plain"
  "2 wc"
  "3 plain"
  "3 wc"
)

latest_run_dir() {
  local task="$1"
  local base="${LOG_ROOT}/${task}/trpo_lag"
  ls -td "${base}"/seed-* 2>/dev/null | head -1
}

# Emit: steps spe learning_iters target_kl cost_limit lag_init lag_lr
# Plain uses WC recipe. L2/L3 = trpo_lag L5 knobs + ppo_lag L6 budget.
recipe_for() {
  local level="$1"
  case "$level" in
    0) echo "5000000 32768 1 0.05 0.25 0.25 0.001" ;;     # L4 WC
    1) echo "6000000 65536 1 0.02 0.25 0.25 0.01" ;;      # L5 WC
    2|3) echo "6000000 65536 1 0.02 0.25 0.25 0.01" ;;    # synth L5 TR + L6 budget
    *) echo "unknown recipe: level=$level" >&2; return 1 ;;
  esac
}

run_one() {
  local level="$1" variant="$2"
  local task experiment steps spe iters tkl cl li llr
  read -r steps spe iters tkl cl li llr < <(recipe_for "$level")

  if [[ "$variant" == "wc" ]]; then
    task="PointLTL${level}MASAR1WC-v0"
    experiment="trpo_lag_l${level}wc_l0l3_s${SEED}"
  else
    task="PointLTL${level}MASAR1-v0"
    experiment="trpo_lag_l${level}_l0l3_s${SEED}"
  fi

  echo "======== TRAIN trpo_lag $task  exp=$experiment  steps=$steps T=$spe  gpu=$DEVICE_ID ========"
  python train/trpo_lag_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps "$steps" --num-envs 8 --steps-per-epoch "$spe" \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters "$iters" \
    --target-kl "$tkl" --gamma 0.995 --lam 0.98 --lam-c 0.98 \
    --max-grad-norm 40 --hidden-sizes 64 64 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True \
    --cost-limit "$cl" \
    --lagrangian-multiplier-init "$li" \
    --lagrangian-multiplier-lr "$llr"

  local run_dir
  run_dir="$(latest_run_dir "$task")"
  if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
    echo "ERROR: no run dir under ${LOG_ROOT}/${task}/trpo_lag/" | tee -a "$RESULTS"
    return 1
  fi

  echo "======== EVAL  $run_dir  episodes=$EVAL_EPISODES ========"
  {
    echo "---- trpo_lag / $task / $experiment ----"
    echo "run_dir=$run_dir"
    python eval_safepo_sa_env.py \
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
