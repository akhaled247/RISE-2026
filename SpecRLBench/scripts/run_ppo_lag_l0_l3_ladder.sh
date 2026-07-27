#!/usr/bin/env bash
# SafePO PPO-Lag ladder: PointLTL{0,1,2,3}MASAR1[-WC]-v0 — train then 50-ep eval.
# Recipes from L4/L5/L6 WC wins (plain inherits WC). L0←L4, L1←L5, L2–L3←L6.
set -euo pipefail
cd "$(dirname "$0")/.."   # SpecRLBench root (script lives in scripts/)

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
RESULTS="${RESULTS:-${LOG_ROOT}/ladder_ppo_lag_l0_l3_results.txt}"

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
  local base="${LOG_ROOT}/${task}/ppo_lag"
  ls -td "${base}"/seed-* 2>/dev/null | head -1
}

# Emit: steps spe target_kl cost_limit lag_init lag_lr
# Plain uses WC recipe (no non-WC Lag wins).
recipe_for() {
  local level="$1"
  case "$level" in
    0) echo "5000000 32768 0.05 0.0 0.0 0.01" ;;           # L4 WC
    1) echo "6000000 65536 0.02 0.25 0.25 0.01" ;;          # L5 WC
    2|3) echo "6000000 65536 0.05 0.25 0.25 0.01" ;;        # L6 WC
    *) echo "unknown recipe: level=$level" >&2; return 1 ;;
  esac
}

run_one() {
  local level="$1" variant="$2"
  local task experiment steps spe tkl cl li llr
  read -r steps spe tkl cl li llr < <(recipe_for "$level")

  if [[ "$variant" == "wc" ]]; then
    task="PointLTL${level}MASAR1WC-v0"
    experiment="ppo_lag_l${level}wc_l0l3_s${SEED}"
  else
    task="PointLTL${level}MASAR1-v0"
    experiment="ppo_lag_l${level}_l0l3_s${SEED}"
  fi

  echo "======== TRAIN ppo_lag $task  exp=$experiment  steps=$steps T=$spe  gpu=$DEVICE_ID ========"
  python train/ppo_lag_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps "$steps" --num-envs 8 --steps-per-epoch "$spe" \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 10 \
    --target-kl "$tkl" --gamma 0.995 --lam 0.98 --lam-c 0.98 \
    --clip-ratio 0.2 --max-grad-norm 40 --hidden-sizes 64 64 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True --lr_end_factor 1.0 --ent-coef 0.0 \
    --cost-limit "$cl" \
    --lagrangian-multiplier-init "$li" \
    --lagrangian-multiplier-lr "$llr"

  local run_dir
  run_dir="$(latest_run_dir "$task")"
  if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
    echo "ERROR: no run dir under ${LOG_ROOT}/${task}/ppo_lag/" | tee -a "$RESULTS"
    return 1
  fi

  echo "======== EVAL  $run_dir  episodes=$EVAL_EPISODES ========"
  {
    echo "---- ppo_lag / $task / $experiment ----"
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
