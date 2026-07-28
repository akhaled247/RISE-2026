#!/usr/bin/env bash
# SafePO TRPO ladder: PointLTL{0,1,2,3}MASAR1[-WC]-v0 — train then 50-ep eval.
# Recipes from L4/L5/L6 wins: L0←L4, L1←L5, L2–L3←L6.
# No L6 WC win → L2/L3 WC inherit L6 plain.
set -euo pipefail
cd "$(dirname "$0")/.."   # RISE-Training root (script lives in scripts/)

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
RESULTS="${RESULTS:-${LOG_ROOT}/ladder_trpo_l0_l3_results.txt}"

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
  local base="${LOG_ROOT}/${task}/trpo"
  ls -td "${base}"/seed-* 2>/dev/null | head -1
}

# Emit: steps spe learning_iters target_kl gamma lam
recipe_for() {
  local level="$1" variant="$2"
  case "${level}_${variant}" in
    0_plain) echo "4000000 32768 1 0.05 0.995 0.98" ;;   # L4 plain
    0_wc)    echo "5000000 32768 1 0.05 0.995 0.98" ;;   # L4 WC
    1_plain) echo "4000000 32768 1 0.02 0.99 0.95" ;;    # L5 plain
    1_wc)    echo "5000000 65536 10 0.05 0.995 0.98" ;;  # L5 WC
    2_plain|3_plain) echo "4000000 32768 1 0.05 0.995 0.98" ;;  # L6 plain
    2_wc|3_wc)       echo "4000000 32768 1 0.05 0.995 0.98" ;;  # inherit L6 plain
    *) echo "unknown recipe: level=$level variant=$variant" >&2; return 1 ;;
  esac
}

run_one() {
  local level="$1" variant="$2"
  local task experiment steps spe iters tkl gamma lam
  read -r steps spe iters tkl gamma lam < <(recipe_for "$level" "$variant")

  if [[ "$variant" == "wc" ]]; then
    task="PointLTL${level}MASAR1WC-v0"
    experiment="trpo_l${level}wc_l0l3_s${SEED}"
  else
    task="PointLTL${level}MASAR1-v0"
    experiment="trpo_l${level}_l0l3_s${SEED}"
  fi

  echo "======== TRAIN trpo $task  exp=$experiment  steps=$steps T=$spe  gpu=$DEVICE_ID ========"
  python train/trpo_train_env.py \
    --task "$task" --seed "$SEED" \
    --experiment "$experiment" \
    --log-dir "$LOG_ROOT" \
    --total-steps "$steps" --num-envs 8 --steps-per-epoch "$spe" \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters "$iters" \
    --target-kl "$tkl" --gamma "$gamma" --lam "$lam" --lam-c "$lam" \
    --max-grad-norm 40 --hidden-sizes 64 64 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True

  local run_dir
  run_dir="$(latest_run_dir "$task")"
  if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
    echo "ERROR: no run dir under ${LOG_ROOT}/${task}/trpo/" | tee -a "$RESULTS"
    return 1
  fi

  echo "======== EVAL  $run_dir  episodes=$EVAL_EPISODES ========"
  {
    echo "---- trpo / $task / $experiment ----"
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
