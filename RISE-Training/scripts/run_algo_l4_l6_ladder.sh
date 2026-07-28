#!/usr/bin/env bash
# SafePO ladder: PPO / PPO-Lag / TRPO / TRPO-Lag × L4–L6 ± WC
# Base = PPO L4WC win recipe; per-algo deltas (iters/kl/Lag). Train → 50-ep eval.
set -euo pipefail
cd "$(dirname "$0")/.."   # RISE-Training root (script lives in scripts/)

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
RESULTS="${RESULTS:-${LOG_ROOT}/ladder_algo_results.txt}"
# Override example: ALGOS="ppo_lag trpo trpo_lag"
ALGOS="${ALGOS:-ppo ppo_lag trpo trpo_lag}"

mkdir -p "$LOG_ROOT"
: > "$RESULTS"

# LEVEL VARIANT TOTAL_STEPS
ENV_JOBS=(
  "4 plain 4000000"
  "4 wc    4000000"
  "5 plain 4000000"
  "5 wc    4000000"
  "6 plain 5000000"
  "6 wc    5000000"
)

latest_run_dir() {
  local task="$1" algo="$2"
  local base="${LOG_ROOT}/${task}/${algo}"
  ls -td "${base}"/seed-* 2>/dev/null | head -1
}

train_cmd() {
  local algo="$1" task="$2" experiment="$3" steps="$4"
  local script
  case "$algo" in
    ppo)      script=train/ppo_train_env.py ;;
    ppo_lag)  script=train/ppo_lag_train_env.py ;;
    trpo)     script=train/trpo_train_env.py ;;
    trpo_lag) script=train/trpo_lag_train_env.py ;;
    *) echo "unknown algo: $algo"; return 1 ;;
  esac

  local -a common=(
    --task "$task" --seed "$SEED"
    --experiment "$experiment"
    --log-dir "$LOG_ROOT"
    --total-steps "$steps" --num-envs 8 --steps-per-epoch 32768
    --actor-lr 5e-5 --critic-lr 1e-3
    --batch-size 256
    --gamma 0.99 --lam 0.95 --lam-c 0.95
    --max-grad-norm 40 --hidden-sizes 64 64
    --device "$DEVICE" --device-id "$DEVICE_ID"
    --write-terminal False --use-tensorboard True
    --parallel True
  )

  case "$algo" in
    ppo)
      python "$script" "${common[@]}" \
        --learning-iters 10 --target-kl 0.05 \
        --clip-ratio 0.2 --lr_end_factor 1.0
      ;;
    ppo_lag)
      python "$script" "${common[@]}" \
        --learning-iters 10 --target-kl 0.05 \
        --clip-ratio 0.2 --lr_end_factor 1.0 \
        --cost-limit 0.0 \
        --lagrangian-multiplier-init 1.0 \
        --lagrangian-multiplier-lr 0.01
      ;;
    trpo)
      # No clip / lr_end_factor (CG trust region; no actor LinearLR)
      python "$script" "${common[@]}" \
        --learning-iters 1 --target-kl 0.02
      ;;
    trpo_lag)
      python "$script" "${common[@]}" \
        --learning-iters 1 --target-kl 0.02 \
        --cost-limit 0.0 \
        --lagrangian-multiplier-init 1.0 \
        --lagrangian-multiplier-lr 0.01
      ;;
  esac
}

run_one() {
  local algo="$1" level="$2" variant="$3" steps="$4"
  local task experiment
  if [[ "$variant" == "wc" ]]; then
    task="PointLTL${level}MASAR1WC-v0"
    experiment="${algo}_l${level}wc_B_s${SEED}"
  else
    task="PointLTL${level}MASAR1-v0"
    experiment="${algo}_l${level}_B_s${SEED}"
  fi

  echo "======== TRAIN $algo $task  exp=$experiment  steps=$steps  gpu=$DEVICE_ID ========"
  train_cmd "$algo" "$task" "$experiment" "$steps"

  local run_dir
  run_dir="$(latest_run_dir "$task" "$algo")"
  if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
    echo "ERROR: no run dir ${LOG_ROOT}/${task}/${algo}/" | tee -a "$RESULTS"
    return 1
  fi

  echo "======== EVAL $run_dir ========"
  {
    echo "---- $algo / $task / $experiment ----"
    echo "run_dir=$run_dir"
    python eval_safepo_env.py \
      --run-dir "$run_dir" \
      --eval-episodes "$EVAL_EPISODES" \
      --device "$DEVICE" \
      --seed "$SEED"
    echo
  } | tee -a "$RESULTS"
}

for algo in $ALGOS; do
  for row in "${ENV_JOBS[@]}"; do
    # shellcheck disable=SC2086
    run_one "$algo" $row
  done
done

echo "All done → $RESULTS"
