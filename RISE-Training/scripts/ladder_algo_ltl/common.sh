#!/usr/bin/env bash
# Shared settings for ladder_algo_ltl training + eval scripts.
# Train + eval: PointLTL{0,1,2}MASAR1[-WC]-v0 (override single job with LEVEL + VARIANT).

RISE_TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-1}"
SEED="${SEED:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"

# Train jobs (L0–L2 plain + WC). Override with LEVEL=0 VARIANT=wc for a single job.
TRAIN_JOBS=(
  # "0 plain"
  "0 wc"
  # "1 plain"
  "1 wc"
  # "2 plain"
  "2 wc"
)

# Eval-only jobs (L0 + L1 plain + WC; eval_*.sh without retraining).
EVAL_JOBS=(
  "0 plain"
  "0 wc"
  "1 plain"
  "1 wc"
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

filtered_train_jobs() {
  if [[ -n "${LEVEL:-}" && -n "${VARIANT:-}" ]]; then
    echo "${LEVEL} ${VARIANT}"
    return
  fi
  printf '%s\n' "${TRAIN_JOBS[@]}"
}

filtered_eval_jobs() {
  if [[ -n "${LEVEL:-}" && -n "${VARIANT:-}" ]]; then
    echo "${LEVEL} ${VARIANT}"
    return
  fi
  printf '%s\n' "${EVAL_JOBS[@]}"
}

# Per-level recipes (from run_ppo_l0_l3_ladder / vault #32 rematch grid).
# Emits: total_steps steps_per_epoch gamma lam target_kl
recipe_for() {
  local level="$1" variant="$2"
  case "${level}_${variant}" in
    0_plain) echo "5000000 32768 0.995 0.98 0.05" ;;
    0_wc)    echo "5000000 65536 0.995 0.98 0.05" ;;
    1_plain) echo "5000000 32768 0.995 0.98 0.05" ;;
    1_wc)    echo "5000000 65536 0.995 0.98 0.05" ;;
    2_plain) echo "5000000 32768 0.995 0.98 0.05" ;;
    2_wc)    echo "5000000 65536 0.995 0.98 0.05" ;;
    *) echo "unknown recipe: level=$level variant=$variant" >&2; return 1 ;;
  esac
}

# Lag rematch: λ_init=0 (vault mem.safepo — avoid λ stuck when cost sparse).
lag_lambda_init() {
  echo "0.0"
}

lag_lambda_lr() {
  echo "${LAG_LAMBDA_LR:-0.01}"
}

lag_cost_limit() {
  echo "${LAG_COST_LIMIT:-0.25}"
}

ppo_ent_coef() {
  echo "${PPO_ENT_COEF:-0.02}"
}

latest_run_dir() {
  local task="$1" algo="$2"
  ls -td "${LOG_ROOT}/${task}/${algo}"/seed-* 2>/dev/null | head -1
}

results_file_for() {
  local algo="$1"
  echo "${LOG_ROOT}/ladder_algo_ltl_${algo}_results.txt"
}

eval_run_dir() {
  local algo="$1" task="$2" experiment="$3" run_dir="$4"
  local results
  results="$(results_file_for "$algo")"

  if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
    {
      echo "---- ERROR ${algo} / $task / $experiment ----"
      echo "no run dir under ${LOG_ROOT}/${task}/${algo}/"
      echo
    } | tee -a "$results"
    return 1
  fi

  echo "======== EVAL  $run_dir  episodes=$EVAL_EPISODES ========"
  {
    echo "---- ${algo} / $task / $experiment ----"
    echo "run_dir=$run_dir"
    python eval_safepo_sa_env.py \
      --run-dir "$run_dir" \
      --eval-episodes "$EVAL_EPISODES" \
      --device "$DEVICE" \
      --seed "$SEED"
    echo
  } | tee -a "$results"
}

eval_latest_for_job() {
  local algo="$1" level="$2" variant="$3"
  local task experiment run_dir
  task="$(task_for "$level" "$variant")"
  experiment="$(experiment_for "$algo" "$level" "$variant")"
  run_dir="$(latest_run_dir "$task" "$algo")"
  eval_run_dir "$algo" "$task" "$experiment" "$run_dir"
}
