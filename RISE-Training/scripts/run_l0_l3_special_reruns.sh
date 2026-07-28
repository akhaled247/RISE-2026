  #!/usr/bin/env bash
  # Ad-hoc L0–L3 special reruns → append eval into ladder_*_l0_l3_results.txt
  set -euo pipefail
  cd "$(dirname "$0")/.."   # RISE-Training root (or: cd /path/to/RISE-Training)
  DEVICE="${DEVICE:-cuda}"
  DEVICE_ID="${DEVICE_ID:-1}"
  SEED="${SEED:-0}"
  EVAL_EPISODES="${EVAL_EPISODES:-50}"
  LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
  mkdir -p "$LOG_ROOT"
  SEP="$(printf '*%.0s' {1..40})"
  latest_run_dir() {
    local task="$1" algo="$2"
    ls -td "${LOG_ROOT}/${task}/${algo}"/seed-* 2>/dev/null | head -1
  }
  eval_append() {
    local algo="$1" task="$2" experiment="$3"
    local results="${LOG_ROOT}/ladder_${algo}_l0_l3_results.txt"
    local run_dir
    run_dir="$(latest_run_dir "$task" "$algo")"
    if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
      {
        echo "$SEP"
        echo "ERROR: no run dir under ${LOG_ROOT}/${task}/${algo}/"
        echo "$SEP"
        echo
      } | tee -a "$results"
      return 1
    fi
    echo "======== EVAL  $run_dir  episodes=$EVAL_EPISODES ========"
    {
      echo "$SEP"
      echo "---- SPECIAL $algo / $task / $experiment ----"
      echo "run_dir=$run_dir"
      python eval_safepo_sa_env.py \
        --run-dir "$run_dir" \
        --eval-episodes "$EVAL_EPISODES" \
        --device "$DEVICE" \
        --seed "$SEED"
      echo "$SEP"
      echo
    } | tee -a "$results"
  }
  # ---------------------------------------------------------------------------
  # 5) TRPO  PointLTL2MASAR1WC-v0
  # ---------------------------------------------------------------------------
  TASK="PointLTL2MASAR1WC-v0"
  ALGO="trpo"
  EXP="trpo_l2wc_special_s${SEED}"
  echo "======== TRAIN $ALGO $TASK  exp=$EXP  gpu=$DEVICE_ID ========"
  python train/trpo_train_env.py \
    --task "$TASK" --seed "$SEED" \
    --experiment "$EXP" \
    --log-dir "$LOG_ROOT" \
    --total-steps 5000000 --num-envs 8 --steps-per-epoch 65536 \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 10 \
    --target-kl 0.05 --gamma 0.995 --lam 0.98 --lam-c 0.98 \
    --max-grad-norm 40 --hidden-sizes 64 64 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True
  eval_append "$ALGO" "$TASK" "$EXP"
  # ---------------------------------------------------------------------------
  # 6) TRPO-Lag  PointLTL2MASAR1WC-v0
  # ---------------------------------------------------------------------------
  TASK="PointLTL2MASAR1WC-v0"
  ALGO="trpo_lag"
  EXP="trpo_lag_l2wc_special_s${SEED}"
  echo "======== TRAIN $ALGO $TASK  exp=$EXP  gpu=$DEVICE_ID ========"
  python train/trpo_lag_train_env.py \
    --task "$TASK" --seed "$SEED" \
    --experiment "$EXP" \
    --log-dir "$LOG_ROOT" \
    --total-steps 6000000 --num-envs 8 --steps-per-epoch 65536 \
    --cost-limit 0.25 \
    --lagrangian-multiplier-init 0.25 --lagrangian-multiplier-lr 1e-2 \
    --actor-lr 5e-5 --critic-lr 1e-3 \
    --batch-size 256 --learning-iters 1 \
    --target-kl 0.02 --gamma 0.995 --lam 0.98 --lam-c 0.98 \
    --max-grad-norm 40 --hidden-sizes 64 64 \
    --device "$DEVICE" --device-id "$DEVICE_ID" \
    --write-terminal False --use-tensorboard True \
    --parallel True
  eval_append "$ALGO" "$TASK" "$EXP"
  echo "All special reruns done. Appended into ladder_*_l0_l3_results.txt under $LOG_ROOT"