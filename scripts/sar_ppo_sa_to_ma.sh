#!/usr/bin/env bash
# PPO: train single-agent on MASAR1WC → deploy eval on MASAR2WC.
# Run in its own terminal:
#   ./scripts/sar_ppo_sa_to_ma.sh
#   SEED=0 DEVICE_ID=0 ./scripts/sar_ppo_sa_to_ma.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/RISE-Training"

TASK="${TASK:-PointLTL0MASAR1WC-v0}"
EVAL_ENV="${EVAL_ENV:-PointLTL0MASAR2WC-v0}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-0}"
LOG_ROOT="${LOG_ROOT:-./_training_logs/safepo}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EXPERIMENT="${EXPERIMENT:-ppo_sar_sa_to_ma_s${SEED}}"

latest_run_dir() {
  ls -td "${LOG_ROOT}/${TASK}/ppo"/seed-* 2>/dev/null | head -1
}

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "======== PPO train ${TASK} seed=${SEED} gpu=${DEVICE_ID} ========"
  python train/ppo_train_env.py \
    --task "$TASK" \
    --seed "$SEED" \
    --experiment "$EXPERIMENT" \
    --log-dir "$LOG_ROOT" \
    --total-steps 10000000 \
    --num-envs 16 \
    --steps-per-epoch 65536 \
    --actor-lr 5e-5 \
    --critic-lr 1e-3 \
    --batch-size 256 \
    --learning-iters 10 \
    --target-kl 0.05 \
    --gamma 0.995 \
    --lam 0.98 \
    --lam-c 0.98 \
    --clip-ratio 0.2 \
    --max-grad-norm 40 \
    --hidden-sizes 64 64 \
    --ent-coef 0.02 \
    --device "$DEVICE" \
    --device-id "$DEVICE_ID" \
    --write-terminal False \
    --use-tensorboard True \
    --parallel True \
    --lr_end_factor 1.0
fi

RUN_DIR="${RUN_DIR:-$(latest_run_dir)}"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: no PPO run dir under ${LOG_ROOT}/${TASK}/ppo/" >&2
  exit 1
fi
echo "run_dir=${RUN_DIR}"

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

echo "======== PPO MA deploy eval ${EVAL_ENV} episodes=${EVAL_EPISODES} ========"
python eval_safepo_sa_on_ma_env.py \
  --run-dir "$RUN_DIR" \
  --eval-env "$EVAL_ENV" \
  --eval-episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "Done. MA eval summary: ${RUN_DIR}/eval_summary_ma_deploy.json"
