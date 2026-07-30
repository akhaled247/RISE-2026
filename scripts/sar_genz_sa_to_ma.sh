#!/usr/bin/env bash
# GenZ-LTL RCO: train single-agent on MASAR1WC → deploy eval on MASAR2WC.
# Speed-tuned for 32-thread CPU (16 env workers, 80 RCO epochs, async vec).
# Run in its own terminal:
#   ./scripts/sar_genz_sa_to_ma.sh
#   SEED=0 DEVICE=cuda:0 ./scripts/sar_genz_sa_to_ma.sh
#   EVAL_ONLY=1 NAME=GenZ-SAR ./scripts/sar_genz_sa_to_ma.sh  # skip train, run MA eval
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/GenZ-LTL"

NAME="${NAME:-GenZ-SAR}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
ENV="${ENV:-PointLTL0MASAR1WC-v0}"
EVAL_ENV="${EVAL_ENV:-PointLTL0MASAR2WC-v0}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
FORMULA="${FORMULA:-((!surface_0 & !surface_1 & !all_surface) U all_entrapped) & F all_surface}"

export PYTHONPATH=src

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "======== GenZ RCO train ${ENV} name=${NAME}-s${SEED} ========"
  python run_sar.py \
    --script train_rco \
    --name "$NAME" \
    --env "$ENV" \
    --curriculum "$ENV" \
    --model_config "$ENV" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --num_steps 15000000 \
    --num_procs 16 \
    --steps_per_process 4096 \
    --batch_size 2048 \
    --epochs 80 \
    --save_interval 10 \
    --discount 0.998 \
    --lr 0.0003 \
    --entropy_coef 0.003 \
    --vec_backend safety_async \
    --sar_env_backend specrl \
    --fast_action_bridge
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

echo "======== GenZ MA deploy eval ${EVAL_ENV} exp=${NAME}-s${SEED} ========"
python src/evaluation/simulate_ma_sar.py \
  --exp "${NAME}-s${SEED}" \
  --seed "$SEED" \
  --formula "$FORMULA" \
  --eval_env "$EVAL_ENV" \
  --train_env "$ENV" \
  --num_episodes "$EVAL_EPISODES"

echo "Done. GenZ MA eval for ${NAME}-s${SEED}"
