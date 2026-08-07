#!/usr/bin/env bash
# GenZ-LTL RCO: train SA on MASAR1WC (RISE async) → MA deploy eval (RISE).
#   ./scripts/sar_genz_sa_to_ma.sh
#   SEED=0 DEVICE=cuda:0 ./scripts/sar_genz_sa_to_ma.sh
#   EVAL_ONLY=1 NAME=GenZ-SAR ./scripts/sar_genz_sa_to_ma.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NAME="${NAME:-GenZ-SAR}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
ENV="${ENV:-PointLTL0MASAR1WC-v0}"
EVAL_ENV="${EVAL_ENV:-PointLTL0MASAR2WC-v0}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
FORMULA="${FORMULA:-(!(any_walls | any_surface) U all_entrapped) & (!any_walls U all_surface)}"

export PYTHONPATH="${REPO_ROOT}/RISE-Training:${REPO_ROOT}/GenZ-LTL/src:${PYTHONPATH:-}"

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "======== GenZ RCO async train ${ENV} name=${NAME}-s${SEED} ========"
  python RISE-Training/train/genz_rco_async.py \
    --name "${NAME}-s${SEED}" \
    --env "$ENV" \
    --curriculum "$ENV" \
    --model_config "$ENV" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --num_steps 15000000 \
    --num_procs 24 \
    --steps_per_process 4096 \
    --batch_size 2048 \
    --epochs 80 \
    --save_interval 10 \
    --discount 0.998 \
    --lr 0.0003 \
    --entropy_coef 0.003 \
    --sar_env_backend specrl
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

echo "======== GenZ MA deploy eval ${EVAL_ENV} exp=${NAME}-s${SEED} ========"
python RISE-Training/eval_genz_ma_sar.py \
  --exp "${NAME}-s${SEED}" \
  --seed "$SEED" \
  --formula "$FORMULA" \
  --eval_env "$EVAL_ENV" \
  --train_env "$ENV" \
  --num_episodes "$EVAL_EPISODES"

echo "Done. GenZ MA eval for ${NAME}-s${SEED}"
