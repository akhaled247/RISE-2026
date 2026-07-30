#!/usr/bin/env bash
# Resume GenZ-LTL RCO from experiments/rco/<env>/<name>/<seed>/status.pth.
# Hyperparams match FileTransfer GenZ-2300-Runs snapshots (not run_sar.py defaults).
#
# Usage:
#   ./scripts/genz_rco_continue_direct.sh
#   ./scripts/genz_rco_continue_rco_sar_s0.sh
#   NAME=GenZ-SAR-s0 LR=0.0003 MAX_LAG=5 ./scripts/genz_rco_continue.sh
#
# Optional restore from FileTransfer copy on depend:
#   RESTORE_FROM=~/RISE-2026/FileTransfer/singleagent_runs/GenZ-2300-Runs/GenZ-LTL-RCO-Direct/0 \
#     ./scripts/genz_rco_continue_direct.sh
#
# Eval only (MA deploy) — run SA gate first: NAME=GenZ-SAR-s0 ./scripts/genz_sa_eval_gate.sh
#   EVAL_ONLY=1 NAME=GenZ-SAR-s0 ./scripts/genz_rco_continue.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/GenZ-LTL"

NAME="${NAME:?Set NAME to the experiment folder (e.g. GenZ-LTL-RCO-Direct or GenZ-RCO-SAR-s0)}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
ENV="${ENV:-PointLTL0MASAR1WC-v0}"
EVAL_ENV="${EVAL_ENV:-PointLTL0MASAR2WC-v0}"
NUM_STEPS="${NUM_STEPS:-15000000}"
NUM_PROCS="${NUM_PROCS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
FORMULA="${FORMULA:-((!surface_0 U entrapped_0) & F surface_0) & ((!surface_1 U entrapped_1) & F surface_1)}"

# RCO hyperparams (override via env; presets set these in wrapper scripts)
LR="${LR:-0.0003}"
MAX_LAG="${MAX_LAG:-5}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
STEPS_PER_PROCESS="${STEPS_PER_PROCESS:-4096}"
TARGET_COST="${TARGET_COST:--0.05}"
MIN_LAG="${MIN_LAG:-0.01}"
DISCOUNT="${DISCOUNT:-0.998}"
ENTROPY_COEF="${ENTROPY_COEF:-0.003}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

EXP_DIR="experiments/rco/${ENV}/${NAME}/${SEED}"

export PYTHONPATH=src

# Preflight: Rabinizer + Java (Büchi search during MA eval)
python -c "from envs.sar_deploy import check_rabinizer; check_rabinizer()"

if [[ "${EVAL_ONLY:-0}" == "1" && ! -f "${EXP_DIR}/status.pth" ]]; then
  echo "ERROR: EVAL_ONLY=1 but no ${EXP_DIR}/status.pth" >&2
  exit 1
fi

if [[ -n "${RESTORE_FROM:-}" ]]; then
  if [[ ! -d "${RESTORE_FROM}" ]]; then
    echo "ERROR: RESTORE_FROM=${RESTORE_FROM} is not a directory" >&2
    exit 1
  fi
  if [[ ! -f "${EXP_DIR}/status.pth" ]]; then
    echo "Restoring checkpoint tree from ${RESTORE_FROM} -> ${EXP_DIR}"
    mkdir -p "${EXP_DIR}"
    cp -a "${RESTORE_FROM}/." "${EXP_DIR}/"
  else
    echo "status.pth already at ${EXP_DIR}; skipping RESTORE_FROM"
  fi
fi

if [[ ! -f "${EXP_DIR}/status.pth" && "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "ERROR: no ${EXP_DIR}/status.pth — copy FileTransfer run or set RESTORE_FROM" >&2
  exit 1
fi

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "======== GenZ RCO resume ${ENV} name=${NAME} seed=${SEED} ========"
  echo "experiment_dir=${EXP_DIR}"
  python src/train/train_rco.py \
    --name "$NAME" \
    --env "$ENV" \
    --seed "$SEED" \
    --num_steps "$NUM_STEPS" \
    --num_procs "$NUM_PROCS" \
    --device "$DEVICE" \
    --vec_backend safety_async \
    --sar_env_backend specrl \
    --fast_action_bridge \
    --model_config "$ENV" \
    --curriculum "$ENV" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --steps_per_process "$STEPS_PER_PROCESS" \
    --lr "$LR" \
    --max_lag "$MAX_LAG" \
    --min_lag "$MIN_LAG" \
    --target_cost "$TARGET_COST" \
    --discount "$DISCOUNT" \
    --entropy_coef "$ENTROPY_COEF" \
    --target_kl 0.015 \
    --save_interval "$SAVE_INTERVAL"
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

echo "======== GenZ MA deploy eval ${EVAL_ENV} exp=${NAME} seed=${SEED} ========"
python src/evaluation/simulate_ma_sar.py \
  --exp "$NAME" \
  --seed "$SEED" \
  --formula "$FORMULA" \
  --eval_env "$EVAL_ENV" \
  --train_env "$ENV" \
  --num_episodes "$EVAL_EPISODES"

echo "Done. experiment_dir=${EXP_DIR}"
