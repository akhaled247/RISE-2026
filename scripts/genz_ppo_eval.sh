#!/usr/bin/env bash
# PPO SAR eval: SA on MASAR1WC or MA deploy on MASAR2WC (auto-detects PPO vs RCO in simulate.py).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/GenZ-LTL"
export PYTHONPATH=src

TRAIN_ENV="${TRAIN_ENV:-PointLTL0MASAR1WC-v0}"
NAME="${NAME:?Set NAME}"
SEED="${SEED:-0}"
MODE="${MODE:-sa}"  # sa | ma
EPISODES="${EPISODES:-100}"
FORMULA_SA="${FORMULA_SA:-(!surface_0 U entrapped_0) & F surface_0}"
FORMULA_MA="${FORMULA_MA:-((!surface_0 & !surface_1) U all_entrapped) & (F surface_0 & F surface_1)}"

if [[ -n "${RESTORE_FROM:-}" ]]; then
  DEST="$REPO_ROOT/GenZ-LTL/experiments/rco/${TRAIN_ENV}/${NAME}/${SEED}"
  if [[ ! -f "$DEST/status.pth" ]]; then
    if [[ -f "${RESTORE_FROM}/status.pth" ]]; then
      RESTORE_FROM="$RESTORE_FROM" NAME="$NAME" SEED="$SEED" TRAIN_ENV="$TRAIN_ENV" \
        bash "$REPO_ROOT/scripts/genz_restore_ppo_checkpoint.sh"
    else
      echo "No status.pth at $DEST or RESTORE_FROM ($RESTORE_FROM)" >&2
      exit 1
    fi
  fi
fi

case "$MODE" in
  sa)
    python src/evaluation/simulate.py \
      --env "$TRAIN_ENV" \
      --exp "$NAME" \
      --seed "$SEED" \
      --formula "$FORMULA_SA" \
      --num_episodes "$EPISODES"
    ;;
  ma)
    python src/evaluation/simulate_ma_ppo.py \
      --train_env "$TRAIN_ENV" \
      --eval_env PointLTL0MASAR2WC-v0 \
      --exp "$NAME" \
      --seed "$SEED" \
      --formula "$FORMULA_MA" \
      --num_episodes "$EPISODES"
    ;;
  *)
    echo "MODE must be sa or ma (got $MODE)" >&2
    exit 1
    ;;
esac
