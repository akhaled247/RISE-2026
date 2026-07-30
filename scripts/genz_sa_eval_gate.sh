#!/usr/bin/env bash
# SA eval gate: run before MA deploy to verify checkpoint on MASAR1WC.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/GenZ-LTL"
export PYTHONPATH=src
NAME="${NAME:?Set NAME}"
SEED="${SEED:-0}"
FORMULA="${FORMULA:-(!surface_0 U entrapped_0) & F surface_0}"
EPISODES="${EPISODES:-20}"
python src/evaluation/simulate.py \
  --env PointLTL0MASAR1WC-v0 \
  --exp "$NAME" \
  --seed "$SEED" \
  --formula "$FORMULA" \
  --num_episodes "$EPISODES"
