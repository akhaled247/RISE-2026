#!/usr/bin/env bash
# Copy a FileTransfer (or other) checkpoint into GenZ-LTL experiments/rco layout.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_ENV="${TRAIN_ENV:-PointLTL0MASAR1WC-v0}"
NAME="${NAME:?Set NAME (experiment folder name)}"
SEED="${SEED:-0}"
RESTORE_FROM="${RESTORE_FROM:?Set RESTORE_FROM to the seed directory (contains status.pth)}"
DEST="$REPO_ROOT/GenZ-LTL/experiments/rco/${TRAIN_ENV}/${NAME}/${SEED}"
mkdir -p "$DEST"
cp -a "${RESTORE_FROM}/." "$DEST/"
echo "Restored checkpoint to $DEST"
ls -la "$DEST"
