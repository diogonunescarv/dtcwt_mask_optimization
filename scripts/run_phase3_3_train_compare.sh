#!/usr/bin/env bash
# Phase 3.3 — run Adam then Protego with the same data split / epochs / FRs.
# Intended for manual execution (full train; slow). Results feed the 3.3 notebook.
#
#   bash scripts/run_phase3_3_train_compare.sh
#   EPOCHS=5 CELEBA_TRAIN_FRAC=0.5 DATA_SEED=42 bash scripts/run_phase3_3_train_compare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export CELEBA_TRAIN_FRAC="${CELEBA_TRAIN_FRAC:-0.5}"
export DATA_SEED="${DATA_SEED:-42}"
export FR_PRESET="${FR_PRESET:-small}"
export DEVICE="${DEVICE:-cuda:0}"
export DATASET="${DATASET:-merged_lfw_celeba}"
export CKPT_EVERY="${CKPT_EVERY:-1000}"

echo "=== Phase 3.3 compare: Adam ==="
OPTIMIZER=adam EXP_NAME="${BASE_EXP:-merged_p33}_adam" \
  bash "${SCRIPT_DIR}/run_phase3_3_train.sh" "$@"

echo "=== Phase 3.3 compare: Protego (sign-SGD) ==="
OPTIMIZER=protego EXP_NAME="${BASE_EXP:-merged_p33}_protego" \
  bash "${SCRIPT_DIR}/run_phase3_3_train.sh" "$@"

echo "=== Done. Pointers under /mnt/study-data/dcarvalho/tests/LATEST_phase3_3_* ==="
