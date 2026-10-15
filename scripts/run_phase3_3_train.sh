#!/usr/bin/env bash
# Phase 3.3 — offline DT-CWT meta-mask: identity ensemble + SSIM/L∞ (no K-disp).
# Dataset: all LFW faces_train + seeded CelebA fraction (default 0.5).
# Ensemble default: fr_preset=small (3 FRs). Full ensemble via FR_PRESET=full later.
#
# Masks  → /mnt/study-data/dcarvalho/tests/YYYYMMDD_HHMMSS_phase3_3_<exp>/
# Metrics→ /mnt/study-data/dcarvalho/metrics/YYYYMMDD_HHMMSS_phase3_3_<exp>/
#
# Examples (run manually; full train is slow):
#   bash scripts/run_phase3_3_train.sh
#   OPTIMIZER=protego bash scripts/run_phase3_3_train.sh
#   EPOCHS=5 CELEBA_TRAIN_FRAC=0.5 bash scripts/run_phase3_3_train.sh --exp_name my_run
#   # resume after disconnect (same run folder):
#   bash scripts/run_phase3_3_train.sh --resume /mnt/study-data/dcarvalho/tests/<run_id>/checkpoint_latest.pt
#
# Rolling checkpoint: checkpoint_latest.pt overwritten every CKPT_EVERY steps (default 1000).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}/protego_base${PYTHONPATH:+:$PYTHONPATH}"

OPTIMIZER="${OPTIMIZER:-adam}"
EXP_NAME="${EXP_NAME:-merged_p33_${OPTIMIZER}}"

python protego_base/train_meta_mask.py \
  --phase33 \
  --quality \
  --fr_preset "${FR_PRESET:-small}" \
  --optimizer "${OPTIMIZER}" \
  --exp_name "${EXP_NAME}" \
  --dataset "${DATASET:-merged_lfw_celeba}" \
  --celeba_train_frac "${CELEBA_TRAIN_FRAC:-0.5}" \
  --data_seed "${DATA_SEED:-42}" \
  --device "${DEVICE:-cuda:0}" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --ckpt_every "${CKPT_EVERY:-1000}" \
  "$@"
