#!/usr/bin/env bash
# Phase 3.1 — offline DT-CWT meta-mask training with quality controls.
# Masks  → /mnt/study-data/dcarvalho/tests/YYYYMMDD_HHMMSS_phase3_1_<exp>/
# Metrics→ /mnt/study-data/dcarvalho/metrics/YYYYMMDD_HHMMSS_phase3_1_<exp>/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}/protego_base${PYTHONPATH:+:$PYTHONPATH}"

# Quality is on by default (SSIM hinge + L∞). Example:
#   bash scripts/run_phase3_1_train.sh --exp_name lfw_K4_q --epochs 3 --max_train 512 --K 4
python protego_base/train_meta_mask.py \
  --quality \
  --exp_name "${EXP_NAME:-meta_mask_q}" \
  --dataset "${DATASET:-lfw_frontal}" \
  --device "${DEVICE:-cuda:0}" \
  "$@"
