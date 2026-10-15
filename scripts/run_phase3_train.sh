#!/usr/bin/env bash
# Phase 3 — offline DT-CWT meta-mask training.
# Masks  → /mnt/study-data/dcarvalho/tests/YYYYMMDD_HHMMSS_phase3_<exp>/
# Metrics→ /mnt/study-data/dcarvalho/metrics/YYYYMMDD_HHMMSS_phase3_<exp>/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}/protego_base${PYTHONPATH:+:$PYTHONPATH}"

# Phase 3 ablation: dispersion only (--no_quality). For SSIM/L∞ use run_phase3_1_train.sh.
# Override freely, e.g.:
#   bash scripts/run_phase3_train.sh --exp_name lfw_K4 --epochs 3 --max_train 256
python protego_base/train_meta_mask.py \
  --no_quality \
  --exp_name "${EXP_NAME:-meta_mask}" \
  --dataset "${DATASET:-lfw_frontal}" \
  --device "${DEVICE:-cuda:0}" \
  "$@"
