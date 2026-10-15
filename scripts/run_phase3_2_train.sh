#!/usr/bin/env bash
# Phase 3.2 — offline DT-CWT meta-mask with identity + batch Gram + multi-FR.
# Dataset default: merged LFW train + CelebA-HQ.
#
# Ensemble (keep multi-FR idea; choose size via flag):
#   FR_PRESET=small  → 3 diverse FRs (default; friendlier to ~11GB GPUs)
#   FR_PRESET=full   → Protego BASIC_POOL w/o ir50_adaface_casia (9 FRs)
#   or pass --fr_models a,b,c / --max_frs N
#
# Masks  → /mnt/study-data/dcarvalho/tests/YYYYMMDD_HHMMSS_phase3_2_<exp>/
# Metrics→ /mnt/study-data/dcarvalho/metrics/YYYYMMDD_HHMMSS_phase3_2_<exp>/
#
# Examples:
#   bash scripts/run_phase3_2_train.sh --exp_name merged_K4 --epochs 3 --max_train 4096
#   FR_PRESET=full bash scripts/run_phase3_2_train.sh --exp_name merged_K4_full ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}/protego_base${PYTHONPATH:+:$PYTHONPATH}"

python protego_base/train_meta_mask.py \
  --phase32 \
  --quality \
  --fr_preset "${FR_PRESET:-small}" \
  --exp_name "${EXP_NAME:-merged_p32}" \
  --dataset "${DATASET:-merged_lfw_celeba}" \
  --device "${DEVICE:-cuda:0}" \
  --epochs "${EPOCHS:-3}" \
  --max_train "${MAX_TRAIN:-4096}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --K "${K:-4}" \
  "$@"
