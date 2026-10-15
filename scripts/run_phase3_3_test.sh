#!/usr/bin/env bash
# Phase 3.3 functional test (identity ensemble + SSIM; Adam + Protego smoke).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}/protego_base${PYTHONPATH:+:$PYTHONPATH}"

python tests/test_phase3_3.py "$@"
