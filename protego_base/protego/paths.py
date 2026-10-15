"""
Absolute output paths and versioned run directories (PLAN.md).

Masks  → /mnt/study-data/dcarvalho/tests/
Logs   → /mnt/study-data/dcarvalho/metrics/

Folder naming (avoids overwrite, matches prior study-data style)::

    YYYYMMDD_HHMMSS_phase3_<exp_name>/
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

MASK_ROOT = "/mnt/study-data/dcarvalho/tests"
METRICS_ROOT = "/mnt/study-data/dcarvalho/metrics"

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_exp_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name.strip()).strip("._-")
    return cleaned or "run"


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dirs(
    phase: str,
    exp_name: str,
    mask_root: str = MASK_ROOT,
    metrics_root: str = METRICS_ROOT,
    stamp: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Create unique sibling directories for masks and metrics.

    Returns:
        (run_id, mask_dir, metrics_dir)
        where ``run_id`` is ``YYYYMMDD_HHMMSS_phaseX_expname``.
    """
    stamp = stamp or timestamp_now()
    base = f"{stamp}_{sanitize_exp_name(phase)}_{sanitize_exp_name(exp_name)}"
    # Guarantee uniqueness even if two runs start in the same second.
    for suffix in ("", "_b", "_c", "_d", "_e"):
        run_id = f"{base}{suffix}"
        mask_dir = os.path.join(mask_root, run_id)
        metrics_dir = os.path.join(metrics_root, run_id)
        if os.path.exists(mask_dir) or os.path.exists(metrics_dir):
            continue
        os.makedirs(mask_dir, exist_ok=False)
        os.makedirs(metrics_dir, exist_ok=False)
        return run_id, mask_dir, metrics_dir
    # Last resort: include microseconds.
    run_id = f"{base}_{datetime.now().strftime('%f')}"
    mask_dir = os.path.join(mask_root, run_id)
    metrics_dir = os.path.join(metrics_root, run_id)
    os.makedirs(mask_dir, exist_ok=False)
    os.makedirs(metrics_dir, exist_ok=False)
    return run_id, mask_dir, metrics_dir


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
