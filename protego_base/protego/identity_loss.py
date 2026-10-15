"""
Phase 3.2 — identity push-away loss (Protego-style cosine similarity).

Minimize ``cos(f_orig, f_prot)`` so that lower values mean stronger
de-identification (same convention used in logs / notebooks).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def identity_cosine_loss(
    feat_orig: torch.Tensor,
    feat_prot: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cosine similarity between original and protected embeddings.

    Args:
        feat_orig: ``(N, D)`` FR features of clean faces.
        feat_prot: ``(N, D)`` FR features of protected faces.

    Returns:
        (loss scalar to minimize, mean cosine similarity scalar for logging).
    """
    if feat_orig.shape != feat_prot.shape:
        raise ValueError(
            f"feat shape mismatch orig={tuple(feat_orig.shape)} prot={tuple(feat_prot.shape)}"
        )
    if feat_orig.ndim != 2:
        raise ValueError(f"features must be (N,D), got {tuple(feat_orig.shape)}")

    # Match Protego: cosine_similarity without forcing pre-normalize on both,
    # but L2-normalize first for a stable unit-sphere cos (lower = better de-ID).
    o = F.normalize(feat_orig, p=2, dim=-1)
    p = F.normalize(feat_prot, p=2, dim=-1)
    cos_sim = (o * p).sum(dim=-1).mean()
    return cos_sim, cos_sim.detach()
