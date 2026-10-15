"""
Frequency Dispersion Loss (Phase 3).

    L_CWT-Dispersion = - log det G
    G_{u,v} = F(X_prot^{(u)})^T  F(X_prot^{(v)})

where the rows of the feature matrix are the K local phase variants of one face.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Build a regularized Gram matrix of L2-normalized features.

    Args:
        features: ``(K, D)`` or ``(N, K, D)``
        eps: diagonal jitter for numerical stability (same spirit as Protego).

    Returns:
        ``(K, K)`` or ``(N, K, K)``
    """
    if features.ndim == 2:
        feat = F.normalize(features, p=2, dim=-1)
        gram = feat @ feat.transpose(-1, -2)
        k = gram.shape[-1]
        eye = torch.eye(k, device=gram.device, dtype=gram.dtype)
        return gram + eps * eye
    if features.ndim == 3:
        feat = F.normalize(features, p=2, dim=-1)
        gram = torch.matmul(feat, feat.transpose(-1, -2))
        k = gram.shape[-1]
        eye = torch.eye(k, device=gram.device, dtype=gram.dtype)
        return gram + eps * eye.unsqueeze(0)
    raise ValueError(f"features must be (K,D) or (N,K,D), got {tuple(features.shape)}")


def frequency_dispersion_loss(
    features: torch.Tensor,
    eps: float = 1e-4,
    det_eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ``L = -log det G`` averaged over the batch when ``features`` is ``(N,K,D)``.

    Returns:
        (loss scalar, gram) where gram is the (mean) Gram used for logging.
    """
    gram = gram_matrix(features, eps=eps)
    if gram.ndim == 2:
        # slogdet is more stable than log(det(.))
        sign, logabsdet = torch.linalg.slogdet(gram)
        # If somehow non-PD, push loss up (rare with eps*I).
        loss = -logabsdet
        if sign <= 0:
            loss = loss + 10.0
        return loss, gram

    signs, logabsdets = torch.linalg.slogdet(gram)  # (N,)
    loss = -logabsdets.mean()
    # Mild penalty if any batch element is not PD.
    bad = (signs <= 0).float().mean()
    loss = loss + 10.0 * bad
    return loss, gram.mean(dim=0)


def dispersion_loss_from_variants(
    fr_callable,
    variants: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """
    End-to-end helper: ``variants (N,K,C,H,W)`` → FR features → dispersion loss.

    ``fr_callable`` must accept ``(B,C,H,W)`` and return ``(B,D)``. FR weights
    should be frozen by the caller (``torch.no_grad`` on params / ``eval``).
    """
    if variants.ndim != 5:
        raise ValueError(f"variants must be (N,K,C,H,W), got {tuple(variants.shape)}")
    n, k, c, h, w = variants.shape
    flat = variants.reshape(n * k, c, h, w)
    feats = fr_callable(flat)  # (N*K, D)
    feats = feats.view(n, k, -1)
    loss, gram = frequency_dispersion_loss(feats, eps=eps)
    with torch.no_grad():
        # Off-diagonal mean |cos| — lower means more dispersed.
        g = gram_matrix(feats, eps=0.0).mean(dim=0)
        k_dim = g.shape[0]
        eye = torch.eye(k_dim, device=g.device, dtype=torch.bool)
        off_mean = g[~eye].abs().mean().item() if k_dim > 1 else 0.0
    stats = {
        "loss": float(loss.detach().cpu()),
        "gram_offdiag_abs_mean": off_mean,
        "feat_norm_mean": float(feats.detach().norm(dim=-1).mean().cpu()),
    }
    return loss, stats
