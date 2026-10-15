"""
Phase 3.1 — pixel-space quality controls for DT-CWT meta-mask training.

Mirrors Protego's SSIM hinge + adaptive weight and L∞ budget, applied after
IDT-CWT reconstruction (not in wavelet coefficient space).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from pytorch_msssim import ssim as calc_ssim


def apply_linf_budget(
    recon: torch.Tensor,
    orig: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clamp spatial residual to ``[-epsilon, epsilon]`` then clamp pixels to ``[0, 1]``.

    Args:
        recon: reconstructed images ``(N, C, H, W)`` or ``(N, K, C, H, W)``.
        orig: originals ``(N, C, H, W)`` (broadcast over K when recon is 5-D).
        epsilon: L∞ budget (same spirit as Protego ``cfgs.epsilon``).

    Returns:
        (protected, residual) with the same shape as ``recon``.
        ``residual`` is the clamped perturbation before the final ``[0, 1]`` clamp.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    if recon.ndim == 5:
        if orig.ndim != 4:
            raise ValueError(
                f"orig must be (N,C,H,W) when recon is (N,K,C,H,W), got {tuple(orig.shape)}"
            )
        if recon.shape[0] != orig.shape[0] or recon.shape[2:] != orig.shape[1:]:
            raise ValueError(
                f"shape mismatch recon={tuple(recon.shape)} orig={tuple(orig.shape)}"
            )
        orig_b = orig.unsqueeze(1)  # (N, 1, C, H, W)
    elif recon.ndim == 4:
        if orig.shape != recon.shape:
            raise ValueError(
                f"shape mismatch recon={tuple(recon.shape)} orig={tuple(orig.shape)}"
            )
        orig_b = orig
    else:
        raise ValueError(f"recon must be 4-D or 5-D, got {tuple(recon.shape)}")

    residual = (recon - orig_b).clamp(-epsilon, epsilon)
    protected = (orig_b + residual).clamp(0.0, 1.0)
    return protected, residual


def ssim_hinge_loss(
    protected: torch.Tensor,
    orig: torch.Tensor,
    min_ssim: float,
    weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Protego-style perceptual hinge: ``weight * max(min_ssim - SSIM, 0)``.

    Args:
        protected: ``(N, C, H, W)`` or ``(N, K, C, H, W)``.
        orig: ``(N, C, H, W)`` (broadcast over K when needed).
        min_ssim: target lower bound (e.g. 0.95).
        weight: adaptive perceptual weight ``w``.

    Returns:
        (percep_term scalar tensor with grad, ssim_value scalar tensor detached-safe).
    """
    if protected.ndim == 5:
        n, k, c, h, w = protected.shape
        prot_flat = protected.reshape(n * k, c, h, w)
        orig_flat = orig.unsqueeze(1).expand(n, k, c, h, w).reshape(n * k, c, h, w)
    elif protected.ndim == 4:
        prot_flat = protected
        orig_flat = orig
    else:
        raise ValueError(f"protected must be 4-D or 5-D, got {tuple(protected.shape)}")

    ssim_value = calc_ssim(prot_flat, orig_flat, data_range=1.0, size_average=True)
    zero = torch.zeros((), device=ssim_value.device, dtype=ssim_value.dtype)
    percep_term = float(weight) * torch.max(min_ssim - ssim_value, zero)
    return percep_term, ssim_value


def adapt_percep_weight(
    ssim_value: Union[float, torch.Tensor],
    min_ssim: float,
    weight: float,
    min_weight: float = 1.0 / 129.0,
    max_weight: float = 129.0,
) -> float:
    """
    Protego adaptive weight: ×2 if SSIM too low, ÷2 if too high, else reset to 1.

    Thresholds: ``min_ssim * 0.99`` / ``min_ssim * 1.01``.
    """
    s = float(ssim_value.detach().cpu()) if isinstance(ssim_value, torch.Tensor) else float(ssim_value)
    w = float(weight)
    if s >= min_ssim * 1.01 and w > min_weight:
        return w / 2.0
    if s <= min_ssim * 0.99 and w <= max_weight:
        return w * 2.0
    if min_ssim * 0.99 < s < min_ssim * 1.01:
        return 1.0
    return w
