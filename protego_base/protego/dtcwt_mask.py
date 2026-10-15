"""
Differentiable DT-CWT meta-mask module (Phase 2).

Pipeline
--------
1. ``DTCWTForward`` on an RGB batch ``(N, C, H, W)``.
2. Add a learnable meta-mask in **magnitude / phase** on DT-CWT levels **2 and 3**.
3. Level **1** (finest / highest-frequency band) is **frozen**: no parameters and
   no mask applied — reduces JPEG-compression sensitivity (PLAN.md).
4. ``DTCWTInverse`` reconstructs the image.

Coefficient layout (``pytorch_wavelets``, ``ri_dim=-1``)::

    yh[j] : (N, C, 6, H_j, W_j, 2)   # last dim = (real, imag)
    j = 0 -> Level 1 (frozen)
    j = 1 -> Level 2 (editable)
    j = 2 -> Level 3 (editable)   when J=3
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pytorch_wavelets is required for Phase 2. "
        "Install with: pip install pytorch-wavelets PyWavelets"
    ) from exc


# 1-based DT-CWT levels that receive the meta-mask (PLAN.md).
DEFAULT_EDITABLE_LEVELS: Tuple[int, ...] = (2, 3)


def _spatial_size_for_level(img_size: int, level: int) -> Tuple[int, int]:
    """High-pass spatial size at DT-CWT ``level`` (1-based) for square inputs."""
    # Level 1 -> img_size/2, Level 2 -> img_size/4, ...
    s = img_size // (2**level)
    if s < 1:
        raise ValueError(
            f"img_size={img_size} too small for DT-CWT level {level} (got spatial {s})."
        )
    return s, s


def coeffs_to_mag_phase(band: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert real/imag band ``(..., 2)`` to magnitude and phase."""
    real = band[..., 0]
    imag = band[..., 1]
    mag = torch.sqrt(real * real + imag * imag + eps)
    phase = torch.atan2(imag, real)
    return mag, phase


def mag_phase_to_coeffs(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Convert magnitude/phase back to real/imag band ``(..., 2)``."""
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)


def apply_mag_phase_delta(
    band: torch.Tensor,
    delta_mag: torch.Tensor,
    delta_phase: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Legacy helper: perturb band magnitude/phase additively (Phase 2 tests).

    Prefer :func:`add_complex_meta_mask` for the PLAN formula
    ``W + ΔW · e^{iφ}``.
    """
    mag, phase = coeffs_to_mag_phase(band, eps=eps)
    mag_new = torch.clamp(mag + delta_mag, min=0.0)
    phase_new = phase + delta_phase
    return mag_phase_to_coeffs(mag_new, phase_new)


def polar_to_real_imag(mag: torch.Tensor, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert polar ``(mag, phase)`` of ΔW into real/imag parts."""
    return mag * torch.cos(phase), mag * torch.sin(phase)


def rotate_real_imag(
    real: torch.Tensor, imag: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate complex coeffs by phase ``phi`` (broadcastable)."""
    c, s = torch.cos(phi), torch.sin(phi)
    return real * c - imag * s, real * s + imag * c


def add_complex_meta_mask(
    band: torch.Tensor,
    delta_mag: torch.Tensor,
    delta_phase: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply PLAN formula on one high-pass band: ``W ← W + ΔW · e^{iφ}``.

    ``delta_mag`` / ``delta_phase`` parameterize the complex meta-mask ΔW in polar
    form (shapes broadcastable to ``band`` without the trailing ri dim).
    ``phi`` is an optional extra phase (scalar, ``(N,)``, or fully broadcastable).
    """
    dw_r, dw_i = polar_to_real_imag(delta_mag, delta_phase)
    if phi is not None:
        while phi.ndim < dw_r.ndim:
            phi = phi.unsqueeze(-1)
        dw_r, dw_i = rotate_real_imag(dw_r, dw_i, phi)
    out = band.clone()
    out[..., 0] = out[..., 0] + dw_r
    out[..., 1] = out[..., 1] + dw_i
    return out


class DTCWTMetaMaskModule(nn.Module):
    """
    Image → DT-CWT →(+ meta_mask on L2/L3)→ IDT-CWT → image.

    Parameters
    ----------
    channels :
        Number of image channels (3 for RGB).
    img_size :
        Assumed square spatial size (e.g. 224). Used to size meta-mask tensors.
    J :
        Number of DT-CWT scales (must be >= max(editable_levels)).
    editable_levels :
        1-based levels that own learnable mag/phase deltas. Level 1 must NOT
        be included (frozen by design).
    biort / qshift :
        Filter banks passed to ``pytorch_wavelets``.
    """

    def __init__(
        self,
        channels: int = 3,
        img_size: int = 224,
        J: int = 3,
        editable_levels: Sequence[int] = DEFAULT_EDITABLE_LEVELS,
        biort: str = "near_sym_b",
        qshift: str = "qshift_b",
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.img_size = int(img_size)
        self.J = int(J)
        self.biort = biort
        self.qshift = qshift

        levels = tuple(int(l) for l in editable_levels)
        if any(l < 1 for l in levels):
            raise ValueError(f"editable_levels must be 1-based, got {levels}")
        if 1 in levels:
            raise ValueError(
                "Level 1 must stay frozen (JPEG robustness). "
                "Remove 1 from editable_levels."
            )
        if max(levels) > self.J:
            raise ValueError(f"editable_levels {levels} exceed J={self.J}")
        self.editable_levels: Tuple[int, ...] = levels

        self.xfm = DTCWTForward(J=self.J, biort=biort, qshift=qshift)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift)

        # Learnable universal meta-mask: one (mag, phase) pair per editable level.
        # Shapes match yh[level-1] without the batch dim: (1, C, 6, H, W).
        self.delta_mag = nn.ParameterDict()
        self.delta_phase = nn.ParameterDict()
        for level in self.editable_levels:
            h, w = _spatial_size_for_level(self.img_size, level)
            key = str(level)
            mag = torch.zeros(1, self.channels, 6, h, w)
            phase = torch.zeros(1, self.channels, 6, h, w)
            if init_std > 0:
                mag.normal_(0.0, init_std)
                phase.normal_(0.0, init_std)
            self.delta_mag[key] = nn.Parameter(mag)
            self.delta_phase[key] = nn.Parameter(phase)

    @property
    def frozen_levels(self) -> Tuple[int, ...]:
        """1-based levels with no meta-mask parameters (includes level 1)."""
        return tuple(l for l in range(1, self.J + 1) if l not in self.editable_levels)

    def meta_mask_state(self) -> Dict[str, torch.Tensor]:
        """Return a detachable copy of mag/phase deltas for checkpointing."""
        state: Dict[str, torch.Tensor] = {}
        for level in self.editable_levels:
            key = str(level)
            state[f"delta_mag_{key}"] = self.delta_mag[key].detach().cpu().clone()
            state[f"delta_phase_{key}"] = self.delta_phase[key].detach().cpu().clone()
        return state

    def load_meta_mask_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Load mag/phase deltas from ``meta_mask_state()`` / a ``.pt`` checkpoint."""
        for level in self.editable_levels:
            key = str(level)
            self.delta_mag[key].data.copy_(state[f"delta_mag_{key}"].to(self.delta_mag[key].device))
            self.delta_phase[key].data.copy_(
                state[f"delta_phase_{key}"].to(self.delta_phase[key].device)
            )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shaped (N,C,H,W), got {tuple(x.shape)}")
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            return F.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        return x

    def _apply_mask_to_yh(
        self,
        yh: List[torch.Tensor],
        phi: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply ``W + ΔW e^{iφ}`` on editable levels; leave Level 1 untouched."""
        yh_out: List[torch.Tensor] = []
        for j, band in enumerate(yh):
            level = j + 1
            if level not in self.editable_levels:
                yh_out.append(band)
                continue
            key = str(level)
            yh_out.append(
                add_complex_meta_mask(
                    band,
                    self.delta_mag[key],
                    self.delta_phase[key],
                    phi=phi,
                )
            )
        return yh_out

    def forward(
        self,
        x: torch.Tensor,
        extra_phase: Optional[torch.Tensor] = None,
        return_coeffs: bool = False,
    ):
        """
        Apply the meta-mask in the DT-CWT domain and reconstruct.

        Uses ``W_prot = W_x + ΔW_meta · e^{iφ}`` on levels 2–3 (Level 1 frozen).

        Args:
            x: ``(N, C, H, W)`` float tensor (typically [0, 1]).
            extra_phase: optional phase ``φ`` (scalar / ``(N,)`` / broadcastable).
            return_coeffs: if True, also return ``(yl, yh_out)``.
        """
        x_in = self._prepare_input(x)
        yl, yh = self.xfm(x_in)
        yh_out = self._apply_mask_to_yh(yh, phi=extra_phase)
        recon = self.ifm((yl, yh_out))

        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if return_coeffs:
            return recon, yl, yh_out
        return recon

    def forward_k_variants(
        self,
        x: torch.Tensor,
        phases: torch.Tensor,
        clamp01: bool = True,
    ) -> torch.Tensor:
        """
        Build the local mini-batch of K phase variants (Phase 3).

        Args:
            x: ``(N, C, H, W)``
            phases: ``(K,)`` phase shifts ``φ_k`` applied as ``ΔW · e^{iφ_k}``.

        Returns:
            ``(N, K, C, H, W)`` reconstructed protected images.
        """
        if phases.ndim != 1:
            raise ValueError(f"phases must be 1-D (K,), got {tuple(phases.shape)}")
        x_in = self._prepare_input(x)
        n = x_in.shape[0]
        k = int(phases.shape[0])
        yl, yh = self.xfm(x_in)

        # Tile batch: [x0,x0,...,x1,x1,...] with K copies each → (N*K, ...)
        yl_rep = yl.repeat_interleave(k, dim=0)
        yh_rep = [band.repeat_interleave(k, dim=0) for band in yh]
        # phases for tiled batch: [φ0..φ_{K-1}, φ0..φ_{K-1}, ...]
        phi = phases.to(device=x_in.device, dtype=x_in.dtype).repeat(n)  # (N*K,)

        yh_out = self._apply_mask_to_yh(yh_rep, phi=phi)
        recon = self.ifm((yl_rep, yh_out))  # (N*K, C, H, W)
        if clamp01:
            recon = recon.clamp(0.0, 1.0)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        c, h, w = recon.shape[1:]
        return recon.view(n, k, c, h, w)

    def zero_level1_mask_contribution(self) -> None:
        """
        Documentation/helper: level 1 has no parameters by construction.

        Exposed so Phase-2 tests can assert the design invariant explicitly.
        """
        assert 1 not in self.editable_levels
        assert "1" not in self.delta_mag
        assert "1" not in self.delta_phase


def perfect_reconstruction_error(
    module: DTCWTMetaMaskModule,
    x: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute reconstruction error with a zero meta-mask (identity path)."""
    was_training = module.training
    module.eval()
    # Temporarily zero deltas without mutating caller state permanently.
    saved = module.meta_mask_state()
    with torch.no_grad():
        for level in module.editable_levels:
            module.delta_mag[str(level)].zero_()
            module.delta_phase[str(level)].zero_()
        y = module(x)
        err = (y - x).abs().mean()
        module.load_meta_mask_state(saved)
    if was_training:
        module.train()
    return err
