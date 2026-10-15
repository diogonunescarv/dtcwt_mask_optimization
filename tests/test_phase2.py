#!/usr/bin/env python3
"""
Phase 2 functional test — differentiable DT-CWT meta-mask module.

Validates:
  1. Near-perfect reconstruction with a zero meta-mask.
  2. Level 1 has no learnable parameters (frozen for JPEG robustness).
  3. Gradients flow from the reconstructed image back to L2/L3 mag/phase masks.
  4. Level-1 coefficients are unchanged by the meta-mask path.

Run:
    source .venv/bin/activate
    bash scripts/run_phase2_test.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTEGO_BASE = os.path.join(REPO_ROOT, "protego_base")
if PROTEGO_BASE not in sys.path:
    sys.path.insert(0, PROTEGO_BASE)


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def test_level1_frozen(module) -> None:
    print("\n== 1) Level 1 frozen (no parameters) ==")
    assert 1 not in module.editable_levels
    assert "1" not in module.delta_mag
    assert "1" not in module.delta_phase
    module.zero_level1_mask_contribution()
    names = [n for n, _ in module.named_parameters()]
    assert not any("delta_mag.1" in n or "delta_phase.1" in n or n.endswith(".1") for n in names)
    for n in names:
        assert "2" in n or "3" in n, n
    _ok(f"Only L2/L3 params present: {names}")
    _ok(f"frozen_levels={module.frozen_levels}")


def test_perfect_reconstruction(module, device, batch: int = 2) -> None:
    print("\n== 2) Near-perfect reconstruction (zero mask) ==")
    import torch
    from protego.dtcwt_mask import perfect_reconstruction_error

    x = torch.rand(batch, 3, module.img_size, module.img_size, device=device)
    err = perfect_reconstruction_error(module, x).item()
    if err > 1e-5:
        _fail(f"Reconstruction MAE too high: {err}")
    _ok(f"zero-mask MAE={err:.3e}")


def test_level1_coeffs_unchanged(module, device) -> None:
    print("\n== 3) Level-1 coefficients unchanged by meta-mask ==")
    import torch

    # Non-trivial mask on L2/L3
    with torch.no_grad():
        module.delta_mag["2"].uniform_(-0.05, 0.05)
        module.delta_phase["2"].uniform_(-0.1, 0.1)
        module.delta_mag["3"].uniform_(-0.05, 0.05)
        module.delta_phase["3"].uniform_(-0.1, 0.1)

    x = torch.rand(1, 3, module.img_size, module.img_size, device=device)
    with torch.no_grad():
        yl, yh = module.xfm(x)
        _, _, yh_masked = module(x, return_coeffs=True)

    # yh[0] == level 1
    diff = (yh_masked[0] - yh[0]).abs().max().item()
    if diff > 1e-6:
        _fail(f"Level-1 coeffs changed by meta-mask (max abs diff={diff})")
    _ok(f"yh[0] (Level 1) identical (max abs diff={diff:.3e})")

    # L2/L3 should differ
    d2 = (yh_masked[1] - yh[1]).abs().max().item()
    d3 = (yh_masked[2] - yh[2]).abs().max().item()
    if d2 < 1e-8 and d3 < 1e-8:
        _fail("L2/L3 coeffs unchanged despite non-zero meta-mask")
    _ok(f"L2/L3 coeffs modified (max|ΔL2|={d2:.3e}, max|ΔL3|={d3:.3e})")


def test_gradient_flow(module, device) -> None:
    print("\n== 4) Gradient flow: recon → meta_mask (L2/L3) ==")
    import torch

    module.zero_grad(set_to_none=True)
    # Start from small random deltas so path is non-degenerate.
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(-0.01, 0.01)

    x = torch.rand(2, 3, module.img_size, module.img_size, device=device)
    # Downstream loss on reconstructed pixels (proxy for FR loss in later phases).
    recon = module(x)
    # Encourage sensitivity to the mask: distance from original.
    loss = (recon - x).pow(2).mean()
    loss.backward()

    for level in (2, 3):
        g_mag = module.delta_mag[str(level)].grad
        g_phase = module.delta_phase[str(level)].grad
        if g_mag is None or g_phase is None:
            _fail(f"Level {level}: grad is None")
        mag_norm = g_mag.abs().sum().item()
        phase_norm = g_phase.abs().sum().item()
        if mag_norm <= 0.0 and phase_norm <= 0.0:
            _fail(f"Level {level}: both mag and phase grads are zero")
        _ok(
            f"Level {level}: ||∂L/∂mag||1={mag_norm:.4e}, "
            f"||∂L/∂phase||1={phase_norm:.4e}"
        )

    # Sanity: no accidental level-1 Parameter appeared.
    assert "1" not in module.delta_mag
    _ok("No Level-1 parameters received gradients (none exist)")


def test_real_face_smoke(module, device) -> None:
    print("\n== 5) Smoke test on a real LFW face ==")
    import torch
    from protego.datasets import list_image_paths, resolve_split_dir
    from protego.utils import load_imgs

    paths = list_image_paths(resolve_split_dir("lfw_frontal", "train"))[:2]
    imgs = load_imgs(img_paths=paths, img_sz=module.img_size, drange=1, device=device)
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(-1e-3, 1e-3)
    recon = module(imgs)
    loss = recon.mean()
    loss.backward()
    assert recon.shape == imgs.shape
    assert module.delta_mag["2"].grad is not None
    _ok(f"LFW batch recon shape={tuple(recon.shape)}, grad L2 mag OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 functional tests")
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 functional test — DT-CWT meta-mask")
    print("=" * 60)

    import torch
    from protego.dtcwt_mask import DTCWTMetaMaskModule

    device = torch.device(args.device)
    module = DTCWTMetaMaskModule(
        channels=3, img_size=args.img_size, J=3, editable_levels=(2, 3)
    ).to(device)

    failures = []
    steps = [
        ("level1_frozen", lambda: test_level1_frozen(module)),
        ("recon", lambda: test_perfect_reconstruction(module, device)),
        ("l1_unchanged", lambda: test_level1_coeffs_unchanged(module, device)),
        ("grads", lambda: test_gradient_flow(module, device)),
        ("lfw_smoke", lambda: test_real_face_smoke(module, device)),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"\n!! Step '{name}' failed: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Phase 2 RESULT: FAILED ({len(failures)} step(s))")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print("Phase 2 RESULT: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
