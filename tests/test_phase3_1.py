#!/usr/bin/env python3
"""
Phase 3.1 functional test — quality controls on DT-CWT meta-mask training.

Validates:
  1. SSIM hinge is zero when SSIM is high and positive when SSIM is low.
  2. Post-clamp residual satisfies L∞ ≤ epsilon.
  3. Smoke training with quality writes phase3_1 dirs and SSIM CSV columns.
  4. Gradients reach meta-mask only (FR frozen) under quality path.

Run:
    source .venv/bin/activate
    bash scripts/run_phase3_1_test.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTEGO_BASE = os.path.join(REPO_ROOT, "protego_base")
if PROTEGO_BASE not in sys.path:
    sys.path.insert(0, PROTEGO_BASE)

MASK_ROOT = "/mnt/study-data/dcarvalho/tests"
METRICS_ROOT = "/mnt/study-data/dcarvalho/metrics"


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def test_ssim_hinge() -> None:
    print("\n== 1) SSIM hinge zero / positive ==")
    import torch
    from protego.quality_loss import adapt_percep_weight, ssim_hinge_loss

    x = torch.rand(2, 3, 64, 64)
    term_hi, ssim_hi = ssim_hinge_loss(x, x, min_ssim=0.95, weight=1.0)
    if float(term_hi) > 1e-6:
        _fail(f"Expected zero hinge on identical images, got {float(term_hi)}")
    if float(ssim_hi) < 0.99:
        _fail(f"Expected SSIM≈1 on identical images, got {float(ssim_hi)}")

    noisy = (x + 0.5).clamp(0, 1)
    term_lo, ssim_lo = ssim_hinge_loss(noisy, x, min_ssim=0.95, weight=2.0)
    if float(ssim_lo) >= 0.95:
        _fail(f"Expected low SSIM for heavy noise, got {float(ssim_lo)}")
    if float(term_lo) <= 0:
        _fail(f"Expected positive hinge, got {float(term_lo)}")
    expected = 2.0 * max(0.95 - float(ssim_lo), 0.0)
    if abs(float(term_lo) - expected) > 1e-5:
        _fail(f"Hinge mismatch: {float(term_lo)} vs {expected}")

    # 5-D variants path
    prot5 = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
    term5, ssim5 = ssim_hinge_loss(prot5, x, min_ssim=0.95, weight=1.0)
    if float(term5) > 1e-6 or float(ssim5) < 0.99:
        _fail("5-D identical variants should have zero hinge / high SSIM")

    w = adapt_percep_weight(0.90, min_ssim=0.95, weight=1.0)
    if w != 2.0:
        _fail(f"Expected weight×2 when SSIM low, got {w}")
    w2 = adapt_percep_weight(0.98, min_ssim=0.95, weight=2.0)
    if w2 != 1.0:
        _fail(f"Expected weight÷2 when SSIM high, got {w2}")
    w3 = adapt_percep_weight(0.95, min_ssim=0.95, weight=4.0)
    if w3 != 1.0:
        _fail(f"Expected weight reset to 1 near target, got {w3}")

    _ok(f"ssim_hi={float(ssim_hi):.4f} ssim_lo={float(ssim_lo):.4f} hinge={float(term_lo):.4f}")


def test_linf_budget() -> None:
    print("\n== 2) L∞ residual clamp ==")
    import torch
    from protego.quality_loss import apply_linf_budget

    eps = 16.0 / 255.0
    orig = torch.rand(2, 3, 32, 32)
    recon = orig + 0.5  # far beyond epsilon
    prot, residual = apply_linf_budget(recon, orig, eps)
    linf = float(residual.abs().amax())
    if linf > eps + 1e-6:
        _fail(f"residual L∞={linf} exceeds epsilon={eps}")
    if float((prot - (orig + residual).clamp(0, 1)).abs().amax()) > 1e-6:
        _fail("protected != orig + residual (clamped)")

    # 5-D broadcast
    recon5 = orig.unsqueeze(1) + torch.tensor([0.5, -0.5]).view(1, 2, 1, 1, 1)
    prot5, res5 = apply_linf_budget(recon5, orig, eps)
    if float(res5.abs().amax()) > eps + 1e-6:
        _fail("5-D residual exceeds epsilon")
    if prot5.shape != (2, 2, 3, 32, 32):
        _fail(f"Unexpected protected shape {tuple(prot5.shape)}")

    _ok(f"linf={linf:.6f} <= epsilon={eps:.6f}")


def test_quality_grad(device: str) -> None:
    print("\n== 3) Quality path grads → meta-mask only ==")
    import torch
    from protego.dispersion_loss import dispersion_loss_from_variants
    from protego.dtcwt_mask import DTCWTMetaMaskModule
    from protego.FacialRecognition import FR
    from protego.quality_loss import apply_linf_budget, ssim_hinge_loss

    device_t = torch.device(device)
    mask = DTCWTMetaMaskModule(img_size=224, init_std=1e-2).to(device_t)
    for p in list(mask.xfm.parameters()) + list(mask.ifm.parameters()):
        p.requires_grad_(False)

    fr = FR(model_name="inception_facenet_vgg", device=device_t)
    fr.fr_model.model.eval()
    for p in fr.fr_model.model.parameters():
        p.requires_grad_(False)

    x = torch.rand(2, 3, 224, 224, device=device_t)
    phases = torch.linspace(-0.1, 0.1, steps=3, device=device_t)
    recon = mask.forward_k_variants(x, phases, clamp01=False)
    prot, _ = apply_linf_budget(recon, x, epsilon=16.0 / 255.0)

    def _fr(batch: torch.Tensor) -> torch.Tensor:
        return fr(batch)

    disp, _stats = dispersion_loss_from_variants(_fr, prot, eps=1e-4)
    percep, ssim_v = ssim_hinge_loss(prot, x, min_ssim=0.95, weight=1.0)
    loss = disp + percep
    loss.backward()

    g2 = mask.delta_mag["2"].grad
    g3 = mask.delta_mag["3"].grad
    if g2 is None or g3 is None or (g2.abs().sum() + g3.abs().sum()) <= 0:
        _fail("No gradient reached meta-mask mag parameters")
    for p in fr.fr_model.model.parameters():
        if p.grad is not None:
            _fail("FR received gradients")
    _ok(f"loss={float(loss):.4f} ssim={float(ssim_v):.4f} mask grads OK")


def test_smoke_train_quality(device: str) -> str:
    print("\n== 4) Smoke train with quality → phase3_1 artifacts ==")
    cmd = [
        sys.executable,
        os.path.join(PROTEGO_BASE, "train_meta_mask.py"),
        "--smoke",
        "--quality",
        "--exp_name",
        "func_test_q",
        "--device",
        device,
        "--dataset",
        "lfw_frontal",
        "--fr_model",
        "inception_facenet_vgg",
        "--max_train",
        "8",
        "--epsilon",
        str(16.0 / 255.0),
        "--min_ssim",
        "0.95",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PROTEGO_BASE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        _fail(f"train_meta_mask.py --quality failed with code {proc.returncode}")

    ptr = os.path.join(MASK_ROOT, "LATEST_phase3_1_smoke_func_test_q.txt")
    if not os.path.isfile(ptr):
        _fail(f"Missing latest pointer {ptr}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    with open(ptr, encoding="utf-8") as f:
        mask_path = f.read().strip()
    if not os.path.isfile(mask_path):
        _fail(f"Mask file missing: {mask_path}")

    mask_dir = os.path.dirname(mask_path)
    run_id = os.path.basename(mask_dir)
    if "phase3_1" not in run_id:
        _fail(f"Expected phase3_1 in run_id, got {run_id}")

    metrics_dir = os.path.join(METRICS_ROOT, run_id)
    log_csv = os.path.join(metrics_dir, "train_log.csv")
    for required in (
        os.path.join(mask_dir, "meta_mask.pt"),
        os.path.join(mask_dir, "config.json"),
        log_csv,
        os.path.join(metrics_dir, "summary.json"),
        os.path.join(metrics_dir, "config.json"),
    ):
        if not os.path.isfile(required):
            _fail(f"Missing artifact: {required}")

    with open(os.path.join(metrics_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("quality"):
        _fail("config.quality should be true")
    if cfg.get("phase") != "phase3_1":
        _fail(f"config.phase should be phase3_1, got {cfg.get('phase')}")

    with open(log_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _fail("train_log.csv is empty")
    required_cols = {"ssim", "percep_term", "percep_weight", "linf_max", "disp_loss", "loss"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        _fail(f"CSV missing columns: {missing}")
    # L∞ of logged residual should not exceed epsilon (allow tiny float slack).
    eps = float(cfg["epsilon"])
    for row in rows:
        if float(row["linf_max"]) > eps + 1e-5:
            _fail(f"logged linf_max={row['linf_max']} > epsilon={eps}")

    with open(os.path.join(metrics_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("final_ssim") is None:
        _fail("summary.final_ssim missing")

    ckpt = __import__("torch").load(mask_path, map_location="cpu", weights_only=False)
    assert "meta_mask" in ckpt and "delta_mag_2" in ckpt["meta_mask"]
    _ok(f"smoke run_id={run_id} final_ssim={summary['final_ssim']:.4f}")
    _ok(f"mask -> {mask_path}")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip smoke train subprocess (unit tests only).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3.1 functional test — quality meta-mask training")
    print("=" * 60)

    failures = []
    steps = [
        ("ssim_hinge", test_ssim_hinge),
        ("linf_budget", test_linf_budget),
        ("quality_grad", lambda: test_quality_grad(args.device)),
    ]
    if not args.skip_train:
        steps.append(("smoke_train_quality", lambda: test_smoke_train_quality(args.device)))

    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"\n!! Step '{name}' failed: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Phase 3.1 RESULT: FAILED ({len(failures)} step(s))")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print("Phase 3.1 RESULT: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
