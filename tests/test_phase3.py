#!/usr/bin/env python3
"""
Phase 3 functional test — Frequency Dispersion Loss + offline meta-mask train.

Validates:
  1. ``-log det G`` is finite and differentiable w.r.t. meta-mask.
  2. Smoke training writes versioned dirs under study-data (no overwrite).
  3. Only meta-mask parameters receive gradients (FR frozen).
  4. Two consecutive smoke runs get distinct folder names.

Run:
    source .venv/bin/activate
    bash scripts/run_phase3_test.sh
"""

from __future__ import annotations

import argparse
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


def test_dispersion_loss_grad(device: str) -> None:
    print("\n== 1) Frequency Dispersion Loss + grad to meta-mask ==")
    import torch
    from protego.dispersion_loss import frequency_dispersion_loss
    from protego.dtcwt_mask import DTCWTMetaMaskModule
    from protego.FacialRecognition import FR

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
    variants = mask.forward_k_variants(x, phases)
    assert variants.shape == (2, 3, 3, 224, 224), variants.shape

    flat = variants.reshape(2 * 3, 3, 224, 224)
    feats = fr(flat).view(2, 3, -1)
    loss, gram = frequency_dispersion_loss(feats)
    assert torch.isfinite(loss), float(loss)
    loss.backward()

    g2 = mask.delta_mag["2"].grad
    g3 = mask.delta_mag["3"].grad
    if g2 is None or g3 is None or (g2.abs().sum() + g3.abs().sum()) <= 0:
        _fail("No gradient reached meta-mask mag parameters")
    for p in fr.fr_model.model.parameters():
        if p.grad is not None:
            _fail("FR received gradients")
    _ok(f"loss={float(loss):.4f} gram={tuple(gram.shape)} mask grads OK")


def test_versioned_dirs() -> None:
    print("\n== 2) Versioned run dirs (no overwrite) ==")
    from protego.paths import make_run_dirs, sanitize_exp_name

    assert sanitize_exp_name("a/b c") == "a_b_c"
    a_id, a_mask, a_met = make_run_dirs("phase3", "unit_dirs_a", MASK_ROOT, METRICS_ROOT)
    time.sleep(1.1)  # ensure timestamp second ticks
    b_id, b_mask, b_met = make_run_dirs("phase3", "unit_dirs_b", MASK_ROOT, METRICS_ROOT)
    assert a_id != b_id
    assert a_mask != b_mask and a_met != b_met
    assert os.path.isdir(a_mask) and os.path.isdir(a_met)
    assert a_mask.startswith(MASK_ROOT)
    assert a_met.startswith(METRICS_ROOT)
    assert "phase3" in a_id
    _ok(f"created distinct runs: {a_id} vs {b_id}")
    # cleanup unit dirs
    for d in (a_mask, a_met, b_mask, b_met):
        try:
            os.rmdir(d)
        except OSError:
            pass


def test_smoke_train(device: str) -> str:
    print("\n== 3) Smoke training writes mask + metrics ==")
    cmd = [
        sys.executable,
        os.path.join(PROTEGO_BASE, "train_meta_mask.py"),
        "--smoke",
        "--no_quality",
        "--exp_name",
        "func_test",
        "--device",
        device,
        "--dataset",
        "lfw_frontal",
        "--fr_model",
        "inception_facenet_vgg",
        "--max_train",
        "8",
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
        _fail(f"train_meta_mask.py failed with code {proc.returncode}")

    # Locate the run via LATEST pointer written by the trainer.
    ptr = os.path.join(MASK_ROOT, "LATEST_phase3_smoke_func_test.txt")
    if not os.path.isfile(ptr):
        _fail(f"Missing latest pointer {ptr}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    with open(ptr, encoding="utf-8") as f:
        mask_path = f.read().strip()
    if not os.path.isfile(mask_path):
        _fail(f"Mask file missing: {mask_path}")
    mask_dir = os.path.dirname(mask_path)
    run_id = os.path.basename(mask_dir)
    metrics_dir = os.path.join(METRICS_ROOT, run_id)
    for required in (
        os.path.join(mask_dir, "meta_mask.pt"),
        os.path.join(mask_dir, "config.json"),
        os.path.join(metrics_dir, "train_log.csv"),
        os.path.join(metrics_dir, "summary.json"),
        os.path.join(metrics_dir, "config.json"),
    ):
        if not os.path.isfile(required):
            _fail(f"Missing artifact: {required}")

    with open(os.path.join(metrics_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    assert summary.get("final_loss") is not None
    ckpt = __import__("torch").load(mask_path, map_location="cpu", weights_only=False)
    assert "meta_mask" in ckpt and "delta_mag_2" in ckpt["meta_mask"]
    _ok(f"smoke run_id={run_id} final_loss={summary['final_loss']:.4f}")
    _ok(f"mask -> {mask_path}")
    _ok(f"metrics -> {metrics_dir}")
    return run_id


def test_second_run_unique(device: str, first_run_id: str) -> None:
    print("\n== 4) Second smoke run gets a unique folder ==")
    time.sleep(1.1)
    cmd = [
        sys.executable,
        os.path.join(PROTEGO_BASE, "train_meta_mask.py"),
        "--smoke",
        "--no_quality",
        "--exp_name",
        "func_test",
        "--device",
        device,
        "--max_train",
        "8",
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
        _fail("second smoke train failed")
    ptr = os.path.join(MASK_ROOT, "LATEST_phase3_smoke_func_test.txt")
    with open(ptr, encoding="utf-8") as f:
        mask_path = f.read().strip()
    second_id = os.path.basename(os.path.dirname(mask_path))
    if second_id == first_run_id:
        _fail(f"Second run reused folder {first_run_id}")
    # Both dirs must still exist (no overwrite/delete of the first).
    if not os.path.isdir(os.path.join(MASK_ROOT, first_run_id)):
        _fail("First run mask dir disappeared")
    if not os.path.isdir(os.path.join(METRICS_ROOT, first_run_id)):
        _fail("First run metrics dir disappeared")
    _ok(f"unique runs preserved: {first_run_id} and {second_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only unit-test loss/dirs (skip smoke train subprocess).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 functional test — meta-mask training")
    print("=" * 60)

    failures = []
    first_run = None
    steps = [
        ("dispersion_grad", lambda: test_dispersion_loss_grad(args.device)),
        ("versioned_dirs", test_versioned_dirs),
    ]
    if not args.skip_train:
        def _smoke():
            nonlocal first_run
            first_run = test_smoke_train(args.device)

        def _unique():
            assert first_run is not None
            test_second_run_unique(args.device, first_run)

        steps.append(("smoke_train", _smoke))
        steps.append(("unique_runs", _unique))

    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"\n!! Step '{name}' failed: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Phase 3 RESULT: FAILED ({len(failures)} step(s))")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print("Phase 3 RESULT: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
