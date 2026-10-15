#!/usr/bin/env python3
"""
Phase 3.2 functional test — identity cos + batch Gram + multi-FR ensemble.

Validates:
  1. identity_cosine_loss is finite and lower=better (identical → ~1).
  2. Batch Gram on (N,D) features is differentiable.
  3. Ensemble mean path sends grads to meta-mask only (FR frozen).
  4. Smoke train with 2 FRs writes phase3_2 artifacts + cos_similarity CSV.

Run:
    source .venv/bin/activate
    bash scripts/run_phase3_2_test.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTEGO_BASE = os.path.join(REPO_ROOT, "protego_base")
if PROTEGO_BASE not in sys.path:
    sys.path.insert(0, PROTEGO_BASE)

MASK_ROOT = "/mnt/study-data/dcarvalho/tests"
METRICS_ROOT = "/mnt/study-data/dcarvalho/metrics"

# Light pair for smoke (time / VRAM).
SMOKE_FR_MODELS = "inception_facenet_vgg,inception_facenet_casia"


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def test_identity_cosine() -> None:
    print("\n== 1) identity_cosine_loss (lower = better de-ID) ==")
    import torch
    from protego.identity_loss import identity_cosine_loss

    a = torch.randn(4, 32)
    loss_same, sim_same = identity_cosine_loss(a, a.clone())
    if float(sim_same) < 0.99:
        _fail(f"identical features should have cos≈1, got {float(sim_same)}")
    b = -a
    loss_opp, sim_opp = identity_cosine_loss(a, b)
    if float(sim_opp) > float(sim_same):
        _fail("opposite features should have lower cos than identical")
    if not torch.isfinite(loss_same):
        _fail("identity loss not finite")
    _ok(f"cos_same={float(sim_same):.4f} cos_opp={float(sim_opp):.4f}")


def test_batch_gram() -> None:
    print("\n== 2) Batch Gram dispersion on (N,D) ==")
    import torch
    from protego.dispersion_loss import frequency_dispersion_loss

    feats = torch.randn(4, 64, requires_grad=True)
    loss, gram = frequency_dispersion_loss(feats, eps=1e-4)
    if gram.shape != (4, 4):
        _fail(f"expected gram (4,4), got {tuple(gram.shape)}")
    loss.backward()
    if feats.grad is None or feats.grad.abs().sum() <= 0:
        _fail("no grad on batch features")
    _ok(f"loss={float(loss):.4f} gram={tuple(gram.shape)}")


def test_ensemble_grad(device: str) -> None:
    print("\n== 3) Ensemble path grads → meta-mask only ==")
    import torch
    from protego.dispersion_loss import dispersion_loss_from_variants, frequency_dispersion_loss
    from protego.dtcwt_mask import DTCWTMetaMaskModule
    from protego.FacialRecognition import FR
    from protego.identity_loss import identity_cosine_loss
    from protego.quality_loss import apply_linf_budget, ssim_hinge_loss

    device_t = torch.device(device)
    mask = DTCWTMetaMaskModule(img_size=224, init_std=1e-2).to(device_t)
    for p in list(mask.xfm.parameters()) + list(mask.ifm.parameters()):
        p.requires_grad_(False)

    fr_names = [n.strip() for n in SMOKE_FR_MODELS.split(",")]
    frs = []
    for name in fr_names:
        fr = FR(model_name=name, device=device_t)
        fr.fr_model.model.eval()
        for p in fr.fr_model.model.parameters():
            p.requires_grad_(False)
        frs.append(fr)

    x = torch.rand(2, 3, 224, 224, device=device_t)
    phases = torch.linspace(-0.1, 0.1, steps=3, device=device_t)
    variants = mask.forward_k_variants(x, phases, clamp01=False)
    variants, _ = apply_linf_budget(variants, x, epsilon=16.0 / 255.0)
    recon0 = mask(x, extra_phase=torch.zeros((), device=device_t))
    prot0, _ = apply_linf_budget(recon0, x, epsilon=16.0 / 255.0)

    ensemble = []
    for fr in frs:

        def _fr(batch, _fr=fr):
            return _fr(batch)

        lk, _ = dispersion_loss_from_variants(_fr, variants, eps=1e-4)
        fp = fr(prot0)
        fo = fr(x)
        lb, _ = frequency_dispersion_loss(fp, eps=1e-4)
        lid, _ = identity_cosine_loss(fo, fp)
        ensemble.append(lk + lb + lid)

    percep, ssim_v = ssim_hinge_loss(prot0, x, min_ssim=0.95, weight=1.0)
    loss = torch.mean(torch.stack(ensemble)) + percep
    loss.backward()

    g2 = mask.delta_mag["2"].grad
    g3 = mask.delta_mag["3"].grad
    if g2 is None or g3 is None or (g2.abs().sum() + g3.abs().sum()) <= 0:
        _fail("No gradient reached meta-mask")
    for fr in frs:
        for p in fr.fr_model.model.parameters():
            if p.grad is not None:
                _fail("FR received gradients")
    _ok(
        f"loss={float(loss):.4f} ssim={float(ssim_v):.4f} n_frs={len(frs)} mask grads OK"
    )


def test_smoke_train(device: str) -> str:
    print("\n== 4) Smoke train phase3_2 (2 FRs) ==")
    cmd = [
        sys.executable,
        os.path.join(PROTEGO_BASE, "train_meta_mask.py"),
        "--smoke",
        "--phase32",
        "--quality",
        "--exp_name",
        "func_test_p32",
        "--device",
        device,
        "--dataset",
        "lfw_frontal",
        "--fr_models",
        SMOKE_FR_MODELS,
        "--max_train",
        "8",
        "--batch_size",
        "2",
        "--K",
        "3",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PROTEGO_BASE + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        _fail(f"train_meta_mask.py --phase32 failed code={proc.returncode}")

    ptr = os.path.join(MASK_ROOT, "LATEST_phase3_2_smoke_func_test_p32.txt")
    if not os.path.isfile(ptr):
        _fail(f"Missing pointer {ptr}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    with open(ptr, encoding="utf-8") as f:
        mask_path = f.read().strip()
    if not os.path.isfile(mask_path):
        _fail(f"Mask missing: {mask_path}")

    mask_dir = os.path.dirname(mask_path)
    run_id = os.path.basename(mask_dir)
    if "phase3_2" not in run_id:
        _fail(f"Expected phase3_2 in run_id, got {run_id}")

    metrics_dir = os.path.join(METRICS_ROOT, run_id)
    log_csv = os.path.join(metrics_dir, "train_log.csv")
    for required in (
        os.path.join(mask_dir, "meta_mask.pt"),
        os.path.join(mask_dir, "config.json"),
        log_csv,
        os.path.join(metrics_dir, "summary.json"),
    ):
        if not os.path.isfile(required):
            _fail(f"Missing artifact: {required}")

    with open(os.path.join(metrics_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("phase32"):
        _fail("config.phase32 should be true")
    if cfg.get("phase") != "phase3_2":
        _fail(f"config.phase={cfg.get('phase')}")
    if len(cfg.get("train_fr_names", [])) < 2:
        _fail("expected >=2 train_fr_names in config")

    with open(log_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _fail("empty train_log.csv")
    need = {"cos_similarity", "disp_k_loss", "disp_batch_loss", "ssim", "loss"}
    missing = need - set(rows[0].keys())
    if missing:
        _fail(f"CSV missing columns: {missing}")

    with open(os.path.join(metrics_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("final_cos_similarity") is None:
        _fail("summary.final_cos_similarity missing")

    _ok(
        f"run_id={run_id} cos_similarity={summary['final_cos_similarity']:.4f} "
        f"(lower=better) ssim={summary['final_ssim']:.4f}"
    )
    _ok(f"mask -> {mask_path}")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3.2 functional test — identity + batch Gram + ensemble")
    print("=" * 60)

    failures = []
    steps = [
        ("identity_cosine", test_identity_cosine),
        ("batch_gram", test_batch_gram),
        ("ensemble_grad", lambda: test_ensemble_grad(args.device)),
    ]
    if not args.skip_train:
        steps.append(("smoke_train", lambda: test_smoke_train(args.device)))

    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"\n!! Step '{name}' failed: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Phase 3.2 RESULT: FAILED ({len(failures)} step(s))")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print("Phase 3.2 RESULT: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
