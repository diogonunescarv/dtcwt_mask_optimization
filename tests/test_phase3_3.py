#!/usr/bin/env python3
"""
Phase 3.3 functional test — identity ensemble + SSIM (no K-disp / batch Gram).

Validates:
  1. Merged frac split is reproducible under (data_seed, celeba_train_frac).
  2. Identity-only path sends grads to meta-mask (FR frozen); no K-variants needed.
  3. Smoke train with Adam writes phase3_3 artifacts + optimizer=adam.
  4. Smoke train with Protego (sign-SGD) writes phase3_3 + optimizer=protego.

Run:
    source .venv/bin/activate
    bash scripts/run_phase3_3_test.sh
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


def test_frac_split_reproducible() -> None:
    print("\n== 1) CelebA frac split reproducible under data_seed ==")
    from protego.datasets import build_merged_frac_split

    t1, e1, s1 = build_merged_frac_split(celeba_train_frac=0.5, data_seed=42, max_train=32)
    t2, e2, s2 = build_merged_frac_split(celeba_train_frac=0.5, data_seed=42, max_train=32)
    if t1 != t2 or e1 != e2:
        _fail("same data_seed should yield identical train/eval path lists")
    if s1 != s2:
        _fail("same data_seed should yield identical split_stats")

    t3, e3, _s3 = build_merged_frac_split(celeba_train_frac=0.5, data_seed=99, max_train=32)
    if t3 == t1:
        _fail("different data_seed should change train set (with high probability)")

    # Holdout + LFW test should be non-empty when not capping eval.
    _t, e_full, stats = build_merged_frac_split(celeba_train_frac=0.5, data_seed=42)
    if stats["n_celeba_eval"] < 1 and stats["n_celeba_total"] > 1:
        _fail("expected non-empty CelebA holdout for frac=0.5")
    if stats["n_lfw_train"] < 1:
        _fail("expected LFW train images")
    if len(e_full) != stats["n_eval_total"]:
        _fail("eval length mismatch vs stats")
    _ok(
        f"n_train={stats['n_train_total']} n_eval={stats['n_eval_total']} "
        f"celeba_train={stats['n_celeba_train']} celeba_eval={stats['n_celeba_eval']}"
    )


def test_identity_only_grad(device: str) -> None:
    print("\n== 2) Identity+SSIM path grads → meta-mask only (no K) ==")
    import torch
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
    n_e = float(len(frs))
    total = None
    for i, fr in enumerate(frs):
        recon = mask(x, extra_phase=torch.zeros((), device=device_t))
        prot, _ = apply_linf_budget(recon, x, epsilon=16.0 / 255.0)
        loss_id, _ = identity_cosine_loss(fr(x), fr(prot))
        fr_loss = loss_id / n_e
        if i == 0:
            percep, ssim_v = ssim_hinge_loss(prot, x, min_ssim=0.95, weight=1.0)
            total = fr_loss + percep
        else:
            total = total + fr_loss  # type: ignore[operator]
    assert total is not None
    total.backward()

    g2 = mask.delta_mag["2"].grad
    g3 = mask.delta_mag["3"].grad
    if g2 is None or g3 is None or (g2.abs().sum() + g3.abs().sum()) <= 0:
        _fail("No gradient reached meta-mask")
    for fr in frs:
        for p in fr.fr_model.model.parameters():
            if p.grad is not None:
                _fail("FR received gradients")
    _ok(f"loss={float(total):.4f} ssim={float(ssim_v):.4f} n_frs={len(frs)} mask grads OK")


def _run_smoke_train(device: str, optimizer: str, exp_suffix: str) -> str:
    cmd = [
        sys.executable,
        os.path.join(PROTEGO_BASE, "train_meta_mask.py"),
        "--smoke",
        "--phase33",
        "--quality",
        "--optimizer",
        optimizer,
        "--exp_name",
        f"func_test_p33_{exp_suffix}",
        "--device",
        device,
        "--dataset",
        "merged_lfw_celeba",
        "--fr_models",
        SMOKE_FR_MODELS,
        "--max_train",
        "8",
        "--batch_size",
        "2",
        "--celeba_train_frac",
        "0.5",
        "--data_seed",
        "42",
        "--ckpt_every",
        "1",
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
        _fail(f"train_meta_mask.py --phase33 --optimizer {optimizer} failed code={proc.returncode}")

    ptr = os.path.join(MASK_ROOT, f"LATEST_phase3_3_smoke_func_test_p33_{exp_suffix}.txt")
    if not os.path.isfile(ptr):
        _fail(f"Missing pointer {ptr}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    with open(ptr, encoding="utf-8") as f:
        mask_path = f.read().strip()
    if not os.path.isfile(mask_path):
        _fail(f"Mask missing: {mask_path}")

    mask_dir = os.path.dirname(mask_path)
    run_id = os.path.basename(mask_dir)
    if "phase3_3" not in run_id:
        _fail(f"Expected phase3_3 in run_id, got {run_id}")

    metrics_dir = os.path.join(METRICS_ROOT, run_id)
    log_csv = os.path.join(metrics_dir, "train_log.csv")
    for required in (
        os.path.join(mask_dir, "meta_mask.pt"),
        os.path.join(mask_dir, "checkpoint_latest.pt"),
        os.path.join(mask_dir, "config.json"),
        log_csv,
        os.path.join(metrics_dir, "summary.json"),
    ):
        if not os.path.isfile(required):
            _fail(f"Missing artifact: {required}")

    import torch

    ckpt_roll = torch.load(
        os.path.join(mask_dir, "checkpoint_latest.pt"),
        map_location="cpu",
        weights_only=False,
    )
    if "global_step" not in ckpt_roll or "meta_mask" not in ckpt_roll:
        _fail("checkpoint_latest.pt missing global_step/meta_mask")

    with open(os.path.join(metrics_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("phase33"):
        _fail("config.phase33 should be true")
    if cfg.get("phase") != "phase3_3":
        _fail(f"config.phase={cfg.get('phase')}")
    if cfg.get("optimizer") != optimizer:
        _fail(f"config.optimizer={cfg.get('optimizer')} expected {optimizer}")
    if cfg.get("K") not in (0, None):
        # Phase 3.3 should not rely on K variants.
        if int(cfg.get("K", 0)) != 0:
            _fail(f"Phase 3.3 config.K should be 0, got {cfg.get('K')}")
    if len(cfg.get("train_fr_names", [])) < 2:
        _fail("expected >=2 train_fr_names in config")
    if cfg.get("celeba_train_frac") is None:
        _fail("expected celeba_train_frac in config for phase33 merged")

    with open(log_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _fail("empty train_log.csv")
    need = {"cos_similarity", "ssim", "loss", "optimizer"}
    missing = need - set(rows[0].keys())
    if missing:
        _fail(f"CSV missing columns: {missing}")
    if rows[0].get("optimizer") != optimizer:
        _fail(f"CSV optimizer={rows[0].get('optimizer')}")

    with open(os.path.join(metrics_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("final_cos_similarity") is None:
        _fail("summary.final_cos_similarity missing")
    if summary.get("optimizer") != optimizer:
        _fail(f"summary.optimizer={summary.get('optimizer')}")

    _ok(
        f"optimizer={optimizer} run_id={run_id} "
        f"cos={summary['final_cos_similarity']:.4f} ssim={summary['final_ssim']:.4f}"
    )
    _ok(f"mask -> {mask_path}")
    return run_id


def test_smoke_adam(device: str) -> str:
    print("\n== 3) Smoke train phase3_3 (Adam) ==")
    return _run_smoke_train(device, "adam", "adam")


def test_smoke_protego(device: str) -> str:
    print("\n== 4) Smoke train phase3_3 (Protego sign-SGD) ==")
    return _run_smoke_train(device, "protego", "protego")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3.3 functional test — identity ensemble + SSIM")
    print("=" * 60)

    failures = []
    steps = [
        ("frac_split", test_frac_split_reproducible),
        ("identity_only_grad", lambda: test_identity_only_grad(args.device)),
    ]
    if not args.skip_train:
        steps.append(("smoke_adam", lambda: test_smoke_adam(args.device)))
        steps.append(("smoke_protego", lambda: test_smoke_protego(args.device)))

    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"\n!! Step '{name}' failed: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Phase 3.3 RESULT: FAILED ({len(failures)} step(s))")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print("Phase 3.3 RESULT: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
