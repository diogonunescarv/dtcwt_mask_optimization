#!/usr/bin/env python3
"""
Phase 1 functional test — Protego base cleanup & adaptation.

Validates:
  1. Flat dataset loaders resolve absolute paths under /mnt/databases/faces/.
  2. Batches load without requiring identity pairs.
  3. FR models (FacialRecognition.FR) can be constructed and extract embeddings.
  4. Default path does NOT import / initialize UV Mapping or SMIRK.
  5. --original flag is wired into configs (UV path remains gated).

Run from repo root:
    source .venv/bin/activate
    python tests/test_phase1.py
    # or:
    bash scripts/run_phase1_test.sh
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

# Ensure protego_base is on sys.path (FR_DB / protego packages live there).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTEGO_BASE = os.path.join(REPO_ROOT, "protego_base")
if PROTEGO_BASE not in sys.path:
    sys.path.insert(0, PROTEGO_BASE)


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def test_dataset_paths_and_loading(max_images: int = 8, batch_size: int = 4) -> None:
    print("\n== 1) Dataset absolute paths & flat loading ==")
    from protego.datasets import (
        DATASET_ROOTS,
        FaceFolderDataset,
        build_flat_data_dict,
        list_image_paths,
        make_dataloader,
        resolve_split_dir,
    )

    assert os.path.isdir(DATASET_ROOTS["lfw_frontal"]), DATASET_ROOTS["lfw_frontal"]
    assert os.path.isdir(DATASET_ROOTS["celebahq_pp_frontal"]), DATASET_ROOTS["celebahq_pp_frontal"]
    _ok(f"Roots exist: {DATASET_ROOTS}")

    lfw_train = resolve_split_dir("lfw_frontal", "train")
    lfw_test = resolve_split_dir("lfw_frontal", "test")
    celeb = resolve_split_dir("celebahq_pp_frontal", "train")
    for d in (lfw_train, lfw_test, celeb):
        assert os.path.isdir(d), d
    _ok(f"Splits resolve: train={lfw_train}, test={lfw_test}, celeb={celeb}")

    # No identity subfolders: images live directly in the split dirs.
    sample = list_image_paths(lfw_train)[:max_images]
    assert sample, "lfw faces_train empty"
    assert all(os.path.isfile(p) for p in sample)
    _ok(f"Listed {len(sample)} LFW train images (flat, no identity pairs)")

    data = build_flat_data_dict(
        "lfw_frontal", max_train=max_images, max_eval=max_images
    )
    assert "lfw_frontal" in data
    assert "train" in data["lfw_frontal"] and "eval" in data["lfw_frontal"]
    assert len(data["lfw_frontal"]["train"]) == max_images
    # Single synthetic protectee key — not per-identity folders.
    assert list(data.keys()) == ["lfw_frontal"]
    _ok("build_flat_data_dict returns Protego-compatible dict without identity pairs")

    ds = FaceFolderDataset(sample, img_size=224)
    img, path = ds[0]
    assert img.shape == (3, 224, 224), img.shape
    assert img.dtype == __import__("torch").float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    _ok(f"FaceFolderDataset item shape={tuple(img.shape)} path={os.path.basename(path)}")

    dl = make_dataloader(sample, batch_size=batch_size, img_size=224, shuffle=False)
    batch_imgs, batch_paths = next(iter(dl))
    assert batch_imgs.shape[0] == batch_size
    assert batch_imgs.shape[1:] == (3, 224, 224)
    _ok(f"DataLoader batch shape={tuple(batch_imgs.shape)}")

    celeb_data = build_flat_data_dict(
        "celebahq_pp_frontal", max_train=max_images, max_eval=max_images
    )
    assert len(celeb_data["celebahq_pp_frontal"]["train"]) == max_images
    _ok("celebahq_pp_frontal flat loader works")


def test_uv_smirk_gated(original: bool = False) -> None:
    print("\n== 2) UV Mapping / SMIRK gating (--original) ==")
    # Fresh process modules: ensure UVMapping is not already imported.
    for mod_name in list(sys.modules):
        if mod_name == "protego.UVMapping" or mod_name.startswith("protego.UVMapping"):
            del sys.modules[mod_name]
        if mod_name.startswith("smirk"):
            del sys.modules[mod_name]

    import protego.run_exp as run_exp

    importlib.reload(run_exp)

    # Importing run_exp must NOT pull UVMapping / smirk at module level.
    assert "protego.UVMapping" not in sys.modules, (
        "UVMapping was imported at module level — should be lazy / --original only"
    )
    assert not any(m == "smirk" or m.startswith("smirk.") for m in sys.modules), (
        "smirk package leaked into sys.modules without --original"
    )
    _ok("run_exp import does not load UVMapping or smirk")

    from omegaconf import OmegaConf

    cfgs_default = OmegaConf.create({"original": False, "three_d": True, "bin_mask": True})
    mapper = run_exp._init_uvmapper_if_original(cfgs_default, __import__("torch").device("cpu"))
    assert mapper is None
    assert "protego.UVMapping" not in sys.modules
    _ok("_init_uvmapper_if_original(original=False) returns None and skips import")

    # Flag wiring (do not actually init SMIRK weights — assets may be absent).
    cfgs_orig = OmegaConf.create({"original": True})
    assert run_exp._use_original_pipeline(cfgs_orig) is True
    assert run_exp._use_original_pipeline(cfgs_default) is False
    _ok("--original / cfgs.original gate helpers behave correctly")

    if original:
        # Optional deep check: attempt real UVGenerator init (needs SMIRK assets).
        print("  [INFO] --check-original-init requested; attempting UVGenerator load...")
        try:
            _ = run_exp._init_uvmapper_if_original(cfgs_orig, __import__("torch").device("cpu"))
            _ok("UVGenerator initialized under --original")
        except Exception as exc:  # noqa: BLE001
            _fail(f"UVGenerator init failed with assets present? {exc}")
    else:
        _ok("Skipping live SMIRK init (pass --check-original-init to force)")


def test_fr_feature_extraction(device: str = "cpu", model_name: str = "inception_facenet_vgg") -> None:
    print(f"\n== 3) FR model load + feature extraction ({model_name} @ {device}) ==")
    import torch
    from protego.FacialRecognition import BASIC_POOL, FR, SPECIAL_POOL
    from protego.datasets import list_image_paths, resolve_split_dir

    assert model_name in BASIC_POOL + SPECIAL_POOL or model_name.startswith("inception_")
    _ok(f"FR pool lists available (BASIC={len(BASIC_POOL)}, SPECIAL={len(SPECIAL_POOL)})")

    fr = FR(model_name=model_name, device=torch.device(device))
    _ok(f"Loaded FR '{fr.model_name}' embedding_dim={fr.embedding_dim} img_size={fr.img_size}")

    paths = list_image_paths(resolve_split_dir("lfw_frontal", "train"))[:4]
    from protego.utils import load_imgs

    imgs = load_imgs(img_paths=paths, img_sz=224, drange=1, device=torch.device(device))
    assert imgs.ndim == 4 and imgs.shape[0] == 4
    with torch.no_grad():
        feats = fr(imgs)
    assert feats.ndim == 2 and feats.shape[0] == 4
    assert feats.shape[1] == fr.embedding_dim
    _ok(f"Feature tensor shape={tuple(feats.shape)}")


def test_train_cli_parses_original() -> None:
    print("\n== 4) train.py CLI exposes --original and dataset paths ==")
    import ast

    train_path = os.path.join(PROTEGO_BASE, "train.py")
    with open(train_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "--original" in src
    assert "build_flat_data_dict" in src
    assert "/mnt/databases/faces" in src or "DATASET_ROOTS" in src
    # Sanity: file still parses.
    ast.parse(src)
    _ok("train.py documents --original and flat dataset loading")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 functional tests")
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fr-model",
        default="inception_facenet_vgg",
        help="FR model name (FaceNet auto-downloads weights).",
    )
    parser.add_argument(
        "--skip-fr",
        action="store_true",
        help="Skip FR weight download / inference (dataset + gating only).",
    )
    parser.add_argument(
        "--check-original-init",
        action="store_true",
        help="Also try initializing UVGenerator (requires SMIRK assets).",
    )
    parser.add_argument("--max-images", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1 functional test — Protego adaptation")
    print("=" * 60)
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"PROTEGO_BASE={PROTEGO_BASE}")

    failures = []
    steps = [
        ("datasets", lambda: test_dataset_paths_and_loading(max_images=args.max_images)),
        ("uv_gate", lambda: test_uv_smirk_gated(original=args.check_original_init)),
        ("train_cli", test_train_cli_parses_original),
    ]
    if not args.skip_fr:
        steps.append(
            ("fr", lambda: test_fr_feature_extraction(device=args.device, model_name=args.fr_model))
        )

    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"\n!! Step '{name}' failed: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Phase 1 RESULT: FAILED ({len(failures)} step(s))")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    print("Phase 1 RESULT: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
