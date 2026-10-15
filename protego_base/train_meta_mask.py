#!/usr/bin/env python3
"""
Phase 3 / 3.1 / 3.2 / 3.3 — Offline training of the universal DT-CWT meta-mask.

Phase 3: Frequency Dispersion over K phase variants (``--no_quality``).
Phase 3.1: + SSIM hinge + L∞ (``--quality``, default when not phase32/33).
Phase 3.2: + identity cos + batch Gram + multi-FR ensemble (``--phase32``).
Phase 3.3: identity cos ensemble + SSIM/L∞ only — no K-disp / batch Gram (``--phase33``).

Outputs (versioned, never overwrite)::

    /mnt/study-data/dcarvalho/tests/YYYYMMDD_HHMMSS_phase3[_1|_2|_3]_<exp>/meta_mask.pt
    /mnt/study-data/dcarvalho/metrics/YYYYMMDD_HHMMSS_phase3[_1|_2|_3]_<exp>/...
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Allow `python train_meta_mask.py` from protego_base/ without exporting PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from tqdm import tqdm

from protego.FacialRecognition import BASIC_POOL, FR
from protego.datasets import (
    DATASET_CHOICES,
    DATASET_ROOTS,
    MERGED_DATASET,
    build_flat_data_dict,
    make_dataloader,
)
from protego.dispersion_loss import dispersion_loss_from_variants, frequency_dispersion_loss
from protego.dtcwt_mask import DTCWTMetaMaskModule
from protego.identity_loss import identity_cosine_loss
from protego.paths import MASK_ROOT, METRICS_ROOT, make_run_dirs, write_json
from protego.quality_loss import adapt_percep_weight, apply_linf_budget, ssim_hinge_loss

# Single rolling checkpoint filename (always overwritten — one file only).
CHECKPOINT_NAME = "checkpoint_latest.pt"

# Protego default epsilon = 16/255.
_DEFAULT_EPSILON = 16.0 / 255.0

# Protego train surrogates: BASIC_POOL without the usual held-out intruder.
DEFAULT_TRAIN_FR_NAMES: List[str] = [n for n in BASIC_POOL if n != "ir50_adaface_casia"]

# Compact ensemble for small GPUs: still multi-FR (arch + loss diversity), not single-model.
SMALL_TRAIN_FR_NAMES: List[str] = [
    "ir50_arcface_casia",
    "mobilefacenet_arcface_casia",
    "ir18_adaface_webface",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3/3.1/3.2/3.3: train DT-CWT meta-mask")
    p.add_argument("--exp_name", type=str, default="meta_mask")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--dataset",
        type=str,
        default="lfw_frontal",
        choices=list(DATASET_CHOICES),
    )
    p.add_argument(
        "--fr_model",
        type=str,
        default="inception_facenet_vgg",
        help="Single FR for Phase 3/3.1 (ignored when --fr_models is set / Phase 3.2/3.3 default).",
    )
    p.add_argument(
        "--fr_models",
        type=str,
        default=None,
        help="Comma-separated FR ensemble (Phase 3.2/3.3). Overrides --fr_preset.",
    )
    p.add_argument(
        "--fr_preset",
        type=str,
        default="full",
        choices=("full", "small"),
        help=(
            "Phase 3.2/3.3 ensemble preset when --fr_models is unset. "
            "'full' = Protego BASIC_POOL w/o ir50_adaface_casia (9 FRs); "
            "'small' = 3 diverse FRs for limited VRAM "
            "(ir50_arcface_casia, mobilefacenet_arcface_casia, ir18_adaface_webface)."
        ),
    )
    p.add_argument(
        "--max_frs",
        type=int,
        default=None,
        help="Optional cap on ensemble size after preset/--fr_models (e.g. --max_frs 3).",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--K", type=int, default=4, help="Local phase variants per image (3/3.1/3.2)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Default: 1e-2 (Adam) or 0.01*epsilon (Protego optimizer).",
    )
    p.add_argument("--phase_scale", type=float, default=0.15, help="|φ_k| ~ U(0, scale)")
    p.add_argument("--gram_eps", type=float, default=1e-4)
    p.add_argument("--init_std", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Seed for CelebA train/holdout split (Phase 3.3 merged frac split).",
    )
    p.add_argument(
        "--celeba_train_frac",
        type=float,
        default=0.5,
        help="Fraction of CelebA-HQ used for train in Phase 3.3 merged split (rest = eval).",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--mask_root",
        type=str,
        default=MASK_ROOT,
        help="Root for versioned mask folders",
    )
    p.add_argument(
        "--metrics_root",
        type=str,
        default=METRICS_ROOT,
        help="Root for versioned metrics folders",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run for functional tests (overrides epochs/max_train/K).",
    )
    p.add_argument(
        "--ckpt_every",
        type=int,
        default=1000,
        help=(
            "Save rolling checkpoint_latest.pt every N steps (overwrite one file). "
            "0 disables periodic checkpoints."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to checkpoint_latest.pt to resume. Continues in the same run folder "
            "(appends train_log.csv). Mid-epoch resume restarts that epoch's dataloader."
        ),
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "protego"),
        help="adam = Adam; protego = sign-SGD on delta_mag/phase (Protego-style).",
    )
    # Phase 3.1 quality controls (on by default).
    p.add_argument(
        "--epsilon",
        type=float,
        default=_DEFAULT_EPSILON,
        help="L∞ pixel residual budget after IDT-CWT (default 16/255).",
    )
    p.add_argument(
        "--min_ssim",
        type=float,
        default=0.95,
        help="SSIM hinge target (Protego-style).",
    )
    quality = p.add_mutually_exclusive_group()
    quality.add_argument(
        "--quality",
        dest="quality",
        action="store_true",
        help="Enable Phase 3.1 quality terms (default).",
    )
    quality.add_argument(
        "--no_quality",
        dest="quality",
        action="store_false",
        help="Phase 3 ablation: dispersion loss only.",
    )
    p.set_defaults(quality=True)

    # Phase 3.2
    p.add_argument(
        "--phase32",
        action="store_true",
        help="Enable Phase 3.2: identity cos + batch Gram + multi-FR ensemble.",
    )
    # Phase 3.3
    p.add_argument(
        "--phase33",
        action="store_true",
        help="Enable Phase 3.3: identity cos ensemble + SSIM/L∞ only (no K-disp / batch Gram).",
    )
    p.add_argument("--lambda_k", type=float, default=1.0, help="Weight for K-phase dispersion.")
    p.add_argument("--lambda_batch", type=float, default=1.0, help="Weight for batch Gram.")
    p.add_argument("--lambda_id", type=float, default=1.0, help="Weight for identity cos.")
    p.add_argument(
        "--no_batch_gram",
        action="store_true",
        help="Ablation: disable batch Gram term (Phase 3.2).",
    )
    p.add_argument(
        "--no_identity",
        action="store_true",
        help="Ablation: disable identity cos term (Phase 3.2).",
    )
    return p.parse_args()


def freeze_fr(fr: FR) -> None:
    fr.fr_model.model.eval()
    for param in fr.fr_model.model.parameters():
        param.requires_grad_(False)


def _fr_to(fr: FR, device: torch.device) -> None:
    fr.fr_model.model.to(device)


def resolve_fr_names(args: argparse.Namespace) -> List[str]:
    if args.fr_models:
        names = [n.strip() for n in args.fr_models.split(",") if n.strip()]
        if not names:
            raise ValueError("--fr_models is empty")
    elif args.phase32 or args.phase33:
        if args.fr_preset == "small":
            names = list(SMALL_TRAIN_FR_NAMES)
        else:
            names = list(DEFAULT_TRAIN_FR_NAMES)
    else:
        names = [args.fr_model]

    if args.max_frs is not None:
        if args.max_frs < 1:
            raise ValueError(f"--max_frs must be >= 1, got {args.max_frs}")
        names = names[: args.max_frs]
    return names


def _protego_sign_step(mask_mod: DTCWTMetaMaskModule, lr: float) -> None:
    """Protego-style update: param -= lr * sign(grad). No wavelet L∞ projection."""
    with torch.no_grad():
        for p in mask_mod.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            p.add_(-lr * p.grad.sign())
            p.grad = None


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    """Write via temp file then rename so a crash mid-write does not corrupt the only ckpt."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_train_checkpoint(
    *,
    mask_mod: DTCWTMetaMaskModule,
    opt: Optional[torch.optim.Optimizer],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    percep_loss_weight: float,
    last_row: Optional[Dict[str, Any]],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "meta_mask": mask_mod.meta_mask_state(),
        "module_state_dict": {
            k: v.detach().cpu()
            for k, v in mask_mod.state_dict().items()
            if k.startswith("delta_")
        },
        "config": cfg,
        "editable_levels": list(mask_mod.editable_levels),
        "img_size": args.img_size,
        "J": 3,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "percep_loss_weight": float(percep_loss_weight),
        "optimizer": args.optimizer,
        "lr": float(args.lr),
        "elapsed_seconds": float(elapsed_seconds),
        "last_row": last_row,
        "is_periodic_checkpoint": True,
    }
    if opt is not None:
        payload["optimizer_state_dict"] = opt.state_dict()
    return payload


def save_rolling_checkpoint(mask_dir: str, payload: Dict[str, Any], label: str) -> str:
    path = os.path.join(mask_dir, CHECKPOINT_NAME)
    _atomic_torch_save(payload, path)
    print(
        f"[{label}] checkpoint step={payload.get('global_step')} "
        f"epoch={payload.get('epoch')} -> {path}"
    )
    return path


def main() -> int:
    args = parse_args()
    if args.phase32 and args.phase33:
        raise SystemExit("Cannot combine --phase32 and --phase33; pick one.")
    if args.phase32 or args.phase33:
        # Both imply quality controls.
        args.quality = True

    # Resolve learning rate defaults per optimizer.
    if args.lr is None:
        if args.optimizer == "protego":
            args.lr = 0.01 * float(args.epsilon)
        else:
            args.lr = 1e-2

    if args.smoke:
        args.epochs = 1
        args.max_train = args.max_train or 8
        args.batch_size = min(args.batch_size, 2)
        args.K = min(args.K, 3)
        args.exp_name = f"smoke_{args.exp_name}"
        if args.ckpt_every <= 0:
            args.ckpt_every = 1

    if args.ckpt_every < 0:
        raise SystemExit("--ckpt_every must be >= 0")

    resume_ckpt: Optional[Dict[str, Any]] = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise SystemExit(f"--resume file not found: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        print(f"[resume] loaded {args.resume} "
              f"(step={resume_ckpt.get('global_step')} epoch={resume_ckpt.get('epoch')})")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    if args.phase33:
        phase_tag = "phase3_3"
        label = "Phase3.3"
    elif args.phase32:
        phase_tag = "phase3_2"
        label = "Phase3.2"
    elif args.quality:
        phase_tag = "phase3_1"
        label = "Phase3.1"
    else:
        phase_tag = "phase3"
        label = "Phase3"

    if resume_ckpt is not None:
        # Continue in the same run folder as the checkpoint.
        mask_dir = os.path.dirname(os.path.abspath(args.resume))
        run_id = os.path.basename(mask_dir)
        metrics_dir = os.path.join(args.metrics_root, run_id)
        os.makedirs(metrics_dir, exist_ok=True)
        # Prefer CLI/flags, but keep run identity from the interrupted job.
        resumed_cfg = resume_ckpt.get("config") or {}
        if resumed_cfg.get("exp_name"):
            args.exp_name = resumed_cfg["exp_name"]
        if resumed_cfg.get("phase") and resumed_cfg["phase"] != phase_tag:
            print(
                f"[WARN] resume phase={resumed_cfg.get('phase')} vs current {phase_tag}; "
                "using current flags for the remaining loop."
            )
    else:
        run_id, mask_dir, metrics_dir = make_run_dirs(
            phase=phase_tag,
            exp_name=args.exp_name,
            mask_root=args.mask_root,
            metrics_root=args.metrics_root,
        )
    fr_names = resolve_fr_names(args)
    print(f"[{label}] run_id={run_id}")
    print(f"[{label}] masks   -> {mask_dir}")
    print(f"[{label}] metrics -> {metrics_dir}")
    print(
        f"[{label}] quality={args.quality} phase32={args.phase32} phase33={args.phase33} "
        f"optimizer={args.optimizer} epsilon={args.epsilon} min_ssim={args.min_ssim}"
    )
    print(f"[{label}] FRs ({len(fr_names)}): {fr_names}")

    dataset_root = (
        f"merged:{DATASET_ROOTS['lfw_frontal']}+{DATASET_ROOTS['celebahq_pp_frontal']}"
        if args.dataset == MERGED_DATASET
        else DATASET_ROOTS[args.dataset]
    )

    # Phase 3.3: LFW-all + seeded CelebA fraction. Other phases keep legacy merge.
    use_frac_split = bool(args.phase33 and args.dataset == MERGED_DATASET)
    celeba_frac: Optional[float] = args.celeba_train_frac if use_frac_split else None

    cfg = {
        "exp_name": args.exp_name,
        "run_id": run_id,
        "phase": phase_tag,
        "dataset": args.dataset,
        "dataset_root": dataset_root,
        "fr_model": fr_names[0],
        "train_fr_names": fr_names,
        "n_frs": len(fr_names),
        "fr_preset": args.fr_preset if (args.phase32 or args.phase33) else None,
        "max_frs": args.max_frs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "K": args.K if not args.phase33 else 0,
        "epochs": args.epochs,
        "max_train": args.max_train,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "phase_scale": args.phase_scale,
        "gram_eps": args.gram_eps,
        "init_std": args.init_std,
        "seed": args.seed,
        "data_seed": args.data_seed if use_frac_split else None,
        "celeba_train_frac": celeba_frac,
        "device": str(device),
        "mask_dir": mask_dir,
        "metrics_dir": metrics_dir,
        "smoke": bool(args.smoke),
        "quality": bool(args.quality),
        "phase32": bool(args.phase32),
        "phase33": bool(args.phase33),
        "epsilon": args.epsilon,
        "min_ssim": args.min_ssim,
        "lambda_k": 0.0 if args.phase33 else args.lambda_k,
        "lambda_batch": (
            0.0
            if args.phase33 or args.no_batch_gram
            else args.lambda_batch
        ),
        "lambda_id": (
            args.lambda_id
            if args.phase33
            else (0.0 if args.no_identity else args.lambda_id)
        ),
        "ckpt_every": int(args.ckpt_every),
        "resumed_from": args.resume,
    }

    data = build_flat_data_dict(
        args.dataset,
        max_train=args.max_train,
        max_eval=0 if args.smoke else None,
        seed=args.seed,
        celeba_train_frac=celeba_frac,
        data_seed=args.data_seed if use_frac_split else None,
    )
    train_paths: List[str] = data[args.dataset]["train"]
    split_stats = data[args.dataset].get("split_stats")
    if split_stats:
        cfg.update({f"split_{k}": v for k, v in split_stats.items()})
        # Flat aliases matching the plan naming.
        cfg["n_lfw_train"] = split_stats.get("n_lfw_train")
        cfg["n_celeba_train"] = split_stats.get("n_celeba_train")
        cfg["n_celeba_eval"] = split_stats.get("n_celeba_eval")
        cfg["n_train_total"] = split_stats.get("n_train_total")
        cfg["n_eval_total"] = split_stats.get("n_eval_total")

    write_json(os.path.join(metrics_dir, "config.json"), cfg)
    write_json(os.path.join(mask_dir, "config.json"), cfg)

    if not train_paths:
        raise RuntimeError("Empty training set.")
    loader = make_dataloader(
        train_paths,
        batch_size=args.batch_size,
        img_size=args.img_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"[{label}] train images={len(train_paths)} batches/epoch≈{len(loader)}")
    print(f"[{label}] ckpt_every={args.ckpt_every} -> {os.path.join(mask_dir, CHECKPOINT_NAME)}")
    if split_stats:
        print(
            f"[{label}] split data_seed={args.data_seed} celeba_frac={celeba_frac} "
            f"lfw_train={split_stats['n_lfw_train']} "
            f"celeba_train={split_stats['n_celeba_train']} "
            f"celeba_eval={split_stats['n_celeba_eval']}"
        )

    frs: List[FR] = []
    for name in fr_names:
        if name not in BASIC_POOL and not name.startswith("inception_"):
            print(f"[WARN] FR '{name}' not in BASIC_POOL; attempting load anyway.")
        fr = FR(model_name=name, device=device)
        freeze_fr(fr)
        frs.append(fr)
    # Multi-FR: keep weights on CPU between steps to fit ~11GB GPUs.
    multi_fr = (args.phase32 or args.phase33) and len(frs) > 1
    if multi_fr:
        for fr in frs:
            _fr_to(fr, torch.device("cpu"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{label}] FR offload enabled ({len(frs)} models on CPU between steps)")

    mask_mod = DTCWTMetaMaskModule(
        channels=3,
        img_size=args.img_size,
        J=3,
        editable_levels=(2, 3),
        init_std=args.init_std,
    ).to(device)
    for p in mask_mod.xfm.parameters():
        p.requires_grad_(False)
    for p in mask_mod.ifm.parameters():
        p.requires_grad_(False)

    if resume_ckpt is not None:
        mask_mod.load_meta_mask_state(resume_ckpt["meta_mask"])

    trainable = [p for p in mask_mod.parameters() if p.requires_grad]
    opt = None
    if args.optimizer == "adam":
        opt = torch.optim.Adam(trainable, lr=args.lr)
        if resume_ckpt is not None and resume_ckpt.get("optimizer_state_dict"):
            try:
                opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except (ValueError, RuntimeError) as exc:
                print(f"[WARN] could not load Adam state ({exc}); continuing with fresh Adam.")
    print(f"[{label}] optimizer={args.optimizer} lr={args.lr}")

    log_fields = [
        "epoch",
        "step",
        "loss",
        "disp_loss",
        "disp_k_loss",
        "disp_batch_loss",
        "cos_similarity",
        "gram_offdiag_abs_mean",
        "feat_norm_mean",
        "ssim",
        "percep_term",
        "percep_weight",
        "linf_max",
        "n_frs",
        "lr",
        "optimizer",
        "seconds",
    ]
    log_csv = os.path.join(metrics_dir, "train_log.csv")
    log_exists = os.path.isfile(log_csv) and os.path.getsize(log_csv) > 0
    if not log_exists:
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    global_step = int(resume_ckpt["global_step"]) if resume_ckpt else 0
    start_epoch = int(resume_ckpt["epoch"]) if resume_ckpt else 0
    history: List[Dict[str, Any]] = []
    if resume_ckpt and resume_ckpt.get("last_row"):
        history.append(resume_ckpt["last_row"])
    percep_loss_weight = (
        float(resume_ckpt["percep_loss_weight"])
        if resume_ckpt and "percep_loss_weight" in resume_ckpt
        else 1.0
    )
    lam_k = args.lambda_k
    lam_b = 0.0 if args.no_batch_gram else args.lambda_batch
    lam_id = 0.0 if args.no_identity else args.lambda_id
    elapsed_offset = float(resume_ckpt.get("elapsed_seconds", 0.0)) if resume_ckpt else 0.0
    t0 = time.time()
    if resume_ckpt:
        print(
            f"[{label}] resuming at epoch={start_epoch + 1}/{args.epochs} "
            f"global_step={global_step} (dataloader restarts that epoch)"
        )

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for imgs, _paths in pbar:
            imgs = imgs.to(device)

            if opt is not None:
                opt.zero_grad(set_to_none=True)
            else:
                mask_mod.zero_grad(set_to_none=True)

            if args.phase33:
                # Identity ensemble + SSIM only (no K variants / batch Gram).
                n_e = float(len(frs))
                offload = len(frs) > 1
                if offload:
                    for fr in frs:
                        _fr_to(fr, torch.device("cpu"))

                sum_id = 0.0
                linf_max = 0.0
                ssim_f = 0.0
                percep_f = 0.0
                total_f = 0.0

                for i, fr in enumerate(frs):
                    if offload:
                        _fr_to(fr, device)

                    recon = mask_mod(imgs, extra_phase=torch.zeros((), device=device))
                    prot, residual = apply_linf_budget(recon, imgs, args.epsilon)
                    linf_max = max(
                        linf_max, float(residual.detach().abs().amax().cpu())
                    )

                    feats_prot = fr(prot)
                    feats_orig = fr(imgs)
                    loss_id, cos_det = identity_cosine_loss(feats_orig, feats_prot)
                    fr_loss = (lam_id * loss_id) / n_e

                    if i == 0:
                        percep_term, ssim_value = ssim_hinge_loss(
                            prot, imgs, args.min_ssim, percep_loss_weight
                        )
                        (fr_loss + percep_term).backward()
                        ssim_f = float(ssim_value.detach().cpu())
                        percep_f = float(percep_term.detach().cpu())
                        total_f += percep_f
                    else:
                        fr_loss.backward()

                    total_f += float(fr_loss.detach().cpu())
                    sum_id += float(cos_det.cpu())

                    if offload:
                        _fr_to(fr, torch.device("cpu"))
                        torch.cuda.empty_cache()

                cos_sim_f = sum_id / n_e
                disp_k_f = 0.0
                disp_b_f = 0.0
                off_f = 0.0
                fnorm_f = 0.0
                disp_legacy = 0.0
                loss = torch.tensor(total_f)

            elif args.phase32:
                phases = (torch.rand(args.K, device=device) * 2 - 1) * args.phase_scale
                # Memory-safe ensemble: one FR at a time (recompute mask forward,
                # optional CPU offload). Avoids holding 9 FR graphs + weights on GPU.
                n_e = float(len(frs))
                offload = len(frs) > 1
                if offload:
                    for fr in frs:
                        _fr_to(fr, torch.device("cpu"))

                sum_k = 0.0
                sum_b = 0.0
                sum_id = 0.0
                sum_off = 0.0
                sum_fnorm = 0.0
                linf_max = 0.0
                ssim_f = 0.0
                percep_f = 0.0
                total_f = 0.0

                for i, fr in enumerate(frs):
                    if offload:
                        _fr_to(fr, device)

                    variants = mask_mod.forward_k_variants(imgs, phases, clamp01=False)
                    variants, residual_k = apply_linf_budget(variants, imgs, args.epsilon)
                    linf_max = max(
                        linf_max, float(residual_k.detach().abs().amax().cpu())
                    )

                    recon0 = mask_mod(imgs, extra_phase=torch.zeros((), device=device))
                    prot0, residual0 = apply_linf_budget(recon0, imgs, args.epsilon)
                    linf_max = max(
                        linf_max, float(residual0.detach().abs().amax().cpu())
                    )

                    def _fr_call(batch: torch.Tensor, _fr=fr) -> torch.Tensor:
                        return _fr(batch)

                    loss_k, stats_k = dispersion_loss_from_variants(
                        _fr_call, variants, eps=args.gram_eps
                    )
                    feats_prot = fr(prot0)
                    feats_orig = fr(imgs)
                    loss_b, _gram_b = frequency_dispersion_loss(
                        feats_prot, eps=args.gram_eps
                    )
                    loss_id, cos_det = identity_cosine_loss(feats_orig, feats_prot)
                    fr_loss = (lam_k * loss_k + lam_b * loss_b + lam_id * loss_id) / n_e

                    if i == 0:
                        percep_term, ssim_value = ssim_hinge_loss(
                            prot0, imgs, args.min_ssim, percep_loss_weight
                        )
                        (fr_loss + percep_term).backward()
                        ssim_f = float(ssim_value.detach().cpu())
                        percep_f = float(percep_term.detach().cpu())
                        total_f += percep_f
                    else:
                        fr_loss.backward()

                    total_f += float(fr_loss.detach().cpu())
                    sum_k += float(loss_k.detach().cpu())
                    sum_b += float(loss_b.detach().cpu())
                    sum_id += float(cos_det.cpu())
                    sum_off += stats_k["gram_offdiag_abs_mean"]
                    sum_fnorm += stats_k["feat_norm_mean"]

                    if offload:
                        _fr_to(fr, torch.device("cpu"))
                        torch.cuda.empty_cache()

                disp_k_f = sum_k / n_e
                disp_b_f = sum_b / n_e
                cos_sim_f = sum_id / n_e  # lower = better de-ID
                off_f = sum_off / n_e
                fnorm_f = sum_fnorm / n_e
                disp_legacy = disp_k_f
                loss = torch.tensor(total_f)  # for logging only; grads already applied
            else:
                phases = (torch.rand(args.K, device=device) * 2 - 1) * args.phase_scale
                variants = mask_mod.forward_k_variants(imgs, phases, clamp01=False)

                if args.quality:
                    variants, residual_k = apply_linf_budget(variants, imgs, args.epsilon)
                    linf_max = float(residual_k.detach().abs().amax().cpu())
                else:
                    variants = variants.clamp(0.0, 1.0)
                    linf_max = float(
                        (variants.detach() - imgs.unsqueeze(1)).abs().amax().cpu()
                    )

                fr = frs[0]

                def _fr(batch: torch.Tensor) -> torch.Tensor:
                    return fr(batch)

                disp_loss, stats = dispersion_loss_from_variants(
                    _fr, variants, eps=args.gram_eps
                )
                if args.quality:
                    percep_term, ssim_value = ssim_hinge_loss(
                        variants, imgs, args.min_ssim, percep_loss_weight
                    )
                    loss = disp_loss + percep_term
                    ssim_f = float(ssim_value.detach().cpu())
                    percep_f = float(percep_term.detach().cpu())
                else:
                    loss = disp_loss
                    ssim_f = 0.0
                    percep_f = 0.0
                disp_k_f = stats["loss"]
                disp_b_f = 0.0
                cos_sim_f = 0.0
                off_f = stats["gram_offdiag_abs_mean"]
                fnorm_f = stats["feat_norm_mean"]
                disp_legacy = stats["loss"]
                loss.backward()
                total_f = float(loss.detach().cpu())

            for fr in frs:
                # Models may be on CPU (offload); check grads only if present.
                for param in fr.fr_model.model.parameters():
                    if param.grad is not None:
                        raise RuntimeError("FR model received gradients; expected freeze.")

            if args.optimizer == "protego":
                _protego_sign_step(mask_mod, args.lr)
            else:
                assert opt is not None
                opt.step()

            if args.quality:
                percep_loss_weight = adapt_percep_weight(
                    ssim_f, args.min_ssim, percep_loss_weight
                )

            global_step += 1
            if not (args.phase32 or args.phase33):
                total_f = float(loss.detach().cpu())
            elapsed_now = elapsed_offset + (time.time() - t0)
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": total_f,
                "disp_loss": disp_legacy,
                "disp_k_loss": disp_k_f,
                "disp_batch_loss": disp_b_f,
                "cos_similarity": cos_sim_f,
                "gram_offdiag_abs_mean": off_f,
                "feat_norm_mean": fnorm_f,
                "ssim": ssim_f,
                "percep_term": percep_f,
                "percep_weight": percep_loss_weight,
                "linf_max": linf_max,
                "n_frs": len(frs),
                "lr": args.lr,
                "optimizer": args.optimizer,
                "seconds": elapsed_now,
            }
            history.append(row)
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)
            postfix = {
                "loss": f"{total_f:.4f}",
            }
            if args.phase33 or args.phase32:
                postfix["cos"] = f"{cos_sim_f:.4f}"
            else:
                postfix["disp"] = f"{disp_legacy:.4f}"
            if args.quality:
                postfix["ssim"] = f"{ssim_f:.4f}"
            pbar.set_postfix(**postfix)

            if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                save_rolling_checkpoint(
                    mask_dir,
                    build_train_checkpoint(
                        mask_mod=mask_mod,
                        opt=opt,
                        cfg=cfg,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        percep_loss_weight=percep_loss_weight,
                        last_row=row,
                        elapsed_seconds=elapsed_now,
                    ),
                    label,
                )

    elapsed = elapsed_offset + (time.time() - t0)
    # Final rolling checkpoint (same single file) + canonical meta_mask.pt
    final_payload = build_train_checkpoint(
        mask_mod=mask_mod,
        opt=opt,
        cfg=cfg,
        args=args,
        epoch=max(args.epochs - 1, 0),
        global_step=global_step,
        percep_loss_weight=percep_loss_weight,
        last_row=history[-1] if history else None,
        elapsed_seconds=elapsed,
    )
    final_payload["is_periodic_checkpoint"] = False
    final_payload["final_loss"] = history[-1]["loss"] if history else None
    if args.ckpt_every > 0:
        save_rolling_checkpoint(mask_dir, final_payload, label)

    ckpt = {
        "meta_mask": final_payload["meta_mask"],
        "module_state_dict": final_payload["module_state_dict"],
        "config": cfg,
        "editable_levels": list(mask_mod.editable_levels),
        "img_size": args.img_size,
        "J": 3,
        "elapsed_seconds": elapsed,
        "final_loss": history[-1]["loss"] if history else None,
    }
    mask_path = os.path.join(mask_dir, "meta_mask.pt")
    _atomic_torch_save(ckpt, mask_path)

    summary = {
        "run_id": run_id,
        "phase": phase_tag,
        "quality": bool(args.quality),
        "phase32": bool(args.phase32),
        "phase33": bool(args.phase33),
        "optimizer": args.optimizer,
        "train_fr_names": fr_names,
        "mask_path": mask_path,
        "metrics_dir": metrics_dir,
        "steps": global_step,
        "elapsed_seconds": elapsed,
        "final_loss": history[-1]["loss"] if history else None,
        "final_disp_loss": history[-1]["disp_loss"] if history else None,
        "final_disp_k_loss": history[-1]["disp_k_loss"] if history else None,
        "final_disp_batch_loss": history[-1]["disp_batch_loss"] if history else None,
        "final_cos_similarity": history[-1]["cos_similarity"] if history else None,
        "final_ssim": history[-1]["ssim"] if history else None,
        "final_gram_offdiag_abs_mean": history[-1]["gram_offdiag_abs_mean"] if history else None,
        "final_percep_weight": history[-1]["percep_weight"] if history else None,
    }
    write_json(os.path.join(metrics_dir, "summary.json"), summary)
    write_json(os.path.join(mask_dir, "summary.json"), summary)
    latest_ptr = os.path.join(args.mask_root, f"LATEST_{phase_tag}_{args.exp_name}.txt")
    with open(latest_ptr, "w", encoding="utf-8") as f:
        f.write(mask_path + "\n")

    print(f"[{label}] Saved mask -> {mask_path}")
    print(f"[{label}] Saved logs -> {log_csv}")
    print(f"[{label}] Done in {elapsed:.1f}s | final_loss={summary['final_loss']}")
    if args.phase32 or args.phase33:
        print(
            f"[{label}] final cos_similarity={summary['final_cos_similarity']:.4f} "
            f"(lower=better de-ID) | final_ssim={summary['final_ssim']} "
            f"| optimizer={args.optimizer}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
