"""
Training / evaluation entry point (adapted from Protego).

Phase 1 adaptations vs upstream Protego:
  * Datasets default to absolute paths under /mnt/databases/faces/
    (lfw_frontal, celebahq_pp_frontal) — flat folders, no identity pairs.
  * UV Mapping + SMIRK are DISABLED by default.
    Pass ``--original`` to restore Protego's 3D UV texture pipeline.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import argparse
import math
import sys

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from omegaconf import OmegaConf

from protego.utils import get_usable_img_paths
from protego.protego_train import train_protego_mask
from protego.run_exp import run
from protego.FacialRecognition import BASIC_POOL
from protego.datasets import DATASET_ROOTS, build_flat_data_dict
from protego import BASE_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protego / DT-CWT mask training entry point (Phase 1 adapted)."
    )
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name.")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda:0 | mps | ...")
    parser.add_argument(
        "--original",
        action="store_true",
        help=(
            "Use Protego's ORIGINAL UV Mapping + SMIRK 3D texture pipeline. "
            "Default (flag omitted) is the DT-CWT path: UV/SMIRK stay inactive."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lfw_frontal",
        choices=sorted(DATASET_ROOTS.keys()),
        help="Flat face dataset under /mnt/databases/faces/ (no identity pairs).",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Optional cap on number of training images (useful for smoke tests).",
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        default=None,
        help="Optional cap on number of eval images.",
    )
    parser.add_argument(
        "--legacy_facescrub",
        action="store_true",
        help=(
            "[ORIGINAL only] Use Protego's FaceScrub identity-folder layout under "
            "face_db/face_scrub instead of the flat /mnt/databases/faces/ datasets."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    args = parse_args()

    configs = {
        # Running env
        "global_random_seed": 42,
        "device": "cuda:0",
        "exp_name": "default",

        # [DT-CWT ADAPTATION] False => skip UV Mapping / SMIRK (default).
        # True when CLI --original is passed.
        "original": False,

        # Training data
        "train_portion": 0.6,  # Only used by --legacy_facescrub identity folders.
        "uv_gen_align_ldmk": False,  # [ORIGINAL] Don't change unless debugging SMIRK align.
        "uv_gen_batch": 8,  # [ORIGINAL] UV generation batch size.
        "need_cropping": False,  # Our frontal crops are already tight 224x224.
        "fd_name": "mtcnn",
        "crop_loosen": 1.0,
        "shuffle": False,

        # Training configs
        # three_d / bin_mask are forced False in run_exp when original=False.
        "three_d": True,  # [ORIGINAL] UV warp. Overridden off without --original.
        "epoch_num": 100,
        "batch_size": 16,
        "epsilon": 16 / 255.0,
        "min_ssim": 0.95,
        "learning_rate": 0.01 * (16 / 255.0),
        "mask_size": 224,
        "mask_random_seed": 114,
        "bin_mask": True,  # [ORIGINAL] Restrict to SMIRK visibility. Off without --original.
        "train_fr_names": [n for n in BASIC_POOL if n != "ir50_adaface_casia"],

        # Eval configs
        "mask_name": ["default", "univ_mask.npy"],
        "eval_db": "face_scrub",
        "eval_fr_names": ["ir50_adaface_casia"],
        "save_univ_mask": True,
        "visualize_interval": 15,
        "query_portion": 0.5,
        "vis_eval": True,
        "lpips_backbone": "vgg",
        "eval_scene": 0,
        "dataset_name": "lfw_frontal",
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    cfgs = OmegaConf.create(configs)
    cfgs.exp_name = args.exp_name
    if args.device is not None:
        cfgs.device = args.device
    cfgs.original = bool(args.original)
    cfgs.dataset_name = args.dataset
    # Keep three_d / bin_mask aligned with the chosen pipeline up-front.
    if not cfgs.original:
        cfgs.three_d = False
        cfgs.bin_mask = False
        print(
            "[DT-CWT] Running WITHOUT UV Mapping / SMIRK. "
            "Pass --original to restore Protego's 3D PPT pipeline."
        )
    else:
        print("[ORIGINAL] UV Mapping + SMIRK pipeline ENABLED (--original).")

    torch.manual_seed(cfgs.global_random_seed)

    if args.legacy_facescrub:
        # ----- ORIGINAL Protego FaceScrub identity-folder layout -----
        if not cfgs.original:
            print(
                "[WARN] --legacy_facescrub is intended for --original. "
                "Continuing with identity folders but UV still disabled."
            )
        train_portion = cfgs.train_portion
        shuffle_data = cfgs.shuffle
        train_data_path = os.path.join(BASE_PATH, "face_db", "face_scrub")
        protectees = sorted(
            [name for name in os.listdir(train_data_path) if not name.startswith((".", "_"))]
        )
        data = {}
        for protectee in protectees:
            protectee_path = os.path.join(train_data_path, protectee)
            imgs = get_usable_img_paths(protectee_path)
            train_num = math.floor(len(imgs) * train_portion)
            if shuffle_data:
                rand_gen = torch.Generator()
                rand_gen.manual_seed(cfgs.global_random_seed)
                indices = torch.randperm(len(imgs), generator=rand_gen).tolist()
                imgs = [imgs[i] for i in indices]
            data[protectee] = {"train": imgs[:train_num], "eval": imgs[train_num:]}
    else:
        # ----- [DT-CWT ADAPTATION] Flat absolute-path datasets (no identity pairs) -----
        print(
            f"[DT-CWT] Loading flat dataset '{cfgs.dataset_name}' "
            f"from {DATASET_ROOTS[cfgs.dataset_name]}"
        )
        data = build_flat_data_dict(
            dataset_name=cfgs.dataset_name,
            train_split="train",
            eval_split="test",
            max_train=args.max_train,
            max_eval=args.max_eval,
        )
        n_train = len(data[cfgs.dataset_name]["train"])
        n_eval = len(data[cfgs.dataset_name]["eval"])
        print(f"[DT-CWT] train images={n_train}, eval images={n_eval} (no identity pairs).")

    # Train a PPT for every protectee, then run the cross-protectee retrieval evaluation.
    run(cfgs, mode="train", data=data, train=train_protego_mask)

    # To evaluate previously trained PPTs without retraining, use the eval mode instead:
    # run(cfgs, mode='eval', data=data)
