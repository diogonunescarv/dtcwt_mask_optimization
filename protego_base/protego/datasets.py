"""
Dataset loaders adapted for this project.

Protego originally expects FaceScrub-style identity folders:
    face_db/<dataset>/<protectee>/{img1.jpg, img2.jpg, ...}

Our datasets are flat face folders WITHOUT identity pairs / multi-image
identity grouping:
    /mnt/databases/faces/lfw_frontal/{faces_train,faces_test}/*.jpg
    /mnt/databases/faces/celebahq_pp_frontal/*.jpg

Phase 1+ pipelines therefore treat each image independently.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset

# Absolute dataset roots (project convention / PLAN.md).
DATASET_ROOTS: Dict[str, str] = {
    "lfw_frontal": "/mnt/databases/faces/lfw_frontal",
    "celebahq_pp_frontal": "/mnt/databases/faces/celebahq_pp_frontal",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_image_paths(directory: str, recursive: bool = False) -> List[str]:
    """Return sorted absolute paths of usable images under ``directory``."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.startswith((".", "_")):
                    continue
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(directory):
            if name.startswith((".", "_")):
                continue
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(directory, name))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")
    return paths


def resolve_split_dir(dataset_name: str, split: str = "train") -> str:
    """
    Resolve the absolute directory for a named dataset split.

    Args:
        dataset_name: ``lfw_frontal`` or ``celebahq_pp_frontal``.
        split: ``train`` / ``test`` / ``all``.
            - lfw_frontal: uses ``faces_train`` / ``faces_test``.
            - celebahq_pp_frontal: flat folder; ``train``/``test``/``all``
              all map to the same root (no official split on disk).
    """
    if dataset_name not in DATASET_ROOTS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {sorted(DATASET_ROOTS)}"
        )
    root = DATASET_ROOTS[dataset_name]

    if dataset_name == "lfw_frontal":
        if split == "train":
            return os.path.join(root, "faces_train")
        if split == "test":
            return os.path.join(root, "faces_test")
        if split == "all":
            # Caller should concatenate both splits if needed.
            raise ValueError("Use split='train' or 'test' for lfw_frontal.")
        raise ValueError(f"Unsupported split for lfw_frontal: {split}")

    # celebahq_pp_frontal: flat directory of images, no identity subfolders.
    return root


def build_flat_data_dict(
    dataset_name: str,
    train_split: str = "train",
    eval_split: Optional[str] = "test",
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a Protego-compatible ``data`` dict WITHOUT identity grouping.

    Protego's ``run()`` iterates over protectees. Because we do not have
    multiple images per identity, we expose a single synthetic protectee
    named after the dataset, containing independent face images.

    Returns:
        ``{dataset_name: {"train": [...], "eval": [...]}}``
    """
    train_dir = resolve_split_dir(dataset_name, split=train_split)
    train_paths = list_image_paths(train_dir)
    if max_train is not None:
        train_paths = train_paths[:max_train]

    if eval_split is None:
        eval_paths: List[str] = []
    else:
        if dataset_name == "celebahq_pp_frontal":
            # No official test split: reuse the same pool, disjoint by index
            # when possible, otherwise take a trailing slice for smoke tests.
            all_paths = list_image_paths(resolve_split_dir(dataset_name, "all"))
            if max_train is not None and max_train < len(all_paths):
                eval_paths = all_paths[max_train:]
            else:
                # Keep a small disjoint-ish eval tail for functional checks.
                eval_paths = all_paths[-max(1, len(all_paths) // 10) :]
            if max_eval is not None:
                eval_paths = eval_paths[:max_eval]
        else:
            eval_dir = resolve_split_dir(dataset_name, split=eval_split)
            eval_paths = list_image_paths(eval_dir)
            if max_eval is not None:
                eval_paths = eval_paths[:max_eval]

    return {dataset_name: {"train": train_paths, "eval": eval_paths}}


class FaceFolderDataset(Dataset):
    """
    Flat face-image dataset (no labels / no identity pairs).

    Each item is a float32 RGB tensor in ``[0, 1]`` with shape ``(3, H, W)``.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        img_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.img_size = img_size
        self.transform = transform
        if not self.image_paths:
            raise ValueError("FaceFolderDataset received an empty path list.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        img = PIL.Image.open(path).convert("RGB")
        if self.img_size > 0 and (img.width != self.img_size or img.height != self.img_size):
            img = img.resize((self.img_size, self.img_size), PIL.Image.Resampling.LANCZOS)

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            # Default: CHW float32 in [0, 1]
            array = np.asarray(img, dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        return tensor, path


def make_dataloader(
    image_paths: Sequence[str],
    batch_size: int = 16,
    img_size: int = 224,
    shuffle: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Convenience DataLoader for flat face folders (images only, no UV)."""
    dataset = FaceFolderDataset(image_paths=image_paths, img_size=img_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
