"""
msi_dataset.py - Custom PyTorch Dataset for 9-band MSI apple bruise segmentation.

Responsibility:
    - Load .npy images (H, W, 9) and masks (H, W) from disk.
    - Apply per-channel normalization using precomputed mean/std.
    - Resize to target input size.
    - Apply spatial augmentations (training only).
    - Provide DataLoader construction with 5-fold split support.

I/O:
    Input paths (from config.yaml):
        data.image_dir  -> .npy images   (default: data/images/)
        data.mask_dir   -> .npy masks    (default: data/masks/)
        data.split_dir  -> fold JSON     (default: data/splits/)
        data.norm_stats -> norm JSON     (default: data/norm_stats.json)
    Output:
        dict with 'image' (C, H, W) float32 tensor, 'mask' (H, W) long tensor,
        'name' str sample identifier.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class MSIDataset(Dataset):
    """Dataset for 9-band MSI .npy images and binary masks.

    Args:
        sample_names: List of sample identifiers (without extension).
        image_dir: Directory containing .npy image files.
        mask_dir: Directory containing .npy mask files.
        input_size: Target spatial size (H, W) for resize.
        mean: Per-channel mean for normalization (length 9).
        std: Per-channel std for normalization (length 9).
        transform: Optional albumentations-style transform.
    """

    def __init__(
        self,
        sample_names: List[str],
        image_dir: str,
        mask_dir: str,
        input_size: int = 512,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.sample_names = sample_names
        # ========== 📂 DATA INPUT PATH (从 config 传入) ==========
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # =========================================================
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.std = np.array(std, dtype=np.float32) if std is not None else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name = self.sample_names[idx]
        img_path = os.path.join(self.image_dir, f"{name}.npy")
        mask_path = os.path.join(self.mask_dir, f"{name}.npy")

        # Load .npy arrays
        image = np.load(img_path).astype(np.float32)  # (H, W, 9)
        mask = np.load(mask_path).astype(np.uint8)     # (H, W)

        # Resize to target size
        if image.shape[0] != self.input_size or image.shape[1] != self.input_size:
            image = cv2.resize(image, (self.input_size, self.input_size),
                               interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.input_size, self.input_size),
                              interpolation=cv2.INTER_NEAREST)

        # Normalize per channel
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / (self.std + 1e-8)

        # Apply spatial augmentations (albumentations expects HWC, uint8 mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to torch tensors: image (9, H, W), mask (H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask.copy()).long()

        return {"image": image, "mask": mask, "name": name}


def load_norm_stats(norm_path: str) -> Tuple[List[float], List[float]]:
    """Load precomputed per-channel mean and std from JSON.

    Args:
        norm_path: Path to norm_stats.json.

    Returns:
        (mean_list, std_list) each of length num_channels.
    """
    if not os.path.exists(norm_path):
        logger.warning(
            "Norm stats file not found at %s. Using default (zeros mean, ones std). "
            "Run scripts/compute_norm_stats.py first.", norm_path
        )
        return [0.0] * 9, [1.0] * 9

    with open(norm_path, "r") as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def load_fold_split(split_dir: str, fold: int) -> Tuple[List[str], List[str]]:
    """Load train/val sample names for a given fold.

    Args:
        split_dir: Directory containing fold JSON files.
        fold: Fold index.

    Returns:
        (train_names, val_names).
    """
    split_file = os.path.join(split_dir, "splits.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            "Run scripts/prepare_splits.py first."
        )
    with open(split_file, "r") as f:
        splits = json.load(f)

    fold_key = f"fold_{fold}"
    if fold_key not in splits:
        raise KeyError(f"Fold '{fold_key}' not found in {split_file}")

    return splits[fold_key]["train"], splits[fold_key]["val"]


def get_dataloaders(
    cfg: Dict[str, Any],
    fold: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for a specific fold.

    Args:
        cfg: Full configuration dictionary.
        fold: Fold index (0-based).

    Returns:
        (train_loader, val_loader).
    """
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    acfg = cfg["augmentation"]

    # ========== 📂 DATA INPUT PATHS (从 config.yaml 读取) ==========
    image_dir = dcfg["image_dir"]
    mask_dir = dcfg["mask_dir"]
    split_dir = dcfg["split_dir"]
    norm_path = dcfg.get("norm_stats", "data/norm_stats.json")
    # ===============================================================

    # Load split and normalization stats
    train_names, val_names = load_fold_split(split_dir, fold)
    mean, std = load_norm_stats(norm_path)
    input_size = dcfg.get("input_size", 512)

    logger.info("Fold %d: %d train, %d val samples", fold, len(train_names), len(val_names))

    # Build datasets
    train_ds = MSIDataset(
        sample_names=train_names,
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_size=input_size,
        mean=mean,
        std=std,
        transform=get_train_transforms(acfg, input_size),
    )
    val_ds = MSIDataset(
        sample_names=val_names,
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_size=input_size,
        mean=mean,
        std=std,
        transform=get_val_transforms(input_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.get("batch_size", 4),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=tcfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.get("batch_size", 4),
        shuffle=False,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=tcfg.get("pin_memory", True),
        drop_last=False,
    )

    return train_loader, val_loader
