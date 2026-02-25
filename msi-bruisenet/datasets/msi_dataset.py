"""
msi_dataset.py — Custom Dataset for 9-band MSI apple bruise segmentation
=========================================================================

Key I/O:
    Input paths (from config.yaml):
        - data.image_dir  → .npy images  (H, W, 9)  float32/uint16
        - data.mask_dir   → .npy masks   (H, W)      uint8  {0, 1}
        - data.split_dir  → fold JSON    list of sample names
        - data.norm_stats → norm_stats.json  {mean: [...], std: [...]}
    Output:
        image tensor (9, H, W) float32, mask tensor (H, W) int64
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import apply_spectral_augmentation

logger = logging.getLogger(__name__)


class MSIDataset(Dataset):
    """PyTorch Dataset for loading MSI .npy images and binary masks.

    Args:
        sample_names: List of sample identifiers (e.g. ['sample_001', 'sample_002']).
        image_dir: Path to directory containing .npy image files.
        mask_dir: Path to directory containing .npy mask files.
        norm_stats_path: Path to norm_stats.json with per-channel mean/std.
        spatial_transform: albumentations Compose for spatial augmentation.
        spectral_augment: Whether to apply per-channel brightness/noise augmentation.
        augmentation_cfg: 'augmentation' section from config for spectral params.
        input_size: Target spatial size (used if spatial_transform is None).
    """

    def __init__(
        self,
        sample_names: List[str],
        image_dir: str,
        mask_dir: str,
        norm_stats_path: Optional[str] = None,
        spatial_transform: Optional[Callable] = None,
        spectral_augment: bool = False,
        augmentation_cfg: Optional[Dict[str, Any]] = None,
        input_size: int = 512,
    ) -> None:
        super().__init__()
        # ========== 📂 DATA INPUT PATH (用户需修改) ==========
        self.image_dir = image_dir      # .npy 图像目录
        self.mask_dir = mask_dir        # .npy 掩码目录
        # =====================================================
        self.sample_names = sample_names
        self.spatial_transform = spatial_transform
        self.spectral_augment = spectral_augment
        self.augmentation_cfg = augmentation_cfg or {}
        self.input_size = input_size

        # Load normalisation statistics
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        if norm_stats_path and os.path.isfile(norm_stats_path):
            with open(norm_stats_path, "r") as f:
                stats = json.load(f)
            self.mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
            self.std = np.array(stats["std"], dtype=np.float32).reshape(1, 1, -1)
            logger.info("Loaded normalisation stats from %s", norm_stats_path)
        else:
            logger.warning(
                "Norm stats not found at %s — images will NOT be normalised. "
                "Run scripts/compute_norm_stats.py first.",
                norm_stats_path,
            )

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.sample_names[index]

        # --- Load image (.npy) ---
        img_path = os.path.join(self.image_dir, f"{name}.npy")
        image = np.load(img_path).astype(np.float32)  # (H, W, 9)

        # --- Load mask (.npy) ---
        mask_path = os.path.join(self.mask_dir, f"{name}.npy")
        mask = np.load(mask_path).astype(np.uint8)     # (H, W)

        # --- Normalise ---
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / (self.std + 1e-8)

        # --- Spatial augmentation (albumentations) ---
        if self.spatial_transform is not None:
            transformed = self.spatial_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # --- Spectral augmentation (brightness + noise) ---
        if self.spectral_augment:
            br = self.augmentation_cfg.get("brightness_range", [0.8, 1.2])
            ns = self.augmentation_cfg.get("gaussian_noise_std", 0.01)
            image = apply_spectral_augmentation(
                image, brightness_range=tuple(br), noise_std=ns
            )

        # --- Convert to tensors ---
        # image: (H, W, 9) → (9, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).long()

        return image_tensor, mask_tensor


def load_fold_split(
    split_dir: str, fold: int
) -> Tuple[List[str], List[str]]:
    """Load train/val sample names for a specific fold.

    Args:
        split_dir: Directory containing fold split JSON files.
        fold: Fold index (0-based).

    Returns:
        (train_names, val_names) lists of sample name strings.
    """
    # ========== 📂 DATA INPUT PATH (用户需修改) ==========
    split_path = os.path.join(split_dir, "splits.json")
    # =====================================================
    with open(split_path, "r") as f:
        splits = json.load(f)

    fold_key = f"fold_{fold}"
    if fold_key not in splits:
        raise KeyError(f"Fold key '{fold_key}' not found in {split_path}")

    return splits[fold_key]["train"], splits[fold_key]["val"]


def get_dataloaders(
    fold: int,
    cfg: Dict[str, Any],
    spatial_train_transform: Optional[Callable] = None,
    spatial_val_transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders for a given fold.

    Args:
        fold: Fold index.
        cfg: Full config dict.
        spatial_train_transform: Training spatial augmentation.
        spatial_val_transform: Validation spatial augmentation.

    Returns:
        (train_loader, val_loader).
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    aug_cfg = cfg.get("augmentation", {})

    train_names, val_names = load_fold_split(data_cfg["split_dir"], fold)

    train_ds = MSIDataset(
        sample_names=train_names,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        norm_stats_path=data_cfg.get("norm_stats"),
        spatial_transform=spatial_train_transform,
        spectral_augment=True,
        augmentation_cfg=aug_cfg,
        input_size=data_cfg.get("input_size", 512),
    )
    val_ds = MSIDataset(
        sample_names=val_names,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        norm_stats_path=data_cfg.get("norm_stats"),
        spatial_transform=spatial_val_transform,
        spectral_augment=False,
        input_size=data_cfg.get("input_size", 512),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    return train_loader, val_loader
