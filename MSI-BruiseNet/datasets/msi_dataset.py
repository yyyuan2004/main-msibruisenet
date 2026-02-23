"""Custom dataset for loading MSI image/mask NPY files.

Key I/O:
- Reads from image/mask dirs and split JSON
- Outputs normalized CHW tensor and HW mask tensor
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import build_train_transform, build_val_transform


class MSIDataset(Dataset):
    """MSI segmentation dataset backed by .npy files."""

    # ========== 📂 DATA INPUT PATH (用户需修改) ==========
    # Paths are read from config.yaml and documented here for quick navigation.
    # DATA_ROOT = "data/"
    # IMAGE_DIR = "data/images/"
    # MASK_DIR  = "data/masks/"
    # SPLIT_DIR = "data/splits/"
    # =====================================================

    def __init__(self, cfg: Dict[str, Any], fold: int, split: str, seed: int, train: bool = True) -> None:
        self.cfg = cfg
        self.input_size = int(cfg["data"]["input_size"])
        self.image_dir = Path(cfg["data"]["image_dir"])
        self.mask_dir = Path(cfg["data"]["mask_dir"])
        split_path = Path(cfg["data"]["split_dir"]) / f"folds_seed{seed}.json"
        if not split_path.exists():
            split_path = Path(cfg["data"]["split_dir"]) / "folds.json"
        with split_path.open("r", encoding="utf-8") as f:
            folds = json.load(f)
        self.ids: List[str] = folds[f"fold_{fold}"][split]
        self.transform = build_train_transform(cfg) if train else build_val_transform()
        self.mean, self.std = self._load_norm_stats(Path(cfg["data"]["root"]) / "norm_stats.json")

    @staticmethod
    def _load_norm_stats(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            return np.array(stats["mean"], dtype=np.float32), np.array(stats["std"], dtype=np.float32)
        return np.zeros(9, dtype=np.float32), np.ones(9, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sid = self.ids[idx]
        image = np.load(self.image_dir / f"{sid}.npy")
        mask = np.load(self.mask_dir / f"{sid}.npy")
        image = image.astype(np.float32)
        if image.ndim != 3 or image.shape[-1] != int(self.cfg["data"]["num_channels"]):
            raise ValueError(f"Invalid MSI shape for {sid}: {image.shape}")
        mask = mask.astype(np.uint8)

        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = (image - self.mean) / (self.std + 1e-6)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long(), sid


def create_dataloader(cfg: Dict[str, Any], fold: int, split: str, seed: int, train: bool) -> DataLoader:
    """Create dataloader from config."""
    ds = MSIDataset(cfg, fold=fold, split=split, seed=seed, train=train)
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
