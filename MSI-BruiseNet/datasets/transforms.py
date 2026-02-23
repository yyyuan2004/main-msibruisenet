"""Data augmentation utilities for MSI segmentation."""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A


def build_train_transform(cfg: Dict[str, Any]) -> A.Compose:
    """Build train transform pipeline."""
    aug = cfg["augmentation"]
    transforms = [
        A.HorizontalFlip(p=float(aug["h_flip"])),
        A.VerticalFlip(p=float(aug["v_flip"])),
        A.Rotate(limit=int(aug["rotate_limit"]), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(float(aug["brightness_range"][0]) - 1.0, float(aug["brightness_range"][1]) - 1.0), contrast_limit=0.1, p=0.3),
        A.GaussNoise(std_range=(0.0, float(aug["gaussian_noise_std"])), p=0.3),
    ]
    if bool(aug["elastic_transform"]):
        transforms.append(A.ElasticTransform(p=0.2))
    return A.Compose(transforms)


def build_val_transform() -> A.Compose:
    """Validation transform (identity)."""
    return A.Compose([])
