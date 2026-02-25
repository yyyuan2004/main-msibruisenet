"""
transforms.py - Spatial augmentation pipelines for MSI data.

Responsibility:
    - Provide training augmentations (flip, rotate, elastic, brightness, noise).
    - Provide validation transforms (no augmentation, just ensure correct format).
    - All transforms operate on (H, W, C) numpy arrays (albumentations convention).

I/O:
    Input  : augmentation config dict from config.yaml
    Output : albumentations.Compose objects
"""

from typing import Any, Dict, Optional

import albumentations as A


def get_train_transforms(aug_cfg: Dict[str, Any], input_size: int = 512) -> A.Compose:
    """Build training augmentation pipeline.

    Args:
        aug_cfg: Augmentation config section from config.yaml.
        input_size: Target spatial size (square).

    Returns:
        albumentations.Compose transform.
    """
    transforms_list = []

    # Horizontal flip
    if aug_cfg.get("h_flip", 0) > 0:
        transforms_list.append(A.HorizontalFlip(p=aug_cfg["h_flip"]))

    # Vertical flip
    if aug_cfg.get("v_flip", 0) > 0:
        transforms_list.append(A.VerticalFlip(p=aug_cfg["v_flip"]))

    # Random rotation
    rotate_limit = aug_cfg.get("rotate_limit", 0)
    if rotate_limit > 0:
        transforms_list.append(A.Rotate(limit=rotate_limit, p=0.5, border_mode=0))

    # Elastic transform
    if aug_cfg.get("elastic_transform", False):
        transforms_list.append(
            A.ElasticTransform(alpha=50, sigma=10, p=0.3)
        )

    # Brightness adjustment (per-pixel multiplicative)
    br = aug_cfg.get("brightness_range", None)
    if br is not None and len(br) == 2:
        # Use RandomBrightnessContrast as a proxy; brightness_limit is symmetric
        limit = max(abs(br[0] - 1.0), abs(br[1] - 1.0))
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=limit, contrast_limit=0, p=0.5
            )
        )

    # Gaussian noise
    noise_std = aug_cfg.get("gaussian_noise_std", 0)
    if noise_std > 0:
        transforms_list.append(
            A.GaussNoise(p=0.3)
        )

    return A.Compose(transforms_list)


def get_val_transforms(input_size: int = 512) -> Optional[A.Compose]:
    """Build validation transform pipeline (no augmentation).

    Args:
        input_size: Target spatial size (square).

    Returns:
        None (no transforms needed; resize is handled in Dataset).
    """
    # Resize is handled in the Dataset class; no augmentation for validation.
    return None
