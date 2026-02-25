"""
transforms.py — Spatial augmentation pipeline for MSI data
===========================================================

Key I/O:
    Input : (H, W, 9) float32 image + (H, W) uint8 mask
    Output: augmented image + mask (same shapes)

Uses albumentations for spatial transforms that apply identically to
both the multi-channel image and the binary mask.
"""

from typing import Any, Dict, Optional, Tuple

import albumentations as A
import numpy as np


def build_train_transforms(cfg: Dict[str, Any], input_size: int = 512) -> A.Compose:
    """Build training augmentation pipeline from config.

    Args:
        cfg: The 'augmentation' section of config.yaml.
        input_size: Target spatial size (H=W).

    Returns:
        albumentations.Compose pipeline.
    """
    aug_list = [
        A.Resize(input_size, input_size, always_apply=True),
    ]

    if cfg.get("h_flip", 0) > 0:
        aug_list.append(A.HorizontalFlip(p=cfg["h_flip"]))

    if cfg.get("v_flip", 0) > 0:
        aug_list.append(A.VerticalFlip(p=cfg["v_flip"]))

    rotate_limit = cfg.get("rotate_limit", 0)
    if rotate_limit > 0:
        aug_list.append(A.Rotate(limit=rotate_limit, border_mode=0, p=0.5))

    if cfg.get("elastic_transform", False):
        aug_list.append(
            A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.3)
        )

    # Note: brightness and noise are handled per-channel in the dataset
    # because albumentations expects 1-3 channel images by default.

    return A.Compose(aug_list)


def build_val_transforms(input_size: int = 512) -> A.Compose:
    """Build validation/test augmentation (resize only).

    Args:
        input_size: Target spatial size (H=W).
    """
    return A.Compose([
        A.Resize(input_size, input_size, always_apply=True),
    ])


def apply_spectral_augmentation(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.01,
) -> np.ndarray:
    """Apply per-channel brightness jitter and Gaussian noise.

    Args:
        image: (H, W, C) float32 normalised image.
        brightness_range: (low, high) multiplicative brightness factor.
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        Augmented image (H, W, C) float32.
    """
    # Random brightness per channel
    c = image.shape[-1]
    factors = np.random.uniform(brightness_range[0], brightness_range[1], size=(1, 1, c)).astype(np.float32)
    image = image * factors

    # Additive Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
        image = image + noise

    return image
