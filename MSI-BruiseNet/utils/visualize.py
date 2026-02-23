"""Visualization helpers for attention, residual and prediction overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_overlay(image_hw9: np.ndarray, pred: np.ndarray, target: np.ndarray, save_path: Path) -> None:
    """Save prediction overlay visualization."""
    base = image_hw9[..., 0]
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[..., 1] = np.clip(rgb[..., 1] + pred * 0.5, 0, 1)
    rgb[..., 0] = np.clip(rgb[..., 0] + target * 0.5, 0, 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
