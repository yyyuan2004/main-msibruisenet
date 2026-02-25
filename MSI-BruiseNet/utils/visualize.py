"""
visualize.py - Visualization utilities for MSI-BruiseNet.

Implements:
    - Prediction overlay on pseudo-RGB composite.
    - Attention weight heatmap visualization.
    - Spectral residual heatmap.
    - Side-by-side comparison plots.

I/O:
    Input  : numpy arrays (images, masks, predictions, attention maps)
    Output : matplotlib figures saved as PNG to output directory
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def make_pseudo_rgb(msi_image: np.ndarray, bands: Tuple[int, int, int] = (4, 2, 0)) -> np.ndarray:
    """Create pseudo-RGB from MSI image by selecting 3 bands.

    Args:
        msi_image: (H, W, 9) MSI image array.
        bands: Tuple of 3 band indices for R, G, B channels.

    Returns:
        (H, W, 3) float32 array normalized to [0, 1].
    """
    rgb = msi_image[:, :, list(bands)].astype(np.float32)
    for c in range(3):
        vmin, vmax = rgb[:, :, c].min(), rgb[:, :, c].max()
        if vmax > vmin:
            rgb[:, :, c] = (rgb[:, :, c] - vmin) / (vmax - vmin)
        else:
            rgb[:, :, c] = 0.0
    return rgb


def plot_prediction_overlay(
    msi_image: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    save_path: str,
    sample_name: str = "",
    alpha: float = 0.4,
) -> None:
    """Plot pseudo-RGB with GT and prediction overlays side by side.

    Args:
        msi_image: (H, W, 9) raw MSI image.
        mask_gt: (H, W) ground truth binary mask.
        mask_pred: (H, W) predicted binary mask.
        save_path: Path to save the figure.
        sample_name: Sample identifier for title.
        alpha: Overlay transparency.
    """
    rgb = make_pseudo_rgb(msi_image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Pseudo-RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Pseudo-RGB")
    axes[0].axis("off")

    # Ground truth overlay
    overlay_gt = rgb.copy()
    overlay_gt[mask_gt == 1] = overlay_gt[mask_gt == 1] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    axes[1].imshow(overlay_gt)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction overlay
    overlay_pred = rgb.copy()
    overlay_pred[mask_pred == 1] = overlay_pred[mask_pred == 1] * (1 - alpha) + np.array([0, 0, 1]) * alpha
    axes[2].imshow(overlay_pred)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.suptitle(sample_name, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction overlay to %s", save_path)


def plot_attention_heatmap(
    attention_map: np.ndarray,
    save_path: str,
    title: str = "Attention Map",
    cmap: str = "jet",
) -> None:
    """Plot attention weight heatmap.

    Args:
        attention_map: (H, W) attention weights.
        save_path: Path to save the figure.
        title: Figure title.
        cmap: Matplotlib colormap name.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attention_map, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_residual_heatmap(
    spectral_residual: np.ndarray,
    save_path: str,
    title: str = "Spectral Residual",
) -> None:
    """Plot spectral residual heatmap (mean across channels).

    Args:
        spectral_residual: (C, H, W) or (H, W) spectral residual.
        save_path: Path to save figure.
        title: Figure title.
    """
    if spectral_residual.ndim == 3:
        residual = spectral_residual.mean(axis=0)
    else:
        residual = spectral_residual

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(residual, cmap="RdBu_r")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Method Comparison",
) -> None:
    """Bar chart comparing metrics across methods.

    Args:
        results: {method_name: {metric_name: value}}.
        save_path: Path to save figure.
        title: Figure title.
    """
    methods = list(results.keys())
    if not methods:
        return

    metrics = list(results[methods[0]].keys())
    x = np.arange(len(methods))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in methods]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
