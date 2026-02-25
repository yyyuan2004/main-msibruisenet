"""
visualize.py — Visualization utilities for MSI-BruiseNet
=========================================================

Key I/O:
    Input : predictions, ground truth masks, attention maps, spectral data
    Output: PNG images saved to outputs/predictions/ or outputs/spectral_analysis/

Functions:
    - overlay_prediction  : Overlay predicted mask on a selected band image
    - plot_attention_map  : Visualise attention weight maps
    - plot_spectral_heatmap : Plot spectral residual as a heatmap
    - plot_confusion_matrix : Pretty confusion matrix figure
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Colour map: background = transparent, bruise = red overlay
BRUISE_COLOUR = np.array([255, 0, 0], dtype=np.uint8)
ALPHA = 0.4


def overlay_prediction(
    image_band: np.ndarray,
    pred: np.ndarray,
    target: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "",
) -> None:
    """Overlay predicted (and optional ground truth) mask on a single band image.

    Args:
        image_band: (H, W) single-channel image (e.g. band index 4 = 805 nm).
        pred: (H, W) predicted binary mask.
        target: (H, W) ground truth mask (optional).
        save_path: If provided, save the figure to this path.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 2 if target is not None else 1, figsize=(12, 5))
    if target is None:
        axes = [axes]

    # Normalise band for display
    band_norm = (image_band - image_band.min()) / (image_band.max() - image_band.min() + 1e-8)
    band_rgb = np.stack([band_norm] * 3, axis=-1)

    # Prediction overlay
    overlay = band_rgb.copy()
    overlay[pred == 1] = overlay[pred == 1] * (1 - ALPHA) + (BRUISE_COLOUR / 255.0) * ALPHA
    axes[0].imshow(overlay)
    axes[0].set_title(f"Prediction — {title}")
    axes[0].axis("off")

    if target is not None:
        gt_overlay = band_rgb.copy()
        gt_overlay[target == 1] = (
            gt_overlay[target == 1] * (1 - ALPHA) + (np.array([0, 255, 0]) / 255.0) * ALPHA
        )
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved overlay to %s", save_path)
    plt.close(fig)


def plot_attention_map(
    attn_weights: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Attention Map",
) -> None:
    """Visualise a 2D attention weight map.

    Args:
        attn_weights: (H, W) attention map in [0, 1].
        save_path: Output file path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_weights, cmap="jet", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_heatmap(
    residual: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Spectral Residual Heatmap",
) -> None:
    """Plot the mean spectral residual magnitude as a heatmap.

    Args:
        residual: (C, H, W) or (H, W, C) spectral residual.
        save_path: Output file path.
        title: Figure title.
    """
    if residual.ndim == 3:
        if residual.shape[0] < residual.shape[-1]:
            # (C, H, W) → mean over channels
            heatmap = np.abs(residual).mean(axis=0)
        else:
            heatmap = np.abs(residual).mean(axis=-1)
    else:
        heatmap = np.abs(residual)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap="hot")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Background", "Bruise"],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix.

    Args:
        cm: (num_classes, num_classes) confusion matrix.
        class_names: List of class label strings.
        save_path: Output file path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
