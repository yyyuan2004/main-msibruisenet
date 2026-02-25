"""
spectral_analysis.py - Spectral correlation pre-experiment for MSI data.

Responsibility:
    - Compute 9×9 Pearson correlation matrices for normal vs. bruise regions.
    - Compute difference matrix ΔR = R_normal - R_bruise.
    - Plot spectral mean curves (normal vs. bruise) across 9 bands.
    - Save results as PNG heatmaps and JSON.

I/O:
    Input  : data/images/*.npy, data/masks/*.npy (from config paths)
    Output : outputs/spectral_analysis/ (PNG + JSON)
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

DEFAULT_WAVELENGTHS = [713, 736, 759, 782, 805, 828, 851, 874, 897, 920]


def collect_spectral_pixels(
    image_dir: str,
    mask_dir: str,
    sample_names: List[str],
    max_pixels: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect spectral vectors for normal and bruise pixels.

    Args:
        image_dir: Path to .npy image directory.
        mask_dir: Path to .npy mask directory.
        sample_names: List of sample names (without .npy extension).
        max_pixels: Maximum pixels to collect per class (for memory).

    Returns:
        (normal_pixels, bruise_pixels) each of shape (N, 9).
    """
    normal_list: List[np.ndarray] = []
    bruise_list: List[np.ndarray] = []

    for name in sample_names:
        img = np.load(os.path.join(image_dir, f"{name}.npy")).astype(np.float32)
        mask = np.load(os.path.join(mask_dir, f"{name}.npy")).astype(np.uint8)

        # Flatten spatial dims
        h, w, c = img.shape
        flat_img = img.reshape(-1, c)  # (H*W, 9)
        flat_mask = mask.reshape(-1)

        normal_list.append(flat_img[flat_mask == 0])
        bruise_list.append(flat_img[flat_mask == 1])

    normal_all = np.concatenate(normal_list, axis=0) if normal_list else np.empty((0, 9))
    bruise_all = np.concatenate(bruise_list, axis=0) if bruise_list else np.empty((0, 9))

    # Subsample if too many pixels
    if len(normal_all) > max_pixels:
        idx = np.random.choice(len(normal_all), max_pixels, replace=False)
        normal_all = normal_all[idx]
    if len(bruise_all) > max_pixels:
        idx = np.random.choice(len(bruise_all), max_pixels, replace=False)
        bruise_all = bruise_all[idx]

    return normal_all, bruise_all


def compute_correlation_matrix(pixels: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix across spectral bands.

    Args:
        pixels: (N, 9) spectral pixel vectors.

    Returns:
        (9, 9) correlation matrix.
    """
    if len(pixels) < 2:
        return np.eye(pixels.shape[1] if pixels.ndim == 2 else 9)
    return np.corrcoef(pixels, rowvar=False)


def run_spectral_analysis(
    image_dir: str,
    mask_dir: str,
    sample_names: List[str],
    output_dir: str,
    wavelengths: List[int] = None,
) -> Dict[str, Any]:
    """Run full spectral correlation analysis and save results.

    Args:
        image_dir: Path to image directory.
        mask_dir: Path to mask directory.
        sample_names: List of sample names.
        output_dir: Directory to save outputs.
        wavelengths: List of wavelength labels.

    Returns:
        Dict with correlation matrices and spectral statistics.
    """
    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS
    wl_labels = [str(w) for w in wavelengths]

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Collecting spectral pixels from %d samples...", len(sample_names))

    normal_pixels, bruise_pixels = collect_spectral_pixels(
        image_dir, mask_dir, sample_names
    )
    logger.info("Normal pixels: %d, Bruise pixels: %d", len(normal_pixels), len(bruise_pixels))

    # Correlation matrices
    r_normal = compute_correlation_matrix(normal_pixels)
    r_bruise = compute_correlation_matrix(bruise_pixels)
    delta_r = r_normal - r_bruise

    # Spectral mean curves
    normal_mean = normal_pixels.mean(axis=0) if len(normal_pixels) > 0 else np.zeros(len(wavelengths))
    normal_std = normal_pixels.std(axis=0) if len(normal_pixels) > 0 else np.zeros(len(wavelengths))
    bruise_mean = bruise_pixels.mean(axis=0) if len(bruise_pixels) > 0 else np.zeros(len(wavelengths))
    bruise_std = bruise_pixels.std(axis=0) if len(bruise_pixels) > 0 else np.zeros(len(wavelengths))

    # --- Plot 1: Normal correlation matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(r_normal, annot=True, fmt=".2f", xticklabels=wl_labels,
                yticklabels=wl_labels, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Normal Region - Band Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "corr_normal.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Bruise correlation matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(r_bruise, annot=True, fmt=".2f", xticklabels=wl_labels,
                yticklabels=wl_labels, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Bruise Region - Band Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "corr_bruise.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Difference matrix ---
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(delta_r, annot=True, fmt=".2f", xticklabels=wl_labels,
                yticklabels=wl_labels, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("ΔR = R_normal - R_bruise")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "corr_delta.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: Spectral mean curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(wavelengths))
    ax.errorbar(x, normal_mean, yerr=normal_std, label="Normal", marker="o", capsize=3)
    ax.errorbar(x, bruise_mean, yerr=bruise_std, label="Bruise", marker="s", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(wl_labels, rotation=45)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean Reflectance")
    ax.set_title("Spectral Mean Curves: Normal vs. Bruise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectral_curves.png"), dpi=150)
    plt.close(fig)

    # Save JSON summary
    summary = {
        "num_normal_pixels": int(len(normal_pixels)),
        "num_bruise_pixels": int(len(bruise_pixels)),
        "normal_mean": normal_mean.tolist(),
        "normal_std": normal_std.tolist(),
        "bruise_mean": bruise_mean.tolist(),
        "bruise_std": bruise_std.tolist(),
        "R_normal": r_normal.tolist(),
        "R_bruise": r_bruise.tolist(),
        "delta_R": delta_r.tolist(),
    }
    with open(os.path.join(output_dir, "spectral_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Spectral analysis saved to %s", output_dir)
    return summary
