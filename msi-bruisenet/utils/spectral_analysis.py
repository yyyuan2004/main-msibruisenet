"""
spectral_analysis.py — Spectral correlation pre-experiment analysis
====================================================================

Key I/O:
    Input : data/images/ (.npy) + data/masks/ (.npy)
    Output: outputs/spectral_analysis/
        - correlation_normal.png   : 9×9 Pearson correlation heatmap (normal region)
        - correlation_bruise.png   : 9×9 Pearson correlation heatmap (bruise region)
        - correlation_diff.png     : Delta-R heatmap (normal - bruise)
        - spectral_curves.png      : Mean spectral curves (normal vs bruise)
        - spectral_stats.json      : Numerical correlation matrices and mean spectra

This module can be used as a library or run as a standalone script via
    python utils/spectral_analysis.py --config configs/config.yaml
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def collect_spectral_data(
    image_dir: str,
    mask_dir: str,
    sample_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect per-pixel spectra for normal and bruise regions.

    Args:
        image_dir: Path to .npy image directory.
        mask_dir: Path to .npy mask directory.
        sample_names: Optional list of sample names. If None, scan image_dir.

    Returns:
        normal_spectra: (N_normal, 9) array of normal pixel spectra.
        bruise_spectra: (N_bruise, 9) array of bruise pixel spectra.
    """
    if sample_names is None:
        files = sorted([f[:-4] for f in os.listdir(image_dir) if f.endswith(".npy")])
    else:
        files = sample_names

    normal_list: List[np.ndarray] = []
    bruise_list: List[np.ndarray] = []

    for name in files:
        img = np.load(os.path.join(image_dir, f"{name}.npy")).astype(np.float32)  # (H,W,9)
        mask = np.load(os.path.join(mask_dir, f"{name}.npy")).astype(np.uint8)    # (H,W)

        # Flatten spatial dims
        pixels = img.reshape(-1, img.shape[-1])  # (H*W, 9)
        labels = mask.reshape(-1)                 # (H*W,)

        normal_list.append(pixels[labels == 0])
        bruise_list.append(pixels[labels == 1])

    normal_spectra = np.concatenate(normal_list, axis=0) if normal_list else np.empty((0, 9))
    bruise_spectra = np.concatenate(bruise_list, axis=0) if bruise_list else np.empty((0, 9))

    logger.info(
        "Collected %d normal pixels and %d bruise pixels",
        normal_spectra.shape[0], bruise_spectra.shape[0],
    )
    return normal_spectra, bruise_spectra


def compute_correlation_matrix(spectra: np.ndarray) -> np.ndarray:
    """Compute 9×9 Pearson correlation matrix from pixel spectra.

    Args:
        spectra: (N, 9) array.

    Returns:
        (9, 9) correlation matrix.
    """
    if spectra.shape[0] < 2:
        return np.eye(spectra.shape[1])
    return np.corrcoef(spectra.T)


def run_spectral_analysis(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    wavelengths: Optional[List[float]] = None,
    sample_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full spectral correlation analysis and save results.

    Args:
        image_dir: Path to .npy image directory.
        mask_dir: Path to .npy mask directory.
        output_dir: Directory to save outputs.
        wavelengths: List of wavelength labels for axes.
        sample_names: Optional list of sample names.

    Returns:
        Dict with correlation matrices and mean spectra (also saved as JSON).
    """
    os.makedirs(output_dir, exist_ok=True)
    if wavelengths is None:
        wavelengths = [713, 736, 759, 782, 805, 828, 851, 874, 897, 920]
    wl_labels = [str(w) for w in wavelengths[:9]]

    normal_spectra, bruise_spectra = collect_spectral_data(image_dir, mask_dir, sample_names)

    # --- Correlation matrices ---
    R_normal = compute_correlation_matrix(normal_spectra)
    R_bruise = compute_correlation_matrix(bruise_spectra)
    delta_R = R_normal - R_bruise

    # --- Plot correlation heatmaps ---
    for mat, name, title in [
        (R_normal, "correlation_normal", "Normal Region Correlation"),
        (R_bruise, "correlation_bruise", "Bruise Region Correlation"),
        (delta_R, "correlation_diff", "ΔR (Normal − Bruise)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        vmin = -1 if "diff" in name else 0
        sns.heatmap(
            mat[:9, :9], ax=ax, annot=True, fmt=".2f",
            xticklabels=wl_labels, yticklabels=wl_labels,
            cmap="RdBu_r" if "diff" in name else "YlOrRd",
            vmin=vmin, vmax=1, square=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Wavelength (nm)")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s.png", name)

    # --- Mean spectral curves ---
    mean_normal = normal_spectra.mean(axis=0) if normal_spectra.shape[0] > 0 else np.zeros(9)
    mean_bruise = bruise_spectra.mean(axis=0) if bruise_spectra.shape[0] > 0 else np.zeros(9)
    std_normal = normal_spectra.std(axis=0) if normal_spectra.shape[0] > 0 else np.zeros(9)
    std_bruise = bruise_spectra.std(axis=0) if bruise_spectra.shape[0] > 0 else np.zeros(9)

    wl_arr = np.array(wavelengths[:9])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_arr, mean_normal[:9], "b-o", label="Normal")
    ax.fill_between(wl_arr, mean_normal[:9] - std_normal[:9], mean_normal[:9] + std_normal[:9], alpha=0.2, color="blue")
    ax.plot(wl_arr, mean_bruise[:9], "r-s", label="Bruise")
    ax.fill_between(wl_arr, mean_bruise[:9] - std_bruise[:9], mean_bruise[:9] + std_bruise[:9], alpha=0.2, color="red")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Mean Spectral Curves: Normal vs Bruise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectral_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spectral_curves.png")

    # --- Save numerical results ---
    results = {
        "R_normal": R_normal[:9, :9].tolist(),
        "R_bruise": R_bruise[:9, :9].tolist(),
        "delta_R": delta_R[:9, :9].tolist(),
        "mean_normal": mean_normal[:9].tolist(),
        "mean_bruise": mean_bruise[:9].tolist(),
        "std_normal": std_normal[:9].tolist(),
        "std_bruise": std_bruise[:9].tolist(),
        "wavelengths": wavelengths[:9],
        "n_normal_pixels": int(normal_spectra.shape[0]),
        "n_bruise_pixels": int(bruise_spectra.shape[0]),
    }
    with open(os.path.join(output_dir, "spectral_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved spectral_stats.json")

    return results


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Spectral correlation pre-analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    output_dir = os.path.join(cfg["output"]["root"], "spectral_analysis")

    run_spectral_analysis(
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        output_dir=output_dir,
        wavelengths=data_cfg.get("wavelengths"),
    )
