"""Spectral correlation pre-analysis for normal vs bruise regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def run_spectral_analysis(image_dir: Path, mask_dir: Path, out_dir: Path) -> Dict[str, List[List[float]]]:
    """Compute spectral correlation matrices and plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    normal_pixels, bruise_pixels = [], []
    for img_path in sorted(image_dir.glob("*.npy")):
        sid = img_path.stem
        mask_path = mask_dir / f"{sid}.npy"
        if not mask_path.exists():
            continue
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        flat = img.reshape(-1, img.shape[-1])
        m = mask.reshape(-1)
        if (m == 0).any():
            normal_pixels.append(flat[m == 0])
        if (m == 1).any():
            bruise_pixels.append(flat[m == 1])

    normal = np.concatenate(normal_pixels, axis=0) if normal_pixels else np.zeros((1, 9), dtype=np.float32)
    bruise = np.concatenate(bruise_pixels, axis=0) if bruise_pixels else np.zeros((1, 9), dtype=np.float32)
    r_normal = np.corrcoef(normal, rowvar=False)
    r_bruise = np.corrcoef(bruise, rowvar=False)
    delta = r_normal - r_bruise

    for mat, name in [(r_normal, "R_normal"), (r_bruise, "R_bruise"), (delta, "Delta_R")]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(mat, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(normal.mean(axis=0), label="normal")
    plt.plot(bruise.mean(axis=0), label="bruise")
    plt.legend()
    plt.title("Mean spectral signature")
    plt.tight_layout()
    plt.savefig(out_dir / "spectral_curve.png", dpi=150)
    plt.close()

    result = {
        "R_normal": r_normal.tolist(),
        "R_bruise": r_bruise.tolist(),
        "Delta_R": delta.tolist(),
        "normal_mean": normal.mean(axis=0).tolist(),
        "bruise_mean": bruise.mean(axis=0).tolist(),
    }
    with (out_dir / "spectral_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
