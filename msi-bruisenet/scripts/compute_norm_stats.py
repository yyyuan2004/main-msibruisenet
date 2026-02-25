"""
compute_norm_stats.py — Compute per-channel mean and std for MSI images
========================================================================

Key I/O:
    Input : data/images/ — all .npy files (H, W, 9)
    Output: data/norm_stats.json — {"mean": [m1..m9], "std": [s1..s9]}

Uses Welford's online algorithm to handle large datasets without
loading all images into memory simultaneously.

Usage:
    python scripts/compute_norm_stats.py --config configs/config.yaml
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_norm_stats(image_dir: str, num_channels: int = 9) -> Dict[str, list]:
    """Compute per-channel mean and std across all .npy images.

    Uses a two-pass approach for numerical stability:
    Pass 1 — compute global mean per channel.
    Pass 2 — compute global std per channel.

    Args:
        image_dir: Directory containing .npy image files.
        num_channels: Expected number of channels.

    Returns:
        Dict with 'mean' and 'std' lists (length = num_channels).
    """
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
    if not files:
        logger.error("No .npy files found in %s", image_dir)
        sys.exit(1)

    logger.info("Computing normalisation stats over %d files...", len(files))

    # Pass 1: Mean
    pixel_sum = np.zeros(num_channels, dtype=np.float64)
    pixel_count = 0
    for fname in tqdm(files, desc="Pass 1 (mean)"):
        img = np.load(os.path.join(image_dir, fname)).astype(np.float64)
        # Handle (H, W, C) or (C, H, W)
        if img.ndim == 3 and img.shape[0] == num_channels:
            img = img.transpose(1, 2, 0)  # → (H, W, C)
        h, w, c = img.shape
        assert c == num_channels, f"Expected {num_channels} channels, got {c} in {fname}"
        pixel_sum += img.reshape(-1, c).sum(axis=0)
        pixel_count += h * w

    mean = pixel_sum / pixel_count

    # Pass 2: Std
    pixel_sq_diff_sum = np.zeros(num_channels, dtype=np.float64)
    for fname in tqdm(files, desc="Pass 2 (std)"):
        img = np.load(os.path.join(image_dir, fname)).astype(np.float64)
        if img.ndim == 3 and img.shape[0] == num_channels:
            img = img.transpose(1, 2, 0)
        pixels = img.reshape(-1, img.shape[-1])
        pixel_sq_diff_sum += ((pixels - mean) ** 2).sum(axis=0)

    std = np.sqrt(pixel_sq_diff_sum / pixel_count)

    logger.info("Mean: %s", mean.tolist())
    logger.info("Std : %s", std.tolist())

    return {"mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel normalisation statistics")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # ========== 📂 DATA INPUT PATH (用户需修改) ==========
    image_dir = data_cfg["image_dir"]
    # =====================================================

    stats = compute_norm_stats(image_dir, num_channels=data_cfg.get("num_channels", 9))

    out_path = data_cfg.get("norm_stats", os.path.join(data_cfg["root"], "norm_stats.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved normalisation stats to %s", out_path)


if __name__ == "__main__":
    main()
