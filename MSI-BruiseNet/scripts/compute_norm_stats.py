#!/usr/bin/env python3
"""
compute_norm_stats.py - Compute per-channel normalization statistics for MSI data.

Responsibility:
    - Iterate all .npy images in data/images/.
    - Compute per-channel mean and std using Welford's online algorithm
      (memory-efficient: does not load all images at once).
    - Save results to data/norm_stats.json.

I/O:
    Input  : data/images/*.npy   (H, W, 9) float32/uint16
    Output : data/norm_stats.json  {"mean": [...], "std": [...]}
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("compute_norm_stats")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute 9-channel mean/std from MSI .npy images")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Config overrides in key=value format")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-notation config overrides."""
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[keys[-1]] = parsed
    return cfg


def compute_stats_welford(image_dir: str, num_channels: int = 9) -> Dict[str, List[float]]:
    """Compute per-channel mean and std using Welford's online algorithm.

    This method is memory-efficient: processes one image at a time.

    Args:
        image_dir: Path to .npy image directory.
        num_channels: Number of spectral channels.

    Returns:
        {"mean": [m1, ..., m9], "std": [s1, ..., s9]}
    """
    count = np.zeros(num_channels, dtype=np.float64)
    mean = np.zeros(num_channels, dtype=np.float64)
    m2 = np.zeros(num_channels, dtype=np.float64)

    npy_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {image_dir}")

    for fname in npy_files:
        img = np.load(os.path.join(image_dir, fname)).astype(np.float64)  # (H, W, C)
        if img.ndim != 3 or img.shape[2] != num_channels:
            logger.warning("Skipping %s: unexpected shape %s", fname, img.shape)
            continue

        # Flatten spatial dims: (H*W, C)
        pixels = img.reshape(-1, num_channels)
        n = pixels.shape[0]

        for c in range(num_channels):
            for val in [pixels[:, c].mean()]:
                # Batch Welford update using per-image mean
                pass
            # More efficient: use batch update
            batch_mean = pixels[:, c].mean()
            batch_var = pixels[:, c].var()
            batch_count = n

            delta = batch_mean - mean[c]
            total_count = count[c] + batch_count
            new_mean = mean[c] + delta * batch_count / total_count
            m2[c] += batch_var * batch_count + delta ** 2 * count[c] * batch_count / total_count
            mean[c] = new_mean
            count[c] = total_count

        logger.info("Processed %s (%d pixels)", fname, n)

    std = np.sqrt(m2 / count)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # ========== 📂 DATA INPUT PATH (从 config.yaml 读取) ==========
    image_dir = cfg["data"]["image_dir"]
    num_channels = cfg["data"].get("num_channels", 9)
    # ===============================================================

    # ========== 📂 OUTPUT PATH ==========
    output_path = cfg["data"].get("norm_stats", "data/norm_stats.json")
    # =====================================

    logger.info("Computing normalization statistics from %s ...", image_dir)
    stats = compute_stats_welford(image_dir, num_channels=num_channels)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved to %s", output_path)
    logger.info("Mean: %s", [f"{v:.6f}" for v in stats["mean"]])
    logger.info("Std:  %s", [f"{v:.6f}" for v in stats["std"]])


if __name__ == "__main__":
    main()
