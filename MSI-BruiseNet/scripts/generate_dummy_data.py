#!/usr/bin/env python3
"""
generate_dummy_data.py - Generate synthetic MSI data for smoke-testing the pipeline.

Responsibility:
    - Create fake (H, W, 9) float32 .npy images with Gaussian background
      and random elliptical "bruise" regions.
    - Create matching (H, W) uint8 binary masks.
    - Save to data/images/ and data/masks/.
    - This allows the full train/eval pipeline to be tested without real data.

I/O:
    Input  : command-line args (num_samples, height, width)
    Output : data/images/sample_NNN.npy, data/masks/sample_NNN.npy
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("generate_dummy_data")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic MSI data for smoke testing")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of dummy samples to generate")
    parser.add_argument("--height", type=int, default=256,
                        help="Image height in pixels")
    parser.add_argument("--width", type=int, default=256,
                        help="Image width in pixels")
    parser.add_argument("--num-channels", type=int, default=9,
                        help="Number of spectral channels")
    parser.add_argument("--max-bruises", type=int, default=5,
                        help="Maximum number of bruise regions per image")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file if it exists."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def generate_single_sample(
    height: int,
    width: int,
    num_channels: int,
    max_bruises: int,
    rng: np.random.RandomState,
) -> tuple:
    """Generate a single synthetic MSI image and mask.

    The image simulates a multi-band reflectance image with:
    - Background: smooth gradient + Gaussian noise per channel
    - Bruise regions: elliptical areas with modified spectral response

    Args:
        height: Image height.
        width: Image width.
        num_channels: Number of spectral channels.
        max_bruises: Maximum bruise regions per image.
        rng: NumPy random state for reproducibility.

    Returns:
        (image, mask) where image is (H, W, C) float32 and mask is (H, W) uint8.
    """
    # Generate background: base reflectance per band + spatial gradient + noise
    base_reflectance = rng.uniform(0.3, 0.8, size=(1, 1, num_channels)).astype(np.float32)
    y_grad = np.linspace(0.95, 1.05, height).reshape(-1, 1, 1).astype(np.float32)
    x_grad = np.linspace(0.98, 1.02, width).reshape(1, -1, 1).astype(np.float32)

    image = base_reflectance * y_grad * x_grad
    image += rng.normal(0, 0.02, size=(height, width, num_channels)).astype(np.float32)

    mask = np.zeros((height, width), dtype=np.uint8)

    # Generate random elliptical bruise regions
    num_bruises = rng.randint(1, max_bruises + 1)
    for _ in range(num_bruises):
        cy = rng.randint(height // 4, 3 * height // 4)
        cx = rng.randint(width // 4, 3 * width // 4)
        ry = rng.randint(height // 20, height // 6)
        rx = rng.randint(width // 20, width // 6)

        yy, xx = np.ogrid[:height, :width]
        ellipse = ((yy - cy) ** 2 / (ry ** 2 + 1e-8) +
                   (xx - cx) ** 2 / (rx ** 2 + 1e-8)) <= 1.0

        mask[ellipse] = 1

        # Modify spectral response in bruise region:
        # - Decrease reflectance in some bands (simulate absorption)
        # - Increase in others
        spectral_shift = rng.uniform(-0.15, 0.05, size=(1, 1, num_channels)).astype(np.float32)
        image[ellipse] += spectral_shift[0, 0, :]

    # Clip to valid range
    image = np.clip(image, 0.0, 1.0)

    return image, mask


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)

    # ========== 📂 OUTPUT PATHS (从 config.yaml 读取或使用默认值) ==========
    image_dir = cfg.get("data", {}).get("image_dir", "data/images/")
    mask_dir = cfg.get("data", {}).get("mask_dir", "data/masks/")
    # ======================================================================

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    logger.info("Generating %d dummy samples (%dx%dx%d) ...",
                args.num_samples, args.height, args.width, args.num_channels)

    for i in range(args.num_samples):
        name = f"sample_{i + 1:03d}"
        image, mask = generate_single_sample(
            args.height, args.width, args.num_channels,
            args.max_bruises, rng,
        )

        np.save(os.path.join(image_dir, f"{name}.npy"), image)
        np.save(os.path.join(mask_dir, f"{name}.npy"), mask)

        bruise_ratio = mask.sum() / mask.size * 100
        logger.info("  %s: bruise coverage %.1f%%", name, bruise_ratio)

    logger.info("Done. Images saved to %s, masks saved to %s", image_dir, mask_dir)
    logger.info("Next steps:")
    logger.info("  1. python scripts/compute_norm_stats.py")
    logger.info("  2. python scripts/prepare_splits.py")
    logger.info("  3. python scripts/train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
