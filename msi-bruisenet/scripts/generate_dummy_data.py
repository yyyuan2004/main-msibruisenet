"""
generate_dummy_data.py — Generate synthetic data for smoke testing
===================================================================

Key I/O:
    Input : configs/config.yaml (reads data paths and num_channels)
    Output:
        - data/images/sample_001..sample_020.npy — random MSI images
        - data/masks/sample_001..sample_020.npy  — random binary masks

Creates dummy .npy files so that the full training pipeline can be
tested end-to-end without real data.

Usage:
    python scripts/generate_dummy_data.py --config configs/config.yaml
    python scripts/generate_dummy_data.py --num-samples 50 --height 256 --width 256
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_dummy_data(
    image_dir: str,
    mask_dir: str,
    num_samples: int = 20,
    height: int = 256,
    width: int = 256,
    num_channels: int = 9,
    seed: int = 42,
) -> None:
    """Generate random .npy images and binary masks for testing.

    Args:
        image_dir: Output directory for .npy images.
        mask_dir: Output directory for .npy masks.
        num_samples: Number of samples to generate.
        height: Image height.
        width: Image width.
        num_channels: Number of spectral channels.
        seed: Random seed.
    """
    np.random.seed(seed)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(1, num_samples + 1):
        name = f"sample_{i:03d}"

        # Generate a random MSI image with realistic-ish values
        # Simulate reflectance in [0, 1] range with some spatial structure
        base = np.random.uniform(0.3, 0.7, size=(height, width, 1)).astype(np.float32)
        noise = np.random.normal(0, 0.05, size=(height, width, num_channels)).astype(np.float32)
        image = np.clip(base + noise, 0, 1).astype(np.float32)

        # Generate a random binary mask with circular "bruise" regions
        mask = np.zeros((height, width), dtype=np.uint8)
        num_bruises = np.random.randint(0, 4)
        for _ in range(num_bruises):
            cy = np.random.randint(20, height - 20)
            cx = np.random.randint(20, width - 20)
            radius = np.random.randint(10, min(40, height // 4))
            yy, xx = np.ogrid[:height, :width]
            circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            mask[circle] = 1

        np.save(os.path.join(image_dir, f"{name}.npy"), image)
        np.save(os.path.join(mask_dir, f"{name}.npy"), mask)

    logger.info(
        "Generated %d dummy samples in %s and %s", num_samples, image_dir, mask_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy data for smoke testing")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # ========== 📂 DATA INPUT PATH (用户需修改) ==========
    image_dir = data_cfg["image_dir"]
    mask_dir = data_cfg["mask_dir"]
    # =====================================================

    generate_dummy_data(
        image_dir=image_dir,
        mask_dir=mask_dir,
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        num_channels=data_cfg.get("num_channels", 9),
    )


if __name__ == "__main__":
    main()
