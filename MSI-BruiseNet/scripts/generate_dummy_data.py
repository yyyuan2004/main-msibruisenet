"""Generate dummy MSI data for smoke testing without real labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    split_dir = root / "splits"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    for i in range(args.num_samples):
        sid = f"sample_{i+1:03d}"
        img = rng.normal(0.5, 0.2, size=(args.size, args.size, 9)).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        mask = np.zeros((args.size, args.size), dtype=np.uint8)
        cx, cy = rng.integers(40, args.size - 40, size=2)
        r = int(rng.integers(10, 35))
        cv2.circle(mask, (int(cx), int(cy)), r, 1, -1)
        np.save(image_dir / f"{sid}.npy", img)
        np.save(mask_dir / f"{sid}.npy", mask)


if __name__ == "__main__":
    main()
