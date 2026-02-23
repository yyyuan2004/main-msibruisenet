"""Compute per-channel normalization stats from MSI NPY files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = Path(cfg["data"]["image_dir"])
    out_path = Path(cfg["data"]["root"]) / "norm_stats.json"

    sums = np.zeros(int(cfg["data"]["num_channels"]), dtype=np.float64)
    sq_sums = np.zeros_like(sums)
    count = 0

    for p in image_dir.glob("*.npy"):
        x = np.load(p).astype(np.float64)
        pixels = x.reshape(-1, x.shape[-1])
        sums += pixels.sum(axis=0)
        sq_sums += (pixels ** 2).sum(axis=0)
        count += pixels.shape[0]

    if count == 0:
        mean = np.zeros_like(sums)
        std = np.ones_like(sums)
    else:
        mean = sums / count
        var = np.maximum(sq_sums / count - mean ** 2, 1e-12)
        std = np.sqrt(var)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
