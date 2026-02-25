"""
prepare_splits.py — Generate 5-fold cross-validation split indices
===================================================================

Key I/O:
    Input : data/images/ directory (scans for .npy filenames)
    Output: data/splits/splits.json — JSON with fold_0..fold_4 train/val lists

Usage:
    python scripts/prepare_splits.py --config configs/config.yaml
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import yaml
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def discover_samples(image_dir: str) -> List[str]:
    """Discover sample names from the image directory.

    Args:
        image_dir: Path to directory with .npy image files.

    Returns:
        Sorted list of sample name strings (without .npy extension).
    """
    if not os.path.isdir(image_dir):
        logger.error("Image directory does not exist: %s", image_dir)
        sys.exit(1)

    names = sorted([f[:-4] for f in os.listdir(image_dir) if f.endswith(".npy")])
    logger.info("Found %d samples in %s", len(names), image_dir)
    return names


def generate_splits(
    sample_names: List[str], num_folds: int = 5, seed: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    """Generate stratified K-fold splits.

    Args:
        sample_names: List of all sample identifiers.
        num_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys fold_0..fold_{K-1}, each containing train/val lists.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits: Dict[str, Dict[str, List[str]]] = {}

    names_array = np.array(sample_names)
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(names_array)):
        splits[f"fold_{fold_idx}"] = {
            "train": names_array[train_indices].tolist(),
            "val": names_array[val_indices].tolist(),
        }
        logger.info(
            "Fold %d: %d train, %d val",
            fold_idx, len(train_indices), len(val_indices),
        )

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5-fold cross-validation splits")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    # ========== 📂 DATA INPUT PATH (用户需修改) ==========
    image_dir = data_cfg["image_dir"]
    split_dir = data_cfg["split_dir"]
    # =====================================================

    sample_names = discover_samples(image_dir)
    if len(sample_names) == 0:
        logger.error("No .npy files found. Please place your data in %s", image_dir)
        sys.exit(1)

    num_folds = train_cfg.get("num_folds", 5)
    splits = generate_splits(sample_names, num_folds=num_folds, seed=42)

    os.makedirs(split_dir, exist_ok=True)
    out_path = os.path.join(split_dir, "splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Saved splits to %s", out_path)


if __name__ == "__main__":
    main()
