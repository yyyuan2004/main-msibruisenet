#!/usr/bin/env python3
"""
prepare_splits.py - Generate 5-fold cross-validation split indices.

Responsibility:
    - Scan data/images/ for all .npy sample files.
    - Generate K-Fold (default 5) train/val splits.
    - Save splits to data/splits/splits.json.

I/O:
    Input  : data/images/*.npy  (file names scanned, not loaded)
    Output : data/splits/splits.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import yaml
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("prepare_splits")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate 5-fold split indices")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Config overrides in key=value format")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
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


def scan_samples(image_dir: str) -> List[str]:
    """Scan image directory for .npy files and return sorted sample names.

    Args:
        image_dir: Path to directory containing .npy image files.

    Returns:
        Sorted list of sample names (without .npy extension).
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    names = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".npy"):
            names.append(fname[:-4])  # strip .npy

    if not names:
        raise RuntimeError(f"No .npy files found in {image_dir}")

    return names


def generate_splits(
    sample_names: List[str],
    num_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate K-Fold cross-validation splits.

    Args:
        sample_names: List of sample identifiers.
        num_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        Dict with fold_0..fold_N-1, each containing train/val name lists.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits: Dict[str, Dict[str, List[str]]] = {}

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(sample_names)):
        splits[f"fold_{fold_idx}"] = {
            "train": [sample_names[i] for i in train_indices],
            "val": [sample_names[i] for i in val_indices],
        }

    return splits


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
    split_dir = cfg["data"]["split_dir"]
    # ===============================================================

    num_folds = cfg["train"].get("num_folds", 5)

    logger.info("Scanning samples from %s ...", image_dir)
    sample_names = scan_samples(image_dir)
    logger.info("Found %d samples", len(sample_names))

    logger.info("Generating %d-fold splits (seed=%d) ...", num_folds, args.seed)
    splits = generate_splits(sample_names, num_folds=num_folds, seed=args.seed)

    # Save
    os.makedirs(split_dir, exist_ok=True)
    output_path = os.path.join(split_dir, "splits.json")
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info("Splits saved to %s", output_path)
    for fold_key, fold_data in splits.items():
        logger.info("  %s: %d train, %d val",
                     fold_key, len(fold_data["train"]), len(fold_data["val"]))


if __name__ == "__main__":
    main()
