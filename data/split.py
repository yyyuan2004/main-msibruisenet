"""Data splitting utilities for train/val/test and k-fold cross-validation."""

import os
import numpy as np
from sklearn.model_selection import KFold


def get_data_splits(data_dir, image_dir="images", seed=42,
                    train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split data into train/val/test sets with a fixed random seed.

    Args:
        data_dir: Root data directory.
        image_dir: Subdirectory containing .npy spectral files.
        seed: Random seed for reproducibility.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.

    Returns:
        dict with keys 'train', 'val', 'test', each mapping to a list of file stems.
    """
    image_root = os.path.join(data_dir, image_dir)
    stems = sorted([f[:-4] for f in os.listdir(image_root) if f.endswith(".npy")])

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(stems))

    n_train = int(len(stems) * train_ratio)
    n_val = int(len(stems) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [stems[i] for i in train_idx],
        "val": [stems[i] for i in val_idx],
        "test": [stems[i] for i in test_idx],
    }


def get_kfold_splits(data_dir, image_dir="images", n_splits=5, seed=42):
    """Generate k-fold cross-validation splits.

    For each fold, one fold is held out as test, and the remaining
    are split into train (80%) and val (20%).

    Args:
        data_dir: Root data directory.
        image_dir: Subdirectory containing .npy spectral files.
        n_splits: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, each with keys 'train', 'val', 'test'.
    """
    image_root = os.path.join(data_dir, image_dir)
    stems = sorted([f[:-4] for f in os.listdir(image_root) if f.endswith(".npy")])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_val_idx, test_idx in kf.split(stems):
        # Further split train_val into train and val
        rng = np.random.RandomState(seed)
        n_val = max(1, int(len(train_val_idx) * 0.2))
        perm = rng.permutation(len(train_val_idx))
        val_idx = train_val_idx[perm[:n_val]]
        train_idx = train_val_idx[perm[n_val:]]

        folds.append({
            "train": [stems[i] for i in train_idx],
            "val": [stems[i] for i in val_idx],
            "test": [stems[i] for i in test_idx],
        })

    return folds
