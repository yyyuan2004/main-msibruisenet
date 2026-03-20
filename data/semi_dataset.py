"""Semi-supervised dataset for Mean Teacher framework.

Provides two dataset classes:
    - MSIDataset (from dataset.py): standard labeled dataset
    - UnlabeledMSIDataset: loads unlabeled .npy images, applies both weak and
      strong augmentation to the SAME image, returning a pair (weak_img, strong_img)
      for consistency regularization.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class UnlabeledMSIDataset(Dataset):
    """Dataset for unlabeled MSI images with dual augmentation (weak + strong).

    For each image, returns TWO augmented versions:
        - weak_image:  lightly augmented (for Teacher pseudo-labeling)
        - strong_image: heavily augmented (for Student prediction)

    Args:
        file_list: List of file stems (without extension).
        data_dir: Root data directory.
        image_dir: Subdirectory name for spectral images.
        weak_transform: Weak augmentation pipeline (flips/rotation only).
        strong_transform: Strong augmentation pipeline (cutout, blur, etc.).
    """

    def __init__(self, file_list, data_dir="data", image_dir="images",
                 weak_transform=None, strong_transform=None):
        self.file_list = file_list
        self.image_root = os.path.join(data_dir, image_dir)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        stem = self.file_list[idx]

        # Load spectral image: (H, W, 9) -> (9, H, W)
        image_path = os.path.join(self.image_root, stem + ".npy")
        image = np.load(image_path).astype(np.float32)
        image = image.transpose(2, 0, 1)  # (9, H, W)

        # Create a dummy mask for transform API compatibility (not used for loss)
        _, h, w = image.shape
        dummy_mask = np.zeros((h, w), dtype=np.int64)

        # Apply weak augmentation → Teacher input
        if self.weak_transform is not None:
            weak_image, _ = self.weak_transform(image.copy(), dummy_mask.copy())
        else:
            weak_image = image.copy()

        # Apply strong augmentation → Student input
        if self.strong_transform is not None:
            strong_image, _ = self.strong_transform(image.copy(), dummy_mask.copy())
        else:
            strong_image = image.copy()

        weak_image = torch.from_numpy(np.ascontiguousarray(weak_image)).float()
        strong_image = torch.from_numpy(np.ascontiguousarray(strong_image)).float()

        return weak_image, strong_image, stem
