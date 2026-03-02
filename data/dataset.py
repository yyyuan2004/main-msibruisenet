"""MSI Dataset for 9-channel near-infrared multispectral apple defect segmentation."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MSIDataset(Dataset):
    """Dataset for loading 9-channel .npy spectral images and corresponding masks.

    Args:
        file_list: List of file stems (without extension).
        data_dir: Root data directory.
        image_dir: Subdirectory name for spectral images.
        mask_dir: Subdirectory name for mask files.
        transform: Spatial transform function that takes (image, mask) and returns (image, mask).
        num_classes: Number of segmentation classes.
    """

    def __init__(self, file_list, data_dir="data", image_dir="images",
                 mask_dir="masks", transform=None, num_classes=2):
        self.file_list = file_list
        self.image_root = os.path.join(data_dir, image_dir)
        self.mask_root = os.path.join(data_dir, mask_dir)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.file_list)

    def _load_mask(self, stem):
        """Load mask file, supporting both .npy and .png formats."""
        npy_path = os.path.join(self.mask_root, stem + ".npy")
        png_path = os.path.join(self.mask_root, stem + ".png")

        if os.path.exists(npy_path):
            mask = np.load(npy_path).astype(np.int64)
        elif os.path.exists(png_path):
            mask = np.array(Image.open(png_path)).astype(np.int64)
        else:
            raise FileNotFoundError(
                f"Mask not found for '{stem}'. Looked for:\n  {npy_path}\n  {png_path}"
            )
        return mask

    def __getitem__(self, idx):
        stem = self.file_list[idx]

        # Load spectral image: (H, W, 9) -> (9, H, W)
        image_path = os.path.join(self.image_root, stem + ".npy")
        image = np.load(image_path).astype(np.float32)  # (H, W, 9)
        image = image.transpose(2, 0, 1)  # (9, H, W)

        # Load mask: (H, W)
        mask = self._load_mask(stem)

        # Apply spatial transforms (both image and mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return image, mask, stem


def get_file_stems(data_dir, image_dir="images"):
    """Scan the image directory and return sorted list of file stems."""
    image_root = os.path.join(data_dir, image_dir)
    stems = []
    for fname in sorted(os.listdir(image_root)):
        if fname.endswith(".npy"):
            stems.append(fname[:-4])  # remove .npy extension
    return stems
