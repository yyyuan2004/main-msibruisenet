"""MSI Dataset for 9-channel near-infrared multispectral apple defect segmentation."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


def apply_lcn(image, kernel_size=31, eps=1e-6):
    """Apply Local Contrast Normalization per band.

    For each band: x_norm = (x - local_mean) / (local_std + eps)
    where local_mean/std are computed over a spatial window.

    Args:
        image: (C, H, W) float32 array.
        kernel_size: Window size for local statistics.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized image with same shape.
    """
    C, H, W = image.shape
    out = np.empty_like(image)
    ksize = (kernel_size, kernel_size)
    for c in range(C):
        band = image[c]
        local_mean = cv2.blur(band, ksize)
        local_sq_mean = cv2.blur(band ** 2, ksize)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0) + eps)
        out[c] = (band - local_mean) / local_std
    return out


class MSIDataset(Dataset):
    """Dataset for loading 9-channel .npy spectral images and corresponding masks.

    Args:
        file_list: List of file stems (without extension).
        data_dir: Root data directory.
        image_dir: Subdirectory name for spectral images.
        mask_dir: Subdirectory name for mask files.
        transform: Spatial transform function that takes (image, mask) and returns (image, mask).
        num_classes: Number of segmentation classes.
        use_lcn: Whether to apply Local Contrast Normalization.
        lcn_kernel_size: LCN window size.
        lcn_eps: LCN epsilon for numerical stability.
        use_pca: Whether to apply PCA dimensionality reduction.
        pca_matrix_path: Path to precomputed pca_matrix.npz.
    """

    def __init__(self, file_list, data_dir="data", image_dir="images",
                 mask_dir="masks", transform=None, num_classes=2,
                 use_lcn=False, lcn_kernel_size=31, lcn_eps=1e-6,
                 use_pca=False, pca_matrix_path=""):
        self.file_list = file_list
        self.image_root = os.path.join(data_dir, image_dir)
        self.mask_root = os.path.join(data_dir, mask_dir)
        self.transform = transform
        self.num_classes = num_classes

        # LCN config
        self.use_lcn = use_lcn
        self.lcn_kernel_size = lcn_kernel_size
        self.lcn_eps = lcn_eps

        # PCA config
        self.use_pca = use_pca
        self.pca_components = None
        self.pca_mean = None
        if use_pca and pca_matrix_path:
            data = np.load(pca_matrix_path)
            self.pca_components = data["components"]  # (n_components, 9)
            self.pca_mean = data["mean"]              # (9,)

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

        # Optional LCN preprocessing (before augmentation)
        if self.use_lcn:
            image = apply_lcn(image, self.lcn_kernel_size, self.lcn_eps)

        # Optional PCA dimensionality reduction: (9, H, W) -> (n_components, H, W)
        if self.use_pca and self.pca_components is not None:
            C, H, W = image.shape
            centered = image - self.pca_mean[:, None, None]
            image = self.pca_components @ centered.reshape(C, -1)  # (n_comp, H*W)
            image = image.reshape(self.pca_components.shape[0], H, W)

        # Load mask: (H, W)
        mask = self._load_mask(stem)

        # Apply spatial transforms (both image and mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return image, mask, stem


def get_dataset_kwargs(cfg):
    """Extract LCN/PCA dataset kwargs from config dict."""
    data_cfg = cfg.get("data", {})
    return {
        "use_lcn": data_cfg.get("use_lcn", False),
        "lcn_kernel_size": data_cfg.get("lcn_kernel_size", 31),
        "lcn_eps": data_cfg.get("lcn_eps", 1e-6),
        "use_pca": data_cfg.get("use_pca", False),
        "pca_matrix_path": data_cfg.get("pca_matrix_path", ""),
    }


def get_file_stems(data_dir, image_dir="images"):
    """Scan the image directory and return sorted list of file stems."""
    image_root = os.path.join(data_dir, image_dir)
    stems = []
    for fname in sorted(os.listdir(image_root)):
        if fname.endswith(".npy"):
            stems.append(fname[:-4])  # remove .npy extension
    return stems
