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


def unsharp_mask(image, sigma=1.0, alpha=1.5):
    """Apply Unsharp Masking per band: sharpened = image + alpha * (image - blurred).

    Args:
        image: (C, H, W) float32 array.
        sigma: Gaussian blur sigma.
        alpha: Sharpening strength.

    Returns:
        Sharpened image with same shape.
    """
    C, H, W = image.shape
    out = np.empty_like(image)
    for c in range(C):
        blurred = cv2.GaussianBlur(image[c], (0, 0), sigma)
        out[c] = image[c] + alpha * (image[c] - blurred)
    return out


class MSIDataset(Dataset):
    """Dataset for loading 9-channel .npy spectral images and corresponding masks.

    Returns 5 values: (image, mask, image_raw, apple_mask, stem).
    - image: preprocessed (possibly band-selected, sharpened, LCN, PCA) and augmented tensor.
    - mask: ground truth defect mask tensor.
    - image_raw: original full-band image tensor (before any preprocessing), for visualization.
    - apple_mask: binary apple region mask (1=apple, 0=background), shape (H, W).
    - stem: file name stem string.
    """

    def __init__(self, file_list, data_dir="data", image_dir="images",
                 mask_dir="masks", transform=None, num_classes=2,
                 band_indices=None,
                 use_sharpen=False, sharpen_sigma=1.0, sharpen_alpha=1.5,
                 use_lcn=False, lcn_kernel_size=31, lcn_eps=1e-6,
                 use_pca=False, pca_matrix_path="",
                 apple_mask_threshold=0.05,
                 lazy_load=False):
        self.file_list = file_list
        self.image_root = os.path.join(data_dir, image_dir)
        self.mask_root = os.path.join(data_dir, mask_dir)
        self.whole_root = os.path.join(data_dir, "whole")
        self.transform = transform
        self.num_classes = num_classes

        # Band selection
        self.band_indices = band_indices

        # Sharpen config
        self.use_sharpen = use_sharpen
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_alpha = sharpen_alpha

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

        # Apple mask: detect availability of whole/ directory at init time
        self.apple_mask_threshold = apple_mask_threshold
        self._whole_dir_exists = os.path.isdir(self.whole_root)

        # Lazy loading: use mmap + only read selected bands (for large HSI files)
        self.lazy_load = lazy_load

    def __len__(self):
        return len(self.file_list)

    @property
    def apple_mask_source(self):
        """Return string describing the apple mask source for logging."""
        if self._whole_dir_exists:
            return "whole_mask_npy"
        return f"threshold_fallback(thr={self.apple_mask_threshold})"

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

    def _load_apple_mask(self, stem, image):
        """Load whole-apple mask from data_dir/whole/, or auto-estimate.

        Source priority:
            1. whole/ directory: load <stem>.npy, binarize (>0 → 1.0)
            2. Threshold fallback: mean_reflectance > apple_mask_threshold

        Use ``self.apple_mask_source`` to check which source is active.

        Args:
            stem: File name stem.
            image: (C, H, W) raw spectral image (before band selection).

        Returns:
            apple_mask: (H, W) float32, 1.0=apple / 0.0=background.
        """
        if self._whole_dir_exists:
            whole_path = os.path.join(self.whole_root, stem + ".npy")
            if os.path.exists(whole_path):
                m = np.load(whole_path).astype(np.float32)
                return (m > 0).astype(np.float32)
        # Fallback: auto-estimate from mean reflectance
        return (image.mean(axis=0) > self.apple_mask_threshold).astype(np.float32)

    def __getitem__(self, idx):
        stem = self.file_list[idx]

        image_path = os.path.join(self.image_root, stem + ".npy")

        if self.lazy_load and self.band_indices is not None:
            # mmap-based partial read: only materialize the selected bands.
            # Avoids reading the full 200+ channel cube for each sample.
            mm = np.load(image_path, mmap_mode="r")  # (H, W, C_total)
            # Fancy indexing on the last axis triggers a copy of just those bands.
            image = np.array(mm[..., self.band_indices], dtype=np.float32)
            image = image.transpose(2, 0, 1)  # (C_sel, H, W)
            # For band search, image_raw is only used as a placeholder in the tuple.
            image_raw = image
            apple_mask = self._load_apple_mask(stem, image)  # uses selected-band mean
        else:
            # Load spectral image: (H, W, C) -> (C, H, W)
            image = np.load(image_path).astype(np.float32)
            image = image.transpose(2, 0, 1)

            # Keep a copy of the raw full-band image for visualization
            image_raw = image.copy()

            # Load apple mask (before band selection, needs full-band for fallback)
            apple_mask = self._load_apple_mask(stem, image)  # (H, W)

            # Band selection
            if self.band_indices is not None:
                image = image[self.band_indices]

        # Optional Unsharp Masking
        if self.use_sharpen:
            image = unsharp_mask(image, self.sharpen_sigma, self.sharpen_alpha)

        # Optional LCN preprocessing (before augmentation)
        if self.use_lcn:
            image = apply_lcn(image, self.lcn_kernel_size, self.lcn_eps)

        # Optional PCA dimensionality reduction: (C, H, W) -> (n_components, H, W)
        if self.use_pca and self.pca_components is not None:
            C, H, W = image.shape
            centered = image - self.pca_mean[:, None, None]
            image = self.pca_components @ centered.reshape(C, -1)  # (n_comp, H*W)
            image = image.reshape(self.pca_components.shape[0], H, W)

        # Load mask: (H, W)
        mask = self._load_mask(stem)

        # Apply spatial transforms (both image and mask; apple_mask syncs with mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        image_raw = torch.from_numpy(np.ascontiguousarray(image_raw)).float()
        apple_mask = torch.from_numpy(np.ascontiguousarray(apple_mask)).float()

        return image, mask, image_raw, apple_mask, stem


def get_dataset_kwargs(cfg):
    """Extract band_indices / sharpen / LCN / PCA dataset kwargs from config dict."""
    data_cfg = cfg.get("data", {})
    return {
        "band_indices": data_cfg.get("band_indices", None),
        "use_sharpen": data_cfg.get("use_sharpen", False),
        "sharpen_sigma": data_cfg.get("sharpen_sigma", 1.0),
        "sharpen_alpha": data_cfg.get("sharpen_alpha", 1.5),
        "use_lcn": data_cfg.get("use_lcn", False),
        "lcn_kernel_size": data_cfg.get("lcn_kernel_size", 31),
        "lcn_eps": data_cfg.get("lcn_eps", 1e-6),
        "use_pca": data_cfg.get("use_pca", False),
        "pca_matrix_path": data_cfg.get("pca_matrix_path", ""),
        "apple_mask_threshold": data_cfg.get("apple_mask_threshold", 0.05),
        "lazy_load": data_cfg.get("lazy_load", False),
    }


def get_file_stems(data_dir, image_dir="images"):
    """Scan the image directory and return sorted list of file stems."""
    image_root = os.path.join(data_dir, image_dir)
    stems = []
    for fname in sorted(os.listdir(image_root)):
        if fname.endswith(".npy"):
            stems.append(fname[:-4])  # remove .npy extension
    return stems
