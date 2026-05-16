"""Spatial augmentations for MSI segmentation.

All transforms operate on numpy arrays and apply identical spatial
transformations to the spectral image (C, H, W), defect mask (H, W), and
optionally the apple foreground mask (H, W). No color/brightness augmentation is
applied to masks; spectral intensity augmentations only modify the image.
"""

import numpy as np
import cv2


def _pack_return(image, mask, apple_mask):
    """Return a 2-tuple or 3-tuple depending on whether apple_mask is used."""
    if apple_mask is None:
        return image, mask
    return image, mask, apple_mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, apple_mask=None):
        if apple_mask is None:
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask

        for t in self.transforms:
            image, mask, apple_mask = t(image, mask, apple_mask)
        return image, mask, apple_mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() < self.p:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
            if apple_mask is not None:
                apple_mask = apple_mask[:, ::-1].copy()
        return _pack_return(image, mask, apple_mask)


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() < self.p:
            image = image[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
            if apple_mask is not None:
                apple_mask = apple_mask[::-1, :].copy()
        return _pack_return(image, mask, apple_mask)


class RandomRotation90:
    """Randomly rotate by 0, 90, 180, or 270 degrees."""

    def __call__(self, image, mask, apple_mask=None):
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
            if apple_mask is not None:
                apple_mask = np.rot90(apple_mask, k, axes=(0, 1)).copy()
        return _pack_return(image, mask, apple_mask)


class RandomCrop:
    """Random crop to target size. If image is smaller, no crop is applied."""

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, image, mask, apple_mask=None):
        _, h, w = image.shape
        if h <= self.crop_h or w <= self.crop_w:
            return _pack_return(image, mask, apple_mask)

        top = np.random.randint(0, h - self.crop_h + 1)
        left = np.random.randint(0, w - self.crop_w + 1)
        image = image[:, top:top + self.crop_h, left:left + self.crop_w].copy()
        mask = mask[top:top + self.crop_h, left:left + self.crop_w].copy()
        if apple_mask is not None:
            apple_mask = apple_mask[top:top + self.crop_h, left:left + self.crop_w].copy()
        return _pack_return(image, mask, apple_mask)


class ElasticTransform:
    """Simple elastic deformation using random displacement fields."""

    def __init__(self, alpha=50, sigma=7, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() >= self.p:
            return _pack_return(image, mask, apple_mask)

        _, h, w = image.shape
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), self.sigma
        ) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        warped_image = np.stack([
            cv2.remap(image[c], map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)
            for c in range(image.shape[0])
        ], axis=0)

        warped_mask = cv2.remap(
            mask.astype(np.float32), map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101
        ).astype(mask.dtype)

        if apple_mask is not None:
            apple_mask = cv2.remap(
                apple_mask.astype(np.float32), map_x, map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101
            ).astype(apple_mask.dtype)

        return _pack_return(warped_image, warped_mask, apple_mask)


class Cutout:
    """Random rectangular erasing for the spectral image only."""

    def __init__(self, num_holes=1, max_h_frac=0.3, max_w_frac=0.3, p=0.5):
        self.num_holes = num_holes
        self.max_h_frac = max_h_frac
        self.max_w_frac = max_w_frac
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() >= self.p:
            return _pack_return(image, mask, apple_mask)

        image = image.copy()
        _, h, w = image.shape

        for _ in range(self.num_holes):
            cut_h = int(h * np.random.uniform(0.1, self.max_h_frac))
            cut_w = int(w * np.random.uniform(0.1, self.max_w_frac))

            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)

            y1 = max(0, cy - cut_h // 2)
            y2 = min(h, cy + cut_h // 2)
            x1 = max(0, cx - cut_w // 2)
            x2 = min(w, cx + cut_w // 2)

            image[:, y1:y2, x1:x2] = 0.0

        return _pack_return(image, mask, apple_mask)


class GaussianBlur:
    """Apply Gaussian blur to the spectral image only."""

    def __init__(self, kernel_range=(3, 7), sigma_range=(0.5, 2.0), p=0.5):
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() >= self.p:
            return _pack_return(image, mask, apple_mask)

        k = np.random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
        sigma = np.random.uniform(*self.sigma_range)

        blurred = np.stack([
            cv2.GaussianBlur(image[c], (k, k), sigma)
            for c in range(image.shape[0])
        ], axis=0)

        return _pack_return(blurred, mask, apple_mask)


class IntensityJitter:
    """Random per-channel intensity scaling for the spectral image only."""

    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() >= self.p:
            return _pack_return(image, mask, apple_mask)

        c = image.shape[0]
        scales = np.random.uniform(
            self.scale_range[0], self.scale_range[1], size=(c, 1, 1)
        ).astype(np.float32)
        image = image * scales
        return _pack_return(image, mask, apple_mask)


class GaussianNoise:
    """Add small Gaussian noise to the spectral image only."""

    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, image, mask, apple_mask=None):
        if np.random.random() < self.p:
            noise = np.random.randn(*image.shape).astype(np.float32) * self.std
            image = image + noise
        return _pack_return(image, mask, apple_mask)


class Resize:
    """Resize image, defect mask, and optionally apple foreground mask."""

    def __init__(self, size):
        if isinstance(size, int):
            self.h, self.w = size, size
        else:
            self.h, self.w = size

    def __call__(self, image, mask, apple_mask=None):
        _, h, w = image.shape
        if h == self.h and w == self.w:
            return _pack_return(image, mask, apple_mask)

        resized_image = np.stack([
            cv2.resize(image[c], (self.w, self.h),
                       interpolation=cv2.INTER_LINEAR)
            for c in range(image.shape[0])
        ], axis=0)

        resized_mask = cv2.resize(
            mask.astype(np.float32), (self.w, self.h),
            interpolation=cv2.INTER_NEAREST
        ).astype(mask.dtype)

        if apple_mask is not None:
            apple_mask = cv2.resize(
                apple_mask.astype(np.float32), (self.w, self.h),
                interpolation=cv2.INTER_NEAREST
            ).astype(apple_mask.dtype)

        return _pack_return(resized_image, resized_mask, apple_mask)


class CenterCrop:
    """Center crop to target size."""

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, image, mask, apple_mask=None):
        _, h, w = image.shape
        if h <= self.crop_h and w <= self.crop_w:
            return _pack_return(image, mask, apple_mask)

        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        image = image[:, top:top + self.crop_h, left:left + self.crop_w].copy()
        mask = mask[top:top + self.crop_h, left:left + self.crop_w].copy()
        if apple_mask is not None:
            apple_mask = apple_mask[top:top + self.crop_h, left:left + self.crop_w].copy()
        return _pack_return(image, mask, apple_mask)


def get_train_transforms(cfg):
    """Build training augmentation pipeline from config."""
    aug_cfg = cfg["train"]["augment"]
    crop_size = cfg["data"].get("crop_size", cfg["data"]["image_size"])

    transforms = []

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(RandomHorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", True):
        transforms.append(RandomVerticalFlip(p=0.5))

    if aug_cfg.get("random_rotation", True):
        transforms.append(RandomRotation90())

    if aug_cfg.get("elastic_transform", False):
        transforms.append(ElasticTransform(p=0.3))

    if aug_cfg.get("gaussian_noise", False):
        std = aug_cfg.get("gaussian_noise_std", 0.01)
        transforms.append(GaussianNoise(std=std, p=0.3))

    if crop_size < cfg["data"]["image_size"]:
        transforms.append(RandomCrop(crop_size))

    return Compose(transforms)


def get_val_transforms(cfg):
    """Build validation/test transform pipeline without random augmentation."""
    crop_size = cfg["data"].get("crop_size", cfg["data"]["image_size"])
    transforms = []
    if crop_size < cfg["data"]["image_size"]:
        transforms.append(CenterCrop(crop_size))
    return Compose(transforms)
