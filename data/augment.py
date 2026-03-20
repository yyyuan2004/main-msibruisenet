"""Spatial augmentations for MSI segmentation.

All transforms operate on numpy arrays and apply identical spatial
transformations to both the image (C, H, W) and the mask (H, W).
No color/brightness augmentation is applied — spectral reflectance
values have physical meaning and must not be altered.
"""

import numpy as np
import cv2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = image[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        return image, mask


class RandomRotation90:
    """Randomly rotate by 0, 90, 180, or 270 degrees."""

    def __call__(self, image, mask):
        k = np.random.randint(0, 4)
        if k > 0:
            # image: (C, H, W), rotate last two dims
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return image, mask


class RandomCrop:
    """Random crop to target size. If image is smaller, no crop is applied."""

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, image, mask):
        _, h, w = image.shape
        if h <= self.crop_h and w <= self.crop_w:
            return image, mask

        top = np.random.randint(0, max(h - self.crop_h, 1))
        left = np.random.randint(0, max(w - self.crop_w, 1))
        image = image[:, top:top + self.crop_h, left:left + self.crop_w].copy()
        mask = mask[top:top + self.crop_h, left:left + self.crop_w].copy()
        return image, mask


class ElasticTransform:
    """Simple elastic deformation using random displacement fields."""

    def __init__(self, alpha=50, sigma=7, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() >= self.p:
            return image, mask

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

        # Apply to each channel of image
        warped_image = np.stack([
            cv2.remap(image[c], map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)
            for c in range(image.shape[0])
        ], axis=0)

        # Apply to mask with nearest interpolation
        warped_mask = cv2.remap(
            mask.astype(np.float32), map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101
        ).astype(mask.dtype)

        return warped_image, warped_mask


class Cutout:
    """Random rectangular erasing (Cutout) for strong augmentation.

    Randomly erases a rectangular region of the image, filling it with zeros.
    The mask is NOT modified (label information is preserved).

    Args:
        num_holes: Number of rectangular patches to erase.
        max_h_size: Maximum height of erased patch (fraction of image H).
        max_w_size: Maximum width of erased patch (fraction of image W).
        p: Probability of applying cutout.
    """

    def __init__(self, num_holes=1, max_h_frac=0.3, max_w_frac=0.3, p=0.5):
        self.num_holes = num_holes
        self.max_h_frac = max_h_frac
        self.max_w_frac = max_w_frac
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() >= self.p:
            return image, mask

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

        return image, mask


class GaussianBlur:
    """Apply Gaussian blur to the spectral image (strong augmentation).

    Args:
        kernel_range: Range of kernel sizes (must be odd numbers).
        sigma_range: Range of sigma values.
        p: Probability of applying blur.
    """

    def __init__(self, kernel_range=(3, 7), sigma_range=(0.5, 2.0), p=0.5):
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() >= self.p:
            return image, mask

        # Pick random odd kernel size
        k = np.random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
        sigma = np.random.uniform(*self.sigma_range)

        blurred = np.stack([
            cv2.GaussianBlur(image[c], (k, k), sigma)
            for c in range(image.shape[0])
        ], axis=0)

        return blurred, mask


class IntensityJitter:
    """Random per-channel intensity scaling (strong augmentation for MSI).

    Unlike RGB brightness/contrast jitter, this applies independent per-band
    scaling to simulate spectral response variations while keeping physical
    meaning roughly intact.

    Args:
        scale_range: (min_scale, max_scale) for per-channel multiplicative jitter.
        p: Probability of applying.
    """

    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() >= self.p:
            return image, mask

        c = image.shape[0]
        scales = np.random.uniform(
            self.scale_range[0], self.scale_range[1], size=(c, 1, 1)
        ).astype(np.float32)
        image = image * scales
        return image, mask


class GaussianNoise:
    """Add small Gaussian noise to the spectral image."""

    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            noise = np.random.randn(*image.shape).astype(np.float32) * self.std
            image = image + noise
        return image, mask


class Resize:
    """Resize image and mask to target size."""

    def __init__(self, size):
        if isinstance(size, int):
            self.h, self.w = size, size
        else:
            self.h, self.w = size

    def __call__(self, image, mask):
        _, h, w = image.shape
        if h == self.h and w == self.w:
            return image, mask

        # image: (C, H, W) -> resize each channel
        resized_image = np.stack([
            cv2.resize(image[c], (self.w, self.h),
                       interpolation=cv2.INTER_LINEAR)
            for c in range(image.shape[0])
        ], axis=0)

        resized_mask = cv2.resize(
            mask.astype(np.float32), (self.w, self.h),
            interpolation=cv2.INTER_NEAREST
        ).astype(mask.dtype)

        return resized_image, resized_mask


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
    """Build validation/test transform pipeline (no augmentation)."""
    crop_size = cfg["data"].get("crop_size", cfg["data"]["image_size"])
    transforms = []
    # For val/test, center-crop if crop_size < image_size
    # Otherwise just pass through
    if crop_size < cfg["data"]["image_size"]:
        transforms.append(CenterCrop(crop_size))
    return Compose(transforms)


def get_weak_transforms(cfg):
    """Build weak augmentation pipeline for Mean Teacher (Teacher branch).

    Weak augmentation: only spatial flips + rotation (no destructive transforms).
    Used for generating high-quality pseudo-labels from the Teacher model.
    """
    crop_size = cfg["data"].get("crop_size", cfg["data"]["image_size"])

    transforms = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
    ]

    if crop_size < cfg["data"]["image_size"]:
        transforms.append(RandomCrop(crop_size))

    return Compose(transforms)


def get_strong_transforms(cfg):
    """Build strong augmentation pipeline for Mean Teacher (Student branch).

    Strong augmentation: spatial flips/rotation + destructive transforms
    (cutout, blur, intensity jitter, elastic deformation, noise).
    Forces the Student to learn robust features despite heavy perturbation.
    """
    crop_size = cfg["data"].get("crop_size", cfg["data"]["image_size"])

    transforms = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        # Destructive transforms
        Cutout(num_holes=2, max_h_frac=0.3, max_w_frac=0.3, p=0.5),
        GaussianBlur(kernel_range=(3, 7), sigma_range=(0.5, 2.0), p=0.3),
        IntensityJitter(scale_range=(0.8, 1.2), p=0.5),
        ElasticTransform(alpha=50, sigma=7, p=0.3),
        GaussianNoise(std=0.02, p=0.3),
    ]

    if crop_size < cfg["data"]["image_size"]:
        transforms.append(RandomCrop(crop_size))

    return Compose(transforms)


class CenterCrop:
    """Center crop to target size."""

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, image, mask):
        _, h, w = image.shape
        if h <= self.crop_h and w <= self.crop_w:
            return image, mask

        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        image = image[:, top:top + self.crop_h, left:left + self.crop_w].copy()
        mask = mask[top:top + self.crop_h, left:left + self.crop_w].copy()
        return image, mask
