"""SDA v2 spectral anomaly feature extraction.

Computes interpretable per-pixel anomaly maps from raw multispectral input:
    - spectral_std:   Per-pixel standard deviation across bands
    - sam:            Spectral Angle Mapper to healthy reference
    - snv_l2:         SNV-transformed L2 distance to healthy reference
    - mahalanobis:    Mahalanobis distance to apple foreground distribution
    - raw_l2:         Raw L2 distance to healthy reference (optional)

All features are computed within the apple foreground mask and normalized
to [0, 1] within the foreground region.
"""

import math
import torch
import torch.nn.functional as F


def _make_gaussian_kernel(sigma, device, dtype):
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    x = torch.arange(kernel_size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size), radius


def gaussian_blur_2d(x, sigma):
    """Apply Gaussian blur to (B, C, H, W) tensor using depthwise conv."""
    if sigma <= 0.5:
        return x
    kernel, pad = _make_gaussian_kernel(sigma, x.device, x.dtype)
    B, C, H, W = x.shape
    kernel = kernel.expand(C, -1, -1, -1)
    return F.conv2d(x, kernel, padding=pad, groups=C)


def compute_snv(x):
    """Standard Normal Variate: per-pixel normalization across bands.
    Args: x (B, C, H, W)
    Returns: (B, C, H, W) SNV-transformed tensor.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std


def compute_healthy_reference(x, apple_mask):
    """Per-image healthy reference = mean spectrum over apple foreground.

    Since defects are a small fraction of the apple area, the foreground
    mean is a good approximation of the healthy reference.

    Args:
        x: (B, C, H, W) raw spectral input.
        apple_mask: (B, 1, H, W) binary mask (1=apple, 0=bg).
    Returns:
        (B, C, 1, 1) reference spectrum per image.
    """
    masked = x * apple_mask
    count = apple_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    return masked.sum(dim=(2, 3), keepdim=True) / count


def compute_spectral_std_map(x):
    """Per-pixel std across spectral bands.
    Args: x (B, C, H, W)
    Returns: (B, 1, H, W)
    """
    return x.std(dim=1, keepdim=True)


def compute_sam_to_reference(x, ref):
    """Spectral Angle Mapper: angle between each pixel and reference spectrum.
    Args:
        x: (B, C, H, W)
        ref: (B, C, 1, 1) reference spectrum.
    Returns: (B, 1, H, W) angle in radians [0, pi].
    """
    x_norm = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-8)
    ref_norm = torch.norm(ref, dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = (x * ref).sum(dim=1, keepdim=True) / (x_norm * ref_norm)
    return torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))


def compute_snv_l2_to_reference(x, ref):
    """L2 distance between SNV(pixel) and SNV(reference).
    Args:
        x: (B, C, H, W)
        ref: (B, C, 1, 1)
    Returns: (B, 1, H, W)
    """
    x_snv = compute_snv(x)
    ref_snv = compute_snv(ref)
    return torch.norm(x_snv - ref_snv, dim=1, keepdim=True)


def compute_raw_l2_to_reference(x, ref):
    """Raw L2 distance between each pixel and reference spectrum.
    Args:
        x: (B, C, H, W)
        ref: (B, C, 1, 1)
    Returns: (B, 1, H, W)
    """
    return torch.norm(x - ref, dim=1, keepdim=True)


def compute_mahalanobis(x, apple_mask):
    """Mahalanobis distance of each pixel to the apple foreground distribution.

    Per-image: estimate mean + covariance from foreground pixels, then compute
    distance for every pixel. With C=4 bands, the C×C covariance inversion
    is negligible.

    Args:
        x: (B, C, H, W)
        apple_mask: (B, 1, H, W)
    Returns: (B, 1, H, W)
    """
    B, C, H, W = x.shape
    result = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

    for b in range(B):
        mask_b = apple_mask[b, 0].bool()
        fg = x[b, :, mask_b]  # (C, N_fg)
        if fg.shape[1] < C + 1:
            continue

        mean = fg.mean(dim=1, keepdim=True)  # (C, 1)
        centered = fg - mean
        cov = (centered @ centered.T) / (centered.shape[1] - 1)
        cov = cov + 1e-4 * torch.eye(C, device=x.device, dtype=x.dtype)

        try:
            cov_inv = torch.linalg.inv(cov)
        except Exception:
            cov_inv = torch.eye(C, device=x.device, dtype=x.dtype)

        x_flat = x[b].reshape(C, -1)  # (C, H*W)
        diff = x_flat - mean
        mahal_sq = (diff * (cov_inv @ diff)).sum(dim=0)  # (H*W,)
        result[b, 0] = mahal_sq.clamp(min=0).sqrt().reshape(H, W)

    return result


def compute_texture_energy(x, sigma_t=5.0):
    """Local texture energy: squared Laplacian of mean-band image, smoothed.

    High values indicate natural fruit texture / high-frequency noise that
    should be suppressed in the anomaly map.

    Args:
        x: (B, C, H, W) raw input.
        sigma_t: Gaussian smoothing sigma for texture map.
    Returns: (B, 1, H, W)
    """
    x_mean = x.mean(dim=1, keepdim=True)
    laplacian = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    T = F.conv2d(x_mean, laplacian, padding=1)
    T = T.pow(2)
    if sigma_t > 0.5:
        T = gaussian_blur_2d(T, sigma_t)
    return T


def normalize_within_mask(feat, apple_mask):
    """Normalize feature map to [0, 1] within apple foreground per image.

    Background pixels are set to zero.

    Args:
        feat: (B, 1, H, W)
        apple_mask: (B, 1, H, W) binary.
    Returns: (B, 1, H, W) normalized.
    """
    B = feat.shape[0]
    out = torch.zeros_like(feat)
    for b in range(B):
        mask_b = apple_mask[b, 0].bool()
        vals = feat[b, 0, mask_b]
        if vals.numel() == 0:
            continue
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-8:
            out[b, 0, mask_b] = 0.0
        else:
            out[b, 0] = (feat[b, 0] - vmin) / (vmax - vmin)
            out[b, 0] = out[b, 0] * mask_b.float()
    return out


# ---- Registry for feature computation ----

FEATURE_REGISTRY = {
    "spectral_std": lambda x, ref, mask: compute_spectral_std_map(x),
    "sam": lambda x, ref, mask: compute_sam_to_reference(x, ref),
    "snv_l2": lambda x, ref, mask: compute_snv_l2_to_reference(x, ref),
    "mahalanobis": lambda x, ref, mask: compute_mahalanobis(x, mask),
    "raw_l2": lambda x, ref, mask: compute_raw_l2_to_reference(x, ref),
}


def compute_sda_features(x, apple_mask, feature_names):
    """Compute a stack of anomaly feature maps.

    Args:
        x: (B, C, H, W) raw spectral input.
        apple_mask: (B, 1, H, W) binary mask.
        feature_names: List of feature names from FEATURE_REGISTRY.
    Returns:
        (B, len(feature_names), H, W) normalized feature maps.
    """
    ref = compute_healthy_reference(x, apple_mask)
    maps = []
    for name in feature_names:
        fn = FEATURE_REGISTRY[name]
        feat = fn(x, ref, apple_mask)
        feat = normalize_within_mask(feat, apple_mask)
        maps.append(feat)
    return torch.cat(maps, dim=1)
