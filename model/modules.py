"""Pluggable feature enhancement modules.

Modules:
    - SEBlock: Channel attention via global average pooling (Squeeze-and-Excitation).
    - SpectralConv1D: 1D convolution along the spectral (band) dimension.
    - InputBandSE: Per-image dynamic band weighting via GAP+FC.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Performs: GAP -> FC(C, C//r) -> ReLU -> FC(C//r, C) -> Sigmoid -> channel-wise scaling.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck (default 16).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class SpectralConv1D(nn.Module):
    """1D convolution along the spectral (band) dimension.

    Learns local correlations between adjacent NIR bands (23nm spacing).
    Includes a residual connection.

    Args:
        num_channels: Number of channels to process.
        kernel_size: 1D convolution kernel size (default 3).
    """

    def __init__(self, num_channels=16, kernel_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat = x_flat.reshape(B * H * W, 1, C)
        x_flat = self.conv(x_flat)
        x_flat = x_flat.reshape(B, H * W, C).permute(0, 2, 1)
        x_out = x_flat.view(B, C, H, W)
        return self.relu(self.bn(x_out + x))


class InputBandSE(nn.Module):
    """Input-level Squeeze-and-Excitation for spectral band weighting.

    Per-image dynamic weights via GAP -> FC -> ReLU -> FC -> Sigmoid.
    """

    def __init__(self, num_bands=9, reduction=2):
        super().__init__()
        mid = max(num_bands // reduction, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_bands, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_bands, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

    def get_weights(self, x):
        """Return per-image band weights as (B, C) numpy array."""
        with torch.no_grad():
            w = self.pool(x)
            w = self.fc(w)
        return w.squeeze(-1).squeeze(-1).cpu().numpy()
