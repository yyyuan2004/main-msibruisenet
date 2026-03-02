"""Pluggable feature enhancement modules for ablation study.

Modules:
    - SE (Squeeze-and-Excitation): Channel attention via global average pooling.
    - SpectralConv1D: 1D convolution along the spectral (band) dimension.
    - ConvGLU: Convolutional Gated Linear Unit — a channel mixer (NOT attention).
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

    This module is inserted ONCE after the encoder S1 output, operating
    on the 9-band input (before the first block output which has 16 channels,
    actually it operates on the original input or right after the first stage).

    Note: This module is designed for the input space (9 bands). When placed
    after S1, the feature has 16 channels — the conv operates on the channel
    dimension treating them as a 1D sequence.

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
        # Reshape: apply 1D conv along channel dimension for each spatial position
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        x_flat = x_flat.reshape(B * H * W, 1, C)        # (B*H*W, 1, C)
        x_flat = self.conv(x_flat)                        # (B*H*W, 1, C)
        x_flat = x_flat.reshape(B, H * W, C).permute(0, 2, 1)  # (B, C, H*W)
        x_out = x_flat.view(B, C, H, W)
        return self.relu(self.bn(x_out + x))  # Residual connection


class ConvGLU(nn.Module):
    """Convolutional Gated Linear Unit — a channel mixer from TransNeXt (CVPR 2024).

    This is NOT an attention module. It is a feed-forward network variant that
    replaces Conv-BN-ReLU blocks in the decoder.

    Structure:
        Input x (B, C_in, H, W)
        ├─ Value branch: Linear(C_in -> hidden) -> value
        ├─ Gate branch:  Linear(C_in -> hidden) -> DWConv3x3 -> GELU -> gate
        └─ Output: (value * gate) -> Linear(hidden -> C_out)

    Args:
        in_channels: Input channel count (after concat in decoder).
        out_channels: Output channel count for this decoder stage.
        expansion_ratio: Expansion ratio for hidden dimension (default 4).
    """

    def __init__(self, in_channels, out_channels, expansion_ratio=4):
        super().__init__()
        hidden = int(in_channels * expansion_ratio * 2 / 3)
        hidden = max(hidden, 8)  # ensure minimum hidden size

        # Value branch
        self.fc_value = nn.Conv2d(in_channels, hidden, 1)

        # Gate branch
        self.fc_gate = nn.Conv2d(in_channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()

        # Output projection
        self.fc_out = nn.Conv2d(hidden, out_channels, 1)

        # Batch normalization on output
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        value = self.fc_value(x)
        gate = self.act(self.dwconv(self.fc_gate(x)))
        out = self.fc_out(value * gate)
        return self.bn(out)
