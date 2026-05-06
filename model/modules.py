"""Pluggable feature enhancement modules for ablation study.

Modules:
    - SE (Squeeze-and-Excitation): Channel attention via global average pooling.
    - CBAM (Convolutional Block Attention Module): Channel + spatial attention.
    - ASPP (Atrous Spatial Pyramid Pooling): Multi-scale context aggregation.
    - SpectralConv1D: 1D convolution along the spectral (band) dimension.
    - InputBandSE: Per-image dynamic band weighting via GAP+FC.
    - BandAttention: Static per-band learnable weighting.
    - GlobalSaliencyBranch: Low-resolution spatial attention for bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM.

    Uses both average-pooled and max-pooled features through a shared MLP.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(spatial))


class CBAMBlock(nn.Module):
    """CBAM: Channel Attention -> Spatial Attention (sequential)."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ASPPConv(nn.Module):
    """Single ASPP branch: dilated Conv3x3 -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """ASPP global pooling branch."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv(self.gap(x))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.relu(self.bn(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Aggregates 5 parallel branches (1x1 + 3 dilated 3x3 + global pool) and
    projects back to ``out_channels``.
    """

    def __init__(self, in_channels=320, out_channels=256,
                 atrous_rates=(6, 12, 18), dropout=0.5):
        super().__init__()

        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.branches = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)
        return self.project(x)


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


class BandAttention(nn.Module):
    """Learnable per-band weighting at input level.

    Each band has an independent learnable scalar (sigmoid-gated).
    Initialized to zeros so all bands start at 0.5.
    """

    def __init__(self, num_bands=9):
        super().__init__()
        self.band_logits = nn.Parameter(torch.zeros(1, num_bands, 1, 1))

    def forward(self, x):
        weights = torch.sigmoid(self.band_logits)
        return x * weights

    def get_weights(self):
        return torch.sigmoid(self.band_logits).detach().cpu().squeeze().numpy()


class GlobalSaliencyBranch(nn.Module):
    """Low-resolution branch producing a spatial attention map.

    Downsamples input -> 3 conv layers -> 1x1 attention head ->
    interpolate to bottleneck size.
    """

    def __init__(self, in_channels=9, downsample_factor=4):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.attention_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, bottleneck_size):
        x_lr = F.interpolate(
            x, scale_factor=1.0 / self.downsample_factor,
            mode='bilinear', align_corners=False,
        )
        feat = self.conv(x_lr)
        attn = self.attention_head(feat)
        return F.interpolate(
            attn, size=bottleneck_size,
            mode='bilinear', align_corners=False,
        )

    def get_attention_map(self, x, bottleneck_size):
        with torch.no_grad():
            return self.forward(x, bottleneck_size).cpu().numpy()
