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


###############################################################################
# [NEW] CBAM — Convolutional Block Attention Module
# Placed at every skip connection level in the decoder (similar position to SE).
# Unlike SE which only does channel attention, CBAM adds spatial attention on top.
#   1) Channel Attention: GAP + GMP -> shared MLP -> sigmoid -> channel-wise scale
#   2) Spatial Attention: AvgPool(ch) + MaxPool(ch) -> Conv7x7 -> sigmoid -> spatial scale
###############################################################################

class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM.

    Uses both average-pooled and max-pooled features through a shared MLP.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck.
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
    """Spatial attention sub-module of CBAM.

    Applies Conv7x7 on channel-wise avg/max pooled feature maps.

    Args:
        kernel_size: Convolution kernel size (default 7).
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)        # (B, 1, H, W)
        spatial = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return x * torch.sigmoid(self.conv(spatial))


class CBAMBlock(nn.Module):
    """CBAM: Channel Attention -> Spatial Attention (sequential).

    Placed at skip connections in the decoder, same position as SE.
    Compared to SE, CBAM adds spatial attention which helps localize
    defect regions more precisely.

    Args:
        channels: Number of input/output channels.
        reduction: Channel attention reduction ratio (default 16).
        spatial_kernel: Spatial attention conv kernel size (default 7).
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)  # Step 1: channel attention
        x = self.spatial_attn(x)  # Step 2: spatial attention
        return x


###############################################################################
# [NEW] ASPP — Atrous Spatial Pyramid Pooling
# Placed between encoder bottleneck (S5, 320ch, 16x16) and decoder input.
# Uses parallel dilated convolutions at rates {1, 6, 12, 18} + global pooling
# to capture multi-scale contextual information.
# Particularly useful for segmentation where defects vary in size.
###############################################################################

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
    """ASPP global pooling branch: GAP -> Conv1x1 -> BN -> ReLU -> upsample."""

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
        # BN applied AFTER upsample to avoid batch_size=1 issue on (B,C,1,1)
        return self.relu(self.bn(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Aggregates 5 parallel branches:
        - 1x1 conv (rate=1)
        - 3x3 conv (rate=6)
        - 3x3 conv (rate=12)
        - 3x3 conv (rate=18)
        - Global average pooling branch
    Concatenate all -> 1x1 conv projection -> dropout.

    Args:
        in_channels: Input channels (encoder bottleneck, e.g. 320).
        out_channels: Output channels after projection (e.g. 256).
        atrous_rates: Tuple of dilation rates (default (6, 12, 18)).
        dropout: Dropout rate after projection (default 0.5).
    """

    def __init__(self, in_channels=320, out_channels=256,
                 atrous_rates=(6, 12, 18), dropout=0.5):
        super().__init__()

        modules = []
        # Branch 1: 1x1 convolution (equivalent to dilation=1)
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        # Branches 2-4: dilated 3x3 convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        # Branch 5: global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.branches = nn.ModuleList(modules)

        # Projection: concat 5 branches -> 1x1 conv
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

    Unlike BandAttention (static per-band scalars), this module computes
    per-image dynamic weights via GAP -> FC -> ReLU -> FC -> Sigmoid.
    This allows the model to adapt band importance based on each image's
    global spectral statistics.

    Args:
        num_bands: Number of input spectral bands (default 9).
        reduction: Reduction ratio for FC bottleneck (default 2, so 9->4->9).
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
        w = self.pool(x)    # (B, C, 1, 1)
        w = self.fc(w)      # (B, C, 1, 1)
        return x * w

    def get_weights(self, x):
        """Return per-image band weights as numpy array.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, C) numpy array of sigmoid-gated weights.
        """
        with torch.no_grad():
            w = self.pool(x)
            w = self.fc(w)
        return w.squeeze(-1).squeeze(-1).cpu().numpy()  # (B, C)


class BandAttention(nn.Module):
    """Learnable per-band weighting at input level.

    Each spectral band gets an independent importance weight.
    Weights are passed through sigmoid so they stay in (0, 1).
    Initialized to zeros -> sigmoid(0) = 0.5 -> all bands start equal.

    After training, call get_weights() to inspect which bands
    the model considers most/least important. This is directly
    interpretable and publishable as a band importance figure.

    Only num_bands learnable parameters -- zero overfitting risk.

    Args:
        num_bands: Number of input spectral bands.
    """

    def __init__(self, num_bands=9):
        super().__init__()
        self.band_logits = nn.Parameter(torch.zeros(1, num_bands, 1, 1))

    def forward(self, x):
        weights = torch.sigmoid(self.band_logits)
        return x * weights

    def get_weights(self):
        """Return learned band importance weights as numpy array."""
        return torch.sigmoid(self.band_logits).detach().cpu().squeeze().numpy()


class GlobalSaliencyBranch(nn.Module):
    """Low-resolution branch that produces a spatial attention map.

    Downsamples the input 4x, extracts global saliency via 3 lightweight
    conv layers, and outputs a single-channel attention map. This map is
    interpolated to match the main branch's bottleneck spatial size and
    multiplied onto the bottleneck features, guiding the decoder to focus
    on suspicious regions.

    The attention map is interpretable: after training, it shows which
    spatial regions the model considers most likely to contain defects.

    Args:
        in_channels: Number of input bands (e.g. 9).
        downsample_factor: How much to shrink input before conv (default 4).
    """

    def __init__(self, in_channels=9, downsample_factor=4):
        super().__init__()
        self.downsample_factor = downsample_factor

        # (9, H/4, W/4) → (32, H/8, W/8) → (64, H/16, W/16) → (64, H/32, W/32)
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
        """
        Args:
            x: (B, C, H, W) original input image.
            bottleneck_size: (H_b, W_b) spatial size of the bottleneck features
                             to match (e.g. (32, 32) for EfficientNet-B0,
                             (16, 16) for MobileNetV2 with 512 input).
        Returns:
            attention_map: (B, 1, H_b, W_b) spatial attention map.
        """
        x_lr = F.interpolate(
            x, scale_factor=1.0 / self.downsample_factor,
            mode='bilinear', align_corners=False
        )
        feat = self.conv(x_lr)
        attn = self.attention_head(feat)
        attn = F.interpolate(
            attn, size=bottleneck_size,
            mode='bilinear', align_corners=False
        )
        return attn

    def get_attention_map(self, x, bottleneck_size):
        """Return attention map for visualization (detached numpy)."""
        with torch.no_grad():
            return self.forward(x, bottleneck_size).cpu().numpy()


class SpectralDifferenceAttention(nn.Module):
    """Spectral Difference Attention (SDA).

    Computes the spectral residual (pixel spectrum minus local-mean spectrum)
    and uses its L2 norm as a spatial anomaly map.  Healthy pixels have near-zero
    residuals (spectrally homogeneous neighbourhoods), whereas defect and
    boundary pixels produce large residuals.

    Two usage modes controlled by ``mode``:
        * ``"input"`` — placed before/after the encoder on raw MSI input
          (replaces InputBandSE position).  Config field: ``use_sda_input``.
        * ``"skip"``  — placed at each decoder skip connection
          (replaces SE / CBAM position).  Config field: ``skip_module: "sda"``.

    Args:
        mode: ``"input"`` or ``"skip"``.  Only cosmetic (both share the same
              forward path); kept for clarity in logs / repr.
        learnable_gate: If True, adds a 1×1 conv + sigmoid gate on the
                        anomaly map (2 learnable params).  Otherwise pure
                        computation (0 params).
    """

    def __init__(self, mode="input", learnable_gate=True):
        super().__init__()
        self.mode = mode
        self.learnable_gate = learnable_gate
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        if learnable_gate:
            self.gate_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

    def _anomaly_map(self, x):
        """Compute normalized anomaly map from input tensor."""
        x_local_mean = self.avg_pool(x)
        spectral_residual = x - x_local_mean
        anomaly_map = torch.norm(spectral_residual, dim=1, keepdim=True)
        anomaly_map = anomaly_map / (anomaly_map.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return anomaly_map

    def forward(self, x):
        anomaly_map = self._anomaly_map(x)
        if self.learnable_gate:
            gate = torch.sigmoid(self.gate_conv(anomaly_map))
        else:
            gate = anomaly_map
        return x * (1 + gate)

    def get_anomaly_map(self, x):
        """Return anomaly map as numpy array for visualization.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, 1, H, W) numpy array in [0, 1].
        """
        with torch.no_grad():
            return self._anomaly_map(x).cpu().numpy()
