"""
attention.py - Attention modules for ablation experiments.

Implements:
    - LSAA  (Local Spectral Anomaly Attention) — proposed method
    - ConvGLU (Gated Linear Unit fusion)
    - SE    (Squeeze-and-Excitation)
    - CBAM  (Convolutional Block Attention Module)
    - ECA   (Efficient Channel Attention)
    - Identity (no-op baseline for ablation)

I/O:
    All attention modules: (B, C, H, W) -> (B, C, H, W)
    ConvGLU: ((B, C, H, W), (B, C, H, W)) -> (B, C, H, W)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LSAA — Local Spectral Anomaly Attention
# =============================================================================

class LSAA(nn.Module):
    """Local Spectral Anomaly Attention.

    Computes a local spectral residual and generates anomaly-aware weights
    to highlight bruise-related spectral deviations.

    Args:
        channels: Number of input feature channels.
        kernel_size: Local average-pooling window size.
        reduction: Channel reduction ratio for the bottleneck.
        bypass: Whether to add a residual (identity) connection.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        reduction: int = 4,
        bypass: bool = True,
    ) -> None:
        super().__init__()
        self.bypass = bypass
        mid = max(channels // reduction, 1)

        # Local background estimator
        self.local_avg = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        # Anomaly weight generator
        self.weight_gen = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Encoder feature map (B, C, H, W).

        Returns:
            LSAA-modulated feature (B, C, H, W).
        """
        f_bar = self.local_avg(x)        # local background estimate
        r_s = x - f_bar                  # spectral residual
        w_a = self.weight_gen(r_s)       # anomaly weights in [0, 1]
        out = x * w_a
        if self.bypass:
            out = out + x                # residual connection
        return out


# =============================================================================
# ConvGLU — Gated Linear Unit Fusion
# =============================================================================

class ConvGLU(nn.Module):
    """ConvGLU gated fusion of encoder (LSAA-enhanced) and decoder features.

    Args:
        channels: Number of channels in each input feature map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, f_enc: torch.Tensor, f_dec: torch.Tensor) -> torch.Tensor:
        """Gated fusion.

        Args:
            f_enc: LSAA-enhanced encoder skip feature (B, C, H, W).
            f_dec: Upsampled decoder feature (B, C, H, W).

        Returns:
            Fused feature (B, C, H, W).
        """
        gate = self.gate_conv(torch.cat([f_enc, f_dec], dim=1))
        return gate * f_enc + (1.0 - gate) * f_dec


# =============================================================================
# ConcatFusion — Simple concatenation + conv (ablation baseline for ConvGLU)
# =============================================================================

class ConcatFusion(nn.Module):
    """Simple concatenation + 1x1 conv fusion (ablation baseline).

    Args:
        channels: Number of channels in each input feature map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_enc: torch.Tensor, f_dec: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([f_enc, f_dec], dim=1))


# =============================================================================
# SE — Squeeze-and-Excitation
# =============================================================================

class SE(nn.Module):
    """Squeeze-and-Excitation block.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# =============================================================================
# CBAM — Convolutional Block Attention Module
# =============================================================================

class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(x.mean(dim=[2, 3]))
        max_out = self.mlp(x.amax(dim=[2, 3]))
        return torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =============================================================================
# ECA — Efficient Channel Attention
# =============================================================================

class ECA(nn.Module):
    """Efficient Channel Attention.

    Kernel size is adaptively determined from channel count.

    Args:
        channels: Number of input channels.
        gamma: Mapping parameter for adaptive kernel size.
        b: Mapping parameter for adaptive kernel size.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        k = int(abs(math.log2(channels) + b) / gamma)
        k = k if k % 2 else k + 1  # ensure odd
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)    # (B, C, 1, 1)
        return x * torch.sigmoid(y)


# =============================================================================
# Identity — No-op attention (ablation baseline)
# =============================================================================

class Identity(nn.Module):
    """Identity (no attention). Used as ablation baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# =============================================================================
# Factory
# =============================================================================

_ATTENTION_REGISTRY = {
    "lsaa": LSAA,
    "se": SE,
    "cbam": CBAM,
    "eca": ECA,
    "none": Identity,
}


def build_attention(
    name: str,
    channels: int,
    kernel_size: int = 5,
    reduction: int = 4,
    bypass: bool = True,
) -> nn.Module:
    """Build an attention module by name.

    Args:
        name: One of 'lsaa', 'se', 'cbam', 'eca', 'none'.
        channels: Number of input feature channels.
        kernel_size: LSAA local window size (ignored for others).
        reduction: Channel reduction ratio (used by LSAA/SE/CBAM).
        bypass: LSAA residual connection flag.

    Returns:
        An nn.Module implementing the attention mechanism.
    """
    name = name.lower()
    if name not in _ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention '{name}'. Choose from {list(_ATTENTION_REGISTRY.keys())}")
    cls = _ATTENTION_REGISTRY[name]
    if cls is Identity:
        return cls()
    if cls is LSAA:
        return cls(channels, kernel_size=kernel_size, reduction=reduction, bypass=bypass)
    if cls is ECA:
        return cls(channels)
    # SE, CBAM
    return cls(channels, reduction=reduction)


def build_fusion(name: str, channels: int) -> nn.Module:
    """Build a fusion module by name.

    Args:
        name: 'convglu' or 'concat'.
        channels: Number of channels per input.

    Returns:
        Fusion module.
    """
    if name == "convglu":
        return ConvGLU(channels)
    elif name == "concat":
        return ConcatFusion(channels)
    else:
        raise ValueError(f"Unknown fusion '{name}'. Choose from ['convglu', 'concat']")
