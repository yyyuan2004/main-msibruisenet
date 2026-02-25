"""
attention.py — Attention & Fusion modules for MSI-BruiseNet
============================================================

Implements the following modules for the main method and ablation studies:

Attention modules (applied on encoder skip features):
    - LSAA     : Local Spectral Anomaly Attention (proposed)
    - SE       : Squeeze-and-Excitation
    - CBAM     : Convolutional Block Attention Module
    - ECA      : Efficient Channel Attention
    - Identity : No attention (pass-through)

Fusion modules (merging encoder skip + decoder upsampled features):
    - ConvGLU  : Gated linear unit based fusion (proposed)
    - ConcatFusion : Simple concatenation + conv reduction
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Attention Modules
# =============================================================================

class LSAA(nn.Module):
    """Local Spectral Anomaly Attention.

    Computes a local spectral residual and derives an anomaly-aware attention
    weight map that modulates the encoder features.

    Args:
        channels: Number of input channels C.
        kernel_size: Local averaging window size k.
        reduction: Channel reduction ratio for the bottleneck.
        bypass: If True, add residual connection (F_e * W_a + F_e).
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
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        mid_ch = max(channels // reduction, 1)
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, f_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_e: Encoder feature map (B, C, H, W).
        Returns:
            LSAA-modulated feature map (B, C, H, W).
        """
        f_bar = self.avg_pool(f_e)           # local background estimate
        r_s = f_e - f_bar                    # spectral residual
        w_a = self.weight_net(r_s)           # anomaly-aware weight [0, 1]
        if self.bypass:
            return f_e * w_a + f_e           # modulation + residual
        return f_e * w_a


class SE(nn.Module):
    """Squeeze-and-Excitation block.

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_ch = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio for channel attention.
        kernel_size: Kernel size for spatial attention conv.
    """

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        mid_ch = max(channels // reduction, 1)
        # Channel attention
        self.ca_mlp = nn.Sequential(
            nn.Linear(channels, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, channels, bias=False),
        )
        # Spatial attention
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Channel attention
        avg_out = self.ca_mlp(x.mean(dim=[2, 3]))
        max_out = self.ca_mlp(x.amax(dim=[2, 3]))
        ca = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * ca
        # Spatial attention
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.amax(dim=1, keepdim=True)
        sa = self.sa_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        return x * sa


class ECA(nn.Module):
    """Efficient Channel Attention.

    Uses adaptive kernel size based on channel count.

    Args:
        channels: Number of input/output channels.
        gamma: Mapping parameter for kernel size computation.
        b_param: Mapping parameter for kernel size computation.
    """

    def __init__(self, channels: int, gamma: int = 2, b_param: int = 1) -> None:
        super().__init__()
        k = int(abs(math.log2(channels) + b_param) / gamma)
        k = k if k % 2 else k + 1  # ensure odd
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        y = y.unsqueeze(1)  # (B, 1, C)
        y = torch.sigmoid(self.conv(y))  # (B, 1, C)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class Identity(nn.Module):
    """Identity (no attention) — used as a baseline in ablation studies."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# =============================================================================
# Fusion Modules
# =============================================================================

class ConvGLU(nn.Module):
    """Gated Linear Unit fusion for combining encoder skip and decoder features.

    Args:
        enc_channels: Channels from the (attention-enhanced) encoder feature.
        dec_channels: Channels from the decoder upsampled feature.
        out_channels: Output channels after fusion.
    """

    def __init__(self, enc_channels: int, dec_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(enc_channels + dec_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        # Align channel dimensions if needed
        self.align_enc = (
            nn.Conv2d(enc_channels, out_channels, 1, bias=False)
            if enc_channels != out_channels
            else nn.Identity()
        )
        self.align_dec = (
            nn.Conv2d(dec_channels, out_channels, 1, bias=False)
            if dec_channels != out_channels
            else nn.Identity()
        )

    def forward(self, f_lsaa: torch.Tensor, f_d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_lsaa: Attention-enhanced encoder feature (B, C_enc, H, W).
            f_d: Decoder upsampled feature (B, C_dec, H, W).
        Returns:
            Fused feature (B, out_channels, H, W).
        """
        gate = self.gate_conv(torch.cat([f_lsaa, f_d], dim=1))
        f_enc = self.align_enc(f_lsaa)
        f_dec = self.align_dec(f_d)
        return gate * f_enc + (1 - gate) * f_dec


class ConcatFusion(nn.Module):
    """Simple concatenation + 1x1 conv reduction (ablation baseline).

    Args:
        enc_channels: Channels from the encoder feature.
        dec_channels: Channels from the decoder feature.
        out_channels: Output channels after fusion.
    """

    def __init__(self, enc_channels: int, dec_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(enc_channels + dec_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_enc: torch.Tensor, f_dec: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([f_enc, f_dec], dim=1))


# =============================================================================
# Factory helpers
# =============================================================================

ATTENTION_REGISTRY = {
    "lsaa": LSAA,
    "se": SE,
    "cbam": CBAM,
    "eca": ECA,
    "none": Identity,
}

FUSION_REGISTRY = {
    "convglu": ConvGLU,
    "concat": ConcatFusion,
}


def build_attention(name: str, channels: int, **kwargs) -> nn.Module:
    """Build an attention module by name.

    Args:
        name: One of 'lsaa', 'se', 'cbam', 'eca', 'none'.
        channels: Number of input channels.
        **kwargs: Extra arguments forwarded to the constructor.
    """
    name = name.lower()
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type '{name}'. Choose from {list(ATTENTION_REGISTRY)}")
    cls = ATTENTION_REGISTRY[name]
    if cls is Identity:
        return cls()
    if name == "lsaa":
        return cls(
            channels,
            kernel_size=kwargs.get("kernel_size", 5),
            reduction=kwargs.get("reduction", 4),
            bypass=kwargs.get("bypass", True),
        )
    return cls(channels)


def build_fusion(name: str, enc_ch: int, dec_ch: int, out_ch: int) -> nn.Module:
    """Build a fusion module by name.

    Args:
        name: One of 'convglu', 'concat'.
        enc_ch: Encoder skip channels.
        dec_ch: Decoder upsampled channels.
        out_ch: Output channels.
    """
    name = name.lower()
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion type '{name}'. Choose from {list(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](enc_ch, dec_ch, out_ch)
