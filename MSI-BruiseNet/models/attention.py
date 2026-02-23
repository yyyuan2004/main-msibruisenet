"""Attention and skip-fusion modules for MSI bruise segmentation.

Key I/O:
- Input: encoder/decoder features (B, C, H, W)
- Output: enhanced fused features for decoder skip connection
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAttention(nn.Module):
    """No-op attention for ablation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LSAA(nn.Module):
    """Local Spectral Anomaly Attention."""

    def __init__(self, channels: int, kernel_size: int = 5, reduction: int = 4, bypass: bool = True) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.bypass = bypass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_bg = self.pool(x)
        spectral_residual = x - local_bg
        w = self.attn(spectral_residual)
        out = x * w
        return out + x if self.bypass else out


class SEAttention(nn.Module):
    """Squeeze-and-Excitation."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mlp(x)


class ECAAttention(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAMAttention(nn.Module):
    """CBAM attention module."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True).values
        ch = torch.sigmoid(self.channel(torch.mean(x, dim=(2, 3), keepdim=True)) + self.channel(torch.amax(x, dim=(2, 3), keepdim=True)))
        x = x * ch
        sp = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sp


class ConvGLUFusion(nn.Module):
    """ConvGLU-style skip fusion between encoder and decoder features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, f_lsaa: torch.Tensor, f_dec: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_lsaa, f_dec], dim=1))
        return g * f_lsaa + (1.0 - g) * f_dec


def get_attention(name: str, channels: int, kernel_size: int = 5, reduction: int = 4, bypass: bool = True) -> nn.Module:
    """Factory for attention modules used in ablation."""
    name = name.lower()
    if name == "lsaa":
        return LSAA(channels, kernel_size=kernel_size, reduction=reduction, bypass=bypass)
    if name == "se":
        return SEAttention(channels)
    if name == "cbam":
        return CBAMAttention(channels)
    if name == "eca":
        return ECAAttention(channels)
    return IdentityAttention()
