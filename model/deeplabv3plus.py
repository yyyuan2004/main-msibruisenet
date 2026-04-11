"""Fang 2025 Improved DeepLabV3+ for MSI apple bruise segmentation.

Reference: Fang et al. 2025, "Multispectral imaging-based detection of apple
bruises using segmentation network and classification model"

Key differences from standard DeepLabV3+:
    - Depthwise Separable Convolutions in ASPP (instead of standard conv)
    - ECA (Efficient Channel Attention) after ASPP
    - ASPP rates: [4, 8, 12, 16] (instead of [6, 12, 18])
    - MobileNetV2 encoder adapted for 9-channel MSI input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MobileNetV2Encoder


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: DWConv + Pointwise Conv."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class ECA(nn.Module):
    """Efficient Channel Attention (ECA-Net).

    Uses 1D convolution on channel descriptors instead of FC layers.
    Kernel size k is adaptively determined from channel count:
        k = |log2(C) / gamma + b / gamma|_odd

    Args:
        channels: Number of input channels.
        gamma: ECA gamma parameter (default 2).
        b: ECA b parameter (default 1).
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 == 1 else k + 1  # ensure odd
        k = max(k, 3)  # minimum kernel size 3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        # (B, C, 1, 1) -> (B, 1, C)
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)  # (B, 1, C)
        y = torch.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class FangASPP(nn.Module):
    """ASPP with Depthwise Separable Convolutions (Fang 2025).

    Branches:
        - 1x1 DSConv
        - 3x3 DSConv, rate=4
        - 3x3 DSConv, rate=8
        - 3x3 DSConv, rate=12
        - 3x3 DSConv, rate=16
        - Image Pooling (GAP -> 1x1 Conv -> Upsample)

    Args:
        in_channels: Input channels from encoder bottleneck.
        out_channels: Output channels after projection.
        atrous_rates: Tuple of dilation rates (default (4, 8, 12, 16)).
    """

    def __init__(self, in_channels, out_channels=256,
                 atrous_rates=(4, 8, 12, 16)):
        super().__init__()

        modules = []
        # Branch 1: 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        # Branches 2-5: DSConv with dilation
        for rate in atrous_rates:
            modules.append(DepthwiseSeparableConv(
                in_channels, out_channels, kernel_size=3,
                padding=rate, dilation=rate
            ))
        # Branch 6: Image pooling (BN applied after upsample to avoid bs=1 issue)
        self.pool_gap = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool_bn = nn.BatchNorm2d(out_channels)
        self.pool_relu = nn.ReLU(inplace=True)

        self.branches = nn.ModuleList(modules)

        num_branches = 1 + len(atrous_rates) + 1  # 1x1 + dilated + pooling
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[2:]
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))

        # Image pooling branch: GAP -> Conv -> Upsample -> BN -> ReLU
        pool_out = self.pool_conv(self.pool_gap(x))
        pool_out = F.interpolate(pool_out, size=size, mode="bilinear",
                                 align_corners=False)
        pool_out = self.pool_relu(self.pool_bn(pool_out))
        outputs.append(pool_out)

        x = torch.cat(outputs, dim=1)
        return self.project(x)


class DeepLabV3PlusFang(nn.Module):
    """Fang 2025 Improved DeepLabV3+ for 9-channel MSI segmentation.

    Architecture:
        Encoder (MobileNetV2) -> ASPP (DSConv) -> ECA -> Decoder

    Decoder:
        - Low-level features from encoder stride 1/4 (S2, 24ch)
        - 1x1 conv to reduce low-level channels
        - Upsample high-level + concat low-level -> 3x3 conv -> upsample 4x

    Args:
        num_classes: Number of segmentation classes.
        in_channels: Number of input spectral bands.
        pretrained: Use ImageNet pretrained encoder.
        aspp_out_channels: ASPP output channels.
        low_level_channels: Reduced low-level feature channels.
    """

    def __init__(self, num_classes=2, in_channels=9, pretrained=True,
                 aspp_out_channels=256, low_level_channels=48):
        super().__init__()

        self.encoder = MobileNetV2Encoder(
            in_channels=in_channels, pretrained=pretrained
        )
        enc_channels = self.encoder.get_output_channels()
        # [16, 24, 32, 96, 320]

        # ASPP with DSConv at bottleneck (S5, 320ch)
        self.aspp = FangASPP(
            in_channels=enc_channels[4],
            out_channels=aspp_out_channels,
        )

        # ECA after ASPP
        self.eca = ECA(aspp_out_channels)

        # Low-level feature processing (from S2, stride 1/4, 24ch)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(enc_channels[1], low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_channels, 256, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]
        low_level = features[1]     # S2: stride 1/4, 24ch
        high_level = features[4]    # S5: stride 1/32, 320ch

        # ASPP + ECA
        high_level = self.aspp(high_level)
        high_level = self.eca(high_level)

        # Upsample high-level to match low-level spatial size
        high_level = F.interpolate(
            high_level, size=low_level.shape[2:],
            mode="bilinear", align_corners=False
        )

        # Reduce low-level channels
        low_level = self.low_level_conv(low_level)

        # Concat and decode
        x = torch.cat([high_level, low_level], dim=1)
        x = self.decoder(x)

        # Upsample to input resolution
        x = F.interpolate(x, size=input_size, mode="bilinear",
                          align_corners=False)
        return x


def build_deeplabv3plus_fang(cfg):
    """Build Fang 2025 DeepLabV3+ from config."""
    model_cfg = cfg["model"]
    return DeepLabV3PlusFang(
        num_classes=model_cfg.get("num_classes", 2),
        in_channels=cfg["data"].get("num_channels", 9),
        pretrained=model_cfg.get("encoder_pretrained", True),
        aspp_out_channels=model_cfg.get("aspp_out_channels", 256),
        low_level_channels=model_cfg.get("low_level_channels", 48),
    )
