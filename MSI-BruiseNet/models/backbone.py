"""
backbone.py - MobileNetV2 encoder adapted for 9-channel MSI input.

Responsibility:
    - Load torchvision MobileNetV2 with optional ImageNet pretrained weights.
    - Replace the first Conv2d(3, 32, 3, stride=2) with Conv2d(9, 32, 3, stride=2).
    - Weight-transfer strategy: repeat 3-ch weights ×3 along channel dim, then /3.
    - Expose 4-level feature maps at 1/2, 1/4, 1/8, 1/16 spatial resolution.

I/O:
    Input  : (B, 9, H, W) float32 tensor
    Output : list of 4 feature tensors [F1, F2, F3, F4]
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models


# MobileNetV2 inverted-residual block indices for each resolution stage.
# features[0]   -> stride 2 (1/2)   32 ch
# features[1:3] -> stride 4 (1/4)  24 ch  (inverted residual blocks 1-2)
# features[3:6] -> stride 8 (1/8)  32 ch  (inverted residual blocks 3-5)
# features[6:13]-> stride 16 (1/16) 96 ch (inverted residual blocks 6-12, last conv 160->96 in block 13 is 1/16 too)
# We use features[0:2], [2:4], [4:7], [7:14] which give 4 stages.
# Actually MobileNetV2 features has 19 children (0..18).
# Standard grouping for UNet:
#   Stage 0: features[0:2]  -> output channels = 16, stride 2
#   Stage 1: features[2:4]  -> output channels = 24, stride 4
#   Stage 2: features[4:7]  -> output channels = 32, stride 8
#   Stage 3: features[7:14] -> output channels = 96, stride 16

_STAGE_INDICES = [
    (0, 2),    # 1/2  -> 16 ch
    (2, 4),    # 1/4  -> 24 ch
    (4, 7),    # 1/8  -> 32 ch
    (7, 14),   # 1/16 -> 96 ch
]

STAGE_OUT_CHANNELS = [16, 24, 32, 96]


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 backbone producing 4-level feature maps for 9-ch MSI input.

    Args:
        in_channels: Number of input channels (default 9 for MSI).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, in_channels: int = 9, pretrained: bool = True) -> None:
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None,
        )

        # ------------------------------------------------------------------
        # Adapt first conv layer: Conv2d(3, 32, 3, stride=2) -> Conv2d(in_channels, 32, 3, stride=2)
        # Strategy: repeat 3-ch pretrained weights along input-channel dim,
        #           then divide by (in_channels // 3) to preserve activation magnitude.
        # ------------------------------------------------------------------
        old_conv: nn.Conv2d = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            old_weight = old_conv.weight.data  # (32, 3, 3, 3)
            repeats = (in_channels + 2) // 3   # ceil(9/3) = 3
            new_weight = old_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
            new_weight = new_weight / (in_channels / 3.0)
            new_conv.weight.data = new_weight
        base.features[0][0] = new_conv

        # Store feature sub-sequences as separate nn.Sequential stages.
        self.stages = nn.ModuleList()
        for start, end in _STAGE_INDICES:
            self.stages.append(nn.Sequential(*list(base.features.children())[start:end]))

    @property
    def out_channels_list(self) -> List[int]:
        """Return output channels for each stage."""
        return list(STAGE_OUT_CHANNELS)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, 9, H, W).

        Returns:
            List of 4 tensors: [F1(1/2), F2(1/4), F3(1/8), F4(1/16)].
        """
        features: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
