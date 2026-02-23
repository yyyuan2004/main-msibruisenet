"""MobileNetV2 backbone for MSI (9-channel) input.

Key I/O:
- Input: MSI tensor (B, 9, H, W)
- Output: multi-scale features F1..F4
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 encoder adapted to 9-channel MSI input."""

    def __init__(self, in_channels: int = 9, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v2(weights=weights)
        first_conv = model.features[0][0]
        if in_channels != 3:
            new_conv = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
            with torch.no_grad():
                if pretrained:
                    pretrained_weight = first_conv.weight.data
                    repeat_factor = in_channels // 3
                    new_weight = pretrained_weight.repeat(1, repeat_factor, 1, 1) / float(repeat_factor)
                    new_conv.weight.copy_(new_weight)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out")
            model.features[0][0] = new_conv
        self.features = model.features
        self.out_channels = [16, 24, 32, 96]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i == 1:
                feats.append(x)  # 1/2
            elif i == 3:
                feats.append(x)  # 1/4
            elif i == 6:
                feats.append(x)  # 1/8
            elif i == 13:
                feats.append(x)  # 1/16
        return feats
