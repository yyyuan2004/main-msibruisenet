"""
backbone.py — MobileNetV2 Encoder adapted for 9-channel MSI input
=================================================================

Key I/O:
    Input : (B, 9, H, W) multi-spectral image tensor
    Output: list of 4 feature maps at 1/2, 1/4, 1/8, 1/16 resolution

The first Conv2d(3, 32, 3, stride=2) is replaced with Conv2d(9, 32, 3, stride=2).
ImageNet pretrained weights for the first layer are repeated along the channel
dimension and divided by 3 to preserve activation magnitude.
"""

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 backbone with 9-channel input adaptation.

    Args:
        in_channels: Number of input channels (default 9 for MSI).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    # Indices into MobileNetV2 `features` that correspond to the 4 output stages.
    # These yield feature maps at strides 2, 4, 8, 16.
    STAGE_INDICES = [1, 3, 6, 13]

    def __init__(self, in_channels: int = 9, pretrained: bool = True) -> None:
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # --- Adapt first conv layer from 3 channels to `in_channels` ---
        original_conv: nn.Conv2d = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained:
            # Repeat 3-channel weights to cover `in_channels` and normalise
            pretrained_weight = original_conv.weight.data  # (32, 3, 3, 3)
            repeats = (in_channels + 2) // 3  # ceil division
            new_weight = pretrained_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
            new_weight = new_weight / (in_channels / 3.0)
            new_conv.weight.data = new_weight
        base.features[0][0] = new_conv

        self.features = base.features
        self.out_channels_list = self._get_out_channels()

    def _get_out_channels(self) -> List[int]:
        """Return the number of output channels at each stage."""
        channels: List[int] = []
        for idx in self.STAGE_INDICES:
            block = self.features[idx]
            # InvertedResidual stores output channels in `out_channels` attribute
            if hasattr(block, "out_channels"):
                channels.append(block.out_channels)
            else:
                # Fallback: inspect last conv/bn in the block
                for m in reversed(list(block.modules())):
                    if isinstance(m, nn.BatchNorm2d):
                        channels.append(m.num_features)
                        break
        return channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of 4 tensors [F1, F2, F3, F4] at strides [2, 4, 8, 16].
        """
        features: List[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.STAGE_INDICES:
                features.append(x)
            if i == self.STAGE_INDICES[-1]:
                break
        return features
