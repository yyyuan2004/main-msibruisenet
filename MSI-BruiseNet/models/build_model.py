"""
build_model.py - Factory function to assemble the complete MSI-BruiseNet.

Responsibility:
    - Read model config and construct encoder + decoder.
    - Expose a single `build_model(cfg)` entry point.

I/O:
    Input  : (B, 9, H, W) MSI tensor
    Output : (B, num_classes, H, W) segmentation logits
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .backbone import MobileNetV2Encoder
from .decoder import UNetDecoder


class MSIBruiseNet(nn.Module):
    """MSI-BruiseNet: 9-band multispectral apple bruise segmentation network.

    Architecture:
        MobileNetV2 encoder (9-ch adapted) -> LSAA + ConvGLU skip fusion -> UNet decoder

    Args:
        in_channels: Input spectral bands (default 9).
        num_classes: Number of output classes (default 2: background + bruise).
        pretrained: Load ImageNet pretrained encoder weights.
        attention: Attention module name ('lsaa', 'se', 'cbam', 'eca', 'none').
        fusion: Fusion module name ('convglu', 'concat').
        bypass: LSAA residual connection.
        lsaa_kernel_size: LSAA local window size.
        lsaa_reduction: LSAA channel reduction ratio.
        decoder_channels: List of decoder output channels per level.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 2,
        pretrained: bool = True,
        attention: str = "lsaa",
        fusion: str = "convglu",
        bypass: bool = True,
        lsaa_kernel_size: int = 5,
        lsaa_reduction: int = 4,
        decoder_channels: List[int] = None,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        self.encoder = MobileNetV2Encoder(
            in_channels=in_channels,
            pretrained=pretrained,
        )

        enc_channels = self.encoder.out_channels_list
        # decoder_channels must have same length as encoder stages
        assert len(decoder_channels) == len(enc_channels), (
            f"decoder_channels ({len(decoder_channels)}) must match "
            f"encoder stages ({len(enc_channels)})"
        )

        self.decoder = UNetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            attention_name=attention,
            fusion_name=fusion,
            lsaa_kernel_size=lsaa_kernel_size,
            lsaa_reduction=lsaa_reduction,
            bypass=bypass,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 9, H, W) multispectral input tensor.

        Returns:
            (B, num_classes, H, W) segmentation logits.
        """
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features, input_size)
        return logits


def build_model(cfg: Dict[str, Any]) -> MSIBruiseNet:
    """Build MSI-BruiseNet from a configuration dictionary.

    Args:
        cfg: Full configuration dict (typically loaded from config.yaml).

    Returns:
        Instantiated MSIBruiseNet model.
    """
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    model = MSIBruiseNet(
        in_channels=dcfg.get("num_channels", 9),
        num_classes=dcfg.get("num_classes", 2),
        pretrained=mcfg.get("pretrained", True),
        attention=mcfg.get("attention", "lsaa"),
        fusion=mcfg.get("fusion", "convglu"),
        bypass=mcfg.get("bypass", True),
        lsaa_kernel_size=mcfg.get("lsaa_kernel_size", 5),
        lsaa_reduction=mcfg.get("lsaa_reduction", 4),
        decoder_channels=mcfg.get("decoder_channels", [256, 128, 64, 32]),
    )
    return model
