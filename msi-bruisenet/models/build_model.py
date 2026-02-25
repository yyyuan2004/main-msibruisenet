"""
build_model.py — Factory to assemble the full MSI-BruiseNet
============================================================

Key I/O:
    Input : config dict (from config.yaml)
    Output: nn.Module — complete MSI-BruiseNet model

Architecture overview:
    MobileNetV2 Encoder (9-ch input, ImageNet pretrained)
       → 4-level features at stride 2, 4, 8, 16
    UNet Decoder with LSAA + ConvGLU at each skip connection
       → (B, 2, H, W) segmentation logits
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import MobileNetV2Encoder
from .decoder import UNetDecoder


class MSIBruiseNet(nn.Module):
    """MSI-BruiseNet: 9-band multispectral apple bruise segmentation network.

    Args:
        in_channels: Number of input spectral bands (default 9).
        num_classes: Number of segmentation classes (default 2).
        pretrained_backbone: Load ImageNet pretrained MobileNetV2 weights.
        decoder_channels: Channel list for each decoder stage.
        attention_cfg: Attention module configuration dict.
        fusion_name: Fusion strategy name ('convglu' or 'concat').
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        decoder_channels: Optional[List[int]] = None,
        attention_cfg: Optional[Dict[str, Any]] = None,
        fusion_name: str = "convglu",
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
        if attention_cfg is None:
            attention_cfg = {"name": "lsaa", "kernel_size": 5, "reduction": 4, "bypass": True}

        self.encoder = MobileNetV2Encoder(
            in_channels=in_channels, pretrained=pretrained_backbone
        )
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels_list,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            attention_cfg=attention_cfg,
            fusion_name=fusion_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input MSI tensor (B, 9, H, W).
        Returns:
            Segmentation logits (B, num_classes, H, W) at the original input resolution.
        """
        input_size = list(x.shape[2:])
        features = self.encoder(x)
        logits = self.decoder(features, input_size=input_size)
        return logits


def build_model(cfg: Dict[str, Any]) -> MSIBruiseNet:
    """Build MSI-BruiseNet from a flat or nested config dict.

    Expected config keys (nested under 'model' and 'data'):
        data.num_channels, data.num_classes,
        model.pretrained, model.attention, model.fusion, model.bypass,
        model.lsaa_kernel_size, model.lsaa_reduction, model.decoder_channels
    """
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    in_channels = data_cfg.get("num_channels", 9)
    num_classes = data_cfg.get("num_classes", 2)
    pretrained = model_cfg.get("pretrained", True)
    decoder_channels = model_cfg.get("decoder_channels", [256, 128, 64, 32])

    attn_name = model_cfg.get("attention", "lsaa")
    attention_cfg = {
        "name": attn_name,
        "kernel_size": model_cfg.get("lsaa_kernel_size", 5),
        "reduction": model_cfg.get("lsaa_reduction", 4),
        "bypass": model_cfg.get("bypass", True),
    }
    fusion_name = model_cfg.get("fusion", "convglu")

    model = MSIBruiseNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        decoder_channels=decoder_channels,
        attention_cfg=attention_cfg,
        fusion_name=fusion_name,
    )
    return model
