"""Model builder for MSI-BruiseNet."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .backbone import MobileNetV2Encoder
from .decoder import UNetDecoder


class MSIBruiseNet(nn.Module):
    """Complete MSI-BruiseNet model."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.input_size = int(cfg["data"]["input_size"])
        self.encoder = MobileNetV2Encoder(
            in_channels=int(cfg["data"]["num_channels"]),
            pretrained=bool(cfg["model"]["pretrained"]),
        )
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=list(cfg["model"]["decoder_channels"]),
            num_classes=int(cfg["data"]["num_classes"]),
            attention_name=str(cfg["model"]["attention"]),
            lsaa_kernel_size=int(cfg["model"]["lsaa_kernel_size"]),
            lsaa_reduction=int(cfg["model"]["lsaa_reduction"]),
            use_convglu=bool(cfg["model"].get("use_convglu", True)),
            lsaa_bypass=bool(cfg["model"].get("lsaa_bypass", True)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.decoder(feats, out_size=x.shape[-1])


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build MSI-BruiseNet from config dict."""
    return MSIBruiseNet(cfg)
