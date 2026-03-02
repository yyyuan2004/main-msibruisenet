"""Full model assembly: MobileNetV2 encoder + UNet decoder + optional modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MobileNetV2Encoder
from .decoder import UNetDecoder
from .modules import SpectralConv1D


class MobileNetV2UNet(nn.Module):
    """MobileNetV2-UNet for pixel-level semantic segmentation of MSI data.

    Configurable skip connection modules for ablation study:
        - baseline: No enhancement modules
        - +SE: SE block at each skip connection
        - +1D-SpConv: Spectral 1D conv after encoder S1
        - +ConvGLU: ConvGLU replacing Conv-BN-ReLU at skip connections
        - +1D-SpConv+SE: Combination of spectral conv and SE

    Args:
        num_classes: Number of segmentation classes.
        in_channels: Number of input spectral bands (default 9).
        pretrained: Use ImageNet pretrained encoder weights.
        skip_module: "none", "se", or "convglu".
        se_reduction: SE reduction ratio.
        convglu_expansion: ConvGLU expansion ratio.
        use_spectral_conv: Whether to insert SpectralConv1D after S1.
        spectral_conv_kernel_size: Kernel size for spectral 1D conv.
    """

    def __init__(self, num_classes=2, in_channels=9, pretrained=True,
                 skip_module="none", se_reduction=16, convglu_expansion=4,
                 use_spectral_conv=False, spectral_conv_kernel_size=3):
        super().__init__()

        self.encoder = MobileNetV2Encoder(
            in_channels=in_channels, pretrained=pretrained
        )

        # Optional spectral conv after S1 (16 channels at stride 1/2)
        self.use_spectral_conv = use_spectral_conv
        if use_spectral_conv:
            # S1 output has 16 channels
            self.spectral_conv = SpectralConv1D(
                num_channels=16,
                kernel_size=spectral_conv_kernel_size
            )

        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.get_output_channels(),
            num_classes=num_classes,
            skip_module=skip_module,
            se_reduction=se_reduction,
            convglu_expansion=convglu_expansion,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 9, H, W).

        Returns:
            Logits of shape (B, num_classes, H, W) — upsampled to input resolution.
        """
        input_size = x.shape[2:]

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]

        # Apply spectral conv after S1 if configured
        if self.use_spectral_conv:
            features[0] = self.spectral_conv(features[0])

        logits = self.decoder(features)  # (B, C, H/2, W/2)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)

        return logits


def build_model(cfg):
    """Build model from config dict.

    Args:
        cfg: Configuration dictionary.

    Returns:
        MobileNetV2UNet model instance.
    """
    model_cfg = cfg["model"]

    model = MobileNetV2UNet(
        num_classes=model_cfg.get("num_classes", 2),
        in_channels=cfg["data"].get("num_channels", 9),
        pretrained=model_cfg.get("encoder_pretrained", True),
        skip_module=model_cfg.get("skip_module", "none"),
        se_reduction=model_cfg.get("se_reduction", 16),
        convglu_expansion=model_cfg.get("convglu_expansion_ratio", 4),
        use_spectral_conv=model_cfg.get("use_spectral_conv", False),
        spectral_conv_kernel_size=model_cfg.get("spectral_conv_kernel_size", 3),
    )

    return model
