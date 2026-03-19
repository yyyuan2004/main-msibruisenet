"""Full model assembly: Encoder + UNet decoder + optional modules.

Supports two encoder backends (via config `encoder_name`):
    - "mobilenetv2": MobileNetV2, channels [16,24,32,96,320], stride down to 1/32
    - "efficientnet_b0": EfficientNet-B0, channels [16,24,40,112,1280], stride down to 1/16

The decoder channel widths auto-adapt to the encoder's output channels.
All other components (ASPP, skip modules, SpectralConv1D) work with both encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MobileNetV2Encoder, EfficientNetB0Encoder
from .decoder import UNetDecoder
from .modules import SpectralConv1D, ASPP


# Encoder registry: name -> class
ENCODERS = {
    "mobilenetv2": MobileNetV2Encoder,
    "efficientnet_b0": EfficientNetB0Encoder,
}


class SegmentationModel(nn.Module):
    """Encoder-UNet segmentation model for MSI data.

    Configurable components:
        - Encoder: MobileNetV2 or EfficientNet-B0 (via encoder_name)
        - Skip modules: none / se / cbam / convglu / cbam_convglu
        - ASPP at bottleneck (optional)
        - SpectralConv1D after S1 (optional)

    Args:
        num_classes: Number of segmentation classes.
        in_channels: Number of input spectral bands (default 9).
        encoder_name: "mobilenetv2" or "efficientnet_b0".
        pretrained: Use ImageNet pretrained encoder weights.
        skip_module: Skip connection module type.
        se_reduction: SE/CBAM reduction ratio.
        convglu_expansion: ConvGLU expansion ratio.
        use_spectral_conv: Insert SpectralConv1D after S1.
        spectral_conv_kernel_size: SpectralConv1D kernel size.
        use_aspp: Insert ASPP at bottleneck.
        aspp_out_channels: ASPP output channels.
        aspp_atrous_rates: ASPP dilation rates.
        aspp_dropout: ASPP dropout rate.
    """

    def __init__(self, num_classes=2, in_channels=9,
                 encoder_name="efficientnet_b0", pretrained=True,
                 skip_module="none", se_reduction=16, convglu_expansion=4,
                 use_spectral_conv=False, spectral_conv_kernel_size=3,
                 use_aspp=False, aspp_out_channels=256,
                 aspp_atrous_rates=(6, 12, 18), aspp_dropout=0.5):
        super().__init__()

        # Build encoder
        encoder_cls = ENCODERS.get(encoder_name)
        if encoder_cls is None:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Available: {list(ENCODERS.keys())}"
            )
        self.encoder = encoder_cls(
            in_channels=in_channels, pretrained=pretrained
        )

        enc_channels = self.encoder.get_output_channels()
        # e.g. MobileNetV2: [16, 24, 32, 96, 320]
        # e.g. EfficientNet-B0: [16, 24, 40, 112, 1280]

        # Optional spectral conv after S1
        self.use_spectral_conv = use_spectral_conv
        if use_spectral_conv:
            self.spectral_conv = SpectralConv1D(
                num_channels=enc_channels[0],  # S1 channels (16 for both)
                kernel_size=spectral_conv_kernel_size
            )

        # Optional ASPP at bottleneck (S5)
        self.use_aspp = use_aspp
        bottleneck_channels = None
        if use_aspp:
            self.aspp = ASPP(
                in_channels=enc_channels[4],
                out_channels=aspp_out_channels,
                atrous_rates=aspp_atrous_rates,
                dropout=aspp_dropout,
            )
            bottleneck_channels = aspp_out_channels

        # Decoder — auto-adapts to encoder's channel widths
        self.decoder = UNetDecoder(
            encoder_channels=enc_channels,
            num_classes=num_classes,
            skip_module=skip_module,
            se_reduction=se_reduction,
            convglu_expansion=convglu_expansion,
            bottleneck_channels=bottleneck_channels,
        )

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]

        if self.use_spectral_conv:
            features[0] = self.spectral_conv(features[0])

        if self.use_aspp:
            features[4] = self.aspp(features[4])

        logits = self.decoder(features)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)

        return logits


# Backward-compatible alias
MobileNetV2UNet = SegmentationModel


def build_model(cfg):
    """Build model from config dict.

    Config fields used:
        model.encoder_name: "mobilenetv2" or "efficientnet_b0" (default "efficientnet_b0")
        model.num_classes, model.skip_module, model.use_aspp, etc.
    """
    model_cfg = cfg["model"]

    model = SegmentationModel(
        num_classes=model_cfg.get("num_classes", 2),
        in_channels=cfg["data"].get("num_channels", 9),
        encoder_name=model_cfg.get("encoder_name", "efficientnet_b0"),
        pretrained=model_cfg.get("encoder_pretrained", True),
        skip_module=model_cfg.get("skip_module", "none"),
        se_reduction=model_cfg.get("se_reduction", 16),
        convglu_expansion=model_cfg.get("convglu_expansion_ratio", 4),
        use_spectral_conv=model_cfg.get("use_spectral_conv", False),
        spectral_conv_kernel_size=model_cfg.get("spectral_conv_kernel_size", 3),
        use_aspp=model_cfg.get("use_aspp", False),
        aspp_out_channels=model_cfg.get("aspp_out_channels", 256),
        aspp_atrous_rates=tuple(model_cfg.get("aspp_atrous_rates", [6, 12, 18])),
        aspp_dropout=model_cfg.get("aspp_dropout", 0.5),
    )

    return model
