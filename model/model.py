"""Full model assembly: Encoder + UNet decoder + optional modules.

Supports three encoder backends (via config `encoder_name`):
    - "mobilenetv2": MobileNetV2, channels [16,24,32,96,320], stride down to 1/32
    - "mobilenetv3": MobileNetV3-Large, channels [16,24,40,112,960], stride down to 1/16
    - "efficientnet_b0": EfficientNet-B0, channels [16,24,40,112,1280], stride down to 1/16

The decoder channel widths auto-adapt to the encoder's output channels.
"""

import torch.nn as nn
import torch.nn.functional as F

from .encoder import MobileNetV2Encoder, MobileNetV3Encoder, EfficientNetB0Encoder
from .decoder import UNetDecoder
from .modules import SpectralConv1D, InputBandSE


ENCODERS = {
    "mobilenetv2": MobileNetV2Encoder,
    "mobilenetv3": MobileNetV3Encoder,
    "efficientnet_b0": EfficientNetB0Encoder,
}


class SegmentationModel(nn.Module):
    """Encoder-UNet segmentation model for MSI data.

    Configurable components:
        - Encoder: MobileNetV2 / EfficientNet-B0 / MobileNetV3
        - Skip modules: none / se
        - SpectralConv1D after S1 (optional)
        - InputBandSE before encoder (optional)
    """

    def __init__(self, num_classes=2, in_channels=9,
                 encoder_name="mobilenetv2", pretrained=True,
                 skip_module="none", se_reduction=16,
                 use_spectral_conv=False, spectral_conv_kernel_size=3,
                 use_band_attention=False, band_se_reduction=2):
        super().__init__()

        self.use_band_attention = use_band_attention
        if use_band_attention:
            self.band_attention = InputBandSE(
                num_bands=in_channels, reduction=band_se_reduction,
            )

        # Encoder
        encoder_cls = ENCODERS.get(encoder_name)
        if encoder_cls is None:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Available: {list(ENCODERS.keys())}"
            )
        self.encoder = encoder_cls(in_channels=in_channels, pretrained=pretrained)

        enc_channels = self.encoder.get_output_channels()

        # Optional spectral conv after S1
        self.use_spectral_conv = use_spectral_conv
        if use_spectral_conv:
            self.spectral_conv = SpectralConv1D(
                num_channels=enc_channels[0],
                kernel_size=spectral_conv_kernel_size,
            )

        # Decoder
        self.decoder = UNetDecoder(
            encoder_channels=enc_channels,
            num_classes=num_classes,
            skip_module=skip_module,
            se_reduction=se_reduction,
        )

    def forward(self, x):
        input_size = x.shape[2:]

        if self.use_band_attention:
            x = self.band_attention(x)

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]

        if self.use_spectral_conv:
            features[0] = self.spectral_conv(features[0])

        logits = self.decoder(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)
        return logits


# Backward-compatible alias
MobileNetV2UNet = SegmentationModel


def build_model(cfg):
    """Build model from config dict.

    Multi-architecture dispatch via cfg["model"]["architecture"]:
        - "default" (or absent): SegmentationModel
        - "smp": segmentation_models_pytorch wrapper
        - "topformer": TopFormer (CVPR 2022)
        - "seaformer": SeaFormer (ICLR 2023)
        - "pidnet": PIDNet (CVPR 2023)
    """
    model_cfg = cfg["model"]
    arch = model_cfg.get("architecture", "default")

    if arch == "smp":
        from .smp_models import build_smp_model
        return build_smp_model(cfg)
    if arch == "topformer":
        from .topformer import build_topformer
        return build_topformer(cfg)
    if arch == "seaformer":
        from .seaformer import build_seaformer
        return build_seaformer(cfg)
    if arch == "pidnet":
        from .pidnet import build_pidnet
        return build_pidnet(cfg)
    if arch != "default":
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available: default, smp, topformer, seaformer, pidnet"
        )

    return SegmentationModel(
        num_classes=model_cfg.get("num_classes", 2),
        in_channels=cfg["data"].get("num_channels", 9),
        encoder_name=model_cfg.get("encoder_name", "mobilenetv2"),
        pretrained=model_cfg.get("encoder_pretrained", True),
        skip_module=model_cfg.get("skip_module", "none"),
        se_reduction=model_cfg.get("se_reduction", 16),
        use_spectral_conv=model_cfg.get("use_spectral_conv", False),
        spectral_conv_kernel_size=model_cfg.get("spectral_conv_kernel_size", 3),
        use_band_attention=model_cfg.get("use_band_attention", False),
        band_se_reduction=model_cfg.get("band_se_reduction", 2),
    )
