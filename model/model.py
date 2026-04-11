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
from .modules import SpectralConv1D, ASPP, BandAttention, InputBandSE, GlobalSaliencyBranch, SEBlock


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
        - BandAttention before encoder (optional)
        - GlobalSaliencyBranch guiding bottleneck (optional)

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
        use_band_attention: Insert BandAttention before encoder input.
        use_global_branch: Insert GlobalSaliencyBranch to guide bottleneck.
        global_downsample: Downsample factor for GlobalSaliencyBranch input.
    """

    def __init__(self, num_classes=2, in_channels=9,
                 encoder_name="efficientnet_b0", pretrained=True,
                 skip_module="none", se_reduction=16, convglu_expansion=4,
                 use_spectral_conv=False, spectral_conv_kernel_size=3,
                 use_aspp=False, aspp_out_channels=256,
                 aspp_atrous_rates=(6, 12, 18), aspp_dropout=0.5,
                 use_band_attention=False,
                 band_attention_type="static", band_se_reduction=2,
                 use_global_branch=False, global_downsample=4):
        super().__init__()

        # Optional band attention at input level (before encoder)
        # "static" = BandAttention (fixed per-band scalars, 9 params)
        # "dynamic" = InputBandSE (per-image GAP->FC->Sigmoid, 85 params)
        self.use_band_attention = use_band_attention
        self.band_attention_type = band_attention_type
        if use_band_attention:
            if band_attention_type == "dynamic":
                self.band_attention = InputBandSE(
                    num_bands=in_channels, reduction=band_se_reduction
                )
            else:
                self.band_attention = BandAttention(num_bands=in_channels)

        # Optional global saliency branch (guides bottleneck features)
        self.use_global_branch = use_global_branch
        if use_global_branch:
            self.global_branch = GlobalSaliencyBranch(
                in_channels=in_channels,
                downsample_factor=global_downsample,
            )

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

        # SE on S5 bottleneck (always enabled)
        # S5 is the most information-dense layer and the only one that
        # feeds directly into the decoder without a skip connection.
        se_in = bottleneck_channels if bottleneck_channels else enc_channels[4]
        self.bottleneck_se = SEBlock(se_in, reduction=se_reduction)

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

        # Save original input for global branch (before band attention)
        if self.use_global_branch:
            x_input = x

        if self.use_band_attention:
            x = self.band_attention(x)

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]

        if self.use_spectral_conv:
            features[0] = self.spectral_conv(features[0])

        if self.use_aspp:
            features[4] = self.aspp(features[4])

        # SE channel attention on bottleneck
        features[4] = self.bottleneck_se(features[4])

        # Global saliency guidance: multiply bottleneck by spatial attention
        if self.use_global_branch:
            bottleneck_size = features[4].shape[2:]
            attn_map = self.global_branch(x_input, bottleneck_size)
            features[4] = features[4] * attn_map

        logits = self.decoder(features)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)

        return logits


# Backward-compatible alias
MobileNetV2UNet = SegmentationModel


def build_model(cfg):
    """Build model from config dict.

    Supports multi-architecture dispatch via cfg["model"]["architecture"]:
        - "default" (or absent): SegmentationModel (MobileNetV2/EfficientNet-B0 UNet)
        - "deeplabv3plus_fang": Fang 2025 Improved DeepLabV3+
        - "smp": segmentation_models_pytorch wrapper

    Existing configs without "architecture" field default to SegmentationModel.
    """
    model_cfg = cfg["model"]
    arch = model_cfg.get("architecture", "default")

    if arch == "deeplabv3plus_fang":
        from .deeplabv3plus import build_deeplabv3plus_fang
        return build_deeplabv3plus_fang(cfg)
    elif arch == "smp":
        from .smp_models import build_smp_model
        return build_smp_model(cfg)
    elif arch != "default":
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available: default, deeplabv3plus_fang, smp"
        )

    # Default: SegmentationModel
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
        use_band_attention=model_cfg.get("use_band_attention", False),
        band_attention_type=model_cfg.get("band_attention_type", "static"),
        band_se_reduction=model_cfg.get("band_se_reduction", 2),
        use_global_branch=model_cfg.get("use_global_branch", False),
        global_downsample=model_cfg.get("global_downsample", 4),
    )

    return model
