"""Full model assembly: Encoder + UNet decoder + optional modules.

Supports three encoder backends (via config `encoder_name`):
    - "mobilenetv2": MobileNetV2, channels [16,24,32,96,320], stride down to 1/32
    - "mobilenetv3": MobileNetV3-Large, channels [16,24,40,112,960], stride down to 1/16
    - "efficientnet_b0": EfficientNet-B0, channels [16,24,40,112,1280], stride down to 1/16

The decoder channel widths auto-adapt to the encoder's output channels.
All other components (ASPP, skip modules, SpectralConv1D) work with all encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MobileNetV2Encoder, MobileNetV3Encoder, EfficientNetB0Encoder
from .decoder import UNetDecoder
from .modules import (
    SpectralConv1D, ASPP, BandAttention, InputBandSE,
    GlobalSaliencyBranch, SEBlock, SpectralDifferenceAttention, SDAModuleV2,
)


# Encoder registry: name -> class
ENCODERS = {
    "mobilenetv2": MobileNetV2Encoder,
    "mobilenetv3": MobileNetV3Encoder,
    "efficientnet_b0": EfficientNetB0Encoder,
}


class SegmentationModel(nn.Module):
    """Encoder-UNet segmentation model for MSI data.

    Configurable components:
        - Encoder: MobileNetV2 / EfficientNet-B0 / MobileNetV3 (via encoder_name)
        - Skip modules: none / se / cbam
        - ASPP at bottleneck (optional)
        - SpectralConv1D after S1 (optional)
        - BandAttention / InputBandSE before encoder (optional)
        - GlobalSaliencyBranch guiding bottleneck (optional)
    """

    def __init__(self, num_classes=2, in_channels=9,
                 encoder_name="mobilenetv2", pretrained=True,
                 skip_module="none", se_reduction=16,
                 use_spectral_conv=False, spectral_conv_kernel_size=3,
                 use_aspp=False, aspp_out_channels=256,
                 aspp_atrous_rates=(6, 12, 18), aspp_dropout=0.5,
                 use_band_attention=False,
                 band_attention_type="static", band_se_reduction=2,
                 use_global_branch=False, global_downsample=4,
                 use_sda_input=False, sda_learnable_gate=True,
                 sda_v2_config=None):
        super().__init__()

        # ---- SDA v2 (new interpretable module) ----
        self.sda_v2_enabled = False
        self.sda_v2_position = "none"
        self._sda_v2_extra_enc_ch = 0
        self._sda_v2_extra_s2_ch = 0
        sda_extra_input = 0

        if sda_v2_config is not None and sda_v2_config.get("enabled", False):
            self.sda_v2_enabled = True
            self.sda_v2_position = sda_v2_config.get("position", "input")
            self.sda_v2 = SDAModuleV2(
                feature_names=sda_v2_config.get(
                    "features", ["spectral_std", "sam", "snv_l2", "mahalanobis"]),
                sigma_a=sda_v2_config.get("sigma_a", 3.0),
                sigma_t=sda_v2_config.get("sigma_t", 5.0),
                gate_mode=sda_v2_config.get("gate_mode", "concat"),
                use_soft_gate=sda_v2_config.get("use_soft_gate", True),
            )
            n_sda = self.sda_v2.out_channels
            if self.sda_v2_position in ("input", "multiscale"):
                sda_extra_input = n_sda
                self._sda_v2_extra_enc_ch = n_sda
            if self.sda_v2_position in ("s2", "multiscale"):
                self._sda_v2_extra_s2_ch = n_sda

        # ---- Legacy SDA v1 (kept for backward compat) ----
        self.use_sda_input = use_sda_input
        if use_sda_input and not self.sda_v2_enabled:
            self.sda_input = SpectralDifferenceAttention(
                mode="input", learnable_gate=sda_learnable_gate,
            )

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

        # Build encoder (extra channels from SDA-input concat)
        encoder_cls = ENCODERS.get(encoder_name)
        if encoder_cls is None:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Available: {list(ENCODERS.keys())}"
            )
        self.encoder = encoder_cls(
            in_channels=in_channels + sda_extra_input, pretrained=pretrained
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
        dec_enc_channels = list(enc_channels)
        if self._sda_v2_extra_s2_ch > 0:
            dec_enc_channels[1] = enc_channels[1] + self._sda_v2_extra_s2_ch

        sda_decoder_ch = 0
        if self.sda_v2_enabled and self.sda_v2_position == "decoder":
            sda_decoder_ch = self.sda_v2.out_channels

        self.decoder = UNetDecoder(
            encoder_channels=dec_enc_channels,
            num_classes=num_classes,
            skip_module=skip_module,
            se_reduction=se_reduction,
            bottleneck_channels=bottleneck_channels,
            sda_decoder_extra_ch=sda_decoder_ch,
        )

    def forward(self, x, apple_mask=None):
        input_size = x.shape[2:]

        # Save raw input for SDA v2 and global branch
        x_raw = x
        if self.use_global_branch:
            x_input = x

        # ---- SDA v2: compute anomaly features from raw input ----
        sda_maps = None
        if self.sda_v2_enabled:
            sda_maps = self.sda_v2(x_raw, apple_mask=apple_mask)
            if self.sda_v2_position in ("input", "multiscale"):
                x = torch.cat([x, sda_maps], dim=1)

        # ---- Legacy SDA v1 ----
        if self.use_sda_input and not self.sda_v2_enabled:
            x = self.sda_input(x, apple_mask=apple_mask)

        if self.use_band_attention:
            x = self.band_attention(x)

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]

        # ---- SDA v2 s2 injection ----
        if self.sda_v2_enabled and self.sda_v2_position in ("s2", "multiscale"):
            sda_s2 = F.interpolate(
                sda_maps, size=features[1].shape[2:],
                mode="bilinear", align_corners=False,
            )
            features[1] = torch.cat([features[1], sda_s2], dim=1)

        if self.use_spectral_conv:
            features[0] = self.spectral_conv(features[0])

        if self.use_aspp:
            features[4] = self.aspp(features[4])

        # SE channel attention on bottleneck
        features[4] = self.bottleneck_se(features[4])

        # Global saliency guidance
        if self.use_global_branch:
            bottleneck_size = features[4].shape[2:]
            attn_map = self.global_branch(x_input, bottleneck_size)
            features[4] = features[4] * attn_map

        # ---- Decoder (pass SDA maps for decoder_sda position) ----
        if self.sda_v2_enabled and self.sda_v2_position == "decoder":
            logits = self.decoder(features, sda_maps=sda_maps)
        else:
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
    sda_v2_config = model_cfg.get("sda_v2", None)

    model = SegmentationModel(
        num_classes=model_cfg.get("num_classes", 2),
        in_channels=cfg["data"].get("num_channels", 9),
        encoder_name=model_cfg.get("encoder_name", "mobilenetv2"),
        pretrained=model_cfg.get("encoder_pretrained", True),
        skip_module=model_cfg.get("skip_module", "none"),
        se_reduction=model_cfg.get("se_reduction", 16),
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
        use_sda_input=model_cfg.get("use_sda_input", False),
        sda_learnable_gate=model_cfg.get("sda_learnable_gate", True),
        sda_v2_config=sda_v2_config,
    )

    return model
