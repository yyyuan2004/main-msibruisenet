"""
decoder.py — UNet-style Decoder for MSI-BruiseNet
===================================================

Key I/O:
    Input : List of 4 encoder feature maps + attention/fusion config
    Output: (B, num_classes, H, W) segmentation logits

Each decoder stage:
    1. Bilinear upsample the previous decoder feature
    2. Apply attention (e.g. LSAA) to the corresponding encoder skip feature
    3. Fuse via ConvGLU (or concat) the skip and upsampled features
    4. Refine with 2 × (Conv3x3 → BN → ReLU)
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention, build_fusion


class ConvBNReLU(nn.Module):
    """Conv3x3 → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    """Single decoder stage: upsample → attention on skip → fuse → refine.

    Args:
        enc_channels: Channels of the encoder skip feature at this level.
        dec_in_channels: Channels of the incoming decoder feature (from deeper level).
        out_channels: Output channels of this decoder stage.
        attention_cfg: Dict with keys 'name' and optional kwargs for attention.
        fusion_name: Name of the fusion module ('convglu' or 'concat').
    """

    def __init__(
        self,
        enc_channels: int,
        dec_in_channels: int,
        out_channels: int,
        attention_cfg: Dict[str, Any],
        fusion_name: str = "convglu",
    ) -> None:
        super().__init__()
        self.attention = build_attention(
            name=attention_cfg.get("name", "lsaa"),
            channels=enc_channels,
            **{k: v for k, v in attention_cfg.items() if k != "name"},
        )
        self.fusion = build_fusion(fusion_name, enc_channels, dec_in_channels, out_channels)
        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, f_dec: torch.Tensor, f_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_dec: Decoder feature from the deeper stage (B, C_dec, H/2, W/2).
            f_enc: Encoder skip feature at this level (B, C_enc, H, W).
        Returns:
            Refined decoder feature (B, out_ch, H, W).
        """
        # Upsample decoder feature to match encoder spatial size
        f_dec_up = F.interpolate(
            f_dec, size=f_enc.shape[2:], mode="bilinear", align_corners=False
        )
        # Apply attention on encoder skip
        f_skip = self.attention(f_enc)
        # Fuse
        f_fused = self.fusion(f_skip, f_dec_up)
        # Refine
        return self.refine(f_fused)


class UNetDecoder(nn.Module):
    """Full UNet decoder with multi-stage skip connections.

    Args:
        encoder_channels: List of encoder output channels [F1_ch, F2_ch, F3_ch, F4_ch]
                          from shallow (stride 2) to deep (stride 16).
        decoder_channels: List of decoder output channels per stage (4 values).
        num_classes: Number of segmentation classes.
        attention_cfg: Attention config dict.
        fusion_name: Fusion module name.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int = 2,
        attention_cfg: Optional[Dict[str, Any]] = None,
        fusion_name: str = "convglu",
    ) -> None:
        super().__init__()
        if attention_cfg is None:
            attention_cfg = {"name": "lsaa"}

        # Build decoder stages from deep to shallow.
        # Stage 0: F4(enc) + F4(dec_in=F4_ch) → dec_channels[0]
        # Stage 1: F3(enc) + dec_channels[0]   → dec_channels[1]
        # Stage 2: F2(enc) + dec_channels[1]   → dec_channels[2]
        # Stage 3: F1(enc) + dec_channels[2]   → dec_channels[3]
        enc_ch = list(reversed(encoder_channels))  # deep → shallow
        self.stages = nn.ModuleList()
        dec_in = enc_ch[0]  # deepest encoder output is the initial decoder input
        for i, (e_ch, d_ch) in enumerate(zip(enc_ch[1:], decoder_channels)):
            # For the first stage, dec_in is the deepest encoder channel
            self.stages.append(
                DecoderStage(e_ch, dec_in, d_ch, attention_cfg, fusion_name)
            )
            dec_in = d_ch

        # Final segmentation head: upsample to input resolution + 1×1 conv
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(
        self, features: List[torch.Tensor], input_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [F1, F2, F3, F4] (shallow → deep).
            input_size: [H, W] of the original input for final upsampling.
        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        # Reverse so that index 0 is the deepest (F4)
        feats = list(reversed(features))
        x = feats[0]  # deepest encoder feature as initial decoder input
        for i, stage in enumerate(self.stages):
            x = stage(x, feats[i + 1])

        logits = self.seg_head(x)
        if input_size is not None:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
        return logits
