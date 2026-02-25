"""
decoder.py - UNet-style decoder with bilinear upsampling and skip-connection fusion.

Responsibility:
    - Accept 4-level encoder features + attention/fusion modules.
    - At each level: upsample -> align channels -> fuse with skip -> 2×Conv-BN-ReLU.
    - Output a final segmentation logit map at original spatial resolution.

I/O:
    Input  : List of 4 encoder features [F1..F4] (low-to-high resolution indices)
    Output : (B, num_classes, H, W) logit tensor
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention, build_fusion


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm -> ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single decoder level: upsample + attention on skip + fusion + 2×ConvBNReLU.

    Args:
        enc_ch: Encoder skip-connection channels.
        dec_in_ch: Decoder input channels (from lower level).
        out_ch: Output channels for this level.
        attention_name: Attention type for skip features.
        fusion_name: Fusion method ('convglu' or 'concat').
        lsaa_kernel_size: LSAA window size.
        lsaa_reduction: LSAA channel reduction ratio.
        bypass: LSAA residual connection.
    """

    def __init__(
        self,
        enc_ch: int,
        dec_in_ch: int,
        out_ch: int,
        attention_name: str = "lsaa",
        fusion_name: str = "convglu",
        lsaa_kernel_size: int = 5,
        lsaa_reduction: int = 4,
        bypass: bool = True,
    ) -> None:
        super().__init__()
        # Align decoder channels to match encoder skip channels before fusion
        self.dec_align = ConvBNReLU(dec_in_ch, enc_ch, kernel_size=1)

        # Attention on encoder skip features
        self.attention = build_attention(
            attention_name, enc_ch,
            kernel_size=lsaa_kernel_size,
            reduction=lsaa_reduction,
            bypass=bypass,
        )

        # Fusion of (attended skip, upsampled decoder)
        self.fusion = build_fusion(fusion_name, enc_ch)

        # 2× Conv-BN-ReLU after fusion
        self.conv_block = nn.Sequential(
            ConvBNReLU(enc_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            enc_feat: Encoder skip feature (B, enc_ch, H, W).
            dec_feat: Decoder feature from lower level (B, dec_in_ch, H/2, W/2).

        Returns:
            Decoder output (B, out_ch, H, W).
        """
        # Upsample decoder feature to encoder spatial size
        dec_up = F.interpolate(dec_feat, size=enc_feat.shape[2:], mode="bilinear", align_corners=False)
        dec_up = self.dec_align(dec_up)

        # Attention on skip
        enc_att = self.attention(enc_feat)

        # Gated fusion
        fused = self.fusion(enc_att, dec_up)

        return self.conv_block(fused)


class UNetDecoder(nn.Module):
    """UNet decoder with configurable attention and fusion at each skip connection.

    Args:
        encoder_channels: List of encoder output channels [stage0..stage3].
        decoder_channels: List of decoder output channels (length must be len(encoder_channels)).
        num_classes: Number of segmentation classes.
        attention_name: Attention module name.
        fusion_name: Fusion module name.
        lsaa_kernel_size: LSAA window size.
        lsaa_reduction: LSAA reduction ratio.
        bypass: LSAA residual flag.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int = 2,
        attention_name: str = "lsaa",
        fusion_name: str = "convglu",
        lsaa_kernel_size: int = 5,
        lsaa_reduction: int = 4,
        bypass: bool = True,
    ) -> None:
        super().__init__()
        assert len(decoder_channels) == len(encoder_channels), (
            "decoder_channels length must match encoder_channels length"
        )

        self.blocks = nn.ModuleList()
        # Build from deepest to shallowest
        for i in reversed(range(len(encoder_channels))):
            if i == len(encoder_channels) - 1:
                # Deepest level: no lower decoder input, use encoder directly
                dec_in_ch = encoder_channels[i]
            else:
                dec_in_ch = decoder_channels[i + 1]
            self.blocks.append(
                DecoderBlock(
                    enc_ch=encoder_channels[i],
                    dec_in_ch=dec_in_ch,
                    out_ch=decoder_channels[i],
                    attention_name=attention_name,
                    fusion_name=fusion_name,
                    lsaa_kernel_size=lsaa_kernel_size,
                    lsaa_reduction=lsaa_reduction,
                    bypass=bypass,
                )
            )

        # Final 1×1 classification head
        self.seg_head = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)

    def forward(self, encoder_features: List[torch.Tensor], input_size: tuple) -> torch.Tensor:
        """Forward pass.

        Args:
            encoder_features: List of encoder features [F1(1/2), F2(1/4), F3(1/8), F4(1/16)].
            input_size: Original (H, W) for final upsampling.

        Returns:
            Logit tensor (B, num_classes, H, W).
        """
        # Start from deepest feature
        x = encoder_features[-1]
        # Decode from deep to shallow (blocks stored deepest-first)
        for idx, block in enumerate(self.blocks):
            enc_idx = len(encoder_features) - 1 - idx
            if idx == 0:
                # Deepest level: attention + self-fusion
                enc_att = block.attention(encoder_features[enc_idx])
                dec_up = block.dec_align(x)
                x = block.fusion(enc_att, dec_up)
                x = block.conv_block(x)
            else:
                x = block(encoder_features[enc_idx], x)

        # Upsample to original input resolution
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.seg_head(x)
