"""UNet-like decoder for MSI-BruiseNet.

Key I/O:
- Input: encoder pyramid features and decoder channels config
- Output: segmentation logits (B, num_classes, H, W)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ConvGLUFusion, get_attention


class ConvBlock(nn.Module):
    """Two-layer Conv-BN-ReLU block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """UNet decoder with configurable attention and ConvGLU skip fusion."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
        attention_name: str,
        lsaa_kernel_size: int,
        lsaa_reduction: int,
        use_convglu: bool,
        lsaa_bypass: bool,
    ) -> None:
        super().__init__()
        self.proj4 = nn.Conv2d(encoder_channels[3], decoder_channels[0], 1)
        self.up3 = ConvBlock(decoder_channels[0], decoder_channels[1])
        self.up2 = ConvBlock(decoder_channels[1], decoder_channels[2])
        self.up1 = ConvBlock(decoder_channels[2], decoder_channels[3])
        self.up0 = ConvBlock(decoder_channels[3], decoder_channels[3])

        skip_ch = [encoder_channels[2], encoder_channels[1], encoder_channels[0]]
        dec_ch = [decoder_channels[0], decoder_channels[1], decoder_channels[2]]

        self.skip_proj = nn.ModuleList([nn.Conv2d(s, d, 1) for s, d in zip(skip_ch, dec_ch)])
        self.attn = nn.ModuleList([
            get_attention(attention_name, d, kernel_size=lsaa_kernel_size, reduction=lsaa_reduction, bypass=lsaa_bypass)
            for d in dec_ch
        ])
        self.use_convglu = use_convglu
        self.glu = nn.ModuleList([ConvGLUFusion(d) for d in dec_ch])
        self.head = nn.Conv2d(decoder_channels[3], num_classes, 1)

    def _fuse(self, idx: int, f_enc: torch.Tensor, f_dec: torch.Tensor) -> torch.Tensor:
        f_enc = self.skip_proj[idx](f_enc)
        f_enc = self.attn[idx](f_enc)
        if self.use_convglu:
            return self.glu[idx](f_enc, f_dec)
        return torch.cat([f_enc, f_dec], dim=1)[:, : f_dec.shape[1], :, :]

    def forward(self, feats: List[torch.Tensor], out_size: int) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        x = self.proj4(f4)

        x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self._fuse(0, f3, x)
        x = self.up3(x)

        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self._fuse(1, f2, x)
        x = self.up2(x)

        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self._fuse(2, f1, x)
        x = self.up1(x)

        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
        x = self.up0(x)
        return self.head(x)
