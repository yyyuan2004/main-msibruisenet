"""UNet decoder blocks with pluggable feature enhancement modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SEBlock


class ConvBNReLU(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """Single UNet decoder level.

    Operation:
        Bilinear Upsample(2x) -> Concat(skip) -> [optional SE] -> Conv-BN-ReLU x2

    Args:
        in_channels: Channels from upsampled feature (before concat).
        skip_channels: Channels from skip connection.
        out_channels: Output channels for this decoder level.
        skip_module: "none" / "se".
        se_reduction: SE reduction ratio.
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 skip_module="none", se_reduction=16):
        super().__init__()

        concat_channels = in_channels + skip_channels
        self.skip_module_type = skip_module

        if skip_module == "se":
            self.attn = SEBlock(concat_channels, reduction=se_reduction)
        else:
            self.attn = None

        self.conv1 = ConvBNReLU(concat_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        if self.attn is not None:
            x = self.attn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    """Full UNet decoder with 4 upsampling levels.

    Args:
        encoder_channels: List of encoder output channels [S1..S5].
        decoder_channels: List of decoder output channels [D1..D4].
        num_classes: Number of segmentation classes.
        skip_module: "none" / "se".
        se_reduction: SE reduction ratio.
    """

    def __init__(self, encoder_channels=None, decoder_channels=None,
                 num_classes=2, skip_module="none", se_reduction=16):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 24, 32, 96, 320]
        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16]

        in_ch = [encoder_channels[4], decoder_channels[0], decoder_channels[1], decoder_channels[2]]
        skip_ch = [encoder_channels[3], encoder_channels[2], encoder_channels[1], encoder_channels[0]]

        self.blocks = nn.ModuleList([
            DecoderBlock(
                in_channels=in_ch[i],
                skip_channels=skip_ch[i],
                out_channels=decoder_channels[i],
                skip_module=skip_module,
                se_reduction=se_reduction,
            )
            for i in range(4)
        ])

        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: List [S1, S2, S3, S4, S5] from encoder.
        Returns:
            Logits of shape (B, num_classes, H/2, W/2).
        """
        s1, s2, s3, s4, s5 = features

        x = self.blocks[0](s5, s4)
        x = self.blocks[1](x, s3)
        x = self.blocks[2](x, s2)
        x = self.blocks[3](x, s1)

        return self.seg_head(x)
