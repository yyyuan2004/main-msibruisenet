"""UNet decoder blocks with pluggable feature enhancement modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SEBlock, CBAMBlock, SpectralDifferenceAttention


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
        Bilinear Upsample(2x) -> Concat(skip) -> [optional module] -> Conv-BN-ReLU -> Conv-BN-ReLU

    The optional module inserted after concat depends on config:
        - "none": standard double convolution
        - "se": SE block after concat, before convolutions
        - "cbam": CBAM block after concat, before convolutions

    Args:
        in_channels: Channels from upsampled feature (before concat).
        skip_channels: Channels from skip connection.
        out_channels: Output channels for this decoder level.
        skip_module: Module type — "none", "se", "cbam", or "sda".
        se_reduction: SE/CBAM reduction ratio.
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 skip_module="none", se_reduction=16, **kwargs):
        super().__init__()

        concat_channels = in_channels + skip_channels
        self.skip_module_type = skip_module

        if skip_module == "se":
            self.se = SEBlock(concat_channels, reduction=se_reduction)
            self.conv1 = ConvBNReLU(concat_channels, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)
        elif skip_module == "cbam":
            self.cbam = CBAMBlock(concat_channels, reduction=se_reduction)
            self.conv1 = ConvBNReLU(concat_channels, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)
        elif skip_module == "sda":
            self.sda = SpectralDifferenceAttention(mode="skip", learnable_gate=True)
            self.conv1 = ConvBNReLU(concat_channels, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)
        else:
            # Baseline: standard double Conv-BN-ReLU
            self.conv1 = ConvBNReLU(concat_channels, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x, skip):
        """
        Args:
            x: Upsampled feature from deeper level.
            skip: Skip connection feature from encoder.
        """
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)

        if self.skip_module_type == "se":
            x = self.se(x)
        elif self.skip_module_type == "cbam":
            x = self.cbam(x)
        elif self.skip_module_type == "sda":
            x = self.sda(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UNetDecoder(nn.Module):
    """Full UNet decoder with 4 upsampling levels.

    Args:
        encoder_channels: List of encoder output channels [16, 24, 32, 96, 320].
        decoder_channels: List of decoder output channels [128, 64, 32, 16].
        num_classes: Number of segmentation classes.
        skip_module: Module type for skip connections ("none", "se", "cbam").
        se_reduction: SE/CBAM reduction ratio.
        bottleneck_channels: If ASPP is used upstream, this overrides encoder_channels[4].
        sda_decoder_extra_ch: Extra channels from SDA maps injected at decoder levels 2-3.
    """

    def __init__(self, encoder_channels=None, decoder_channels=None,
                 num_classes=2, skip_module="none", se_reduction=16,
                 bottleneck_channels=None, sda_decoder_extra_ch=0, **kwargs):

        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 24, 32, 96, 320]
        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16]

        self.sda_decoder_extra_ch = sda_decoder_extra_ch
        bottleneck = bottleneck_channels if bottleneck_channels else encoder_channels[4]

        in_ch = [bottleneck, decoder_channels[0], decoder_channels[1], decoder_channels[2]]
        skip_ch = [encoder_channels[3], encoder_channels[2], encoder_channels[1], encoder_channels[0]]

        if sda_decoder_extra_ch > 0:
            skip_ch[2] = skip_ch[2] + sda_decoder_extra_ch
            skip_ch[3] = skip_ch[3] + sda_decoder_extra_ch

        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(DecoderBlock(
                in_channels=in_ch[i],
                skip_channels=skip_ch[i],
                out_channels=decoder_channels[i],
                skip_module=skip_module,
                se_reduction=se_reduction,
            ))

        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features, sda_maps=None):
        """
        Args:
            features: List [S1, S2, S3, S4, S5] from encoder.
            sda_maps: Optional (B, N_sda, H, W) anomaly maps at full resolution.
                      Concatenated with S2 and S1 skips when sda_decoder_extra_ch > 0.
        Returns:
            Logits of shape (B, num_classes, H/2, W/2).
        """
        s1, s2, s3, s4, s5 = features

        x = self.blocks[0](s5, s4)
        x = self.blocks[1](x, s3)

        if sda_maps is not None and self.sda_decoder_extra_ch > 0:
            sda_s2 = F.interpolate(sda_maps, size=s2.shape[2:],
                                   mode="bilinear", align_corners=False)
            s2 = torch.cat([s2, sda_s2], dim=1)

        x = self.blocks[2](x, s2)

        if sda_maps is not None and self.sda_decoder_extra_ch > 0:
            sda_s1 = F.interpolate(sda_maps, size=s1.shape[2:],
                                   mode="bilinear", align_corners=False)
            s1 = torch.cat([s1, sda_s1], dim=1)

        x = self.blocks[3](x, s1)

        return self.seg_head(x)
