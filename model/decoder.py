"""UNet decoder blocks with pluggable feature enhancement modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# [CHANGED] Added CBAMBlock import for the "+CBAM" ablation variant
from .modules import SEBlock, CBAMBlock, ConvGLU


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
        - "convglu": ConvGLU replaces the double Conv-BN-ReLU block
        - "cbam_convglu": [NEW] CBAM attention -> ConvGLU channel mixing (fused)
            设计动机: CBAM先做通道+空间注意力筛选（"看哪里"），
            ConvGLU再做门控通道变换（"怎么融合"），两者串行、各司其职。
            CBAM不改变通道数，只做注意力重标定；
            ConvGLU负责通道降维和非线性混合。

    Args:
        in_channels: Channels from upsampled feature (before concat).
        skip_channels: Channels from skip connection.
        out_channels: Output channels for this decoder level.
        skip_module: Module type — "none", "se", "cbam", "convglu", or "cbam_convglu".
        se_reduction: SE/CBAM reduction ratio (used if skip_module="se" or "cbam" or "cbam_convglu").
        convglu_expansion: ConvGLU expansion ratio (only used if skip_module="convglu" or "cbam_convglu").
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 skip_module="none", se_reduction=16, convglu_expansion=4):
        super().__init__()

        concat_channels = in_channels + skip_channels
        self.skip_module_type = skip_module

        if skip_module == "se":
            self.se = SEBlock(concat_channels, reduction=se_reduction)
            self.conv1 = ConvBNReLU(concat_channels, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)
        elif skip_module == "cbam":
            # [NEW] CBAM: channel + spatial attention at skip connections
            self.cbam = CBAMBlock(concat_channels, reduction=se_reduction)
            self.conv1 = ConvBNReLU(concat_channels, out_channels)
            self.conv2 = ConvBNReLU(out_channels, out_channels)
        elif skip_module == "cbam_convglu":
            # [NEW] 融合模式: CBAM注意力筛选 → ConvGLU通道混合
            # 数据流: concat(416ch) → CBAM(416→416, 注意力重标定)
            #       → ConvGLU(416→128, 门控通道变换)
            # CBAM负责"看哪里重要"（通道+空间注意力）
            # ConvGLU负责"怎么融合"（GELU门控的通道降维+混合）
            self.cbam = CBAMBlock(concat_channels, reduction=se_reduction)
            self.convglu = ConvGLU(concat_channels, out_channels,
                                   expansion_ratio=convglu_expansion)
        elif skip_module == "convglu":
            self.convglu = ConvGLU(concat_channels, out_channels,
                                   expansion_ratio=convglu_expansion)
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
        # Bilinear upsample to match skip spatial dimensions
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        if self.skip_module_type == "se":
            x = self.se(x)
            x = self.conv1(x)
            x = self.conv2(x)
        elif self.skip_module_type == "cbam":
            # [NEW] CBAM: channel attn -> spatial attn -> double conv
            x = self.cbam(x)
            x = self.conv1(x)
            x = self.conv2(x)
        elif self.skip_module_type == "cbam_convglu":
            # [NEW] 融合执行: 先CBAM注意力，再ConvGLU通道混合
            x = self.cbam(x)     # 注意力重标定（不改shape）
            x = self.convglu(x)  # 门控通道变换（降维到out_channels）
        elif self.skip_module_type == "convglu":
            x = self.convglu(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        return x


class UNetDecoder(nn.Module):
    """Full UNet decoder with 4 upsampling levels.

    Decoder channel configuration (from bottleneck to high resolution):
        D4: 320↑ + 96 = 416 -> 128
        D3: 128↑ + 32 = 160 -> 64
        D2:  64↑ + 24 =  88 -> 32
        D1:  32↑ + 16 =  48 -> 16

    Args:
        encoder_channels: List of encoder output channels [16, 24, 32, 96, 320].
        decoder_channels: List of decoder output channels [128, 64, 32, 16].
        num_classes: Number of segmentation classes.
        skip_module: Module type for skip connections ("none", "se", "cbam", "convglu").
        se_reduction: SE/CBAM reduction ratio.
        convglu_expansion: ConvGLU expansion ratio.
        bottleneck_channels: If ASPP is used upstream, this overrides encoder_channels[4].
            E.g., ASPP(320 -> 256) means bottleneck_channels=256. [NEW]
    """

    def __init__(self, encoder_channels=None, decoder_channels=None,
                 num_classes=2, skip_module="none", se_reduction=16,
                 convglu_expansion=4, bottleneck_channels=None):

        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 24, 32, 96, 320]
        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16]

        # [CHANGED] If ASPP transforms bottleneck channels, use the new value
        # instead of encoder_channels[4] (original 320).
        bottleneck = bottleneck_channels if bottleneck_channels else encoder_channels[4]

        # S5(320 or ASPP out) is bottleneck, S4(96)..S1(16) are skip connections
        # D4: bottleneck + 96 -> 128
        # D3: 128 + 32 -> 64
        # D2:  64 + 24 -> 32
        # D1:  32 + 16 -> 16
        in_ch = [bottleneck, decoder_channels[0], decoder_channels[1], decoder_channels[2]]
        skip_ch = [encoder_channels[3], encoder_channels[2], encoder_channels[1], encoder_channels[0]]

        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(DecoderBlock(
                in_channels=in_ch[i],
                skip_channels=skip_ch[i],
                out_channels=decoder_channels[i],
                skip_module=skip_module,
                se_reduction=se_reduction,
                convglu_expansion=convglu_expansion,
            ))

        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: List [S1, S2, S3, S4, S5] from encoder.

        Returns:
            Logits of shape (B, num_classes, H/2, W/2).
            Note: output is at stride 2 (half input resolution) since S1 is at stride 2.
        """
        s1, s2, s3, s4, s5 = features

        x = self.blocks[0](s5, s4)  # D4: bottleneck + S4
        x = self.blocks[1](x, s3)   # D3: D4 + S3
        x = self.blocks[2](x, s2)   # D2: D3 + S2
        x = self.blocks[3](x, s1)   # D1: D2 + S1

        return self.seg_head(x)
