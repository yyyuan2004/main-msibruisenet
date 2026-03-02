"""MobileNetV2 encoder adapted for 9-channel multispectral input."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 backbone extracting 5 levels of features for UNet skip connections.

    Feature extraction points (from MobileNetV2 features):
        S1: features[0:2]   -> stride 1/2,  channels 16
        S2: features[2:4]   -> stride 1/4,  channels 24
        S3: features[4:7]   -> stride 1/8,  channels 32
        S4: features[7:14]  -> stride 1/16, channels 96
        S5: features[14:18] -> stride 1/32, channels 320 (bottleneck)

    Args:
        in_channels: Number of input channels (default 9 for MSI).
        pretrained: Whether to use ImageNet pretrained weights.
    """

    # Feature extraction boundaries (exclusive end index for each stage)
    STAGE_ENDS = [2, 4, 7, 14, 18]
    OUT_CHANNELS = [16, 24, 32, 96, 320]

    def __init__(self, in_channels=9, pretrained=True):
        super().__init__()

        if pretrained:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v2(weights=None)

        # Adapt the first conv layer from 3 channels to in_channels
        original_conv = backbone.features[0][0]  # Conv2d(3, 32, 3, stride=2, padding=1)
        new_conv = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Copy pretrained weights for first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
            # Kaiming init for remaining channels
            nn.init.kaiming_normal_(
                new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu"
            )

        backbone.features[0][0] = new_conv

        # Store all feature blocks as a list for sequential processing
        self.blocks = nn.ModuleList(backbone.features)

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, 9, H, W).

        Returns:
            List of 5 feature tensors [S1, S2, S3, S4, S5].
        """
        features = []
        stage_idx = 0

        for i, block in enumerate(self.blocks):
            x = block(x)
            if stage_idx < len(self.STAGE_ENDS) and i == self.STAGE_ENDS[stage_idx] - 1:
                features.append(x)
                stage_idx += 1

        return features  # [S1, S2, S3, S4, S5]

    def get_output_channels(self):
        """Return output channel counts for each stage."""
        return list(self.OUT_CHANNELS)
