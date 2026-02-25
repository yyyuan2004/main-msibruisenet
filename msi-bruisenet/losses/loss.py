"""
loss.py — Loss functions for MSI-BruiseNet
============================================

Key I/O:
    Input : logits (B, C, H, W), targets (B, H, W)
    Output: scalar loss value

Implements:
    L = L_CE + lambda_dice * L_Dice + lambda_aux * L_aux

Where:
    - L_CE   : Standard Cross-Entropy
    - L_Dice : Dice Loss for class imbalance
    - L_aux  : Spectral residual auxiliary loss (optional, boundary supervision)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary/multi-class segmentation.

    Args:
        smooth: Smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.
        Returns:
            Scalar dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BoundaryAuxLoss(nn.Module):
    """Spectral-residual auxiliary boundary loss.

    L_aux = -log(sigmoid(R_s * M_boundary))
    where M_boundary is a dilated ground-truth boundary mask.

    Args:
        dilation: Number of pixels to dilate the boundary mask.
    """

    def __init__(self, dilation: int = 3) -> None:
        super().__init__()
        self.dilation = dilation

    def _compute_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute dilated boundary mask from integer labels.

        Args:
            targets: (B, H, W) integer mask.
        Returns:
            (B, 1, H, W) float boundary mask.
        """
        targets_float = targets.unsqueeze(1).float()
        kernel_size = 2 * self.dilation + 1
        # Erode and dilate to find boundary
        eroded = -F.max_pool2d(
            -targets_float, kernel_size=kernel_size, stride=1, padding=self.dilation
        )
        dilated = F.max_pool2d(
            targets_float, kernel_size=kernel_size, stride=1, padding=self.dilation
        )
        boundary = (dilated - eroded).clamp(0, 1)
        return boundary

    def forward(
        self, spectral_residual: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spectral_residual: (B, C, H, W) spectral residual from LSAA.
            targets: (B, H, W) integer mask.
        Returns:
            Scalar auxiliary loss.
        """
        boundary = self._compute_boundary_mask(targets)  # (B, 1, H, W)
        # Average residual across channels
        r_s_mean = spectral_residual.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Resize if spatial sizes differ
        if r_s_mean.shape[2:] != boundary.shape[2:]:
            r_s_mean = F.interpolate(
                r_s_mean, size=boundary.shape[2:], mode="bilinear", align_corners=False
            )

        # L_aux = -log(sigmoid(R_s * M_boundary))
        product = r_s_mean * boundary
        loss = -F.logsigmoid(product)
        # Only average over boundary pixels
        num_boundary = boundary.sum().clamp(min=1)
        return (loss * boundary).sum() / num_boundary


class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + optional Auxiliary.

    Args:
        cfg: The 'loss' section from config.yaml.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.ce_weight = cfg.get("ce_weight", 1.0)
        self.dice_weight = cfg.get("dice_weight", 1.0)
        self.aux_weight = cfg.get("aux_weight", 0.0)

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.aux_loss = BoundaryAuxLoss(dilation=cfg.get("boundary_dilation", 3))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        spectral_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) segmentation logits.
            targets: (B, H, W) integer class labels.
            spectral_residual: (B, C, H, W) optional residual for aux loss.
        Returns:
            Scalar combined loss.
        """
        loss = self.ce_weight * self.ce_loss(logits, targets)
        loss = loss + self.dice_weight * self.dice_loss(logits, targets)

        if self.aux_weight > 0 and spectral_residual is not None:
            loss = loss + self.aux_weight * self.aux_loss(spectral_residual, targets)

        return loss
