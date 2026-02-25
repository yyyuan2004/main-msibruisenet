"""
loss.py - Loss functions for MSI bruise semantic segmentation.

Implements:
    - CrossEntropyLoss wrapper
    - DiceLoss (soft Dice for binary / multi-class)
    - SpectralResidualAuxLoss (boundary-aware spectral residual supervision)
    - CombinedLoss: L = w_ce * CE + w_dice * Dice + w_aux * AuxLoss

I/O:
    Input  : logits (B, C, H, W), targets (B, H, W), optional spectral_residual
    Output : scalar loss tensor
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
        """Compute Dice loss.

        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.

        Returns:
            Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute per-class Dice
        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice.mean()


class SpectralResidualAuxLoss(nn.Module):
    """Auxiliary loss for spectral residual supervision at boundary regions.

    L_aux = -log(sigmoid(R_s * M_boundary))

    where R_s is the spectral residual and M_boundary is a dilated boundary mask.

    Args:
        dilation: Number of pixels to dilate the boundary mask.
    """

    def __init__(self, dilation: int = 3) -> None:
        super().__init__()
        self.dilation = dilation

    def _compute_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute dilated boundary mask from segmentation targets.

        Args:
            targets: (B, H, W) integer labels.

        Returns:
            (B, 1, H, W) binary boundary mask.
        """
        # Use morphological gradient: dilate - erode
        t_float = targets.float().unsqueeze(1)  # (B, 1, H, W)
        k = 2 * self.dilation + 1
        kernel = torch.ones(1, 1, k, k, device=targets.device)
        dilated = F.conv2d(t_float, kernel, padding=self.dilation)
        eroded = -F.conv2d(-t_float, kernel, padding=self.dilation)
        boundary = ((dilated > 0) & (eroded <= 0)).float()
        return boundary

    def forward(
        self,
        spectral_residual: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral residual auxiliary loss.

        Args:
            spectral_residual: (B, C, H, W) from LSAA module.
            targets: (B, H, W) integer labels.

        Returns:
            Scalar auxiliary loss.
        """
        boundary = self._compute_boundary_mask(targets)  # (B, 1, H, W)
        # Pool spectral residual across channels
        r_s = spectral_residual.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Resize if needed
        if r_s.shape[2:] != boundary.shape[2:]:
            r_s = F.interpolate(r_s, size=boundary.shape[2:], mode="bilinear",
                                align_corners=False)

        # L_aux = -log(sigmoid(R_s * M_boundary)) at boundary pixels
        activated = torch.sigmoid(r_s * boundary)
        # Only compute loss at boundary pixels
        mask_sum = boundary.sum()
        if mask_sum > 0:
            loss = -(torch.log(activated + 1e-8) * boundary).sum() / mask_sum
        else:
            loss = torch.tensor(0.0, device=r_s.device)
        return loss


class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + optional spectral residual auxiliary.

    L = ce_weight * CE + dice_weight * Dice + aux_weight * AuxLoss

    Args:
        ce_weight: Weight for cross-entropy loss.
        dice_weight: Weight for Dice loss.
        aux_weight: Weight for auxiliary spectral residual loss (0 = disabled).
        boundary_dilation: Dilation pixels for boundary mask in aux loss.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.0,
        boundary_dilation: int = 3,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.aux_loss = SpectralResidualAuxLoss(dilation=boundary_dilation) if aux_weight > 0 else None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        spectral_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: (B, C, H, W) model output logits.
            targets: (B, H, W) ground-truth labels.
            spectral_residual: Optional (B, C', H', W') spectral residual for aux loss.

        Returns:
            Scalar combined loss.
        """
        loss = self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)

        if self.aux_weight > 0 and self.aux_loss is not None and spectral_residual is not None:
            loss = loss + self.aux_weight * self.aux_loss(spectral_residual, targets)

        return loss


def build_loss(cfg: Dict[str, Any]) -> CombinedLoss:
    """Build loss function from config.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        CombinedLoss instance.
    """
    lcfg = cfg["loss"]
    return CombinedLoss(
        ce_weight=lcfg.get("ce_weight", 1.0),
        dice_weight=lcfg.get("dice_weight", 1.0),
        aux_weight=lcfg.get("aux_weight", 0.0),
        boundary_dilation=lcfg.get("boundary_dilation", 3),
    )
