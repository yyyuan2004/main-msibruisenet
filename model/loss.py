"""Loss functions for semantic segmentation.

Available losses:
    - DiceLoss: per-class Dice, returns 1 - mean_dice.
    - FocalLoss: focal modulation for class imbalance.
    - SegmentationLoss: combined CE/Focal + Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation.

    Computes per-class Dice and returns 1 - mean_dice.
    Supports multi-class via one-hot encoding.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # Sum over batch and spatial dims
        intersection = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Balancing factor. If float, used for the positive class.
               If None, no balancing is applied.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.
        """
        num_classes = logits.shape[1]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None and num_classes == 2:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class SegmentationLoss(nn.Module):
    """Combined segmentation loss: CE/Focal + Dice.

    loss = ce_weight * CE + dice_weight * Dice

    Args:
        loss_type: "ce_dice" or "focal_dice" or "focal".
        ce_weight: Weight for CE/Focal component.
        dice_weight: Weight for Dice component.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
    """

    def __init__(self, loss_type="ce_dice", ce_weight=0.5, dice_weight=0.5,
                 focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if loss_type == "focal_dice":
            self.ce_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        elif loss_type == "focal":
            self.ce_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
            self.dice_weight = 0.0
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
