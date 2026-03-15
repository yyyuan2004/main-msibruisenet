"""Loss functions for semantic segmentation.

Change log (Spectral Smoothness Regularization):
    - [NEW] SpectralSmoothnessLoss: Penalizes large differences between
      predictions at adjacent spectral bands' feature maps. This is a
      REGULARIZATION TERM (not a standalone loss), added to the existing
      CE + Dice loss to encourage spatially smooth predictions that respect
      the physical smoothness of NIR spectral reflectance.
    - [CHANGED] SegmentationLoss: Added optional spectral_smoothness_weight
      parameter. When > 0, the spectral smoothness regularizer is included:
      loss = ce_w * CE + dice_w * Dice + ss_w * SpectralSmoothnessLoss
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
            # Apply alpha weighting for binary case
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


###############################################################################
# [NEW] Spectral Smoothness Regularization Loss
#
# Physical motivation: In 9-band NIR multispectral imaging (23nm spacing),
# adjacent bands are highly correlated. The model's prediction should not
# change drastically when the input varies only slightly across adjacent
# spectral bands. This regularizer operates on the INPUT spectral image,
# penalizing sharp gradients of the predicted soft probabilities when
# weighted by the spectral smoothness of the input.
#
# Implementation: For each pixel, we compute the L2 norm of the difference
# between the model's soft predictions at neighboring spatial positions,
# weighted by how similar those positions are in spectral space. This
# encourages the segmentation boundary to align with genuine spectral
# discontinuities (i.e., defect boundaries) rather than noise.
#
# Simplified version used here: Penalize the total variation of predicted
# probabilities (spatial gradient smoothness), which acts as a generic
# smoothness prior suitable for the extreme low-sample scenario.
###############################################################################

class SpectralSmoothnessLoss(nn.Module):
    """Spectral smoothness regularization for MSI segmentation.

    Encourages smooth predictions by penalizing the total variation
    (spatial gradient magnitude) of the predicted probability maps.
    This is especially beneficial for few-shot MSI data where the model
    might overfit to noisy spectral patterns.

    loss = mean(|P[:,:,i+1,j] - P[:,:,i,j]|^2 + |P[:,:,i,j+1] - P[:,:,i,j]|^2)

    where P = softmax(logits) is the predicted probability map.
    """

    def forward(self, logits, targets=None):
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: Not used — included for API consistency with other losses.

        Returns:
            Scalar total variation loss over predicted probabilities.
        """
        probs = F.softmax(logits, dim=1)

        # Spatial gradients along height and width
        diff_h = probs[:, :, 1:, :] - probs[:, :, :-1, :]  # (B, C, H-1, W)
        diff_w = probs[:, :, :, 1:] - probs[:, :, :, :-1]  # (B, C, H, W-1)

        # L2 penalty (mean squared gradient)
        loss = (diff_h ** 2).mean() + (diff_w ** 2).mean()
        return loss


class SegmentationLoss(nn.Module):
    """Combined loss: CE (or Focal) + Dice + optional Spectral Smoothness.

    [CHANGED] Added spectral_smoothness_weight parameter. When > 0, the loss
    becomes: loss = ce_w * CE + dice_w * Dice + ss_w * SpectralSmoothness

    Args:
        loss_type: "ce_dice" or "focal_dice".
        ce_weight: Weight for CE/Focal component.
        dice_weight: Weight for Dice component.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
        spectral_smoothness_weight: Weight for spectral smoothness regularizer.
            Default 0.0 (disabled). Recommended: 0.1 for mild, 0.3 for strong. [NEW]
    """

    def __init__(self, loss_type="ce_dice", ce_weight=0.5, dice_weight=0.5,
                 focal_gamma=2.0, focal_alpha=0.25,
                 spectral_smoothness_weight=0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        # [NEW] Spectral smoothness regularization weight
        self.ss_weight = spectral_smoothness_weight

        if loss_type == "focal_dice":
            self.ce_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss()

        # [NEW] Only instantiate if weight > 0
        if self.ss_weight > 0:
            self.spectral_smoothness = SpectralSmoothnessLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        total = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        # [NEW] Add spectral smoothness regularization if configured
        if self.ss_weight > 0:
            loss_ss = self.spectral_smoothness(logits)
            total = total + self.ss_weight * loss_ss

        return total
