"""Loss functions for semantic segmentation.

Change log:
    - [v1] SpectralSmoothnessLoss: Total variation regularizer on predicted
      probability maps. Encourages smooth predictions, suppresses noise.
    - [v2 NEW] EdgePreservingLoss: 边缘感知正则。在GT边缘区域*最大化*预测梯度
      幅值，鼓励模型保留清晰的缺陷边界。与SpectralSmoothnessLoss互补：
        - SpectralSmoothnessLoss: 在平坦区域抑制噪声（最小化梯度）
        - EdgePreservingLoss:    在边缘区域保留细节（最大化梯度）
      两者结合 = 各向异性平滑（anisotropic smoothing）。
    - [v2 CHANGED] SegmentationLoss: 总损失变为
      loss = ce_w*CE + dice_w*Dice + ss_w*SpSmooth + edge_w*EdgePreserve
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


###############################################################################
# [NEW] Edge-Preserving Regularization Loss
#
# 来源: 改编自DualAnoDiff的边缘保持正则思想。
# 原理: 从GT mask提取边缘区域（膨胀-腐蚀），在边缘像素处*最大化*预测概率图
# 的空间梯度幅值。这迫使模型在缺陷边界处产生陡峭的置信度跳变，而非模糊过渡。
#
# 与SpectralSmoothnessLoss的互补关系:
#   - SpectralSmoothnessLoss: 全局TV正则 → 平坦区域抑制噪声
#   - EdgePreservingLoss: 边缘局部 → 保留清晰边界
#   两者结合 = 各向异性平滑: 平坦处光滑，边缘处锐利。
#
# 实现: 不依赖形态学操作（避免引入额外依赖），直接用GT mask的空间梯度
# 作为边缘指示器。GT梯度非零处即为边缘。
###############################################################################

class EdgePreservingLoss(nn.Module):
    """Edge-preserving regularization: maximize prediction gradients at GT edges.

    This loss extracts edges from the ground truth mask (via spatial gradient),
    and MAXIMIZES the predicted probability gradient magnitude at those edges.
    The result is sharper defect boundaries.

    The loss is negative (we maximize gradients), so it's negated before returning.
    """

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw predictions.
            targets: (B, H, W) integer class labels.

        Returns:
            Scalar edge-preserving loss (lower = sharper edges).
        """
        probs = F.softmax(logits, dim=1)

        # --- Step 1: 从GT mask提取边缘指示图 ---
        # targets的空间梯度: 类别标签变化处 = 边缘
        targets_float = targets.float().unsqueeze(1)  # (B, 1, H, W)
        edge_h = torch.abs(targets_float[:, :, 1:, :] - targets_float[:, :, :-1, :])
        edge_w = torch.abs(targets_float[:, :, :, 1:] - targets_float[:, :, :, :-1])
        # 二值化: 有梯度的地方就是边缘
        edge_h = (edge_h > 0).float()  # (B, 1, H-1, W)
        edge_w = (edge_w > 0).float()  # (B, 1, H, W-1)

        # --- Step 2: 计算预测概率的空间梯度幅值 ---
        grad_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])  # (B, C, H-1, W)
        grad_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])  # (B, C, H, W-1)

        # --- Step 3: 在边缘区域最大化梯度 ---
        # edge_h/w: (B, 1, ...) 会broadcast到 (B, C, ...)
        edge_grad_h = (grad_h * edge_h).sum() / (edge_h.sum() * probs.shape[1] + 1e-6)
        edge_grad_w = (grad_w * edge_w).sum() / (edge_w.sum() * probs.shape[1] + 1e-6)

        # 取负: 最大化梯度 = 最小化负梯度
        return -(edge_grad_h + edge_grad_w)


class SegmentationLoss(nn.Module):
    """Combined loss: CE/Focal + Dice + SpectralSmoothnessLoss + EdgePreservingLoss.

    [CHANGED v2] 总损失公式:
        loss = ce_w * CE + dice_w * Dice
             + ss_w * SpectralSmoothnessLoss    (平坦区域抑制噪声)
             + edge_w * EdgePreservingLoss       (边缘区域保留细节)  [NEW]

    Args:
        loss_type: "ce_dice" or "focal_dice".
        ce_weight: Weight for CE/Focal component.
        dice_weight: Weight for Dice component.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
        spectral_smoothness_weight: Weight for spectral smoothness regularizer.
            Default 0.0 (disabled). Recommended: 0.1 for mild regularization.
        edge_preserve_weight: Weight for edge-preserving regularizer. [NEW]
            Default 0.0 (disabled). Recommended: 0.05~0.1, too large causes
            high-frequency artifacts, too small has no effect. Start with 0.05.
    """

    def __init__(self, loss_type="ce_dice", ce_weight=0.5, dice_weight=0.5,
                 focal_gamma=2.0, focal_alpha=0.25,
                 spectral_smoothness_weight=0.0,
                 edge_preserve_weight=0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ss_weight = spectral_smoothness_weight
        # [NEW] 边缘感知正则权重
        self.edge_weight = edge_preserve_weight

        if loss_type == "focal_dice":
            self.ce_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss()

        if self.ss_weight > 0:
            self.spectral_smoothness = SpectralSmoothnessLoss()

        # [NEW] 边缘保持正则: 仅当权重>0时实例化
        if self.edge_weight > 0:
            self.edge_preserve = EdgePreservingLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        total = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        # 光谱平滑正则: 平坦区域抑制噪声
        if self.ss_weight > 0:
            loss_ss = self.spectral_smoothness(logits)
            total = total + self.ss_weight * loss_ss

        # [NEW] 边缘保持正则: 缺陷边界处保留清晰梯度
        if self.edge_weight > 0:
            loss_edge = self.edge_preserve(logits, targets)
            total = total + self.edge_weight * loss_edge

        return total
