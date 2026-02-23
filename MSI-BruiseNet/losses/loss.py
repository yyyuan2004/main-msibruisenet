"""Loss functions for MSI bruise segmentation.

Includes CE + Dice and optional boundary-guided auxiliary loss.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """CE + Dice + optional auxiliary boundary loss."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.ce_weight = float(cfg["loss"]["ce_weight"])
        self.dice_weight = float(cfg["loss"]["dice_weight"])
        self.aux_weight = float(cfg["loss"]["aux_weight"])
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_f = target.float()
        inter = (probs * target_f).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        return 1 - ((2 * inter + eps) / (denom + eps)).mean()

    @staticmethod
    def aux_spectral_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bruise_prob = torch.softmax(logits, dim=1)[:, 1]
        boundary = target.float() - F.avg_pool2d(target.float().unsqueeze(1), 3, 1, 1).squeeze(1)
        boundary = boundary.abs().clamp(0, 1)
        score = (bruise_prob * boundary).mean()
        return -torch.log(torch.sigmoid(score) + 1e-6)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = self.dice_loss(logits, target)
        aux = self.aux_spectral_loss(logits, target) if self.aux_weight > 0 else torch.tensor(0.0, device=logits.device)
        return self.ce_weight * ce + self.dice_weight * dice + self.aux_weight * aux
