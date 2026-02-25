"""
losses package
==============
Combined loss functions for MSI-BruiseNet training:
- Cross-Entropy Loss
- Dice Loss (class imbalance)
- Spectral Residual Auxiliary Loss (optional boundary supervision)
"""

from .loss import CombinedLoss

__all__ = ["CombinedLoss"]
