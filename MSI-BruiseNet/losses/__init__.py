"""
losses - Loss functions for MSI bruise segmentation.

Provides:
    - CombinedLoss: CE + Dice + optional spectral residual auxiliary loss.
"""

from .loss import CombinedLoss, build_loss  # noqa: F401

__all__ = ["CombinedLoss", "build_loss"]
