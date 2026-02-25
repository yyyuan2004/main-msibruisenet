"""
utils - Utility functions for MSI-BruiseNet.

Submodules:
    metrics           : IoU, Dice, F1, Precision, Recall; area-stratified eval.
    visualize         : Attention maps, spectral heatmaps, prediction overlays.
    spectral_analysis : Band correlation matrices and spectral curve analysis.
    seed              : Global random seed fixing for reproducibility.
"""

from .seed import set_global_seed  # noqa: F401

__all__ = ["set_global_seed"]
