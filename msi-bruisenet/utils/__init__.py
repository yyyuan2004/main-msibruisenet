"""
utils package
=============
Utility functions for MSI-BruiseNet:
- metrics.py           : IoU, Dice, F1, Precision, Recall; area-stratified evaluation
- visualize.py         : Attention maps, spectral heatmaps, prediction overlays
- spectral_analysis.py : Pre-experiment spectral correlation analysis
- seed.py              : Global random seed fixing for reproducibility
"""

from .seed import set_global_seed
from .metrics import compute_metrics

__all__ = ["set_global_seed", "compute_metrics"]
