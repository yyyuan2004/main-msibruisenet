"""
models package
==============
Contains all model components for MSI-BruiseNet:
- backbone.py   : MobileNetV2 encoder adapted for 9-channel MSI input
- decoder.py    : UNet-style decoder with skip connections
- attention.py  : LSAA, ConvGLU, SE, CBAM, ECA modules for ablation
- build_model.py: Factory function to assemble the full MSI-BruiseNet
"""

from .build_model import build_model

__all__ = ["build_model"]
