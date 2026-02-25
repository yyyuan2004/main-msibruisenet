"""
models - MSI-BruiseNet model components.

Submodules:
    backbone   : MobileNetV2 encoder adapted for 9-channel MSI input.
    decoder    : UNet-style decoder with bilinear upsampling + skip fusion.
    attention  : LSAA, SE, CBAM, ECA, and Identity attention modules.
    build_model: Factory function that assembles the full MSI-BruiseNet.
"""

from .build_model import build_model  # noqa: F401

__all__ = ["build_model"]
