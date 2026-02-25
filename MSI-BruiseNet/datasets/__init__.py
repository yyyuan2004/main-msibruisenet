"""
datasets - Data loading and augmentation for MSI apple bruise segmentation.

Submodules:
    msi_dataset : Custom PyTorch Dataset for .npy MSI images and masks.
    transforms  : Spatial augmentation pipeline (flip, rotate, elastic, etc.).
"""

from .msi_dataset import MSIDataset, get_dataloaders  # noqa: F401

__all__ = ["MSIDataset", "get_dataloaders"]
