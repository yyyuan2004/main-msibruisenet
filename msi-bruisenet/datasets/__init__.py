"""
datasets package
================
Custom PyTorch Dataset and spatial augmentation transforms for
9-band multispectral imaging (MSI) apple bruise segmentation.

- msi_dataset.py : MSIDataset class — loads .npy images/masks, normalises, augments
- transforms.py  : Spatial augmentation pipeline (flip, rotate, elastic, etc.)
"""

from .msi_dataset import MSIDataset

__all__ = ["MSIDataset"]
