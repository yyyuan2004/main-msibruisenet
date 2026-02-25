# Data Directory

> **USER ACTION REQUIRED**: Place your MSI data here.

## Structure

```
data/
├── images/          # (H, W, 9) float32 .npy multispectral images
├── masks/           # (H, W) uint8 .npy binary masks (0=background, 1=bruise)
├── splits/          # Auto-generated: 5-fold cross-validation JSON
└── norm_stats.json  # Auto-generated: per-channel mean/std
```

## Image Specification

- **Format**: NumPy `.npy` files
- **Shape**: `(H, W, 9)` — height x width x 9 spectral bands
- **Bands**: 713, 736, 759, 782, 805, 828, 851, 874, 897, 920 nm
- **dtype**: `float32` (recommended) or `uint16`
- **Naming**: `sample_001.npy`, `sample_002.npy`, ...

## Mask Specification

- **Format**: NumPy `.npy` files
- **Shape**: `(H, W)` — must match the corresponding image dimensions
- **dtype**: `uint8`
- **Values**: `0` = background, `1` = bruise
- **Naming**: Must match image filenames exactly

## Quick Test

Generate dummy data for smoke testing:
```bash
python scripts/generate_dummy_data.py --config configs/config.yaml
```
