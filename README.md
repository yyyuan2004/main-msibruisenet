# MSI-Bruise-Baseline: MobileNetV2-UNet + Attention Module Ablation

Pixel-level semantic segmentation of apple defects using 9-channel near-infrared multispectral images (MSI), with ablation study on pluggable feature enhancement modules.

## Architecture

- **Encoder**: MobileNetV2 (ImageNet pretrained, first layer adapted for 9-channel input)
- **Decoder**: UNet-style with bilinear upsampling (no ConvTranspose2d)
- **Output**: Per-pixel class logits at full input resolution

## Ablation Configurations

| Config | Encoder After S1 | Skip Connection | Description |
|--------|-----------------|-----------------|-------------|
| `baseline` | None | Standard Conv-BN-ReLU | Pure MobileNetV2-UNet |
| `+SE` | None | SE block each level | Channel attention baseline |
| `+1D-SpConv` | SpectralConv1D | None | Spectral band modeling |
| `+ConvGLU` | None | ConvGLU replaces Conv-BN-ReLU | Channel mixer (TransNeXt) |
| `+1D-SpConv+SE` | SpectralConv1D | SE block each level | Combined approach |

## Data Setup

Place your data in the following structure:

```
data/
├── images/    # .npy spectral files, shape (H, W, 9), float32 reflectance
└── masks/     # .npy or .png mask files, shape (H, W), integer class labels
```

- File names must match between `images/` and `masks/` (e.g., `apple_001.npy` <-> `apple_001.npy` or `apple_001.png`)
- Image shape: (512, 512, 9) — 9 NIR bands
- Mask values: 0=background, 1=defect (or multi-class: 0=bg, 1=type_A, 2=type_B, ...)
- Update `num_classes` in config files if using more than 2 classes

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Spectral Pre-analysis

Run this first to understand your data:

```bash
python utils/spectral_analysis.py --data_dir data --output_dir outputs/spectral_analysis
```

Outputs:
- Band correlation heatmaps (normal vs defect)
- Mean spectral curves with std bands
- PCA cumulative variance explained
- 3-band linear regression R2 table

### 3. Train a Single Configuration

```bash
python train.py --config configs/baseline.yaml --seed 42
```

### 4. Run Full Ablation Study

```bash
bash run_ablation.sh
```

This runs all 5 configs x 3 seeds (15 training runs) + evaluation + result aggregation.

### 5. Evaluate a Trained Model

```bash
python eval.py --checkpoint outputs/baseline_seed42/checkpoints/best_model.pth --seed 42 --split test
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=1e-4 |
| Learning Rate | 1e-3, CosineAnnealing to 1e-6 |
| Epochs | 200 |
| Batch Size | 4 (reduce to 2 if OOM) |
| Loss | 0.5xCE + 0.5xDice (or Focal+Dice) |
| Input Size | 512x512, random crop to 384x384 |
| Augmentation | H-flip, V-flip, 90/180/270 degree rotation |

## Evaluation Metrics

- **mIoU** (primary metric)
- F1-score (per-class + macro)
- Precision / Recall (per-class + macro)

All reported as mean +/- std over 3 random seeds.

## Project Structure

```
├── configs/              # YAML configurations for each ablation variant
│   ├── baseline.yaml
│   ├── se.yaml
│   ├── spconv.yaml
│   ├── convglu.yaml
│   └── spconv_se.yaml
├── data/
│   ├── dataset.py        # MSIDataset: loads 9ch .npy + masks
│   ├── augment.py        # Spatial augmentations (image+mask synchronized)
│   └── split.py          # Train/val/test splitting utilities
├── model/
│   ├── encoder.py        # MobileNetV2 encoder (9ch adapted)
│   ├── decoder.py        # UNet decoder blocks
│   ├── modules.py        # SE / SpectralConv1D / ConvGLU implementations
│   ├── model.py          # Full model assembly
│   └── loss.py           # CE+Dice / Focal+Dice losses
├── utils/
│   ├── metrics.py        # Segmentation metrics (mIoU, F1, Precision, Recall)
│   └── spectral_analysis.py  # Spectral pre-analysis script
├── train.py              # Training loop
├── eval.py               # Evaluation + visualization
├── aggregate_results.py  # Collect results into ablation table
├── run_ablation.sh       # One-click ablation runner
└── requirements.txt      # Python dependencies
```

## Notes

- **No color/brightness augmentation**: spectral reflectance values have physical meaning
- **ConvGLU is a channel mixer**, not an attention module (from TransNeXt, CVPR 2024)
- **1D-SpConv is inserted once** after encoder S1, not at every skip connection
- **Extreme low-sample regime** (~153 samples): module differences may be within statistical noise — this is itself a valuable finding
- If OOM on 6GB GPU: reduce batch_size to 2 or crop_size to 384
