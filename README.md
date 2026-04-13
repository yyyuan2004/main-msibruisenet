[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/main-msibruisenet)
# MSI-Bruise-Baseline: MobileNetV2-UNet + Attention Module Ablation

Pixel-level semantic segmentation of apple defects using 9-channel near-infrared multispectral images (MSI), with ablation study on pluggable feature enhancement modules.

## Architecture

- **Encoder**: MobileNetV2 (ImageNet pretrained, first layer adapted for 9-channel input)
- **Decoder**: UNet-style with bilinear upsampling (no ConvTranspose2d)
- **Output**: Per-pixel class logits at full input resolution

## Ablation Configurations

| Config | Bottleneck | Skip Connection | Loss | Description |
|--------|-----------|-----------------|------|-------------|
| `baseline` | -- | Conv-BN-ReLU | CE+Dice | Pure MobileNetV2-UNet |
| `+SE` | -- | SE block | CE+Dice | Channel attention |
| `+1D-SpConv` | -- | Conv-BN-ReLU | CE+Dice | Spectral band modeling |
| `+ConvGLU` | -- | ConvGLU | CE+Dice | Channel mixer (TransNeXt) |
| `+1D-SpConv+SE` | -- | SE block | CE+Dice | Combined approach |
| `+ASPP` | ASPP | Conv-BN-ReLU | CE+Dice+SpSmooth | Multi-scale context |
| `+CBAM` | -- | CBAM | CE+Dice+SpSmooth | Channel+spatial attention |
| `fused` | ASPP | CBAM->ConvGLU | CE+Dice+SpSmooth+Edge | Full fusion |

## Data Setup

### Current data path (default)

数据集默认指向外部绝对路径，**不在项目目录内**:

```
/home/yy/datasets/153/
├── images/    # .npy spectral files, shape (H, W, 9), float32 reflectance
└── masks/     # .npy or .png mask files, shape (H, W), integer class labels
```

### How to change data path (换设备/换数据集时修改)

**只需修改 config YAML 文件中的 `data_dir` 字段**，所有 config 统一位于 `configs/` 目录下:

```yaml
# configs/baseline.yaml (以及 se.yaml, spconv.yaml, convglu.yaml, ... 等所有config)
data:
  data_dir: "/home/yy/datasets/153"   # <-- 修改这一行为你的数据集路径
  image_dir: "images"                  # images子目录名（通常不需要改）
  mask_dir: "masks"                    # masks子目录名（通常不需要改）
```

需要修改的文件列表（共8个config）:
- `configs/baseline.yaml`
- `configs/se.yaml`
- `configs/spconv.yaml`
- `configs/convglu.yaml`
- `configs/spconv_se.yaml`
- `configs/aspp.yaml`
- `configs/cbam.yaml`
- `configs/fused.yaml`

也可以用命令行一键替换所有config中的数据路径:
```bash
# 例如: 将数据路径从 /home/yy/datasets/153 改为 /data/apple_msi
sed -i 's|data_dir: ".*"|data_dir: "/data/apple_msi"|' configs/*.yaml
```

光谱预分析脚本也支持命令行指定数据路径:
```bash
python utils/spectral_analysis.py --data_dir /your/data/path
```

### Data format requirements

- File names must match between `images/` and `masks/` (e.g., `apple_001.npy` <-> `apple_001.npy` or `apple_001.png`)
- Image shape: (512, 512, 9) -- 9 NIR bands, float32 reflectance
- Mask values: 0=background, 1=defect (or multi-class: 0=bg, 1=type_A, 2=type_B, ...)
- Update `num_classes` in config files if using more than 2 classes

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Spectral Pre-analysis

```bash
python -m utils.spectral_analysis --data_dir /home/yy/datasets/153 --output_dir outputs/spectral_analysis
```

### 3. Train a Single Configuration

```bash
python train.py --config configs/baseline.yaml --seed 42
```

### 4. Run Full Ablation Study

```bash
bash run_ablation.sh
```

This runs all 8 configs x 3 seeds (24 training runs) + evaluation + result aggregation.

### 5. Evaluate a Trained Model

```bash
# Basic evaluation
python eval.py --checkpoint outputs/baseline_seed42/checkpoints/best_model.pth --seed 42 --split test

# With post-processing sharpening (guided filter)
python eval.py --checkpoint outputs/fused_seed42/checkpoints/best_model.pth --seed 42 --postprocess guided
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=1e-4 |
| Learning Rate | 1e-3, CosineAnnealing to 1e-6 |
| Epochs | 200 |
| Batch Size | 4 (reduce to 2 if OOM) |
| Loss | 0.5xCE + 0.5xDice (+ SpSmooth + EdgePreserve for fused) |
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
│   ├── baseline.yaml     # (data_dir is configured here)
│   ├── se.yaml
│   ├── spconv.yaml
│   ├── convglu.yaml
│   ├── spconv_se.yaml
│   ├── aspp.yaml
│   ├── cbam.yaml
│   └── fused.yaml        # ASPP + CBAM->ConvGLU + dual regularization
├── data/
│   ├── dataset.py        # MSIDataset: loads 9ch .npy + masks
│   ├── augment.py        # Spatial augmentations (image+mask synchronized)
│   └── split.py          # Train/val/test splitting utilities
├── model/
│   ├── encoder.py        # MobileNetV2 encoder (9ch adapted)
│   ├── decoder.py        # UNet decoder blocks (supports cbam_convglu fusion)
│   ├── modules.py        # SE / CBAM / ASPP / SpectralConv1D / ConvGLU
│   ├── model.py          # Full model assembly
│   └── loss.py           # CE+Dice / Focal+Dice / SpSmooth / EdgePreserve
├── utils/
│   ├── metrics.py        # Segmentation metrics (mIoU, F1, Precision, Recall)
│   ├── postprocess.py    # Inference-time sharpening (unsharp mask / guided filter)
│   └── spectral_analysis.py  # Spectral pre-analysis script
├── train.py              # Training loop
├── eval.py               # Evaluation + visualization + optional post-processing
├── aggregate_results.py  # Collect results into ablation table
├── run_ablation.sh       # One-click ablation runner
└── requirements.txt      # Python dependencies
```

## Notes

- **No color/brightness augmentation**: spectral reflectance values have physical meaning
- **ConvGLU is a channel mixer**, not an attention module (from TransNeXt, CVPR 2024)
- **1D-SpConv is inserted once** after encoder S1, not at every skip connection
- **Extreme low-sample regime** (~153 samples): module differences may be within statistical noise -- this is itself a valuable finding
- If OOM on 6GB GPU: reduce batch_size to 2 or crop_size to 384
