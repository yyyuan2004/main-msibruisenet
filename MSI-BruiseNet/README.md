# MSI-BruiseNet

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**9-Band Multispectral Apple Bruise Semantic Segmentation**

---

## Abstract

MSI-BruiseNet is a lightweight semantic segmentation network designed for detecting and localizing bruise damage on apples using 9-band snapshot multispectral imaging (MSI) in the 713–920 nm near-infrared range. The architecture combines a MobileNetV2 encoder (adapted for 9-channel input) with a UNet-style decoder, enhanced by a novel **Local Spectral Anomaly Attention (LSAA)** module and **ConvGLU** gated skip-connection fusion. LSAA exploits local spectral residuals to highlight anomalous bruise signatures, while ConvGLU provides learnable gating for encoder-decoder feature fusion. The project supports 5-fold cross-validation with multiple random seeds, comprehensive ablation experiments, and area-stratified evaluation — all designed to run on a single laptop GPU (RTX 3060, 6 GB VRAM).

---

## Architecture

```
Input: (B, 9, H, W)  ← 9-channel MSI .npy
  │
  ├── MobileNetV2 Encoder (first conv: 9→32 ch, rest: ImageNet pretrained)
  │     ├── Stage 0: F1 (1/2,  16 ch)
  │     ├── Stage 1: F2 (1/4,  24 ch)
  │     ├── Stage 2: F3 (1/8,  32 ch)
  │     └── Stage 3: F4 (1/16, 96 ch)
  │
  ├── Skip Connections (at each level):
  │     F_skip = ConvGLU( LSAA(F_encoder), F_decoder_up )
  │
  │     LSAA:  F_bar = AvgPool(F_e, k=5)          ← local background
  │            R_s   = F_e - F_bar                  ← spectral residual
  │            W_a   = σ(BN(Conv3x3(ReLU(BN(Conv1x1(R_s))))))
  │            F_out = F_e * W_a + F_e              ← modulation + bypass
  │
  │     ConvGLU: G = σ(Conv3x3(cat(F_lsaa, F_dec)))
  │              F_out = G * F_lsaa + (1-G) * F_dec
  │
  ├── UNet Decoder (bilinear upsample + skip fusion + 2× Conv-BN-ReLU)
  │
  └── 1×1 Conv → (B, 2, H, W) logits (background / bruise)
```

---

## Installation

```bash
git clone <this-repo-url>
cd MSI-BruiseNet
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.12+, CUDA 11.x (optional, for GPU training).

---

## Data Preparation

### ⚠️ USER ACTION REQUIRED

You must provide your own MSI data. Place files in the `data/` directory following this structure:

```
data/
├── images/                    # ⚠️ Place your .npy images here
│   ├── sample_001.npy         #    Shape: (H, W, 9), dtype: float32
│   ├── sample_002.npy
│   └── ...
├── masks/                     # ⚠️ Place your .npy masks here
│   ├── sample_001.npy         #    Shape: (H, W), dtype: uint8, values: {0, 1}
│   ├── sample_002.npy
│   └── ...
├── splits/                    # Auto-generated (do not modify manually)
│   └── splits.json
└── norm_stats.json            # Auto-generated
```

### Image Format

| Property | Specification |
|----------|---------------|
| Format | `.npy` (NumPy binary array) |
| Shape | `(H, W, 9)` — height × width × 9 spectral bands |
| dtype | `float32` (recommended) or `uint16` |
| Bands | 713, 736, 759, 782, 805, 828, 851, 874, 897, 920 nm |
| Naming | `sample_001.npy`, `sample_002.npy`, ... |

### Mask Format

| Property | Specification |
|----------|---------------|
| Format | `.npy` (NumPy binary array) |
| Shape | `(H, W)` — must match corresponding image |
| dtype | `uint8` |
| Values | `0` = background, `1` = bruise |
| Naming | Must exactly match corresponding image filename |

### Smoke Test with Dummy Data

To verify the pipeline works without real data:

```bash
python scripts/generate_dummy_data.py --num-samples 20 --height 256 --width 256
```

---

## Configuration

All hyperparameters are managed in `configs/config.yaml`. Key fields you may need to modify:

| Field | Description | Default |
|-------|-------------|---------|
| `data.image_dir` | Path to MSI image .npy files | `data/images/` |
| `data.mask_dir` | Path to mask .npy files | `data/masks/` |
| `data.input_size` | Training resize resolution | `512` |
| `train.batch_size` | Batch size (reduce if OOM) | `4` |
| `train.epochs` | Total training epochs | `200` |
| `train.lr` | Learning rate | `1e-4` |
| `model.attention` | Attention module: `lsaa`/`se`/`cbam`/`eca`/`none` | `lsaa` |
| `evaluation.pixel_per_cm` | Camera calibration factor | `50.0` |

Override any config value from the command line:

```bash
python scripts/train.py --config configs/config.yaml --override train.batch_size=2 model.attention=se
```

---

## Quick Start

```bash
# Step 0: (Optional) Generate dummy data for testing
python scripts/generate_dummy_data.py --num-samples 20

# Step 1: Compute per-channel normalization statistics
python scripts/compute_norm_stats.py

# Step 2: Generate 5-fold cross-validation splits
python scripts/prepare_splits.py

# Step 3: (Optional) Spectral pre-analysis
python utils/spectral_analysis.py  # or use notebooks/spectral_preanalysis.ipynb

# Step 4: Train (5-fold × 3 seeds = 15 runs)
python scripts/train.py --config configs/config.yaml

# Step 5: Evaluate
python scripts/evaluate.py --config configs/config.yaml \
    --checkpoint outputs/checkpoints/default_fold0_seed42/best.pth \
    --save-predictions
```

### Training a Single Fold

```bash
python scripts/train.py --config configs/config.yaml --fold 0 --seed-idx 0 --tag quick_test
```

### Resume Training

```bash
python scripts/train.py --config configs/config.yaml \
    --resume outputs/checkpoints/default_fold0_seed42/latest.pth
```

---

## Ablation & Baselines

### Ablation Experiments

```bash
bash scripts/ablation.sh
```

This runs:
1. **Attention type ablation:** LSAA vs SE vs CBAM vs ECA vs None
2. **LSAA kernel size:** k = 3, 5, 7, 9
3. **Fusion method:** ConvGLU vs simple concatenation
4. **Bypass residual:** with vs without

### Baseline Comparison

```bash
bash scripts/baseline.sh
```

Compares: UNet (vanilla) / UNet+SE / UNet+CBAM / UNet+ECA / MSI-BruiseNet (ours).

---

## Output Structure

```
outputs/
├── checkpoints/               # Model weights (.pth)
│   └── <tag>_fold<N>_seed<S>/
│       ├── best.pth           # Best validation mIoU
│       └── latest.pth         # Latest checkpoint
├── logs/                      # TensorBoard logs
│   └── <tag>_fold<N>_seed<S>/
├── predictions/               # Inference results
│   └── <tag>/
│       ├── sample_001_pred.npy
│       └── sample_001_overlay.png
├── results/                   # Metrics summaries
│   ├── metrics_summary_<tag>.csv
│   ├── metrics_by_area_<tag>.csv
│   ├── aggregated_<tag>.json
│   └── model_complexity_<tag>.json
└── spectral_analysis/         # Spectral pre-analysis
    ├── corr_normal.png
    ├── corr_bruise.png
    ├── corr_delta.png
    ├── spectral_curves.png
    └── spectral_analysis.json
```

---

## Key File-Path Reference Table

| What | Where in Code | Config Key | Default Path |
|------|---------------|------------|--------------|
| MSI images (.npy) | `datasets/msi_dataset.py` L63 | `data.image_dir` | `data/images/` |
| Masks (.npy) | `datasets/msi_dataset.py` L64 | `data.mask_dir` | `data/masks/` |
| Fold splits | `datasets/msi_dataset.py` L138 | `data.split_dir` | `data/splits/` |
| Norm stats | `datasets/msi_dataset.py` L175 | `data.norm_stats` | `data/norm_stats.json` |
| Checkpoints | `scripts/train.py` L282 | `output.checkpoints` | `outputs/checkpoints/` |
| TensorBoard logs | `scripts/train.py` L283 | `output.logs` | `outputs/logs/` |
| Predictions | `scripts/evaluate.py` L163 | `output.predictions` | `outputs/predictions/` |
| Metrics CSV | `scripts/evaluate.py` L164 | `output.results` | `outputs/results/` |

---

## Project Structure

```
MSI-BruiseNet/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusions
├── configs/
│   └── config.yaml                 # Centralized hyperparameters
├── data/                           # ⚠️ User data (see Data Preparation)
│   ├── images/                     # .npy MSI images
│   ├── masks/                      # .npy binary masks
│   ├── splits/                     # 5-fold split JSON
│   └── README_DATA.md              # Data format documentation
├── models/
│   ├── __init__.py
│   ├── backbone.py                 # MobileNetV2 encoder (9-ch adapted)
│   ├── decoder.py                  # UNet decoder
│   ├── attention.py                # LSAA / SE / CBAM / ECA / Identity
│   └── build_model.py              # Model factory
├── datasets/
│   ├── __init__.py
│   ├── msi_dataset.py              # Dataset: load .npy, normalize, augment
│   └── transforms.py               # Spatial augmentations
├── losses/
│   ├── __init__.py
│   └── loss.py                     # CE + Dice + optional spectral aux loss
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # IoU / Dice / F1 / area-stratified eval
│   ├── visualize.py                # Prediction overlays, attention heatmaps
│   ├── spectral_analysis.py        # Band correlation analysis
│   └── seed.py                     # Global seed fixing
├── scripts/
│   ├── train.py                    # Training entry (5-fold × 3 seeds)
│   ├── evaluate.py                 # Inference + metrics + visualization
│   ├── prepare_splits.py           # Generate 5-fold split JSON
│   ├── compute_norm_stats.py       # Compute 9-channel mean/std
│   ├── generate_dummy_data.py      # Synthetic data for smoke testing
│   ├── ablation.sh                 # Ablation experiment batch script
│   └── baseline.sh                 # Baseline comparison batch script
├── outputs/                        # Auto-generated training outputs
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── results/
└── notebooks/
    └── spectral_preanalysis.ipynb  # Spectral analysis Jupyter notebook
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{msi_bruisenet2024,
  title   = {MSI-BruiseNet: Local Spectral Anomaly Attention for Multispectral Apple Bruise Segmentation},
  author  = {Your Name},
  year    = {2024},
  note    = {GitHub repository},
  url     = {https://github.com/your-username/MSI-BruiseNet}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---
---

## 中文说明 (Chinese README)

### 项目简介

MSI-BruiseNet 是一个轻量级语义分割网络，专为利用 9 波段快照式多光谱成像（MSI，713–920 nm 近红外波段）检测和定位苹果瘀伤损伤而设计。该架构结合了 MobileNetV2 编码器（适配 9 通道输入）和 UNet 风格解码器，并通过新颖的**局部光谱异常注意力（LSAA）**模块和 **ConvGLU** 门控跳跃连接融合进行增强。LSAA 利用局部光谱残差突出异常瘀伤信号，ConvGLU 提供可学习的编码器-解码器特征融合门控。项目支持 5 折交叉验证 × 多随机种子、全面的消融实验和按面积分层评估——所有功能均可在单块笔记本 GPU（RTX 3060，6 GB 显存）上运行。

---

### 网络架构

```
输入: (B, 9, H, W)  ← 9通道 MSI .npy
  │
  ├── MobileNetV2 编码器（首层: 9→32 通道，其余加载 ImageNet 预训练权重）
  │     ├── Stage 0: F1 (1/2,  16 ch)
  │     ├── Stage 1: F2 (1/4,  24 ch)
  │     ├── Stage 2: F3 (1/8,  32 ch)
  │     └── Stage 3: F4 (1/16, 96 ch)
  │
  ├── 每级跳跃连接:
  │     F_skip = ConvGLU( LSAA(F_encoder), F_decoder_up )
  │
  │     LSAA:  局部背景估计 → 谱残差 → 异常权重 → 特征调制 + 残差
  │     ConvGLU: 门控融合编码器和解码器特征
  │
  ├── UNet 解码器（双线性上采样 + 跳跃融合 + 2× Conv-BN-ReLU）
  │
  └── 1×1 卷积 → (B, 2, H, W) 预测（背景 / 瘀伤）
```

---

### 安装

```bash
git clone <仓库地址>
cd MSI-BruiseNet
pip install -r requirements.txt
```

**环境要求：** Python 3.8+，PyTorch 1.12+，CUDA 11.x（可选，用于 GPU 训练）。

---

### 数据准备

#### ⚠️ 需要用户操作

您需要提供自己的 MSI 数据，并按以下结构放置在 `data/` 目录下：

```
data/
├── images/                    # ⚠️ 将 .npy 图像放在这里
│   ├── sample_001.npy         #    形状: (H, W, 9)，类型: float32
│   ├── sample_002.npy
│   └── ...
├── masks/                     # ⚠️ 将 .npy 掩码放在这里
│   ├── sample_001.npy         #    形状: (H, W)，类型: uint8，值: {0, 1}
│   ├── sample_002.npy
│   └── ...
├── splits/                    # 自动生成（无需手动修改）
│   └── splits.json
└── norm_stats.json            # 自动生成
```

#### 图像格式详解

`.npy` 是 NumPy 的二进制存储格式，可以高效地保存多维数组。

**创建 .npy 文件的方法：**

```python
import numpy as np

# 假设你有一个 9 波段的多光谱图像数组
# image 的形状应为 (高度, 宽度, 9)
image = np.zeros((512, 512, 9), dtype=np.float32)  # 示例

# 保存为 .npy
np.save("data/images/sample_001.npy", image)

# 读取验证
loaded = np.load("data/images/sample_001.npy")
print(loaded.shape)  # (512, 512, 9)
print(loaded.dtype)  # float32
```

| 属性 | 图像文件 | 掩码文件 |
|------|---------|---------|
| 格式 | `.npy` | `.npy` |
| 形状 | `(H, W, 9)` | `(H, W)` |
| 数据类型 | `float32`（推荐）或 `uint16` | `uint8` |
| 像素值 | 反射率值 | `0`=背景，`1`=瘀伤 |
| 命名 | `sample_001.npy` | 与图像文件**完全一致** |

#### 使用假数据进行冒烟测试

无需真实数据即可验证流程是否跑通：

```bash
python scripts/generate_dummy_data.py --num-samples 20 --height 256 --width 256
```

---

### 配置说明

所有超参数集中在 `configs/config.yaml` 管理。需要用户修改的关键字段：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `data.image_dir` | MSI 图像 .npy 文件路径 | `data/images/` |
| `data.mask_dir` | 掩码 .npy 文件路径 | `data/masks/` |
| `data.input_size` | 训练时图像缩放尺寸 | `512` |
| `train.batch_size` | 批次大小（显存不够时调小） | `4` |
| `train.epochs` | 总训练轮数 | `200` |
| `model.attention` | 注意力模块选择 | `lsaa` |
| `evaluation.pixel_per_cm` | 相机标定系数（像素/厘米） | `50.0` |

命令行覆盖配置：

```bash
python scripts/train.py --config configs/config.yaml --override train.batch_size=2
```

---

### 快速开始

```bash
# 第 0 步：（可选）生成假数据用于测试
python scripts/generate_dummy_data.py --num-samples 20

# 第 1 步：计算各通道归一化统计量
python scripts/compute_norm_stats.py

# 第 2 步：生成 5 折交叉验证划分
python scripts/prepare_splits.py

# 第 3 步：（可选）光谱预分析
# 使用命令行或 Jupyter Notebook
# python utils/spectral_analysis.py
# 或打开 notebooks/spectral_preanalysis.ipynb

# 第 4 步：训练（5 折 × 3 随机种子 = 15 次实验）
python scripts/train.py --config configs/config.yaml

# 第 5 步：评估
python scripts/evaluate.py --config configs/config.yaml \
    --checkpoint outputs/checkpoints/default_fold0_seed42/best.pth \
    --save-predictions
```

#### 训练单个 fold

```bash
python scripts/train.py --config configs/config.yaml --fold 0 --seed-idx 0 --tag quick_test
```

#### 断点续训

```bash
python scripts/train.py --config configs/config.yaml \
    --resume outputs/checkpoints/default_fold0_seed42/latest.pth
```

---

### 消融实验与基线对比

#### 消融实验

```bash
bash scripts/ablation.sh
```

包含以下消融：
1. **注意力模块消融：** LSAA / SE / CBAM / ECA / 无注意力
2. **LSAA 窗口大小：** k = 3, 5, 7, 9
3. **融合方式：** ConvGLU vs 简单拼接
4. **残差连接：** 有 vs 无 bypass

#### 基线对比

```bash
bash scripts/baseline.sh
```

对比方法：UNet（原始）/ UNet+SE / UNet+CBAM / UNet+ECA / MSI-BruiseNet（本文方法）。

---

### 输出目录结构

```
outputs/
├── checkpoints/               # 模型权重 (.pth)
│   └── <标签>_fold<N>_seed<S>/
│       ├── best.pth           # 最佳验证 mIoU 的权重
│       └── latest.pth         # 最新检查点
├── logs/                      # TensorBoard 日志
├── predictions/               # 推理结果
│   └── <标签>/
│       ├── sample_001_pred.npy     # 预测掩码
│       └── sample_001_overlay.png  # 可视化叠加图
├── results/                   # 指标汇总
│   ├── metrics_summary_<标签>.csv     # 各次实验的指标
│   ├── metrics_by_area_<标签>.csv     # 按面积分层的 IoU
│   ├── aggregated_<标签>.json         # 均值 ± 标准差
│   └── model_complexity_<标签>.json   # 参数量 / FLOPs / FPS
└── spectral_analysis/         # 光谱预分析结果
```

---

### 关键文件路径参考表

| 内容 | 代码位置 | 配置键 | 默认路径 |
|------|---------|--------|---------|
| MSI 图像 | `datasets/msi_dataset.py` L63 | `data.image_dir` | `data/images/` |
| 掩码文件 | `datasets/msi_dataset.py` L64 | `data.mask_dir` | `data/masks/` |
| 交叉验证划分 | `datasets/msi_dataset.py` L138 | `data.split_dir` | `data/splits/` |
| 归一化统计 | `datasets/msi_dataset.py` L175 | `data.norm_stats` | `data/norm_stats.json` |
| 模型权重 | `scripts/train.py` L282 | `output.checkpoints` | `outputs/checkpoints/` |
| TensorBoard 日志 | `scripts/train.py` L283 | `output.logs` | `outputs/logs/` |
| 预测结果 | `scripts/evaluate.py` L163 | `output.predictions` | `outputs/predictions/` |
| 指标 CSV | `scripts/evaluate.py` L164 | `output.results` | `outputs/results/` |

---

### 常见问题 FAQ

#### 1. 如何将 TIFF/ENVI 格式的多光谱图像转为 .npy？

```python
import numpy as np

# === TIFF 格式 ===
from PIL import Image
# 如果是多页 TIFF（每页一个波段）
bands = []
for band_idx in range(9):
    img = Image.open(f"band_{band_idx}.tif")
    bands.append(np.array(img))
msi_image = np.stack(bands, axis=-1).astype(np.float32)  # (H, W, 9)
np.save("data/images/sample_001.npy", msi_image)

# === ENVI 格式 ===
# 需要安装 spectral 库: pip install spectral
import spectral
img = spectral.open_image("image.hdr")
data = img.load()  # 加载为 numpy 数组
# 选择需要的 9 个波段（假设波段索引已知）
band_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
msi_image = data[:, :, band_indices].astype(np.float32)
np.save("data/images/sample_001.npy", msi_image)
```

#### 2. 图像尺寸不一致怎么办？

不需要手动统一尺寸。训练时会自动将所有图像 resize 到 `config.yaml` 中设定的 `data.input_size`（默认 512×512）。不同尺寸的原始图像可以混合使用。

#### 3. 显存不够（GPU Out of Memory）怎么办？

按优先级依次尝试：
1. **降低 batch_size：** `--override train.batch_size=2`（甚至 `=1`）
2. **降低输入分辨率：** `--override data.input_size=256`
3. **关闭弹性变形增强：** `--override augmentation.elastic_transform=false`

推荐的 RTX 3060 (6GB) 配置：`batch_size=4`, `input_size=512`。

#### 4. 没有 GPU 怎么办？

项目支持 CPU 训练，代码会自动检测：
- 有 GPU → 使用 CUDA
- 无 GPU → 使用 CPU（训练速度会显著降低，建议减少 epochs 和 input_size）

```bash
# CPU 训练示例（缩小规模以加速）
python scripts/train.py --config configs/config.yaml \
    --override train.batch_size=2 data.input_size=256 train.epochs=50
```

---

### 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@misc{msi_bruisenet2024,
  title   = {MSI-BruiseNet: Local Spectral Anomaly Attention for Multispectral Apple Bruise Segmentation},
  author  = {Your Name},
  year    = {2024},
  note    = {GitHub repository},
  url     = {https://github.com/your-username/MSI-BruiseNet}
}
```

### 许可证

本项目基于 MIT 许可证发布。
