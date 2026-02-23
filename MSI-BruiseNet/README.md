# MSI-BruiseNet

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A lightweight PyTorch project for **9-band snapshot multispectral imaging (MSI) apple bruise semantic segmentation**, designed for complete training/inference on edge-friendly hardware.

## Abstract
MSI-BruiseNet is a compact segmentation framework for apple bruise detection from 9-band MSI cubes stored as `.npy` files. It combines a 9-channel MobileNetV2 encoder, a UNet-like decoder, and Local Spectral Anomaly Attention (LSAA) with ConvGLU skip fusion. The project provides reproducible 5-fold × multi-seed experiments, ablation scripts, spectral pre-analysis tools, and edge-oriented complexity reporting.

## Architecture Diagram
```mermaid
flowchart TD
    A[Input MSI (B,9,H,W)] --> B[MobileNetV2 Encoder\nConv2d 9->32 at stem]
    B --> F1[F1 1/2]
    B --> F2[F2 1/4]
    B --> F3[F3 1/8]
    B --> F4[F4 1/16]
    F4 --> D[UNet Decoder Upsampling]
    F3 --> L3[LSAA + ConvGLU]
    F2 --> L2[LSAA + ConvGLU]
    F1 --> L1[LSAA + ConvGLU]
    L3 --> D
    L2 --> D
    L1 --> D
    D --> H[1x1 Conv]
    H --> O[Softmax -> (B,2,H,W)]
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation
⚠️ **USER ACTION REQUIRED**: provide your own MSI data.

- MSI image path: `data/images/`
- Mask path: `data/masks/`
- Fold split JSON path: `data/splits/`

### `.npy` format requirements
- Images:
  - path: `data/images/`
  - file suffix: `.npy`
  - shape: `(H, W, 9)`
  - dtype: `float32` or `uint16`
  - naming: `sample_001.npy`, `sample_002.npy`, ...
- Masks:
  - path: `data/masks/`
  - shape: `(H, W)`
  - dtype: `uint8`
  - label values: `0=background`, `1=bruise`
  - naming must strictly match image IDs.

### Data tree
```text
data/
├── images/
├── masks/
└── splits/
```

## Configuration
All hyper-parameters and paths are centralized in `configs/config.yaml`.

Important fields users should edit:
- `data.image_dir`, `data.mask_dir`, `data.split_dir`
- `data.input_size`
- `train.batch_size`, `train.epochs`, `train.seeds`
- `evaluation.pixel_per_cm`

## Quick Start
```bash
# Step 0 (optional): generate dummy data for smoke test
python scripts/generate_dummy_data.py --root data --num-samples 20 --size 256

# Step 1: compute channel-wise normalization statistics
python scripts/compute_norm_stats.py --config configs/config.yaml

# Step 2: generate 5-fold splits
python scripts/prepare_splits.py --config configs/config.yaml --seed 42

# Step 3: spectral pre-analysis (optional)
python scripts/spectral_analysis.py --config configs/config.yaml
# or use notebooks/spectral_preanalysis.ipynb

# Step 4: train
python scripts/train.py --config configs/config.yaml

# Step 5: evaluate
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_default_seed42_fold0.pth
```

## Ablation & Baselines
```bash
bash scripts/ablation.sh
bash scripts/baseline.sh
```

## Output Structure
```text
outputs/
├── checkpoints/      # best_*.pth
├── logs/             # TensorBoard events
├── predictions/      # pred .npy + overlay .png
├── results/          # CSV / JSON metrics
└── spectral_analysis/
```

## Key File-Path Reference Table

| What | Where in Code | Config Key | Default Path |
|------|--------------|------------|-------------|
| MSI images (.npy) | `datasets/msi_dataset.py` L32 | `data.image_dir` | `data/images/` |
| Masks (.npy) | `datasets/msi_dataset.py` L33 | `data.mask_dir` | `data/masks/` |
| Fold splits | `datasets/msi_dataset.py` L34 | `data.split_dir` | `data/splits/` |
| Norm stats | `datasets/msi_dataset.py` L49 | auto-detect | `data/norm_stats.json` |
| Checkpoints | `scripts/train.py` L83 | `output.checkpoints` | `outputs/checkpoints/` |
| TensorBoard logs | `scripts/train.py` L84 | `output.logs` | `outputs/logs/` |
| Predictions | `scripts/evaluate.py` L40 | `output.predictions` | `outputs/predictions/` |
| Metrics CSV/JSON | `scripts/evaluate.py` L41 | `output.results` | `outputs/results/` |

## Citation
```bibtex
@misc{msi-bruisenet-2026,
  title={MSI-BruiseNet: Lightweight Multispectral Apple Bruise Segmentation},
  author={Your Name},
  year={2026}
}
```

## License
MIT License.

---

## 中文说明 (Chinese README)

# MSI-BruiseNet（中文）

这是一个轻量级 PyTorch 项目，用于 **9 波段快照式多光谱成像（MSI）苹果瘀伤语义分割**，可在中等硬件（如 RTX 3060 Laptop 6GB）完成训练与推理。

## 摘要
MSI-BruiseNet 针对苹果瘀伤分割任务，输入为 `.npy` 多光谱立方体（9 通道），采用 MobileNetV2 编码器 + UNet 解码器结构，在跳跃连接位置引入 LSAA（局部谱异常注意力）与 ConvGLU 门控融合。项目内置 5-fold × 多随机种子的训练流程、消融实验脚本、光谱相关性预分析和模型复杂度统计，适合科研复现与边缘部署验证。

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备（重点）
⚠️ **用户必须自行准备数据**，项目不会自带真实苹果数据。

### 1) 图像文件（MSI）
- 目录：`data/images/`
- 格式：NumPy `.npy`
- 数组 shape：`(H, W, 9)`（高、宽、9个通道）
- dtype：推荐 `float32`（也支持 `uint16`）
- 命名：`sample_001.npy`, `sample_002.npy` ...

### 2) 掩码文件（GT）
- 目录：`data/masks/`
- 格式：`.npy`
- shape：`(H, W)`（必须与对应图像尺寸一致）
- dtype：`uint8`
- 像素取值：`0` 背景，`1` 瘀伤
- 命名：必须与图像文件严格同名（仅目录不同）

### 3) 划分索引
- 目录：`data/splits/`
- 脚本：`scripts/prepare_splits.py`
- 输出：`folds.json` / `folds_seed*.json`

### 4) 数据目录结构
```text
data/
├── images/   # 放 MSI .npy
├── masks/    # 放 mask .npy
└── splits/   # 脚本生成 5-fold JSON
```

## 配置说明
所有参数集中在 `configs/config.yaml`。
你需要重点修改：
- 数据路径：`data.image_dir`, `data.mask_dir`, `data.split_dir`
- 输入尺寸：`data.input_size`
- 训练参数：`train.batch_size`, `train.epochs`, `train.seeds`
- 评估参数：`evaluation.pixel_per_cm`

## 快速开始
```bash
python scripts/generate_dummy_data.py --root data --num-samples 20 --size 256
python scripts/compute_norm_stats.py --config configs/config.yaml
python scripts/prepare_splits.py --config configs/config.yaml --seed 42
python scripts/spectral_analysis.py --config configs/config.yaml
python scripts/train.py --config configs/config.yaml
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_default_seed42_fold0.pth
```

## 消融与基线
```bash
bash scripts/ablation.sh
bash scripts/baseline.sh
```

## 输出目录说明
- `outputs/checkpoints/`：最佳权重 `.pth`
- `outputs/logs/`：TensorBoard 日志
- `outputs/predictions/`：预测 `.npy` + 叠加可视化 `.png`
- `outputs/results/`：指标汇总 CSV / JSON

## FAQ（常见问题）
1. **如何将 TIFF/ENVI 转成 `.npy`？**
   - 可用 `rasterio`、`spectral`、`GDAL` 读取后转换为 NumPy 数组，再 `np.save()`。
   - 最终需整理为 `(H, W, 9)`。
2. **图像尺寸不一致怎么办？**
   - 数据加载器会自动 resize 到 `config.yaml` 中的 `data.input_size`（默认 512）。
3. **显存不够怎么办？**
   - 优先降低 `train.batch_size`，其次降低 `data.input_size`。
4. **没有 GPU 可以训练吗？**
   - 可以，代码支持 CPU 训练，但速度会明显变慢。

## 引用
如用于论文或报告，请在仓库后续更新时补充正式引用信息。

## 许可证
MIT。
