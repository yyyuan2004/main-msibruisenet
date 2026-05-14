[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/main-msibruisenet)

# MSI-Bruise-Seg：苹果多光谱图像瘀伤分割（MobileNetV2-UNet + 注意力模块消融）

基于近红外多光谱图像（MSI）的苹果瘀伤（defect）像素级语义分割项目。
以轻量级 MobileNetV2-UNet 为骨架，对比若干"可插拔"模块（1D 频谱卷积、SE/CBAM、动态 BandSE、全局显著性分支）以及多个外部参考架构（SMP U-Net/UNet++/Linknet/MAnet/DeepLabV3+/FPN），在相同训练/评估配方下做消融对照。

---

## 1. 数据集

```
/root/autodl-tmp/datasets/185_9bands/
├── images/    # .npy, shape (H, W, 9), float32 反射率
└── masks/     # .npy 或 .png, shape (H, W), 0=背景 / 1=瘀伤
```

- 共 185 张标注样本，9 个近红外波段（565–730 nm）
- `images/` 与 `masks/` 文件名严格一一对应

> 换设备/换数据集时只需改各 `configs/*.yaml` 的 `data.data_dir` 字段。

---

## 2. 整体工作流

每个 config 对应如下 10 步统一流程：

```
 [1] 数据读取     ── 9ch .npy + 二值 mask
 [2] 预处理       ── 默认 band_indices=[0,2,4,5] 选 4 波段 / 可选 unsharp mask / LCN / PCA
 [3] 数据划分     ── train:val = 7:3（默认）或 k-fold 交叉验证（--kfold N）
 [4] 数据增强     ── 水平/垂直翻转、90°/180°/270° 旋转（无颜色增强）
 [5] 模型前向     ── Encoder → (可选 ASPP/SpConv/BandSE/GlobalBranch) → Decoder → 1x1 seg head
 [6] 损失函数     ── 0.5·CE + 0.5·Dice（Fang 版用 Focal）
 [7] 优化器       ── AdamW + (可选 LinearWarmup → ) CosineAnnealingLR(η_min=1e-6)
 [8] 训练策略     ── 500 epochs；early stopping on class-1 IoU (patience=50)
 [9] 评估         ── val set：class-1 IoU / F1 / Precision / Recall
 [10] 可视化      ── train_eval.py 自动产出 metric 曲线 + 预测对比图；
                    k-fold 模式额外生成 per-fold 表格 + 柱状图
```

---

## 3. 项目结构

```
.
├── configs/                                    # 每个消融实验一个 YAML
│   ├── baseline.yaml                           # 纯 MobileNetV2-UNet
│   ├── spconv_se.yaml                          # + 1D SpectralConv + SE skip
│   ├── input_band_se.yaml                      # + 输入端动态 BandSE
│   ├── global_branch.yaml                      # + 全局显著性旁路分支
│   ├── smp_unet_resnet34.yaml                  # SMP U-Net + ResNet34
│   ├── smp_unetplusplus_resnet34.yaml          # SMP UNet++ + ResNet34
│   ├── smp_linknet_resnet34.yaml               # SMP Linknet + ResNet34
│   ├── smp_manet_resnet34.yaml                 # SMP MAnet + ResNet34
│   ├── smp_deeplabv3plus_mobilenetv2.yaml      # SMP DeepLabV3+ + MobileNetV2
│   ├── smp_fpn_efficientnetb0.yaml             # SMP FPN + EfficientNet-B0
│   └── deeplabv3plus_fang.yaml                 # Fang 2025 改进版 DLv3+
├── data/
│   ├── dataset.py                              # MSIDataset: 9ch .npy + band 选择
│   ├── augment.py                              # 空间增强（image/mask 同步）
│   └── split.py                                # 7:3 划分 / k-fold
├── model/
│   ├── encoder.py                              # MobileNetV2 / V3 / EfficientNet-B0 (9ch 适配)
│   ├── decoder.py                              # UNet 解码器（skip: none/se/cbam）
│   ├── modules.py                              # SE / CBAM / ASPP / SpectralConv1D / BandSE / GlobalSaliency
│   ├── smp_models.py                           # segmentation_models_pytorch 封装
│   ├── deeplabv3plus.py                        # Fang 改进版 DeepLabV3+
│   ├── model.py                                # build_model 工厂
│   └── loss.py                                 # CE+Dice / Focal
├── utils/
│   ├── metrics.py                              # IoU / F1 / Precision / Recall
│   ├── pca_transform.py                        # PCA 投影矩阵生成
│   └── spectral_analysis.py                    # 光谱预分析
├── scripts/
│   ├── band_search.py                          # C(9,k) 波段子集穷举搜索
│   └── band_selection_comparison.py            # SPA/CARS/MI vs Exhaustive
├── train.py                                    # 训练主循环（含 early stopping）
├── eval.py                                     # 评估 + 可视化
├── train_eval.py                               # 单 config 一键：训练→评估→曲线图（含 k-fold）
├── run_ablation.sh                             # 全量消融（11 configs × 3 seeds）
├── aggregate_results.py                        # 聚合所有种子结果为表格
└── requirements.txt
```

---

## 4. 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. （可选）光谱预分析
python -m utils.spectral_analysis \
    --data_dir /root/autodl-tmp/datasets/185_9bands \
    --output_dir outputs/spectral_analysis

# 3. 单 config 训练+评估+出图
python train_eval.py \
    --config configs/baseline.yaml \
    --seed 42 \
    --output_dir outputs/baseline_seed42

# 3'. 单 config 5-fold 交叉验证（结束后自动美观汇总所有 fold）
python train_eval.py \
    --config configs/baseline.yaml \
    --seed 42 \
    --kfold 5

# 4. 全量消融（11 configs × 3 seeds，7:3 划分）
bash run_ablation.sh

# 4'. 全量消融 + k-fold（11 configs × 3 seeds × 5 folds）
bash run_ablation.sh --kfold 5

# 5. 聚合结果（7:3 模式）
python aggregate_results.py
# → outputs/ablation_table.txt
```

---

## 5. 统一训练超参

| 参数 | 取值 |
|------|------|
| Optimizer | AdamW, weight_decay=1e-4 |
| Learning Rate | 5e-4，CosineAnnealing to 1e-6 |
| Warmup | 可选（`use_warmup`），默认关闭；linear warmup 5 epochs, start_factor=0.01 |
| Epochs | 500 |
| Batch Size | 32 |
| Loss | 0.5·CE + 0.5·Dice（`deeplabv3plus_fang` 用 Focal γ=2, α=0.25） |
| Input Size | 512×512，random crop 384×384 |
| Augmentation | h-flip / v-flip / {90,180,270}° 旋转 |
| Early Stopping | class-1 IoU, patience=50 |
| save_interval | 99999（只保存 best_model.pth） |
| 默认波段 | `band_indices=[0,2,4,5]`，`num_channels=4` |
| 数据划分 | 7:3 单 split（默认），或 k-fold 交叉验证（`--kfold N`） |

> early stopping 基于验证集 **class-1 (defect) IoU**；仅当 IoU 提升时才覆盖写
> `best_model.pth`，到达 patience 时提前终止。

---

## 6. 对比实验总览

**自研 MobileNetV2-UNet 家族**
| Config | 核心改动 |
|---|---|
| `baseline` | 纯 MobileNetV2 + UNet，无任何增强模块 |
| `spconv_se` | 在 S1 后插入 1D SpectralConv + 每个 skip 处加 SE |
| `input_band_se` | 输入端动态 BandSE（逐图像自适应波段加权） |
| `global_branch` | 并行低分辨率全局显著性分支（弱监督空间先验） |

**外部参考架构（cross-method 对照）**
| Config | 架构 | 说明 |
|---|---|---|
| `smp_unet_resnet34` | U-Net + ResNet34 | 经典 U-Net + 强 encoder |
| `smp_unetplusplus_resnet34` | UNet++ + ResNet34 | 嵌套跳连，缩小语义鸿沟 |
| `smp_linknet_resnet34` | Linknet + ResNet34 | 轻量 U-Net 替代（add 而非 concat） |
| `smp_manet_resnet34` | MAnet + ResNet34 | 多尺度注意力 decoder |
| `smp_fpn_efficientnetb0` | FPN + EfficientNet-B0 | 强 encoder + FPN 多尺度 |
| `smp_deeplabv3plus_mobilenetv2` | DeepLabV3+ + MobileNetV2 | 同 encoder 下不同 decoder |
| `deeplabv3plus_fang` | Fang 2025 改进版 DLv3+ | DSConv-ASPP + ECA + Focal |

---

## 7. 可选预处理与训练开关

```yaml
data:
  num_channels: 4               # 必须与 band_indices 长度一致
  band_indices: [0, 2, 4, 5]    # 默认从 9 波段中选 4 个；null 表示用全 9 波段
  use_sharpen: false            # 是否做 Unsharp Masking
  sharpen_sigma: 1.0
  sharpen_alpha: 1.5

train:
  use_warmup: false             # 学习率线性预热到目标 lr，再走 CosineAnnealing
  warmup_epochs: 5
  warmup_start_factor: 0.01
```

- **波段子集搜索**：`python scripts/band_search.py --k 4`
- **波段方法对比**：`python scripts/band_selection_comparison.py`（SPA/CARS/MI vs Exhaustive，CNN 下游）
- **Unsharp Masking**：`configs/baseline_sharpen.yaml`
- **K-fold 交叉验证**：`python train_eval.py --config X --kfold 5`

---

## 8. 评估指标与 K-Fold 汇总

每个 (config, seed) 评估输出：
- **class-1 IoU**（主要指标，用于 early stopping）
- mIoU, F1(macro), Precision(macro), Recall(macro)
- 所有 7:3 实验在 3 个随机种子上取 mean ± std（`aggregate_results.py`）

**K-Fold 模式自动汇总：** 训练完所有 fold 后，`aggregate_kfold_results()` 会同时生成
3 种可读形式：

1. **控制台 pretty 表格**（unicode 边框 + 每折一行 + mean/std 行 + 最佳 fold 标记 `*`）
2. **`kfold_summary.md`** —— Markdown 表格，可直接粘到论文/报告
3. **`kfold_summary.png`** —— per-fold 多指标分组柱状图，每个指标的 mean 用虚线标出
4. **`kfold_summary.json`** —— 含每折原始指标 + mean/std，便于再分析

控制台示例：

```
╔══════════════════════════════════════════════════════════════════╗
║              K-Fold Summary — baseline   (5 folds)               ║
╠══════════════════════════════════════════════════════════════════╣
║ Fold | IoU(c1) |    mIoU |      F1 |    Prec |  Recall           ║
╠══════════════════════════════════════════════════════════════════╣
║   1  |  0.7240 |  0.8330 |  0.8821 |  0.8915 |  0.8732           ║
║   2* |  0.7301 |  0.8362 |  0.8855 |  0.8942 |  0.8770           ║
║   3  |  0.7185 |  0.8295 |  0.8784 |  0.8870 |  0.8702           ║
║   4  |  0.7158 |  0.8281 |  0.8770 |  0.8851 |  0.8689           ║
║   5  |  0.7196 |  0.8307 |  0.8800 |  0.8893 |  0.8718           ║
╠══════════════════════════════════════════════════════════════════╣
║ mean |  0.7216 |  0.8315 |  0.8806 |  0.8894 |  0.8722           ║
║ std  |  0.0050 |  0.0029 |  0.0030 |  0.0034 |  0.0028           ║
╚══════════════════════════════════════════════════════════════════╝
  * = best IoU(c1) fold (fold 2)
```

---

## 9. 输出目录结构

**单 split 模式（默认 7:3）：**
```
outputs/<config>_seed<seed>/
├── checkpoints/best_model.pth
├── training_log.json
├── visualization/
│   ├── loss_curve.png
│   ├── iou_f1_curve.png
│   ├── precision_recall_curve.png
│   └── metrics_summary.png
└── eval_results/
    ├── results.json
    ├── confusion_matrix.png
    └── visualizations/
```

**K-fold 模式：**
```
outputs/<config>_seed<seed>_kfold<N>/
├── fold0/                       # 同上结构
├── fold1/
├── ...
├── kfold_summary.json           # 每折原始指标 + mean ± std
├── kfold_summary.md             # markdown 表格
└── kfold_summary.png            # per-fold 分组柱状图
```

---
