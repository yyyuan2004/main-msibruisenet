[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/main-msibruisenet)

# MSI-Bruise-Seg：苹果多光谱图像瘀伤分割（MobileNetV2-UNet + 注意力/频谱模块消融）

基于 9 波段近红外多光谱图像（MSI）的苹果瘀伤（defect）像素级语义分割项目。
以轻量级 MobileNetV2-UNet 为骨架，对比若干"可插拔"增强模块（1D 频谱卷积 + SE、输入端 BandSE、全局显著性分支）以及 3 个外部参考架构（SMP U-Net/ResNet34、SMP FPN/EfficientNet-B0、Fang 2025 改进版 DeepLabV3+），在相同训练/评估配方下做消融对照。

---

## 1. 数据集

```
/root/autodl-tmp/datasets/185_9bands/
├── images/    # .npy, shape (H, W, 9), float32 反射率
└── masks/     # .npy 或 .png, shape (H, W), 0=背景 / 1=瘀伤
```

- 共 185 张标注样本，9 个近红外波段（620–780 nm）
- `images/` 与 `masks/` 文件名严格一一对应
- 默认按 7:3 划分 train/val（无独立 test）

> 换设备/换数据集时只需改各 `configs/*.yaml` 的 `data.data_dir` 字段。

---

## 2. 整体工作流

每个 config 对应如下 10 步统一流程：

```
 [1] 数据读取     ── 9ch .npy + 二值 mask
 [2] 预处理       ── (可选) band_indices / unsharp mask / LCN / PCA
 [3] 数据划分     ── train:val = 7:3（固定 seed）
 [4] 数据增强     ── 水平/垂直翻转、90°/180°/270° 旋转（无颜色增强）
 [5] 模型前向     ── Encoder → (可选 ASPP/SpConv/BandSE/GlobalBranch) → Decoder → 1x1 seg head
 [6] 损失函数     ── 0.5·CE + 0.5·Dice（Fang 版用 Focal）
 [7] 优化器       ── AdamW + CosineAnnealingLR(T_max=epochs, η_min=1e-6)
 [8] 训练策略     ── 500 epochs；early stopping on class-1 IoU (patience=50)
 [9] 评估         ── val set：class-1 IoU / F1 / Precision / Recall
 [10] 可视化      ── train_eval.py 自动产出 metric 曲线 + 预测对比图
```

---

## 3. 项目结构

```
.
├── configs/                             # 每个消融实验一个 YAML
│   ├── baseline.yaml                    # 纯 MobileNetV2-UNet
│   ├── spconv_se.yaml                   # + 1D SpectralConv + SE
│   ├── input_band_se.yaml               # + 输入端动态 BandSE
│   ├── global_branch.yaml               # + 全局显著性旁路分支
│   ├── smp_deeplabv3plus_mobilenetv2.yaml
│   ├── smp_fpn_efficientnetb0.yaml
│   ├── smp_unet_resnet34.yaml
│   └── deeplabv3plus_fang.yaml          # Fang 2025 改进版 DLv3+（DSConv-ASPP + ECA）
├── data/
│   ├── dataset.py                       # MSIDataset: 9ch .npy + 可选 band 选择/锐化
│   ├── augment.py                       # 空间增强（image/mask 同步）
│   └── split.py                         # 7:3 划分
├── model/
│   ├── encoder.py                       # MobileNetV2 / MobileNetV3 / EfficientNet-B0 (9ch 适配)
│   ├── decoder.py                       # UNet 解码器（skip: none/se/cbam）
│   ├── modules.py                       # SE / CBAM / ASPP / SpectralConv1D / BandSE / GlobalBranch
│   ├── smp_models.py                    # segmentation_models_pytorch 封装
│   ├── deeplabv3plus.py                 # Fang 改进版 DeepLabV3+
│   ├── model.py                         # build_model 工厂
│   └── loss.py                          # CE+Dice / Focal
├── utils/
│   ├── metrics.py                       # IoU / F1 / Precision / Recall
│   ├── postprocess.py                   # 推理后处理（unsharp / guided filter）
│   └── spectral_analysis.py             # 光谱预分析
├── scripts/
│   └── band_search.py                   # C(9,k) 波段子集穷举搜索
├── train.py                             # 训练主循环（含 early stopping）
├── eval.py                              # 评估 + 可视化
├── train_eval.py                        # 单 config 一键：训练→评估→曲线图
├── run_ablation.sh                      # 全量消融（8 configs × 3 seeds）
├── aggregate_results.py                 # 聚合所有种子结果为表格
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

# 4. 全量消融（8 configs × 3 seeds = 24 次）
bash run_ablation.sh

# 5. 聚合结果
python aggregate_results.py
# → outputs/ablation_table.txt
```

---

## 5. 统一训练超参

| 参数 | 取值 |
|------|------|
| Optimizer | AdamW, weight_decay=1e-4 |
| Learning Rate | 5e-4，CosineAnnealing to 1e-6 |
| Epochs | 500 |
| Batch Size | 32 |
| Loss | 0.5·CE + 0.5·Dice（`deeplabv3plus_fang` 用 Focal γ=2, α=0.25） |
| Input Size | 512×512，random crop 384×384 |
| Augmentation | h-flip / v-flip / {90,180,270}° 旋转 |
| Early Stopping | class-1 IoU, patience=50 |
| save_interval | 99999（只保存 best_model.pth） |

> `train.py` 的 early stopping 基于验证集 **class-1 (defect) IoU** 相对历史最佳是否有提升；
> 仅当 IoU 提升时才覆盖写 `best_model.pth`，到达 patience 时提前终止。

---

## 6. 对比实验总览（保留的 8 个 config）

**自研 MobileNetV2-UNet 家族**
| Config | 核心改动 |
|---|---|
| `baseline` | 纯 MobileNetV2 + UNet，无任何增强模块 |
| `spconv_se` | 在 S1 后插入 1D SpectralConv + SE 通道注意力 |
| `input_band_se` | 输入端动态 BandSE（逐图像自适应波段加权） |
| `global_branch` | 并行低分辨率全局显著性分支（弱监督空间先验） |

**外部参考架构（cross-method 对照）**
| Config | 架构 | 说明 |
|---|---|---|
| `smp_unet_resnet34` | U-Net + ResNet34 | 经典 U-Net + 强 encoder |
| `smp_fpn_efficientnetb0` | FPN + EfficientNet-B0 | 强 encoder + FPN 多尺度 |
| `smp_deeplabv3plus_mobilenetv2` | DeepLabV3+ + MobileNetV2 | 同 encoder 下不同 decoder |
| `deeplabv3plus_fang` | Fang 2025 改进版 DLv3+ | DSConv-ASPP + ECA + Focal |

---

## 7. 可选预处理

所有 config 都暴露以下开关（默认关闭）：

```yaml
data:
  band_indices: null        # 例: [0, 3, 5, 8] 表示只用这 4 个波段
  use_sharpen: false        # 是否做 Unsharp Masking
  sharpen_sigma: 1.0        # 高斯模糊 σ
  sharpen_alpha: 1.5        # 锐化强度
```

- **波段子集搜索**：`python scripts/band_search.py --k 4` 会在 C(9,k) 里穷举验证哪几个波段组合最优。
- **Unsharp Masking**：`configs/baseline_sharpen.yaml` 提供了开启锐化的对照 config。

---

## 8. 评估指标

- **class-1 IoU**（主要指标，用于 early stopping）
- mIoU, F1(macro), Precision(macro), Recall(macro)
- 所有指标在 3 个随机种子（42/123/456）上取 mean ± std
- `aggregate_results.py` 会输出统一格式的消融表到 `outputs/ablation_table.txt`

---

## 9. 输出目录结构（每个 run）

```
outputs/<config>_seed<seed>/
├── checkpoints/
│   └── best_model.pth
├── logs/
│   └── training_log.json        # 每 epoch 的 loss / metric / lr
├── metric_curves/
│   ├── loss_curve.png
│   ├── iou_f1_curve.png
│   └── summary.png              # 3-panel 合图
├── eval_results/
│   ├── results.json
│   └── visualizations/          # 预测对比图（含多波段 grid + sharpen 对比）
└── augment_preview/             # （--vis_augment 时生成）
```

---

## 10. 注意事项

- **不做颜色/亮度增强**：光谱反射率具有物理含义
- **小样本体制**（~185 张）：模块间差异可能落在统计噪声内——本身即是有价值的发现
- **6 GB GPU 显存不足时**：降 `batch_size=16` 或 `crop_size=320`
- `save_interval=99999` 意为仅保存 best，不做周期 checkpoint
