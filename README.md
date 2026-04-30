[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/main-msibruisenet)

# MSI-Bruise-Seg：苹果多光谱图像瘀伤分割（MobileNetV2-UNet + 注意力/频谱模块消融）

基于 9 波段近红外多光谱图像（MSI）的苹果瘀伤（defect）像素级语义分割项目。
以轻量级 MobileNetV2-UNet 为骨架，对比若干"可插拔"模块（1D 频谱卷积 + SE、全局显著性分支）以及 3 个外部参考架构（SMP U-Net/ResNet34、SMP FPN/EfficientNet-B0、Fang 2025 改进版 DeepLabV3+），在相同训练/评估配方下做消融对照。

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
│   ├── sda_input.yaml                   # SDA v2 — input 位置 concat
│   ├── sda_s2.yaml                      # SDA v2 — S2 特征 concat
│   ├── sda_decoder.yaml                 # SDA v2 — decoder 后段 concat
│   ├── sda_multiscale.yaml              # SDA v2 — input + S2 多尺度
│   ├── smp_deeplabv3plus_mobilenetv2.yaml
│   ├── smp_fpn_efficientnetb0.yaml
│   ├── smp_unet_resnet34.yaml
│   └── deeplabv3plus_fang.yaml          # Fang 2025 改进版 DLv3+（DSConv-ASPP + ECA）
├── data/
│   ├── dataset.py                       # MSIDataset: 9ch .npy + apple_mask + band 选择
│   ├── augment.py                       # 空间增强（image/mask 同步）
│   └── split.py                         # 7:3 划分 / k-fold
├── model/
│   ├── encoder.py                       # MobileNetV2 / MobileNetV3 / EfficientNet-B0 (9ch 适配)
│   ├── decoder.py                       # UNet 解码器（skip: none/se/cbam; 支持 SDA concat）
│   ├── modules.py                       # SE / CBAM / ASPP / SpectralConv1D / BandSE / SDAModuleV2
│   ├── smp_models.py                    # segmentation_models_pytorch 封装
│   ├── deeplabv3plus.py                 # Fang 改进版 DeepLabV3+
│   ├── model.py                         # build_model 工厂（含 SDA v2 多位置注入）
│   └── loss.py                          # CE+Dice / Focal
├── utils/
│   ├── metrics.py                       # IoU / F1 / Precision / Recall
│   ├── sda_features.py                  # SDA v2 光谱异常特征提取
│   ├── postprocess.py                   # 推理后处理（unsharp / guided filter）
│   └── spectral_analysis.py             # 光谱预分析
├── scripts/
│   └── band_search.py                   # C(9,k) 波段子集穷举搜索
├── train.py                             # 训练主循环（含 early stopping）
├── eval.py                              # 评估 + 可视化 + SDA v2 heatmap
├── train_eval.py                        # 单 config 一键：训练→评估→曲线图
├── run_ablation.sh                      # 全量消融（12 configs × 3 seeds）
├── run_sda_ablation.sh                  # SDA v2 专项消融（5 configs paired comparison）
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

# 3'. 单 config 5-fold 交叉验证
python train_eval.py \
    --config configs/baseline.yaml \
    --seed 42 \
    --kfold 5

# 4. 全量消融（8 configs × 3 seeds = 24 次，7:3 划分）
bash run_ablation.sh

# 4'. 全量消融 + k-fold（8 configs × 3 seeds × 5 folds = 120 次）
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

> `train.py` 的 early stopping 基于验证集 **class-1 (defect) IoU** 相对历史最佳是否有提升；
> 仅当 IoU 提升时才覆盖写 `best_model.pth`，到达 patience 时提前终止。

---

## 6. 对比实验总览

**自研 MobileNetV2-UNet 家族**
| Config | 核心改动 |
|---|---|
| `baseline` | 纯 MobileNetV2 + UNet，无任何增强模块 |
| `spconv_se` | 在 S1 后插入 1D SpectralConv + SE 通道注意力 |
| `input_band_se` | 输入端动态 BandSE（逐图像自适应波段加权） |
| `global_branch` | 并行低分辨率全局显著性分支（弱监督空间先验） |

**SDA v2 可解释光谱异常注意力（4 种插入位置消融）**
| Config | 插入位置 | 说明 |
|---|---|---|
| `sda_input` | 输入端 | 异常特征图与输入 concat → encoder 接收 4+4=8 通道 |
| `sda_s2` | S2 特征后 | 异常特征图下采样后 concat 到 S2 skip connection |
| `sda_decoder` | Decoder 后段 | 异常特征图注入 decoder 的 S1/S2 skip connection |
| `sda_multiscale` | 输入 + S2 | 同时在输入端和 S2 两个尺度注入异常特征 |

**外部参考架构（cross-method 对照）**
| Config | 架构 | 说明 |
|---|---|---|
| `smp_unet_resnet34` | U-Net + ResNet34 | 经典 U-Net + 强 encoder |
| `smp_fpn_efficientnetb0` | FPN + EfficientNet-B0 | 强 encoder + FPN 多尺度 |
| `smp_deeplabv3plus_mobilenetv2` | DeepLabV3+ + MobileNetV2 | 同 encoder 下不同 decoder |
| `deeplabv3plus_fang` | Fang 2025 改进版 DLv3+ | DSConv-ASPP + ECA + Focal |

---

## 7. 可选预处理与训练开关

所有 config 都暴露以下开关：

```yaml
data:
  num_channels: 4               # 必须与 band_indices 长度一致
  band_indices: [0, 2, 4, 5]    # 默认从 9 波段中选 4 个；null 表示用全 9 波段
  use_sharpen: false            # 是否做 Unsharp Masking
  sharpen_sigma: 1.0            # 高斯模糊 σ
  sharpen_alpha: 1.5            # 锐化强度

train:
  # 学习率 warmup（线性预热到目标 lr，再走 CosineAnnealing）
  use_warmup: false             # 总开关
  warmup_epochs: 5              # 预热长度
  warmup_start_factor: 0.01     # 起始 lr = lr * start_factor
```

- **波段子集搜索**：`python scripts/band_search.py --k 4` 在 C(9,k) 里穷举验证哪几个波段组合最优。
- **Unsharp Masking**：`configs/baseline_sharpen.yaml` 提供开启锐化的对照 config。
- **K-fold 交叉验证**：`python train_eval.py --config X --kfold 5` 自动跑 5 折并输出 `kfold_summary.json`（mean ± std）。
- **Warmup**：在 config 中设 `use_warmup: true` 即可启用线性预热。

---

## 8. SDA v2：可解释光谱异常注意力模块

### 设计动机

诊断实验表明：像素级 raw_spectrum + SNV_spectrum 的 logistic regression 已有很强判别力。
baseline 的强表现主要来自局部像素光谱可分性，CNN 补充空间平滑和上下文。
因此 SDA v2 不再以 mean_reflectance 残差作为异常图，改为利用 raw + SNV 的互补光谱结构。

### 异常特征图

| 特征名 | 计算方式 | 物理含义 |
|---|---|---|
| `spectral_std` | 像素级波段标准差 | 光谱变异程度（缺陷区反射率波段间差异大） |
| `sam` | Spectral Angle Mapper | 与健康参考光谱的夹角（方向差异） |
| `snv_l2` | SNV 变换后 L2 距离 | 归一化后的光谱形状差异 |
| `mahalanobis` | Mahalanobis 距离 | 考虑波段间相关性的统计异常度 |
| `raw_l2` | 原始 L2 距离（可选） | 与健康参考的绝对差异 |

- 健康参考 = 苹果前景像素的均值光谱（per-image 在线计算）
- 所有特征在苹果前景内归一化到 [0, 1]，背景强制为 0
- 苹果前景 mask 优先读取 `data_dir/whole/` 目录，fallback 使用阈值估计

### 软门控机制

```
A_low = GaussianBlur(anomaly_map, sigma_a)
T     = texture_energy (Laplacian², 平滑 sigma_t)
G     = apple_mask × σ((A_low - τ_a) / s_a) × σ((τ_t - T) / s_t)
```

- `tau_a`, `tau_t`, `s_a`, `s_t`：可学习参数（自动梯度优化）
- `sigma_a`, `sigma_t`：config 可调（默认 3.0, 5.0）
- 纹理抑制：高频果皮纹理被 `sigmoid(τ_t - T)` 自动压制
- `gate_mode`: `concat`（默认，不改变 backbone 特征） / `multiply` / `none`

### 配置示例

```yaml
model:
  sda_v2:
    enabled: true
    position: "input"      # input / s2 / decoder / multiscale
    features: ["spectral_std", "sam", "snv_l2", "mahalanobis"]
    sigma_a: 3.0
    sigma_t: 5.0
    gate_mode: "concat"
    use_soft_gate: true
```

### SDA 专项消融实验

```bash
# 5 configs × 3 seeds = 15 次（paired comparison）
bash run_sda_ablation.sh

# 5-fold 交叉验证
bash run_sda_ablation.sh --kfold 5

# 自定义 seed
bash run_sda_ablation.sh --seeds 42,123
```

### 实验输出额外记录

训练结束时自动记录 `sda_config` 到 result dict：

```json
{
  "sda_enabled": true,
  "sda_position": "input",
  "sda_feature_names": ["spectral_std", "sam", "snv_l2", "mahalanobis"],
  "tau_t": 0.30,  "tau_a": 0.50,
  "sigma_a": 3.0, "sigma_t": 5.0,
  "gate_mode": "concat",
  "use_whole_mask": true,
  "apple_mask_source": "whole_mask_npy"
}
```

---

## 9. 评估指标

- **class-1 IoU**（主要指标，用于 early stopping）
- mIoU, F1(macro), Precision(macro), Recall(macro)
- 所有指标在 3 个随机种子上取 mean ± std
- `aggregate_results.py` 会输出统一格式的消融表到 `outputs/ablation_table.txt`

---

## 10. 输出目录结构

**单 split 模式（默认 7:3）：**
```
outputs/<config>_seed<seed>/
├── checkpoints/best_model.pth
├── training_log.json            # 每 epoch 的 loss / metric / lr
├── visualization/
│   ├── loss_curve.png
│   ├── iou_f1_curve.png
│   ├── precision_recall_curve.png
│   └── metrics_summary.png      # 3-panel 合图
└── eval_results/
    ├── results.json
    ├── confusion_matrix.png
    └── visualizations/          # 预测对比图（含多波段 grid + sharpen 对比）
```

**K-fold 模式：**
```
outputs/<config>_seed<seed>_kfold<N>/
├── fold0/                       # 同上结构
├── fold1/
├── ...
└── kfold_summary.json           # 跨折聚合 mean ± std
```

---

