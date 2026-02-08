# 多光谱水果缺陷分割（UNet Baseline）

这是一个用于**多光谱快照水果缺陷识别**的基础实验代码，使用 **UNet + 语义分割**。

## 1. 安装依赖

```bash
pip install torch torchvision numpy pillow
```

## 2. 数据集组织方式

请按下面结构准备数据：

```text
dataset_root/
  train/
    images/
      000.npy
      001.npy
    masks/
      000.png
      001.png
  val/
    images/
      100.npy
    masks/
      100.png
```

- `images/*.npy`：多光谱图像，shape 为 `(H, W, C)`。
- `masks/*.png`：单通道标注图，像素值是类别 ID（例如：`0=背景`, `1=缺陷`）。

## 3. 训练示例

```bash
python train_unet_multispectral.py \
  --data_root ./dataset_root \
  --in_channels 8 \
  --num_classes 2 \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-3
```

参数说明：

- `--in_channels`：光谱通道数（例如 8、16）。
- `--num_classes`：分割类别数。
- `--save_path`：最佳模型保存路径（默认 `checkpoints/unet_best.pth`）。

## 4. 下一步建议

你可以在这个 baseline 上继续做：

1. 使用 Dice Loss / Focal Loss 处理类别不平衡。
2. 加入数据增强（随机裁剪、翻转、光谱通道扰动）。
3. 评估 IoU / mIoU，而不只是像素准确率。
4. 尝试更强模型（UNet++、DeepLabV3+）。
