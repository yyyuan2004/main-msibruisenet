"""自动化训练→评估→可视化工作流。

功能:
    1. 运行训练 (train.py 的 train 函数)
    2. 自动在 val set 上运行评估 (eval.py 的逻辑)
    3. 生成科研级别的逐epoch指标曲线图 (PNG)
    4. 可选: 生成数据增强可视化对比图

用法:
    python train_eval.py --config configs/baseline.yaml --seed 42
    python train_eval.py --config configs/se.yaml --seed 42 --vis_augment

注意:
    - 本脚本不影响原有 train.py 和 eval.py 的独立使用
    - 默认 split 比例: train:val = 7:3 (test=0)
    - eval 在 val set 上执行
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from train import train, set_seed
from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import (
    get_train_transforms, get_val_transforms,
    Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90,
    RandomCrop, ElasticTransform, Cutout, GaussianBlur, IntensityJitter,
    GaussianNoise, Resize,
)
from data.split import get_data_splits
from model.model import build_model
from utils.metrics import SegmentationMetrics
from eval import evaluate, plot_confusion_matrix, visualize_predictions, print_results, analyze_band_weights, _normalize_band


# ---------------------------------------------------------------------------
# 科研绘图: 逐epoch指标曲线
# ---------------------------------------------------------------------------

def plot_metric_curves(log_path, output_dir, experiment_name):
    """从 training_log.json 生成科研级别的逐epoch指标曲线图。

    输出图片:
        - loss_curve.png: 训练/验证损失曲线
        - iou_f1_curve.png: Class1 IoU 与 F1 曲线 (主图)
        - precision_recall_curve.png: Precision 与 Recall 曲线
        - lr_curve.png: 学习率变化曲线
        - metrics_summary.png: 所有关键指标汇总图
    """
    with open(log_path, "r") as f:
        logs = json.load(f)

    if not logs:
        print("WARNING: training_log.json is empty, skipping curve plotting.")
        return

    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    epochs = [e["epoch"] for e in logs]
    train_loss = [e["train_loss"] for e in logs]
    val_loss = [e["val_loss"] for e in logs]
    iou_c1 = [e["IoU_class1"] for e in logs]
    f1 = [e["F1_macro"] for e in logs]
    precision = [e["Precision_macro"] for e in logs]
    recall = [e["Recall_macro"] for e in logs]
    lr = [e["lr"] for e in logs]

    # 全局绘图风格
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
    })

    def _annotate_max(ax, x, y, label, color):
        """在曲线最大值处标注。"""
        idx = int(np.argmax(y))
        xmax, ymax = x[idx], y[idx]
        ax.annotate(
            f"max={ymax:.4f}\n(epoch {xmax})",
            xy=(xmax, ymax), fontsize=8,
            xytext=(15, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    def _annotate_min(ax, x, y, label, color):
        """在曲线最小值处标注。"""
        idx = int(np.argmin(y))
        xmin, ymin = x[idx], y[idx]
        ax.annotate(
            f"min={ymin:.4f}\n(epoch {xmin})",
            xy=(xmin, ymin), fontsize=8,
            xytext=(15, -25), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    def _add_mean_line(ax, y, color, label):
        """添加均值水平线。"""
        mean_val = np.mean(y)
        ax.axhline(y=mean_val, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            0.98, mean_val, f"mean={mean_val:.4f}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=color, alpha=0.7,
        )

    # --- 1. Loss 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color="#2196F3", label="Train Loss")
    ax.plot(epochs, val_loss, color="#F44336", label="Val Loss")
    _annotate_min(ax, epochs, val_loss, "Val Loss", "#F44336")
    _add_mean_line(ax, val_loss, "#F44336", "val mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training & Validation Loss — {experiment_name}")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "loss_curve.png"), dpi=200)
    plt.close(fig)

    # --- 2. IoU(class1) + F1 曲线 (主图) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, iou_c1, color="#4CAF50", label="IoU (defect, class 1)")
    ax.plot(epochs, f1, color="#FF9800", label="F1 (macro)")
    _annotate_max(ax, epochs, iou_c1, "IoU", "#4CAF50")
    _annotate_max(ax, epochs, f1, "F1", "#FF9800")
    _add_mean_line(ax, iou_c1, "#4CAF50", "IoU mean")
    _add_mean_line(ax, f1, "#FF9800", "F1 mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"IoU (Defect) & F1 Score — {experiment_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "iou_f1_curve.png"), dpi=200)
    plt.close(fig)

    # --- 3. Precision + Recall 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, precision, color="#9C27B0", label="Precision (macro)")
    ax.plot(epochs, recall, color="#00BCD4", label="Recall (macro)")
    _annotate_max(ax, epochs, precision, "Precision", "#9C27B0")
    _annotate_max(ax, epochs, recall, "Recall", "#00BCD4")
    _add_mean_line(ax, precision, "#9C27B0", "Prec mean")
    _add_mean_line(ax, recall, "#00BCD4", "Rec mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision & Recall — {experiment_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "precision_recall_curve.png"), dpi=200)
    plt.close(fig)

    # --- 4. 汇总图 (1x3 子图：Loss | IoU+F1 | Precision+Recall) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) Loss
    axes[0].plot(epochs, train_loss, color="#2196F3", label="Train Loss")
    axes[0].plot(epochs, val_loss, color="#F44336", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # (1) IoU + F1
    axes[1].plot(epochs, iou_c1, color="#4CAF50", label="IoU(defect)")
    axes[1].plot(epochs, f1, color="#FF9800", label="F1(macro)")
    axes[1].set_title("IoU & F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")

    # (2) Precision + Recall
    axes[2].plot(epochs, precision, color="#9C27B0", label="Precision")
    axes[2].plot(epochs, recall, color="#00BCD4", label="Recall")
    axes[2].set_title("Precision & Recall")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")

    # 在汇总图里标注最优值
    best_iou_idx = int(np.argmax(iou_c1))
    best_f1_idx = int(np.argmax(f1))
    summary_text = (
        f"Best IoU(defect): {iou_c1[best_iou_idx]:.4f} @ epoch {epochs[best_iou_idx]}  |  "
        f"Best F1(macro): {f1[best_f1_idx]:.4f} @ epoch {epochs[best_f1_idx]}  |  "
        f"Mean IoU(defect): {np.mean(iou_c1):.4f}  |  "
        f"Mean F1(macro): {np.mean(f1):.4f}"
    )
    fig.text(0.5, 0.01, summary_text, ha="center", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="#90CAF9"))

    fig.suptitle(f"Training Metrics Summary — {experiment_name}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(os.path.join(vis_dir, "metrics_summary.png"), dpi=200)
    plt.close(fig)

    print(f"Metric curves saved to {vis_dir}")


# ---------------------------------------------------------------------------
# 数据增强可视化
# ---------------------------------------------------------------------------

def visualize_augmentations(cfg, output_dir, num_samples=3):
    """可视化各种数据增强方法的效果，原图与增强后对比。

    对每种增强方法分别生成一张对比图 (原图 | 增强后)，保存到
    output_dir/visualization/augmentations/ 目录。
    """
    data_dir = cfg["data"]["data_dir"]
    image_dir = cfg["data"]["image_dir"]
    mask_dir = cfg["data"]["mask_dir"]

    image_root = os.path.join(data_dir, image_dir)
    if not os.path.isdir(image_root):
        print(f"WARNING: image directory not found: {image_root}, skipping augment visualization.")
        return

    stems = sorted([f[:-4] for f in os.listdir(image_root) if f.endswith(".npy")])
    if not stems:
        print("WARNING: no .npy files found, skipping augment visualization.")
        return

    vis_dir = os.path.join(output_dir, "visualization", "augmentations")
    os.makedirs(vis_dir, exist_ok=True)

    vis_bands = tuple(cfg.get("eval", {}).get("vis_bands", [0, 4, 8]))
    num_samples = min(num_samples, len(stems))
    selected_stems = stems[:num_samples]

    # 定义要可视化的增强方法
    augment_methods = {
        "HorizontalFlip": RandomHorizontalFlip(p=1.0),
        "VerticalFlip": RandomVerticalFlip(p=1.0),
        "Rotation90": RandomRotation90(),
        "ElasticTransform": ElasticTransform(alpha=50, sigma=7, p=1.0),
        "Cutout": Cutout(num_holes=2, max_h_frac=0.3, max_w_frac=0.3, p=1.0),
        "GaussianBlur": GaussianBlur(kernel_range=(5, 5), sigma_range=(1.5, 1.5), p=1.0),
        "IntensityJitter": IntensityJitter(scale_range=(0.7, 1.3), p=1.0),
        "GaussianNoise": GaussianNoise(std=0.02, p=1.0),
    }

    def to_rgb(img, bands):
        """将多光谱图像的指定波段转为伪彩色 RGB。"""
        rgb = np.stack([img[b] for b in bands], axis=-1)
        for c in range(3):
            vmin, vmax = np.percentile(rgb[:, :, c], [2, 98])
            if vmax - vmin > 1e-6:
                rgb[:, :, c] = np.clip((rgb[:, :, c] - vmin) / (vmax - vmin), 0, 1)
            else:
                rgb[:, :, c] = 0.0
        return rgb

    for stem in selected_stems:
        image = np.load(os.path.join(image_root, stem + ".npy")).astype(np.float32)
        image = image.transpose(2, 0, 1)  # (C, H, W)

        mask_npy = os.path.join(data_dir, mask_dir, stem + ".npy")
        mask_png = os.path.join(data_dir, mask_dir, stem + ".png")
        if os.path.exists(mask_npy):
            mask = np.load(mask_npy).astype(np.int64)
        elif os.path.exists(mask_png):
            from PIL import Image
            mask = np.array(Image.open(mask_png)).astype(np.int64)
        else:
            mask = np.zeros(image.shape[1:], dtype=np.int64)

        n_methods = len(augment_methods)
        fig, axes = plt.subplots(n_methods, 4, figsize=(16, 3.5 * n_methods))

        for row, (name, aug) in enumerate(augment_methods.items()):
            np.random.seed(42)
            aug_image, aug_mask = aug(image.copy(), mask.copy())

            orig_rgb = to_rgb(image, vis_bands)
            aug_rgb = to_rgb(aug_image, vis_bands)

            # 原图
            axes[row, 0].imshow(orig_rgb)
            axes[row, 0].set_title("Original Image", fontsize=9)
            axes[row, 0].axis("off")

            # 原始mask
            axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("Original Mask", fontsize=9)
            axes[row, 1].axis("off")

            # 增强后图像
            axes[row, 2].imshow(aug_rgb)
            axes[row, 2].set_title(f"After {name}", fontsize=9)
            axes[row, 2].axis("off")

            # 增强后mask
            axes[row, 3].imshow(aug_mask, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].set_title(f"Mask after {name}", fontsize=9)
            axes[row, 3].axis("off")

            # 行标签
            axes[row, 0].set_ylabel(name, fontsize=10, rotation=0,
                                     labelpad=80, va="center")

        fig.suptitle(f"Data Augmentation Visualization — {stem}", fontsize=14)
        fig.tight_layout(rect=[0.08, 0, 1, 0.97])
        fig.savefig(os.path.join(vis_dir, f"augment_{stem}.png"), dpi=150)
        plt.close(fig)

    print(f"Augmentation visualizations saved to {vis_dir}")


# ---------------------------------------------------------------------------
# 评估阶段 (复用 eval.py 的逻辑)
# ---------------------------------------------------------------------------

def run_eval(cfg, seed, output_dir):
    """在 val set 上运行评估，生成所有评估输出。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: best checkpoint not found at {ckpt_path}, skipping eval.")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Data
    data_dir = cfg["data"]["data_dir"]
    splits = get_data_splits(
        data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        seed=seed,
    )

    val_transform = get_val_transforms(cfg)
    ds_kwargs = get_dataset_kwargs(cfg)
    val_dataset = MSIDataset(
        splits["val"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
    )

    # Model
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Band attention analysis
    analyze_band_weights(model, val_loader, device, eval_dir, cfg["experiment_name"])

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # Evaluate
    results, all_preds, all_masks, all_images, all_images_raw, all_stems = evaluate(
        model, val_loader, metrics, device, num_classes
    )

    # Print
    print_results(results, cfg["experiment_name"])

    # Save results JSON
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results_serializable[k] = v.tolist()
        else:
            results_serializable[k] = v

    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(results, eval_dir, num_classes)

    # Visualization
    vis_bands = tuple(cfg.get("eval", {}).get("vis_bands", [0, 4, 8]))
    num_vis = cfg.get("eval", {}).get("num_vis_samples", 10)
    visualize_predictions(
        all_images, all_preds, all_masks, all_stems,
        eval_dir, vis_bands=vis_bands, num_samples=num_vis,
        images_raw=all_images_raw,
    )

    print(f"Evaluation results saved to {eval_dir}")
    return results


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="自动化训练→评估→可视化工作流"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<experiment>_seed<seed>)")
    parser.add_argument("--vis_augment", action="store_true",
                        help="Generate augmentation visualization")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only run eval + plotting (requires existing checkpoint)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, only run training + plotting")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg["experiment_name"]

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs", f"{experiment_name}_seed{args.seed}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config copy
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("=" * 70)
    print(f" Automated Train-Eval Workflow")
    print(f" Experiment: {experiment_name}")
    print(f" Seed: {args.seed}")
    print(f" Output: {args.output_dir}")
    print("=" * 70)

    # ---- Step 1: Training ----
    if not args.skip_train:
        print("\n[Step 1/4] Training...")
        train_result = train(cfg, args.seed, args.output_dir)
        print(f"Training result: {train_result}")
    else:
        print("\n[Step 1/4] Training skipped (--skip_train)")

    # ---- Step 2: Metric curves ----
    log_path = os.path.join(args.output_dir, "training_log.json")
    if os.path.exists(log_path):
        print("\n[Step 2/4] Generating metric curves...")
        plot_metric_curves(log_path, args.output_dir, experiment_name)
    else:
        print("\n[Step 2/4] No training_log.json found, skipping curve plotting.")

    # ---- Step 3: Evaluation on val set ----
    if not args.skip_eval:
        print("\n[Step 3/4] Evaluating on validation set...")
        run_eval(cfg, args.seed, args.output_dir)
    else:
        print("\n[Step 3/4] Evaluation skipped (--skip_eval)")

    # ---- Step 4: Augmentation visualization ----
    if args.vis_augment:
        print("\n[Step 4/4] Generating augmentation visualizations...")
        visualize_augmentations(cfg, args.output_dir, num_samples=3)
    else:
        print("\n[Step 4/4] Augmentation visualization skipped (use --vis_augment to enable)")

    print("\n" + "=" * 70)
    print(" Workflow complete!")
    print(f" All outputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
