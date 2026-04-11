"""Evaluation script: compute metrics, confusion matrix, visualize segmentation results,
and (if applicable) analyze band attention weights."""

import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import get_val_transforms
from data.split import get_data_splits
from model.model import build_model
from utils.metrics import SegmentationMetrics


@torch.no_grad()
def evaluate(model, dataloader, metrics, device, num_classes):
    """Run evaluation and collect predictions."""
    model.eval()
    metrics.reset()

    all_preds = []
    all_masks = []
    all_images = []
    all_stems = []

    for images, masks, stems in dataloader:
        images_dev = images.to(device)
        masks_dev = masks.to(device)

        logits = model(images_dev)
        preds = logits.argmax(dim=1)

        metrics.update(preds, masks_dev)

        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.numpy())
        all_images.append(images.numpy())
        all_stems.extend(stems)

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    results = metrics.compute()
    return results, all_preds, all_masks, all_images, all_stems


def plot_confusion_matrix(results, output_dir, num_classes):
    """Plot and save confusion matrix."""
    cm = results.get("confusion_matrix")
    if cm is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    classes = [f"Class {i}" for i in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def visualize_predictions(images, preds, masks, stems, output_dir,
                          vis_bands=(0, 4, 8), num_samples=10):
    """Visualize segmentation results: pseudo-color image | prediction | ground truth."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))

    cmap = ListedColormap(["black", "red", "blue", "green", "yellow"][:max(preds.max() + 1, 2)])

    for i in range(num_samples):
        img = images[i]
        pred = preds[i]
        mask = masks[i]
        stem = stems[i]

        rgb = np.stack([img[b] for b in vis_bands], axis=-1)
        for c in range(3):
            vmin, vmax = np.percentile(rgb[:, :, c], [2, 98])
            if vmax - vmin > 1e-6:
                rgb[:, :, c] = np.clip((rgb[:, :, c] - vmin) / (vmax - vmin), 0, 1)
            else:
                rgb[:, :, c] = 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Pseudo-color (bands {vis_bands})")
        axes[0].axis("off")

        axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=max(preds.max(), 1))
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap=cmap, vmin=0, vmax=max(masks.max(), 1))
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        fig.suptitle(stem)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"{stem}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {num_samples} visualizations to {vis_dir}")


def print_results(results, experiment_name):
    """Pretty-print evaluation results. Primary metric: Class 1 (defect) IoU."""
    iou_per_class = results["IoU_per_class"]
    defect_iou = float(iou_per_class[1]) if len(iou_per_class) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Evaluation Results: {experiment_name}")
    print(f"{'=' * 60}")
    print(f"  IoU (defect, class 1): {defect_iou:.4f}  <-- primary metric")
    print(f"  F1 (macro):            {results['F1_macro']:.4f}")
    print(f"  Precision (macro):     {results['Precision_macro']:.4f}")
    print(f"  Recall (macro):        {results['Recall_macro']:.4f}")

    if "IoU_per_class" in results:
        print(f"\n  Per-class metrics:")
        num_classes = len(iou_per_class)
        for c in range(num_classes):
            label = "bg" if c == 0 else "defect"
            print(f"    Class {c} ({label}): IoU={iou_per_class[c]:.4f} | "
                  f"F1={results['F1_per_class'][c]:.4f} | "
                  f"Prec={results['Precision_per_class'][c]:.4f} | "
                  f"Rec={results['Recall_per_class'][c]:.4f}")
    print(f"{'=' * 60}\n")


def analyze_band_weights(model, dataloader, device, output_dir, experiment_name):
    """Extract and visualize band attention weights.

    Supports both BandAttention (static, no input needed) and InputBandSE
    (dynamic, per-image weights via GAP+FC). Saves bar chart + text file.

    Skipped silently if model does not have band_attention.
    """
    if not hasattr(model, 'band_attention'):
        return

    from model.modules import InputBandSE
    is_dynamic = isinstance(model.band_attention, InputBandSE)

    print("\nAnalyzing band attention weights...")
    model.eval()

    if is_dynamic:
        # InputBandSE: collect per-image weights
        all_weights = []
        with torch.no_grad():
            for images, _, _ in dataloader:
                images_dev = images.to(device)
                w = model.band_attention.get_weights(images_dev)  # (B, C)
                all_weights.append(w)
        all_weights = np.concatenate(all_weights, axis=0)  # (N, C)
        avg_weights = all_weights.mean(axis=0)
        std_weights = all_weights.std(axis=0)

        # Print per-band statistics
        print("  Band Importance Weights (dynamic, per-image statistics):")
        print(f"  {'Band':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for i in range(all_weights.shape[1]):
            print(f"  Band {i+1:>2}: {avg_weights[i]:.4f}  {std_weights[i]:.4f}  "
                  f"{all_weights[:, i].min():.4f}  {all_weights[:, i].max():.4f}")

        # Bar chart with error bars
        fig, ax = plt.subplots(figsize=(10, 5))
        bands = range(1, len(avg_weights) + 1)
        bars = ax.bar(bands, avg_weights, yerr=std_weights, capsize=4,
                      color='steelblue', edgecolor='black', alpha=0.8)
        for bar_item in bars:
            yval = bar_item.get_height()
            ax.text(bar_item.get_x() + bar_item.get_width() / 2, yval + 0.01,
                    f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Band Index', fontsize=12)
        ax.set_ylabel('Weight (mean +/- std)', fontsize=12)
        ax.set_title(f'Dynamic Band Importance ({experiment_name})', fontsize=14)
        ax.set_xticks(bands)
        ax.set_ylim(0, min(max(avg_weights) + 0.15, 1.0))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save text with statistics
        with open(os.path.join(output_dir, "band_weights.txt"), "w") as f:
            f.write(f"Experiment: {experiment_name} (InputBandSE dynamic)\n\n")
            f.write(f"{'Band':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
            f.write("-" * 50 + "\n")
            for i in range(all_weights.shape[1]):
                f.write(f"Band {i+1:<4}  {avg_weights[i]:>10.6f} {std_weights[i]:>10.6f} "
                        f"{all_weights[:, i].min():>10.6f} {all_weights[:, i].max():>10.6f}\n")
    else:
        # Static BandAttention: fixed per-band scalars
        avg_weights = model.band_attention.get_weights()  # (C,) numpy

        print("  Band Importance Weights (static, shared across all images):")
        for i, w in enumerate(avg_weights):
            bar = "#" * int(w * 30)
            print(f"    Band {i+1}: {w:.4f}  {bar}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bands = range(1, len(avg_weights) + 1)
        bars = ax.bar(bands, avg_weights, color='steelblue', edgecolor='black', alpha=0.8)
        for bar_item in bars:
            yval = bar_item.get_height()
            ax.text(bar_item.get_x() + bar_item.get_width() / 2, yval + 0.01,
                    f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Band Index', fontsize=12)
        ax.set_ylabel('Learned Importance Weight', fontsize=12)
        ax.set_title(f'Band Importance Analysis ({experiment_name})', fontsize=14)
        ax.set_xticks(bands)
        ax.set_ylim(0, max(avg_weights) + 0.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save text
        with open(os.path.join(output_dir, "band_weights.txt"), "w") as f:
            f.write(f"Experiment: {experiment_name} (BandAttention static)\n\n")
            f.write(f"{'Band':<10} {'Weight':>10}\n")
            f.write("-" * 22 + "\n")
            for i, w in enumerate(avg_weights):
                f.write(f"Band {i+1:<4}  {w:>10.6f}\n")

    fig.savefig(os.path.join(output_dir, "band_weights.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved band_weights.png and band_weights.txt to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (if not embedded in checkpoint)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (must match training split)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num_vis", type=int, default=10,
                        help="Number of samples to visualize")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    elif "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        raise ValueError("Config not found. Provide --config or use a checkpoint with embedded config.")

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.checkpoint), "..", "eval_results"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_dir = cfg["data"]["data_dir"]
    splits = get_data_splits(
        data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        seed=args.seed,
    )

    val_transform = get_val_transforms(cfg)
    ds_kwargs = get_dataset_kwargs(cfg)
    dataset = MSIDataset(
        splits[args.split], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
    )

    # Model
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Band attention weights analysis (auto-skipped if model has no band_attention)
    analyze_band_weights(model, dataloader, device, args.output_dir, cfg["experiment_name"])

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # Evaluate
    results, all_preds, all_masks, all_images, all_stems = evaluate(
        model, dataloader, metrics, device, num_classes
    )

    # Print results
    print_results(results, cfg["experiment_name"])

    # Save results
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results_serializable[k] = v.tolist()
        else:
            results_serializable[k] = v

    import json
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(results, args.output_dir, num_classes)

    # Visualization
    vis_bands = tuple(cfg.get("eval", {}).get("vis_bands", [0, 4, 8]))
    visualize_predictions(
        all_images, all_preds, all_masks, all_stems,
        args.output_dir, vis_bands=vis_bands, num_samples=args.num_vis
    )

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
