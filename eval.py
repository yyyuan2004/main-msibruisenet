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


def _eval_apple_mask_kwarg(model, apple_masks, device):
    """Return apple_mask kwarg dict for model forward if model uses SDA."""
    if getattr(model, 'sda_v2_enabled', False):
        return {"apple_mask": apple_masks.unsqueeze(1).to(device)}
    if getattr(model, 'use_sda_input', False):
        return {"apple_mask": apple_masks.unsqueeze(1).to(device)}
    return {}


@torch.no_grad()
def evaluate(model, dataloader, metrics, device, num_classes):
    """Run evaluation and collect predictions."""
    model.eval()
    metrics.reset()

    all_preds = []
    all_masks = []
    all_images = []
    all_images_raw = []
    all_stems = []

    for images, masks, images_raw, apple_masks, stems in dataloader:
        images_dev = images.to(device)
        masks_dev = masks.to(device)
        sda_kw = _eval_apple_mask_kwarg(model, apple_masks, device)

        logits = model(images_dev, **sda_kw)
        preds = logits.argmax(dim=1)

        metrics.update(preds, masks_dev)

        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.numpy())
        all_images.append(images.numpy())
        all_images_raw.append(images_raw.numpy())
        all_stems.extend(stems)

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_images_raw = np.concatenate(all_images_raw, axis=0)

    results = metrics.compute()
    return results, all_preds, all_masks, all_images, all_images_raw, all_stems


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


def _normalize_band(band):
    """Normalize a single band to [0, 1] using 2-98 percentile clipping."""
    vmin, vmax = np.percentile(band, [2, 98])
    if vmax - vmin > 1e-6:
        return np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return np.zeros_like(band)


def visualize_predictions(images, preds, masks, stems, output_dir,
                          vis_bands=(0, 4, 8), num_samples=10,
                          images_raw=None):
    """Visualize segmentation results with all-band grid + sharpen comparison.

    Layout per sample:
        Row 1: all individual bands of the raw image (9 bands in a row)
        Row 2: pseudo-color (raw) | pseudo-color (processed) | prediction | ground truth
        If images_raw is provided and differs from images (sharpen active),
        the second row shows the before/after comparison.
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))
    cmap = ListedColormap(["black", "red", "blue", "green", "yellow"][:max(preds.max() + 1, 2)])

    for i in range(num_samples):
        img = images[i]  # preprocessed (C, H, W)
        pred = preds[i]
        mask = masks[i]
        stem = stems[i]
        img_raw = images_raw[i] if images_raw is not None else img

        n_bands_raw = img_raw.shape[0]
        n_cols = max(n_bands_raw, 4)

        fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))

        # Row 1: individual raw bands
        for b in range(n_bands_raw):
            axes[0, b].imshow(_normalize_band(img_raw[b]), cmap="gray", vmin=0, vmax=1)
            axes[0, b].set_title(f"Band {b}", fontsize=8)
            axes[0, b].axis("off")
        for b in range(n_bands_raw, n_cols):
            axes[0, b].axis("off")

        # Row 2: pseudo-color raw | pseudo-color processed | prediction | ground truth
        rgb_raw = np.stack([_normalize_band(img_raw[b]) for b in vis_bands], axis=-1)
        rgb_proc = np.stack([_normalize_band(img[b]) for b in vis_bands if b < img.shape[0]], axis=-1)
        if rgb_proc.shape[-1] < 3:
            rgb_proc = rgb_raw

        axes[1, 0].imshow(rgb_raw)
        axes[1, 0].set_title("Raw pseudo-color", fontsize=9)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(rgb_proc)
        axes[1, 1].set_title("Processed pseudo-color", fontsize=9)
        axes[1, 1].axis("off")

        axes[1, 2].imshow(pred, cmap=cmap, vmin=0, vmax=max(preds.max(), 1))
        axes[1, 2].set_title("Prediction", fontsize=9)
        axes[1, 2].axis("off")

        axes[1, 3].imshow(mask, cmap=cmap, vmin=0, vmax=max(masks.max(), 1))
        axes[1, 3].set_title("Ground Truth", fontsize=9)
        axes[1, 3].axis("off")

        for b in range(4, n_cols):
            axes[1, b].axis("off")

        fig.suptitle(stem, fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"{stem}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {num_samples} visualizations to {vis_dir}")


def save_sda_anomaly_maps(model, dataloader, device, output_dir):
    """Save SDA anomaly heatmaps for every val sample.

    Supports both legacy SDA v1 (single anomaly map) and SDA v2
    (multiple named feature maps). Skipped silently if model has neither.
    """
    has_v1 = hasattr(model, 'sda_input')
    has_v2 = getattr(model, 'sda_v2_enabled', False)
    if not has_v1 and not has_v2:
        return

    heatmap_dir = os.path.join(output_dir, "sda_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    model.eval()
    count = 0
    with torch.no_grad():
        for images, _masks, _raw, apple_masks, stems in dataloader:
            images_dev = images.to(device)
            apple_mask_dev = apple_masks.unsqueeze(1).to(device)

            if has_v2:
                feature_maps = model.sda_v2.get_feature_maps(
                    images_dev, apple_mask=apple_mask_dev)
                feature_names = model.sda_v2.feature_names
                for j in range(feature_maps.shape[0]):
                    n_feat = len(feature_names)
                    fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 4))
                    if n_feat == 1:
                        axes = [axes]
                    for fi, fname in enumerate(feature_names):
                        amap = feature_maps[j, fi]
                        axes[fi].imshow(amap, cmap="hot", vmin=0, vmax=1)
                        axes[fi].set_title(fname, fontsize=9)
                        axes[fi].axis("off")
                    fig.suptitle(f"SDA v2 — {stems[j]}", fontsize=11)
                    fig.tight_layout()
                    fig.savefig(os.path.join(
                        heatmap_dir, f"{stems[j]}_sda_v2.png"), dpi=150)
                    plt.close(fig)
                    count += 1
            elif has_v1:
                am_kw = {}
                if getattr(model, 'use_sda_input', False):
                    am_kw["apple_mask"] = apple_mask_dev
                anomaly_maps = model.sda_input.get_anomaly_map(
                    images_dev, **am_kw)
                for j in range(anomaly_maps.shape[0]):
                    amap = anomaly_maps[j, 0]
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(amap, cmap="hot", vmin=0, vmax=1)
                    ax.set_title(f"SDA anomaly — {stems[j]}")
                    ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(
                        heatmap_dir, f"{stems[j]}_anomaly.png"), dpi=150)
                    plt.close(fig)
                    count += 1

    print(f"Saved {count} SDA anomaly heatmaps to {heatmap_dir}")


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
            for images, _, _, _, _ in dataloader:
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

    # SDA anomaly heatmaps (auto-skipped if model has no sda_input)
    save_sda_anomaly_maps(model, dataloader, device, args.output_dir)

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # Evaluate
    results, all_preds, all_masks, all_images, all_images_raw, all_stems = evaluate(
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
        args.output_dir, vis_bands=vis_bands, num_samples=args.num_vis,
        images_raw=all_images_raw,
    )

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
