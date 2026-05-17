"""Evaluation script: compute metrics, confusion matrix, visualize segmentation results,
and generate TP/FP/FN error analysis overlays."""

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
import matplotlib.patches as mpatches

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
    all_images_raw = []
    all_stems = []

    for images, masks, images_raw, _apple_masks, stems in dataloader:
        images_dev = images.to(device, non_blocking=True)
        masks_dev = masks.to(device, non_blocking=True)

        logits = model(images_dev)
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
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))
    cmap = ListedColormap(["black", "red", "blue", "green", "yellow"][:max(preds.max() + 1, 2)])

    for i in range(num_samples):
        img = images[i]
        pred = preds[i]
        mask = masks[i]
        stem = stems[i]
        img_raw = images_raw[i] if images_raw is not None else img

        n_bands_raw = img_raw.shape[0]
        n_cols = max(n_bands_raw, 4)

        fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))

        for b in range(n_bands_raw):
            axes[0, b].imshow(_normalize_band(img_raw[b]), cmap="gray", vmin=0, vmax=1)
            axes[0, b].set_title(f"Band {b}", fontsize=8)
            axes[0, b].axis("off")
        for b in range(n_bands_raw, n_cols):
            axes[0, b].axis("off")

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


def visualize_error_analysis(images, preds, masks, stems, output_dir,
                             vis_bands=(0, 4, 8), num_samples=10):
    """Generate per-sample TP/FP/FN error analysis overlay images.

    IMPORTANT: ``images`` must be the *processed* (post-transform) images,
    NOT images_raw.  preds and masks are computed on the transformed images,
    so the spatial dimensions must match.

    Color coding (for defect class = 1):
        - TP (green):  correctly predicted defect
        - FP (red):    predicted defect but actually background
        - FN (blue):   missed defect (GT=1, pred=0)
        - TN:          transparent (background correctly predicted)

    Output: one PNG per sample with pseudo-color background + colored overlay.
    """
    err_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(err_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        img_raw = images[i]
        pred = preds[i]
        mask = masks[i]
        stem = stems[i]

        # Binary masks for defect class (class=1)
        tp = (pred == 1) & (mask == 1)
        fp = (pred == 1) & (mask == 0)
        fn = (pred == 0) & (mask == 1)

        # Build pseudo-color background
        bands_avail = [b for b in vis_bands if b < img_raw.shape[0]]
        while len(bands_avail) < 3:
            bands_avail.append(bands_avail[-1] if bands_avail else 0)
        bg = np.stack([_normalize_band(img_raw[b]) for b in bands_avail[:3]], axis=-1)

        # Create RGBA overlay
        H, W = mask.shape
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        alpha = 0.55
        overlay[tp] = [0.0, 0.8, 0.0, alpha]   # green = TP
        overlay[fp] = [0.9, 0.0, 0.0, alpha]    # red   = FP
        overlay[fn] = [0.0, 0.2, 0.9, alpha]    # blue  = FN

        # Compute pixel counts for annotation
        n_tp, n_fp, n_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        iou = n_tp / max(n_tp + n_fp + n_fn, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # (0) Pseudo-color input
        axes[0].imshow(bg)
        axes[0].set_title("Input (pseudo-color)", fontsize=11)
        axes[0].axis("off")

        # (1) Error overlay
        axes[1].imshow(bg)
        axes[1].imshow(overlay)
        axes[1].set_title("Error Analysis", fontsize=11)
        axes[1].axis("off")
        legend_patches = [
            mpatches.Patch(color=(0.0, 0.8, 0.0), label=f"TP ({n_tp:,} px)"),
            mpatches.Patch(color=(0.9, 0.0, 0.0), label=f"FP ({n_fp:,} px)"),
            mpatches.Patch(color=(0.0, 0.2, 0.9), label=f"FN ({n_fn:,} px)"),
        ]
        axes[1].legend(handles=legend_patches, loc="lower right", fontsize=8,
                       framealpha=0.8)

        # (2) GT vs Pred side-by-side in a single panel
        combined = np.zeros((H, W, 3), dtype=np.float32)
        combined[mask == 1] = [1.0, 1.0, 1.0]  # GT defect = white
        combined[pred == 1, 0] = 1.0            # Pred defect adds red channel
        axes[2].imshow(combined)
        axes[2].set_title("GT (white) + Pred (red)", fontsize=11)
        axes[2].axis("off")

        fig.suptitle(f"{stem}  |  IoU={iou:.4f}  TP={n_tp}  FP={n_fp}  FN={n_fn}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(err_dir, f"{stem}_error.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {num_samples} error analysis images to {err_dir}")


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
        from utils.config import load_config
        cfg = load_config(args.config)
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
    _nw = cfg["train"].get("num_workers", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=_nw,
        pin_memory=True,
        persistent_workers=_nw > 0,
        prefetch_factor=2 if _nw > 0 else None,
    )

    # Model
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

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

    # TP/FP/FN error analysis overlay (use processed images, not raw —
    # preds/masks are at post-transform resolution)
    visualize_error_analysis(
        all_images, all_preds, all_masks, all_stems,
        args.output_dir, vis_bands=vis_bands, num_samples=args.num_vis,
    )

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
