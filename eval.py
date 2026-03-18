"""Evaluation script: compute metrics, confusion matrix, and visualize segmentation results."""

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

from data.dataset import MSIDataset
from data.augment import get_val_transforms
from data.split import get_data_splits
from model.model import build_model
from utils.metrics import SegmentationMetrics
# [NEW] 推理后处理: 引导滤波锐化 / Unsharp Masking
from utils.postprocess import refine_predictions


@torch.no_grad()
def evaluate(model, dataloader, metrics, device, num_classes,
             postprocess="none", postprocess_cfg=None):
    """Run evaluation and collect predictions.

    [CHANGED] Added postprocess parameter for inference-time refinement.
    When postprocess != "none", applies sharpening/guided filtering to
    predicted probability maps before argmax.

    Args:
        postprocess: "none", "unsharp", "guided", or "unsharp+guided".
        postprocess_cfg: Dict of post-processing parameters (optional).
    """
    model.eval()
    metrics.reset()

    if postprocess_cfg is None:
        postprocess_cfg = {}

    all_preds = []
    all_masks = []
    all_images = []
    all_stems = []

    for images, masks, stems in dataloader:
        images_dev = images.to(device)
        masks_dev = masks.to(device)

        logits = model(images_dev)

        if postprocess != "none":
            # [NEW] 推理后处理: 对softmax概率图做锐化
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds_np = refine_predictions(
                probs, images_np=images.numpy(),
                method=postprocess, **postprocess_cfg
            )
            preds = torch.from_numpy(preds_np).to(device)
        else:
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

    # Annotate cells
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

    # Color map for masks
    cmap = ListedColormap(["black", "red", "blue", "green", "yellow"][:max(preds.max() + 1, 2)])

    for i in range(num_samples):
        img = images[i]   # (C, H, W)
        pred = preds[i]   # (H, W)
        mask = masks[i]   # (H, W)
        stem = stems[i]

        # Create pseudo-color from selected bands
        rgb = np.stack([img[b] for b in vis_bands], axis=-1)  # (H, W, 3)
        # Normalize to [0, 1] for display
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2-UNet segmentation")
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
    # [NEW] 推理后处理选项
    parser.add_argument("--postprocess", type=str, default="none",
                        choices=["none", "unsharp", "guided", "unsharp+guided"],
                        help="Post-processing method for prediction refinement")
    parser.add_argument("--sharpen_sigma", type=float, default=1.0,
                        help="Unsharp mask Gaussian sigma")
    parser.add_argument("--sharpen_strength", type=float, default=1.5,
                        help="Unsharp mask strength")
    parser.add_argument("--guide_radius", type=int, default=4,
                        help="Guided filter radius")
    parser.add_argument("--guide_eps", type=float, default=0.01,
                        help="Guided filter regularization eps")
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
    dataset = MSIDataset(
        splits[args.split], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
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

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # [CHANGED] 传递后处理参数
    postprocess_cfg = {
        "sharpen_sigma": args.sharpen_sigma,
        "sharpen_strength": args.sharpen_strength,
        "guide_radius": args.guide_radius,
        "guide_eps": args.guide_eps,
    }
    # Evaluate
    results, all_preds, all_masks, all_images, all_stems = evaluate(
        model, dataloader, metrics, device, num_classes,
        postprocess=args.postprocess, postprocess_cfg=postprocess_cfg
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
