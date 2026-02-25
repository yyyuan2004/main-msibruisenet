"""
evaluate.py — Inference + metrics computation + area-stratified evaluation
===========================================================================

Key I/O:
    Input:
        - configs/config.yaml
        - outputs/checkpoints/*.pth    (trained model weights)
        - data/images/, data/masks/    (test data)
    Output:
        - outputs/predictions/         (predicted masks .npy + overlay .png)
        - outputs/results/             (metrics CSV / JSON)
            ├── metrics_summary.csv
            ├── metrics_by_area.csv
            └── model_complexity.csv   (Params / FLOPs / FPS)

Usage:
    python scripts/evaluate.py --config configs/config.yaml \\
        --checkpoint outputs/checkpoints/fold0_seed42_best.pth
    python scripts/evaluate.py --config configs/config.yaml --all-folds
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.msi_dataset import MSIDataset, load_fold_split
from datasets.transforms import build_val_transforms
from models.build_model import build_model
from utils.metrics import compute_metrics, compute_area_stratified_metrics
from utils.visualize import overlay_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_model_complexity(
    model: torch.nn.Module, input_size: int = 512, num_channels: int = 9
) -> Dict[str, Any]:
    """Compute model parameters, FLOPs, and FPS.

    Args:
        model: The model.
        input_size: Spatial input size.
        num_channels: Number of input channels.

    Returns:
        Dict with 'params_M', 'flops_G', 'fps'.
    """
    device = next(model.parameters()).device
    dummy = torch.randn(1, num_channels, input_size, input_size, device=device)

    # Parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # FLOPs (requires thop)
    flops = 0.0
    try:
        from thop import profile
        flops_val, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = flops_val / 1e9
    except ImportError:
        logger.warning("thop not installed; FLOPs will be 0. Install with: pip install thop")

    # FPS
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        num_runs = 50
        for _ in range(num_runs):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        fps = num_runs / (t1 - t0)

    return {"params_M": round(params, 2), "flops_G": round(flops, 2), "fps": round(fps, 1)}


def evaluate_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    fold: int = 0,
    save_predictions: bool = True,
    tag: str = "",
) -> Dict[str, Any]:
    """Evaluate a single checkpoint on its validation fold.

    Args:
        cfg: Full config dict.
        checkpoint_path: Path to .pth checkpoint.
        fold: Fold index for validation data.
        save_predictions: Whether to save prediction masks and overlays.
        tag: Experiment tag for output organization.

    Returns:
        Dict of evaluation metrics.
    """
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]
    eval_cfg = cfg.get("evaluation", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = data_cfg.get("input_size", 512)

    # Build model and load weights
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)

    # Load validation data
    _, val_names = load_fold_split(data_cfg["split_dir"], fold)
    spatial_val = build_val_transforms(input_size)
    val_ds = MSIDataset(
        sample_names=val_names,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        norm_stats_path=data_cfg.get("norm_stats"),
        spatial_transform=spatial_val,
    )

    # ========== 📂 OUTPUT PATH (推理结果) ==========
    pred_dir = os.path.join(output_cfg["predictions"], tag) if tag else output_cfg["predictions"]
    result_dir = output_cfg["results"]
    # =====================================================
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Inference
    all_metrics: List[Dict[str, float]] = []
    all_area_metrics: List[Dict[str, Any]] = []

    for idx in tqdm(range(len(val_ds)), desc="Evaluating"):
        image_tensor, mask_tensor = val_ds[idx]
        image_batch = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_batch)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        target = mask_tensor.numpy()

        # Per-sample metrics
        m = compute_metrics(pred, target, data_cfg.get("num_classes", 2))
        m["sample"] = val_names[idx]
        all_metrics.append(m)

        # Area-stratified metrics
        area_m = compute_area_stratified_metrics(
            pred, target,
            area_bins_cm2=eval_cfg.get("area_bins_cm2", [0, 1.0, 3.0, 1000.0]),
            pixel_per_cm=eval_cfg.get("pixel_per_cm", 50.0),
        )
        all_area_metrics.append({"sample": val_names[idx], **area_m})

        # Save predictions
        if save_predictions:
            np.save(os.path.join(pred_dir, f"{val_names[idx]}_pred.npy"), pred.astype(np.uint8))
            # Overlay visualization (use band 4 ≈ 805nm)
            raw_img = np.load(os.path.join(data_cfg["image_dir"], f"{val_names[idx]}.npy"))
            band_idx = min(4, raw_img.shape[-1] - 1)
            band = raw_img[:, :, band_idx] if raw_img.ndim == 3 else raw_img[band_idx]
            overlay_prediction(
                band, pred, target,
                save_path=os.path.join(pred_dir, f"{val_names[idx]}_overlay.png"),
                title=f"{val_names[idx]} (mIoU={m['mIoU']:.3f})",
            )

    # Aggregate metrics
    metric_keys = ["mIoU", "IoU_bruise", "Dice", "F1", "Precision", "Recall"]
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in metric_keys}
    std_metrics = {f"{k}_std": float(np.std([m[k] for m in all_metrics])) for k in metric_keys}

    # Save per-sample CSV
    csv_path = os.path.join(result_dir, f"eval_per_sample{'_' + tag if tag else ''}.csv")
    if all_metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        logger.info("Saved per-sample metrics to %s", csv_path)

    # Save area-stratified CSV
    area_csv_path = os.path.join(result_dir, f"metrics_by_area{'_' + tag if tag else ''}.csv")
    with open(area_csv_path, "w") as f:
        json.dump(all_area_metrics, f, indent=2)
    logger.info("Saved area-stratified metrics to %s", area_csv_path)

    # Model complexity
    complexity = compute_model_complexity(model, input_size, data_cfg.get("num_channels", 9))
    complexity_path = os.path.join(result_dir, "model_complexity.csv")
    with open(complexity_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=complexity.keys())
        writer.writeheader()
        writer.writerow(complexity)
    logger.info("Saved model complexity to %s", complexity_path)

    result = {**avg_metrics, **std_metrics, **complexity, "fold": fold, "checkpoint": checkpoint_path}
    logger.info("Evaluation results: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in result.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MSI-BruiseNet")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag")
    parser.add_argument("--no-save", action="store_true", help="Skip saving predictions")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if args.override:
        from scripts.train import parse_overrides
        cfg = parse_overrides(args.override, cfg)

    if not args.checkpoint:
        logger.error("Please provide --checkpoint path")
        sys.exit(1)

    evaluate_checkpoint(
        cfg,
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        save_predictions=not args.no_save,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
