#!/usr/bin/env python3
"""
evaluate.py - Inference and evaluation script for MSI-BruiseNet.

Responsibility:
    - Load a trained checkpoint and run inference on validation/test data.
    - Compute per-sample and aggregated metrics (mIoU, Dice, F1, Precision, Recall).
    - Compute area-stratified evaluation.
    - Compute model complexity (Params, FLOPs, FPS).
    - Save predictions as .npy and overlay visualizations as .png.
    - Save metrics to CSV/JSON.

I/O:
    Input  : checkpoint .pth, config.yaml, data/*.npy
    Output : outputs/predictions/, outputs/results/
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import get_dataloaders
from models import build_model
from utils.metrics import (
    compute_metrics,
    compute_area_stratified_metrics,
    aggregate_metrics,
)
from utils.visualize import plot_prediction_overlay
from utils.seed import set_global_seed

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MSI-BruiseNet")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index to evaluate on")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Config overrides in key=value format")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save prediction .npy and overlay .png")
    parser.add_argument("--tag", type=str, default="eval",
                        help="Evaluation tag for output naming")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-notation config overrides."""
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[keys[-1]] = parsed
    return cfg


def compute_model_complexity(
    model: nn.Module,
    input_size: tuple = (1, 9, 512, 512),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Compute model parameters, FLOPs, and inference FPS.

    Args:
        model: The segmentation model.
        input_size: Input tensor shape (B, C, H, W).
        device: Compute device.

    Returns:
        Dict with params, flops, fps.
    """
    from thop import profile

    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # FLOPs and params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # FPS measurement (average over 50 forward passes after 10 warmups)
    model.to(device)
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        n_runs = 50
        for _ in range(n_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
    fps = n_runs / elapsed

    return {
        "params_M": params / 1e6,
        "flops_G": flops / 1e9,
        "fps": fps,
    }


@torch.no_grad()
def run_evaluation(
    cfg: Dict[str, Any],
    model: nn.Module,
    fold: int,
    device: torch.device,
    save_predictions: bool = False,
    tag: str = "eval",
) -> Dict[str, Any]:
    """Run full evaluation pipeline.

    Args:
        cfg: Config dict.
        model: Loaded model.
        fold: Fold index.
        device: Compute device.
        save_predictions: Whether to save prediction files.
        tag: Output tag.

    Returns:
        Dict with all evaluation results.
    """
    model.eval()
    _, val_loader = get_dataloaders(cfg, fold)

    # ========== 📂 OUTPUT PATHS ==========
    pred_dir = os.path.join(cfg["output"]["predictions"], tag)
    result_dir = os.path.join(cfg["output"]["results"])
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    # =======================================

    ecfg = cfg["evaluation"]
    area_bins = ecfg.get("area_bins_cm2", [0, 1.0, 3.0, 1000.0])
    pixel_per_cm = ecfg.get("pixel_per_cm", 50.0)

    all_metrics: List[Dict[str, float]] = []
    all_area_metrics: List[Dict[str, Any]] = []

    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        names = batch["name"]

        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = masks.numpy()

        for i in range(preds.shape[0]):
            pred_i = preds[i]
            target_i = targets[i]
            name_i = names[i]

            # Per-sample metrics
            m = compute_metrics(pred_i, target_i)
            m["name"] = name_i
            all_metrics.append(m)

            # Area-stratified metrics
            area_m = compute_area_stratified_metrics(
                pred_i, target_i, area_bins, pixel_per_cm
            )
            all_area_metrics.append({"name": name_i, **area_m})

            # Save predictions
            if save_predictions:
                np.save(os.path.join(pred_dir, f"{name_i}_pred.npy"), pred_i)
                # Load original image for visualization
                img_path = os.path.join(cfg["data"]["image_dir"], f"{name_i}.npy")
                if os.path.exists(img_path):
                    msi_img = np.load(img_path).astype(np.float32)
                    plot_prediction_overlay(
                        msi_img, target_i, pred_i,
                        os.path.join(pred_dir, f"{name_i}_overlay.png"),
                        sample_name=name_i,
                    )

    # Aggregate metrics
    metric_keys = [k for k in all_metrics[0] if k != "name"]
    agg = aggregate_metrics([{k: m[k] for k in metric_keys} for m in all_metrics])

    # Save per-sample CSV
    csv_path = os.path.join(result_dir, f"metrics_per_sample_{tag}.csv")
    if all_metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)
    logger.info("Per-sample metrics saved to %s", csv_path)

    # Save aggregated metrics JSON
    agg_path = os.path.join(result_dir, f"metrics_aggregated_{tag}.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Aggregated metrics saved to %s", agg_path)

    # Save area-stratified metrics CSV
    area_csv_path = os.path.join(result_dir, f"metrics_by_area_{tag}.csv")
    if all_area_metrics:
        with open(area_csv_path, "w", newline="") as f:
            # Flatten nested dicts for CSV
            flat_rows = []
            for row in all_area_metrics:
                flat = {"name": row["name"]}
                for k, v in row.items():
                    if k == "name":
                        continue
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}_{kk}"] = vv
                    else:
                        flat[k] = v
                flat_rows.append(flat)
            if flat_rows:
                writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)
    logger.info("Area-stratified metrics saved to %s", area_csv_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("Evaluation Results (%d samples):", len(all_metrics))
    for k, v in agg.items():
        logger.info("  %s: %.4f +/- %.4f", k, v["mean"], v["std"])
    logger.info("=" * 60)

    return {"per_sample": all_metrics, "aggregated": agg, "area_stratified": all_area_metrics}


def main() -> None:
    """Main evaluation entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build and load model
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", args.checkpoint, ckpt.get("epoch", -1))

    # Model complexity
    logger.info("Computing model complexity...")
    complexity = compute_model_complexity(model, device=device)
    logger.info("Params: %.2f M | FLOPs: %.2f G | FPS: %.1f",
                complexity["params_M"], complexity["flops_G"], complexity["fps"])

    # Save complexity
    result_dir = cfg["output"]["results"]
    os.makedirs(result_dir, exist_ok=True)
    complexity_path = os.path.join(result_dir, f"model_complexity_{args.tag}.json")
    with open(complexity_path, "w") as f:
        json.dump(complexity, f, indent=2)

    # Run evaluation
    run_evaluation(
        cfg, model, args.fold, device,
        save_predictions=args.save_predictions,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
