"""
train.py — Training entry point for MSI-BruiseNet
===================================================

Key I/O:
    Input : configs/config.yaml (hyperparameters, paths)
    Output:
        - outputs/checkpoints/  — model weights (.pth)
        - outputs/logs/         — TensorBoard event files
        - outputs/results/      — metrics CSV/JSON summaries

Supports:
    - 5-fold × 3-seed cross-validation
    - --config for YAML config
    - --override for CLI overrides (e.g. model.attention=se)
    - --resume for checkpoint resumption
    - --tag for experiment naming

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --override model.attention=se --tag ablation_se
    python scripts/train.py --config configs/config.yaml --resume outputs/checkpoints/fold0_seed42_best.pth
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.msi_dataset import get_dataloaders
from datasets.transforms import build_train_transforms, build_val_transforms
from losses.loss import CombinedLoss
from models.build_model import build_model
from utils.metrics import compute_metrics
from utils.seed import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_overrides(override_strs: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply CLI overrides to config dict (e.g. model.attention=se).

    Args:
        override_strs: List of 'key=value' strings with dot-separated keys.
        cfg: Config dict to modify in-place.

    Returns:
        Modified config dict.
    """
    for item in override_strs:
        if "=" not in item:
            logger.warning("Invalid override (no '='): %s", item)
            continue
        key_path, value = item.split("=", 1)
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse value as int/float/bool
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        d[keys[-1]] = value
        logger.info("Override: %s = %s", key_path, value)
    return cfg


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Tuple[float, int]:
    """Train for one epoch.

    Args:
        model: The model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epoch: Current epoch number.
        writer: TensorBoard writer.
        global_step: Running step counter.

    Returns:
        (average_loss, updated_global_step).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if writer is not None:
            writer.add_scalar("train/loss_step", loss.item(), global_step)

    avg_loss = total_loss / max(num_batches, 1)
    if writer is not None:
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: The model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.
        epoch: Current epoch number.
        writer: TensorBoard writer.

    Returns:
        Dict of validation metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for images, masks in tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()

        for p, t in zip(preds, targets):
            all_preds.append(p)
            all_targets.append(t)

    avg_loss = total_loss / max(len(loader), 1)

    # Compute metrics over all validation samples
    from utils.metrics import compute_metrics_batch
    metrics = compute_metrics_batch(all_preds, all_targets)
    metrics["val_loss"] = avg_loss

    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
        for k, v in metrics.items():
            if k != "val_loss":
                writer.add_scalar(f"val/{k}", v, epoch)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        metrics: Current validation metrics.
        path: File path to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info("Saved checkpoint to %s (epoch %d, mIoU=%.4f)", path, epoch, metrics.get("mIoU", 0))


def run_training(cfg: Dict[str, Any], tag: str = "", resume_path: str = "") -> List[Dict[str, Any]]:
    """Run the full training loop: num_folds × num_seeds.

    Args:
        cfg: Full config dict.
        tag: Experiment tag for organizing outputs.
        resume_path: Path to a checkpoint to resume from.

    Returns:
        List of result dicts for each (fold, seed) run.
    """
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    seeds = train_cfg.get("seeds", [42, 123, 2024])
    num_folds = train_cfg.get("num_folds", 5)
    epochs = train_cfg.get("epochs", 200)
    val_interval = train_cfg.get("val_interval", 10)
    input_size = data_cfg.get("input_size", 512)

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for fold in range(num_folds):
            exp_name = f"fold{fold}_seed{seed}"
            if tag:
                exp_name = f"{tag}/{exp_name}"

            logger.info("=" * 60)
            logger.info("Starting %s", exp_name)
            logger.info("=" * 60)

            set_global_seed(seed)

            # Build model
            model = build_model(cfg).to(device)

            # Build data loaders
            spatial_train = build_train_transforms(cfg.get("augmentation", {}), input_size)
            spatial_val = build_val_transforms(input_size)
            train_loader, val_loader = get_dataloaders(
                fold, cfg, spatial_train, spatial_val
            )

            # Optimizer
            lr = train_cfg.get("lr", 1e-4)
            wd = train_cfg.get("weight_decay", 1e-4)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            # Scheduler
            warmup_epochs = train_cfg.get("warmup_epochs", 10)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
            )

            # Loss
            criterion = CombinedLoss(cfg.get("loss", {}))

            # TensorBoard
            log_dir = os.path.join(output_cfg["logs"], exp_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

            # Checkpoint path
            ckpt_dir = os.path.join(output_cfg["checkpoints"], exp_name if tag else "")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Resume
            start_epoch = 0
            best_miou = 0.0
            if resume_path and os.path.isfile(resume_path):
                ckpt = torch.load(resume_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_miou = ckpt.get("metrics", {}).get("mIoU", 0.0)
                logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

            global_step = start_epoch * len(train_loader)

            # Training loop
            for epoch in range(start_epoch, epochs):
                # Warmup: linearly increase LR
                if epoch < warmup_epochs:
                    warmup_lr = lr * (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warmup_lr

                avg_loss, global_step = train_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    epoch, writer, global_step,
                )

                # Step scheduler after warmup
                if epoch >= warmup_epochs:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", current_lr, epoch)

                # Validation
                if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
                    metrics = validate(model, val_loader, criterion, device, epoch, writer)
                    logger.info(
                        "Epoch %d — loss=%.4f, mIoU=%.4f, Dice=%.4f",
                        epoch, avg_loss, metrics["mIoU"], metrics["Dice"],
                    )

                    # Save best checkpoint
                    if metrics["mIoU"] > best_miou:
                        best_miou = metrics["mIoU"]
                        best_path = os.path.join(ckpt_dir, f"{exp_name}_best.pth")
                        save_checkpoint(model, optimizer, epoch, metrics, best_path)

            # Save final checkpoint
            final_path = os.path.join(ckpt_dir, f"{exp_name}_final.pth")
            final_metrics = validate(model, val_loader, criterion, device, epochs - 1, writer)
            save_checkpoint(model, optimizer, epochs - 1, final_metrics, final_path)

            writer.close()

            result = {
                "fold": fold,
                "seed": seed,
                "tag": tag,
                "best_mIoU": best_miou,
                **final_metrics,
            }
            all_results.append(result)
            logger.info("Finished %s — best mIoU=%.4f", exp_name, best_miou)

    return all_results


def save_summary(results: List[Dict[str, Any]], output_dir: str, tag: str = "") -> None:
    """Save aggregated results to CSV and JSON.

    Args:
        results: List of per-experiment result dicts.
        output_dir: Directory to write result files.
        tag: Experiment tag.
    """
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    fname = f"metrics_summary_{tag}.csv" if tag else "metrics_summary.csv"
    csv_path = os.path.join(output_dir, fname)
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved metrics CSV to %s", csv_path)

    # Summary statistics
    metric_keys = ["mIoU", "Dice", "F1", "Precision", "Recall"]
    summary: Dict[str, Dict[str, float]] = {}
    for mk in metric_keys:
        vals = [r[mk] for r in results if mk in r]
        if vals:
            summary[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    json_fname = f"summary_stats_{tag}.json" if tag else "summary_stats.json"
    json_path = os.path.join(output_dir, json_fname)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary stats to %s", json_path)

    # Print summary
    logger.info("=" * 50)
    logger.info("Summary over %d runs:", len(results))
    for mk, stats in summary.items():
        logger.info("  %s: %.4f +/- %.4f", mk, stats["mean"], stats["std"])
    logger.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSI-BruiseNet")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if args.override:
        cfg = parse_overrides(args.override, cfg)

    results = run_training(cfg, tag=args.tag, resume_path=args.resume)
    save_summary(results, cfg["output"]["results"], tag=args.tag)


if __name__ == "__main__":
    main()
