#!/usr/bin/env python3
"""
train.py - Main training entry point for MSI-BruiseNet.

Responsibility:
    - Run 5-fold cross-validation × 3 random seeds (configurable).
    - Train MSI-BruiseNet with AdamW + cosine LR scheduler + warmup.
    - Log metrics to TensorBoard.
    - Save best checkpoints per fold/seed.
    - Support --resume for checkpoint continuation.
    - Support --override for config overrides (e.g., model.attention=se).
    - Support --tag for experiment naming.

I/O:
    Input  : configs/config.yaml, data/images/*.npy, data/masks/*.npy, data/splits/splits.json
    Output : outputs/checkpoints/*.pth, outputs/logs/, outputs/results/
"""

import argparse
import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import get_dataloaders
from losses import build_loss
from models import build_model
from utils.metrics import compute_metrics, aggregate_metrics
from utils.seed import set_global_seed

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MSI-BruiseNet")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Config overrides in key=value format (e.g., model.attention=se)")
    parser.add_argument("--tag", type=str, default="default",
                        help="Experiment tag for organizing outputs")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only a specific fold (0-indexed). Default: all folds.")
    parser.add_argument("--seed-idx", type=int, default=None,
                        help="Run only a specific seed index. Default: all seeds.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file.

    Args:
        config_path: Path to config YAML.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-notation config overrides.

    Args:
        cfg: Base config dict.
        overrides: List of "key.subkey=value" strings.

    Returns:
        Modified config dict.
    """
    for item in overrides:
        if "=" not in item:
            logger.warning("Skipping invalid override: %s", item)
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse value as appropriate type
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[keys[-1]] = parsed
        logger.info("Override: %s = %s", key, parsed)
    return cfg


def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """Create warmup + cosine annealing LR scheduler.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch.

    Args:
        model: The segmentation model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epoch: Current epoch number.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate model on validation set.

    Args:
        model: The segmentation model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        (val_loss, metrics_dict).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_metrics: List[Dict[str, float]] = []

    for batch in tqdm(loader, desc="[Val]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        num_batches += 1

        # Compute per-sample metrics
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        for i in range(preds.shape[0]):
            m = compute_metrics(preds[i], targets[i])
            all_metrics.append(m)

    avg_loss = total_loss / max(num_batches, 1)
    agg = aggregate_metrics(all_metrics)
    avg_metrics = {k: v["mean"] for k, v in agg.items()}
    return avg_loss, avg_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    fold: int,
    seed: int,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        epoch: Current epoch.
        metrics: Current validation metrics.
        save_path: File path to save.
        fold: Current fold index.
        seed: Current seed value.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "fold": fold,
        "seed": seed,
    }, save_path)
    logger.info("Checkpoint saved to %s", save_path)


def train_single_run(
    cfg: Dict[str, Any],
    fold: int,
    seed: int,
    tag: str,
    resume_path: Optional[str] = None,
) -> Dict[str, float]:
    """Train a single fold/seed combination.

    Args:
        cfg: Full config dict.
        fold: Fold index.
        seed: Random seed.
        tag: Experiment tag.
        resume_path: Optional checkpoint path to resume from.

    Returns:
        Best validation metrics dict.
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training fold=%d seed=%d on %s", fold, seed, device)

    # ========== 📂 OUTPUT PATHS (从 config.yaml 读取) ==========
    ckpt_dir = cfg["output"]["checkpoints"]
    log_dir = cfg["output"]["logs"]
    # ============================================================

    run_name = f"{tag}_fold{fold}_seed{seed}"
    run_ckpt_dir = os.path.join(ckpt_dir, run_name)
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    # Build components
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    train_loader, val_loader = get_dataloaders(cfg, fold)

    tcfg = cfg["train"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=tcfg.get("warmup_epochs", 10),
        total_epochs=tcfg["epochs"],
    )

    writer = SummaryWriter(log_dir=run_log_dir)

    start_epoch = 0
    best_miou = 0.0
    best_metrics: Dict[str, float] = {}

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("metrics", {}).get("mIoU", 0.0)
        logger.info("Resumed from epoch %d (best mIoU=%.4f)", start_epoch, best_miou)

    # Training loop
    val_interval = tcfg.get("val_interval", 10)
    for epoch in range(start_epoch, tcfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Validation
        if (epoch + 1) % val_interval == 0 or epoch == tcfg["epochs"] - 1:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            writer.add_scalar("val/loss", val_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | mIoU=%.4f | Dice=%.4f",
                epoch + 1, tcfg["epochs"], train_loss, val_loss,
                val_metrics.get("mIoU", 0), val_metrics.get("Dice", 0),
            )

            # Save best model
            current_miou = val_metrics.get("mIoU", 0)
            if current_miou > best_miou:
                best_miou = current_miou
                best_metrics = val_metrics.copy()
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    os.path.join(run_ckpt_dir, "best.pth"),
                    fold, seed,
                )

        # Save latest checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metrics,
                os.path.join(run_ckpt_dir, "latest.pth"),
                fold, seed,
            )

    writer.close()
    logger.info("Fold %d Seed %d complete. Best mIoU=%.4f", fold, seed, best_miou)
    return best_metrics


def main() -> None:
    """Main training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    tcfg = cfg["train"]
    seeds = tcfg["seeds"]
    num_folds = tcfg["num_folds"]

    # Determine which folds and seeds to run
    folds = [args.fold] if args.fold is not None else list(range(num_folds))
    seed_indices = [args.seed_idx] if args.seed_idx is not None else list(range(len(seeds)))

    all_results: List[Dict[str, Any]] = []

    for seed_idx in seed_indices:
        seed = seeds[seed_idx]
        for fold in folds:
            logger.info("=" * 60)
            logger.info("Starting: tag=%s fold=%d seed=%d", args.tag, fold, seed)
            logger.info("=" * 60)

            best_metrics = train_single_run(
                cfg, fold, seed, args.tag, resume_path=args.resume,
            )
            all_results.append({
                "tag": args.tag,
                "fold": fold,
                "seed": seed,
                **best_metrics,
            })

    # ========== 📂 RESULTS OUTPUT ==========
    result_dir = cfg["output"]["results"]
    os.makedirs(result_dir, exist_ok=True)

    # Save per-run results CSV
    csv_path = os.path.join(result_dir, f"metrics_summary_{args.tag}.csv")
    if all_results:
        keys = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=keys)
            writer_csv.writeheader()
            writer_csv.writerows(all_results)
        logger.info("Results saved to %s", csv_path)

    # Compute and log aggregated statistics
    metric_keys = [k for k in all_results[0] if k not in ("tag", "fold", "seed")]
    agg = aggregate_metrics([{k: r[k] for k in metric_keys} for r in all_results])
    logger.info("=" * 60)
    logger.info("Aggregated results (%d runs):", len(all_results))
    for k, v in agg.items():
        logger.info("  %s: %.4f ± %.4f", k, v["mean"], v["std"])
    logger.info("=" * 60)

    # Save aggregated JSON
    agg_path = os.path.join(result_dir, f"aggregated_{args.tag}.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Aggregated metrics saved to %s", agg_path)


if __name__ == "__main__":
    main()
