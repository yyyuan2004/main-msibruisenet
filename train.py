"""Training script for MobileNetV2-UNet MSI segmentation with ablation support."""

import argparse
import os
import sys
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import get_train_transforms, get_val_transforms
from data.split import get_data_splits
from model.model import build_model
from model.loss import SegmentationLoss
from utils.metrics import SegmentationMetrics


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, metrics, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metrics.reset()

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    results = metrics.compute()
    return avg_loss, results


def train(cfg, seed, output_dir):
    """Main training loop."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data splits
    data_dir = cfg["data"]["data_dir"]
    splits = get_data_splits(
        data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        seed=seed,
    )
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    # Transforms
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # Datasets
    ds_kwargs = get_dataset_kwargs(cfg)
    train_dataset = MSIDataset(
        splits["train"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=train_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )
    val_dataset = MSIDataset(
        splits["val"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )

    # DataLoaders
    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    # Model
    model = build_model(cfg).to(device)
    param_count = count_parameters(model)
    print(f"Model: {cfg['experiment_name']} | Parameters: {param_count:.2f}M")

    # Loss
    # [CHANGED v2] Added edge_preserve_weight for fused config
    # 总loss = CE + Dice + SpectralSmoothnessLoss + EdgePreservingLoss
    criterion = SegmentationLoss(
        loss_type=train_cfg.get("loss", "ce_dice"),
        ce_weight=train_cfg.get("ce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
        spectral_smoothness_weight=train_cfg.get("spectral_smoothness_weight", 0.0),
        edge_preserve_weight=train_cfg.get("edge_preserve_weight", 0.0),  # [NEW]
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["num_epochs"],
        eta_min=train_cfg.get("eta_min", 1e-6),
    )

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # TensorBoard
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Checkpoint dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_miou = 0.0
    best_epoch = 0
    # Early stopping: 连续 patience 个epoch mIoU无提升则停止训练
    patience = train_cfg.get("early_stopping_patience", 0)  # 0=禁用
    no_improve_count = 0

    print(f"\nStarting training for {train_cfg['num_epochs']} epochs...")
    if patience > 0:
        print(f"Early stopping enabled: patience={patience} epochs")
    for epoch in range(1, train_cfg["num_epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_results = validate(model, val_loader, criterion, metrics, device)

        # Step scheduler
        scheduler.step()

        miou = val_results["mIoU"]
        # 用 class 1 (缺陷类) 的 IoU 作为主指标，而非 mIoU
        defect_iou = float(val_results["IoU_per_class"][1])
        elapsed = time.time() - t0

        # Log to TensorBoard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("val/IoU_class1", defect_iou, epoch)
        writer.add_scalar("val/F1_macro", val_results["F1_macro"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{train_cfg['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"IoU(defect): {defect_iou:.4f} | F1: {val_results['F1_macro']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model + early stopping check (以 class1 IoU 为判据)
        if defect_iou > best_miou:
            best_miou = defect_iou
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "config": cfg,
            }, os.path.join(ckpt_dir, "best_model.pth"))
        else:
            no_improve_count += 1

        if patience > 0 and no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch}: "
                  f"IoU(defect)未提升已达 {patience} epochs. "
                  f"Best IoU(defect): {best_miou:.4f} at epoch {best_epoch}")
            break

        # Periodic checkpoints
        save_interval = train_cfg.get("save_interval", 50)
        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "config": cfg,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    # Save final model
    torch.save({
        "epoch": train_cfg["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou": best_miou,
        "config": cfg,
    }, os.path.join(ckpt_dir, "final_model.pth"))

    writer.close()

    print(f"\nTraining complete. Best IoU(defect): {best_miou:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {ckpt_dir}")

    # Return info for ablation summary
    return {
        "experiment": cfg["experiment_name"],
        "seed": seed,
        "params_M": param_count,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2-UNet for MSI segmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<experiment>_seed<seed>)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs", f"{cfg['experiment_name']}_seed{args.seed}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config copy
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    result = train(cfg, args.seed, args.output_dir)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
