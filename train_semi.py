"""Semi-supervised training with Strong-Weak Mean Teacher framework.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. Supervised Branch (labeled data)                                │
    │    labeled_img → Student → logits → L_sup (CE + Dice + ...)       │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 2. Unsupervised Branch (unlabeled data)                           │
    │    unlabeled_img ──┬── weak_aug  → Teacher → pseudo-label (τ过滤) │
    │                    └── strong_aug → Student → prediction          │
    │    L_unsup = CE(student_pred, filtered_pseudo_label)              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 3. Total Loss                                                     │
    │    L_total = L_sup + λ_u(t) * L_unsup                            │
    │    λ_u(t) ramps up from 0 → λ_max via Gaussian warmup            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 4. EMA Update                                                     │
    │    Teacher_weights = α * Teacher_weights + (1-α) * Student_weights│
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    python train_semi.py --config configs/semi.yaml --seed 42
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.augment import (
    get_train_transforms,
    get_val_transforms,
    get_weak_transforms,
    get_strong_transforms,
)
from data.dataset import MSIDataset
from data.semi_dataset import UnlabeledMSIDataset
from data.split import get_data_splits
from model.loss import SegmentationLoss
from model.model import build_model
from utils.ema import EMATeacher
from utils.metrics import SegmentationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def gaussian_rampup(current_epoch, rampup_length):
    """Gaussian ramp-up curve: 0 → 1 over rampup_length epochs.

    Returns a float in [0, 1] used to scale the unsupervised loss weight.
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current_epoch, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_unlabeled_file_list(unlabeled_dir, image_dir="images"):
    """Scan unlabeled directory for .npy file stems."""
    image_root = os.path.join(unlabeled_dir, image_dir)
    if not os.path.isdir(image_root):
        return []
    stems = sorted([f[:-4] for f in os.listdir(image_root) if f.endswith(".npy")])
    return stems


# ---------------------------------------------------------------------------
# Validation (same as train.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, criterion, metrics, device):
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


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train_semi(cfg, seed, output_dir):
    """Semi-supervised Mean Teacher training loop."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Config shortcuts ----
    train_cfg = cfg["train"]
    semi_cfg = cfg.get("semi", {})
    data_dir = cfg["data"]["data_dir"]

    ema_decay = semi_cfg.get("ema_decay", 0.999)
    pseudo_threshold = semi_cfg.get("pseudo_threshold", 0.80)
    lambda_u_max = semi_cfg.get("lambda_u_max", 1.0)
    rampup_epochs = semi_cfg.get("rampup_epochs", 30)
    unlabeled_dir = semi_cfg.get("unlabeled_dir", "")
    unlabeled_image_dir = semi_cfg.get("unlabeled_image_dir", "images")
    pretrained_ckpt = semi_cfg.get("pretrained_checkpoint", "")

    # ---- Data splits (labeled) ----
    splits = get_data_splits(
        data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        seed=seed,
    )
    print(f"Labeled — Train: {len(splits['train'])}, Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    # ---- Unlabeled data ----
    if unlabeled_dir and os.path.isdir(unlabeled_dir):
        unlabeled_stems = get_unlabeled_file_list(
            unlabeled_dir, unlabeled_image_dir
        )
    else:
        # 如果没有独立的无标签目录，则跳过无监督分支
        unlabeled_stems = []
    print(f"Unlabeled: {len(unlabeled_stems)} samples")

    # ---- Transforms ----
    train_transform = get_train_transforms(cfg)  # for labeled supervised branch
    val_transform = get_val_transforms(cfg)
    weak_transform = get_weak_transforms(cfg)
    strong_transform = get_strong_transforms(cfg)

    # ---- Datasets ----
    train_dataset = MSIDataset(
        splits["train"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=train_transform,
        num_classes=cfg["data"]["num_classes"],
    )
    val_dataset = MSIDataset(
        splits["val"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
    )

    unlabeled_dataset = None
    if unlabeled_stems:
        unlabeled_dataset = UnlabeledMSIDataset(
            unlabeled_stems,
            data_dir=unlabeled_dir,
            image_dir=unlabeled_image_dir,
            weak_transform=weak_transform,
            strong_transform=strong_transform,
        )

    # ---- DataLoaders ----
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    unlabeled_loader = None
    if unlabeled_dataset is not None:
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    # ---- Model (Student) ----
    student = build_model(cfg).to(device)
    param_count = count_parameters(student)
    print(f"Model: {cfg['experiment_name']} | Parameters: {param_count:.2f}M")

    # Load pretrained checkpoint if provided (warm start)
    if pretrained_ckpt and os.path.isfile(pretrained_ckpt):
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded pretrained checkpoint: {pretrained_ckpt}")

    # ---- Teacher (EMA of Student) ----
    teacher = EMATeacher(student, decay=ema_decay).to(device)
    print(f"EMA Teacher initialized (decay={ema_decay})")

    # ---- Loss ----
    criterion = SegmentationLoss(
        loss_type=train_cfg.get("loss", "ce_dice"),
        ce_weight=train_cfg.get("ce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
        spectral_smoothness_weight=train_cfg.get("spectral_smoothness_weight", 0.0),
        edge_preserve_weight=train_cfg.get("edge_preserve_weight", 0.0),
    )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["num_epochs"],
        eta_min=train_cfg.get("eta_min", 1e-6),
    )

    # ---- Metrics & Logging ----
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training state ----
    best_iou = 0.0
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 0)
    no_improve_count = 0
    num_epochs = train_cfg["num_epochs"]

    print(f"\nStarting semi-supervised training for {num_epochs} epochs...")
    print(f"  Pseudo-label threshold τ = {pseudo_threshold}")
    print(f"  λ_u ramp-up: 0 → {lambda_u_max} over {rampup_epochs} epochs")
    if patience > 0:
        print(f"  Early stopping: patience={patience} epochs")

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        student.train()

        # Ramp-up weight for unsupervised loss
        lambda_u = lambda_u_max * gaussian_rampup(epoch, rampup_epochs)

        epoch_sup_loss = 0.0
        epoch_unsup_loss = 0.0
        epoch_total_loss = 0.0
        epoch_pseudo_ratio = 0.0  # fraction of pixels above threshold
        num_labeled_batches = 0
        num_unlabeled_batches = 0

        # Create iterator for unlabeled data (cycles if shorter than labeled)
        unlabeled_iter = None
        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # =============== Step A: Supervised Branch ===============
            logits_sup = student(images)
            loss_sup = criterion(logits_sup, masks)

            # =============== Step B: Unsupervised Branch ==============
            loss_unsup = torch.tensor(0.0, device=device)
            pseudo_ratio = 0.0

            if unlabeled_iter is not None and lambda_u > 0:
                try:
                    weak_imgs, strong_imgs, _ = next(unlabeled_iter)
                except StopIteration:
                    # Restart unlabeled iterator (cycle)
                    unlabeled_iter = iter(unlabeled_loader)
                    weak_imgs, strong_imgs, _ = next(unlabeled_iter)

                weak_imgs = weak_imgs.to(device)
                strong_imgs = strong_imgs.to(device)

                # Teacher: weak augmented → pseudo-label
                teacher_logits = teacher(weak_imgs)  # no grad
                teacher_probs = F.softmax(teacher_logits, dim=1)  # (B, C, H, W)
                max_probs, pseudo_labels = teacher_probs.max(dim=1)  # (B, H, W)

                # Confidence mask: only pixels above threshold
                conf_mask = (max_probs >= pseudo_threshold).float()  # (B, H, W)
                pseudo_ratio = conf_mask.mean().item()

                # Student: strong augmented → prediction
                student_logits = student(strong_imgs)

                # Consistency loss: CE between student prediction and pseudo-label
                # Only on confident pixels (masked)
                if conf_mask.sum() > 0:
                    loss_unsup_unreduced = F.cross_entropy(
                        student_logits, pseudo_labels, reduction="none"
                    )  # (B, H, W)
                    loss_unsup = (loss_unsup_unreduced * conf_mask).sum() / (
                        conf_mask.sum() + 1e-6
                    )

                num_unlabeled_batches += 1
                epoch_pseudo_ratio += pseudo_ratio

            # =============== Total Loss ===============
            loss_total = loss_sup + lambda_u * loss_unsup

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # =============== EMA Update ===============
            teacher.update(student)

            # Accumulate stats
            epoch_sup_loss += loss_sup.item()
            epoch_unsup_loss += loss_unsup.item()
            epoch_total_loss += loss_total.item()
            num_labeled_batches += 1
            global_step += 1

        # ---- End of epoch ----
        scheduler.step()

        avg_sup = epoch_sup_loss / max(num_labeled_batches, 1)
        avg_unsup = epoch_unsup_loss / max(num_unlabeled_batches, 1)
        avg_total = epoch_total_loss / max(num_labeled_batches, 1)
        avg_pseudo_ratio = epoch_pseudo_ratio / max(num_unlabeled_batches, 1)

        # Validate
        val_loss, val_results = validate(
            student, val_loader, criterion, metrics, device
        )
        miou = val_results["mIoU"]
        defect_iou = float(val_results["IoU_per_class"][1])
        elapsed = time.time() - t0

        # TensorBoard logging
        writer.add_scalar("train/loss_sup", avg_sup, epoch)
        writer.add_scalar("train/loss_unsup", avg_unsup, epoch)
        writer.add_scalar("train/loss_total", avg_total, epoch)
        writer.add_scalar("train/lambda_u", lambda_u, epoch)
        writer.add_scalar("train/pseudo_ratio", avg_pseudo_ratio, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("val/IoU_class1", defect_iou, epoch)
        writer.add_scalar("val/F1_macro", val_results["F1_macro"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"L_sup: {avg_sup:.4f} | L_unsup: {avg_unsup:.4f} | "
                f"λ_u: {lambda_u:.3f} | τ_ratio: {avg_pseudo_ratio:.2%} | "
                f"IoU(defect): {defect_iou:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        # Save best model + early stopping (by class1 IoU)
        if defect_iou > best_iou:
            best_iou = defect_iou
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "config": cfg,
            }, os.path.join(ckpt_dir, "best_model.pth"))
        else:
            no_improve_count += 1

        if patience > 0 and no_improve_count >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}: "
                f"IoU(defect)未提升已达 {patience} epochs. "
                f"Best IoU(defect): {best_iou:.4f} at epoch {best_epoch}"
            )
            break

        # Periodic checkpoints
        save_interval = train_cfg.get("save_interval", 50)
        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "config": cfg,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    # Save final model
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_iou": best_iou,
        "config": cfg,
    }, os.path.join(ckpt_dir, "final_model.pth"))

    writer.close()
    print(f"\nTraining complete. Best IoU(defect): {best_iou:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {ckpt_dir}")

    return {
        "experiment": cfg["experiment_name"],
        "seed": seed,
        "params_M": param_count,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Semi-supervised Mean Teacher training for MSI segmentation"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs", f"{cfg['experiment_name']}_seed{args.seed}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    result = train_semi(cfg, args.seed, args.output_dir)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
