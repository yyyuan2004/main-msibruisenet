"""Training entry for MSI-BruiseNet (5-fold x multi-seed)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import create_dataloader
from losses import CombinedLoss
from models import build_model
from utils.config import load_config
from utils.metrics import binary_metrics
from utils.seed import set_global_seed


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_epoch(model: torch.nn.Module, loader: Any, criterion: CombinedLoss, optimizer: Any, device: torch.device, train: bool) -> float:
    model.train() if train else model.eval()
    losses: List[float] = []
    for x, y, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model: torch.nn.Module, loader: Any, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_metrics = []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_np = y.numpy()
        for p, t in zip(pred, y_np):
            all_metrics.append(binary_metrics(p, t))
    keys = all_metrics[0].keys() if all_metrics else ["iou", "dice", "f1", "precision", "recall"]
    return {k: float(np.mean([m[k] for m in all_metrics])) if all_metrics else 0.0 for k in keys}


def save_ckpt(model: torch.nn.Module, path: Path, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)

    # ========== 📂 OUTPUT PATH (训练产出) ==========
    # OUTPUT_ROOT = "outputs/"
    # CKPT_DIR    = "outputs/checkpoints/"
    # LOG_DIR     = "outputs/logs/"
    # RESULT_DIR  = "outputs/results/"
    # =====================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["output"]["checkpoints"])
    log_dir = Path(cfg["output"]["logs"])
    result_dir = Path(cfg["output"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selected_seeds = list(cfg["train"]["seeds"])[: int(cfg["train"].get("num_seeds", len(cfg["train"]["seeds"]))) ]
    for seed in selected_seeds:
        set_global_seed(int(seed))
        for fold in range(int(cfg["train"]["num_folds"])):
            logging.info("Start seed=%s fold=%s", seed, fold)
            model = build_model(cfg).to(device)
            if args.resume:
                state = torch.load(args.resume, map_location=device)
                model.load_state_dict(state["model"], strict=False)
            optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
            scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
            criterion = CombinedLoss(cfg)

            train_loader = create_dataloader(cfg, fold=fold, split="train", seed=int(seed), train=True)
            val_loader = create_dataloader(cfg, fold=fold, split="val", seed=int(seed), train=False)
            writer = SummaryWriter(log_dir=str(log_dir / f"{args.tag}_seed{seed}_fold{fold}"))

            best_iou = -1.0
            best_path = ckpt_dir / f"best_{args.tag}_seed{seed}_fold{fold}.pth"
            for epoch in range(int(cfg["train"]["epochs"])):
                tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
                writer.add_scalar("loss/train", tr_loss, epoch)
                if epoch % 10 == 0 or epoch == int(cfg["train"]["epochs"]) - 1:
                    va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
                    metrics = validate(model, val_loader, device)
                    writer.add_scalar("loss/val", va_loss, epoch)
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                    if metrics["iou"] > best_iou:
                        best_iou = metrics["iou"]
                        save_ckpt(model, best_path, epoch, metrics)
                scheduler.step()
            writer.close()

            best_state = torch.load(best_path, map_location=device)
            model.load_state_dict(best_state["model"])
            final_metrics = validate(model, val_loader, device)
            final_metrics.update({"seed": int(seed), "fold": fold, "tag": args.tag})
            rows.append(final_metrics)
            logging.info("Done seed=%s fold=%s iou=%.4f", seed, fold, final_metrics["iou"])

    out_csv = result_dir / "metrics_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["seed", "fold", "iou"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    if rows:
        for k in ["iou", "dice", "f1", "precision", "recall"]:
            vals = np.array([r[k] for r in rows], dtype=np.float32)
            summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    with (result_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
