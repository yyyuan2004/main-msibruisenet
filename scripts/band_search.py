"""C(9, k) 波段子集穷举搜索脚本。

遍历所有 C(9,k) 波段组合，对每组 band_indices 训练一个轻量模型（少 epoch），
在 val set 上评估 class-1 IoU，找出最优波段子集。

用法:
    python scripts/band_search.py --config configs/baseline.yaml --k 4
    python scripts/band_search.py --config configs/baseline.yaml --k 3 --epochs 50
"""

import argparse
import itertools
import json
import os
import sys
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import get_train_transforms, get_val_transforms
from data.split import get_data_splits
from model.model import build_model
from model.loss import SegmentationLoss
from utils.metrics import SegmentationMetrics
from train import set_seed


def train_and_eval(cfg, seed, band_indices, num_epochs, device):
    """Train a model with given band_indices for num_epochs and return val class-1 IoU."""
    set_seed(seed)

    data_dir = cfg["data"]["data_dir"]
    splits = get_data_splits(
        data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        seed=seed,
    )

    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    ds_kwargs = get_dataset_kwargs(cfg)
    ds_kwargs["band_indices"] = list(band_indices)

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

    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    cfg_copy = json.loads(json.dumps(cfg))
    cfg_copy["data"]["num_channels"] = len(band_indices)
    model = build_model(cfg_copy).to(device)

    criterion = SegmentationLoss(
        loss_type=train_cfg.get("loss", "ce_dice"),
        ce_weight=train_cfg.get("ce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    best_iou = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, masks, _raw, _stems in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        metrics.reset()
        with torch.no_grad():
            for images, masks, _raw, _stems in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images).argmax(dim=1)
                metrics.update(preds, masks)
        results = metrics.compute()
        defect_iou = float(results["IoU_per_class"][1])
        if defect_iou > best_iou:
            best_iou = defect_iou

    return best_iou


def main():
    parser = argparse.ArgumentParser(description="Band subset exhaustive search")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML")
    parser.add_argument("--k", type=int, required=True, help="Number of bands to select")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per combination (quick search)")
    parser.add_argument("--output", type=str, default="outputs/band_search_results.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    total_bands = cfg["data"].get("num_channels", 9)
    combos = list(itertools.combinations(range(total_bands), args.k))
    print(f"Band search: C({total_bands}, {args.k}) = {len(combos)} combinations")
    print(f"Epochs per combo: {args.epochs}, seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    best_iou = 0.0
    best_combo = None

    for i, combo in enumerate(combos):
        t0 = time.time()
        iou = train_and_eval(cfg, args.seed, combo, args.epochs, device)
        elapsed = time.time() - t0

        results.append({"bands": list(combo), "iou": iou})
        if iou > best_iou:
            best_iou = iou
            best_combo = combo

        print(f"[{i+1}/{len(combos)}] bands={list(combo)} IoU={iou:.4f} "
              f"(best={best_iou:.4f} @ {list(best_combo)}) {elapsed:.1f}s")

    results.sort(key=lambda x: x["iou"], reverse=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "k": args.k,
            "total_bands": total_bands,
            "seed": args.seed,
            "epochs_per_combo": args.epochs,
            "best_bands": list(best_combo),
            "best_iou": best_iou,
            "all_results": results,
        }, f, indent=2)

    print(f"\nBest {args.k}-band subset: {list(best_combo)} with IoU={best_iou:.4f}")
    print(f"Top 5 results:")
    for r in results[:5]:
        print(f"  bands={r['bands']} IoU={r['iou']:.4f}")
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
