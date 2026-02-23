"""Evaluation script for MSI-BruiseNet inference and reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from thop import profile
from tqdm import tqdm

from datasets import create_dataloader
from models import build_model
from utils.config import load_config
from utils.metrics import binary_metrics, stratify_by_area
from utils.visualize import save_overlay


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    loader = create_dataloader(cfg, fold=args.fold, split="val", seed=args.seed, train=False)

    pred_dir = Path(cfg["output"]["predictions"])
    result_dir = Path(cfg["output"]["results"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: List[Dict[str, float]] = []
    preds, targets = [], []
    for x, y, sid in tqdm(loader):
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_np = y.numpy()
        for p, t, s in zip(pred, y_np, sid):
            preds.append(p)
            targets.append(t)
            metrics_all.append(binary_metrics(p, t))
            np.save(pred_dir / f"{s}_pred.npy", p.astype(np.uint8))
            save_overlay(np.transpose(x[0].cpu().numpy(), (1, 2, 0)), p, t, pred_dir / f"{s}_overlay.png")

    mean_metrics = {k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0].keys()} if metrics_all else {}
    with (result_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(mean_metrics, f, indent=2)

    area_metrics = stratify_by_area(preds, targets, cfg["evaluation"]["area_bins_cm2"], float(cfg["evaluation"]["pixel_per_cm"]))
    with (result_dir / "metrics_by_area.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_cm2", "iou"])
        for k, v in area_metrics.items():
            w.writerow([k, v])

    dummy = torch.randn(1, int(cfg["data"]["num_channels"]), int(cfg["data"]["input_size"]), int(cfg["data"]["input_size"]), device=device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    with (result_dir / "model_complexity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["params", "flops"])
        w.writerow([int(params), int(flops)])


if __name__ == "__main__":
    main()
