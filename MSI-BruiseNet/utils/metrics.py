"""Metrics for segmentation, including area-stratified analysis."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def binary_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """Compute IoU, Dice, F1, Precision and Recall."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return {"iou": float(iou), "dice": float(dice), "f1": float(dice), "precision": float(precision), "recall": float(recall)}


def stratify_by_area(
    preds: Iterable[np.ndarray],
    targets: Iterable[np.ndarray],
    area_bins_cm2: List[float],
    pixel_per_cm: float,
) -> Dict[str, float]:
    """Compute IoU grouped by lesion area bins (cm²)."""
    bins = {f"{area_bins_cm2[i]}-{area_bins_cm2[i+1]}": [] for i in range(len(area_bins_cm2) - 1)}
    for pred, target in zip(preds, targets):
        area_px = float((target > 0).sum())
        area_cm2 = area_px / (pixel_per_cm ** 2)
        metrics = binary_metrics(pred, target)
        for i in range(len(area_bins_cm2) - 1):
            lo, hi = area_bins_cm2[i], area_bins_cm2[i + 1]
            if lo <= area_cm2 < hi:
                bins[f"{lo}-{hi}"].append(metrics["iou"])
                break
    return {k: float(np.mean(v)) if v else float("nan") for k, v in bins.items()}
