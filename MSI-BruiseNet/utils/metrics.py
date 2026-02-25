"""
metrics.py - Evaluation metrics for semantic segmentation.

Implements:
    - IoU (Intersection over Union) per class and mean
    - Dice coefficient
    - F1 / Precision / Recall
    - Area-stratified evaluation (by bruise region size in cm²)

I/O:
    Input  : prediction (B, H, W) or (H, W), ground truth (H, W)
    Output : dict of metric values
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def compute_confusion(
    pred: np.ndarray, target: np.ndarray, num_classes: int = 2
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: Predicted labels (H, W), integer.
        target: Ground truth labels (H, W), integer.
        num_classes: Number of classes.

    Returns:
        (num_classes, num_classes) confusion matrix.
    """
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def iou_from_confusion(hist: np.ndarray) -> np.ndarray:
    """Compute per-class IoU from confusion matrix.

    Args:
        hist: (num_classes, num_classes) confusion matrix.

    Returns:
        Per-class IoU array.
    """
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / (union + 1e-8)
    return iou


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute standard segmentation metrics.

    Args:
        pred: (H, W) predicted labels.
        target: (H, W) ground truth labels.
        num_classes: Number of classes.

    Returns:
        Dict with keys: mIoU, IoU_bg, IoU_bruise, Dice, F1, Precision, Recall.
    """
    hist = compute_confusion(pred, target, num_classes)
    iou = iou_from_confusion(hist)

    # Bruise class = 1
    tp = hist[1, 1]
    fp = hist[0, 1]
    fn = hist[1, 0]

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return {
        "mIoU": float(iou.mean()),
        "IoU_bg": float(iou[0]),
        "IoU_bruise": float(iou[1]),
        "Dice": float(dice),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
    }


def compute_area_stratified_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    area_bins_cm2: List[float],
    pixel_per_cm: float = 50.0,
) -> Dict[str, Dict[str, float]]:
    """Compute IoU stratified by bruise region area.

    Args:
        pred: (H, W) predicted labels.
        target: (H, W) ground truth labels.
        area_bins_cm2: Area bin boundaries in cm² (e.g., [0, 1.0, 3.0, 1000.0]).
        pixel_per_cm: Pixels per centimeter for area conversion.

    Returns:
        Dict mapping bin names to per-bin metrics.
    """
    pixel_area_cm2 = 1.0 / (pixel_per_cm ** 2)

    # Label connected components in ground truth
    labeled, num_features = ndimage.label(target == 1)
    results: Dict[str, Dict[str, float]] = {}

    for bin_idx in range(len(area_bins_cm2) - 1):
        lo = area_bins_cm2[bin_idx]
        hi = area_bins_cm2[bin_idx + 1]
        bin_name = f"{lo:.1f}-{hi:.1f}cm2"

        # Create mask for regions in this size bin
        bin_mask = np.zeros_like(target, dtype=bool)
        for region_id in range(1, num_features + 1):
            region_pixels = (labeled == region_id).sum()
            region_area_cm2 = region_pixels * pixel_area_cm2
            if lo <= region_area_cm2 < hi:
                bin_mask |= (labeled == region_id)

        if not bin_mask.any():
            results[bin_name] = {"IoU": float("nan"), "count": 0}
            continue

        # Compute IoU only within this bin's regions
        pred_in_bin = pred[bin_mask]
        target_in_bin = target[bin_mask]
        tp = ((pred_in_bin == 1) & (target_in_bin == 1)).sum()
        fp = ((pred_in_bin == 1) & (target_in_bin == 0)).sum()
        fn = ((pred_in_bin == 0) & (target_in_bin == 1)).sum()
        iou = tp / (tp + fp + fn + 1e-8)

        # Count regions in this bin
        count = 0
        for region_id in range(1, num_features + 1):
            region_pixels = (labeled == region_id).sum()
            region_area_cm2 = region_pixels * pixel_area_cm2
            if lo <= region_area_cm2 < hi:
                count += 1

        results[bin_name] = {"IoU": float(iou), "count": count}

    return results


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs (mean ± std).

    Args:
        metrics_list: List of metric dicts from individual runs.

    Returns:
        Dict mapping metric names to {mean, std}.
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return result
