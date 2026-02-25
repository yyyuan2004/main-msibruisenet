"""
metrics.py — Evaluation metrics for semantic segmentation
==========================================================

Key I/O:
    Input : predicted mask (H, W) int, ground truth mask (H, W) int
    Output: dict of metric values

Implements:
    - IoU (Intersection over Union) per class and mean
    - Dice coefficient
    - F1, Precision, Recall
    - Area-stratified metrics (by bruise size in cm²)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_confusion(
    pred: np.ndarray, target: np.ndarray, num_classes: int = 2
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: (H, W) predicted labels.
        target: (H, W) ground truth labels.
        num_classes: Number of classes.

    Returns:
        (num_classes, num_classes) confusion matrix.
    """
    mask = (target >= 0) & (target < num_classes)
    cm = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, num_classes: int = 2
) -> Dict[str, float]:
    """Compute segmentation metrics from a single prediction/target pair.

    Args:
        pred: (H, W) predicted class labels.
        target: (H, W) ground truth class labels.
        num_classes: Number of classes.

    Returns:
        Dict with keys: mIoU, IoU_bg, IoU_bruise, Dice, F1, Precision, Recall.
    """
    cm = compute_confusion(pred, target, num_classes)

    # Per-class IoU
    iou_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou_per_class[c] = tp / denom if denom > 0 else 0.0

    # Bruise class metrics (class 1)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "mIoU": float(iou_per_class.mean()),
        "IoU_bg": float(iou_per_class[0]),
        "IoU_bruise": float(iou_per_class[1]),
        "Dice": float(dice),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
    }


def compute_metrics_batch(
    preds: List[np.ndarray], targets: List[np.ndarray], num_classes: int = 2
) -> Dict[str, float]:
    """Aggregate metrics over a batch of predictions.

    Args:
        preds: List of (H, W) predicted masks.
        targets: List of (H, W) ground truth masks.
        num_classes: Number of classes.

    Returns:
        Dict with averaged metrics.
    """
    all_metrics: Dict[str, List[float]] = {}
    for p, t in zip(preds, targets):
        m = compute_metrics(p, t, num_classes)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def compute_area_stratified_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    area_bins_cm2: List[float],
    pixel_per_cm: float = 50.0,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics stratified by bruise area size.

    Connected components in the target mask are grouped by area, and
    per-component IoU is computed within each bin.

    Args:
        pred: (H, W) predicted mask.
        target: (H, W) ground truth mask.
        area_bins_cm2: Bin edges in cm² (e.g. [0, 1.0, 3.0, 1000.0]).
        pixel_per_cm: Pixels per centimetre (calibration).

    Returns:
        Dict mapping bin labels to metric dicts.
    """
    from scipy import ndimage

    pixel_area_cm2 = 1.0 / (pixel_per_cm ** 2)
    labelled, n_components = ndimage.label(target == 1)

    bin_results: Dict[str, List[float]] = {}
    for i in range(1, n_components + 1):
        comp_mask = labelled == i
        area_cm2 = comp_mask.sum() * pixel_area_cm2

        # Find which bin this component belongs to
        bin_label = None
        for j in range(len(area_bins_cm2) - 1):
            if area_bins_cm2[j] <= area_cm2 < area_bins_cm2[j + 1]:
                bin_label = f"{area_bins_cm2[j]}-{area_bins_cm2[j+1]}_cm2"
                break
        if bin_label is None:
            continue

        # Component-level IoU
        pred_comp = pred[comp_mask]
        tp = (pred_comp == 1).sum()
        fn = (pred_comp == 0).sum()
        # FP in the bounding box region
        rows, cols = np.where(comp_mask)
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        pred_region = pred[r_min:r_max, c_min:c_max]
        target_region = target[r_min:r_max, c_min:c_max]
        fp = ((pred_region == 1) & (target_region == 0)).sum()

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        bin_results.setdefault(bin_label, []).append(float(iou))

    return {
        k: {"IoU_mean": float(np.mean(v)), "IoU_std": float(np.std(v)), "count": len(v)}
        for k, v in bin_results.items()
    }
