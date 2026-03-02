"""Segmentation evaluation metrics: mIoU, F1, Precision, Recall."""

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates confusion matrix and computes segmentation metrics.

    Metrics:
        - IoU per class and mIoU
        - F1-score per class and macro F1
        - Precision per class and macro
        - Recall per class and macro
        - Confusion matrix

    Args:
        num_classes: Number of segmentation classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds, targets):
        """Update confusion matrix with a batch of predictions.

        Args:
            preds: (B, H, W) integer predictions.
            targets: (B, H, W) integer ground truth.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        preds = preds.astype(np.int64).flatten()
        targets = targets.astype(np.int64).flatten()

        # Only count valid pixels
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        # Update confusion matrix
        indices = targets * self.num_classes + preds
        cm_flat = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm_flat.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary with all metrics.
        """
        cm = self.confusion_matrix
        eps = 1e-8

        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # IoU
        iou = tp / (tp + fp + fn + eps)
        miou = iou.mean()

        # Precision
        precision = tp / (tp + fp + eps)
        precision_macro = precision.mean()

        # Recall
        recall = tp / (tp + fn + eps)
        recall_macro = recall.mean()

        # F1
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_macro = f1.mean()

        return {
            "mIoU": float(miou),
            "IoU_per_class": iou,
            "F1_macro": float(f1_macro),
            "F1_per_class": f1,
            "Precision_macro": float(precision_macro),
            "Precision_per_class": precision,
            "Recall_macro": float(recall_macro),
            "Recall_per_class": recall,
            "confusion_matrix": cm,
        }
