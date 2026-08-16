"""
Confusion-matrix based segmentation metrics: per-class IoU, mean IoU (mIoU),
and pixel accuracy. Accumulates across batches so metrics are exact over an
entire epoch/dataset rather than a batch-averaged approximation.
"""
from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion.fill(0)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        preds, targets: (N, H, W) integer class-id tensors (preds already
        argmax'd), or preds may be (N, C, H, W) logits (will be argmax'd here).
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        preds = preds.detach().cpu().numpy().reshape(-1)
        targets = targets.detach().cpu().numpy().reshape(-1)

        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        idx = targets * self.num_classes + preds
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        cm = self.confusion.astype(np.float64)
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, intersection / union, np.nan)
        return iou

    def mean_iou(self) -> float:
        iou = self.per_class_iou()
        valid = ~np.isnan(iou)
        if not valid.any():
            return 0.0
        return float(np.nanmean(iou[valid]))

    def pixel_accuracy(self) -> float:
        cm = self.confusion.astype(np.float64)
        total = cm.sum()
        if total == 0:
            return 0.0
        return float(np.diag(cm).sum() / total)

    def summary(self, class_names: list[str] | None = None) -> dict:
        iou = self.per_class_iou()
        names = class_names or [f"class_{i}" for i in range(self.num_classes)]
        return {
            "mIoU": self.mean_iou(),
            "pixel_accuracy": self.pixel_accuracy(),
            "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, iou)},
        }


def fast_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Convenience one-shot mIoU for a single batch (used inside attack loops
    where accumulating a full-epoch confusion matrix isn't necessary)."""
    m = SegmentationMetrics(num_classes)
    m.update(preds, targets)
    return m.mean_iou()
