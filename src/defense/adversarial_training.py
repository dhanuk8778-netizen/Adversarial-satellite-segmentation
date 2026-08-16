"""
FGSM adversarial training defense (Goodfellow et al., 2015 / Madry et al.,
2018 "for free" family of ideas, simplified to plain FGSM-in-the-loop).

Each training step generates an FGSM adversarial version of the batch
on-the-fly (using the model's *current* weights) and trains on a mix of
clean + adversarial examples. This is cheap (one extra forward+backward per
batch) compared to PGD adversarial training (num_iter extra passes), which
is the classic robustness/cost trade-off this project quantifies: it
recovers a meaningful chunk of mIoU under PGD attack, at a measured cost in
clean-data accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.attacks.fgsm import fgsm_attack
from src.metrics.segmentation_metrics import SegmentationMetrics


def adversarial_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    eps: float = 8 / 255,
    clean_ratio: float = 0.5,
    ignore_index: int = -100,
) -> dict:
    """One epoch of FGSM adversarial training.

    Args:
        clean_ratio: fraction of each batch's loss contributed by the clean
            (unperturbed) examples vs. the FGSM-perturbed examples. 0.5
            (equal weighting) is a standard, robust default; push it lower
            to prioritize robustness further at greater clean-accuracy cost.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        # Generate FGSM adversarial examples w.r.t. current weights.
        # Attack generation itself is done with the model in eval mode
        # (stable BN stats) but gradients still flow to `images`.
        model.eval()
        adv_images = fgsm_attack(model, images, masks, eps=eps, ignore_index=ignore_index)
        model.train()

        optimizer.zero_grad()

        clean_logits = model(images)
        adv_logits = model(adv_images)

        clean_loss = F.cross_entropy(clean_logits, masks, ignore_index=ignore_index)
        adv_loss = F.cross_entropy(adv_logits, masks, ignore_index=ignore_index)
        loss = clean_ratio * clean_loss + (1 - clean_ratio) * adv_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.summary()
