"""
Fast Gradient Sign Method (FGSM) — Goodfellow et al., 2015.

Used two ways in this project:
  1. As a (weaker, single-step) attack for comparison against PGD.
  2. As the inner-loop perturbation generator for adversarial training,
     the defense evaluated in src/defense/adversarial_training.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    clip_min: float = -3.0,
    clip_max: float = 3.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Single-step untargeted FGSM-Linf attack.

    Args mirror `pgd_attack`; see src/attacks/pgd.py for the note on
    epsilon in normalized input space.
    """
    images = torch.clamp(images.clone().detach(), clip_min, clip_max).requires_grad_(True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    grad = torch.autograd.grad(loss, images)[0]

    with torch.no_grad():
        adv_images = images + eps * grad.sign()
        adv_images = torch.clamp(adv_images, clip_min, clip_max)

    return adv_images.detach()
