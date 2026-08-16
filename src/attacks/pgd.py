"""
Projected Gradient Descent (PGD) attack, adapted for dense (per-pixel)
semantic segmentation predictions.

Madry et al., 2018, "Towards Deep Learning Models Resistant to Adversarial
Attacks." The per-pixel cross-entropy is averaged over all spatial
locations to form a single scalar loss to backprop through, and the
perturbation is projected onto an L-infinity ball of radius `eps` after
every step, with a final clip back into valid (normalized) input range.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float | None = None,
    num_iter: int = 10,
    random_start: bool = True,
    clip_min: float = -3.0,
    clip_max: float = 3.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Runs untargeted PGD-Linf against `model` to maximize per-pixel
    cross-entropy w.r.t. the ground-truth labels.

    Args:
        images: (N, C, H, W) clean inputs, already normalized to the same
            space the model was trained on.
        labels: (N, H, W) ground-truth class ids.
        eps: L-infinity perturbation budget, in the *normalized* input
            space. NOTE: the canonical eps=8/255 is defined for images in
            [0, 1]; when inputs are z-score normalized, the effective
            perturbation magnitude is eps/band_std, so keep `eps` and
            `clip_min`/`clip_max` consistent with whatever preprocessing
            the model was trained with (see README "A note on epsilon").
        alpha: per-step size. Defaults to eps / 4, a standard choice that
            lets a 10-step attack comfortably reach the eps-ball boundary.
        num_iter: number of PGD steps (10 in the reported experiment).
        random_start: start from a random point inside the eps-ball
            (standard PGD) rather than at the clean image.
        clip_min/clip_max: valid range of the (normalized) input space,
            used for the final box-projection.
        ignore_index: label value to exclude from the loss (e.g. padding).

    Returns:
        Adversarial images, same shape/dtype as `images`, detached.
    """
    if alpha is None:
        alpha = eps / 4

    model.eval()
    # Clamp the clean reference image into the valid range first: the
    # eps-ball budget is only well-defined relative to an in-range clean
    # pixel, otherwise the final box-projection below could silently
    # inflate the effective perturbation for out-of-range inputs.
    images = torch.clamp(images.clone().detach(), clip_min, clip_max)
    labels = labels.clone().detach()

    if random_start:
        delta = torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(images + delta, clip_min, clip_max).detach()
    else:
        adv_images = images.clone().detach()

    for _ in range(num_iter):
        adv_images.requires_grad_(True)
        logits = model(adv_images)
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            adv_images = adv_images.detach() + alpha * grad.sign()
            # project back onto the eps-ball around the original image
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, clip_min, clip_max)

    return adv_images.detach()


class PGDAttackConfig:
    """Serializable config mirroring the CLI/YAML attack settings."""

    def __init__(self, eps: float = 8 / 255, alpha: float | None = None, num_iter: int = 10, random_start: bool = True):
        self.eps = eps
        self.alpha = alpha if alpha is not None else eps / 4
        self.num_iter = num_iter
        self.random_start = random_start

    def as_dict(self) -> dict:
        return {"eps": self.eps, "alpha": self.alpha, "num_iter": self.num_iter, "random_start": self.random_start}
