"""
Evaluate a trained U-Net's robustness to a 10-iteration PGD-Linf attack
(eps = 8/255), reporting the mIoU collapse relative to the clean baseline.

Usage:
    python -m src.attack_eval --checkpoint checkpoints/unet_baseline_best.pt \
        --train-config configs/train_config.yaml --attack-config configs/attack_config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.data.synthetic import SyntheticSentinel2Dataset
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.models.unet import LAND_COVER_CLASSES, UNet
from src.utils import get_device, load_checkpoint, save_json, set_seed


def build_test_loader(train_cfg: dict, batch_size: int):
    d = train_cfg["data"]
    if d["source"] == "synthetic":
        full = SyntheticSentinel2Dataset(num_samples=d["num_samples"], image_size=d["image_size"], seed=d["seed"])
    else:
        from src.data.dataset import Sentinel2Dataset
        full = Sentinel2Dataset(root=d["root"])

    n = len(full)
    n_val = int(n * d["val_fraction"])
    n_test = int(n * d["test_fraction"])
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(d["seed"])
    _, _, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def run_eval(model, loader, device, num_classes, attack_fn=None, num_batches=None):
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    for i, (images, masks) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        images, masks = images.to(device), masks.to(device)
        if attack_fn is not None:
            images = attack_fn(model, images, masks)
        with torch.no_grad():
            logits = model(images)
        metrics.update(logits, masks)
    return metrics.summary(LAND_COVER_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--attack-config", type=str, default="configs/attack_config.yaml")
    parser.add_argument("--num-batches", type=int, default=None, help="limit eval to N batches (smoke test)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tag", type=str, default="clean_model", help="label for the results file")
    args = parser.parse_args()

    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.attack_config) as f:
        attack_cfg = yaml.safe_load(f)

    set_seed(train_cfg["data"]["seed"])
    device = torch.device("cpu") if args.cpu else get_device()

    model = UNet(**train_cfg["model"]).to(device)
    meta = load_checkpoint(model, args.checkpoint, map_location=device)
    print(f"Loaded checkpoint (meta: {meta.get('epoch', 'n/a')} epochs, val_mIoU={meta.get('val_mIoU', 'n/a')})")

    test_loader = build_test_loader(train_cfg, attack_cfg["eval"]["batch_size"])
    num_batches = args.num_batches or attack_cfg["eval"]["num_batches"]
    num_classes = train_cfg["model"]["num_classes"]

    print("\n=== Clean evaluation ===")
    clean_metrics = run_eval(model, test_loader, device, num_classes, attack_fn=None, num_batches=num_batches)
    print(f"Clean mIoU: {clean_metrics['mIoU']:.4f} | pixel_acc: {clean_metrics['pixel_accuracy']:.4f}")

    pgd_cfg = attack_cfg["pgd"]
    def pgd_fn(m, x, y):
        return pgd_attack(m, x, y, eps=pgd_cfg["eps"], alpha=pgd_cfg["alpha"], num_iter=pgd_cfg["num_iter"], random_start=pgd_cfg["random_start"])

    print(f"\n=== PGD attack (eps={pgd_cfg['eps']:.4f}, iters={pgd_cfg['num_iter']}) ===")
    pgd_metrics = run_eval(model, test_loader, device, num_classes, attack_fn=pgd_fn, num_batches=num_batches)
    print(f"PGD-attacked mIoU: {pgd_metrics['mIoU']:.4f} | pixel_acc: {pgd_metrics['pixel_accuracy']:.4f}")

    fgsm_cfg = attack_cfg["fgsm"]
    def fgsm_fn(m, x, y):
        return fgsm_attack(m, x, y, eps=fgsm_cfg["eps"])

    print(f"\n=== FGSM attack (eps={fgsm_cfg['eps']:.4f}) ===")
    fgsm_metrics = run_eval(model, test_loader, device, num_classes, attack_fn=fgsm_fn, num_batches=num_batches)
    print(f"FGSM-attacked mIoU: {fgsm_metrics['mIoU']:.4f} | pixel_acc: {fgsm_metrics['pixel_accuracy']:.4f}")

    drop_pgd = clean_metrics["mIoU"] - pgd_metrics["mIoU"]
    drop_fgsm = clean_metrics["mIoU"] - fgsm_metrics["mIoU"]
    print(f"\nmIoU drop under PGD:  {drop_pgd:.4f} ({100*drop_pgd/max(clean_metrics['mIoU'],1e-9):.1f}% relative)")
    print(f"mIoU drop under FGSM: {drop_fgsm:.4f} ({100*drop_fgsm/max(clean_metrics['mIoU'],1e-9):.1f}% relative)")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"attack_eval_{args.tag}.json")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "pgd_config": pgd_cfg,
            "fgsm_config": fgsm_cfg,
            "clean": clean_metrics,
            "pgd": pgd_metrics,
            "fgsm": fgsm_metrics,
        },
        out_path,
    )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
