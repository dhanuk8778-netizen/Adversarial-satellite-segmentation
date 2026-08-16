"""
Standalone clean-accuracy evaluation for a checkpoint (no attack). Useful
for quickly checking a checkpoint's test mIoU without running the full
attack suite.

Usage:
    python -m src.evaluate --checkpoint checkpoints/unet_baseline_best.pt --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attack_eval import build_test_loader, run_eval
from src.models.unet import UNet
from src.utils import get_device, load_checkpoint, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu") if args.cpu else get_device()
    model = UNet(**cfg["model"]).to(device)
    meta = load_checkpoint(model, args.checkpoint, map_location=device)
    print(f"Loaded checkpoint: {meta}")

    test_loader = build_test_loader(cfg, batch_size=cfg["train"]["batch_size"])
    metrics = run_eval(model, test_loader, device, cfg["model"]["num_classes"])
    print(f"Test mIoU: {metrics['mIoU']:.4f} | pixel_acc: {metrics['pixel_accuracy']:.4f}")
    for cls, iou in metrics["per_class_iou"].items():
        print(f"  {cls:12s}: {iou}")

    save_json(metrics, "results/eval_only.json")


if __name__ == "__main__":
    main()
