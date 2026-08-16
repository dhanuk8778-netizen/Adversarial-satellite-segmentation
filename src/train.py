"""
Train the clean (non-adversarial) U-Net baseline.

Usage:
    python -m src.train --config configs/train_config.yaml
    python -m src.train --config configs/train_config.yaml --epochs 3 --num-samples 200  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticSentinel2Dataset
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.models.unet import LAND_COVER_CLASSES, UNet
from src.utils import get_device, save_checkpoint, save_json, set_seed, timer


def build_dataloaders(cfg: dict):
    d = cfg["data"]
    if d["source"] == "synthetic":
        full = SyntheticSentinel2Dataset(
            num_samples=d["num_samples"], image_size=d["image_size"], seed=d["seed"]
        )
    else:
        from src.data.dataset import Sentinel2Dataset
        full = Sentinel2Dataset(root=d["root"])

    n = len(full)
    n_val = int(n * d["val_fraction"])
    n_test = int(n * d["test_fraction"])
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(d["seed"])
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, masks)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.summary(LAND_COVER_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    parser.add_argument("--num-samples", type=int, default=None, help="override cfg.data.num_samples")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA/MPS available")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.num_samples is not None:
        cfg["data"]["num_samples"] = args.num_samples

    set_seed(cfg["data"]["seed"])
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = UNet(**cfg["model"]).to(device)
    print(f"UNet parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["amp"] and device.type == "cuda") else None

    best_miou = -1.0
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    run_name = cfg["output"]["run_name"]
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        with timer(f"epoch {epoch}"):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
            val_metrics = evaluate(model, val_loader, device, cfg["model"]["num_classes"])
        scheduler.step()

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
            f"val_mIoU {val_metrics['mIoU']:.4f} | val_pixel_acc {val_metrics['pixel_accuracy']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            bad_epochs = 0
            save_checkpoint(
                model, os.path.join(ckpt_dir, f"{run_name}_best.pt"),
                meta={"epoch": epoch, "val_mIoU": best_miou, "config": cfg},
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    save_checkpoint(model, os.path.join(ckpt_dir, f"{run_name}_final.pt"), meta={"config": cfg})

    # Final test-set evaluation using best checkpoint
    from src.utils import load_checkpoint
    load_checkpoint(model, os.path.join(ckpt_dir, f"{run_name}_best.pt"), map_location=device)
    test_metrics = evaluate(model, test_loader, device, cfg["model"]["num_classes"])
    print(f"TEST  | mIoU {test_metrics['mIoU']:.4f} | pixel_acc {test_metrics['pixel_accuracy']:.4f}")

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    save_json(
        {"history": history, "test_metrics": test_metrics, "best_val_mIoU": best_miou},
        os.path.join(cfg["output"]["results_dir"], f"{run_name}_train_results.json"),
    )


if __name__ == "__main__":
    main()
