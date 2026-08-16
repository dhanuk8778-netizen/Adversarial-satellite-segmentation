"""
Train a robustified U-Net using FGSM adversarial training, then report the
clean-accuracy trade-off and the mIoU recovered under PGD attack relative
to the non-robust baseline.

Usage:
    python -m src.defense_train --train-config configs/train_config.yaml \
        --attack-config configs/attack_config.yaml --baseline-checkpoint checkpoints/unet_baseline_best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.pgd import pgd_attack
from src.attack_eval import run_eval
from src.data.synthetic import SyntheticSentinel2Dataset
from src.defense.adversarial_training import adversarial_train_epoch, evaluate_clean
from src.models.unet import UNet
from src.utils import get_device, load_checkpoint, save_checkpoint, save_json, set_seed, timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--attack-config", type=str, default="configs/attack_config.yaml")
    parser.add_argument("--baseline-checkpoint", type=str, default=None,
                         help="clean-model checkpoint to report the accuracy trade-off against")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None, help="limit attack eval to N batches (smoke test)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.attack_config) as f:
        attack_cfg = yaml.safe_load(f)

    if args.num_samples is not None:
        train_cfg["data"]["num_samples"] = args.num_samples

    adv_cfg = attack_cfg["adversarial_training"]
    epochs = args.epochs or adv_cfg["epochs"]

    set_seed(train_cfg["data"]["seed"])
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Using device: {device}")

    d = train_cfg["data"]
    full = SyntheticSentinel2Dataset(num_samples=d["num_samples"], image_size=d["image_size"], seed=d["seed"]) \
        if d["source"] == "synthetic" else __import__("src.data.dataset", fromlist=["Sentinel2Dataset"]).Sentinel2Dataset(root=d["root"])

    n = len(full)
    n_val = int(n * d["val_fraction"])
    n_test = int(n * d["test_fraction"])
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(d["seed"])
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    bs = adv_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = UNet(**train_cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=adv_cfg["lr"], weight_decay=train_cfg["train"]["weight_decay"])

    num_classes = train_cfg["model"]["num_classes"]
    best_val_miou = -1.0
    ckpt_dir = train_cfg["output"]["checkpoint_dir"]
    run_name = "unet_fgsm_robust"
    history = []

    for epoch in range(1, epochs + 1):
        with timer(f"adv epoch {epoch}"):
            train_stats = adversarial_train_epoch(
                model, train_loader, optimizer, device, eps=adv_cfg["eps"], clean_ratio=adv_cfg["clean_ratio"]
            )
            val_metrics = evaluate_clean(model, val_loader, device, num_classes)
        print(f"epoch {epoch:03d} | train_loss {train_stats['train_loss']:.4f} | val_mIoU {val_metrics['mIoU']:.4f}")
        history.append({"epoch": epoch, **train_stats, **val_metrics})

        if val_metrics["mIoU"] > best_val_miou:
            best_val_miou = val_metrics["mIoU"]
            save_checkpoint(model, os.path.join(ckpt_dir, f"{run_name}_best.pt"),
                             meta={"epoch": epoch, "val_mIoU": best_val_miou, "config": train_cfg})

    load_checkpoint(model, os.path.join(ckpt_dir, f"{run_name}_best.pt"), map_location=device)

    num_batches = args.num_batches or attack_cfg["eval"]["num_batches"]
    print("\n=== Robust model: clean evaluation ===")
    clean_metrics = run_eval(model, test_loader, device, num_classes, attack_fn=None, num_batches=num_batches)
    print(f"Robust clean mIoU: {clean_metrics['mIoU']:.4f}")

    pgd_cfg = attack_cfg["pgd"]
    def pgd_fn(m, x, y):
        return pgd_attack(m, x, y, eps=pgd_cfg["eps"], alpha=pgd_cfg["alpha"], num_iter=pgd_cfg["num_iter"])

    print(f"\n=== Robust model: PGD attack (eps={pgd_cfg['eps']:.4f}, iters={pgd_cfg['num_iter']}) ===")
    pgd_metrics = run_eval(model, test_loader, device, num_classes, attack_fn=pgd_fn, num_batches=num_batches)
    print(f"Robust PGD-attacked mIoU: {pgd_metrics['mIoU']:.4f}")

    result = {
        "robust_clean": clean_metrics,
        "robust_under_pgd": pgd_metrics,
        "history": history,
    }

    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        baseline = UNet(**train_cfg["model"]).to(device)
        load_checkpoint(baseline, args.baseline_checkpoint, map_location=device)
        print("\n=== Baseline (non-robust) model: clean + PGD for comparison ===")
        baseline_clean = run_eval(baseline, test_loader, device, num_classes, attack_fn=None, num_batches=num_batches)
        baseline_pgd = run_eval(baseline, test_loader, device, num_classes, attack_fn=pgd_fn, num_batches=num_batches)
        result["baseline_clean"] = baseline_clean
        result["baseline_under_pgd"] = baseline_pgd

        clean_tradeoff = baseline_clean["mIoU"] - clean_metrics["mIoU"]
        recovered = pgd_metrics["mIoU"] - baseline_pgd["mIoU"]
        sign = "-" if clean_tradeoff >= 0 else "+"
        print(f"\nClean accuracy trade-off: {sign}{abs(clean_tradeoff)*100:.1f} points mIoU "
              f"({'cost' if clean_tradeoff >= 0 else 'no cost -- robust model also improved clean mIoU on this run'})")
        print(f"mIoU recovered under PGD attack: +{recovered*100:.1f} points "
              f"({baseline_pgd['mIoU']*100:.1f}% -> {pgd_metrics['mIoU']*100:.1f}%)")
        result["clean_accuracy_tradeoff_points"] = clean_tradeoff * 100
        result["miou_recovered_points"] = recovered * 100

    os.makedirs("results", exist_ok=True)
    save_json(result, os.path.join("results", "defense_eval.json"))
    print("\nSaved results to results/defense_eval.json")


if __name__ == "__main__":
    main()
