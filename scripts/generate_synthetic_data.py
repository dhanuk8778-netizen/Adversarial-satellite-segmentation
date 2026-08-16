"""
Materialize a few synthetic samples to disk as PNG previews (RGB-ish
composite from bands 3/2/1) + label masks, for quick visual sanity checking.
The training pipeline itself consumes SyntheticSentinel2Dataset on-the-fly
and does not require this script.

Usage:
    python scripts/generate_synthetic_data.py --n 8 --out results/synthetic_preview
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticSentinel2Dataset
from src.models.unet import LAND_COVER_CLASSES

_PALETTE = np.array([
    [30, 90, 200],    # water - blue
    [30, 130, 60],    # forest - green
    [160, 160, 160],  # urban - gray
    [220, 180, 60],   # agriculture - gold
    [180, 140, 100],  # barren - tan
    [90, 180, 160],   # wetland - teal
], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="results/synthetic_preview")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = SyntheticSentinel2Dataset(num_samples=args.n, image_size=args.image_size, seed=args.seed, normalize=False)

    for i in range(args.n):
        image, mask = ds[i]
        rgb = image[[2, 1, 0]].numpy()  # bands: 0=B2(blue) 1=B3(green) 2=B4(red)
        rgb = np.clip(rgb / max(rgb.max(), 1e-6), 0, 1)
        rgb_img = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)

        mask_np = mask.numpy()
        mask_rgb = _PALETTE[mask_np]

        Image.fromarray(rgb_img).save(os.path.join(args.out, f"sample_{i:03d}_rgb.png"))
        Image.fromarray(mask_rgb).save(os.path.join(args.out, f"sample_{i:03d}_mask.png"))

    print(f"Wrote {args.n} preview pairs to {args.out}")
    print("Classes:", LAND_COVER_CLASSES)


if __name__ == "__main__":
    main()
