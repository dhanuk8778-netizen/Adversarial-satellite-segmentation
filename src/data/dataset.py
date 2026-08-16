"""
Real-data Sentinel-2 chip dataset loader.

Expects a directory layout:

    root/
      images/*.tif      # 6-band GeoTIFF chips (B2,B3,B4,B8,B11,B12), any HxW
      masks/*.png        # single-channel label PNGs, values in [0, num_classes)

Image/mask pairs are matched by filename stem. Reading uses `rasterio` if
available (recommended -- correctly handles GeoTIFF band order and nodata),
falling back to `tifffile` for plain multi-band TIFFs. See
scripts/download_sentinel2.md for how to build this directory from public
Sentinel-2 sources.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


def _read_multiband_tif(path: str) -> np.ndarray:
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (C, H, W)
        return arr
    if _HAS_TIFFFILE:
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] <= 16 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)  # HWC -> CHW
        return arr
    raise ImportError(
        "Reading real Sentinel-2 GeoTIFFs requires `rasterio` or `tifffile`. "
        "Install one of them (`pip install rasterio`) or use "
        "src.data.synthetic.SyntheticSentinel2Dataset for offline development."
    )


def _read_mask(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path)).astype(np.int64)


# Sentinel-2 L2A typical reflectance range is 0-10000 (scaled by 10000);
# these stats are placeholders for per-band standardization and should be
# recomputed on your actual training set (see scripts/compute_band_stats.py
# pattern referenced in the README).
DEFAULT_BAND_MEAN = np.array([0.13, 0.14, 0.15, 0.28, 0.22, 0.17], dtype=np.float32)
DEFAULT_BAND_STD = np.array([0.05, 0.05, 0.06, 0.08, 0.07, 0.06], dtype=np.float32)


class Sentinel2Dataset(Dataset):
    def __init__(
        self,
        root: str,
        band_mean: np.ndarray | None = None,
        band_std: np.ndarray | None = None,
        reflectance_scale: float = 10000.0,
        transform=None,
    ):
        self.root = root
        self.image_paths = sorted(glob.glob(os.path.join(root, "images", "*.tif")))
        if not self.image_paths:
            raise FileNotFoundError(
                f"No .tif images found under {os.path.join(root, 'images')}. "
                "See scripts/download_sentinel2.md."
            )
        self.mask_dir = os.path.join(root, "masks")
        self.band_mean = band_mean if band_mean is not None else DEFAULT_BAND_MEAN
        self.band_std = band_std if band_std is not None else DEFAULT_BAND_STD
        self.reflectance_scale = reflectance_scale
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, stem + ".png")

        image = _read_multiband_tif(img_path) / self.reflectance_scale
        image = (image - self.band_mean[:, None, None]) / self.band_std[:, None, None]
        mask = _read_mask(mask_path)

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)

        return image_t, mask_t
