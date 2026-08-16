"""
Synthetic Sentinel-2-style multispectral dataset.

Real Sentinel-2 tiles require pulling scenes from Copernicus / AWS
Open Data (network access not available in every environment, and the
license terms mean raw scenes shouldn't be vendored in a git repo). This
module generates procedurally-structured 6-band imagery with matching
6-class land-cover masks so the *entire pipeline* (training, PGD attack,
FGSM defense) is runnable end-to-end offline, for development, CI, and
unit testing.

To train on real data instead, point `Sentinel2Dataset` (dataset.py) at a
directory of GeoTIFF chips + PNG masks -- see scripts/download_sentinel2.md
for how to source them (e.g. Radiant Earth MLHub, the Sentinel-2 Cloud
Mask Catalogue, or a DynamicWorld / EuroSAT-derived chip export).

Class layout matches src.models.unet.LAND_COVER_CLASSES:
    0 water   1 forest   2 urban   3 agriculture   4 barren   5 wetland
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label
from torch.utils.data import Dataset

NUM_BANDS = 6      # B2, B3, B4, B8, B11, B12 analogue
NUM_CLASSES = 6

# Approximate per-class reflectance signature across the 6 bands, used to
# synthesize spectrally-plausible pixel values (values are illustrative,
# not radiometrically calibrated).
_CLASS_SPECTRA = np.array([
    [0.03, 0.05, 0.04, 0.02, 0.01, 0.01],  # water: low reflectance, low NIR
    [0.04, 0.07, 0.03, 0.35, 0.15, 0.08],  # forest: high NIR
    [0.15, 0.16, 0.18, 0.20, 0.28, 0.25],  # urban: flat, moderately bright
    [0.08, 0.12, 0.09, 0.30, 0.20, 0.14],  # agriculture: NIR bump, lower than forest
    [0.20, 0.22, 0.24, 0.26, 0.30, 0.28],  # barren: bright, flat
    [0.05, 0.09, 0.06, 0.22, 0.12, 0.07],  # wetland: forest/water blend
], dtype=np.float32)


def _random_region_map(size: int, num_classes: int, blobiness: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a spatially-coherent class map via smoothed multi-channel
    noise + argmax (a cheap stand-in for realistic land-cover parcels)."""
    noise = rng.normal(size=(num_classes, size, size)).astype(np.float32)
    for c in range(num_classes):
        noise[c] = gaussian_filter(noise[c], sigma=blobiness)
    # class priors: water/wetland rarer than forest/agriculture/urban/barren
    priors = np.array([0.9, 1.3, 1.1, 1.2, 1.0, 0.7], dtype=np.float32)[:num_classes]
    noise += np.log(priors)[:, None, None]
    class_map = noise.argmax(axis=0).astype(np.int64)
    return class_map


class SyntheticSentinel2Dataset(Dataset):
    """Procedurally generated 6-band multispectral tiles + land-cover masks.

    Deterministic given `seed` + index, so train/val/test splits generated
    with different seeds never collide, and the dataset is reproducible
    without storing any files on disk.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        image_size: int = 128,
        seed: int = 0,
        blobiness: float = 6.0,
        noise_std: float = 0.02,
        normalize: bool = True,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.blobiness = blobiness
        self.noise_std = noise_std
        self.normalize = normalize

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1_000_003 + idx)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        size = self.image_size

        class_map = _random_region_map(size, NUM_CLASSES, self.blobiness, rng)

        image = np.zeros((NUM_BANDS, size, size), dtype=np.float32)
        for c in range(NUM_CLASSES):
            mask_c = class_map == c
            if not mask_c.any():
                continue
            spectrum = _CLASS_SPECTRA[c]
            for b in range(NUM_BANDS):
                image[b][mask_c] = spectrum[b]

        # sensor/atmospheric noise + slight per-tile illumination variation
        image += rng.normal(0, self.noise_std, size=image.shape).astype(np.float32)
        illum = 1.0 + rng.normal(0, 0.05)
        image *= illum
        image = np.clip(image, 0.0, 1.0)

        if self.normalize:
            image = (image - 0.5) / 0.25  # roughly zero-mean, unit-ish scale

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(class_map).long()
        return image_t, mask_t


def class_distribution(dataset: SyntheticSentinel2Dataset, num_batches: int = 20) -> np.ndarray:
    """Quick sanity utility: estimate class pixel frequency over a few samples."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(min(num_batches, len(dataset))):
        _, mask = dataset[i]
        c = np.bincount(mask.numpy().reshape(-1), minlength=NUM_CLASSES)
        counts += c
    return counts / counts.sum()
