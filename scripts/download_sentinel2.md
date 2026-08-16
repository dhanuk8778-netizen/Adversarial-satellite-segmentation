# Sourcing real Sentinel-2 data

This repo ships with a **synthetic** data generator (`src/data/synthetic.py`)
so the full pipeline runs offline with zero setup. To reproduce the reported
results on real imagery, point `src/data/dataset.py`'s `Sentinel2Dataset` at
a directory of 6-band chips + label masks. A few practical ways to build one:

## Option A — EuroSAT / DynamicWorld-style chip exports (easiest)
Public benchmark datasets already distribute pre-cut, labeled Sentinel-2
chips:
- **Radiant Earth MLHub** (https://mlhub.earth/) hosts several land-cover
  datasets with Sentinel-2 imagery and per-pixel or per-chip labels.
- **Google's Dynamic World** (via Earth Engine) provides near-real-time
  9-class land-cover probability maps aligned to Sentinel-2 scenes; export
  chips + argmax labels via the Earth Engine Python API.

## Option B — Copernicus Open Access Hub / AWS Open Data
Pull raw L2A (surface reflectance) scenes directly:
- Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu/)
- Sentinel-2 on AWS Open Data (`s3://sentinel-s2-l2a`, requester-pays)

Then tile scenes into fixed-size chips (e.g. 128x128) using the six bands
referenced throughout this repo (B2, B3, B4, B8, B11, B12 — RGB + NIR + two
SWIR bands, resampled to a common 10-20m resolution), and generate label
masks from a reference land-cover product (e.g. ESA WorldCover, resampled
and re-mapped to this project's 6 classes: water, forest, urban,
agriculture, barren, wetland).

## Expected directory layout

```
data/sentinel2/
  images/
    tile_0001.tif   # 6-band GeoTIFF, float32 or uint16 reflectance
    tile_0002.tif
    ...
  masks/
    tile_0001.png    # single-channel, values 0-5
    tile_0002.png
    ...
```

## Wiring it up
1. Install the optional geo dependency: `pip install rasterio`
2. Set `data.source: real` and `data.root: data/sentinel2` in
   `configs/train_config.yaml`
3. Recompute `DEFAULT_BAND_MEAN` / `DEFAULT_BAND_STD` in
   `src/data/dataset.py` from your actual training split rather than using
   the placeholder values shipped in this repo (a simple running-mean/std
   pass over `images/*.tif` is sufficient).
4. Run `python -m src.train --config configs/train_config.yaml` as usual.
