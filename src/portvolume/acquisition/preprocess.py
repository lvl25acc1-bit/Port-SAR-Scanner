"""SAR preprocessing: calibration and speckle filtering.

Land masking is handled separately by the rasterized OSM water mask
in detection/water_mask.py, not here. Preprocessing produces a clean
dB-scale image with no pixels removed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from portvolume.sites import Site

logger = logging.getLogger(__name__)

# Backscatter floor in dB to avoid log(0)
DB_FLOOR = -30.0


def dn_to_sigma0_db(data: np.ndarray) -> np.ndarray:
    """Convert linear power values to sigma-naught in dB.

    Planetary Computer Sentinel-1 RTC data is already radiometrically
    calibrated to linear power. We convert to dB:
        sigma0_db = 10 * log10(linear)
    with a floor at DB_FLOOR to avoid log(0).
    """
    linear = data.astype(np.float64)
    linear = np.where(linear > 0, linear, 1e-10)
    db = 10.0 * np.log10(linear)
    db = np.clip(db, DB_FLOOR, None)
    return db


def lee_speckle_filter(data: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple Lee speckle filter (local mean approximation)."""
    return uniform_filter(data, size=size, mode="reflect")


def preprocess_scene(
    raw_path: Path,
    site: Site,
    output_dir: Path,
) -> Path:
    """Full preprocessing pipeline for one scene.

    1. Load GeoTIFF with rasterio
    2. Convert to sigma0 dB
    3. Apply Lee speckle filter
    4. Save processed array as GeoTIFF

    No land masking is applied here — that is done by the rasterized
    OSM water mask in the detection step.
    """
    rel = raw_path.name
    site_dir = output_dir / site.id
    site_dir.mkdir(parents=True, exist_ok=True)
    out_path = site_dir / rel

    if out_path.exists():
        logger.debug("Already preprocessed: %s", out_path)
        return out_path

    logger.info("Preprocessing %s for site %s", raw_path.name, site.id)

    with rasterio.open(raw_path) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile.copy()

    # Step 1: Convert to dB
    db = dn_to_sigma0_db(data)

    # Step 2: Speckle filter
    filtered = lee_speckle_filter(db, size=3)

    # Save as float32 GeoTIFF
    profile.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(filtered.astype(np.float32), 1)

    logger.info("Saved preprocessed: %s", out_path.name)
    return out_path
