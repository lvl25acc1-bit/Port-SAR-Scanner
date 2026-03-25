"""Airport activity estimation from SAR backscatter statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label

from portvolume.config import AirportParams
from portvolume.sites import Site

logger = logging.getLogger(__name__)


@dataclass
class AirportObservation:
    """Single-scene airport activity measurement."""

    site_id: str
    scene_id: str
    timestamp: str
    mean_backscatter_db: float
    bright_pixel_count: int
    bright_pixel_fraction: float
    estimated_aircraft_count: int


def estimate_airport_activity(
    processed_path: Path,
    site: Site,
    scene_id: str,
    timestamp: str,
    params: AirportParams,
) -> AirportObservation:
    """Estimate airport activity from SAR backscatter statistics.

    Aircraft on aprons produce strong radar returns against smooth concrete.
    We count bright pixels and clusters as a temporal activity proxy.
    """
    with rasterio.open(processed_path) as src:
        values = src.read(1).astype(np.float64)

    valid_mask = ~np.isnan(values)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return AirportObservation(
            site_id=site.id,
            scene_id=scene_id,
            timestamp=timestamp,
            mean_backscatter_db=np.nan,
            bright_pixel_count=0,
            bright_pixel_fraction=0.0,
            estimated_aircraft_count=0,
        )

    bright_mask = valid_mask & (values > params.threshold_db)
    bright_count = int(bright_mask.sum())
    bright_fraction = bright_count / n_valid

    labeled, n_clusters = label(bright_mask)
    aircraft_count = 0
    for cluster_id in range(1, n_clusters + 1):
        cluster_size = int((labeled == cluster_id).sum())
        if cluster_size >= params.min_cluster_pixels:
            aircraft_count += 1

    mean_db = float(np.nanmean(values))

    logger.info(
        "Airport %s/%s: bright_frac=%.4f, est_aircraft=%d",
        site.id,
        scene_id,
        bright_fraction,
        aircraft_count,
    )

    return AirportObservation(
        site_id=site.id,
        scene_id=scene_id,
        timestamp=timestamp,
        mean_backscatter_db=mean_db,
        bright_pixel_count=bright_count,
        bright_pixel_fraction=bright_fraction,
        estimated_aircraft_count=aircraft_count,
    )
