"""Post-processing: cluster detections, produce GeoDataFrame output.

Post-CFAR filters applied here (after CFAR, before final output):
1. **dB floor** — reject clusters whose peak backscatter is below ``min_db``.
   Applied *after* CFAR so it does not distort the background estimate.
2. **Aspect ratio** — reject clusters more elongated than ``max_aspect_ratio``
   (typical wave streaks are long and thin).
3. **dB contrast** — reject clusters where ``peak_db - mean_db`` is below
   a threshold (uniform clutter patches rather than compact point targets).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass
from shapely.geometry import Point

from portvolume.config import PostDetectionParams

logger = logging.getLogger(__name__)


def _cluster_aspect_ratio(cluster_mask: np.ndarray) -> float:
    """Compute the aspect ratio (major/minor axis) of a binary cluster.

    Uses second-order central moments (inertia tensor).  Returns 1.0 for
    perfectly round clusters, >1 for elongated ones.
    """
    ys, xs = np.where(cluster_mask)
    if len(xs) < 3:
        return 1.0
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    # Inertia tensor components
    ixx = (dy * dy).sum()
    iyy = (dx * dx).sum()
    ixy = -(dx * dy).sum()
    # Eigenvalues of [[iyy, ixy], [ixy, ixx]]
    avg = (ixx + iyy) / 2.0
    diff = np.sqrt(max(((ixx - iyy) / 2.0) ** 2 + ixy**2, 0))
    lam1 = avg + diff
    lam2 = max(avg - diff, 1e-10)
    return np.sqrt(lam1 / lam2)


def cluster_detections(
    binary_mask: np.ndarray,
    min_pixels: int,
    transform,
    crs=None,
    db_image: np.ndarray | None = None,
    post_params: PostDetectionParams | None = None,
) -> gpd.GeoDataFrame:
    """Convert a binary detection mask to a GeoDataFrame of point detections.

    Parameters
    ----------
    binary_mask : 2D boolean array from CFAR detector.
    min_pixels : Minimum cluster size to keep.
    transform : rasterio-style affine transform.
    crs : Coordinate reference system of the raster.
    db_image : Optional dB-scale image for post-detection brightness gating.
    post_params : Optional post-detection filter parameters.

    Returns
    -------
    GeoDataFrame with columns: [geometry, pixel_count, peak_db, mean_db,
    aspect_ratio, peak_to_mean].  CRS is EPSG:4326.
    """
    labeled_array, n_labels = label(binary_mask)

    empty = gpd.GeoDataFrame(
        columns=["geometry", "pixel_count", "peak_db", "mean_db",
                 "aspect_ratio", "db_contrast"],
        geometry="geometry",
        crs="EPSG:4326",
    )
    if n_labels == 0:
        return empty

    records = []
    rejected = {"size": 0, "brightness": 0, "aspect": 0, "contrast": 0}

    for cluster_id in range(1, n_labels + 1):
        cluster_mask = labeled_array == cluster_id
        pixel_count = int(cluster_mask.sum())

        if pixel_count < min_pixels:
            rejected["size"] += 1
            continue

        # Compute cluster metrics from dB image
        peak_db = mean_db = aspect = contrast = 0.0
        if db_image is not None:
            cluster_vals = db_image[cluster_mask]
            valid = cluster_vals[~np.isnan(cluster_vals)]
            if len(valid) > 0:
                peak_db = float(valid.max())
                mean_db = float(valid.mean())
                contrast = peak_db - mean_db  # dB difference
            aspect = _cluster_aspect_ratio(cluster_mask)

        # Apply post-detection filters
        if post_params and db_image is not None:
            if peak_db < post_params.min_db:
                rejected["brightness"] += 1
                continue
            if aspect > post_params.max_aspect_ratio:
                rejected["aspect"] += 1
                continue
            if contrast < post_params.min_db_contrast and pixel_count > 5:
                rejected["contrast"] += 1
                continue

        # Centroid
        centroid = center_of_mass(cluster_mask)
        row, col = centroid[0], centroid[1]
        lon, lat = transform * (col, row)

        records.append({
            "geometry": Point(lon, lat),
            "pixel_count": pixel_count,
            "peak_db": round(peak_db, 2),
            "mean_db": round(mean_db, 2),
            "aspect_ratio": round(aspect, 2),
            "db_contrast": round(contrast, 2),
        })

    native_crs = crs if crs else "EPSG:4326"
    if not records:
        return empty
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=native_crs)

    if crs and str(crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    logger.info(
        "Clustered %d detections from %d labels (min_px=%d). "
        "Rejected: size=%d, brightness=%d, aspect=%d, contrast=%d",
        len(gdf), n_labels, min_pixels,
        rejected["size"], rejected["brightness"],
        rejected["aspect"], rejected["contrast"],
    )
    return gdf


def save_detections(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    fmt: str = "parquet",
) -> None:
    """Save detection GeoDataFrame to disk.

    Parameters
    ----------
    gdf : GeoDataFrame of detections.
    output_path : Where to save. Extension is overridden by fmt.
    fmt : "parquet" or "geojson".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        out = output_path.with_suffix(".parquet")
        gdf.to_parquet(out)
    elif fmt == "geojson":
        out = output_path.with_suffix(".geojson")
        gdf.to_file(out, driver="GeoJSON")
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    logger.info("Saved %d detections to %s", len(gdf), out)


def scene_detection_summary(
    gdf: gpd.GeoDataFrame,
    site_id: str,
    scene_id: str,
    timestamp: str,
) -> dict:
    """Produce a summary dict for one scene's detections."""
    return {
        "site_id": site_id,
        "scene_id": scene_id,
        "timestamp": timestamp,
        "vessel_count": len(gdf),
        "total_pixel_area": int(gdf["pixel_count"].sum()) if len(gdf) > 0 else 0,
        "mean_cluster_size": (
            float(gdf["pixel_count"].mean()) if len(gdf) > 0 else 0.0
        ),
    }
