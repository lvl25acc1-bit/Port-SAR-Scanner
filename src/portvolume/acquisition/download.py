"""Download Sentinel-1 GRD assets via windowed COG reads."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import box as shapely_box, shape as shapely_shape

logger = logging.getLogger(__name__)

# Minimum overlap fraction between scene footprint and site bbox
MIN_OVERLAP_FRACTION = 0.05
# Minimum pixels on each axis for a valid crop
MIN_PIXELS = 10


def _check_overlap(item: dict, bbox: tuple) -> float:
    """Return the fraction of the site bbox covered by the scene footprint."""
    site_box = shapely_box(*bbox)
    try:
        scene_geom = shapely_shape(item["geometry"])
        intersection = site_box.intersection(scene_geom)
        return intersection.area / site_box.area
    except Exception:
        return 0.0


def download_scene_asset(
    item: dict,
    asset_key: str,
    output_dir: Path,
    site_id: str,
    bbox: tuple[float, float, float, float],
) -> Path | None:
    """Download a single asset (VV or VH band) clipped to the site bbox.

    Uses rasterio windowed reads to fetch only the pixels within the site
    bbox directly from the remote COG, avoiding download of the full tile.

    Returns path to saved file, or None if the asset key is missing or
    the scene doesn't sufficiently overlap the bbox.
    """
    if asset_key not in item["assets"]:
        logger.warning(
            "Asset %r not found in scene %s. Available: %s",
            asset_key,
            item["id"],
            list(item["assets"].keys()),
        )
        return None

    site_dir = output_dir / site_id
    site_dir.mkdir(parents=True, exist_ok=True)

    out_path = site_dir / f"{item['id']}_{asset_key}.tif"
    if out_path.exists():
        logger.debug("Already downloaded: %s", out_path)
        return out_path

    # Check overlap before attempting download
    overlap = _check_overlap(item, bbox)
    if overlap < MIN_OVERLAP_FRACTION:
        logger.info(
            "Skipping %s/%s: only %.1f%% overlap with site bbox",
            item["id"],
            asset_key,
            overlap * 100,
        )
        return None

    href = item["assets"][asset_key]["href"]
    logger.info("Downloading %s/%s (%.0f%% overlap)", item["id"], asset_key, overlap * 100)

    try:
        west, south, east, north = bbox

        with rasterio.open(href) as src:
            # Reproject bbox from EPSG:4326 to the raster's CRS if needed
            raster_crs = src.crs
            if raster_crs and not raster_crs.to_epsg() == 4326:
                transformer = Transformer.from_crs(
                    "EPSG:4326", raster_crs, always_xy=True
                )
                west_t, south_t = transformer.transform(west, south)
                east_t, north_t = transformer.transform(east, north)
            else:
                west_t, south_t, east_t, north_t = west, south, east, north

            # Compute the read window from reprojected bounds
            window = from_bounds(west_t, south_t, east_t, north_t, src.transform)

            # Clamp window to the actual raster extent
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )

            if window.width < MIN_PIXELS or window.height < MIN_PIXELS:
                logger.info(
                    "Skipping %s/%s: crop too small (%dx%d px)",
                    item["id"],
                    asset_key,
                    int(window.width),
                    int(window.height),
                )
                return None

            # Read the windowed data
            data = src.read(1, window=window)

            # Compute the transform for the cropped window
            win_transform = src.window_transform(window)

            # Write to local GeoTIFF
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                transform=win_transform,
                driver="GTiff",
                compress="deflate",
            )

            with rasterio.open(str(out_path), "w", **profile) as dst:
                dst.write(data, 1)

        size_mb = out_path.stat().st_size / 1e6
        logger.info(
            "Saved %s (%dx%d px, %.1f MB)",
            out_path.name,
            data.shape[1],
            data.shape[0],
            size_mb,
        )
        return out_path

    except Exception:
        logger.exception("Failed to download %s/%s", item["id"], asset_key)
        if out_path.exists():
            out_path.unlink()
        return None


def download_site_scenes(
    site_id: str,
    items: list[dict],
    asset_keys: list[str],
    bbox: tuple[float, float, float, float],
    output_dir: Path,
) -> list[Path]:
    """Download all scenes for a site.

    Returns list of successfully downloaded file paths.
    """
    downloaded: list[Path] = []

    for item in items:
        for key in asset_keys:
            path = download_scene_asset(
                item=item,
                asset_key=key,
                output_dir=output_dir,
                site_id=site_id,
                bbox=bbox,
            )
            if path is not None:
                downloaded.append(path)

    logger.info(
        "Downloaded %d files for site %s (%d scenes × %d assets)",
        len(downloaded),
        site_id,
        len(items),
        len(asset_keys),
    )
    return downloaded
