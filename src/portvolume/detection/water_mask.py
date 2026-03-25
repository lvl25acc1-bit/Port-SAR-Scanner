"""Rasterized water mask for pre-CFAR land exclusion.

Mask source precedence:
1. **Manual mask** — if ``config/water_masks/{site_id}.gpkg`` (or .geojson)
   exists it is treated as authoritative.  No automated layers are added,
   no buffer is applied, and ``all_touched=False`` so the rasterisation
   matches the hand-drawn boundary exactly.
2. Automated layered approach (per-region):
   a. Europe: EU-Hydro (official Copernicus hydrography) as primary.
   b. Everywhere: JRC Global Surface Water occurrence as secondary/fallback.
   c. OSM harbour-only tags as supplement for man-made port basins.
   d. Union all layers, buffer in projected CRS, rasterize to scene grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box as shapely_box, shape as shapely_shape

from portvolume.sites import Site

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_M = 30
JRC_OCCURRENCE_THRESHOLD = 25

# Supported extensions for manual water-mask files (checked in order).
_MANUAL_MASK_EXTENSIONS = [".gpkg", ".geojson", ".json"]

_vector_cache: dict[str, gpd.GeoDataFrame] = {}

# Sites in regions covered by EU-Hydro
EU_HYDRO_REGIONS = {"europe"}


class WaterMaskError(RuntimeError):
    """Raised when a valid water mask cannot be constructed."""


# ---------------------------------------------------------------------------
# Layer 1: EU-Hydro (Europe only — official Copernicus hydrography)
# ---------------------------------------------------------------------------

_EUHYDRO_BASE = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services"
    "/EUHydro/EUHydro_RiverNetworkDatabase/MapServer"
)
_EUHYDRO_LAYERS = [
    (0, "Coastal_polygon"),
    (2, "InlandWater"),
    (19, "River_Net_polygon"),
]


def _fetch_euhydro(site: Site) -> gpd.GeoDataFrame:
    """Fetch EU-Hydro coastal, inland water, and river polygons via ArcGIS REST."""
    bbox_str = f"{site.bbox[0]},{site.bbox[1]},{site.bbox[2]},{site.bbox[3]}"
    gdfs = []

    for layer_id, layer_name in _EUHYDRO_LAYERS:
        url = (
            f"{_EUHYDRO_BASE}/{layer_id}/query"
            f"?geometry={bbox_str}"
            "&geometryType=esriGeometryEnvelope"
            "&inSR=4326&outSR=4326"
            "&spatialRel=esriSpatialRelIntersects"
            "&outFields=OBJECTID"
            "&f=geojson"
            "&resultRecordCount=5000"
        )
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            if features:
                gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
                if len(gdf) > 0:
                    gdfs.append(gdf[["geometry"]])
                    logger.info(
                        "EU-Hydro %s for %s: %d polygons",
                        layer_name, site.id, len(gdf),
                    )
        except Exception as exc:
            logger.warning(
                "EU-Hydro %s query failed for %s: %s", layer_name, site.id, exc
            )

    if not gdfs:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    combined = gpd.GeoDataFrame(
        gpd.pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
    )
    logger.info("EU-Hydro total for %s: %d polygons", site.id, len(combined))
    return combined


# ---------------------------------------------------------------------------
# Layer 2: JRC Global Surface Water (global fallback)
# ---------------------------------------------------------------------------

def _fetch_jrc_water(
    site: Site,
    scene_crs,
    scene_transform,
    scene_width: int,
    scene_height: int,
    threshold: int = JRC_OCCURRENCE_THRESHOLD,
) -> np.ndarray:
    """Fetch JRC GSW occurrence raster and reproject to scene grid."""
    import planetary_computer
    from pystac_client import Client

    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(client.search(collections=["jrc-gsw"], bbox=list(site.bbox)).items())
    if not items:
        logger.warning("No JRC GSW tiles for %s", site.id)
        return np.zeros((scene_height, scene_width), dtype=bool)

    href = items[0].assets["occurrence"].href
    with rasterio.open(href) as src:
        window = from_bounds(*site.bbox, src.transform)
        jrc_data = src.read(1, window=window)
        jrc_transform = src.window_transform(window)
        jrc_crs = src.crs

    jrc_water = (jrc_data >= threshold).astype(np.uint8)
    scene_mask = np.zeros((scene_height, scene_width), dtype=np.uint8)
    reproject(
        source=jrc_water, destination=scene_mask,
        src_transform=jrc_transform, src_crs=jrc_crs,
        dst_transform=scene_transform, dst_crs=scene_crs,
        resampling=Resampling.nearest,
    )

    result = scene_mask.astype(bool)
    logger.info("JRC water for %s: %.1f%% (threshold=%d%%)",
                site.id, 100 * result.sum() / result.size, threshold)
    return result


# ---------------------------------------------------------------------------
# Layer 3: OSM harbour supplement (port basins only)
# ---------------------------------------------------------------------------

def _fetch_osm_harbour(site: Site, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    """Fetch OSM harbour/basin/port polygons only (not generic water)."""
    cache_key = f"osm_harbour_{site.id}"
    if cache_key in _vector_cache:
        return _vector_cache[cache_key]

    if cache_dir:
        cache_path = cache_dir / f"osm_harbour_{site.id}.parquet"
        if cache_path.exists():
            gdf = gpd.read_parquet(cache_path)
            _vector_cache[cache_key] = gdf
            return gdf

    import osmnx as ox
    bbox_geom = shapely_box(*site.bbox)
    gdfs = []

    for tags in [{"landuse": "harbour"}, {"landuse": "basin"}, {"industrial": "port"}]:
        try:
            feats = ox.features_from_polygon(bbox_geom, tags=tags)
            if len(feats) > 0:
                gdfs.append(feats[["geometry"]])
        except Exception:
            pass

    if not gdfs:
        result = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    else:
        result = gpd.GeoDataFrame(gpd.pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
        result = result[result.geometry.type.isin(["Polygon", "MultiPolygon"])]

    _vector_cache[cache_key] = result
    if cache_dir and len(result) > 0:
        cache_path = cache_dir / f"osm_harbour_{site.id}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)

    logger.info("OSM harbour for %s: %d polygons", site.id, len(result))
    return result


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------

def _rasterize_vectors(
    gdf: gpd.GeoDataFrame,
    scene_crs,
    scene_transform,
    scene_width: int,
    scene_height: int,
    buffer_m: float,
    label: str,
    site_id: str,
) -> np.ndarray:
    """Buffer vector polygons in metric CRS and rasterize to scene grid."""
    if gdf.empty:
        return np.zeros((scene_height, scene_width), dtype=bool)

    # Project to metric CRS for buffering
    if gdf.crs is None or gdf.crs.is_geographic:
        metric_crs = gdf.estimate_utm_crs()
        gdf_m = gdf.to_crs(metric_crs)
    else:
        gdf_m = gdf

    buffered = gdf_m.geometry.buffer(buffer_m)
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=gdf_m.crs)
    buffered_scene = buffered_gdf.to_crs(scene_crs)

    shapes = [
        (geom, 1) for geom in buffered_scene.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros((scene_height, scene_width), dtype=bool)

    mask = rasterize(
        shapes,
        out_shape=(scene_height, scene_width),
        transform=scene_transform,
        fill=0, dtype=np.uint8, all_touched=True,
    )
    result = mask.astype(bool)
    logger.info("%s mask for %s: %.1f%% water, buffer=%dm",
                label, site_id, 100 * result.sum() / result.size, buffer_m)
    return result


# ---------------------------------------------------------------------------
# Manual mask loader
# ---------------------------------------------------------------------------

def _find_manual_mask(site: Site, mask_dir: Path) -> Path | None:
    """Look for a hand-drawn water mask file for *site*.

    Searches **only** ``mask_dir`` (typically ``config/water_masks/``) for:
      1. ``{site_id}.gpkg``  /  ``{site_id}.geojson``  (canonical)
      2. ``{site_id}_manual_water_mask.gpkg`` etc.  (backward compat)

    No other directories are searched. Returns the first match or ``None``.
    """
    if not mask_dir.is_dir():
        return None

    # Canonical names first
    for ext in _MANUAL_MASK_EXTENSIONS:
        candidate = mask_dir / f"{site.id}{ext}"
        if candidate.exists():
            return candidate

    # Backward-compatible name (same directory only)
    for ext in _MANUAL_MASK_EXTENSIONS:
        candidate = mask_dir / f"{site.id}_manual_water_mask{ext}"
        if candidate.exists():
            return candidate

    return None


def _rasterize_manual_mask(
    mask_path: Path,
    scene_crs,
    scene_transform,
    scene_width: int,
    scene_height: int,
    site_id: str,
) -> np.ndarray:
    """Rasterize a manual vector mask with NO buffer and all_touched=False."""
    gdf = gpd.read_file(mask_path)
    logger.info(
        "Manual mask for %s: %d features from %s (CRS=%s)",
        site_id, len(gdf), mask_path.name, gdf.crs,
    )

    if gdf.empty:
        raise WaterMaskError(
            f"Manual mask {mask_path} for {site_id} contains no features."
        )

    gdf_proj = gdf.to_crs(scene_crs)
    shapes = [
        (geom, 1) for geom in gdf_proj.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        raise WaterMaskError(
            f"Manual mask {mask_path} for {site_id}: no valid geometries "
            f"after reprojection to {scene_crs}."
        )

    mask = rasterize(
        shapes,
        out_shape=(scene_height, scene_width),
        transform=scene_transform,
        fill=0, dtype=np.uint8,
        all_touched=False,  # exact boundary, no inflation
    )
    result = mask.astype(bool)
    pct = 100 * result.sum() / result.size
    logger.info(
        "Manual mask for %s: %.1f%% water (%d/%d px), buffer=0, all_touched=False",
        site_id, pct, int(result.sum()), result.size,
    )
    return result


# ---------------------------------------------------------------------------
# Combined mask builder
# ---------------------------------------------------------------------------

def build_water_raster(
    scene_path: Path,
    site: Site,
    buffer_m: float = DEFAULT_BUFFER_M,
    cache_dir: Path | None = None,
    jrc_threshold: int = JRC_OCCURRENCE_THRESHOLD,
    manual_mask_dir: Path | None = None,
) -> np.ndarray:
    """Build a water mask rasterized to the scene grid.

    **Precedence**: if a manual mask file exists for this site in
    ``manual_mask_dir`` (default: ``config/water_masks/``) it is used
    exclusively — no automated layers are added, no buffer is applied, and
    ``all_touched=False``.

    Otherwise the automated layered approach is used:
      - Europe: EU-Hydro (primary) + JRC (fill) + OSM harbour.
      - Elsewhere: JRC (primary) + OSM harbour.

    Returns 2D boolean (True = water). Raises WaterMaskError if no
    layer produces any water pixels.
    """
    with rasterio.open(scene_path) as src:
        scene_crs = src.crs
        scene_transform = src.transform
        scene_width = src.width
        scene_height = src.height

    # --- Manual mask: authoritative, replaces everything ----------------
    if manual_mask_dir is None:
        from portvolume.config import PROJECT_ROOT
        manual_mask_dir = PROJECT_ROOT / "config" / "water_masks"

    manual_path = _find_manual_mask(site, manual_mask_dir)
    if manual_path is not None:
        logger.info(
            "Using MANUAL mask for %s — skipping EU-Hydro/JRC/OSM.",
            site.id,
        )
        return _rasterize_manual_mask(
            manual_path, scene_crs, scene_transform,
            scene_width, scene_height, site.id,
        )

    # --- Automated layers (fallback) ------------------------------------
    combined = np.zeros((scene_height, scene_width), dtype=bool)

    # Layer 1: EU-Hydro for European sites
    if site.region in EU_HYDRO_REGIONS:
        try:
            euhydro_gdf = _fetch_euhydro(site)
            if not euhydro_gdf.empty:
                euhydro_mask = _rasterize_vectors(
                    euhydro_gdf, scene_crs, scene_transform,
                    scene_width, scene_height, buffer_m,
                    "EU-Hydro", site.id,
                )
                combined |= euhydro_mask
        except Exception as exc:
            logger.warning("EU-Hydro failed for %s: %s", site.id, exc)

    # Layer 2: JRC Global Surface Water
    try:
        jrc_mask = _fetch_jrc_water(
            site, scene_crs, scene_transform, scene_width, scene_height,
            jrc_threshold,
        )
        combined |= jrc_mask
    except Exception as exc:
        logger.warning("JRC failed for %s: %s", site.id, exc)

    # Layer 3: OSM harbour supplement
    try:
        osm_gdf = _fetch_osm_harbour(site, cache_dir)
        if not osm_gdf.empty:
            osm_mask = _rasterize_vectors(
                osm_gdf, scene_crs, scene_transform,
                scene_width, scene_height, buffer_m,
                "OSM harbour", site.id,
            )
            combined |= osm_mask
    except Exception as exc:
        logger.warning("OSM harbour failed for %s: %s", site.id, exc)

    total_pct = 100 * combined.sum() / combined.size

    if combined.sum() == 0:
        raise WaterMaskError(
            f"Water mask for {site.id} is entirely land — all layers failed."
        )

    logger.info("Combined water mask for %s: %.1f%% water", site.id, total_pct)
    return combined
