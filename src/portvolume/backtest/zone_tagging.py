"""Zone tagging for Rotterdam water mask polygons and detection assignment.

The 27 mask polygons have placeholder names. Zone type is assigned by
spatial position (centroid longitude), verified against optical imagery.
Also produces a dissolved AOI polygon for GFW queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Zone classification by polygon index, verified against centroid positions.
# approach: sea approach west of Maasvlakte (lon < 4.05)
# channel: Europoort / Nieuwe Waterweg (lon 4.05-4.15)
# basin: inner port basins (lon > 4.15)
ZONE_MAP: dict[int, str] = {
    0: "channel",    # lon=4.137
    1: "channel",    # lon=4.193 — borderline, but this is Nieuwe Waterweg
    2: "channel",    # lon=4.119
    3: "channel",    # lon=4.150
    4: "channel",    # lon=4.097
    5: "basin",      # lon=4.224
    6: "basin",      # lon=4.188
    7: "basin",      # lon=4.236
    8: "basin",      # lon=4.282
    9: "basin",      # lon=4.306
    10: "basin",     # lon=4.251
    11: "basin",     # lon=4.328
    12: "basin",     # lon=4.325
    13: "basin",     # lon=4.352
    14: "basin",     # lon=4.378
    15: "basin",     # lon=4.410
    16: "basin",     # lon=4.433
    17: "basin",     # lon=4.417
    18: "basin",     # lon=4.393
    19: "basin",     # lon=4.463
    20: "basin",     # lon=4.482
    21: "basin",     # lon=4.500
    22: "basin",     # lon=4.525
    23: "approach",  # lon=3.985 — large sea polygon
    24: "approach",  # lon=4.004 — Maasvlakte entrance
    25: "channel",   # lon=4.065 — Europoort
    26: "channel",   # lon=4.136 — top river
}


def validate_zone_map(gdf: gpd.GeoDataFrame) -> None:
    """Assert that centroid positions are consistent with the zone assignments."""
    for idx, zone_type in ZONE_MAP.items():
        if idx >= len(gdf):
            continue
        centroid = gdf.iloc[idx].geometry.centroid
        lon = centroid.x
        if zone_type == "approach" and lon > 4.08:
            raise ValueError(
                f"Polygon {idx} assigned 'approach' but centroid lon={lon:.3f} > 4.08"
            )
        if zone_type == "basin" and lon < 4.15:
            raise ValueError(
                f"Polygon {idx} assigned 'basin' but centroid lon={lon:.3f} < 4.15"
            )


def tag_zones(gpkg_path: Path) -> gpd.GeoDataFrame:
    """Load rotterdam.gpkg and add zone_id, zone_type, zone_name columns."""
    gdf = gpd.read_file(gpkg_path)
    logger.info("Loaded %d polygons from %s", len(gdf), gpkg_path.name)

    validate_zone_map(gdf)

    gdf["zone_id"] = gdf.index
    gdf["zone_type"] = gdf.index.map(ZONE_MAP).fillna("unknown")
    gdf["zone_name"] = gdf["zone_type"] + "_" + gdf["zone_id"].astype(str)

    counts = gdf["zone_type"].value_counts().to_dict()
    logger.info("Zone types: %s", counts)
    return gdf


def build_dissolved_aoi(zones_gdf: gpd.GeoDataFrame, output_path: Path) -> dict:
    """Dissolve all zone polygons into a single AOI polygon and save as GeoJSON.

    Returns the GeoJSON dict (for direct use in GFW API calls).
    """
    dissolved = unary_union(zones_gdf.geometry)
    geojson = json.loads(gpd.GeoSeries([dissolved], crs=zones_gdf.crs).to_json())
    # Extract the single feature's geometry
    aoi_geojson = geojson["features"][0]["geometry"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aoi_geojson, f)
    logger.info("Saved dissolved AOI to %s", output_path)

    return aoi_geojson


def assign_detections_to_zones(
    detection_dir: Path,
    zones_gdf: gpd.GeoDataFrame,
    scene_ids: list[str],
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Spatial join detection centroids into zone polygons.

    For each scene's detection parquet, assigns zone_type to each detection.
    Returns a summary DataFrame with per-scene, per-zone counts.
    """
    # Prepare zones for spatial join
    zones_simple = zones_gdf[["zone_id", "zone_type", "geometry"]].copy()

    records = []
    for scene_id in scene_ids:
        det_path = detection_dir / f"{scene_id}.parquet"
        if not det_path.exists():
            continue

        det_gdf = gpd.read_parquet(det_path)
        if len(det_gdf) == 0:
            continue

        # Ensure same CRS
        if det_gdf.crs != zones_simple.crs:
            det_gdf = det_gdf.to_crs(zones_simple.crs)

        # Spatial join: each detection point → zone
        joined = gpd.sjoin(det_gdf, zones_simple, how="left", predicate="within")

        # Aggregate by zone_type
        for zone_type in ["approach", "channel", "basin"]:
            zone_dets = joined[joined["zone_type"] == zone_type]
            records.append({
                "scene_id": scene_id,
                "zone_type": zone_type,
                "zone_vessel_count": len(zone_dets),
                "zone_pixel_area": int(zone_dets["pixel_count"].sum()) if len(zone_dets) > 0 else 0,
            })

        # Unassigned detections
        unassigned = joined[joined["zone_type"].isna()]
        if len(unassigned) > 0:
            records.append({
                "scene_id": scene_id,
                "zone_type": None,
                "zone_vessel_count": len(unassigned),
                "zone_pixel_area": int(unassigned["pixel_count"].sum()),
            })

    df = pd.DataFrame(records)
    logger.info(
        "Zone detections: %d records across %d scenes",
        len(df), df["scene_id"].nunique(),
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / "zone_detections.parquet", index=False)

    return df
