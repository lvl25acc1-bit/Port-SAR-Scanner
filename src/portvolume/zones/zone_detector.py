"""Zone-aware vessel detection: spatial join, metrics, and time series."""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from portvolume.zones.zone_config import PortZoneConfig

logger = logging.getLogger(__name__)


def assign_detections_to_zones(
    detections_gdf: gpd.GeoDataFrame,
    zone_config: PortZoneConfig,
) -> gpd.GeoDataFrame:
    """Spatial join adding zone_id and zone_type columns to detections.

    Detections that fall outside all defined zones get zone_type="outside"
    and zone_id="outside".

    Parameters
    ----------
    detections_gdf : GeoDataFrame of point detections (from cluster_detections).
    zone_config : Port zone configuration with polygon geometries.

    Returns
    -------
    GeoDataFrame with original columns plus zone_id and zone_type.
    """
    if detections_gdf.empty:
        result = detections_gdf.copy()
        result["zone_id"] = pd.Series(dtype="str")
        result["zone_type"] = pd.Series(dtype="str")
        return result

    zones_gdf = zone_config.zone_gdf()

    # Ensure matching CRS
    if detections_gdf.crs and zones_gdf.crs and detections_gdf.crs != zones_gdf.crs:
        detections_gdf = detections_gdf.to_crs(zones_gdf.crs)

    # Spatial join: each detection point -> zone polygon
    joined = gpd.sjoin(
        detections_gdf,
        zones_gdf[["zone_id", "zone_type", "geometry"]],
        how="left",
        predicate="within",
    )

    # If a detection falls in overlapping zones, keep only the first match
    joined = joined[~joined.index.duplicated(keep="first")]

    # Fill unmatched detections
    joined["zone_id"] = joined["zone_id"].fillna("outside")
    joined["zone_type"] = joined["zone_type"].fillna("outside")

    # Drop the index_right column added by sjoin
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    logger.info(
        "Zone assignment: %d detections -> %s",
        len(joined),
        joined["zone_type"].value_counts().to_dict(),
    )
    return joined


def compute_zone_metrics(
    zoned_detections: gpd.GeoDataFrame,
    zone_config: PortZoneConfig,
    scene_id: str,
    timestamp: str,
) -> dict:
    """Compute per-zone metrics for a single scene.

    Parameters
    ----------
    zoned_detections : GeoDataFrame with zone_id and zone_type columns.
    zone_config : Port zone configuration (for capacity info).
    scene_id : Identifier for the SAR scene.
    timestamp : ISO timestamp string.

    Returns
    -------
    Dict with keys: scene_id, timestamp, anchorage_count, berth_count,
    channel_count, outside_count, total_count, anchorage_occupancy_ratio,
    congestion_index, queue_length_proxy.
    """
    counts = zoned_detections["zone_type"].value_counts()
    anchorage_count = int(counts.get("anchorage", 0))
    berth_count = int(counts.get("berth", 0))
    channel_count = int(counts.get("channel", 0))
    outside_count = int(counts.get("outside", 0))
    total_count = len(zoned_detections)

    # Anchorage occupancy ratio: vessels / total capacity across anchorage zones
    total_capacity = sum(
        z.capacity_vessels
        for z in zone_config.anchorage_zones
        if z.capacity_vessels is not None
    )
    anchorage_occupancy_ratio = (
        anchorage_count / total_capacity if total_capacity > 0 else None
    )

    # Congestion index: anchorage / (anchorage + berth)
    denom = anchorage_count + berth_count
    congestion_index = anchorage_count / denom if denom > 0 else 0.0

    # Queue length proxy: ships waiting (anchorage + channel)
    queue_length_proxy = anchorage_count + channel_count

    return {
        "scene_id": scene_id,
        "timestamp": timestamp,
        "anchorage_count": anchorage_count,
        "berth_count": berth_count,
        "channel_count": channel_count,
        "outside_count": outside_count,
        "total_count": total_count,
        "anchorage_occupancy_ratio": anchorage_occupancy_ratio,
        "congestion_index": congestion_index,
        "queue_length_proxy": queue_length_proxy,
    }


def build_congestion_timeseries(
    zone_metrics_df: pd.DataFrame,
    metric: str = "congestion_index",
) -> pd.DataFrame:
    """Resample scene-level zone metrics to weekly frequency.

    Uses the same weekly aggregation pattern as aggregate.py.

    Parameters
    ----------
    zone_metrics_df : DataFrame with 'timestamp' and metric columns
        (output of compute_zone_metrics collected over many scenes).
    metric : Which metric column to aggregate (default: congestion_index).

    Returns
    -------
    DataFrame with columns: [week, mean, max, n_scenes].
    """
    if zone_metrics_df.empty:
        return pd.DataFrame(columns=["week", "mean", "max", "n_scenes"])

    df = zone_metrics_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    weekly = df[metric].resample("W-MON").agg(["mean", "max", "count"])
    weekly.columns = ["mean", "max", "n_scenes"]
    weekly = weekly[weekly["n_scenes"] > 0]
    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={"timestamp": "week"})

    logger.info(
        "Built %s timeseries: %d weeks from %d scenes",
        metric,
        len(weekly),
        len(zone_metrics_df),
    )
    return weekly
