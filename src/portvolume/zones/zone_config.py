"""Zone configuration: load and represent port zone definitions from GeoJSON."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely.geometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZoneDefinition:
    """A single zone within a port area."""

    zone_id: str
    zone_type: str  # "anchorage", "berth", "channel", "approach"
    geometry: shapely.geometry.base.BaseGeometry  # Polygon or MultiPolygon
    capacity_vessels: int | None = None


@dataclass(frozen=True)
class PortZoneConfig:
    """All zone definitions for a single port."""

    port_id: str
    zones: tuple[ZoneDefinition, ...]  # tuple for frozen dataclass

    @property
    def anchorage_zones(self) -> list[ZoneDefinition]:
        """Return all zones with zone_type='anchorage'."""
        return [z for z in self.zones if z.zone_type == "anchorage"]

    @property
    def berth_zones(self) -> list[ZoneDefinition]:
        """Return all zones with zone_type='berth'."""
        return [z for z in self.zones if z.zone_type == "berth"]

    @property
    def channel_zones(self) -> list[ZoneDefinition]:
        """Return all zones with zone_type='channel'."""
        return [z for z in self.zones if z.zone_type == "channel"]

    def zone_gdf(self) -> gpd.GeoDataFrame:
        """Build a GeoDataFrame from the zone definitions for spatial joins."""
        records = [
            {
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "geometry": z.geometry,
                "capacity_vessels": z.capacity_vessels,
            }
            for z in self.zones
        ]
        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def load_zone_config(geojson_path: Path) -> PortZoneConfig:
    """Load zone definitions from a GeoJSON file.

    Features must have properties: zone_id, zone_type, and optionally
    capacity_vessels.  The port_id is derived from the file stem.
    """
    gdf = gpd.read_file(geojson_path)
    port_id = geojson_path.stem

    zones: list[ZoneDefinition] = []
    for _, row in gdf.iterrows():
        cap = row.get("capacity_vessels")
        # geopandas reads JSON null as NaN — normalise to None
        if cap is not None and pd.isna(cap):
            cap = None
        elif cap is not None:
            cap = int(cap)
        zones.append(
            ZoneDefinition(
                zone_id=row["zone_id"],
                zone_type=row["zone_type"],
                geometry=row.geometry,
                capacity_vessels=cap,
            )
        )

    logger.info(
        "Loaded %d zones for port %s from %s",
        len(zones),
        port_id,
        geojson_path.name,
    )
    return PortZoneConfig(port_id=port_id, zones=tuple(zones))


def load_all_zone_configs(zones_dir: Path) -> dict[str, PortZoneConfig]:
    """Load all {port_id}.geojson files from a directory.

    Returns a dict mapping port_id -> PortZoneConfig.
    """
    configs: dict[str, PortZoneConfig] = {}
    if not zones_dir.exists():
        logger.warning("Zones directory does not exist: %s", zones_dir)
        return configs

    for geojson_path in sorted(zones_dir.glob("*.geojson")):
        config = load_zone_config(geojson_path)
        configs[config.port_id] = config
        logger.info("Loaded zone config for %s", config.port_id)

    return configs
