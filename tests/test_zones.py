"""Tests for zone detection module — config loading, spatial join, metrics, timeseries."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from portvolume.zones.zone_config import PortZoneConfig, ZoneDefinition, load_zone_config
from portvolume.zones.zone_detector import (
    assign_detections_to_zones,
    build_congestion_timeseries,
    compute_zone_metrics,
)

ROTTERDAM_GEOJSON = Path(__file__).resolve().parent.parent / "config" / "zones" / "rotterdam.geojson"


class TestLoadZoneConfig:
    def test_load_zone_config_rotterdam(self):
        """Load the actual rotterdam.geojson and verify zone types."""
        config = load_zone_config(ROTTERDAM_GEOJSON)

        assert config.port_id == "rotterdam"
        assert len(config.zones) == 6

        zone_types = {z.zone_type for z in config.zones}
        assert "anchorage" in zone_types
        assert "berth" in zone_types
        assert "channel" in zone_types

        # Verify anchorage/berth property helpers
        assert len(config.anchorage_zones) == 2
        assert len(config.berth_zones) == 3
        assert len(config.channel_zones) == 1


class TestAssignDetections:
    @pytest.fixture()
    def rotterdam_config(self) -> PortZoneConfig:
        return load_zone_config(ROTTERDAM_GEOJSON)

    def test_assign_detections_all_in_anchorage(self, rotterdam_config):
        """Create 5 synthetic Point detections inside the anchorage polygon.
        Assert all get zone_type='anchorage'."""
        # Points inside anchorage_north: [3.90, 51.96] to [4.05, 51.99]
        points = [
            Point(3.95, 51.97),
            Point(3.98, 51.975),
            Point(4.00, 51.98),
            Point(4.02, 51.985),
            Point(4.03, 51.97),
        ]
        gdf = gpd.GeoDataFrame(
            {
                "geometry": points,
                "pixel_count": [50] * 5,
                "peak_db": [5.0] * 5,
                "mean_db": [2.0] * 5,
                "aspect_ratio": [1.5] * 5,
                "db_contrast": [3.0] * 5,
            },
            crs="EPSG:4326",
        )

        result = assign_detections_to_zones(gdf, rotterdam_config)

        assert len(result) == 5
        assert (result["zone_type"] == "anchorage").all()

    def test_assign_detections_mixed_zones(self, rotterdam_config):
        """Create detections in different zone polygons. Assert correct zone_type."""
        # Pick points that are unambiguously inside a single zone each.
        # anchorage_north: [3.90, 51.96] to [4.05, 51.99]
        # maasvlakte_berth: [4.00, 51.93] to [4.10, 51.96] — use upper part
        #   (above 51.94 to avoid overlap with approach_channel [4.00, 51.91]-[4.15, 51.94])
        # approach_channel: [4.00, 51.91] to [4.15, 51.94] — use lon > 4.10
        #   (outside maasvlakte_berth [4.00-4.10])
        points = [
            Point(3.95, 51.97),   # anchorage_north
            Point(4.05, 51.955),  # maasvlakte_berth (above channel overlap)
            Point(4.12, 51.915),  # approach_channel only
        ]
        gdf = gpd.GeoDataFrame(
            {
                "geometry": points,
                "pixel_count": [50, 60, 40],
                "peak_db": [5.0, 6.0, 4.0],
                "mean_db": [2.0, 3.0, 1.5],
                "aspect_ratio": [1.5, 1.2, 2.0],
                "db_contrast": [3.0, 3.0, 2.5],
            },
            crs="EPSG:4326",
        )

        result = assign_detections_to_zones(gdf, rotterdam_config)

        zone_types = set(result["zone_type"].tolist())
        assert "anchorage" in zone_types
        assert "berth" in zone_types
        assert "channel" in zone_types
        assert "outside" not in zone_types
        assert len(result) == 3  # no duplicates from overlapping polygons

    def test_outside_detections_labeled(self, rotterdam_config):
        """A detection at [0, 0] (way outside Rotterdam) gets zone_type='outside'."""
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0.0, 0.0)],
                "pixel_count": [30],
                "peak_db": [3.0],
                "mean_db": [1.0],
                "aspect_ratio": [2.0],
                "db_contrast": [2.0],
            },
            crs="EPSG:4326",
        )

        result = assign_detections_to_zones(gdf, rotterdam_config)

        assert len(result) == 1
        assert result.iloc[0]["zone_type"] == "outside"
        assert result.iloc[0]["zone_id"] == "outside"


class TestZoneMetrics:
    def test_congestion_index_calculation(self):
        """Create zoned_detections with 3 anchorage + 7 berth.
        Assert congestion_index == 0.3."""
        zone_types = ["anchorage"] * 3 + ["berth"] * 7
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0)] * 10,
                "pixel_count": [50] * 10,
                "zone_id": [f"z{i}" for i in range(10)],
                "zone_type": zone_types,
            },
            crs="EPSG:4326",
        )

        # Minimal zone config (no capacity needed for congestion_index)
        config = PortZoneConfig(port_id="test", zones=())

        metrics = compute_zone_metrics(gdf, config, "scene_001", "2024-01-15T00:00:00")

        assert metrics["anchorage_count"] == 3
        assert metrics["berth_count"] == 7
        assert metrics["total_count"] == 10
        assert metrics["congestion_index"] == pytest.approx(0.3)
        assert metrics["queue_length_proxy"] == 3  # anchorage + channel (0)


class TestCongestionTimeseries:
    def test_congestion_timeseries_weekly(self):
        """Create daily scene metrics over 3 weeks, assert weekly aggregation."""
        dates = pd.date_range("2024-01-01", periods=21, freq="D")
        records = []
        for i, dt in enumerate(dates):
            records.append(
                {
                    "scene_id": f"scene_{i:03d}",
                    "timestamp": dt.isoformat(),
                    "congestion_index": 0.3 + (i % 7) * 0.01,
                    "anchorage_count": 3,
                    "berth_count": 7,
                }
            )
        df = pd.DataFrame(records)

        result = build_congestion_timeseries(df, metric="congestion_index")

        assert len(result) >= 3  # 21 days spans 3-4 weekly bins depending on alignment
        assert "week" in result.columns
        assert "mean" in result.columns
        assert "max" in result.columns
        assert "n_scenes" in result.columns
        assert result["n_scenes"].sum() == 21
