"""Tests for zone_tagging.py — zone mapping, dissolved AOI, detection assignment."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from portvolume.backtest.zone_tagging import (
    ZONE_MAP,
    assign_detections_to_zones,
    build_dissolved_aoi,
    tag_zones,
    validate_zone_map,
)

GPKG_PATH = Path("/Users/davidbass/Documents/FinancelModels/Port Volume/config/water_masks/rotterdam.gpkg")


class TestZoneMap:
    def test_all_27_polygons_mapped(self):
        assert len(ZONE_MAP) == 27

    def test_valid_zone_types(self):
        for idx, zt in ZONE_MAP.items():
            assert zt in ("approach", "channel", "basin"), f"Invalid zone_type for {idx}: {zt}"

    def test_has_all_three_types(self):
        types = set(ZONE_MAP.values())
        assert types == {"approach", "channel", "basin"}

    @pytest.mark.skipif(not GPKG_PATH.exists(), reason="gpkg not available")
    def test_validate_against_real_gpkg(self):
        gdf = gpd.read_file(GPKG_PATH)
        validate_zone_map(gdf)  # Should not raise


class TestTagZones:
    @pytest.mark.skipif(not GPKG_PATH.exists(), reason="gpkg not available")
    def test_tag_zones_columns(self):
        gdf = tag_zones(GPKG_PATH)
        assert "zone_id" in gdf.columns
        assert "zone_type" in gdf.columns
        assert "zone_name" in gdf.columns
        assert len(gdf) == 27

    @pytest.mark.skipif(not GPKG_PATH.exists(), reason="gpkg not available")
    def test_zone_type_counts(self):
        gdf = tag_zones(GPKG_PATH)
        counts = gdf["zone_type"].value_counts()
        assert counts["basin"] > counts["approach"]
        assert counts["channel"] > counts["approach"]


class TestDissolvedAOI:
    @pytest.mark.skipif(not GPKG_PATH.exists(), reason="gpkg not available")
    def test_dissolved_aoi_valid(self, tmp_path):
        gdf = tag_zones(GPKG_PATH)
        out_path = tmp_path / "aoi.geojson"
        geojson = build_dissolved_aoi(gdf, out_path)

        assert out_path.exists()
        assert geojson["type"] in ("Polygon", "MultiPolygon")
        assert "coordinates" in geojson

        # Reload and verify it's valid geometry
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["type"] in ("Polygon", "MultiPolygon")


class TestDetectionAssignment:
    def test_synthetic_detection_in_basin(self, tmp_path):
        """A detection point inside a basin polygon should get zone_type=basin."""
        # Create a simple zone GeoDataFrame
        basin_poly = Polygon([(4.3, 51.88), (4.4, 51.88), (4.4, 51.90), (4.3, 51.90)])
        zones = gpd.GeoDataFrame(
            {"zone_id": [0], "zone_type": ["basin"], "geometry": [basin_poly]},
            crs="EPSG:4326",
        )

        # Create a detection parquet with one point inside the basin
        det = gpd.GeoDataFrame(
            {"geometry": [Point(4.35, 51.89)], "pixel_count": [50],
             "peak_db": [5.0], "mean_db": [2.0], "aspect_ratio": [1.5], "db_contrast": [3.0]},
            crs="EPSG:4326",
        )
        det_dir = tmp_path / "detections"
        det_dir.mkdir()
        det.to_parquet(det_dir / "test_scene.parquet")

        result = assign_detections_to_zones(det_dir, zones, ["test_scene"])
        basin_row = result[result["zone_type"] == "basin"]
        assert len(basin_row) == 1
        assert basin_row.iloc[0]["zone_vessel_count"] == 1

    def test_detection_outside_zones(self, tmp_path):
        """A detection outside all polygons gets zone_type=None."""
        basin_poly = Polygon([(4.3, 51.88), (4.4, 51.88), (4.4, 51.90), (4.3, 51.90)])
        zones = gpd.GeoDataFrame(
            {"zone_id": [0], "zone_type": ["basin"], "geometry": [basin_poly]},
            crs="EPSG:4326",
        )

        # Point far outside
        det = gpd.GeoDataFrame(
            {"geometry": [Point(5.0, 52.0)], "pixel_count": [30],
             "peak_db": [3.0], "mean_db": [1.0], "aspect_ratio": [2.0], "db_contrast": [2.0]},
            crs="EPSG:4326",
        )
        det_dir = tmp_path / "detections"
        det_dir.mkdir()
        det.to_parquet(det_dir / "outside_scene.parquet")

        result = assign_detections_to_zones(det_dir, zones, ["outside_scene"])
        none_rows = result[result["zone_type"].isna()]
        assert len(none_rows) == 1
