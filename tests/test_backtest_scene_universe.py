"""Tests for scene_universe.py — scene_id parsing, pass_family, orbit direction, QC flags."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from portvolume.backtest.scene_universe import (
    build_scene_universe,
    derive_orbit_direction,
    derive_pass_family,
    parse_scene_id,
    validate_orbit_direction,
)


class TestParseSceneId:
    def test_s1a_format(self):
        sid = "S1A_IW_GRDH_1SDV_20240101T055039_20240101T055104_051909_064584_rtc"
        result = parse_scene_id(sid)
        assert result["satellite"] == "S1A"
        assert result["start_utc"] == datetime(2024, 1, 1, 5, 50, 39, tzinfo=timezone.utc)
        assert result["end_utc"] == datetime(2024, 1, 1, 5, 51, 4, tzinfo=timezone.utc)
        assert result["scene_duration_s"] == 25.0
        assert result["abs_orbit"] == 51909

    def test_s1c_format(self):
        sid = "S1C_IW_GRDH_1SDV_20250427T173223_20250427T173248_002082_rtc"
        result = parse_scene_id(sid)
        assert result["satellite"] == "S1C"
        assert result["start_utc"].year == 2025
        assert result["start_utc"].hour == 17
        assert result["scene_duration_s"] == 25.0
        assert result["abs_orbit"] == 2082

    def test_midpoint_calculation(self):
        sid = "S1A_IW_GRDH_1SDV_20240101T055039_20240101T055104_051909_064584_rtc"
        result = parse_scene_id(sid)
        mid = result["scene_midpoint_utc"]
        assert mid.hour == 5
        assert mid.minute == 50
        assert mid.second == 51 or mid.second == 52  # 39 + 25/2 ≈ 51.5


class TestPassFamily:
    def test_morning(self):
        dt = datetime(2024, 1, 1, 5, 50, 0, tzinfo=timezone.utc)
        assert derive_pass_family(dt) == "morning"

    def test_evening(self):
        dt = datetime(2024, 1, 4, 17, 25, 0, tzinfo=timezone.utc)
        assert derive_pass_family(dt) == "evening"

    def test_noon_boundary(self):
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert derive_pass_family(dt) == "evening"

    def test_just_before_noon(self):
        dt = datetime(2024, 1, 1, 11, 59, 59, tzinfo=timezone.utc)
        assert derive_pass_family(dt) == "morning"


class TestOrbitDirection:
    def test_odd_ascending(self):
        assert derive_orbit_direction(51909) == "ascending"

    def test_even_descending(self):
        assert derive_orbit_direction(2082) == "descending"

    def test_s1c_odd_ascending(self):
        assert derive_orbit_direction(6807) == "ascending"

    def test_validation_ascending_morning(self):
        assert validate_orbit_direction("ascending", "morning") is True

    def test_validation_descending_evening(self):
        assert validate_orbit_direction("descending", "evening") is True

    def test_validation_mismatch(self):
        assert validate_orbit_direction("ascending", "evening") is False


class TestBuildSceneUniverse:
    def test_full_universe(self, tmp_path):
        from pathlib import Path
        csv_path = Path("/Users/davidbass/Documents/FinancelModels/Port Volume/data/rotterdam_vessel_counts_raw.csv")
        if not csv_path.exists():
            pytest.skip("CSV not available")
        df = build_scene_universe(csv_path, cache_dir=tmp_path)

        # Shape
        assert len(df) >= 349
        # Required columns
        for col in [
            "date", "scene_id", "vessel_count", "total_pixel_area",
            "mean_cluster_size", "satellite", "start_utc", "end_utc",
            "scene_midpoint_utc", "scene_duration_s", "abs_orbit",
            "pass_family", "orbit_direction", "footprint_id",
            "raw_partner_missing", "qc_clutter_flag",
        ]:
            assert col in df.columns, f"Missing column: {col}"

        # Satellites
        assert set(df["satellite"].unique()) == {"S1A", "S1C"}

        # Pass families
        assert set(df["pass_family"].unique()) == {"morning", "evening"}

        # QC flag on known gap
        gap_rows = df[df["raw_partner_missing"]]
        assert len(gap_rows) <= 1  # may be 0 if date not in CSV

        # Parquet written
        assert (tmp_path / "scene_universe.parquet").exists()
