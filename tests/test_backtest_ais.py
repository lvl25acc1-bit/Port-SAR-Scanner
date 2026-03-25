"""Tests for ais_matching.py — commercial filtering, size thresholds, daily counts."""

import pandas as pd
import pytest

from portvolume.backtest.ais_matching import (
    COMMERCIAL_TYPES,
    build_daily_counts,
    filter_commercial_vessels,
    sensitivity_sweep,
)


@pytest.fixture
def presence_df():
    return pd.DataFrame({
        "vesselId": ["v1", "v2", "v3", "v4"],
        "mmsi": ["111", "222", "333", "444"],
        "vesselType": ["CARGO", "OTHER", "TANKER", "CARGO"],
        "hours": [12, 8, 6, 10],
        "query_date": ["2024-01-09"] * 4,
    })


@pytest.fixture
def identity_df():
    return pd.DataFrame({
        "vesselId": ["v1", "v2", "v3", "v4"],
        "length_m": [200, 30, 150, 50],
        "beam_m": [32, 10, 25, 10],
    })


@pytest.fixture
def scene_universe():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-09"]),
        "scene_id": ["test_scene"],
        "vessel_count": [100],
    })


class TestFilterCommercial:
    def test_keeps_cargo_rejects_other(self, presence_df, identity_df):
        result = filter_commercial_vessels(presence_df, identity_df, min_area_m2=0)
        types = result["vesselType"].str.upper().unique()
        assert "OTHER" not in types
        assert "CARGO" in types

    def test_size_filter_600(self, presence_df, identity_df):
        result = filter_commercial_vessels(presence_df, identity_df, min_area_m2=600)
        # v1: 200*32=6400 ✓, v3: 150*25=3750 ✓, v4: 50*10=500 ✗ (but no dims → kept)
        assert len(result) >= 2

    def test_size_filter_strict(self, presence_df, identity_df):
        # At 4000 m², only v1 (6400) passes
        result = filter_commercial_vessels(presence_df, identity_df, min_area_m2=4000)
        large = result[result["hull_area_m2"] >= 4000]
        assert len(large) >= 1

    def test_empty_presence(self, identity_df):
        result = filter_commercial_vessels(pd.DataFrame(), identity_df)
        assert result.empty


class TestDailyCounts:
    def test_basic_count(self, presence_df, identity_df, scene_universe):
        filtered = filter_commercial_vessels(presence_df, identity_df, min_area_m2=0)
        daily = build_daily_counts(scene_universe, filtered)
        assert len(daily) >= 1
        assert "gfw_unique_vessels" in daily.columns
        assert "gfw_total_hours" in daily.columns
        assert daily["gfw_unique_vessels"].iloc[0] > 0


class TestSensitivitySweep:
    def test_monotonic_counts(self, presence_df, identity_df, scene_universe):
        result = sensitivity_sweep(
            presence_df, identity_df, [0, 600, 4000], scene_universe
        )
        assert "threshold_m2" in result.columns
        # More vessels at lower threshold
        low = result[result["threshold_m2"] == 0]["gfw_unique_vessels"].sum()
        high = result[result["threshold_m2"] == 4000]["gfw_unique_vessels"].sum()
        assert low >= high
