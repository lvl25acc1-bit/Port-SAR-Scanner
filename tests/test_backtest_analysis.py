"""Tests for analysis.py — weekly aggregation, audit selection, diagnostics."""

import numpy as np
import pandas as pd
import pytest

from portvolume.backtest.analysis import (
    scene_level_diagnostic,
    select_audit_scenes,
    weekly_robustness,
)


@pytest.fixture
def merged_df():
    """Synthetic merged scene table."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2024-06-01", periods=n, freq="3D")
    return pd.DataFrame({
        "date": dates,
        "scene_id": [f"scene_{i}" for i in range(n)],
        "vessel_count": rng.integers(100, 200, n),
        "total_pixel_area": rng.integers(3000, 6000, n),
        "gfw_unique_vessels": rng.integers(80, 180, n),
        "gfw_sum_area_m2": rng.uniform(50000, 150000, n),
        "pass_family": ["morning" if i % 2 == 0 else "evening" for i in range(n)],
        "satellite": ["S1A" if i < 70 else "S1C" for i in range(n)],
        "wind_speed_mps": rng.uniform(2, 15, n),
        "tide_height_cm": rng.uniform(-100, 150, n),
        "minutes_from_high_tide": rng.uniform(-360, 360, n),
    })


class TestSceneDiagnostic:
    def test_returns_metrics(self, merged_df):
        result = scene_level_diagnostic(merged_df)
        assert "count_vs_count" in result
        metrics = result["count_vs_count"]
        assert "spearman_r" in metrics
        assert "mae" in metrics
        assert "caveat" in metrics

    def test_correlated_data_positive_spearman(self):
        """Perfectly correlated data should give positive Spearman."""
        df = pd.DataFrame({
            "vessel_count": range(50, 150),
            "gfw_unique_vessels": range(50, 150),
            "total_pixel_area": range(3000, 3100),
            "gfw_sum_area_m2": range(50000, 50100),
        })
        result = scene_level_diagnostic(df)
        assert result["count_vs_count"]["spearman_r"] > 0.9


class TestWeeklyRobustness:
    def test_has_pass_family_split(self, merged_df):
        controlled = {"residuals": []}
        weekly = weekly_robustness(merged_df, controlled)
        assert "pass_family" in weekly.columns
        families = weekly["pass_family"].unique()
        assert "morning" in families
        assert "evening" in families

    def test_raw_median_present(self, merged_df):
        controlled = {"residuals": []}
        weekly = weekly_robustness(merged_df, controlled)
        assert "raw_median_count" in weekly.columns
        assert weekly["raw_median_count"].notna().all()


class TestAuditSelection:
    def test_returns_20_or_fewer(self, merged_df):
        audit = select_audit_scenes(merged_df, n=20)
        assert len(audit) <= 20
        assert len(audit) > 0

    def test_unique_scene_ids(self, merged_df):
        audit = select_audit_scenes(merged_df, n=20)
        assert audit["scene_id"].nunique() == len(audit)
