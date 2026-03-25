"""Tests for weekly aggregation and normalization."""

import numpy as np
import pandas as pd
import pytest

from portvolume.index.aggregate import aggregate_weekly
from portvolume.index.normalize import zscore_normalize, build_composite_index
from portvolume.sites import Site


class TestAggregateWeekly:
    def test_basic_aggregation(self):
        """30 daily observations over ~5 weeks should produce ~4-5 weekly rows."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "vessel_count": rng.integers(50, 150, 30),
            }
        )

        weekly = aggregate_weekly(df, "vessel_count", "monday")
        assert len(weekly) >= 4
        assert len(weekly) <= 6
        assert "mean" in weekly.columns
        assert "n_scenes" in weekly.columns

    def test_preserves_values(self):
        """Weekly mean should equal the input when there's 1 obs per week."""
        # One observation per Monday for 4 weeks
        dates = pd.date_range("2023-01-02", periods=4, freq="W-MON")
        values = [100.0, 200.0, 150.0, 175.0]
        df = pd.DataFrame({"timestamp": dates, "vessel_count": values})

        weekly = aggregate_weekly(df, "vessel_count", "monday")
        assert len(weekly) == 4
        np.testing.assert_array_almost_equal(weekly["mean"].values, values)

    def test_empty_input(self):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame(columns=["timestamp", "vessel_count"])
        weekly = aggregate_weekly(df, "vessel_count", "monday")
        assert weekly.empty


class TestZScoreNormalize:
    def test_output_shape(self, sample_weekly_detections):
        """Z-score output should have same length as input."""
        series = sample_weekly_detections.set_index("week")["mean"]
        z = zscore_normalize(series, rolling_window=12)
        assert len(z) == len(series)

    def test_approximate_mean_zero(self, sample_weekly_detections):
        """After warmup, z-scores should have approximately zero mean."""
        series = sample_weekly_detections.set_index("week")["mean"]
        z = zscore_normalize(series, rolling_window=12)
        # Skip first 12 (warmup) values
        tail = z.iloc[12:].dropna()
        assert abs(tail.mean()) < 0.5, f"Mean too far from 0: {tail.mean()}"


class TestCompositeIndex:
    def test_single_site(self, sample_weekly_detections, sample_port):
        """Composite of a single site should match its normalized series."""
        indices = {"rotterdam": sample_weekly_detections}
        composite = build_composite_index(
            indices, [sample_port], normalization="raw", rolling_window=52
        )
        assert not composite.empty
        assert "composite_index" in composite.columns
        assert len(composite) == len(sample_weekly_detections)

    def test_empty_input(self):
        """Empty input should return empty DataFrame."""
        composite = build_composite_index({}, [], normalization="zscore")
        assert composite.empty
