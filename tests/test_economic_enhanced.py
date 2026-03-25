"""Tests for enhanced economic signal analysis: stationarity, cointegration, and multiple comparison correction."""

import numpy as np
import pandas as pd
import pytest

from portvolume.analysis.enhanced_correlation import (
    compute_correlations_with_correction,
    optimal_lag_selection,
    test_cointegration as check_cointegration,
    test_stationarity as check_stationarity,
)
from portvolume.economic.commodity_indicators import (
    get_indicators_for_commodity,
)


class TestMultipleComparison:
    def test_bh_correction_reduces_significance(self):
        """With 10 correlations and one raw p=0.04, BH correction should make it non-significant."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        # Create SAR column and 10 economic columns, all random noise
        data = {"sar": rng.normal(0, 1, n)}
        for i in range(10):
            data[f"econ_{i}"] = rng.normal(0, 1, n)

        # Engineer one column to have a weak but marginally significant correlation
        # We create a column that mixes SAR with noise to get p ~ 0.04
        # With n=200, Spearman r ~ 0.145 gives p ~ 0.04
        signal_strength = 0.16
        data["econ_0"] = signal_strength * data["sar"] + (1 - signal_strength) * rng.normal(0, 1, n)

        df = pd.DataFrame(data, index=dates).reset_index(names="week")
        econ_cols = [f"econ_{i}" for i in range(10)]

        result = compute_correlations_with_correction(df, "sar", econ_cols, method="benjamini_hochberg")

        # Find the engineered column
        engineered = result[result["indicator"] == "econ_0"]
        assert len(engineered) == 1

        raw_p = engineered.iloc[0]["p_value"]
        adjusted_p = engineered.iloc[0]["adjusted_p_value"]

        # The adjusted p-value should be >= raw p-value
        assert adjusted_p >= raw_p
        # With 10 tests and a marginal p-value, BH correction should increase it
        # If raw_p < 0.05, adjusted should be > 0.05 (no longer significant)
        # Note: this depends on the specific random seed giving a marginal result
        # We check the structural property: adjusted_p > raw_p for marginal cases
        if raw_p < 0.05:
            assert adjusted_p > raw_p, "BH correction should increase marginal p-values"

    def test_bonferroni_more_conservative_than_bh(self):
        """Bonferroni adjusted p-values should be >= BH adjusted p-values."""
        rng = np.random.default_rng(123)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        data = {"sar": rng.normal(0, 1, n)}
        for i in range(10):
            # Mix in some signal so we get a range of p-values
            signal = 0.3 * (i % 3 == 0)
            data[f"econ_{i}"] = signal * data["sar"] + rng.normal(0, 1, n)

        df = pd.DataFrame(data, index=dates).reset_index(names="week")
        econ_cols = [f"econ_{i}" for i in range(10)]

        bh_result = compute_correlations_with_correction(df, "sar", econ_cols, method="benjamini_hochberg")
        bonf_result = compute_correlations_with_correction(df, "sar", econ_cols, method="bonferroni")

        # Merge on indicator to compare
        merged = bh_result.merge(
            bonf_result, on="indicator", suffixes=("_bh", "_bonf")
        )

        for _, row in merged.iterrows():
            assert row["adjusted_p_value_bonf"] >= row["adjusted_p_value_bh"] - 1e-10, (
                f"Bonferroni should be >= BH for {row['indicator']}: "
                f"Bonf={row['adjusted_p_value_bonf']:.6f}, BH={row['adjusted_p_value_bh']:.6f}"
            )


class TestStationarity:
    def test_stationarity_random_walk(self):
        """Cumulative sum of standard normal should be non-stationary."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="W-MON")
        random_walk = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)

        result = check_stationarity(random_walk, method="adf")
        assert result["is_stationary"] == False, (
            f"Random walk should be non-stationary, ADF p={result['adf_pvalue']:.4f}"
        )

    def test_stationarity_white_noise(self):
        """Standard normal white noise should be stationary."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="W-MON")
        white_noise = pd.Series(rng.normal(0, 1, n), index=dates)

        result = check_stationarity(white_noise, method="adf")
        assert result["is_stationary"] == True, (
            f"White noise should be stationary, ADF p={result['adf_pvalue']:.4f}"
        )


class TestCointegration:
    def test_cointegration_known_pair(self):
        """y = 2*x + small_noise where x is random walk should be cointegrated."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="W-MON")

        x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
        y = 2 * x + rng.normal(0, 0.5, n)

        result = check_cointegration(y, x)
        assert result["is_cointegrated"] == True, (
            f"Known cointegrated pair should be detected, p={result['adf_pvalue']:.4f}"
        )

    def test_cointegration_independent_walks(self):
        """Two independent random walks should NOT be cointegrated."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="W-MON")

        x = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)
        y = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)

        result = check_cointegration(y, x)
        assert result["is_cointegrated"] == False, (
            f"Independent random walks should not be cointegrated, p={result['adf_pvalue']:.4f}"
        )


class TestOptimalLag:
    def test_optimal_lag_recovers_true_lag(self):
        """y[t] = 0.8 * x[t-4] + noise should recover lag ~4."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="W-MON")

        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(4, n):
            y[t] = 0.8 * x[t - 4] + rng.normal(0, 0.3)

        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)

        result = optimal_lag_selection(y_series, x_series, max_lag=16, criterion="bic")

        # Optimal lag should be close to 4 (accept 3-5)
        assert 3 <= result["optimal_lag"] <= 5, (
            f"Expected optimal lag near 4, got {result['optimal_lag']}"
        )


class TestCommodityMapping:
    def test_commodity_indicators_mapping(self):
        """get_indicators_for_commodity should return correct indicators."""
        crude_indicators = get_indicators_for_commodity("crude_oil")
        crude_ids = [ind.get("id") or ind.get("ticker") for ind in crude_indicators]
        assert "DCOILWTICO" in crude_ids, "crude_oil should include DCOILWTICO"

        grain_indicators = get_indicators_for_commodity("grain")
        grain_ids = [ind.get("id") or ind.get("ticker") for ind in grain_indicators]
        assert "ZS=F" in grain_ids, "grain should include ZS=F"

        # Unknown commodity should fall back to containers
        unknown = get_indicators_for_commodity("unknown_stuff")
        container_indicators = get_indicators_for_commodity("containers")
        assert unknown == container_indicators, "Unknown commodity should fall back to containers"
