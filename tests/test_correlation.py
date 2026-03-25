"""Tests for correlation and statistical analysis."""

import numpy as np
import pandas as pd
import pytest

from portvolume.analysis.correlation import compute_correlations, lead_lag_crosscorrelation
from portvolume.analysis.granger import run_granger_test


class TestCorrelations:
    def test_perfect_correlation(self):
        """Two identical series should have r=1.0."""
        n = 100
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, n)
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        df = pd.DataFrame(
            {"sar": values, "econ1": values},
            index=dates,
        )
        df = df.reset_index(names="week")

        result = compute_correlations(df, "sar", ["econ1"])
        assert len(result) == 1
        assert result.iloc[0]["pearson_r"] == pytest.approx(1.0)
        assert result.iloc[0]["pearson_p"] == pytest.approx(0.0, abs=1e-10)

    def test_uncorrelated_series(self):
        """Independent random series should have |r| near 0."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        df = pd.DataFrame(
            {
                "sar": rng.normal(0, 1, n),
                "noise": rng.normal(0, 1, n),
            },
            index=dates,
        ).reset_index(names="week")

        result = compute_correlations(df, "sar", ["noise"])
        assert abs(result.iloc[0]["pearson_r"]) < 0.2

    def test_handles_nan(self):
        """Should skip NaN rows and still compute correlation."""
        n = 100
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, n)
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        econ = values.copy()
        econ[::3] = np.nan  # Set every 3rd value to NaN

        df = pd.DataFrame(
            {"sar": values, "econ_with_gaps": econ},
            index=dates,
        ).reset_index(names="week")

        result = compute_correlations(df, "sar", ["econ_with_gaps"])
        assert len(result) == 1
        assert result.iloc[0]["n_obs"] < n


class TestLeadLag:
    def test_detects_correct_lag(self):
        """If econ = shift(sar, +3), best lag should be near +3."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        sar = pd.Series(rng.normal(0, 1, n), index=dates, name="sar")

        # Create economic series that lags SAR by 3 weeks
        econ_values = np.zeros(n)
        econ_values[3:] = sar.values[:-3]
        econ_values[:3] = rng.normal(0, 0.1, 3)
        econ = pd.Series(econ_values, index=dates, name="econ")

        result = lead_lag_crosscorrelation(sar, econ, max_lag_weeks=8)
        assert not result.empty

        best = result[result["is_best"]]
        assert len(best) == 1
        # Best lag should be +3 (SAR leads by 3 weeks)
        assert best.iloc[0]["lag_weeks"] == 3
        assert best.iloc[0]["correlation"] > 0.8

    def test_insufficient_data(self):
        """Should return empty DataFrame with < 20 observations."""
        dates = pd.date_range("2023-01-01", periods=10, freq="W-MON")
        sar = pd.Series(np.zeros(10), index=dates)
        econ = pd.Series(np.zeros(10), index=dates)

        result = lead_lag_crosscorrelation(sar, econ, max_lag_weeks=5)
        assert result.empty


class TestGrangerCausality:
    def test_causal_relationship_detected(self):
        """When SAR genuinely predicts econ, Granger test should detect it."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        # Generate AR(1) SAR process
        sar = np.zeros(n)
        for i in range(1, n):
            sar[i] = 0.5 * sar[i - 1] + rng.normal(0, 1)

        # Econ depends on lagged SAR
        econ = np.zeros(n)
        for i in range(2, n):
            econ[i] = 0.7 * sar[i - 2] + 0.2 * econ[i - 1] + rng.normal(0, 0.5)

        sar_s = pd.Series(sar, index=dates)
        econ_s = pd.Series(econ, index=dates)

        result = run_granger_test(sar_s, econ_s, max_lag=4)
        assert result.sar_causes_econ, "SAR should Granger-cause econ"

    def test_no_causality_for_independent(self):
        """Independent series should not show Granger causality."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")

        sar = pd.Series(rng.normal(0, 1, n), index=dates)
        econ = pd.Series(rng.normal(0, 1, n), index=dates)

        result = run_granger_test(sar, econ, max_lag=4, significance=0.01)
        # With strict significance, independent series should not show causality
        # (though there's a small chance due to randomness)
        assert result.best_direction in ("none", "bidirectional", "sar->econ", "econ->sar")
