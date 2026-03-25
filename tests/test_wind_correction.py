"""Tests for wind deconfounding module (Module 1)."""

import numpy as np
import pandas as pd
import pytest

from portvolume.detection.wind_correction import (
    WaveAgeParams,
    classify_wave_age,
    compute_cfar_correction_factor,
    compute_wind_adjusted_count,
    fit_wind_model,
)
from portvolume.detection.adaptive_cfar import adaptive_ca_cfar_2d
from portvolume.detection.cfar import ca_cfar_2d


class TestClassifyWaveAge:
    def test_classify_wave_age_high_wind(self):
        """Wind=12 m/s with short period=4s should be young_sea."""
        result = classify_wave_age(
            wind_speed_mps=12.0,
            peak_wave_period_s=4.0,
            significant_wave_height_m=2.0,
        )
        assert result == "young_sea"

    def test_classify_wave_age_swell(self):
        """Wind=3 m/s with long period=14s should be swell."""
        result = classify_wave_age(
            wind_speed_mps=3.0,
            peak_wave_period_s=14.0,
            significant_wave_height_m=1.0,
        )
        assert result == "swell"


class TestCorrectionFactor:
    def test_correction_factor_increases_with_wind(self):
        """Factor at 15 m/s should be greater than at 5 m/s for same wave_age."""
        params = WaveAgeParams()
        factor_low = compute_cfar_correction_factor("old_sea", 5.0, params)
        factor_high = compute_cfar_correction_factor("old_sea", 15.0, params)
        assert factor_high > factor_low


class TestWindAdjustedCount:
    def test_wind_adjusted_count_removes_variance(self):
        """Fit model on synthetic data; adjusted count should decorrelate from wind."""
        rng = np.random.default_rng(42)
        n = 200
        wind = rng.uniform(2.0, 15.0, n)
        noise = rng.normal(0, 3.0, n)
        raw_count = 100.0 - 5.0 * wind + noise

        # Build a DataFrame for fitting
        df = pd.DataFrame({
            "vessel_count": raw_count,
            "wind_speed_mps": wind,
            "pass_family": rng.choice(["morning", "evening"], n),
            "satellite": rng.choice(["S1A", "S1B"], n),
        })

        coefs = fit_wind_model(df, target_col="vessel_count")

        # Compute adjusted counts
        adjusted = []
        for _, row in df.iterrows():
            adj = compute_wind_adjusted_count(
                raw_count=row["vessel_count"],
                wind_speed_mps=row["wind_speed_mps"],
                pass_family=row["pass_family"],
                satellite=row["satellite"],
                model_coefficients=coefs,
            )
            adjusted.append(adj)

        adjusted = np.array(adjusted)

        # Raw should correlate with wind
        raw_corr = abs(np.corrcoef(raw_count, wind)[0, 1])
        assert raw_corr > 0.4, f"Raw correlation with wind too low: {raw_corr:.3f}"

        # Adjusted should not correlate with wind
        adj_corr = abs(np.corrcoef(adjusted, wind)[0, 1])
        assert adj_corr < 0.15, f"Adjusted correlation with wind too high: {adj_corr:.3f}"


class TestAdaptiveCFAR:
    def test_adaptive_cfar_reduces_false_alarms(self):
        """Higher correction factor should reduce detections on elevated background."""
        rng = np.random.default_rng(99)
        # Elevated uniform background simulating high wind clutter
        image = rng.exponential(scale=0.05, size=(256, 256))

        # Add a few real targets
        for r, c in [(50, 50), (150, 150), (200, 100)]:
            image[r - 2 : r + 3, c - 2 : c + 3] = 5.0

        standard = ca_cfar_2d(image, guard_cells=5, background_cells=15, pfa=1e-6)
        adaptive = adaptive_ca_cfar_2d(
            image, guard_cells=5, background_cells=15, pfa=1e-6,
            correction_factor=1.5,
        )

        assert adaptive.sum() <= standard.sum(), (
            f"Adaptive ({adaptive.sum()}) should have fewer or equal detections "
            f"than standard ({standard.sum()})"
        )

    def test_adaptive_cfar_backward_compatible(self):
        """correction_factor=1.0 should produce identical output to ca_cfar_2d."""
        rng = np.random.default_rng(77)
        image = rng.exponential(scale=0.01, size=(256, 256))

        # Add targets
        image[100, 100] = 10.0
        image[200, 200] = 10.0

        standard = ca_cfar_2d(image, guard_cells=5, background_cells=15, pfa=1e-6)
        adaptive = adaptive_ca_cfar_2d(
            image, guard_cells=5, background_cells=15, pfa=1e-6,
            correction_factor=1.0,
        )

        np.testing.assert_array_equal(standard, adaptive)


class TestFitWindModel:
    def test_fit_wind_model_coefficients(self):
        """Recovered wind coefficient should be within 1.5 of true value (-5.0)."""
        rng = np.random.default_rng(123)
        n = 300
        wind = rng.uniform(2.0, 15.0, n)
        noise = rng.normal(0, 5.0, n)
        true_wind_coef = -5.0

        raw_count = 120.0 + true_wind_coef * wind + noise

        df = pd.DataFrame({
            "vessel_count": raw_count,
            "wind_speed_mps": wind,
            "pass_family": rng.choice(["morning", "evening"], n),
            "satellite": rng.choice(["S1A", "S1B"], n),
        })

        coefs = fit_wind_model(df, target_col="vessel_count")
        recovered = coefs["wind_coef"]

        assert abs(recovered - true_wind_coef) < 1.5, (
            f"Recovered wind_coef={recovered:.2f}, expected ~{true_wind_coef}"
        )
