"""Tests for the validation framework (Module 5)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from portvolume.validation.metrics import (
    ValidationResult,
    validate_improvement,
    validate_signal,
)
from portvolume.validation.reference_data import (
    ReferenceDataset,
    align_reference_to_sar,
    create_synthetic_reference,
)
from portvolume.validation.report import generate_scorecard, generate_validation_report
from portvolume.validation.backtest_runner import run_full_validation


class TestValidatePerfectCorrelation:
    def test_validate_perfect_correlation(self):
        """Two identical series -> rho=1.0, passes_threshold=True, directional_accuracy=1.0."""
        n = 52
        rng = np.random.default_rng(42)
        values = rng.normal(100, 15, n)

        sar = pd.Series(values)
        ref = pd.Series(values)

        result = validate_signal(
            sar_series=sar,
            reference_series=ref,
            signal_name="raw_count",
            reference_name="ground_truth",
            reference_source="manual",
            port_id="test_port",
        )

        assert result.spearman_rho == pytest.approx(1.0, abs=1e-6)
        assert result.passes_threshold is True
        assert result.directional_accuracy == pytest.approx(1.0, abs=1e-6)
        assert result.n_observations == n


class TestValidateNoCorrelation:
    def test_validate_no_correlation(self):
        """SAR = random, reference = random (different seed). |rho| < 0.3, passes=False."""
        n = 200
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(999)

        sar = pd.Series(rng1.normal(100, 15, n))
        ref = pd.Series(rng2.normal(50, 10, n))

        result = validate_signal(
            sar_series=sar,
            reference_series=ref,
            signal_name="raw_count",
            reference_name="random_ref",
            reference_source="manual",
            port_id="test_port",
        )

        assert abs(result.spearman_rho) < 0.3
        assert result.passes_threshold is False


class TestValidateImprovementPositive:
    def test_validate_improvement_positive(self):
        """old_result with rho=0.1, new_result with rho=0.5. improvement > 0."""
        old = ValidationResult(
            port_id="p", signal_name="s", reference_name="r",
            reference_source="manual", n_observations=50,
            spearman_rho=0.1, spearman_p=0.5,
            pearson_r=0.1, pearson_p=0.5,
            rmse=1.0, mae=0.8,
            directional_accuracy=0.5,
            optimal_lag_weeks=0, correlation_at_optimal_lag=0.1,
            passes_threshold=False,
        )
        new = ValidationResult(
            port_id="p", signal_name="s", reference_name="r",
            reference_source="manual", n_observations=50,
            spearman_rho=0.5, spearman_p=0.001,
            pearson_r=0.5, pearson_p=0.001,
            rmse=0.7, mae=0.5,
            directional_accuracy=0.7,
            optimal_lag_weeks=0, correlation_at_optimal_lag=0.5,
            passes_threshold=True,
        )

        imp = validate_improvement(old, new)
        assert imp["improvement_pct"] > 0
        assert imp["rho_improved"] is True
        assert imp["delta_rho"] == pytest.approx(0.4, abs=1e-4)


class TestValidateImprovementNotSignificant:
    def test_validate_improvement_not_significant(self):
        """old rho=0.30, new rho=0.32. delta_rho=0.02."""
        old = ValidationResult(
            port_id="p", signal_name="s", reference_name="r",
            reference_source="manual", n_observations=50,
            spearman_rho=0.30, spearman_p=0.05,
            pearson_r=0.30, pearson_p=0.05,
            rmse=1.0, mae=0.8,
            directional_accuracy=0.5,
            optimal_lag_weeks=0, correlation_at_optimal_lag=0.30,
            passes_threshold=True,
        )
        new = ValidationResult(
            port_id="p", signal_name="s", reference_name="r",
            reference_source="manual", n_observations=50,
            spearman_rho=0.32, spearman_p=0.04,
            pearson_r=0.32, pearson_p=0.04,
            rmse=0.95, mae=0.75,
            directional_accuracy=0.55,
            optimal_lag_weeks=0, correlation_at_optimal_lag=0.32,
            passes_threshold=True,
        )

        imp = validate_improvement(old, new)
        assert imp["delta_rho"] == pytest.approx(0.02, abs=1e-4)
        # Verify the math is correct
        assert imp["rho_improved"] is True
        expected_pct = (0.32 - 0.30) / 0.30 * 100
        assert imp["improvement_pct"] == pytest.approx(expected_pct, abs=0.1)


class TestScorecardFormat:
    def test_scorecard_format(self):
        """Generate results for 3 ports, verify scorecard has correct columns and 3+ rows."""
        results = []
        for port_id in ["port_a", "port_b", "port_c"]:
            results.append(ValidationResult(
                port_id=port_id,
                signal_name="raw_count",
                reference_name="ref_1",
                reference_source="manual",
                n_observations=50,
                spearman_rho=0.4 if port_id == "port_a" else 0.2,
                spearman_p=0.01 if port_id == "port_a" else 0.2,
                pearson_r=0.4,
                pearson_p=0.01,
                rmse=1.0,
                mae=0.8,
                directional_accuracy=0.6,
                optimal_lag_weeks=0,
                correlation_at_optimal_lag=0.4,
                passes_threshold=(port_id == "port_a"),
            ))

        scorecard = generate_scorecard(results)

        expected_cols = {
            "port_id", "signal_name", "reference_name", "rho",
            "p_value", "optimal_lag", "passes", "directional_accuracy",
        }
        assert expected_cols.issubset(set(scorecard.columns))
        assert len(scorecard) >= 3
        # Should be sorted by |rho| descending
        assert scorecard.iloc[0]["port_id"] == "port_a"


class TestReportGeneration:
    def test_report_generation(self, tmp_path):
        """Generate report to a tmp dir, verify .csv, .json, .md files exist and are non-empty."""
        results = [
            ValidationResult(
                port_id="rotterdam",
                signal_name="raw_count",
                reference_name="gfw_ais",
                reference_source="gfw_ais",
                n_observations=50,
                spearman_rho=0.35,
                spearman_p=0.01,
                pearson_r=0.30,
                pearson_p=0.02,
                rmse=0.9,
                mae=0.7,
                directional_accuracy=0.65,
                optimal_lag_weeks=1,
                correlation_at_optimal_lag=0.40,
                passes_threshold=True,
            ),
        ]

        out = generate_validation_report(results, tmp_path)

        csv_path = out / "validation_scorecard.csv"
        json_path = out / "validation_report.json"
        md_path = out / "validation_summary.md"

        assert csv_path.exists() and csv_path.stat().st_size > 0
        assert json_path.exists() and json_path.stat().st_size > 0
        assert md_path.exists() and md_path.stat().st_size > 0

        # Validate JSON structure
        with open(json_path) as f:
            data = json.load(f)
        assert data["n_results"] == 1
        assert data["n_passing"] == 1


class TestAlignMonthlyToWeekly:
    def test_align_monthly_to_weekly(self):
        """Monthly reference (12 values) aligned to weekly SAR (52 weeks). Forward-fill -> 52 rows."""
        weekly_dates = pd.date_range("2024-01-01", periods=52, freq="W-MON")
        sar_df = pd.DataFrame({
            "week": weekly_dates,
            "mean": np.random.default_rng(42).normal(100, 10, 52),
        })

        monthly_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        ref = ReferenceDataset(
            name="monthly_ref",
            source="eurostat",
            frequency="monthly",
            coverage=["test_port"],
            data=pd.DataFrame({
                "date": monthly_dates,
                "value": np.arange(100, 112, dtype=float),
            }),
        )

        aligned = align_reference_to_sar(sar_df, ref, sar_date_col="week", sar_value_col="mean")

        # Should have up to 52 rows (limited by reference coverage)
        assert len(aligned) == 52
        assert "sar_value" in aligned.columns
        assert "reference_value" in aligned.columns
        assert aligned["reference_value"].notna().all()


class TestFullValidationE2E:
    def test_full_validation_e2e(self):
        """Create 2 ports, 2 signals each, 1 reference each. Verify 4 results."""
        n = 52
        rng = np.random.default_rng(42)

        signals = {}
        references = {}

        for port_id in ["port_a", "port_b"]:
            signals[port_id] = {
                "raw_count": pd.DataFrame({"value": rng.normal(100, 10, n)}),
                "wind_adjusted": pd.DataFrame({"value": rng.normal(90, 12, n)}),
            }
            references[port_id] = [
                ReferenceDataset(
                    name=f"ref_{port_id}",
                    source="manual",
                    frequency="weekly",
                    coverage=[port_id],
                    data=pd.DataFrame({
                        "date": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
                        "value": rng.normal(100, 10, n),
                    }),
                ),
            ]

        results = run_full_validation(signals, references)

        # 2 ports x 2 signals x 1 reference = 4 results
        assert len(results) == 4
        assert all(isinstance(r, ValidationResult) for r in results)

        # Verify all ports and signals are represented
        port_ids = {r.port_id for r in results}
        assert port_ids == {"port_a", "port_b"}
        signal_names = {r.signal_name for r in results}
        assert signal_names == {"raw_count", "wind_adjusted"}
