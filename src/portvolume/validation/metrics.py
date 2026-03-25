"""Validation metrics: signal-vs-reference comparison with pass/fail thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating one SAR signal against one reference."""

    port_id: str
    signal_name: str  # "raw_count", "wind_adjusted", "anchorage_count", "congestion_index"
    reference_name: str
    reference_source: str
    n_observations: int
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    rmse: float
    mae: float
    directional_accuracy: float  # fraction of periods where week-over-week direction matches
    optimal_lag_weeks: int
    correlation_at_optimal_lag: float
    passes_threshold: bool  # True if |spearman_rho| > minimum_threshold AND spearman_p < 0.05
    note: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "port_id": self.port_id,
            "signal_name": self.signal_name,
            "reference_name": self.reference_name,
            "reference_source": self.reference_source,
            "n_observations": self.n_observations,
            "spearman_rho": round(self.spearman_rho, 4),
            "spearman_p": round(self.spearman_p, 6),
            "pearson_r": round(self.pearson_r, 4),
            "pearson_p": round(self.pearson_p, 6),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "optimal_lag_weeks": self.optimal_lag_weeks,
            "correlation_at_optimal_lag": round(self.correlation_at_optimal_lag, 4),
            "passes_threshold": self.passes_threshold,
            "note": self.note,
        }

    def summary_line(self) -> str:
        """One-line summary: 'port_id | signal | ref | rho=X.XX (p=X.XX) | PASS/FAIL'"""
        verdict = "PASS" if self.passes_threshold else "FAIL"
        return (
            f"{self.port_id} | {self.signal_name} | {self.reference_name} | "
            f"rho={self.spearman_rho:.2f} (p={self.spearman_p:.4f}) | {verdict}"
        )


def validate_signal(
    sar_series: pd.Series,
    reference_series: pd.Series,
    signal_name: str,
    reference_name: str,
    reference_source: str,
    port_id: str,
    min_rho: float = 0.3,
    max_lag: int = 12,
) -> ValidationResult:
    """Run full validation of one SAR signal against one reference.

    Steps:
    1. Align series (drop NaN)
    2. Compute Spearman and Pearson correlations
    3. Compute RMSE and MAE (after z-score normalization of both)
    4. Compute directional accuracy (sign of diff matches)
    5. Find optimal lag via cross-correlation
    6. Determine pass/fail
    """
    # Step 1: align — drop NaN
    combined = pd.DataFrame({"sar": sar_series, "ref": reference_series}).dropna()
    n = len(combined)

    # Edge case: too few observations
    if n < 10:
        logger.warning(
            "Too few observations (%d) for %s / %s at %s",
            n, signal_name, reference_name, port_id,
        )
        return ValidationResult(
            port_id=port_id,
            signal_name=signal_name,
            reference_name=reference_name,
            reference_source=reference_source,
            n_observations=n,
            spearman_rho=0.0,
            spearman_p=1.0,
            pearson_r=0.0,
            pearson_p=1.0,
            rmse=float("nan"),
            mae=float("nan"),
            directional_accuracy=0.0,
            optimal_lag_weeks=0,
            correlation_at_optimal_lag=0.0,
            passes_threshold=False,
            note=f"insufficient observations ({n} < 10)",
        )

    sar_vals = combined["sar"].values.astype(float)
    ref_vals = combined["ref"].values.astype(float)

    # Edge case: constant series
    if np.std(sar_vals) == 0 or np.std(ref_vals) == 0:
        logger.warning("Constant series detected for %s / %s at %s", signal_name, reference_name, port_id)
        return ValidationResult(
            port_id=port_id,
            signal_name=signal_name,
            reference_name=reference_name,
            reference_source=reference_source,
            n_observations=n,
            spearman_rho=0.0,
            spearman_p=1.0,
            pearson_r=0.0,
            pearson_p=1.0,
            rmse=float("nan"),
            mae=float("nan"),
            directional_accuracy=0.0,
            optimal_lag_weeks=0,
            correlation_at_optimal_lag=0.0,
            passes_threshold=False,
            note="constant series — no variance",
        )

    # Step 2: correlations
    spearman_rho, spearman_p = stats.spearmanr(sar_vals, ref_vals)
    pearson_r, pearson_p = stats.pearsonr(sar_vals, ref_vals)

    # Step 3: RMSE and MAE on z-scored values
    sar_z = (sar_vals - np.mean(sar_vals)) / np.std(sar_vals)
    ref_z = (ref_vals - np.mean(ref_vals)) / np.std(ref_vals)
    rmse = float(np.sqrt(np.mean((sar_z - ref_z) ** 2)))
    mae = float(np.mean(np.abs(sar_z - ref_z)))

    # Step 4: directional accuracy
    if n > 1:
        sar_diff = np.diff(sar_vals)
        ref_diff = np.diff(ref_vals)
        # Compare signs (both positive, both negative, or both zero)
        dir_match = np.sign(sar_diff) == np.sign(ref_diff)
        directional_accuracy = float(np.mean(dir_match))
    else:
        directional_accuracy = 0.0

    # Step 5: find optimal lag via cross-correlation
    optimal_lag, corr_at_lag = _find_optimal_lag(sar_vals, ref_vals, max_lag)

    # Step 6: pass/fail
    passes = abs(float(spearman_rho)) >= min_rho and float(spearman_p) < 0.05

    result = ValidationResult(
        port_id=port_id,
        signal_name=signal_name,
        reference_name=reference_name,
        reference_source=reference_source,
        n_observations=n,
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        rmse=rmse,
        mae=mae,
        directional_accuracy=directional_accuracy,
        optimal_lag_weeks=optimal_lag,
        correlation_at_optimal_lag=corr_at_lag,
        passes_threshold=passes,
    )

    logger.info("Validation: %s", result.summary_line())
    return result


def _find_optimal_lag(
    sar_vals: np.ndarray,
    ref_vals: np.ndarray,
    max_lag: int,
) -> tuple[int, float]:
    """Find the lag (in periods) that maximizes Spearman correlation.

    Positive lag = SAR leads reference (SAR is predictive).
    """
    n = len(sar_vals)
    best_lag = 0
    best_corr = 0.0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x = sar_vals[: n - lag]
            y = ref_vals[lag:]
        elif lag < 0:
            x = sar_vals[-lag:]
            y = ref_vals[: n + lag]
        else:
            x = sar_vals
            y = ref_vals

        if len(x) < 10:
            continue

        r, _ = stats.spearmanr(x, y)
        if abs(r) > abs(best_corr):
            best_corr = float(r)
            best_lag = lag

    return best_lag, best_corr


def validate_improvement(
    old_result: ValidationResult,
    new_result: ValidationResult,
) -> dict:
    """Compare two ValidationResults.

    Returns: {delta_rho, delta_rmse, rho_improved: bool, improvement_pct: float}
    """
    delta_rho = new_result.spearman_rho - old_result.spearman_rho

    # For RMSE, handle NaN
    if np.isnan(old_result.rmse) or np.isnan(new_result.rmse):
        delta_rmse = float("nan")
    else:
        delta_rmse = new_result.rmse - old_result.rmse

    rho_improved = abs(new_result.spearman_rho) > abs(old_result.spearman_rho)

    # Improvement percentage (based on absolute rho)
    if abs(old_result.spearman_rho) > 0:
        improvement_pct = (
            (abs(new_result.spearman_rho) - abs(old_result.spearman_rho))
            / abs(old_result.spearman_rho)
            * 100.0
        )
    else:
        improvement_pct = float("inf") if abs(new_result.spearman_rho) > 0 else 0.0

    return {
        "delta_rho": round(delta_rho, 4),
        "delta_rmse": round(delta_rmse, 4) if not np.isnan(delta_rmse) else None,
        "rho_improved": rho_improved,
        "improvement_pct": round(improvement_pct, 2),
    }
