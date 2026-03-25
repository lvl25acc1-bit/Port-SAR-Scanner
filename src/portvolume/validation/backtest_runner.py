"""Backtest runner: multi-port, multi-signal validation orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from portvolume.validation.metrics import ValidationResult, validate_signal
from portvolume.validation.reference_data import ReferenceDataset

logger = logging.getLogger(__name__)


def run_full_validation(
    signals: dict[str, dict[str, pd.DataFrame]],
    references: dict[str, list[ReferenceDataset]],
    min_rho: float = 0.3,
    max_lag: int = 12,
) -> list[ValidationResult]:
    """Run validation across all ports, all signals, all references.

    Args:
        signals: {port_id: {signal_name: series_df}}
            Each series_df must have a numeric column (first non-date column used).
        references: {port_id: [ReferenceDataset, ...]}

    Returns list of ValidationResult for every (port, signal, reference) combination.
    """
    results: list[ValidationResult] = []

    for port_id, port_signals in signals.items():
        port_refs = references.get(port_id, [])
        if not port_refs:
            logger.warning("No reference data for port %s — skipping", port_id)
            continue

        for signal_name, signal_df in port_signals.items():
            # Extract the signal series (assume 'value' column or first numeric column)
            sar_series = _extract_series(signal_df)
            if sar_series is None:
                logger.warning(
                    "Could not extract series from signal %s at %s", signal_name, port_id
                )
                continue

            for ref in port_refs:
                # Extract reference series
                ref_series = _extract_series(ref.data, value_col="value")
                if ref_series is None:
                    continue

                result = validate_signal(
                    sar_series=sar_series,
                    reference_series=ref_series,
                    signal_name=signal_name,
                    reference_name=ref.name,
                    reference_source=ref.source,
                    port_id=port_id,
                    min_rho=min_rho,
                    max_lag=max_lag,
                )
                results.append(result)

    logger.info(
        "Full validation complete: %d results (%d passing)",
        len(results),
        sum(1 for r in results if r.passes_threshold),
    )
    return results


def _extract_series(
    df: pd.DataFrame,
    value_col: str | None = None,
) -> pd.Series | None:
    """Extract a numeric series from a DataFrame."""
    if df is None or df.empty:
        return None

    if value_col and value_col in df.columns:
        return df[value_col].reset_index(drop=True)

    # Fall back to first numeric column
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    return df[numeric_cols[0]].reset_index(drop=True)


def compare_pivot_vs_baseline(
    baseline_results: list[ValidationResult],
    pivot_results: list[ValidationResult],
) -> dict:
    """Compare pivot results against baseline.

    Returns:
    {
        "n_baseline_passing": int,
        "n_pivot_passing": int,
        "best_baseline": {port_id, signal, rho},
        "best_pivot": {port_id, signal, rho},
        "mean_rho_improvement": float,
        "verdict": str  # "improved", "no_change", "degraded"
    }
    """
    n_baseline_pass = sum(1 for r in baseline_results if r.passes_threshold)
    n_pivot_pass = sum(1 for r in pivot_results if r.passes_threshold)

    def _best(results: list[ValidationResult]) -> dict:
        if not results:
            return {"port_id": "", "signal": "", "rho": 0.0}
        best = max(results, key=lambda r: abs(r.spearman_rho))
        return {
            "port_id": best.port_id,
            "signal": best.signal_name,
            "rho": round(best.spearman_rho, 4),
        }

    # Compute mean rho improvement for matched pairs
    baseline_map = {
        (r.port_id, r.signal_name, r.reference_name): r for r in baseline_results
    }
    deltas = []
    for r in pivot_results:
        key = (r.port_id, r.signal_name, r.reference_name)
        if key in baseline_map:
            deltas.append(
                abs(r.spearman_rho) - abs(baseline_map[key].spearman_rho)
            )

    mean_improvement = float(sum(deltas) / len(deltas)) if deltas else 0.0

    # Verdict
    if mean_improvement > 0.02:
        verdict = "improved"
    elif mean_improvement < -0.02:
        verdict = "degraded"
    else:
        verdict = "no_change"

    return {
        "n_baseline_passing": n_baseline_pass,
        "n_pivot_passing": n_pivot_pass,
        "best_baseline": _best(baseline_results),
        "best_pivot": _best(pivot_results),
        "mean_rho_improvement": round(mean_improvement, 4),
        "verdict": verdict,
    }


def load_rotterdam_baseline(backtest_outputs_dir: Path) -> list[ValidationResult]:
    """Load existing Rotterdam backtest results as ValidationResult objects.

    Reads from scene_diagnostic.json and monthly_validation.json.
    Creates ValidationResult entries for the known baseline (rho=-0.068 etc.)
    """
    results: list[ValidationResult] = []
    outputs_dir = Path(backtest_outputs_dir)

    # Scene diagnostic
    scene_path = outputs_dir / "scene_diagnostic.json"
    if scene_path.exists():
        with open(scene_path) as f:
            scene_diag = json.load(f)

        for label, metrics in scene_diag.items():
            if not isinstance(metrics, dict) or "spearman_r" not in metrics:
                continue
            results.append(ValidationResult(
                port_id="rotterdam",
                signal_name=f"scene_{label}",
                reference_name="gfw_ais",
                reference_source="gfw_ais",
                n_observations=metrics.get("n_scenes", 0),
                spearman_rho=metrics["spearman_r"],
                spearman_p=metrics.get("spearman_p", 1.0),
                pearson_r=metrics.get("pearson_r", 0.0),
                pearson_p=metrics.get("pearson_p", 1.0),
                rmse=float("nan"),
                mae=metrics.get("mae", float("nan")),
                directional_accuracy=0.0,
                optimal_lag_weeks=0,
                correlation_at_optimal_lag=metrics["spearman_r"],
                passes_threshold=(
                    abs(metrics["spearman_r"]) >= 0.3
                    and metrics.get("spearman_p", 1.0) < 0.05
                ),
                note="loaded from scene_diagnostic.json baseline",
            ))

    # Monthly validation
    monthly_path = outputs_dir / "monthly_validation.json"
    if monthly_path.exists():
        with open(monthly_path) as f:
            monthly_val = json.load(f)

        for label, metrics in monthly_val.items():
            if not isinstance(metrics, dict) or "spearman_r" not in metrics:
                continue
            results.append(ValidationResult(
                port_id="rotterdam",
                signal_name=f"monthly_{label}",
                reference_name="gfw_ais",
                reference_source="gfw_ais",
                n_observations=metrics.get("n_months", 0),
                spearman_rho=metrics["spearman_r"],
                spearman_p=metrics.get("spearman_p", 1.0),
                pearson_r=metrics.get("pearson_r", 0.0),
                pearson_p=metrics.get("pearson_p", 1.0),
                rmse=float("nan"),
                mae=float("nan"),
                directional_accuracy=0.0,
                optimal_lag_weeks=0,
                correlation_at_optimal_lag=metrics["spearman_r"],
                passes_threshold=(
                    abs(metrics["spearman_r"]) >= 0.3
                    and metrics.get("spearman_p", 1.0) < 0.05
                ),
                note="loaded from monthly_validation.json baseline",
            ))

    logger.info("Loaded %d Rotterdam baseline results from %s", len(results), outputs_dir)
    return results
