#!/usr/bin/env python3
"""Enhanced correlation analysis between Rotterdam SAR index and economic indicators.

Loads Rotterdam weekly SAR data and cached economic data, runs stationarity tests,
computes correlations with multiple-hypothesis correction, and finds optimal lags.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy statsmodels warnings during analysis
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Imports from the project (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from portvolume.analysis.enhanced_correlation import (
        test_stationarity,
        compute_correlations_with_correction,
        optimal_lag_selection,
    )
    HAVE_ENHANCED = True
except ImportError as exc:
    logger.error("Could not import enhanced_correlation: %s", exc)
    HAVE_ENHANCED = False


def _make_json_safe(obj):
    """Recursively convert numpy/pandas types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _make_json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# 1. Load Rotterdam weekly SAR index
# ---------------------------------------------------------------------------
def load_rotterdam_weekly() -> pd.DataFrame:
    """Load the Rotterdam weekly index; rebuild from scene_universe if too few rows."""
    weekly_path = PROJECT_ROOT / "data" / "indices" / "rotterdam_weekly.parquet"
    df = pd.read_parquet(weekly_path)
    print(f"\n{'='*70}")
    print("ROTTERDAM WEEKLY INDEX (from rotterdam_weekly.parquet)")
    print(f"{'='*70}")
    print(f"  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head(10).to_string(index=False))

    if len(df) >= 20:
        return df

    # ---- Too few rows - try to build from scene_universe ----
    print(f"\n  WARNING: Only {len(df)} rows - too few for meaningful analysis (need >=20).")
    print("  Attempting to build weekly series from scene_universe.parquet ...")

    su_path = PROJECT_ROOT / "data" / "backtest" / "scene_universe.parquet"
    if not su_path.exists():
        print(f"  scene_universe.parquet not found at {su_path}. Cannot rebuild.")
        return df

    su = pd.read_parquet(su_path)
    print(f"  scene_universe shape: {su.shape}, columns: {list(su.columns)}")

    # Identify a date column
    date_col = None
    for candidate in ("date", "datetime", "acquisition_date", "acquired", "timestamp", "sensing_time"):
        if candidate in su.columns:
            date_col = candidate
            break
    if date_col is None:
        # Try to parse from the scene_id filename which contains a date string
        if "scene_id" in su.columns:
            print("  No date column found; attempting to parse date from scene_id ...")
            # Typical scene_id: S1A_IW_GRDH_...._YYYYMMDDTHHMMSS_...
            dates = su["scene_id"].str.extract(r"(\d{8}T\d{6})")
            if dates[0].notna().sum() > 0:
                su["_parsed_date"] = pd.to_datetime(dates[0], format="%Y%m%dT%H%M%S", errors="coerce")
                date_col = "_parsed_date"

    if date_col is None:
        print("  Could not identify a date column in scene_universe. Cannot rebuild.")
        return df

    su[date_col] = pd.to_datetime(su[date_col], errors="coerce")
    su = su.dropna(subset=[date_col])
    su["week"] = su[date_col].dt.to_period("W").dt.to_timestamp()

    # Identify a value column to aggregate
    value_col = None
    for candidate in ("mean", "mean_db", "backscatter", "sigma0", "value", "sar_mean"):
        if candidate in su.columns:
            value_col = candidate
            break

    if value_col is not None:
        weekly_rebuilt = (
            su.groupby("week")
            .agg(mean=(value_col, "mean"), max=(value_col, "max"), n_scenes=(value_col, "count"))
            .reset_index()
        )
    else:
        # Fallback: just count scenes per week as an activity proxy
        print("  No SAR value column found; using scene count per week as proxy index.")
        weekly_rebuilt = (
            su.groupby("week")
            .size()
            .reset_index(name="n_scenes")
        )
        weekly_rebuilt["mean"] = weekly_rebuilt["n_scenes"].astype(float)
        weekly_rebuilt["max"] = weekly_rebuilt["mean"]

    print(f"  Rebuilt weekly series: {len(weekly_rebuilt)} rows")
    print(weekly_rebuilt.head(10).to_string(index=False))
    return weekly_rebuilt


# ---------------------------------------------------------------------------
# 2. Load economic data
# ---------------------------------------------------------------------------
def load_economic_data() -> pd.DataFrame:
    """Load and merge FRED + yfinance cached data, resample to weekly."""
    fred_path = PROJECT_ROOT / "data" / "economic" / "fred_data.parquet"
    yf_path = PROJECT_ROOT / "data" / "economic" / "yfinance_data.parquet"

    frames = []
    for path, label in [(fred_path, "FRED"), (yf_path, "yfinance")]:
        if not path.exists():
            print(f"  {label} data not found at {path}")
            continue
        tmp = pd.read_parquet(path)
        print(f"  {label} data: shape={tmp.shape}, columns={list(tmp.columns)}")
        # Ensure index is datetime
        if not isinstance(tmp.index, pd.DatetimeIndex):
            for c in ("date", "week", "Date"):
                if c in tmp.columns:
                    tmp = tmp.set_index(c)
                    break
        if not isinstance(tmp.index, pd.DatetimeIndex):
            tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    # Drop fully-NaN columns
    combined = combined.dropna(axis=1, how="all")
    # Resample to weekly (Friday close) using last valid observation
    weekly = combined.resample("W").last()
    weekly = weekly.ffill(limit=2)  # forward fill up to 2 weeks for sparse monthly series
    print(f"\n  Combined economic data (weekly): {weekly.shape}")
    print(f"  Date range: {weekly.index.min()} to {weekly.index.max()}")
    print(f"  Columns: {list(weekly.columns)}")
    return weekly


# ---------------------------------------------------------------------------
# 3. Merge SAR + economic data
# ---------------------------------------------------------------------------
def merge_sar_economic(sar_df: pd.DataFrame, econ_df: pd.DataFrame) -> pd.DataFrame:
    """Align SAR weekly index with economic weekly data."""
    sar = sar_df.copy()
    if "week" in sar.columns:
        sar["week"] = pd.to_datetime(sar["week"], errors="coerce")
    else:
        print("  No 'week' column in SAR data!")
        return pd.DataFrame()

    # Snap SAR weeks to nearest Sunday (W frequency aligns to Sunday by default)
    sar["week_key"] = sar["week"].dt.to_period("W").dt.to_timestamp()
    sar = sar.set_index("week_key")

    econ = econ_df.copy()
    econ.index = econ.index.to_period("W").to_timestamp()

    merged = sar.join(econ, how="inner")
    print(f"\n  Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" ENHANCED CORRELATION ANALYSIS: Rotterdam SAR vs Economic Indicators")
    print("=" * 70)

    if not HAVE_ENHANCED:
        print("\nERROR: Could not import analysis functions. Aborting.")
        sys.exit(1)

    # ---- Load data ----
    sar_df = load_rotterdam_weekly()
    print(f"\n{'='*70}")
    print("ECONOMIC DATA")
    print(f"{'='*70}")
    econ_df = load_economic_data()

    if econ_df.empty:
        print("\nERROR: No economic data loaded. Aborting.")
        sys.exit(1)

    # ---- Merge ----
    merged = merge_sar_economic(sar_df, econ_df)
    if merged.empty or len(merged) < 5:
        print(f"\nERROR: Merged dataset has only {len(merged)} rows. Cannot proceed.")
        sys.exit(1)

    sar_col = "mean"
    econ_cols = [
        c for c in merged.columns
        if c not in ("week", "mean", "max", "n_scenes", "site_id", "site_type")
    ]
    print(f"  SAR column: '{sar_col}'")
    print(f"  Economic columns ({len(econ_cols)}): {econ_cols}")

    results: dict = {
        "n_observations": int(len(merged)),
        "date_range": [str(merged.index.min()), str(merged.index.max())],
        "sar_column": sar_col,
        "economic_columns": econ_cols,
    }

    # ---- (c) Stationarity test on SAR index ----
    print(f"\n{'='*70}")
    print("STATIONARITY TESTS (SAR 'mean' column)")
    print(f"{'='*70}")
    sar_series = merged[sar_col].dropna()
    stationarity = test_stationarity(sar_series)
    results["stationarity_sar"] = stationarity

    for key, val in stationarity.items():
        print(f"  {key:25s}: {val}")

    if stationarity.get("recommendation") == "insufficient_data":
        print("\n  NOTE: Insufficient data for reliable stationarity testing.")

    # ---- (d) Correlations with BH correction ----
    print(f"\n{'='*70}")
    print("CORRELATIONS WITH BENJAMINI-HOCHBERG CORRECTION")
    print(f"{'='*70}")

    # Only use columns that have enough non-NaN overlap with SAR
    valid_econ_cols = []
    for c in econ_cols:
        overlap = merged[[sar_col, c]].dropna().shape[0]
        if overlap >= 10:
            valid_econ_cols.append(c)
        else:
            print(f"  Skipping '{c}': only {overlap} overlapping observations")

    if valid_econ_cols:
        corr_df = compute_correlations_with_correction(
            merged, sar_col, valid_econ_cols, method="benjamini_hochberg"
        )
        if not corr_df.empty:
            corr_df = corr_df.sort_values("rho", key=abs, ascending=False).reset_index(drop=True)
            print(corr_df.to_string(index=False))
            results["correlations"] = corr_df.to_dict(orient="records")
        else:
            print("  No valid correlations computed (insufficient overlapping data).")
            results["correlations"] = []
    else:
        print("  No economic columns with >= 10 overlapping observations.")
        corr_df = pd.DataFrame()
        results["correlations"] = []

    # ---- (e) Optimal lag for top-3 correlations ----
    print(f"\n{'='*70}")
    print("OPTIMAL LAG SELECTION (top 3 by |rho|)")
    print(f"{'='*70}")

    results["optimal_lags"] = {}

    if not corr_df.empty and len(corr_df) > 0:
        top3 = corr_df.head(3)
        for _, row in top3.iterrows():
            indicator = row["indicator"]
            print(f"\n  --- {indicator} (rho={row['rho']:.4f}) ---")
            sar_s = merged[sar_col].dropna()
            econ_s = merged[indicator].dropna()
            # Align indices
            common_idx = sar_s.index.intersection(econ_s.index)
            sar_s = sar_s.loc[common_idx]
            econ_s = econ_s.loc[common_idx]

            if len(common_idx) < 12:
                print(f"  Only {len(common_idx)} observations - too few for lag selection.")
                results["optimal_lags"][indicator] = {"error": "insufficient_data"}
                continue

            max_lag = min(16, len(common_idx) // 3 - 1)
            if max_lag < 1:
                max_lag = 1

            lag_result = optimal_lag_selection(sar_s, econ_s, max_lag=max_lag)
            results["optimal_lags"][indicator] = lag_result
            print(f"  Optimal lag  : {lag_result['optimal_lag']} weeks")
            print(f"  BIC value    : {lag_result['criterion_value']:.2f}" if np.isfinite(lag_result["criterion_value"]) else f"  BIC value    : inf")
            if lag_result["all_lags"]:
                print("  All lags tested:")
                for entry in lag_result["all_lags"]:
                    xcorr = entry.get("cross_correlation", float("nan"))
                    xcorr_str = f"{xcorr:.4f}" if np.isfinite(xcorr) else "N/A"
                    bic_str = f"{entry['criterion_value']:.2f}" if np.isfinite(entry["criterion_value"]) else "inf"
                    print(f"    lag={entry['lag']:2d}  BIC={bic_str:>10s}  xcorr={xcorr_str}")
    else:
        print("  No correlations available to select top indicators.")

    # ---- (f) Save results ----
    output_dir = PROJECT_ROOT / "data" / "backtest" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "enhanced_analysis_results.json"

    safe_results = _make_json_safe(results)
    with open(output_path, "w") as f:
        json.dump(safe_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {output_path}")
    print(f"{'='*70}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Observations       : {results['n_observations']}")
    print(f"  Date range         : {results['date_range'][0]} to {results['date_range'][1]}")
    print(f"  SAR stationary?    : {stationarity.get('is_stationary', 'N/A')}")
    print(f"  Recommendation     : {stationarity.get('recommendation', 'N/A')}")
    n_sig = sum(1 for r in (results.get('correlations') or []) if r.get('significant'))
    n_total = len(results.get('correlations') or [])
    print(f"  Significant corrs  : {n_sig} / {n_total}")
    if results.get("optimal_lags"):
        for ind, lag_info in results["optimal_lags"].items():
            if isinstance(lag_info, dict) and "optimal_lag" in lag_info:
                print(f"  Optimal lag ({ind}): {lag_info['optimal_lag']} weeks")

    print("\nDone.")


if __name__ == "__main__":
    main()
