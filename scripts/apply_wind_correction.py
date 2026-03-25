"""Apply wind correction to Rotterdam backtest scene data.

Fits a robust linear model to remove wind/pass/satellite confounds,
computes wind-adjusted vessel counts, and builds weekly time series.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project source is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portvolume.detection.wind_correction import fit_wind_model, compute_wind_adjusted_count

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
backtest_dir = PROJECT_ROOT / "data" / "backtest"
output_dir = backtest_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WIND CORRECTION — Rotterdam Backtest")
print("=" * 70)

scene_universe = pd.read_parquet(backtest_dir / "scene_universe.parquet")
wind = pd.read_parquet(backtest_dir / "wind.parquet")

print(f"\nLoaded scene_universe: {len(scene_universe)} scenes, columns: {scene_universe.columns.tolist()}")
print(f"Loaded wind: {len(wind)} rows, columns: {wind.columns.tolist()}")

# ---------------------------------------------------------------------------
# 2. Merge wind data into scene_universe on scene_midpoint_utc
# ---------------------------------------------------------------------------
# Ensure merge key is datetime with consistent timezone
scene_universe["scene_midpoint_utc"] = pd.to_datetime(
    scene_universe["scene_midpoint_utc"], utc=True
)
wind["scene_midpoint_utc"] = pd.to_datetime(
    wind["scene_midpoint_utc"], utc=True
)

# If wind_speed_mps already exists in scene_universe, drop it to avoid _x/_y
if "wind_speed_mps" in scene_universe.columns:
    scene_universe = scene_universe.drop(columns=["wind_speed_mps"])

merged = scene_universe.merge(wind, on="scene_midpoint_utc", how="left")
n_with_wind = merged["wind_speed_mps"].notna().sum()
print(f"\nAfter merge: {len(merged)} scenes, {n_with_wind} with wind data")

# ---------------------------------------------------------------------------
# 3. Fit wind model and compute wind-adjusted counts
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("FITTING WIND MODEL (RLM: vessel_count ~ wind_speed_mps + pass_family + satellite)")
print("-" * 70)

coefficients = fit_wind_model(merged, target_col="vessel_count")

print("\nModel Coefficients:")
for k, v in coefficients.items():
    print(f"  {k:>25s} = {v:>10.4f}")

# Apply wind correction to each scene
merged["wind_adjusted_count"] = merged.apply(
    lambda row: compute_wind_adjusted_count(
        raw_count=row["vessel_count"],
        wind_speed_mps=row["wind_speed_mps"],
        pass_family=row["pass_family"],
        satellite=row["satellite"],
        model_coefficients=coefficients,
    )
    if pd.notna(row["wind_speed_mps"])
    else row["vessel_count"],
    axis=1,
)

print(f"\nWind-adjusted counts computed for {n_with_wind} scenes")
print(f"  Raw vessel_count       — mean: {merged['vessel_count'].mean():.2f}, std: {merged['vessel_count'].std():.2f}")
print(f"  Wind-adjusted count    — mean: {merged['wind_adjusted_count'].mean():.2f}, std: {merged['wind_adjusted_count'].std():.2f}")

# ---------------------------------------------------------------------------
# 4. Build weekly time series
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("WEEKLY TIME SERIES")
print("-" * 70)

merged["date"] = pd.to_datetime(merged["date"])
weekly = (
    merged
    .set_index("date")
    .resample("W-MON")[["vessel_count", "wind_adjusted_count"]]
    .mean()
    .dropna()
    .reset_index()
)

print(f"\n{len(weekly)} weeks in time series")
print(weekly.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
scene_out = output_dir / "wind_adjusted_scenes.parquet"
weekly_out = output_dir / "wind_adjusted_weekly.parquet"

merged.to_parquet(scene_out, index=False)
weekly.to_parquet(weekly_out, index=False)

print(f"\nSaved scene-level results:  {scene_out}")
print(f"Saved weekly results:       {weekly_out}")

# ---------------------------------------------------------------------------
# 6. Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

valid = merged.dropna(subset=["wind_speed_mps", "vessel_count", "wind_adjusted_count"])

raw_var = valid["vessel_count"].var()
adj_var = valid["wind_adjusted_count"].var()
var_reduction = (1 - adj_var / raw_var) * 100

corr_raw_wind = valid["vessel_count"].corr(valid["wind_speed_mps"])
corr_adj_wind = valid["wind_adjusted_count"].corr(valid["wind_speed_mps"])

print(f"\n  Raw vessel_count variance:          {raw_var:.2f}")
print(f"  Wind-adjusted count variance:       {adj_var:.2f}")
print(f"  Variance reduction:                 {var_reduction:.1f}%")
print(f"")
print(f"  Correlation(raw, wind_speed):        {corr_raw_wind:.4f}")
print(f"  Correlation(adjusted, wind_speed):   {corr_adj_wind:.4f}")
print(f"")
print(f"  R-squared (model):                   {coefficients.get('r_squared', float('nan')):.4f}")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
