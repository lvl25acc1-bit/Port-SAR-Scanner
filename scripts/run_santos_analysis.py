"""Santos soy-season analysis: zone-based vessel detection vs soybean futures."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portvolume.zones.zone_config import load_zone_config
from portvolume.zones.zone_detector import assign_detections_to_zones, compute_zone_metrics
from portvolume.analysis.granger import run_granger_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SANTOS_DETECTIONS_DIR = ROOT / "data" / "detections" / "santos"
SANTOS_ZONES_PATH = ROOT / "config" / "zones" / "santos.geojson"
SUMMARIES_CSV = SANTOS_DETECTIONS_DIR / "summaries.csv"
ECONOMIC_DATA = ROOT / "data" / "economic" / "yfinance_data.parquet"
OUTPUT_JSON = SANTOS_DETECTIONS_DIR / "analysis_results.json"
MAX_SOY_GAP_DAYS = 7
MIN_CORR_OBS = 10
LOW_VARIATION_TOP_SHARE = 0.80
LOW_VARIATION_MIN_UNIQUE = 3
GRANGER_MAX_LAG = 4
OUT_OF_TIME_TRAIN_END = pd.Timestamp("2024-12-31")
OUT_OF_TIME_TEST_START = pd.Timestamp("2025-01-01")
HARVEST_MONTHS = (2, 3, 4, 5)


def load_scene_parquets() -> list[tuple[str, str, gpd.GeoDataFrame]]:
    """Load all Santos detection parquet files.

    Returns list of (scene_id, timestamp_str, GeoDataFrame).
    """
    summaries = pd.read_csv(SUMMARIES_CSV)
    scenes = []
    for _, row in summaries.iterrows():
        scene_id = row["scene_id"]
        timestamp = row["timestamp"]
        parquet_path = SANTOS_DETECTIONS_DIR / f"{scene_id}.parquet"
        if parquet_path.exists():
            gdf = gpd.read_parquet(parquet_path)
            scenes.append((scene_id, timestamp, gdf))
        else:
            print(f"  WARNING: Missing parquet for {scene_id}")
    return scenes


def align_series_nearest(
    series: pd.Series,
    target_dates: pd.Series | pd.DatetimeIndex,
    max_gap_days: int,
) -> pd.DataFrame:
    """Align a dated series to target timestamps using nearest match with a max gap."""
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    values = series.astype(float).to_numpy()

    if len(idx) == 0:
        return pd.DataFrame(
            {
                "matched_date": [pd.NaT] * len(target_dates),
                "value": [np.nan] * len(target_dates),
                "gap_days": [np.nan] * len(target_dates),
                "is_valid": [False] * len(target_dates),
            }
        )

    aligned = []
    for target_date in pd.to_datetime(target_dates):
        diffs = np.abs(idx - target_date)
        nearest_idx = int(diffs.argmin())
        gap_days = float(diffs[nearest_idx] / np.timedelta64(1, "D"))
        is_valid = gap_days <= max_gap_days
        aligned.append(
            {
                "matched_date": idx[nearest_idx] if is_valid else pd.NaT,
                "value": float(values[nearest_idx]) if is_valid else np.nan,
                "gap_days": gap_days,
                "is_valid": is_valid,
            }
        )

    return pd.DataFrame(aligned)


def series_diagnostics(series: pd.Series) -> dict:
    """Summarize whether a series has enough variation for inference."""
    clean = pd.Series(series).dropna()
    if clean.empty:
        return {
            "n_obs": 0,
            "n_unique": 0,
            "std": None,
            "top_share": None,
            "informative": False,
            "note": "no overlapping observations",
        }

    value_share = clean.value_counts(normalize=True, dropna=False)
    top_share = float(value_share.iloc[0])
    n_unique = int(clean.nunique())
    std = float(clean.std()) if len(clean) > 1 else 0.0

    notes: list[str] = []
    if n_unique < LOW_VARIATION_MIN_UNIQUE:
        notes.append(f"only {n_unique} unique values")
    if std == 0:
        notes.append("constant series")
    if top_share >= LOW_VARIATION_TOP_SHARE:
        notes.append(f"top value share {top_share:.1%}")

    return {
        "n_obs": int(len(clean)),
        "n_unique": n_unique,
        "std": std,
        "top_share": top_share,
        "informative": not notes,
        "note": "; ".join(notes) if notes else None,
    }


def safe_correlation_summary(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str = "soy_price",
) -> dict:
    """Compute level and differenced correlations with basic robustness diagnostics."""
    mask = df[metric_col].notna() & df[target_col].notna()
    aligned = df.loc[mask, [metric_col, target_col]].astype(float)

    summary: dict = {
        "n_obs": int(len(aligned)),
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "differenced_pearson_r": None,
        "differenced_pearson_p": None,
        "differenced_spearman_rho": None,
        "differenced_spearman_p": None,
        "valid_for_inference": False,
        "metric_diagnostics": series_diagnostics(aligned[metric_col]),
        "target_diagnostics": series_diagnostics(aligned[target_col]),
        "note": None,
    }

    notes: list[str] = []
    if len(aligned) < MIN_CORR_OBS:
        notes.append(f"need at least {MIN_CORR_OBS} overlapping observations")
    if not summary["metric_diagnostics"]["informative"]:
        notes.append(f"{metric_col}: {summary['metric_diagnostics']['note']}")
    if not summary["target_diagnostics"]["informative"]:
        notes.append(f"{target_col}: {summary['target_diagnostics']['note']}")

    if not notes:
        pr, pp = stats.pearsonr(aligned[metric_col], aligned[target_col])
        sr, sp = stats.spearmanr(aligned[metric_col], aligned[target_col])
        summary.update(
            {
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
                "valid_for_inference": True,
            }
        )

    diff_df = aligned.diff().dropna()
    diff_metric_diag = series_diagnostics(diff_df[metric_col])
    diff_target_diag = series_diagnostics(diff_df[target_col])
    if (
        len(diff_df) >= MIN_CORR_OBS
        and diff_metric_diag["informative"]
        and diff_target_diag["informative"]
    ):
        dpr, dpp = stats.pearsonr(diff_df[metric_col], diff_df[target_col])
        dsr, dsp = stats.spearmanr(diff_df[metric_col], diff_df[target_col])
        summary.update(
            {
                "differenced_pearson_r": float(dpr),
                "differenced_pearson_p": float(dpp),
                "differenced_spearman_rho": float(dsr),
                "differenced_spearman_p": float(dsp),
            }
        )
    else:
        diff_notes: list[str] = []
        if len(diff_df) < MIN_CORR_OBS:
            diff_notes.append("insufficient differenced observations")
        if not diff_metric_diag["informative"]:
            diff_notes.append(f"d{metric_col}: {diff_metric_diag['note']}")
        if not diff_target_diag["informative"]:
            diff_notes.append(f"d{target_col}: {diff_target_diag['note']}")
        if diff_notes:
            notes.append(", ".join(diff_notes))

    if notes:
        summary["note"] = " | ".join(notes)

    return summary


def yearly_spearman_summary(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str = "soy_price",
) -> dict[str, dict]:
    """Compute within-year correlations on the overlapping window."""
    records: dict[str, dict] = {}
    if "date" not in df.columns:
        return records

    for year, subset in df.groupby(df["date"].dt.year):
        mask = subset[metric_col].notna() & subset[target_col].notna()
        aligned = subset.loc[mask, [metric_col, target_col]]
        if len(aligned) < MIN_CORR_OBS:
            continue

        metric_diag = series_diagnostics(aligned[metric_col])
        target_diag = series_diagnostics(aligned[target_col])
        if not metric_diag["informative"] or not target_diag["informative"]:
            records[str(year)] = {
                "n_obs": int(len(aligned)),
                "spearman_rho": None,
                "spearman_p": None,
                "note": "; ".join(
                    note
                    for note in (metric_diag["note"], target_diag["note"])
                    if note
                ) or "insufficient variation",
            }
            continue

        rho, p_value = stats.spearmanr(aligned[metric_col], aligned[target_col])
        records[str(year)] = {
            "n_obs": int(len(aligned)),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "note": None,
        }

    return records


def _safe_pairwise_correlations(metric: pd.Series, target: pd.Series) -> dict:
    metric_diag = series_diagnostics(metric)
    target_diag = series_diagnostics(target)
    if not metric_diag["informative"] or not target_diag["informative"]:
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "note": "; ".join(
                note for note in (metric_diag["note"], target_diag["note"]) if note
            ) or "insufficient variation",
        }

    pearson_r, pearson_p = stats.pearsonr(metric, target)
    spearman_rho, spearman_p = stats.spearmanr(metric, target)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "note": None,
    }


def detrended_correlation_summary(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str = "soy_price",
    months: tuple[int, ...] | None = None,
) -> dict:
    """Correlate residuals after removing a linear time trend from both series."""
    subset = df.loc[df[metric_col].notna() & df[target_col].notna(), ["date", metric_col, target_col]].copy()
    if months is not None:
        subset = subset[subset["date"].dt.month.isin(months)].copy()

    summary = {
        "n_obs": int(len(subset)),
        "months_filter": list(months) if months is not None else None,
        "metric_trend_per_year": None,
        "target_trend_per_year": None,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "note": None,
    }
    if len(subset) < MIN_CORR_OBS:
        summary["note"] = f"need >= {MIN_CORR_OBS} observations"
        return summary

    ordinal_dates = subset["date"].map(pd.Timestamp.toordinal).astype(float).to_numpy()
    metric_values = subset[metric_col].astype(float).to_numpy()
    target_values = subset[target_col].astype(float).to_numpy()

    metric_slope, metric_intercept, _, _, _ = stats.linregress(ordinal_dates, metric_values)
    target_slope, target_intercept, _, _, _ = stats.linregress(ordinal_dates, target_values)
    metric_resid = metric_values - (metric_intercept + metric_slope * ordinal_dates)
    target_resid = target_values - (target_intercept + target_slope * ordinal_dates)

    corr = _safe_pairwise_correlations(pd.Series(metric_resid), pd.Series(target_resid))
    summary.update(corr)
    summary["metric_trend_per_year"] = float(metric_slope * 365.25)
    summary["target_trend_per_year"] = float(target_slope * 365.25)
    return summary


def monthly_differenced_granger_summary(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str = "soy_price",
    max_lag: int = GRANGER_MAX_LAG,
) -> dict:
    """Run Granger causality on differenced monthly averages."""
    subset = df.loc[df[metric_col].notna() & df[target_col].notna(), ["date", metric_col, target_col]].copy()
    monthly = (
        subset.assign(month=subset["date"].dt.to_period("M"))
        .groupby("month")
        .agg(metric=(metric_col, "mean"), target=(target_col, "mean"))
    )
    monthly.index = monthly.index.to_timestamp()
    diff_monthly = monthly.diff().dropna()

    summary = {
        "frequency": "monthly",
        "n_months": int(len(monthly)),
        "n_differenced_obs": int(len(diff_monthly)),
        "max_lag": int(max_lag),
        "sar_causes_target": False,
        "target_causes_sar": False,
        "direction": "none",
        "optimal_lag": 0,
        "p_value": None,
        "note": None,
    }
    if len(diff_monthly) < max_lag * 3:
        summary["note"] = f"need >= {max_lag * 3} differenced monthly observations"
        return summary

    granger_result = run_granger_test(
        diff_monthly["metric"],
        diff_monthly["target"],
        max_lag=max_lag,
    )
    summary.update(
        {
            "sar_causes_target": bool(granger_result.sar_causes_econ),
            "target_causes_sar": bool(granger_result.econ_causes_sar),
            "direction": granger_result.best_direction,
            "optimal_lag": int(granger_result.optimal_lag),
            "p_value": float(granger_result.p_value),
        }
    )
    return summary


def out_of_time_regression_summary(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str = "soy_price",
    train_end: pd.Timestamp = OUT_OF_TIME_TRAIN_END,
    test_start: pd.Timestamp = OUT_OF_TIME_TEST_START,
) -> dict:
    """Fit a level regression on 2023-2024 and score it on 2025-2026."""
    subset = df.loc[df[metric_col].notna() & df[target_col].notna(), ["date", metric_col, target_col]].copy()
    train = subset[subset["date"] <= train_end].copy()
    test = subset[subset["date"] >= test_start].copy()

    summary = {
        "train_window_end": train_end.strftime("%Y-%m-%d"),
        "test_window_start": test_start.strftime("%Y-%m-%d"),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "train_slope": None,
        "train_pearson_r": None,
        "train_pearson_p": None,
        "test_r2": None,
        "test_mae": None,
        "test_pearson_r": None,
        "test_pearson_p": None,
        "test_spearman_rho": None,
        "test_spearman_p": None,
        "note": None,
    }
    if len(train) < MIN_CORR_OBS or len(test) < MIN_CORR_OBS:
        summary["note"] = f"need >= {MIN_CORR_OBS} train and test observations"
        return summary

    slope, intercept, train_r, train_p, _ = stats.linregress(
        train[metric_col].astype(float),
        train[target_col].astype(float),
    )
    predicted = intercept + slope * test[metric_col].astype(float).to_numpy()
    actual = test[target_col].astype(float).to_numpy()
    ss_res = float(((actual - predicted) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    mae = float(np.abs(actual - predicted).mean())

    test_corr = _safe_pairwise_correlations(pd.Series(predicted), pd.Series(actual))
    summary.update(
        {
            "train_slope": float(slope),
            "train_pearson_r": float(train_r),
            "train_pearson_p": float(train_p),
            "test_r2": float(1 - ss_res / ss_tot) if ss_tot else None,
            "test_mae": mae,
            "test_pearson_r": test_corr["pearson_r"],
            "test_pearson_p": test_corr["pearson_p"],
            "test_spearman_rho": test_corr["spearman_rho"],
            "test_spearman_p": test_corr["spearman_p"],
            "note": test_corr["note"],
        }
    )
    return summary


def main() -> None:
    print("=" * 70)
    print("SANTOS SOY-SEASON ANALYSIS: SAR Vessel Zones vs Soybean Futures")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load zone config
    # ------------------------------------------------------------------
    print("\n[1] Loading Santos zone configuration...")
    zone_config = load_zone_config(SANTOS_ZONES_PATH)
    print(f"    Loaded {len(zone_config.zones)} zones:")
    for z in zone_config.zones:
        print(f"      - {z.zone_id} ({z.zone_type}), capacity={z.capacity_vessels}")

    # ------------------------------------------------------------------
    # Step 2: Load detection parquets and assign zones
    # ------------------------------------------------------------------
    print("\n[2] Loading detection parquets and assigning zones...")
    scenes = load_scene_parquets()
    print(f"    Loaded {len(scenes)} scenes")

    # Inspect first scene structure
    if scenes:
        _, _, first_gdf = scenes[0]
        print(f"    First scene columns: {list(first_gdf.columns)}")
        print(f"    CRS: {first_gdf.crs}")
        print(f"    Geometry types: {first_gdf.geometry.geom_type.unique().tolist()}")
        print(f"    Bounds: {first_gdf.total_bounds}")

    metrics_list = []
    for scene_id, timestamp, gdf in scenes:
        # Assign detections to zones (CRS handled inside assign_detections_to_zones)
        zoned = assign_detections_to_zones(gdf, zone_config)
        m = compute_zone_metrics(zoned, zone_config, scene_id, timestamp)
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df["date"] = pd.to_datetime(metrics_df["timestamp"]).dt.normalize()
    metrics_df["month"] = pd.to_datetime(metrics_df["timestamp"]).dt.month
    metrics_df["month_name"] = pd.to_datetime(metrics_df["timestamp"]).dt.strftime("%Y-%m")

    print("\n    Scene-level zone metrics:")
    print("    " + "-" * 95)
    print(f"    {'Date':<12} {'Total':>6} {'Anch':>6} {'Berth':>6} {'Chan':>6} {'Outside':>8} {'Congestion':>11}")
    print("    " + "-" * 95)
    for _, row in metrics_df.iterrows():
        print(
            f"    {row['date'].strftime('%Y-%m-%d'):<12} {row['total_count']:>6} {row['anchorage_count']:>6} "
            f"{row['berth_count']:>6} {row['channel_count']:>6} {row['outside_count']:>8} "
            f"{row['congestion_index']:>11.3f}"
        )

    # ------------------------------------------------------------------
    # Step 3: Load soybean futures
    # ------------------------------------------------------------------
    print("\n[3] Loading soybean futures (ZS=F) data...")
    econ_df = pd.read_parquet(ECONOMIC_DATA)
    print(f"    Economic data columns: {list(econ_df.columns)}")
    print(f"    Index: {econ_df.index.name}, dtype: {econ_df.index.dtype}")
    print(f"    Shape: {econ_df.shape}")

    if "ZS=F" not in econ_df.columns:
        print("    ERROR: ZS=F column not found in economic data!")
        return

    soy_prices = econ_df["ZS=F"].dropna()
    print(f"    ZS=F: {len(soy_prices)} observations, range [{soy_prices.min():.2f}, {soy_prices.max():.2f}]")

    # ------------------------------------------------------------------
    # Step 4: Align scene dates with soy prices (nearest date)
    # ------------------------------------------------------------------
    print(f"\n[4] Aligning scene dates with soy price (nearest date, max gap {MAX_SOY_GAP_DAYS} days)...")
    scene_dates = pd.to_datetime(metrics_df["timestamp"])
    aligned_prices = align_series_nearest(soy_prices, scene_dates, max_gap_days=MAX_SOY_GAP_DAYS)

    metrics_df["soy_price"] = aligned_prices["value"]
    metrics_df["soy_date"] = aligned_prices["matched_date"]
    metrics_df["soy_gap_days"] = aligned_prices["gap_days"]
    metrics_df["soy_price_valid"] = aligned_prices["is_valid"]

    for row in metrics_df.itertuples(index=False):
        if row.soy_price_valid:
            print(
                "    "
                f"Scene {row.date.strftime('%Y-%m-%d')} -> Soy {row.soy_date.strftime('%Y-%m-%d')}: "
                f"${row.soy_price:.2f} (gap={row.soy_gap_days:.1f}d)"
            )
        else:
            print(
                "    "
                f"Scene {row.date.strftime('%Y-%m-%d')} -> no soy match within {MAX_SOY_GAP_DAYS} days "
                f"(nearest gap={row.soy_gap_days:.1f}d)"
            )

    valid_soy_df = metrics_df[metrics_df["soy_price_valid"]].copy()
    print(
        f"    Valid soy-aligned scenes: {len(valid_soy_df)} / {len(metrics_df)} "
        f"({metrics_df['soy_price_valid'].mean():.1%})"
    )
    if not valid_soy_df.empty:
        print(
            "    Overlap window: "
            f"{valid_soy_df['date'].min().strftime('%Y-%m-%d')} to "
            f"{valid_soy_df['date'].max().strftime('%Y-%m-%d')}"
        )

    # ------------------------------------------------------------------
    # Step 5: Correlation analysis
    # ------------------------------------------------------------------
    print(f"\n[5] Correlation analysis on valid soy overlap (n={len(valid_soy_df)} scenes)...")
    print("    " + "-" * 75)

    corr_results = {}
    for metric_name, metric_col in [
        ("total_count", "total_count"),
        ("anchorage_count", "anchorage_count"),
        ("berth_count", "berth_count"),
        ("congestion_index", "congestion_index"),
        ("queue_length_proxy", "queue_length_proxy"),
    ]:
        result = safe_correlation_summary(valid_soy_df, metric_col)
        corr_results[metric_name] = result

        if result["valid_for_inference"]:
            sig_marker = " *" if result["spearman_p"] is not None and result["spearman_p"] < 0.10 else ""
            diff_text = ""
            if result["differenced_spearman_rho"] is not None:
                diff_text = (
                    f" | d-rho={result['differenced_spearman_rho']:+.3f} "
                    f"(p={result['differenced_spearman_p']:.3f})"
                )
            print(
                f"    {metric_name:<22} Pearson r={result['pearson_r']:+.3f} "
                f"(p={result['pearson_p']:.3f})  Spearman rho={result['spearman_rho']:+.3f} "
                f"(p={result['spearman_p']:.3f}){sig_marker}{diff_text}"
            )
        else:
            print(
                f"    {metric_name:<22} skipped for inference"
                f" ({result['note'] or 'insufficient variation'})"
            )

    print("    " + "-" * 75)
    print("    (* = Spearman p < 0.10)")

    # ------------------------------------------------------------------
    # Step 6: Seasonal pattern
    # ------------------------------------------------------------------
    print("\n[6] Seasonal pattern: monthly averages...")
    print("    " + "-" * 80)
    print(f"    {'Month':<10} {'Scenes':>7} {'Avg Total':>10} {'Avg Anch':>9} {'Avg Berth':>10} "
          f"{'Avg Congest':>12} {'Avg Soy $':>10}")
    print("    " + "-" * 80)

    monthly_stats = []
    for month_name in sorted(metrics_df["month_name"].unique()):
        subset = metrics_df[metrics_df["month_name"] == month_name]
        row_data = {
            "month": month_name,
            "n_scenes": len(subset),
            "avg_total": float(subset["total_count"].mean()),
            "avg_anchorage": float(subset["anchorage_count"].mean()),
            "avg_berth": float(subset["berth_count"].mean()),
            "avg_congestion": float(subset["congestion_index"].mean()),
            "avg_soy_price": (
                float(subset.loc[subset["soy_price_valid"], "soy_price"].mean())
                if subset["soy_price_valid"].any()
                else None
            ),
            "n_scenes_with_soy": int(subset["soy_price_valid"].sum()),
        }
        monthly_stats.append(row_data)

        soy_season = " <-- SOY SEASON" if int(month_name.split("-")[1]) in HARVEST_MONTHS else ""
        soy_text = f"{row_data['avg_soy_price']:.2f}" if row_data["avg_soy_price"] is not None else "n/a"
        print(
            f"    {row_data['month']:<10} {row_data['n_scenes']:>7} {row_data['avg_total']:>10.1f} "
            f"{row_data['avg_anchorage']:>9.1f} {row_data['avg_berth']:>10.1f} "
            f"{row_data['avg_congestion']:>12.3f} {soy_text:>10s}{soy_season}"
        )

    # Soy season vs off-season comparison
    soy_months = metrics_df[metrics_df["month"].isin(HARVEST_MONTHS)]
    off_months = metrics_df[~metrics_df["month"].isin(HARVEST_MONTHS)]

    print("\n    Soy season (Feb-May) vs off-season summary:")
    if len(soy_months) > 0 and len(off_months) > 0:
        print(f"      Soy season  (n={len(soy_months)}): avg anchorage={soy_months['anchorage_count'].mean():.1f}, "
              f"avg congestion={soy_months['congestion_index'].mean():.3f}")
        print(f"      Off season  (n={len(off_months)}): avg anchorage={off_months['anchorage_count'].mean():.1f}, "
              f"avg congestion={off_months['congestion_index'].mean():.3f}")

        # Mann-Whitney U test for seasonal difference in anchorage counts
        if len(soy_months) >= 3 and len(off_months) >= 3:
            u_stat, u_p = stats.mannwhitneyu(
                soy_months["anchorage_count"].values,
                off_months["anchorage_count"].values,
                alternative="two-sided",
            )
            print(f"      Mann-Whitney U test (anchorage): U={u_stat:.1f}, p={u_p:.3f}")

    # ------------------------------------------------------------------
    # Step 7: Enhanced correlation (if available)
    # ------------------------------------------------------------------
    print("\n[7] Enhanced correlation analysis...")
    enhanced_results = {}
    try:
        from portvolume.analysis.enhanced_correlation import (
            test_stationarity,
            compute_correlations_with_correction,
        )

        if len(valid_soy_df) < MIN_CORR_OBS:
            raise ValueError("not enough overlapping soy observations for enhanced analysis")

        informative_metrics = []
        excluded_metrics = {}
        for col in [
            "anchorage_count",
            "berth_count",
            "congestion_index",
            "total_count",
            "queue_length_proxy",
        ]:
            diagnostics = series_diagnostics(valid_soy_df[col])
            if diagnostics["informative"]:
                informative_metrics.append(col)
            else:
                excluded_metrics[col] = diagnostics["note"]

        corr_df = valid_soy_df[informative_metrics + ["soy_price"]].copy()
        enhanced_results["excluded_metrics"] = excluded_metrics

        # Stationarity tests
        for col in [c for c in ["anchorage_count", "queue_length_proxy", "soy_price"] if c in corr_df.columns]:
            series = pd.Series(corr_df[col].values, index=pd.to_datetime(valid_soy_df["timestamp"]))
            stat_result = test_stationarity(series)
            print(f"    Stationarity({col}): recommendation={stat_result['recommendation']}, "
                  f"ADF p={stat_result['adf_pvalue']:.3f}" if not np.isnan(stat_result['adf_pvalue'])
                  else f"    Stationarity({col}): recommendation={stat_result['recommendation']}")
            enhanced_results[f"stationarity_{col}"] = stat_result

        # Correlations with BH correction
        econ_cols_for_corr = informative_metrics
        corr_with_correction = compute_correlations_with_correction(
            corr_df, "soy_price", econ_cols_for_corr
        )
        if not corr_with_correction.empty:
            print("\n    Correlations with Benjamini-Hochberg correction:")
            for _, row in corr_with_correction.iterrows():
                sig = "YES" if row["significant"] else "no"
                print(
                    f"      {row['indicator']:<22} rho={row['rho']:+.3f}  "
                    f"p={row['p_value']:.3f}  adj_p={row['adjusted_p_value']:.3f}  significant={sig}"
                )
            enhanced_results["bh_corrected"] = corr_with_correction.to_dict(orient="records")
        else:
            print("    Not enough data for BH-corrected correlations (need n>=10)")

    except Exception as e:
        print(f"    Enhanced correlation skipped: {e}")

    # ------------------------------------------------------------------
    # Step 8: Diagnostics against coincident trending
    # ------------------------------------------------------------------
    print("\n[8] Detrending, Granger, and out-of-time diagnostics...")
    signal_diagnostics = {}
    for metric_name in ["anchorage_count", "queue_length_proxy", "total_count"]:
        detrended_full = detrended_correlation_summary(valid_soy_df, metric_name)
        detrended_harvest = detrended_correlation_summary(
            valid_soy_df,
            metric_name,
            months=HARVEST_MONTHS,
        )
        granger_summary = monthly_differenced_granger_summary(valid_soy_df, metric_name)
        out_of_time = out_of_time_regression_summary(valid_soy_df, metric_name)
        within_year = yearly_spearman_summary(valid_soy_df, metric_name)

        signal_diagnostics[metric_name] = {
            "detrended_full_sample": detrended_full,
            "detrended_harvest_only": detrended_harvest,
            "within_year_levels": within_year,
            "monthly_differenced_granger": granger_summary,
            "out_of_time_level_regression": out_of_time,
        }

        print(
            f"    {metric_name:<22} detrended rho="
            f"{detrended_full['spearman_rho'] if detrended_full['spearman_rho'] is not None else 'n/a'}  "
            f"harvest detrended rho="
            f"{detrended_harvest['spearman_rho'] if detrended_harvest['spearman_rho'] is not None else 'n/a'}"
        )
        print(
            f"                          Granger direction={granger_summary['direction']} "
            f"(p={granger_summary['p_value'] if granger_summary['p_value'] is not None else 'n/a'}), "
            f"OOT R2={out_of_time['test_r2'] if out_of_time['test_r2'] is not None else 'n/a'}"
        )

    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    print("\n[9] Saving results...")
    output = {
        "site": "santos",
        "analysis": "soy_season_zone_analysis",
        "n_scenes": len(scenes),
        "date_range": {
            "start": metrics_df["date"].min().strftime("%Y-%m-%d"),
            "end": metrics_df["date"].max().strftime("%Y-%m-%d"),
        },
        "soy_alignment": {
            "indicator": "ZS=F",
            "source_path": str(ECONOMIC_DATA),
            "series_start": pd.to_datetime(soy_prices.index).min().strftime("%Y-%m-%d"),
            "series_end": pd.to_datetime(soy_prices.index).max().strftime("%Y-%m-%d"),
            "n_price_observations": int(len(soy_prices)),
            "max_gap_days": MAX_SOY_GAP_DAYS,
            "n_scenes_with_valid_soy": int(metrics_df["soy_price_valid"].sum()),
            "first_valid_scene": (
                valid_soy_df["date"].min().strftime("%Y-%m-%d") if not valid_soy_df.empty else None
            ),
            "last_valid_scene": (
                valid_soy_df["date"].max().strftime("%Y-%m-%d") if not valid_soy_df.empty else None
            ),
        },
        "zone_config": {
            "n_zones": len(zone_config.zones),
            "zones": [
                {"zone_id": z.zone_id, "zone_type": z.zone_type, "capacity": z.capacity_vessels}
                for z in zone_config.zones
            ],
        },
        "scene_metrics": metrics_df[
            ["scene_id", "timestamp", "date", "month_name", "anchorage_count",
             "berth_count", "channel_count", "outside_count", "total_count",
             "congestion_index", "queue_length_proxy", "soy_price", "soy_date",
             "soy_gap_days", "soy_price_valid"]
        ].assign(
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
            soy_date=lambda d: d["soy_date"].dt.strftime("%Y-%m-%d").where(d["soy_price_valid"], None),
        ).to_dict(orient="records"),
        "correlations": corr_results,
        "yearly_correlations": {
            "anchorage_count": yearly_spearman_summary(valid_soy_df, "anchorage_count"),
            "queue_length_proxy": yearly_spearman_summary(valid_soy_df, "queue_length_proxy"),
            "total_count": yearly_spearman_summary(valid_soy_df, "total_count"),
        },
        "signal_diagnostics": signal_diagnostics,
        "seasonal_pattern": monthly_stats,
        "soy_season_summary": {
            "soy_months_avg_anchorage": float(soy_months["anchorage_count"].mean()) if len(soy_months) > 0 else None,
            "off_months_avg_anchorage": float(off_months["anchorage_count"].mean()) if len(off_months) > 0 else None,
            "soy_months_avg_congestion": float(soy_months["congestion_index"].mean()) if len(soy_months) > 0 else None,
            "off_months_avg_congestion": float(off_months["congestion_index"].mean()) if len(off_months) > 0 else None,
            "anchorage_mannwhitney_p": (
                float(
                    stats.mannwhitneyu(
                        soy_months["anchorage_count"].values,
                        off_months["anchorage_count"].values,
                        alternative="two-sided",
                    ).pvalue
                )
                if len(soy_months) >= 3 and len(off_months) >= 3
                else None
            ),
        },
    }

    # Add enhanced results if available
    if enhanced_results:
        # Convert any non-serializable values
        serializable_enhanced = {}
        for k, v in enhanced_results.items():
            if isinstance(v, dict):
                serializable_enhanced[k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating, float)) and not np.isnan(vv)
                         else str(vv) if isinstance(vv, (np.floating, float)) and np.isnan(vv)
                         else vv)
                    for kk, vv in v.items()
                }
            else:
                serializable_enhanced[k] = v
        output["enhanced_analysis"] = serializable_enhanced

    OUTPUT_JSON.write_text(json.dumps(output, indent=2, default=str))
    print(f"    Results saved to: {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Scenes analyzed: {len(scenes)}")
    print(
        f"  Date range: {metrics_df['date'].min().strftime('%Y-%m-%d')} "
        f"to {metrics_df['date'].max().strftime('%Y-%m-%d')}"
    )
    print(f"  Zone breakdown: {len(zone_config.anchorage_zones)} anchorage, "
          f"{len(zone_config.berth_zones)} berth, {len(zone_config.channel_zones)} channel")
    print(
        f"  Valid soy overlap: {len(valid_soy_df)} scenes "
        f"(max gap {MAX_SOY_GAP_DAYS} days)"
    )
    print(f"\n  Key correlations with soybean futures (ZS=F):")
    for name, res in corr_results.items():
        if res["valid_for_inference"]:
            diff_text = ""
            if res["differenced_spearman_rho"] is not None:
                diff_text = (
                    f", diff rho={res['differenced_spearman_rho']:+.3f} "
                    f"(p={res['differenced_spearman_p']:.3f})"
                )
            print(
                f"    {name:<22} Spearman rho={res['spearman_rho']:+.3f} "
                f"(p={res['spearman_p']:.3f}{diff_text})"
            )
        else:
            print(f"    {name:<22} skipped ({res['note']})")
    if len(soy_months) > 0 and len(off_months) > 0:
        print(f"\n  Soy season effect:")
        print(f"    Avg anchorage count: {soy_months['anchorage_count'].mean():.1f} (soy) vs "
              f"{off_months['anchorage_count'].mean():.1f} (off)")
        print(f"    Avg congestion index: {soy_months['congestion_index'].mean():.3f} (soy) vs "
              f"{off_months['congestion_index'].mean():.3f} (off)")
    print("\n  Trend-robust diagnostics:")
    for metric_name in ["anchorage_count", "queue_length_proxy", "total_count"]:
        diag = signal_diagnostics[metric_name]
        detrended = diag["detrended_full_sample"]
        granger = diag["monthly_differenced_granger"]
        out_of_time = diag["out_of_time_level_regression"]
        print(
            f"    {metric_name:<22} detrended rho="
            f"{detrended['spearman_rho'] if detrended['spearman_rho'] is not None else 'n/a'} "
            f"(p={detrended['spearman_p'] if detrended['spearman_p'] is not None else 'n/a'}), "
            f"Granger={granger['direction']}, "
            f"OOT R2={out_of_time['test_r2'] if out_of_time['test_r2'] is not None else 'n/a'}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
