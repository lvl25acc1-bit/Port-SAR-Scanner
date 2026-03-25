"""Backtest analysis: scene diagnostic, monthly headline, controlled model,
tide-by-zone, weekly robustness, and audit scene selection.

Reads only from canonical parquet tables in data/backtest/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import TheilSenRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge canonical tables
# ---------------------------------------------------------------------------

def load_and_merge(backtest_dir: Path) -> pd.DataFrame:
    """Load all canonical tables and merge into a single scene-level DataFrame."""
    universe = pd.read_parquet(backtest_dir / "scene_universe.parquet")
    universe["date"] = pd.to_datetime(universe["date"])

    # AIS daily counts — join on date
    ais_path = backtest_dir / "ais_daily_counts.parquet"
    if ais_path.exists():
        ais = pd.read_parquet(ais_path)
        ais["query_date"] = pd.to_datetime(ais["query_date"])
        universe = universe.merge(
            ais, left_on="date", right_on="query_date", how="left"
        ).drop(columns=["query_date"], errors="ignore")

    # Wind
    wind_path = backtest_dir / "wind.parquet"
    if wind_path.exists():
        wind = pd.read_parquet(wind_path)
        wind["scene_midpoint_utc"] = pd.to_datetime(wind["scene_midpoint_utc"], utc=True)
        universe["scene_midpoint_utc"] = pd.to_datetime(universe["scene_midpoint_utc"], utc=True)
        universe = universe.merge(wind, on="scene_midpoint_utc", how="left")

    # Tide
    tide_path = backtest_dir / "tide.parquet"
    if tide_path.exists():
        tide = pd.read_parquet(tide_path)
        tide["scene_midpoint_utc"] = pd.to_datetime(tide["scene_midpoint_utc"], utc=True)
        universe = universe.merge(tide, on="scene_midpoint_utc", how="left")

    logger.info("Merged scene table: %d rows, %d columns", len(universe), len(universe.columns))
    return universe


# ---------------------------------------------------------------------------
# Scene-level diagnostic
# ---------------------------------------------------------------------------

def scene_level_diagnostic(merged: pd.DataFrame) -> dict:
    """Scene-level correlation: SAR vs GFW same-date.

    CAVEAT: GFW is daily presence, not overpass-time AIS. Same-date mismatch
    may reflect time aggregation, not detector failure.
    """
    results = {}

    for sar_col, gfw_col, label in [
        ("vessel_count", "gfw_unique_vessels", "count_vs_count"),
        ("total_pixel_area", "gfw_sum_area_m2", "area_vs_area"),
    ]:
        if gfw_col not in merged.columns:
            continue

        valid = merged[[sar_col, gfw_col]].dropna()
        if len(valid) < 10:
            continue

        x, y = valid[sar_col].values, valid[gfw_col].values

        spearman_r, spearman_p = stats.spearmanr(x, y)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        mae = np.mean(np.abs(x - y))
        bias = np.mean(x - y)

        # Theil-Sen robust regression
        ts = TheilSenRegressor(random_state=42)
        ts.fit(x.reshape(-1, 1), y)

        results[label] = {
            "n_scenes": len(valid),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "mae": round(float(mae), 2),
            "bias": round(float(bias), 2),
            "theilsen_slope": round(float(ts.coef_[0]), 4),
            "theilsen_intercept": round(float(ts.intercept_), 2),
            "caveat": "diagnostic only — GFW is daily presence, not overpass-time AIS",
        }

    logger.info("Scene diagnostic: %s", {k: v["spearman_r"] for k, v in results.items()})
    return results


# ---------------------------------------------------------------------------
# Monthly headline validation
# ---------------------------------------------------------------------------

def monthly_validation(merged: pd.DataFrame, ais_monthly: pd.DataFrame) -> dict:
    """Monthly SAR vs GFW — the headline result.

    'Does the SAR series track real large-vessel occupancy trends over time?'
    """
    merged = merged.copy()
    merged["month"] = merged["date"].dt.to_period("M").dt.to_timestamp()

    # Monthly SAR aggregates by pass_family
    sar_monthly = merged.groupby(["month", "pass_family"]).agg(
        sar_median_count=("vessel_count", "median"),
        sar_median_area=("total_pixel_area", "median"),
        n_scenes=("vessel_count", "count"),
    ).reset_index()

    # Also compute overall monthly (both pass families)
    sar_monthly_all = merged.groupby("month").agg(
        sar_median_count=("vessel_count", "median"),
        sar_median_area=("total_pixel_area", "median"),
        n_scenes=("vessel_count", "count"),
    ).reset_index()

    if ais_monthly.empty:
        return {"error": "no monthly AIS data"}

    # Merge SAR monthly with GFW monthly
    combined = sar_monthly_all.merge(ais_monthly, on="month", how="inner")

    results = {}
    for sar_col, gfw_col, label in [
        ("sar_median_count", "gfw_monthly_unique_vessels", "count_vs_count"),
        ("sar_median_area", "gfw_monthly_sum_area_m2", "area_vs_area"),
    ]:
        if gfw_col not in combined.columns:
            continue

        valid = combined[[sar_col, gfw_col]].dropna()
        if len(valid) < 6:
            continue

        x, y = valid[sar_col].values, valid[gfw_col].values
        spearman_r, spearman_p = stats.spearmanr(x, y)
        pearson_r, pearson_p = stats.pearsonr(x, y)

        ts = TheilSenRegressor(random_state=42)
        ts.fit(x.reshape(-1, 1), y)

        results[label] = {
            "n_months": len(valid),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 6),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "theilsen_slope": round(float(ts.coef_[0]), 4),
            "theilsen_intercept": round(float(ts.intercept_), 2),
        }

    logger.info("Monthly validation: %s", {k: v["spearman_r"] for k, v in results.items()})
    return results


# ---------------------------------------------------------------------------
# Controlled model
# ---------------------------------------------------------------------------

def controlled_model(merged: pd.DataFrame) -> dict:
    """Robust regression with nuisance covariates.

    vessel_count ~ gfw_count + C(pass_family) + C(satellite) + wind + tide
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not available; skipping controlled model")
        return {}

    required = ["vessel_count", "gfw_unique_vessels", "pass_family", "satellite"]
    if not all(c in merged.columns for c in required):
        return {"error": "missing required columns"}

    df = merged[merged["gfw_unique_vessels"].notna()].copy()
    if len(df) < 30:
        return {"error": f"too few scenes ({len(df)})"}

    # Encode categoricals
    df["is_evening"] = (df["pass_family"] == "evening").astype(int)
    df["is_s1c"] = (df["satellite"] == "S1C").astype(int)

    predictors = ["gfw_unique_vessels", "is_evening", "is_s1c"]
    if "wind_speed_mps" in df.columns:
        predictors.append("wind_speed_mps")
    if "tide_height_cm" in df.columns:
        predictors.append("tide_height_cm")
    if "minutes_from_high_tide" in df.columns:
        predictors.append("minutes_from_high_tide")

    X = sm.add_constant(df[predictors].astype(float))
    y = df["vessel_count"].astype(float)

    # Drop rows with NaN in any predictor
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    result = model.fit()

    coefs = {}
    for name, val, pval in zip(result.params.index, result.params.values, result.pvalues):
        coefs[name] = {"coef": round(float(val), 4), "p_value": round(float(pval), 6)}

    # Store residuals for weekly aggregation
    df.loc[mask.values, "residual"] = result.resid

    logger.info("Controlled model: R²≈%.3f, %d predictors", result.rsquared if hasattr(result, 'rsquared') else 0, len(predictors))

    return {
        "n_scenes": int(mask.sum()),
        "coefficients": coefs,
        "residuals": df[["scene_id", "date", "pass_family", "residual"]].to_dict("records") if "residual" in df.columns else [],
    }


# ---------------------------------------------------------------------------
# Tide-by-zone
# ---------------------------------------------------------------------------

def tide_by_zone(
    zone_detections: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict:
    """Compare tide effect in channel vs basin zones."""
    if zone_detections.empty or "tide_height_cm" not in merged.columns:
        return {"error": "missing zone or tide data"}

    # Aggregate zone counts per scene
    zone_scene = zone_detections.groupby(["scene_id", "zone_type"]).agg(
        zone_count=("zone_vessel_count", "sum"),
    ).reset_index()

    # Merge with tide
    tide_cols = ["scene_id", "tide_height_cm", "minutes_from_high_tide"]
    available = [c for c in tide_cols if c in merged.columns]
    zone_merged = zone_scene.merge(merged[available], on="scene_id", how="left")

    results = {}
    for zone in ["channel", "basin"]:
        zdf = zone_merged[zone_merged["zone_type"] == zone].dropna(subset=["tide_height_cm"])
        if len(zdf) < 20:
            results[zone] = {"error": f"too few scenes ({len(zdf)})"}
            continue

        r, p = stats.spearmanr(zdf["tide_height_cm"], zdf["zone_count"])
        results[zone] = {
            "n_scenes": len(zdf),
            "spearman_r": round(float(r), 4),
            "spearman_p": round(float(p), 6),
        }

    logger.info("Tide-by-zone: %s", results)
    return results


# ---------------------------------------------------------------------------
# Weekly robustness
# ---------------------------------------------------------------------------

def weekly_robustness(
    merged: pd.DataFrame,
    controlled_results: dict,
) -> pd.DataFrame:
    """Build weekly median (raw) and weekly adjusted residual median.

    Labeled as robustness, not headline.
    """
    df = merged.copy()
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)

    # Raw weekly median by pass_family
    raw = df.groupby(["iso_year", "iso_week", "pass_family"]).agg(
        raw_median_count=("vessel_count", "median"),
        raw_median_area=("total_pixel_area", "median"),
        n_scenes=("vessel_count", "count"),
    ).reset_index()

    # Adjusted: use residuals from controlled model
    residuals = controlled_results.get("residuals", [])
    if residuals:
        res_df = pd.DataFrame(residuals)
        res_df["date"] = pd.to_datetime(res_df["date"])
        res_df["iso_year"] = res_df["date"].dt.isocalendar().year.astype(int)
        res_df["iso_week"] = res_df["date"].dt.isocalendar().week.astype(int)

        adjusted = res_df.groupby(["iso_year", "iso_week", "pass_family"]).agg(
            adjusted_median_residual=("residual", "median"),
        ).reset_index()

        raw = raw.merge(adjusted, on=["iso_year", "iso_week", "pass_family"], how="left")

    logger.info("Weekly robustness: %d rows", len(raw))
    return raw


# ---------------------------------------------------------------------------
# Audit scene selection
# ---------------------------------------------------------------------------

def select_audit_scenes(merged: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Select a stratified 20-scene audit set.

    4 pre-S1C winter (Dec 2024–Feb 2025): 2 low-wind, 2 high-wind
    8 post-S1C morning: low/high wind × low/high tide, both S1A/S1C
    8 post-S1C evening: same
    """
    df = merged.copy()

    # Split pre/post S1C (first S1C scene is ~Apr 2025)
    s1c_start = df[df["satellite"] == "S1C"]["date"].min()
    pre_s1c = df[df["date"] < s1c_start] if pd.notna(s1c_start) else df.head(0)
    post_s1c = df[df["date"] >= s1c_start] if pd.notna(s1c_start) else df

    audit_ids = []

    # 4 pre-S1C winter scenes
    winter = pre_s1c[pre_s1c["date"].dt.month.isin([12, 1, 2])]
    if len(winter) >= 4 and "wind_speed_mps" in winter.columns:
        wind_median = winter["wind_speed_mps"].median()
        low_wind = winter[winter["wind_speed_mps"] <= wind_median].head(2)
        high_wind = winter[winter["wind_speed_mps"] > wind_median].head(2)
        audit_ids.extend(low_wind["scene_id"].tolist())
        audit_ids.extend(high_wind["scene_id"].tolist())
    elif len(winter) > 0:
        audit_ids.extend(winter.head(4)["scene_id"].tolist())

    # 16 post-S1C scenes: 8 morning + 8 evening
    for family in ["morning", "evening"]:
        subset = post_s1c[post_s1c["pass_family"] == family]
        if len(subset) < 4:
            audit_ids.extend(subset["scene_id"].tolist())
            continue

        wind_med = subset["wind_speed_mps"].median() if "wind_speed_mps" in subset.columns else 5
        tide_med = subset["tide_height_cm"].median() if "tide_height_cm" in subset.columns else 0

        for sat in ["S1A", "S1C"]:
            sat_sub = subset[subset["satellite"] == sat]
            if len(sat_sub) < 2:
                audit_ids.extend(sat_sub.head(2)["scene_id"].tolist())
                continue

            has_wind = "wind_speed_mps" in sat_sub.columns and sat_sub["wind_speed_mps"].notna().any()
            has_tide = "tide_height_cm" in sat_sub.columns and sat_sub["tide_height_cm"].notna().any()

            # Pick one from each wind/tide quadrant (degrade gracefully)
            wind_conditions = ["low", "high"] if has_wind else ["all"]
            tide_conditions = ["low", "high"] if has_tide else ["all"]

            for wind_cond in wind_conditions:
                for tide_cond in tide_conditions:
                    if has_wind:
                        w_mask = sat_sub["wind_speed_mps"] <= wind_med if wind_cond == "low" else sat_sub["wind_speed_mps"] > wind_med
                    else:
                        w_mask = pd.Series(True, index=sat_sub.index)

                    if has_tide:
                        t_mask = sat_sub["tide_height_cm"] <= tide_med if tide_cond == "low" else sat_sub["tide_height_cm"] > tide_med
                    else:
                        t_mask = pd.Series(True, index=sat_sub.index)

                    candidates = sat_sub[w_mask & t_mask]
                    if len(candidates) > 0:
                        audit_ids.append(candidates.iloc[0]["scene_id"])

    # Deduplicate and limit
    audit_ids = list(dict.fromkeys(audit_ids))[:n]

    result = merged[merged["scene_id"].isin(audit_ids)].copy()
    logger.info("Audit set: %d scenes selected", len(result))
    return result


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

def save_outputs(
    backtest_dir: Path,
    scene_diag: dict,
    monthly_val: dict,
    controlled: dict,
    tide_zone: dict,
    weekly: pd.DataFrame,
    sensitivity: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Write all analysis outputs."""
    out = backtest_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "scene_diagnostic.json", "w") as f:
        json.dump(scene_diag, f, indent=2, default=str)

    with open(out / "monthly_validation.json", "w") as f:
        json.dump(monthly_val, f, indent=2, default=str)

    with open(out / "controlled_model.json", "w") as f:
        json.dump(controlled, f, indent=2, default=str)

    with open(out / "tide_by_zone.json", "w") as f:
        json.dump(tide_zone, f, indent=2, default=str)

    weekly.to_parquet(out / "weekly_robustness.parquet", index=False)

    if not sensitivity.empty:
        sensitivity.to_parquet(out / "sensitivity_sweep.parquet", index=False)

    audit.to_parquet(out / "audit_20_scenes.parquet", index=False)

    logger.info("All outputs saved to %s", out)
