"""Wind and wave-age deconfounding for SAR vessel detection.

Rotterdam backtest showed wind speed explains ~43% of vessel count variance
(coefficient = -4.98, p<0.001). This module provides:
1. Wave-age classification to adapt CFAR thresholds
2. Post-hoc residual correction to remove wind/pass/satellite confounds
3. RLM fitting for nuisance model coefficients
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default model coefficients from Rotterdam backtest
_DEFAULT_COEFFICIENTS: dict = {
    "wind_coef": -4.98,
    "pass_evening_coef": -3.2,
    "satellite_s1c_coef": 2.1,
    "intercept": 120.0,
}


@dataclass(frozen=True)
class WaveAgeParams:
    """Parameters for wave-age classification and CFAR correction."""

    young_sea_threshold: float = 1.2
    swell_period_threshold: float = 10.0
    correction_factors: dict = field(
        default_factory=lambda: {
            "young_sea": 1.4,
            "old_sea": 1.0,
            "swell": 0.85,
        }
    )
    wind_speed_scale: float = 0.03  # linear scaling added per m/s above 5


def classify_wave_age(
    wind_speed_mps: float,
    peak_wave_period_s: Optional[float] = None,
    significant_wave_height_m: Optional[float] = None,
) -> str:
    """Classify sea state into wave-age categories.

    Uses wave age parameter cp/u* when wave data is available;
    falls back to a wind-only heuristic otherwise.

    Parameters
    ----------
    wind_speed_mps : 10-m wind speed in m/s.
    peak_wave_period_s : Peak wave period in seconds (optional).
    significant_wave_height_m : Significant wave height in metres (optional).

    Returns
    -------
    One of ``"young_sea"``, ``"old_sea"``, or ``"swell"``.
    """
    if peak_wave_period_s is not None and significant_wave_height_m is not None:
        # Estimate phase speed from deep-water dispersion: cp = g*T/(2*pi)
        g = 9.81
        cp = g * peak_wave_period_s / (2.0 * np.pi)

        # Friction velocity approximation: u* ~ 0.04 * U10
        u_star = max(0.04 * wind_speed_mps, 1e-6)
        wave_age = cp / u_star

        if wave_age < 15.0:
            return "young_sea"
        elif peak_wave_period_s >= 10.0:
            return "swell"
        else:
            return "old_sea"

    # Wind-only heuristic fallback
    if wind_speed_mps >= 8.0:
        return "young_sea"
    elif wind_speed_mps <= 3.0:
        return "swell"
    else:
        return "old_sea"


def compute_cfar_correction_factor(
    wave_age_class: str,
    wind_speed_mps: float,
    params: Optional[WaveAgeParams] = None,
) -> float:
    """Compute a multiplicative correction factor for the CFAR threshold.

    Higher factors raise the threshold (fewer false alarms in rough seas).
    Lower factors lower the threshold (more sensitivity in calm seas).

    Parameters
    ----------
    wave_age_class : One of ``"young_sea"``, ``"old_sea"``, ``"swell"``.
    wind_speed_mps : 10-m wind speed in m/s.
    params : Wave-age parameters; uses defaults if *None*.

    Returns
    -------
    Multiplicative correction factor (1.0 = no change).
    """
    if params is None:
        params = WaveAgeParams()

    base = params.correction_factors.get(wave_age_class, 1.0)

    # Add a linear wind-speed scaling above 5 m/s
    wind_excess = max(wind_speed_mps - 5.0, 0.0)
    factor = base + params.wind_speed_scale * wind_excess

    return float(factor)


def compute_wind_adjusted_count(
    raw_count: float,
    wind_speed_mps: float,
    pass_family: str,
    satellite: str,
    model_coefficients: Optional[dict] = None,
) -> float:
    """Remove wind/pass/satellite confounds via post-hoc residual.

    adjusted = raw - (wind_coef * wind + pass_coef * is_evening + sat_coef * is_s1c)

    Parameters
    ----------
    raw_count : Observed vessel count.
    wind_speed_mps : 10-m wind speed in m/s.
    pass_family : ``"morning"`` or ``"evening"`` (Sentinel-1 pass family).
    satellite : ``"S1A"``, ``"S1B"``, or ``"S1C"``.
    model_coefficients : Dict with keys ``wind_coef``, ``pass_evening_coef``,
        ``satellite_s1c_coef``, ``intercept``. Uses Rotterdam defaults if *None*.

    Returns
    -------
    Wind-adjusted vessel count.
    """
    coefs = model_coefficients or _DEFAULT_COEFFICIENTS

    wind_coef = coefs.get("wind_coef", -4.98)
    pass_coef = coefs.get("pass_evening_coef", -3.2)
    sat_coef = coefs.get("satellite_s1c_coef", 2.1)
    intercept = coefs.get("intercept", 120.0)

    is_evening = 1.0 if pass_family.lower() == "evening" else 0.0
    is_s1c = 1.0 if satellite.upper() == "S1C" else 0.0

    # Nuisance prediction (confound signal)
    nuisance = wind_coef * wind_speed_mps + pass_coef * is_evening + sat_coef * is_s1c

    # Adjusted = raw - nuisance (keep the intercept-centred level)
    adjusted = raw_count - nuisance

    return float(adjusted)


def fit_wind_model(
    scene_df: pd.DataFrame,
    target_col: str = "vessel_count",
) -> dict:
    """Fit a robust linear model for confound removal.

    Model: count ~ wind_speed_mps + C(pass_family) + C(satellite)

    Uses Huber's M-estimator (RLM) to down-weight outlier scenes.

    Parameters
    ----------
    scene_df : DataFrame with columns ``wind_speed_mps``, ``pass_family``,
        ``satellite``, and *target_col*.

    Returns
    -------
    Dict with keys ``wind_coef``, ``pass_evening_coef``, ``satellite_s1c_coef``,
    ``intercept``, and ``r_squared``.
    """
    import statsmodels.api as sm

    df = scene_df.dropna(subset=[target_col, "wind_speed_mps"]).copy()
    if len(df) < 10:
        logger.warning(
            "Too few observations (%d) to fit wind model; returning defaults",
            len(df),
        )
        return dict(_DEFAULT_COEFFICIENTS)

    # Encode categoricals
    df["is_evening"] = (df["pass_family"].str.lower() == "evening").astype(float)
    df["is_s1c"] = (df["satellite"].str.upper() == "S1C").astype(float)

    X = df[["wind_speed_mps", "is_evening", "is_s1c"]]
    X = sm.add_constant(X)
    y = df[target_col]

    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    result = rlm_model.fit()

    coefs = {
        "intercept": float(result.params.get("const", 0.0)),
        "wind_coef": float(result.params.get("wind_speed_mps", 0.0)),
        "pass_evening_coef": float(result.params.get("is_evening", 0.0)),
        "satellite_s1c_coef": float(result.params.get("is_s1c", 0.0)),
    }

    # Approximate R-squared from RLM (pseudo)
    ss_res = np.sum((y - result.fittedvalues) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    coefs["r_squared"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info(
        "Wind model fit: wind=%.2f, pass=%.2f, sat=%.2f, R2=%.3f",
        coefs["wind_coef"],
        coefs["pass_evening_coef"],
        coefs["satellite_s1c_coef"],
        coefs["r_squared"],
    )

    return coefs


def fetch_wave_data(
    timestamps: list,
    lat: float,
    lon: float,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch wave period and height from Open-Meteo ERA5 Marine API.

    Parameters
    ----------
    timestamps : Scene timestamps (datetime-like).
    lat, lon : Location coordinates.
    cache_dir : Optional directory for caching results.

    Returns
    -------
    DataFrame with columns: ``timestamp``, ``peak_wave_period_s``,
    ``significant_wave_height_m``, ``wave_age_class``.
    """
    import requests

    cache_path = cache_dir / "wave.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached wave data")
        return pd.read_parquet(cache_path)

    dates = pd.DatetimeIndex(timestamps)
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")

    logger.info("Fetching wave data from Open-Meteo Marine: %s to %s", start_date, end_date)

    url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=wave_period,wave_height,wind_wave_period,wind_wave_height"
        f"&timezone=UTC"
    )

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        periods = hourly.get("wave_period", [])
        heights = hourly.get("wave_height", [])

        if not times:
            raise ValueError("No wave data returned")

        hourly_df = pd.DataFrame({
            "time": pd.to_datetime(times, utc=True),
            "peak_wave_period_s": periods,
            "significant_wave_height_m": heights,
        }).set_index("time")

    except Exception as exc:
        logger.warning("Wave data fetch failed (%s); returning NaN", exc)
        result = pd.DataFrame({
            "timestamp": timestamps,
            "peak_wave_period_s": np.nan,
            "significant_wave_height_m": np.nan,
            "wave_age_class": "old_sea",
        })
        return result

    # Interpolate to each timestamp
    results = []
    for ts in timestamps:
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")

        before = hourly_df.index[hourly_df.index <= ts_pd]
        after = hourly_df.index[hourly_df.index >= ts_pd]

        period = np.nan
        height = np.nan

        if len(before) > 0 and len(after) > 0:
            t0, t1 = before[-1], after[0]
            if t0 == t1:
                period = hourly_df.loc[t0, "peak_wave_period_s"]
                height = hourly_df.loc[t0, "significant_wave_height_m"]
            else:
                frac = (ts_pd - t0).total_seconds() / (t1 - t0).total_seconds()
                period = (
                    hourly_df.loc[t0, "peak_wave_period_s"]
                    + frac * (hourly_df.loc[t1, "peak_wave_period_s"]
                              - hourly_df.loc[t0, "peak_wave_period_s"])
                )
                height = (
                    hourly_df.loc[t0, "significant_wave_height_m"]
                    + frac * (hourly_df.loc[t1, "significant_wave_height_m"]
                              - hourly_df.loc[t0, "significant_wave_height_m"])
                )

        wave_age_class = classify_wave_age(
            wind_speed_mps=0.0,  # wind not available here; use wave data heuristic
            peak_wave_period_s=float(period) if not np.isnan(period) else None,
            significant_wave_height_m=float(height) if not np.isnan(height) else None,
        )

        results.append({
            "timestamp": ts,
            "peak_wave_period_s": round(float(period), 2) if not np.isnan(period) else np.nan,
            "significant_wave_height_m": round(float(height), 2) if not np.isnan(height) else np.nan,
            "wave_age_class": wave_age_class,
        })

    result_df = pd.DataFrame(results)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(cache_path, index=False)
        logger.info("Cached wave data: %d scenes", len(result_df))

    return result_df
