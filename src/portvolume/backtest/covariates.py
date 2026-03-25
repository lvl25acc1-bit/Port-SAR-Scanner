"""Fetch wind and tide covariates for scene midpoints.

Wind: Open-Meteo historical weather API (free, no key).
Tide: Rijkswaterstaat waterinfo API for Hoek van Holland (free).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wind (Open-Meteo ERA5)
# ---------------------------------------------------------------------------

def fetch_wind(
    midpoints: list[datetime],
    lat: float = 51.93,
    lon: float = 4.25,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch hourly 10m wind speed from Open-Meteo and interpolate to scene midpoints.

    Returns DataFrame with columns: scene_midpoint_utc, wind_speed_mps.
    """
    cache_path = cache_dir / "wind.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached wind data")
        return pd.read_parquet(cache_path)

    # Determine date range
    dates = pd.DatetimeIndex(midpoints)
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")

    logger.info("Fetching wind data from Open-Meteo: %s to %s", start_date, end_date)

    # Open-Meteo has a max range per request; chunk by year
    all_hourly = []
    for year_start, year_end in _year_chunks(start_date, end_date):
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={year_start}&end_date={year_end}"
            f"&hourly=wind_speed_10m"
            f"&timezone=UTC"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        speeds = hourly.get("wind_speed_10m", [])
        if times and speeds:
            chunk_df = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "wind_speed_mps": speeds})
            # Open-Meteo returns km/h, convert to m/s
            chunk_df["wind_speed_mps"] = chunk_df["wind_speed_mps"] / 3.6
            all_hourly.append(chunk_df)

    if not all_hourly:
        logger.warning("No wind data returned from Open-Meteo")
        return pd.DataFrame({"scene_midpoint_utc": midpoints, "wind_speed_mps": np.nan})

    hourly_df = pd.concat(all_hourly, ignore_index=True).sort_values("time")
    hourly_df = hourly_df.set_index("time")

    # Interpolate to each scene midpoint
    results = []
    for mp in midpoints:
        mp_ts = pd.Timestamp(mp)
        if mp_ts.tzinfo is None:
            mp_ts = mp_ts.tz_localize("UTC")
        # Find bracketing hours
        before = hourly_df.index[hourly_df.index <= mp_ts]
        after = hourly_df.index[hourly_df.index >= mp_ts]

        if len(before) > 0 and len(after) > 0:
            t0 = before[-1]
            t1 = after[0]
            if t0 == t1:
                wind = hourly_df.loc[t0, "wind_speed_mps"]
            else:
                w0 = hourly_df.loc[t0, "wind_speed_mps"]
                w1 = hourly_df.loc[t1, "wind_speed_mps"]
                frac = (mp_ts - t0).total_seconds() / (t1 - t0).total_seconds()
                wind = w0 + frac * (w1 - w0)
        else:
            wind = np.nan

        results.append({"scene_midpoint_utc": mp, "wind_speed_mps": round(float(wind), 2)})

    result_df = pd.DataFrame(results)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(cache_path, index=False)
        logger.info("Cached wind data: %d scenes", len(result_df))

    return result_df


def _year_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split a date range into per-year chunks for Open-Meteo API limits."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    current = start
    while current <= end:
        year_end = min(pd.Timestamp(f"{current.year}-12-31"), end)
        chunks.append((current.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
        current = pd.Timestamp(f"{current.year + 1}-01-01")
    return chunks


# ---------------------------------------------------------------------------
# Tide (Rijkswaterstaat)
# ---------------------------------------------------------------------------

def fetch_tide(
    midpoints: list[datetime],
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch tidal height from Rijkswaterstaat for Hoek van Holland.

    Uses the ddlpy package if available, otherwise falls back to a simple
    REST query. Interpolates to scene midpoints.

    Returns DataFrame with: scene_midpoint_utc, tide_height_cm, minutes_from_high_tide.
    """
    cache_path = cache_dir / "tide.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached tide data")
        return pd.read_parquet(cache_path)

    logger.info("Fetching tide data from Rijkswaterstaat")

    try:
        return _fetch_tide_ddlpy(midpoints, cache_path)
    except Exception as exc:
        logger.warning("ddlpy tide fetch failed (%s), trying REST fallback", exc)

    try:
        return _fetch_tide_rest(midpoints, cache_path)
    except Exception as exc:
        logger.warning("RWS REST tide fetch failed (%s), using tidal prediction", exc)

    try:
        return _predict_tide_uptide(midpoints, cache_path)
    except Exception:
        logger.exception("All tide sources failed")
        return pd.DataFrame({
            "scene_midpoint_utc": midpoints,
            "tide_height_cm": np.nan,
            "minutes_from_high_tide": np.nan,
        })


def _fetch_tide_ddlpy(midpoints: list[datetime], cache_path: Path | None) -> pd.DataFrame:
    """Fetch via ddlpy (Dutch government open data package)."""
    import ddlpy

    # Find Hoek van Holland station
    locations = ddlpy.locations()
    hvh = locations[locations.index.str.contains("HOEKVHLD", case=False)]
    if hvh.empty:
        # Try broader search
        hvh = locations[locations["Naam"].str.contains("Hoek van Holland", case=False, na=False)]
    if hvh.empty:
        raise ValueError("Hoek van Holland station not found in ddlpy")

    # Get water level measurements
    station = hvh.iloc[0]
    dates = pd.DatetimeIndex(midpoints)
    start = dates.min() - pd.Timedelta(hours=12)
    end = dates.max() + pd.Timedelta(hours=12)

    measurements = ddlpy.measurements(station, start, end)
    if measurements.empty:
        raise ValueError("No tide measurements returned")

    # Parse into time series
    ts = measurements["Meetwaarde.Waarde_Numeriek"].astype(float)
    ts.index = pd.to_datetime(measurements.index, utc=True)
    ts = ts.sort_index()

    return _interpolate_tide(ts, midpoints, cache_path)


def _fetch_tide_rest(midpoints: list[datetime], cache_path: Path | None) -> pd.DataFrame:
    """Fallback: fetch tide from RWS waterwebservices."""
    dates = pd.DatetimeIndex(midpoints)
    start = (dates.min() - pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
    end = (dates.max() + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")

    url = "https://waterwebservices.rijkswaterstaat.nl/ONLINEWAARNEMINGENSERVICES_DBO/OphalenWaarnemingen"
    body = {
        "Locatie": {"X": 4.12, "Y": 51.98, "Code": "HOEKVHLD"},
        "AquoPlusWaarwordenMetadata": {
            "AquoMetadata": {"Grootheid": {"Code": "WATHTE"}},
        },
        "Periode": {"Begindatumtijd": start, "Einddatumtijd": end},
    }

    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("WaarnemingenLijst", [{}])[0].get("MetingenLijst", [])
    if not records:
        raise ValueError("No tide records from RWS REST API")

    times = []
    values = []
    for rec in records:
        t = rec.get("Tijdstip")
        v = rec.get("Meetwaarde", {}).get("Waarde_Numeriek")
        if t and v is not None:
            times.append(pd.Timestamp(t, tz="UTC"))
            values.append(float(v))

    ts = pd.Series(values, index=pd.DatetimeIndex(times)).sort_index()
    return _interpolate_tide(ts, midpoints, cache_path)


def _interpolate_tide(
    tide_series: pd.Series,
    midpoints: list[datetime],
    cache_path: Path | None,
) -> pd.DataFrame:
    """Interpolate tide to scene midpoints and compute minutes_from_high_tide."""
    results = []
    for mp in midpoints:
        mp_ts = pd.Timestamp(mp)
        if mp_ts.tzinfo is None:
            mp_ts = mp_ts.tz_localize("UTC")

        # Interpolate height
        before = tide_series.index[tide_series.index <= mp_ts]
        after = tide_series.index[tide_series.index >= mp_ts]

        if len(before) > 0 and len(after) > 0:
            t0, t1 = before[-1], after[0]
            if t0 == t1:
                height = tide_series.loc[t0]
            else:
                v0 = tide_series.loc[t0]
                v1 = tide_series.loc[t1]
                frac = (mp_ts - t0).total_seconds() / (t1 - t0).total_seconds()
                height = v0 + frac * (v1 - v0)
        else:
            height = np.nan

        # Find nearest high tide (local max within ±12h window)
        window_start = mp_ts - pd.Timedelta(hours=12)
        window_end = mp_ts + pd.Timedelta(hours=12)
        window = tide_series[window_start:window_end]

        if len(window) > 2:
            high_tide_idx = window.idxmax()
            minutes_from_ht = (mp_ts - high_tide_idx).total_seconds() / 60.0
        else:
            minutes_from_ht = np.nan

        results.append({
            "scene_midpoint_utc": mp,
            "tide_height_cm": round(float(height), 1),
            "minutes_from_high_tide": round(float(minutes_from_ht), 0),
        })

    result_df = pd.DataFrame(results)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(cache_path, index=False)
        logger.info("Cached tide data: %d scenes", len(result_df))

    return result_df


def _predict_tide_uptide(
    midpoints: list[datetime],
    cache_path: Path | None,
) -> pd.DataFrame:
    """Predict tide using uptide harmonic constituents for Hoek van Holland.

    Uses the main M2/S2/N2/K1/O1 constituents. Not as accurate as
    measured data but sufficient for covariate analysis.
    """
    import uptide

    logger.info("Using uptide tidal prediction for %d scenes", len(midpoints))

    # Standard harmonic constituents for Hoek van Holland (approximate)
    # Amplitudes in meters, phases in degrees
    tide = uptide.Tides(["M2", "S2", "N2", "K1", "O1"])
    # Hoek van Holland amplitudes (m) and phases (degrees) from tidal tables
    tide.set_initial_time(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    amplitudes = np.array([0.82, 0.22, 0.16, 0.08, 0.06])
    phases = np.deg2rad(np.array([330.0, 10.0, 310.0, 150.0, 280.0]))

    # Generate a dense time series for interpolation of high tides
    dates = pd.DatetimeIndex(midpoints)
    start = dates.min() - pd.Timedelta(hours=13)
    end = dates.max() + pd.Timedelta(hours=13)
    t_dense = pd.date_range(start, end, freq="10min", tz="UTC")

    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def predict_height(dt):
        seconds = (dt - t0).total_seconds()
        h = 0.0
        omegas = tide.omega
        for amp, phase, omega in zip(amplitudes, phases, omegas):
            h += amp * np.cos(omega * seconds - phase)
        return h * 100  # convert m to cm

    # Build dense series
    dense_heights = [predict_height(t.to_pydatetime()) for t in t_dense]
    tide_series = pd.Series(dense_heights, index=t_dense)

    return _interpolate_tide(tide_series, midpoints, cache_path)
