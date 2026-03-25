"""Weekly aggregation of per-scene detection counts into time series."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from portvolume.sites import Site

logger = logging.getLogger(__name__)


def load_detection_summaries(
    detections_dir: Path,
    site_id: str,
) -> pd.DataFrame:
    """Load all detection summary CSVs for a site into a DataFrame.

    Expects files at: {detections_dir}/{site_id}/summaries.csv
    with columns: scene_id, timestamp, vessel_count (or bright_pixel_fraction,
    estimated_aircraft_count for airports).
    """
    summary_path = detections_dir / site_id / "summaries.csv"
    if not summary_path.exists():
        logger.warning("No summaries found for site %s", site_id)
        return pd.DataFrame()

    df = pd.read_csv(summary_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def aggregate_weekly(
    daily_counts: pd.DataFrame,
    value_column: str,
    day_of_week: str = "monday",
) -> pd.DataFrame:
    """Aggregate per-scene detection counts to weekly frequency.

    Parameters
    ----------
    daily_counts : DataFrame with 'timestamp' and value_column.
    value_column : Column to aggregate (e.g., 'vessel_count').
    day_of_week : Which day starts the week (for pd.Grouper freq label).

    Returns
    -------
    DataFrame with columns: [week, mean, max, n_scenes].
    Weeks with 0 observations are not included (will appear as gaps).
    """
    if daily_counts.empty:
        return pd.DataFrame(columns=["week", "mean", "max", "n_scenes"])

    df = daily_counts.copy()
    df = df.set_index("timestamp")

    # Map day_of_week to pandas offset alias
    dow_map = {
        "monday": "W-MON",
        "tuesday": "W-TUE",
        "wednesday": "W-WED",
        "thursday": "W-THU",
        "friday": "W-FRI",
        "saturday": "W-SAT",
        "sunday": "W-SUN",
    }
    freq = dow_map.get(day_of_week.lower(), "W-MON")

    weekly = df[value_column].resample(freq).agg(["mean", "max", "count"])
    weekly.columns = ["mean", "max", "n_scenes"]
    weekly = weekly[weekly["n_scenes"] > 0]  # Drop empty weeks
    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={"timestamp": "week"})

    return weekly


def build_site_index(
    site: Site,
    detections_dir: Path,
    day_of_week: str = "monday",
    value_column: str = "vessel_count",
) -> pd.DataFrame:
    """Build the weekly index for a single site.

    Parameters
    ----------
    site : Site object.
    detections_dir : Root detections directory.
    day_of_week : Week-start day for aggregation.
    value_column : Column to aggregate. Defaults to ``"vessel_count"``;
        pass ``"wind_adjusted_count"`` to use wind-corrected values.
        For airport sites, this is overridden to
        ``"estimated_aircraft_count"`` automatically.
    """
    df = load_detection_summaries(detections_dir, site.id)
    if df.empty:
        return pd.DataFrame()

    if site.site_type == "port":
        value_col = value_column
    else:
        value_col = "estimated_aircraft_count"

    # Fallback: if requested column missing, try default
    if value_col not in df.columns:
        if value_col != "vessel_count" and "vessel_count" in df.columns:
            logger.info(
                "Column %s not found for site %s; falling back to vessel_count",
                value_col,
                site.id,
            )
            value_col = "vessel_count"
        else:
            logger.warning(
                "Column %s not found for site %s. Available: %s",
                value_col,
                site.id,
                list(df.columns),
            )
            return pd.DataFrame()

    weekly = aggregate_weekly(df, value_col, day_of_week)
    weekly["site_id"] = site.id
    weekly["site_type"] = site.site_type
    return weekly


def build_all_indices(
    sites: list[Site],
    detections_dir: Path,
    day_of_week: str = "monday",
) -> dict[str, pd.DataFrame]:
    """Build weekly indices for all sites.

    Returns {site_id: weekly_df}.
    """
    indices = {}
    for site in sites:
        weekly = build_site_index(site, detections_dir, day_of_week)
        if not weekly.empty:
            indices[site.id] = weekly
            logger.info(
                "Built index for %s: %d weeks", site.id, len(weekly)
            )
    return indices
