"""FRED API data fetching and caching."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred

from portvolume.config import FREDSeries

logger = logging.getLogger(__name__)


def get_fred_client(api_key: str) -> Fred:
    """Initialize FRED API client."""
    if not api_key:
        raise ValueError(
            "FRED API key not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set the FRED_API_KEY environment variable."
        )
    return Fred(api_key=api_key)


def fetch_fred_series(
    client: Fred,
    series_id: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.Series:
    """Fetch a single FRED series.

    Returns pd.Series with DatetimeIndex.
    """
    kwargs = {"observation_start": start_date}
    if end_date:
        kwargs["observation_end"] = end_date

    data = client.get_series(series_id, **kwargs)
    data.name = series_id
    data.index = pd.to_datetime(data.index)
    return data


def resample_to_weekly(
    series: pd.Series,
    frequency: str,
) -> pd.Series:
    """Resample a series to weekly frequency.

    Monthly/quarterly data is forward-filled to weekly.
    Daily data is resampled to weekly (Friday close).
    """
    if frequency == "weekly":
        return series
    elif frequency == "daily":
        return series.resample("W-FRI").last()
    else:
        # Monthly, quarterly, etc: forward-fill to daily, then resample weekly
        daily = series.resample("D").ffill()
        return daily.resample("W-FRI").last()


def fetch_all_fred(
    client: Fred,
    series_config: list[FREDSeries],
    start_date: str,
    end_date: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch all configured FRED series and merge into a single DataFrame.

    Caches results as parquet to avoid repeated API calls.
    Columns: one per series_id, index: weekly DatetimeIndex.
    """
    cache_path = None
    if cache_dir:
        cache_path = cache_dir / "fred_data.parquet"
        if cache_path.exists():
            logger.info("Loading cached FRED data from %s", cache_path)
            return pd.read_parquet(cache_path)

    all_series = {}
    for s in series_config:
        try:
            logger.info("Fetching FRED series %s (%s)...", s.id, s.name)
            raw = fetch_fred_series(client, s.id, start_date, end_date)
            weekly = resample_to_weekly(raw, s.frequency)
            all_series[s.id] = weekly
        except Exception:
            logger.exception("Failed to fetch FRED series %s", s.id)

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    df.index.name = "date"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached FRED data to %s", cache_path)

    return df
