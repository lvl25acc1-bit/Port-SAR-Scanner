"""Commodity-specific economic indicator mappings and fetching."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from portvolume.economic.fred import fetch_fred_series, get_fred_client, resample_to_weekly
from portvolume.economic.yfinance_data import fetch_yfinance_series

logger = logging.getLogger(__name__)

COMMODITY_INDICATORS: dict[str, list[dict]] = {
    "crude_oil": [
        {"source": "fred", "id": "DCOILWTICO", "name": "WTI Crude Oil"},
        {"source": "yfinance", "ticker": "BZ=F", "name": "Brent Crude Futures"},
        {"source": "yfinance", "ticker": "CL=F", "name": "WTI Crude Futures"},
    ],
    "iron_ore": [
        {"source": "fred", "id": "PIORECRUSDM", "name": "Iron Ore Price Index"},
        {"source": "yfinance", "ticker": "BHP", "name": "BHP Group (iron ore proxy)"},
    ],
    "grain": [
        {"source": "yfinance", "ticker": "ZS=F", "name": "Soybean Futures"},
        {"source": "yfinance", "ticker": "ZW=F", "name": "Wheat Futures"},
        {"source": "yfinance", "ticker": "ZC=F", "name": "Corn Futures"},
    ],
    "coal": [
        {"source": "fred", "id": "PCOALAUUSDM", "name": "Coal Price Australia"},
    ],
    "lng": [
        {"source": "fred", "id": "PNGASUSUSDM", "name": "US Natural Gas Price"},
        {"source": "yfinance", "ticker": "NG=F", "name": "Natural Gas Futures"},
    ],
    "containers": [
        {"source": "yfinance", "ticker": "BDRY", "name": "Dry Bulk Index ETF"},
        {"source": "fred", "id": "BOPGSTB", "name": "US Trade Balance"},
        {"source": "fred", "id": "INDPRO", "name": "Industrial Production"},
    ],
}


def get_indicators_for_commodity(commodity: str) -> list[dict]:
    """Return indicators for a commodity group. Falls back to containers if unknown."""
    return COMMODITY_INDICATORS.get(commodity, COMMODITY_INDICATORS["containers"])


def get_indicators_for_site(site) -> list[dict]:
    """Return indicators based on site.primary_commodity.

    If site has economic_indicators field, filter to only those IDs.
    """
    commodity = getattr(site, "primary_commodity", "") or ""
    indicators = get_indicators_for_commodity(commodity)

    # If site specifies explicit indicator IDs, filter to those
    economic_ids = getattr(site, "economic_indicators", ()) or ()
    if economic_ids:
        filtered = []
        for ind in indicators:
            ind_id = ind.get("id") or ind.get("ticker", "")
            if ind_id in economic_ids:
                filtered.append(ind)
        # Return filtered if any matched, otherwise return all
        if filtered:
            return filtered

    return indicators


def fetch_commodity_indicators(
    commodity: str,
    start_date: str,
    fred_api_key: str = "",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch all indicators for a commodity, merge to weekly frequency.

    Uses existing fred.py and yfinance_data.py fetch functions.

    Parameters
    ----------
    commodity : Commodity group key (e.g. "crude_oil", "grain").
    start_date : Start date string for data fetch.
    fred_api_key : FRED API key. Required for FRED series.
    cache_dir : Optional cache directory for storing fetched data.

    Returns
    -------
    DataFrame with weekly DatetimeIndex and one column per indicator.
    """
    indicators = get_indicators_for_commodity(commodity)
    all_series: dict[str, pd.Series] = {}

    fred_client = None

    for ind in indicators:
        try:
            if ind["source"] == "fred":
                if not fred_api_key:
                    logger.warning(
                        "Skipping FRED series %s: no API key provided", ind["id"]
                    )
                    continue
                if fred_client is None:
                    fred_client = get_fred_client(fred_api_key)
                raw = fetch_fred_series(fred_client, ind["id"], start_date)
                # Determine frequency heuristic: daily if > 200 obs/year
                freq = "daily" if len(raw) > 200 else "monthly"
                weekly = resample_to_weekly(raw, freq)
                all_series[ind["id"]] = weekly

            elif ind["source"] == "yfinance":
                weekly = fetch_yfinance_series(ind["ticker"], start_date)
                if not weekly.empty:
                    all_series[ind["ticker"]] = weekly

        except Exception:
            label = ind.get("id") or ind.get("ticker", "unknown")
            logger.exception("Failed to fetch indicator %s", label)

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    df.index.name = "date"

    if cache_dir:
        cache_path = cache_dir / f"commodity_{commodity}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached commodity indicators to %s", cache_path)

    return df
