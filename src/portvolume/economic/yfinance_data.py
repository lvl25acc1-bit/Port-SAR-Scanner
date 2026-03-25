"""Yahoo Finance data fetching for commodities and shipping indices."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from portvolume.config import YFinanceTicker

logger = logging.getLogger(__name__)


def fetch_yfinance_series(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.Series:
    """Fetch daily close prices for a ticker and resample to weekly."""
    logger.info("Fetching yfinance ticker %s...", ticker)
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        logger.warning("No data returned for ticker %s", ticker)
        return pd.Series(dtype=float, name=ticker)

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Resample to weekly (Friday close)
    weekly = close.resample("W-FRI").last().dropna()
    weekly.name = ticker
    return weekly


def fetch_all_yfinance(
    tickers_config: list[YFinanceTicker],
    start_date: str,
    end_date: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch all configured tickers and merge into a single DataFrame.

    Caches as parquet.
    """
    cache_path = None
    if cache_dir:
        cache_path = cache_dir / "yfinance_data.parquet"
        if cache_path.exists():
            logger.info("Loading cached yfinance data from %s", cache_path)
            return pd.read_parquet(cache_path)

    all_series = {}
    for t in tickers_config:
        try:
            series = fetch_yfinance_series(t.ticker, start_date, end_date)
            if not series.empty:
                all_series[t.ticker] = series
        except Exception:
            logger.exception("Failed to fetch ticker %s", t.ticker)

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    df.index.name = "date"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached yfinance data to %s", cache_path)

    return df
