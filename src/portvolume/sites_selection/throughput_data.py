"""Port throughput data retrieval and signal-quality ranking."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from portvolume.sites_v2 import CommoditySite, compute_site_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded approximate monthly throughput data (millions of tonnes or
# million barrels, depending on commodity).  Real API integration is a
# future enhancement.
# ---------------------------------------------------------------------------
_THROUGHPUT_DATA: dict[str, list[dict[str, Any]]] = {
    "ras_tanura": [
        {"date": "2024-01-01", "throughput": 6.2, "unit": "million_barrels"},
        {"date": "2024-02-01", "throughput": 5.8, "unit": "million_barrels"},
        {"date": "2024-03-01", "throughput": 6.5, "unit": "million_barrels"},
        {"date": "2024-04-01", "throughput": 7.1, "unit": "million_barrels"},
        {"date": "2024-05-01", "throughput": 6.9, "unit": "million_barrels"},
        {"date": "2024-06-01", "throughput": 5.5, "unit": "million_barrels"},
        {"date": "2024-07-01", "throughput": 7.3, "unit": "million_barrels"},
        {"date": "2024-08-01", "throughput": 6.0, "unit": "million_barrels"},
        {"date": "2024-09-01", "throughput": 6.8, "unit": "million_barrels"},
        {"date": "2024-10-01", "throughput": 7.0, "unit": "million_barrels"},
        {"date": "2024-11-01", "throughput": 5.9, "unit": "million_barrels"},
        {"date": "2024-12-01", "throughput": 6.4, "unit": "million_barrels"},
    ],
    "port_hedland": [
        {"date": "2024-01-01", "throughput": 45.0, "unit": "million_tonnes"},
        {"date": "2024-02-01", "throughput": 40.0, "unit": "million_tonnes"},
        {"date": "2024-03-01", "throughput": 48.0, "unit": "million_tonnes"},
        {"date": "2024-04-01", "throughput": 46.0, "unit": "million_tonnes"},
        {"date": "2024-05-01", "throughput": 44.0, "unit": "million_tonnes"},
        {"date": "2024-06-01", "throughput": 42.0, "unit": "million_tonnes"},
        {"date": "2024-07-01", "throughput": 47.0, "unit": "million_tonnes"},
        {"date": "2024-08-01", "throughput": 43.0, "unit": "million_tonnes"},
        {"date": "2024-09-01", "throughput": 46.5, "unit": "million_tonnes"},
        {"date": "2024-10-01", "throughput": 44.5, "unit": "million_tonnes"},
        {"date": "2024-11-01", "throughput": 41.0, "unit": "million_tonnes"},
        {"date": "2024-12-01", "throughput": 45.5, "unit": "million_tonnes"},
    ],
    "santos": [
        {"date": "2024-01-01", "throughput": 8.0, "unit": "million_tonnes"},
        {"date": "2024-02-01", "throughput": 9.5, "unit": "million_tonnes"},
        {"date": "2024-03-01", "throughput": 12.0, "unit": "million_tonnes"},
        {"date": "2024-04-01", "throughput": 13.5, "unit": "million_tonnes"},
        {"date": "2024-05-01", "throughput": 14.0, "unit": "million_tonnes"},
        {"date": "2024-06-01", "throughput": 11.0, "unit": "million_tonnes"},
        {"date": "2024-07-01", "throughput": 9.0, "unit": "million_tonnes"},
        {"date": "2024-08-01", "throughput": 7.5, "unit": "million_tonnes"},
        {"date": "2024-09-01", "throughput": 8.5, "unit": "million_tonnes"},
        {"date": "2024-10-01", "throughput": 10.0, "unit": "million_tonnes"},
        {"date": "2024-11-01", "throughput": 11.5, "unit": "million_tonnes"},
        {"date": "2024-12-01", "throughput": 12.5, "unit": "million_tonnes"},
    ],
    "rotterdam": [
        {"date": "2024-01-01", "throughput": 38.0, "unit": "million_tonnes"},
        {"date": "2024-02-01", "throughput": 37.5, "unit": "million_tonnes"},
        {"date": "2024-03-01", "throughput": 39.0, "unit": "million_tonnes"},
        {"date": "2024-04-01", "throughput": 38.5, "unit": "million_tonnes"},
        {"date": "2024-05-01", "throughput": 39.5, "unit": "million_tonnes"},
        {"date": "2024-06-01", "throughput": 37.0, "unit": "million_tonnes"},
        {"date": "2024-07-01", "throughput": 38.0, "unit": "million_tonnes"},
        {"date": "2024-08-01", "throughput": 38.5, "unit": "million_tonnes"},
        {"date": "2024-09-01", "throughput": 39.0, "unit": "million_tonnes"},
        {"date": "2024-10-01", "throughput": 38.0, "unit": "million_tonnes"},
        {"date": "2024-11-01", "throughput": 37.5, "unit": "million_tonnes"},
        {"date": "2024-12-01", "throughput": 38.5, "unit": "million_tonnes"},
    ],
}


def fetch_port_throughput(
    port_name: str,
    source: str = "manual",
) -> pd.DataFrame:
    """Fetch monthly throughput data for a port.

    Parameters
    ----------
    port_name : str
        Port identifier (e.g. ``"ras_tanura"``).
    source : str
        Data source.  Currently only ``"manual"`` (hardcoded) is supported.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, throughput, unit]``.
    """
    if source != "manual":
        logger.warning("Only 'manual' source is currently supported; falling back.")

    records = _THROUGHPUT_DATA.get(port_name, [])
    if not records:
        logger.warning("No throughput data for port %r", port_name)
        return pd.DataFrame(columns=["date", "throughput", "unit"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_throughput_variance(
    throughput_df: pd.DataFrame,
    window_months: int = 24,
) -> float:
    """Compute coefficient of variation (std / mean) over a rolling window.

    Parameters
    ----------
    throughput_df : pd.DataFrame
        Must contain a ``throughput`` column.
    window_months : int
        Maximum number of months to consider (most recent).

    Returns
    -------
    float
        Coefficient of variation as a fraction (e.g. 0.18 for 18%).
    """
    if throughput_df.empty:
        return 0.0

    series = throughput_df["throughput"].tail(window_months)
    mean = series.mean()
    if mean == 0:
        return 0.0
    return float(series.std(ddof=0) / mean)


def rank_ports_by_signal_potential(
    ports: list[dict[str, Any]],
) -> pd.DataFrame:
    """Rank ports using :func:`compute_site_score`.

    Parameters
    ----------
    ports : list[dict]
        Each dict must have keys matching :class:`CommoditySite` fields.

    Returns
    -------
    pd.DataFrame
        Columns: ``[port_id, score, primary_commodity, throughput_variance_pct]``,
        sorted by ``score`` descending.
    """
    rows: list[dict[str, Any]] = []
    for p in ports:
        site = CommoditySite(
            id=p["id"],
            name=p.get("name", p["id"]),
            lat=p.get("lat", 0.0),
            lon=p.get("lon", 0.0),
            bbox=tuple(p.get("bbox", (0, 0, 0, 0))),
            site_type=p.get("site_type", "port"),
            primary_commodity=p.get("primary_commodity", ""),
            commodity_group=p.get("commodity_group", ""),
            throughput_variance_pct=p.get("throughput_variance_pct", 0.0),
            anchorage_bbox=tuple(p["anchorage_bbox"]) if p.get("anchorage_bbox") else None,
            economic_indicators=tuple(p.get("economic_indicators", ())),
        )
        rows.append(
            {
                "port_id": site.id,
                "score": compute_site_score(site),
                "primary_commodity": site.primary_commodity,
                "throughput_variance_pct": site.throughput_variance_pct,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
