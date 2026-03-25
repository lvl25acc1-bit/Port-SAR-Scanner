"""Merge SAR activity index with economic data on aligned weekly dates."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def align_sar_and_economic(
    sar_index: pd.DataFrame,
    economic_df: pd.DataFrame,
    sar_date_col: str = "week",
    sar_value_col: str = "composite_index",
    tolerance: pd.Timedelta | None = None,
    site=None,
) -> pd.DataFrame:
    """Merge SAR activity index with economic data on weekly dates.

    Uses pd.merge_asof to align series that may not share exact dates.
    Forward-fills economic data that is monthly/quarterly to weekly.

    Parameters
    ----------
    sar_index : DataFrame with 'week' (datetime) and activity index columns.
    economic_df : DataFrame with DatetimeIndex and economic series columns.
    sar_date_col : Name of the date column in sar_index.
    sar_value_col : Name of the SAR index value column.
    tolerance : Max time difference for merge_asof. Default 7 days.
    site : Optional Site object. When provided with a primary_commodity attribute,
        the economic DataFrame is filtered to only include columns matching that
        commodity's indicators before merging.

    Returns
    -------
    Single DataFrame with SAR index and all economic columns,
    aligned on a common weekly DatetimeIndex.
    """
    if tolerance is None:
        tolerance = pd.Timedelta("7 days")

    # Filter economic data to commodity-specific indicators if site is provided
    if site is not None:
        primary_commodity = getattr(site, "primary_commodity", "") or ""
        if primary_commodity:
            from portvolume.economic.commodity_indicators import get_indicators_for_site

            site_indicators = get_indicators_for_site(site)
            indicator_ids = set()
            for ind in site_indicators:
                if "id" in ind:
                    indicator_ids.add(ind["id"])
                if "ticker" in ind:
                    indicator_ids.add(ind["ticker"])

            # Filter to only matching columns (keep all if none match)
            matching_cols = [c for c in economic_df.columns if c in indicator_ids]
            if matching_cols:
                economic_df = economic_df[matching_cols]
                logger.info(
                    "Filtered economic data to %d commodity-specific columns for %s",
                    len(matching_cols),
                    primary_commodity,
                )

    # Prepare SAR index
    sar = sar_index[[sar_date_col, sar_value_col]].copy()
    sar[sar_date_col] = pd.to_datetime(sar[sar_date_col])
    sar = sar.sort_values(sar_date_col).reset_index(drop=True)

    # Also include n_sites if available
    if "n_sites" in sar_index.columns:
        sar["n_sites"] = sar_index["n_sites"].values

    # Prepare economic data
    econ = economic_df.copy()
    if not isinstance(econ.index, pd.DatetimeIndex):
        econ.index = pd.to_datetime(econ.index)
    econ = econ.sort_index()

    # Forward-fill to handle monthly data gaps
    econ = econ.resample("D").ffill().resample("W-FRI").last()
    econ = econ.reset_index().rename(columns={"date": "econ_date", "index": "econ_date"})

    # Rename the first column if it's not already 'econ_date'
    if "econ_date" not in econ.columns:
        econ = econ.rename(columns={econ.columns[0]: "econ_date"})

    # Merge using merge_asof
    merged = pd.merge_asof(
        sar,
        econ,
        left_on=sar_date_col,
        right_on="econ_date",
        tolerance=tolerance,
        direction="nearest",
    )

    # Drop the redundant economic date column
    if "econ_date" in merged.columns:
        merged = merged.drop(columns=["econ_date"])

    logger.info(
        "Merged SAR index (%d rows) with economic data (%d cols): %d rows",
        len(sar),
        len(economic_df.columns),
        len(merged),
    )

    return merged
