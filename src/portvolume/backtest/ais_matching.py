"""Build AIS scene-level and monthly reference tables from GFW data.

Filters to commercial vessel types, applies size thresholds, and produces
daily and monthly aggregate metrics for backtest comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COMMERCIAL_TYPES = {"CARGO", "CARRIER", "TANKER", "PASSENGER"}


def filter_commercial_vessels(
    presence_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    min_area_m2: float = 600.0,
    min_hours: float = 0.0,
) -> pd.DataFrame:
    """Join presence with identity, filter to commercial types and size/hours threshold.

    Join key: vessel_id first (GFW's native key), MMSI as fallback.
    Size filter: length_m * beam_m >= min_area_m2 (when identity available).
    Hours filter: presence hours >= min_hours (proxy for large berthed vessels
    when identity is unavailable).
    """
    if presence_df.empty:
        return pd.DataFrame()

    # Normalize vessel type to uppercase for matching
    if "vesselType" in presence_df.columns:
        presence_df = presence_df.copy()
        presence_df["vesselType_upper"] = presence_df["vesselType"].str.upper()
    else:
        return pd.DataFrame()

    # Filter to commercial types
    commercial = presence_df[
        presence_df["vesselType_upper"].isin(COMMERCIAL_TYPES)
    ].copy()

    if commercial.empty:
        return pd.DataFrame()

    # Join identity for dimensions (if available)
    has_identity = (
        not identity_df.empty
        and "vesselId" in identity_df.columns
        and "length_m" in identity_df.columns
    )
    if has_identity:
        id_cols = ["vesselId"]
        dim_cols = ["length_m", "beam_m", "gross_tonnage"]
        available = [c for c in dim_cols if c in identity_df.columns]
        commercial = commercial.merge(
            identity_df[id_cols + available].drop_duplicates("vesselId"),
            on="vesselId",
            how="left",
        )

    # Compute hull area and apply size filter
    if has_identity and "length_m" in commercial.columns and "beam_m" in commercial.columns:
        commercial["hull_area_m2"] = (
            commercial["length_m"].fillna(0) * commercial["beam_m"].fillna(0)
        )
        has_dims = commercial["hull_area_m2"] > 0
        passes_size = commercial["hull_area_m2"] >= min_area_m2
        no_dims = ~has_dims
        commercial = commercial[passes_size | no_dims].copy()
        logger.info(
            "Size filter (hull area): %d vessels pass at %.0f m²",
            passes_size.sum(), min_area_m2,
        )
    else:
        commercial["hull_area_m2"] = 0.0

    # Hours filter: proxy for large berthed vessels when no identity
    if min_hours > 0 and "hours" in commercial.columns:
        before = len(commercial)
        commercial = commercial[commercial["hours"] >= min_hours].copy()
        logger.info(
            "Hours filter: %d → %d vessels (min_hours=%.1f)",
            before, len(commercial), min_hours,
        )

    logger.info(
        "Commercial filter: %d → %d vessels (types=%s, min_area=%.0f m², min_hours=%.1f)",
        len(presence_df), len(commercial), COMMERCIAL_TYPES, min_area_m2, min_hours,
    )
    return commercial


def build_daily_counts(
    scene_universe: pd.DataFrame,
    filtered_presence: pd.DataFrame,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build per-scene-date AIS reference metrics.

    For each unique date in the scene universe, compute:
    - gfw_unique_vessels: count of distinct commercial vessels
    - gfw_total_hours: sum of presence hours
    - gfw_size_weighted_hours: sum of hours * hull_area_m2
    - gfw_sum_area_m2: sum of hull_area_m2
    """
    if filtered_presence.empty:
        dates = scene_universe["date"].dt.strftime("%Y-%m-%d").unique()
        empty = pd.DataFrame({
            "query_date": dates,
            "gfw_unique_vessels": 0,
            "gfw_total_hours": 0.0,
            "gfw_size_weighted_hours": 0.0,
            "gfw_sum_area_m2": 0.0,
        })
        return empty

    # Group by date
    grouped = filtered_presence.groupby("query_date").agg(
        gfw_unique_vessels=("vesselId", "nunique"),
        gfw_total_hours=("hours", "sum"),
        gfw_size_weighted_hours=pd.NamedAgg(
            column="hours",
            aggfunc=lambda x: (x * filtered_presence.loc[x.index, "hull_area_m2"]).sum(),
        ),
        gfw_sum_area_m2=("hull_area_m2", "sum"),
    ).reset_index()

    logger.info(
        "Daily AIS counts: %d dates, mean %.0f vessels/day",
        len(grouped), grouped["gfw_unique_vessels"].mean(),
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        grouped.to_parquet(cache_dir / "ais_daily_counts.parquet", index=False)

    return grouped


def build_monthly_counts(
    daily_counts: pd.DataFrame,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Aggregate daily AIS metrics to monthly for headline validation."""
    if daily_counts.empty:
        return pd.DataFrame()

    daily_counts = daily_counts.copy()
    daily_counts["month"] = pd.to_datetime(daily_counts["query_date"]).dt.to_period("M")

    monthly = daily_counts.groupby("month").agg(
        gfw_monthly_unique_vessels=("gfw_unique_vessels", "mean"),
        gfw_monthly_total_hours=("gfw_total_hours", "mean"),
        gfw_monthly_size_weighted_hours=("gfw_size_weighted_hours", "mean"),
        gfw_monthly_sum_area_m2=("gfw_sum_area_m2", "mean"),
        n_days=("query_date", "count"),
    ).reset_index()

    monthly["month"] = monthly["month"].dt.to_timestamp()

    logger.info("Monthly AIS counts: %d months", len(monthly))

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        monthly.to_parquet(cache_dir / "ais_monthly_counts.parquet", index=False)

    return monthly


def sensitivity_sweep(
    presence_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    thresholds: list[float],
    scene_universe: pd.DataFrame,
    hours_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Run filter + daily counts at multiple thresholds.

    When identity data is available, sweeps hull area thresholds.
    When unavailable, sweeps min_hours thresholds as a proxy.
    Returns long-form DataFrame.
    """
    has_identity = (
        not identity_df.empty
        and "vesselId" in identity_df.columns
        and "length_m" in identity_df.columns
    )

    frames = []

    if has_identity:
        # Sweep hull area
        for threshold in thresholds:
            filtered = filter_commercial_vessels(presence_df, identity_df, threshold)
            daily = build_daily_counts(scene_universe, filtered)
            daily["threshold_m2"] = threshold
            daily["threshold_hours"] = 0.0
            frames.append(daily)
    else:
        # Sweep min_hours as proxy
        if hours_thresholds is None:
            hours_thresholds = [0, 2, 4, 6, 8, 12]
        for min_h in hours_thresholds:
            filtered = filter_commercial_vessels(
                presence_df, identity_df, min_area_m2=0, min_hours=min_h
            )
            daily = build_daily_counts(scene_universe, filtered)
            daily["threshold_m2"] = 0
            daily["threshold_hours"] = min_h
            frames.append(daily)

    result = pd.concat(frames, ignore_index=True)
    logger.info("Sensitivity sweep: %d rows", len(result))
    return result
