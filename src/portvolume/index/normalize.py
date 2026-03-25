"""Index normalization and composite index construction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portvolume.sites import Site

logger = logging.getLogger(__name__)


def zscore_normalize(
    series: pd.Series,
    rolling_window: int = 52,
) -> pd.Series:
    """Z-score normalization using a trailing rolling window.

    z = (x - rolling_mean) / rolling_std

    Removes seasonality and level differences between sites,
    producing a comparable "activity anomaly" measure.

    For the first `rolling_window` observations, uses an expanding window.
    """
    rolling_mean = series.rolling(window=rolling_window, min_periods=4).mean()
    rolling_std = series.rolling(window=rolling_window, min_periods=4).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z = (series - rolling_mean) / rolling_std
    return z


def minmax_normalize(
    series: pd.Series,
    rolling_window: int = 52,
) -> pd.Series:
    """Min-max normalization using a trailing rolling window.

    Scales values to [0, 1] range within the window.
    """
    rolling_min = series.rolling(window=rolling_window, min_periods=4).min()
    rolling_max = series.rolling(window=rolling_window, min_periods=4).max()
    denom = rolling_max - rolling_min
    denom = denom.replace(0, np.nan)
    return (series - rolling_min) / denom


def normalize_series(
    series: pd.Series,
    method: str = "zscore",
    rolling_window: int = 52,
) -> pd.Series:
    """Normalize a series using the specified method."""
    if method == "zscore":
        return zscore_normalize(series, rolling_window)
    elif method == "minmax":
        return minmax_normalize(series, rolling_window)
    elif method == "raw":
        return series
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")


def build_composite_index(
    site_indices: dict[str, pd.DataFrame],
    sites: list[Site],
    normalization: str = "zscore",
    rolling_window: int = 52,
    weight_by_throughput: bool = False,
) -> pd.DataFrame:
    """Combine individual site indices into a single global composite.

    Parameters
    ----------
    site_indices : {site_id: weekly DataFrame with 'week' and 'mean' columns}.
    sites : List of Site objects (for throughput weights).
    normalization : "zscore", "minmax", or "raw".
    rolling_window : Window for normalization.
    weight_by_throughput : If True, weight ports by annual TEU.

    Returns
    -------
    DataFrame with columns: [week, composite_index, n_sites].
    """
    if not site_indices:
        return pd.DataFrame(columns=["week", "composite_index", "n_sites"])

    site_lookup = {s.id: s for s in sites}
    normalized_series: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}

    for site_id, df in site_indices.items():
        if df.empty:
            continue

        s = df.set_index("week")["mean"]
        normalized = normalize_series(s, normalization, rolling_window)
        normalized_series[site_id] = normalized

        if weight_by_throughput and site_id in site_lookup:
            w = site_lookup[site_id].annual_teu_millions
            weights[site_id] = w if w > 0 else 1.0
        else:
            weights[site_id] = 1.0

    if not normalized_series:
        return pd.DataFrame(columns=["week", "composite_index", "n_sites"])

    # Align all series on a common date index
    combined = pd.DataFrame(normalized_series)

    # Weighted average (ignoring NaN)
    weight_arr = np.array([weights.get(col, 1.0) for col in combined.columns])
    weight_arr = weight_arr / weight_arr.sum()

    # Manual weighted nanmean
    values = combined.values
    mask = ~np.isnan(values)
    weighted_sum = np.nansum(values * weight_arr[np.newaxis, :], axis=1)
    weight_sum = (mask * weight_arr[np.newaxis, :]).sum(axis=1)
    weight_sum[weight_sum == 0] = np.nan
    composite = weighted_sum / weight_sum

    result = pd.DataFrame(
        {
            "week": combined.index,
            "composite_index": composite,
            "n_sites": mask.sum(axis=1),
        }
    )
    return result.reset_index(drop=True)


def build_regional_indices(
    site_indices: dict[str, pd.DataFrame],
    sites: list[Site],
    normalization: str = "zscore",
    rolling_window: int = 52,
) -> dict[str, pd.DataFrame]:
    """Build sub-indices grouped by region.

    Returns {region: composite DataFrame}.
    """
    site_lookup = {s.id: s for s in sites}

    # Group sites by region
    region_sites: dict[str, list[str]] = {}
    for site_id in site_indices:
        if site_id in site_lookup:
            region = site_lookup[site_id].region or "unknown"
            region_sites.setdefault(region, []).append(site_id)

    regional: dict[str, pd.DataFrame] = {}
    for region, ids in region_sites.items():
        sub_indices = {sid: site_indices[sid] for sid in ids if sid in site_indices}
        sub_sites = [site_lookup[sid] for sid in ids if sid in site_lookup]
        composite = build_composite_index(
            sub_indices, sub_sites, normalization, rolling_window
        )
        if not composite.empty:
            regional[region] = composite
            logger.info("Built %s index: %d weeks", region, len(composite))

    return regional
