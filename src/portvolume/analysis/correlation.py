"""Correlation analysis: Pearson/Spearman and lead-lag cross-correlation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def compute_correlations(
    df: pd.DataFrame,
    sar_column: str,
    econ_columns: list[str],
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations between SAR index and economic series.

    Parameters
    ----------
    df : Merged DataFrame with SAR and economic columns.
    sar_column : Name of the SAR index column.
    econ_columns : Names of economic data columns.

    Returns
    -------
    DataFrame: [econ_series, pearson_r, pearson_p, spearman_r, spearman_p, n_obs].
    """
    records = []
    sar = df[sar_column]

    for col in econ_columns:
        if col not in df.columns:
            continue

        # Drop rows where either is NaN
        mask = sar.notna() & df[col].notna()
        x = sar[mask].values
        y = df[col][mask].values

        if len(x) < 10:
            continue

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        records.append(
            {
                "econ_series": col,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_obs": len(x),
            }
        )

    result_df = pd.DataFrame(records)

    # Apply Benjamini-Hochberg correction for multiple comparisons
    if len(result_df) > 0:
        for p_col in ("pearson_p", "spearman_p"):
            if p_col in result_df.columns:
                _, pvals_corrected, _, _ = multipletests(
                    result_df[p_col].values, alpha=0.05, method="fdr_bh"
                )
                result_df[f"adjusted_{p_col}"] = pvals_corrected

    return result_df


def lead_lag_crosscorrelation(
    sar_series: pd.Series,
    econ_series: pd.Series,
    max_lag_weeks: int = 12,
) -> pd.DataFrame:
    """Compute cross-correlation at lags from -max_lag to +max_lag weeks.

    Convention:
    - Positive lag = SAR leads economic data (SAR is predictive).
    - Negative lag = economic data leads SAR (SAR is lagging).

    Parameters
    ----------
    sar_series : Weekly SAR index values (DatetimeIndex).
    econ_series : Weekly economic series values (DatetimeIndex).
    max_lag_weeks : Maximum lag to test in each direction.

    Returns
    -------
    DataFrame: [lag_weeks, correlation, p_value, is_best].
    """
    # Align on common dates
    combined = pd.DataFrame({"sar": sar_series, "econ": econ_series}).dropna()

    if len(combined) < 20:
        logger.warning(
            "Insufficient data for lead-lag analysis: %d rows", len(combined)
        )
        return pd.DataFrame(
            columns=["lag_weeks", "correlation", "p_value", "is_best"]
        )

    sar_vals = combined["sar"].values
    econ_vals = combined["econ"].values
    n = len(sar_vals)

    records = []
    for lag in range(-max_lag_weeks, max_lag_weeks + 1):
        if lag > 0:
            # SAR leads: compare SAR[:-lag] with econ[lag:]
            x = sar_vals[: n - lag]
            y = econ_vals[lag:]
        elif lag < 0:
            # Econ leads: compare SAR[-lag:] with econ[:lag]
            x = sar_vals[-lag:]
            y = econ_vals[: n + lag]
        else:
            x = sar_vals
            y = econ_vals

        if len(x) < 10:
            continue

        r, p = stats.pearsonr(x, y)
        records.append(
            {
                "lag_weeks": lag,
                "correlation": r,
                "p_value": p,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        best_idx = result["correlation"].abs().idxmax()
        result["is_best"] = False
        result.loc[best_idx, "is_best"] = True

    return result
