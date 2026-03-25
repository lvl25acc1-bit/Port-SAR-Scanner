"""Enhanced correlation analysis with stationarity, cointegration, and multiple comparison correction."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sm

from portvolume.analysis.granger import run_granger_test
from portvolume.analysis.regression import out_of_sample_test

logger = logging.getLogger(__name__)


def test_stationarity(
    series: pd.Series,
    method: str = "both",
) -> dict:
    """Run ADF and/or KPSS stationarity tests.

    Parameters
    ----------
    series : Time series to test.
    method : "adf", "kpss", or "both".

    Returns
    -------
    dict with keys: is_stationary, adf_stat, adf_pvalue, kpss_stat, kpss_pvalue,
                    recommendation ("stationary", "difference", "trend_stationary").
    """
    result: dict = {
        "is_stationary": False,
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "kpss_stat": np.nan,
        "kpss_pvalue": np.nan,
        "recommendation": "difference",
    }

    # Clean the series
    clean = series.dropna()
    if len(clean) < 20:
        logger.warning("Series too short for stationarity test: %d observations", len(clean))
        result["recommendation"] = "insufficient_data"
        return result

    # Check for constant series
    if clean.std() == 0:
        logger.warning("Constant series detected; treating as stationary")
        result["is_stationary"] = True
        result["recommendation"] = "stationary"
        return result

    adf_stationary = None
    kpss_stationary = None

    if method in ("adf", "both"):
        try:
            adf_result = adfuller(clean.values, autolag="AIC")
            result["adf_stat"] = adf_result[0]
            result["adf_pvalue"] = adf_result[1]
            adf_stationary = adf_result[1] < 0.05
        except Exception:
            logger.exception("ADF test failed")

    if method in ("kpss", "both"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(clean.values, regression="c", nlags="auto")
            result["kpss_stat"] = kpss_result[0]
            result["kpss_pvalue"] = kpss_result[1]
            # KPSS null is stationarity, so reject => non-stationary
            kpss_stationary = kpss_result[1] > 0.05
        except Exception:
            logger.exception("KPSS test failed")

    # Determine recommendation based on both tests
    if method == "both" and adf_stationary is not None and kpss_stationary is not None:
        if adf_stationary and kpss_stationary:
            result["is_stationary"] = True
            result["recommendation"] = "stationary"
        elif adf_stationary and not kpss_stationary:
            result["is_stationary"] = False
            result["recommendation"] = "trend_stationary"
        elif not adf_stationary and kpss_stationary:
            result["is_stationary"] = False
            result["recommendation"] = "difference"
        else:
            result["is_stationary"] = False
            result["recommendation"] = "difference"
    elif adf_stationary is not None:
        result["is_stationary"] = adf_stationary
        result["recommendation"] = "stationary" if adf_stationary else "difference"
    elif kpss_stationary is not None:
        result["is_stationary"] = kpss_stationary
        result["recommendation"] = "stationary" if kpss_stationary else "difference"

    return result


def test_cointegration(
    sar_series: pd.Series,
    econ_series: pd.Series,
    method: str = "engle_granger",
) -> dict:
    """Engle-Granger two-step cointegration test.

    1. Regress sar ~ econ via OLS
    2. Test residuals for stationarity (ADF)

    Parameters
    ----------
    sar_series : SAR index series.
    econ_series : Economic indicator series.
    method : Only "engle_granger" is currently supported.

    Returns
    -------
    dict with keys: is_cointegrated, adf_stat, adf_pvalue, regression_coef, regression_r2.
    """
    result: dict = {
        "is_cointegrated": False,
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "regression_coef": np.nan,
        "regression_r2": np.nan,
    }

    # Align and clean
    combined = pd.DataFrame({"sar": sar_series, "econ": econ_series}).dropna()
    if len(combined) < 20:
        logger.warning("Insufficient data for cointegration test: %d rows", len(combined))
        return result

    # Check for constant series
    if combined["sar"].std() == 0 or combined["econ"].std() == 0:
        logger.warning("Constant series detected; cannot test cointegration")
        return result

    try:
        coint_stat, pvalue, crit_values = coint(
            combined["sar"].values, combined["econ"].values
        )
        result["adf_stat"] = coint_stat
        result["adf_pvalue"] = pvalue
        result["is_cointegrated"] = pvalue < 0.05

        # Also fit OLS for coefficient and R-squared
        X = sm.add_constant(combined["econ"].values)
        model = sm.OLS(combined["sar"].values, X).fit()
        result["regression_coef"] = model.params[1]
        result["regression_r2"] = model.rsquared

    except Exception:
        logger.exception("Cointegration test failed")

    return result


def compute_correlations_with_correction(
    df: pd.DataFrame,
    sar_column: str,
    econ_columns: list[str],
    method: str = "benjamini_hochberg",
) -> pd.DataFrame:
    """Compute Spearman correlations with multiple hypothesis correction.

    Parameters
    ----------
    df : Merged DataFrame with SAR and economic columns.
    sar_column : Name of the SAR index column.
    econ_columns : Names of economic data columns.
    method : Multiple testing correction method.
        "benjamini_hochberg" maps to statsmodels "fdr_bh".
        "bonferroni" maps to statsmodels "bonferroni".

    Returns
    -------
    DataFrame with columns: [indicator, rho, p_value, adjusted_p_value, significant].
    """
    method_map = {
        "benjamini_hochberg": "fdr_bh",
        "bonferroni": "bonferroni",
    }
    sm_method = method_map.get(method, method)

    records = []
    sar = df[sar_column]

    for col in econ_columns:
        if col not in df.columns:
            continue

        mask = sar.notna() & df[col].notna()
        x = sar[mask].values
        y = df[col][mask].values

        if len(x) < 10:
            continue

        rho, p_value = stats.spearmanr(x, y)
        records.append(
            {
                "indicator": col,
                "rho": rho,
                "p_value": p_value,
            }
        )

    if not records:
        return pd.DataFrame(columns=["indicator", "rho", "p_value", "adjusted_p_value", "significant"])

    result_df = pd.DataFrame(records)

    # Apply multiple testing correction
    reject, pvals_corrected, _, _ = multipletests(
        result_df["p_value"].values, alpha=0.05, method=sm_method
    )
    result_df["adjusted_p_value"] = pvals_corrected
    result_df["significant"] = reject

    return result_df


def optimal_lag_selection(
    sar_series: pd.Series,
    econ_series: pd.Series,
    max_lag: int = 16,
    criterion: str = "bic",
) -> dict:
    """Select optimal lag using information criteria.

    For each lag 0..max_lag: compute cross-correlation and BIC of VAR model.

    Parameters
    ----------
    sar_series : Weekly SAR index values (DatetimeIndex).
    econ_series : Weekly economic series values (DatetimeIndex).
    max_lag : Maximum lag to test.
    criterion : Information criterion ("bic" or "aic").

    Returns
    -------
    dict with keys: optimal_lag, criterion_value, all_lags (list of dicts).
    """
    result: dict = {
        "optimal_lag": 0,
        "criterion_value": np.inf,
        "all_lags": [],
    }

    combined = pd.DataFrame({"sar": sar_series, "econ": econ_series}).dropna()

    if len(combined) < max_lag + 10:
        logger.warning(
            "Insufficient data for lag selection: %d rows", len(combined)
        )
        return result

    # Check for constant series
    if combined["sar"].std() == 0 or combined["econ"].std() == 0:
        logger.warning("Constant series; cannot select optimal lag")
        return result

    try:
        model = VAR(combined.values)
        # Use at least lag 1 for VAR
        effective_max = min(max_lag, len(combined) // 3)
        if effective_max < 1:
            return result

        lag_order = model.select_order(maxlags=effective_max)

        # Extract the selected lag from the summary table
        # lag_order has .bic, .aic etc as the selected lag (int) and
        # .summary() has the full table. Use fitted models per lag instead.
        # The select_order result stores per-lag IC values in ics dict.
        # Access via the underlying _ics attribute or iterate.
        all_lags = []
        best_lag = 0
        best_value = np.inf

        for lag_val in range(1, effective_max + 1):
            try:
                fitted = model.fit(lag_val)
                if criterion == "bic":
                    val = fitted.bic
                else:
                    val = fitted.aic
                all_lags.append({"lag": lag_val, "criterion_value": val})
                if val < best_value:
                    best_value = val
                    best_lag = lag_val
            except Exception:
                all_lags.append({"lag": lag_val, "criterion_value": np.inf})

        # Also add cross-correlation based lag as supplementary info
        sar_vals = combined["sar"].values
        econ_vals = combined["econ"].values
        n = len(sar_vals)

        for entry in all_lags:
            lag_val = entry["lag"]
            if lag_val < n:
                x = sar_vals[lag_val:]
                y = econ_vals[: n - lag_val]
                if len(x) >= 10:
                    r, _ = stats.pearsonr(x, y)
                    entry["cross_correlation"] = r

        result["optimal_lag"] = best_lag
        result["criterion_value"] = best_value
        result["all_lags"] = all_lags

    except Exception:
        logger.exception("Optimal lag selection failed")

    return result


def commodity_specific_analysis(
    site_index: pd.DataFrame,
    site,
    indicators: pd.DataFrame,
    max_lag_weeks: int = 16,
) -> dict:
    """Full analysis pipeline for one commodity-specific site.

    Steps:
    1. Test stationarity of SAR index and each indicator
    2. Difference non-stationary series
    3. Test cointegration for each SAR-indicator pair
    4. Compute correlations with BH correction
    5. Find optimal lag via BIC for top indicators
    6. Run Granger causality at optimal lag
    7. Run out-of-sample forecast evaluation

    Parameters
    ----------
    site_index : DataFrame with 'week' and 'composite_index' columns.
    site : Site object with primary_commodity attribute.
    indicators : DataFrame with weekly economic indicator columns (DatetimeIndex).
    max_lag_weeks : Maximum lag to test.

    Returns
    -------
    dict with all analysis results.
    """
    from portvolume.economic.merge import align_sar_and_economic

    results: dict = {
        "site_id": getattr(site, "id", "unknown"),
        "commodity": getattr(site, "primary_commodity", "unknown"),
        "stationarity": {},
        "cointegration": {},
        "correlations": pd.DataFrame(),
        "optimal_lags": {},
        "granger": {},
        "out_of_sample": {},
    }

    # Merge SAR index with indicators
    merged = align_sar_and_economic(site_index, indicators)
    econ_cols = [
        c for c in merged.columns
        if c not in ("week", "composite_index", "n_sites")
    ]

    if not econ_cols:
        logger.warning("No economic columns after merge for site %s", results["site_id"])
        return results

    sar_col = "composite_index"

    # Step 1: Stationarity tests
    sar_series = merged.set_index("week")[sar_col]
    results["stationarity"][sar_col] = test_stationarity(sar_series)

    for col in econ_cols:
        econ_s = merged.set_index("week")[col]
        results["stationarity"][col] = test_stationarity(econ_s)

    # Step 2: Difference non-stationary series for correlation analysis
    analysis_df = merged.copy()
    diff_sar_col = sar_col
    if not results["stationarity"][sar_col]["is_stationary"]:
        analysis_df[f"{sar_col}_diff"] = analysis_df[sar_col].diff()
        diff_sar_col = f"{sar_col}_diff"

    diff_econ_cols = []
    for col in econ_cols:
        if not results["stationarity"][col]["is_stationary"]:
            analysis_df[f"{col}_diff"] = analysis_df[col].diff()
            diff_econ_cols.append(f"{col}_diff")
        else:
            diff_econ_cols.append(col)

    # Drop NaN rows introduced by differencing
    analysis_df = analysis_df.dropna(subset=[diff_sar_col] + diff_econ_cols)

    # Step 3: Cointegration tests (on original levels, not differenced)
    for col in econ_cols:
        sar_s = merged.set_index("week")[sar_col]
        econ_s = merged.set_index("week")[col]
        results["cointegration"][col] = test_cointegration(sar_s, econ_s)

    # Step 4: Correlations with BH correction
    if len(analysis_df) >= 10 and diff_econ_cols:
        results["correlations"] = compute_correlations_with_correction(
            analysis_df, diff_sar_col, diff_econ_cols
        )

    # Step 5: Optimal lag for top indicators
    if not results["correlations"].empty:
        top_n = min(5, len(results["correlations"]))
        top_indicators = (
            results["correlations"]
            .nlargest(top_n, "rho", keep="first")["indicator"]
            .tolist()
        )
    else:
        top_indicators = diff_econ_cols[:5]

    for col in top_indicators:
        if col in analysis_df.columns:
            sar_s = analysis_df.set_index("week")[diff_sar_col] if "week" in analysis_df.columns else analysis_df[diff_sar_col]
            econ_s = analysis_df.set_index("week")[col] if "week" in analysis_df.columns else analysis_df[col]
            results["optimal_lags"][col] = optimal_lag_selection(
                sar_s, econ_s, max_lag=max_lag_weeks
            )

    # Step 6: Granger causality at optimal lag
    for col in econ_cols:
        sar_s = merged.set_index("week")[sar_col]
        econ_s = merged.set_index("week")[col]
        opt_lag = max_lag_weeks
        diff_col = f"{col}_diff" if f"{col}_diff" in diff_econ_cols else col
        if diff_col in results["optimal_lags"]:
            opt_lag = results["optimal_lags"][diff_col].get("optimal_lag", max_lag_weeks)
            if opt_lag < 1:
                opt_lag = max_lag_weeks

        granger_result = run_granger_test(
            sar_s, econ_s, max_lag=min(opt_lag, max_lag_weeks)
        )
        results["granger"][col] = {
            "sar_causes_econ": granger_result.sar_causes_econ,
            "econ_causes_sar": granger_result.econ_causes_sar,
            "direction": granger_result.best_direction,
            "optimal_lag": granger_result.optimal_lag,
            "p_value": granger_result.p_value,
        }

    # Step 7: Out-of-sample forecast
    if econ_cols and len(merged) >= 50:
        results["out_of_sample"] = out_of_sample_test(
            merged, sar_col, econ_cols[:5], n_splits=5
        )

    return results
