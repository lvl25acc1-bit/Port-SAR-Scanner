"""Granger causality testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


@dataclass
class GrangerResult:
    """Result of a bidirectional Granger causality test."""

    sar_causes_econ: bool
    econ_causes_sar: bool
    optimal_lag: int
    best_direction: str  # "sar->econ", "econ->sar", "bidirectional", "none"
    f_statistic: float
    p_value: float


def run_granger_test(
    sar_series: pd.Series,
    econ_series: pd.Series,
    max_lag: int = 8,
    significance: float = 0.05,
) -> GrangerResult:
    """Test whether the SAR index Granger-causes the economic series and vice versa.

    Parameters
    ----------
    sar_series : Weekly SAR index (DatetimeIndex).
    econ_series : Weekly economic series (DatetimeIndex).
    max_lag : Maximum number of lags to test.
    significance : P-value threshold for significance.

    Returns
    -------
    GrangerResult with test outcomes in both directions.
    """
    # Align and drop NaN
    combined = pd.DataFrame({"sar": sar_series, "econ": econ_series}).dropna()

    if len(combined) < max_lag * 3:
        logger.warning(
            "Insufficient data for Granger test: %d rows (need %d)",
            len(combined),
            max_lag * 3,
        )
        return GrangerResult(
            sar_causes_econ=False,
            econ_causes_sar=False,
            optimal_lag=0,
            best_direction="none",
            f_statistic=0.0,
            p_value=1.0,
        )

    # Test: does SAR Granger-cause econ?
    # grangercausalitytests expects [effect, cause] column order
    sar_to_econ_data = combined[["econ", "sar"]].values
    econ_to_sar_data = combined[["sar", "econ"]].values

    try:
        results_s2e = grangercausalitytests(
            sar_to_econ_data, maxlag=max_lag, verbose=False
        )
        results_e2s = grangercausalitytests(
            econ_to_sar_data, maxlag=max_lag, verbose=False
        )
    except Exception:
        logger.exception("Granger causality test failed")
        return GrangerResult(
            sar_causes_econ=False,
            econ_causes_sar=False,
            optimal_lag=0,
            best_direction="none",
            f_statistic=0.0,
            p_value=1.0,
        )

    # Find best lag for each direction (lowest p-value from F-test)
    def _best_lag(results: dict) -> tuple[int, float, float]:
        best_p = 1.0
        best_f = 0.0
        best_l = 1
        for lag_val in range(1, max_lag + 1):
            if lag_val not in results:
                continue
            test_result = results[lag_val]
            # test_result is (test_dict, ols_results)
            f_test = test_result[0]["ssr_ftest"]
            p = f_test[1]
            f = f_test[0]
            if p < best_p:
                best_p = p
                best_f = f
                best_l = lag_val
        return best_l, best_f, best_p

    lag_s2e, f_s2e, p_s2e = _best_lag(results_s2e)
    lag_e2s, f_e2s, p_e2s = _best_lag(results_e2s)

    sar_causes = p_s2e < significance
    econ_causes = p_e2s < significance

    if sar_causes and econ_causes:
        direction = "bidirectional"
    elif sar_causes:
        direction = "sar->econ"
    elif econ_causes:
        direction = "econ->sar"
    else:
        direction = "none"

    # Report the more interesting direction (SAR causing econ)
    optimal_lag = lag_s2e if sar_causes else lag_e2s
    best_f = f_s2e if sar_causes else f_e2s
    best_p = p_s2e if sar_causes else p_e2s

    logger.info(
        "Granger test: SAR->econ p=%.4f (lag=%d), econ->SAR p=%.4f (lag=%d) => %s",
        p_s2e,
        lag_s2e,
        p_e2s,
        lag_e2s,
        direction,
    )

    return GrangerResult(
        sar_causes_econ=sar_causes,
        econ_causes_sar=econ_causes,
        optimal_lag=optimal_lag,
        best_direction=direction,
        f_statistic=best_f,
        p_value=best_p,
    )
