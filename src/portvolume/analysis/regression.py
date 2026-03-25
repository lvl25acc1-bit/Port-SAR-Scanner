"""Regression analysis: rolling OLS and out-of-sample validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def rolling_ols(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    window: int = 26,
) -> pd.DataFrame:
    """Rolling OLS regression.

    Returns time series of R-squared and coefficients, useful for checking
    if the SAR-economic relationship is stable over time.

    Parameters
    ----------
    df : DataFrame with target and feature columns.
    target_col : Column name for dependent variable.
    feature_cols : Column names for independent variables.
    window : Rolling window size in weeks.

    Returns
    -------
    DataFrame with columns: [r_squared, *feature_coefficients], indexed by date.
    """
    results = []

    clean = df[[target_col] + feature_cols].dropna()
    if len(clean) < window:
        logger.warning("Not enough data for rolling OLS: %d < %d", len(clean), window)
        return pd.DataFrame()

    for i in range(window, len(clean) + 1):
        window_data = clean.iloc[i - window : i]

        y = window_data[target_col].values
        X = sm.add_constant(window_data[feature_cols].values)

        try:
            model = sm.OLS(y, X).fit()
            row = {"r_squared": model.rsquared}
            for j, col in enumerate(feature_cols):
                row[f"coef_{col}"] = model.params[j + 1]
            results.append(row)
        except Exception:
            results.append({"r_squared": np.nan})

    result_df = pd.DataFrame(results, index=clean.index[window - 1 :])
    return result_df


def out_of_sample_test(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """Time series cross-validated out-of-sample prediction.

    Uses TimeSeriesSplit to avoid look-ahead bias.
    Fits OLS on training fold, predicts on test fold.

    Returns
    -------
    dict with:
    - oos_r_squared: float (can be negative if model is worse than mean)
    - rmse: float
    - directional_accuracy: fraction of weeks where predicted direction matches actual
    - fold_results: list of per-fold metrics
    """
    clean = df[[target_col] + feature_cols].dropna()

    if len(clean) < n_splits * 10:
        logger.warning(
            "Insufficient data for %d-fold CV: %d rows", n_splits, len(clean)
        )
        return {
            "oos_r_squared": np.nan,
            "rmse": np.nan,
            "directional_accuracy": np.nan,
            "fold_results": [],
        }

    y = clean[target_col].values
    X = clean[feature_cols].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_y_true = []
    all_y_pred = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = sm.add_constant(X[train_idx])
        X_test = sm.add_constant(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        try:
            model = sm.OLS(y_train, X_train).fit()
            y_pred = model.predict(X_test)

            # Per-fold metrics
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            fold_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            fold_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            fold_results.append(
                {"fold": fold_idx, "r_squared": fold_r2, "rmse": fold_rmse}
            )

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        except Exception:
            logger.exception("OLS failed on fold %d", fold_idx)

    if not all_y_true:
        return {
            "oos_r_squared": np.nan,
            "rmse": np.nan,
            "directional_accuracy": np.nan,
            "fold_results": [],
        }

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    # Overall OOS R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Directional accuracy: did we predict the correct direction of change?
    if len(y_true) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc = np.mean(true_dir == pred_dir)
    else:
        dir_acc = np.nan

    logger.info(
        "OOS test: R²=%.4f, RMSE=%.4f, Dir.Acc=%.2f%% (%d folds)",
        oos_r2,
        rmse,
        dir_acc * 100 if not np.isnan(dir_acc) else 0,
        n_splits,
    )

    return {
        "oos_r_squared": oos_r2,
        "rmse": rmse,
        "directional_accuracy": dir_acc,
        "fold_results": fold_results,
    }
