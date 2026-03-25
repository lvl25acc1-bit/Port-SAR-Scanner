"""Time series plots for SAR indices and economic data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_site_index(
    weekly_df: pd.DataFrame,
    site_name: str,
    value_col: str = "mean",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot weekly activity index for a single site."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(weekly_df["week"], weekly_df[value_col], linewidth=1, color="steelblue")
    ax.fill_between(
        weekly_df["week"], 0, weekly_df[value_col], alpha=0.2, color="steelblue"
    )

    ax.set_title(f"Weekly Activity Index: {site_name}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Detection Count (weekly mean)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_sar_vs_economic(
    merged_df: pd.DataFrame,
    sar_col: str,
    econ_col: str,
    date_col: str = "week",
    output_path: Path | None = None,
) -> plt.Figure:
    """Dual-axis plot of SAR index vs an economic series."""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    color1 = "steelblue"
    color2 = "darkorange"

    ax1.plot(merged_df[date_col], merged_df[sar_col], color=color1, linewidth=1.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"SAR Index ({sar_col})", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(merged_df[date_col], merged_df[econ_col], color=color2, linewidth=1.5)
    ax2.set_ylabel(econ_col, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"SAR Activity Index vs {econ_col}", fontsize=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_lead_lag(
    lead_lag_df: pd.DataFrame,
    series_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot cross-correlation as a function of lag."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "red" if row.get("is_best", False) else "steelblue"
        for _, row in lead_lag_df.iterrows()
    ]
    ax.bar(lead_lag_df["lag_weeks"], lead_lag_df["correlation"], color=colors)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (weeks) — positive = SAR leads")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(f"Lead-Lag Cross-Correlation: SAR vs {series_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig
