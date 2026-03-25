"""Generate Santos anchorage report: figures + 2-page PDF summary."""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SANTOS_CSV = ROOT / "data" / "detections" / "santos" / "summaries.csv"
RESULTS_JSON = ROOT / "data" / "detections" / "santos" / "analysis_results.json"
YF_PATH = ROOT / "data" / "economic" / "yfinance_data.parquet"
FIG_DIR = ROOT / "data" / "figures" / "santos_report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "grid.color": "#222222",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
})

ACCENT = "#00d4aa"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#4ecdc4"
ACCENT4 = "#ffe66d"
BG = "#0e1117"
SOY_MAX_GAP_DAYS = 7


def load_optional_json(path: Path) -> dict:
    if path.exists():
        return json.load(open(path))
    return {}


def format_pvalue(p_value) -> str:
    if p_value is None or pd.isna(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{float(p_value):.3f}"


def format_signed(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def align_series_nearest(series: pd.Series, target_dates: pd.Series, max_gap_days: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    if len(idx) == 0:
        return pd.DataFrame(
            {
                "matched_date": [pd.NaT] * len(target_dates),
                "value": [np.nan] * len(target_dates),
                "gap_days": [np.nan] * len(target_dates),
                "is_valid": [False] * len(target_dates),
            }
        )

    values = series.astype(float).to_numpy()
    rows = []
    for target in pd.to_datetime(target_dates):
        diffs = np.abs(idx - target)
        nearest_idx = int(diffs.argmin())
        gap_days = float(diffs[nearest_idx] / np.timedelta64(1, "D"))
        is_valid = gap_days <= max_gap_days
        rows.append(
            {
                "matched_date": idx[nearest_idx] if is_valid else pd.NaT,
                "value": float(values[nearest_idx]) if is_valid else np.nan,
                "gap_days": gap_days,
                "is_valid": is_valid,
            }
        )
    return pd.DataFrame(rows)


def correlation_result(results: dict, metric_name: str) -> dict:
    return results.get("correlations", {}).get(metric_name, {})


def load_data():
    santos = pd.read_csv(SANTOS_CSV)
    santos["date"] = pd.to_datetime(santos["timestamp"]).dt.normalize()
    santos = santos.sort_values("date").reset_index(drop=True)

    results = json.load(open(RESULTS_JSON))
    metrics = pd.DataFrame(results["scene_metrics"])
    metrics["date"] = pd.to_datetime(metrics["date"])

    yf = pd.read_parquet(YF_PATH)
    yf.index = pd.to_datetime(yf.index)

    if "soy_price_valid" in metrics.columns:
        metrics["soy_price_valid"] = metrics["soy_price_valid"].astype(bool)
        metrics["soy_date"] = pd.to_datetime(metrics.get("soy_date"), errors="coerce")
        metrics["soy_gap_days"] = pd.to_numeric(metrics.get("soy_gap_days"), errors="coerce")
        metrics["soy_price"] = pd.to_numeric(metrics.get("soy_price"), errors="coerce")
    else:
        aligned = align_series_nearest(yf["ZS=F"].dropna(), metrics["date"], SOY_MAX_GAP_DAYS)
        metrics["soy_price"] = aligned["value"]
        metrics["soy_date"] = aligned["matched_date"]
        metrics["soy_gap_days"] = aligned["gap_days"]
        metrics["soy_price_valid"] = aligned["is_valid"]
    metrics["month"] = metrics["date"].dt.month
    metrics["is_harvest"] = metrics["month"].isin([2, 3, 4, 5])

    ext_results = load_optional_json(ROOT / "data" / "detections" / "santos" / "extended_correlations.json")
    trade_results = load_optional_json(ROOT / "data" / "detections" / "santos" / "trade_correlations.json")
    aps_results = load_optional_json(ROOT / "data" / "backtest" / "santos_soy_vs_anchorage_test.json")

    return santos, metrics, yf, results, ext_results, trade_results, aps_results


def fig1_dual_timeseries(metrics, yf):
    """Anchorage vessel count + soy price; two panels for 2020 and 2023-2026."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Split data into two periods (there's a gap 2021-2022)
    early = metrics[metrics["date"] < "2021-01-01"].copy()
    late = metrics[metrics["date"] >= "2023-01-01"].copy()

    fig, (ax_e, ax_l) = plt.subplots(1, 2, figsize=(14, 3.8),
                                      gridspec_kw={"width_ratios": [1, 3.5]}, sharey=True)

    for ax, data, date_range, show_soy in [(ax_e, early, ("2019-12", "2020-10"), False),
                                             (ax_l, late, ("2023-09", "2026-05"), True)]:
        # Harvest shading
        for year in range(2020, 2027):
            start = pd.Timestamp(f"{year}-02-01")
            end = pd.Timestamp(f"{year}-05-31")
            if start >= pd.Timestamp(date_range[0]) and start <= pd.Timestamp(date_range[1]):
                ax.axvspan(start, end, alpha=0.08, color=ACCENT3, zorder=0)

        # Scatter
        ax.scatter(data["date"], data["anchorage_count"],
                   c=[ACCENT if h else "#666666" for h in data["is_harvest"]],
                   s=40, zorder=5, edgecolors="none", alpha=0.8)

        # Rolling avg
        ds = data.sort_values("date").set_index("date")["anchorage_count"]
        roll = ds.rolling("30D", min_periods=2).mean()
        ax.plot(roll.index, roll.values, color=ACCENT, linewidth=2, alpha=0.9)

        ax.set_xlim(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if show_soy else 2))
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=35)

        # Soy price on right panel
        if show_soy:
            ax2 = ax.twinx()
            soy_w = yf["ZS=F"].loc[date_range[0]:date_range[1]].resample("W").last().dropna()
            ax2.fill_between(soy_w.index, soy_w.values, alpha=0.12, color=ACCENT4)
            ax2.plot(soy_w.index, soy_w.values, color=ACCENT4, alpha=0.5, linewidth=1)
            ax2.set_ylabel("Soy Futures ($/bu)", color=ACCENT4, fontsize=10)
            ax2.tick_params(axis="y", labelcolor=ACCENT4)

    ax_e.set_ylabel("Anchorage Vessel Count", color=ACCENT, fontsize=10)
    ax_e.set_title("2020", fontsize=10, color="#888888")
    ax_l.set_title("2023 - 2026", fontsize=10, color="#888888")

    # Break indicator
    ax_e.spines["right"].set_visible(False)
    ax_l.spines["left"].set_visible(False)
    d = 0.015
    kwargs = dict(transform=ax_e.transAxes, color="#555555", clip_on=False, linewidth=1)
    ax_e.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_e.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs["transform"] = ax_l.transAxes
    ax_l.plot((-d, +d), (-d, +d), **kwargs)
    ax_l.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    fig.suptitle("Santos Anchorage: SAR Vessel Detections vs Soybean Futures",
                 fontsize=13, fontweight="bold", y=1.02)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT, markersize=7, label="Harvest (Feb-May)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markersize=7, label="Off-season"),
        Patch(facecolor=ACCENT4, alpha=0.3, label="Soy price"),
    ]
    ax_l.legend(handles=legend_elements, loc="upper right", framealpha=0.3, edgecolor="#333333", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_timeseries.png")


def fig2_scatter_correlation(metrics, results):
    """Scatter: anchorage count vs soy price with regression line."""
    valid_metrics = metrics[metrics["soy_price_valid"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pairs = [
        ("anchorage_count", "Anchorage Count", ACCENT),
        ("queue_length_proxy", "Queue Length (Anch + Channel)", ACCENT3),
        ("total_count", "Total Vessel Count", ACCENT4),
    ]

    for ax, (col, label, color) in zip(axes, pairs):
        result = correlation_result(results, col)
        x = valid_metrics["soy_price"].values
        y = valid_metrics[col].values

        if len(valid_metrics) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "No valid soy overlap", ha="center", va="center")
            continue

        ax.scatter(x, y, c=color, alpha=0.6, s=40, edgecolors="none")

        # Regression line
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 2:
            slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, alpha=0.8)

        title = (
            f"{label}\n"
            f"rho={format_signed(result.get('spearman_rho'))}, "
            f"p={format_pvalue(result.get('spearman_p'))}, "
            f"n={result.get('n_obs', int(mask.sum()))}"
        )
        if result.get("differenced_spearman_rho") is not None:
            title += (
                f"\ndiff rho={format_signed(result.get('differenced_spearman_rho'))}, "
                f"p={format_pvalue(result.get('differenced_spearman_p'))}"
            )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Soybean Futures ($/bu)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Santos Anchorage: SAR Signals vs Soybean Price", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 02_scatter.png")


def fig3_monthly_seasonal(metrics):
    """Monthly vessel count profile showing seasonality."""
    fig, ax = plt.subplots(figsize=(10, 3.8))

    monthly = metrics.groupby("month").agg(
        mean_anch=("anchorage_count", "mean"),
        std_anch=("anchorage_count", "std"),
        mean_total=("total_count", "mean"),
        n=("total_count", "count"),
    )

    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Only plot months we have data for
    has_data = [m for m in months if m in monthly.index]
    anch_means = [monthly.loc[m, "mean_anch"] if m in monthly.index else 0 for m in months]
    anch_stds = [monthly.loc[m, "std_anch"] if m in monthly.index else 0 for m in months]
    total_means = [monthly.loc[m, "mean_total"] if m in monthly.index else 0 for m in months]
    n_scenes = [int(monthly.loc[m, "n"]) if m in monthly.index else 0 for m in months]

    colors = [ACCENT3 if m in [2, 3, 4, 5] else "#555555" for m in months]

    bars = ax.bar(months, anch_means, color=colors, alpha=0.8, width=0.6, edgecolor="none")
    ax.errorbar(months, anch_means, yerr=anch_stds, fmt="none", ecolor="#999999", capsize=4, alpha=0.5)

    # Add n labels
    for i, (m, n) in enumerate(zip(months, n_scenes)):
        if n > 0:
            ax.text(m, anch_means[i] + anch_stds[i] + 1, f"n={n}", ha="center", fontsize=8, color="#888888")

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Mean Anchorage Vessel Count", fontsize=12)
    ax.set_title("Santos: Monthly Anchorage Activity Profile", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="y", alpha=0.2)

    # Harvest annotation
    ax.annotate("Soy Harvest\nSeason", xy=(3.5, max(anch_means) * 0.95),
                fontsize=11, color=ACCENT3, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=ACCENT3, alpha=0.15, edgecolor=ACCENT3))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_seasonal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_seasonal.png")


def fig4_methodology(metrics, results):
    """Visual overview: SAR image -> detection -> zone -> correlation."""
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Detection zone map (schematic)
    ax1 = fig.add_subplot(gs[:, 0])
    # Draw Santos port schematic
    ax1.set_xlim(-46.45, -46.05)
    ax1.set_ylim(-24.35, -23.95)

    from matplotlib.patches import Rectangle
    # Inner harbor
    ax1.add_patch(Rectangle((-46.35, -23.97), 0.08, 0.08, facecolor="#444444", alpha=0.5, edgecolor="#666666"))
    ax1.text(-46.31, -23.93, "Inner\nHarbor", ha="center", va="center", fontsize=7, color="#999999")

    # Channel
    ax1.add_patch(Rectangle((-46.30, -24.05), 0.12, 0.06, facecolor="#335566", alpha=0.5, edgecolor="#446677"))
    ax1.text(-46.24, -24.02, "Channel", ha="center", va="center", fontsize=7, color="#88aacc")

    # Anchorage zones
    ax1.add_patch(Rectangle((-46.25, -24.25), 0.18, 0.15, facecolor=ACCENT, alpha=0.2, edgecolor=ACCENT))
    ax1.text(-46.16, -24.175, "Guaruja\nAnchorage", ha="center", va="center", fontsize=8, color=ACCENT, fontweight="bold")

    ax1.add_patch(Rectangle((-46.40, -24.30), 0.15, 0.12, facecolor=ACCENT3, alpha=0.2, edgecolor=ACCENT3))
    ax1.text(-46.325, -24.24, "Outer\nRoads", ha="center", va="center", fontsize=7, color=ACCENT3)

    # Bbox outline
    ax1.plot([-46.42, -46.09, -46.09, -46.42, -46.42],
             [-24.33, -24.33, -24.00, -24.00, -24.33],
             color=ACCENT2, linewidth=1.5, linestyle="--", alpha=0.6)
    ax1.text(-46.255, -24.34, "SAR Coverage Area", ha="center", fontsize=7, color=ACCENT2)

    ax1.set_title("Zone Layout", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, alpha=0.15)

    # Panel 2: Histogram of vessel counts
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(metrics["anchorage_count"], bins=20, color=ACCENT, alpha=0.7, edgecolor="none")
    ax2.axvline(metrics["anchorage_count"].mean(), color=ACCENT2, linestyle="--", linewidth=1.5)
    ax2.set_title("Anchorage Count Distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Vessels")
    ax2.grid(True, alpha=0.2)

    # Panel 3: Harvest vs off-season boxplot
    ax3 = fig.add_subplot(gs[1, 1])
    harvest = metrics[metrics["is_harvest"]]["anchorage_count"]
    off = metrics[~metrics["is_harvest"]]["anchorage_count"]
    harvest_p = (
        stats.mannwhitneyu(harvest.values, off.values, alternative="two-sided").pvalue
        if len(harvest) >= 3 and len(off) >= 3
        else np.nan
    )
    bp = ax3.boxplot([harvest.values, off.values], tick_labels=["Harvest\n(Feb-May)", "Off-season"],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(ACCENT3)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#555555")
    bp["boxes"][1].set_alpha(0.6)
    ax3.set_title(f"Seasonal Comparison (p={format_pvalue(harvest_p)})", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Anchorage Count")
    ax3.grid(True, axis="y", alpha=0.2)

    # Panel 4: Correlation scorecard
    ax4 = fig.add_subplot(gs[:, 2:])
    ax4.axis("off")

    scorecard = []
    for signal_key, signal_label in [
        ("anchorage_count", "Anchorage"),
        ("queue_length_proxy", "Queue"),
        ("total_count", "Total"),
        ("berth_count", "Berth"),
        ("congestion_index", "Congestion"),
    ]:
        result = correlation_result(results, signal_key)
        if result.get("valid_for_inference"):
            if result.get("differenced_spearman_rho") is None:
                flag = "LEVEL"
            elif result.get("differenced_spearman_p") is not None and result.get("differenced_spearman_p") < 0.10:
                flag = "ROBUST"
            else:
                flag = "CAUTION"
            scorecard.append(
                (
                    signal_label,
                    format_signed(result.get("spearman_rho")),
                    format_signed(result.get("differenced_spearman_rho")),
                    flag,
                )
            )
        else:
            scorecard.append((signal_label, "n/a", "n/a", "LOW VAR"))

    valid_overlap = int(metrics["soy_price_valid"].sum())
    if metrics["soy_price_valid"].any():
        first_valid = metrics.loc[metrics["soy_price_valid"], "date"].min()
        last_valid = metrics.loc[metrics["soy_price_valid"], "date"].max()
        overlap_text = f"Soy-overlap scenes: {valid_overlap} ({first_valid:%Y-%m} - {last_valid:%Y-%m})"
    else:
        overlap_text = "Soy-overlap scenes: 0"

    ax4.text(0.5, 0.95, "Correlation Scorecard", fontsize=14, fontweight="bold",
             ha="center", va="top", transform=ax4.transAxes)
    ax4.text(0.5, 0.88, overlap_text, fontsize=10,
             ha="center", va="top", transform=ax4.transAxes, color="#888888")

    headers = ["Signal", "rho", "diff rho", "Flag"]
    col_x = [0.05, 0.43, 0.63, 0.83]

    y = 0.78
    for i, h in enumerate(headers):
        ax4.text(col_x[i], y, h, fontsize=10, fontweight="bold", transform=ax4.transAxes, color="#aaaaaa")
    y -= 0.04
    ax4.plot([0.03, 0.97], [y, y], color="#333333", transform=ax4.transAxes, clip_on=False)

    for signal, rho, diff_rho, status in scorecard:
        y -= 0.09
        color = ACCENT if status in {"ROBUST", "LEVEL"} else (ACCENT4 if status == "CAUTION" else ACCENT2)
        ax4.text(col_x[0], y, signal, fontsize=10, transform=ax4.transAxes)
        ax4.text(col_x[1], y, rho, fontsize=11, fontweight="bold", transform=ax4.transAxes, color=color)
        ax4.text(col_x[2], y, diff_rho, fontsize=10, transform=ax4.transAxes)
        ax4.text(col_x[3], y, status, fontsize=10, fontweight="bold", transform=ax4.transAxes, color=color)

    y -= 0.08
    ax4.plot([0.03, 0.97], [y, y], color="#333333", transform=ax4.transAxes, clip_on=False)
    y -= 0.06
    ax4.text(0.5, y, "LEVEL = contemporaneous association only",
             fontsize=9, ha="center", transform=ax4.transAxes, color="#666666")
    y -= 0.06
    ax4.text(0.5, y, "CAUTION = differenced relationship is weak or unstable",
             fontsize=9, ha="center", transform=ax4.transAxes, color="#666666")

    fig.suptitle("", fontsize=1)
    fig.savefig(FIG_DIR / "04_methodology.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_methodology.png")


def _dark_page(pdf):
    pdf.set_fill_color(14, 17, 23)
    pdf.rect(0, 0, 210, 297, "F")


def _section(pdf, text):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _text(pdf, text, size=9):
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(200, 200, 200)
    pdf.multi_cell(0, 4.2, text)
    pdf.ln(1)


def _img(pdf, path, w=190, x=10):
    if Path(path).exists():
        pdf.image(str(path), x=x, w=w)
        pdf.ln(2)


def artifact_is_fresh(artifact_path: Path, reference_path: Path) -> bool:
    return artifact_path.exists() and reference_path.exists() and artifact_path.stat().st_mtime >= reference_path.stat().st_mtime


def _remaining(pdf):
    """Remaining space on page before margin."""
    return pdf.h - pdf.get_y() - 15


def generate_pdf_report(metrics, results, ext_results, trade_results, aps_results):
    """Generate full PDF report with all figures."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("  fpdf not installed, skipping PDF. Install with: pip3 install fpdf2")
        return

    n_scenes = len(metrics)
    date_min = metrics["date"].min().strftime("%b %Y")
    date_max = metrics["date"].max().strftime("%b %Y")
    valid_metrics = metrics[metrics["soy_price_valid"]].copy()
    overlap_n = len(valid_metrics)
    overlap_min = valid_metrics["date"].min() if overlap_n else None
    overlap_max = valid_metrics["date"].max() if overlap_n else None

    anch_res = correlation_result(results, "anchorage_count")
    queue_res = correlation_result(results, "queue_length_proxy")
    total_res = correlation_result(results, "total_count")
    season_summary = results.get("soy_season_summary", {})
    harvest_p = season_summary.get("anchorage_mannwhitney_p")
    signal_diag = results.get("signal_diagnostics", {})
    anch_diag = signal_diag.get("anchorage_count", {})
    anch_detrended = anch_diag.get("detrended_full_sample", {})
    anch_detrended_harvest = anch_diag.get("detrended_harvest_only", {})
    anch_granger = anch_diag.get("monthly_differenced_granger", {})
    anch_oot = anch_diag.get("out_of_time_level_regression", {})
    anch_yearly = results.get("yearly_correlations", {}).get("anchorage_count", {})
    yearly_signs = [
        np.sign(v.get("spearman_rho"))
        for v in anch_yearly.values()
        if v.get("spearman_rho") is not None
    ]
    sign_flips = bool(yearly_signs) and (min(yearly_signs) < 0 < max(yearly_signs))

    queue_proxy_rows = ext_results.get("correlations", {}).get("queue_length_proxy", [])
    proxy_coverage = ext_results.get("alignment", {}).get("coverage", {})
    diff_proxy_rows = {
        row["indicator"]: row
        for row in ext_results.get("differenced_correlations", {}).get("queue_length_proxy", [])
    }

    trade_corr = trade_results.get("correlations", {})
    grain_scenario = trade_corr.get("sar_vs_grain_throughput", {})
    soy_scenario = trade_corr.get("sar_vs_soy_exports", {})
    aps_tests = aps_results.get("tests", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(15, 12, 15)  # L, T, R margins

    # ================================================================
    # PAGE 1 — Title + Summary + Time Series
    # ================================================================
    pdf.add_page()
    _dark_page(pdf)

    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 15, "Santos Anchorage", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 8, "SAR-Based Soy Exposure Review", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"{datetime.now().strftime('%B %d, %Y')}  |  {n_scenes} Sentinel-1 scenes  |  {date_min} - {date_max}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)

    # Key finding box
    pdf.set_fill_color(0, 212, 170)
    pdf.rect(15, pdf.get_y(), 180, 12, "F")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(14, 17, 23)
    overlap_text = (
        f"Soy overlap: {overlap_n} scenes ({overlap_min:%Y-%m} - {overlap_max:%Y-%m})"
        if overlap_n
        else "Soy overlap: no valid price matches"
    )
    pdf.cell(0, 6, overlap_text, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(
        0,
        6,
        (
            f"Anchorage rho={format_signed(anch_res.get('spearman_rho'))}, "
            f"detrended rho={format_signed(anch_detrended.get('spearman_rho'))}"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.ln(3)

    _section(pdf, "Executive Summary")
    _text(pdf, (
        f"{n_scenes} Sentinel-1 SAR scenes over Santos anchorage ({date_min} to {date_max}). "
        f"Only {overlap_n} scenes overlap the local soybean futures series within {SOY_MAX_GAP_DAYS} days, "
        "so all price correlations are computed on that later window. "
        f"Anchorage counts show a positive level correlation with soy futures "
        f"(rho={format_signed(anch_res.get('spearman_rho'))}, p={format_pvalue(anch_res.get('spearman_p'))}), "
        f"but that relationship weakens after detrending "
        f"(detrended rho={format_signed(anch_detrended.get('spearman_rho'))}, "
        f"p={format_pvalue(anch_detrended.get('spearman_p'))}) and after first differencing "
        f"(diff rho={format_signed(anch_res.get('differenced_spearman_rho'))}, "
        f"p={format_pvalue(anch_res.get('differenced_spearman_p'))}). "
        f"Within-year signs flip across the overlap window, so the level relationship is not stable evidence of a standalone price signal. "
        f"Harvest months average {season_summary.get('soy_months_avg_anchorage', float('nan')):.1f} anchorage vessels "
        f"versus {season_summary.get('off_months_avg_anchorage', float('nan')):.1f} off-season "
        f"(p={format_pvalue(harvest_p)})."
    ), size=8)

    _img(pdf, FIG_DIR / "01_timeseries.png", w=180, x=15)

    # ================================================================
    # PAGE 2 — Scatter + Scorecard
    # ================================================================
    pdf.add_page()
    _dark_page(pdf)

    _section(pdf, "Signal Review: SAR vs Soybean Futures")
    _img(pdf, FIG_DIR / "02_scatter.png", w=180, x=15)

    _section(pdf, "Zone Analysis & Correlation Scorecard")
    _img(pdf, FIG_DIR / "04_methodology.png", w=180, x=15)

    # ================================================================
    # PAGE 3 — Extended Indicators
    # ================================================================
    pdf.add_page()
    _dark_page(pdf)

    _section(pdf, "Extended Proxy Checks")
    if queue_proxy_rows:
        coverage_counts = [proxy_coverage.get(row["indicator"], {}).get("n_valid_scenes", 0) for row in queue_proxy_rows]
        _text(pdf, (
            f"External indicators are aligned by nearest date within +/-{ext_results.get('alignment', {}).get('max_gap_days', SOY_MAX_GAP_DAYS)} days. "
            f"Coverage across the saved proxy set ranges from {min(coverage_counts)} to {max(coverage_counts)} scenes. "
            "These are descriptive overlap checks, not independent validation."
        ))

        pdf.set_font("Courier", "B", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, "  Indicator                    rho     adj. p   n     diff", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(50, 50, 50)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(1)

        pdf.set_font("Courier", "", 8)
        for row in queue_proxy_rows:
            indicator = row["indicator"]
            diff_row = diff_proxy_rows.get(indicator, {})
            coverage_row = proxy_coverage.get(indicator, {})
            passing = bool(row.get("significant"))
            if passing:
                pdf.set_text_color(0, 212, 170)
            else:
                pdf.set_text_color(80, 80, 80)
            pdf.cell(
                0,
                4.5,
                f"  {indicator:<26s} {format_signed(row.get('rho')):>6s}   "
                f"{format_pvalue(row.get('adjusted_p_value')):>7s}   "
                f"{coverage_row.get('n_valid_scenes', 0):>3d}   "
                f"{format_signed(diff_row.get('rho')):>6s}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
        pdf.ln(3)
    else:
        _text(pdf, "No extended proxy results are currently available.")

    ext_json_path = ROOT / "data" / "detections" / "santos" / "extended_correlations.json"
    if queue_proxy_rows and artifact_is_fresh(FIG_DIR / "05_extended_correlations.png", ext_json_path):
        _img(pdf, FIG_DIR / "05_extended_correlations.png", w=175, x=18)

    # Seasonality
    _section(pdf, "Seasonal Pattern: Soy Harvest Effect")
    _text(pdf, (
        "Anchorage counts rise during the Brazilian soy harvest window (Feb-May). "
        f"Harvest vs off-season Mann-Whitney p = {format_pvalue(harvest_p)}."
    ))
    _img(pdf, FIG_DIR / "03_seasonal.png", w=170, x=20)

    # ================================================================
    # PAGE 4 — Trade Validation + Methodology
    # ================================================================
    pdf.add_page()
    _dark_page(pdf)

    _section(pdf, "Trade / Throughput Cross-checks")
    trade_lines = [
        "The saved trade script should be read as scenario analysis, not validation."
    ]
    if grain_scenario:
        trade_lines.append(
            f"Synthetic grain scenario: rho={format_signed(grain_scenario.get('spearman_rho'))}, "
            f"p={format_pvalue(grain_scenario.get('p_value'))}, n={grain_scenario.get('n_months')} months."
        )
    if soy_scenario:
        trade_lines.append(
            f"Synthetic soy-export scenario: rho={format_signed(soy_scenario.get('spearman_rho'))}, "
            f"p={format_pvalue(soy_scenario.get('p_value'))}, n={soy_scenario.get('n_months')} months."
        )
    if aps_tests:
        trade_lines.append(
            "Direct 2024 APS comparison is mixed / negative: "
            f"soy complex rho={format_signed(aps_tests.get('soy_complex_t', {}).get('spearman_rho'))}, "
            f"soy grain rho={format_signed(aps_tests.get('soy_grain_t', {}).get('spearman_rho'))}, "
            f"soy meal rho={format_signed(aps_tests.get('soy_meal_t', {}).get('spearman_rho'))}."
        )
    _text(pdf, " ".join(trade_lines))
    trade_json_path = ROOT / "data" / "detections" / "santos" / "trade_correlations.json"
    if trade_results and artifact_is_fresh(FIG_DIR / "06_trade_validation.png", trade_json_path):
        _img(pdf, FIG_DIR / "06_trade_validation.png")

    _section(pdf, "Trend-Robust Diagnostics")
    granger_direction = anch_granger.get("direction", "none")
    if granger_direction == "econ->sar":
        granger_text = "Monthly differenced Granger points from soy to anchorage, not SAR to soy."
    elif granger_direction == "sar->econ":
        granger_text = "Monthly differenced Granger suggests SAR may lead soy."
    else:
        granger_text = "Monthly differenced Granger shows no significant SAR-leading result."
    _text(pdf, (
        f"Detrended anchorage rho={format_signed(anch_detrended.get('spearman_rho'))} "
        f"(p={format_pvalue(anch_detrended.get('spearman_p'))}); "
        f"harvest-only detrended rho={format_signed(anch_detrended_harvest.get('spearman_rho'))} "
        f"(p={format_pvalue(anch_detrended_harvest.get('spearman_p'))}). "
        f"{granger_text} "
        f"Train 2023-2024 / test 2025-2026 level regression has test R2={format_signed(anch_oot.get('test_r2'))}. "
        f"Year-by-year anchorage signs {'flip' if sign_flips else 'do not flip'} across the overlap window."
    ))

    _section(pdf, "Methodology")
    _text(pdf, (
        f"Data: {n_scenes} Sentinel-1 GRD RTC scenes (IW mode, 10m, VV pol.) via Microsoft Planetary Computer. "
        "Anchorage-only bbox [-46.42, -24.33, -46.09, -24.00] isolates Guaruja offshore anchorage, "
        "excluding inner harbor berths.\n\n"
        "Detection: 2D CA-CFAR (Pfa=1e-8, guard=5, bg=15) with adaptive wind correction via ERA5 wave age. "
        "Post-filtering: aspect ratio < 6.0, min brightness > -12 dB, dB contrast > 2.0.\n\n"
        "Zones: Spatial join to GeoJSON polygons (3 anchorage, 1 berth, 2 channel). "
        "Primary signal: queue_length_proxy = anchorage + channel count.\n\n"
        f"Economic alignment: nearest-date match within {results.get('soy_alignment', {}).get('max_gap_days', SOY_MAX_GAP_DAYS)} days; "
        "scenes outside that overlap are excluded from price correlations. "
        "Low-variation metrics such as berth_count and congestion_index are flagged instead of treated as primary evidence.\n\n"
        "Statistics: Spearman rank correlation at levels, detrended residual checks, first-difference checks, "
        "within-year splits, monthly differenced Granger, BH correction for multi-indicator screens, and out-of-time regression tests."
    ))

    _section(pdf, "Key Conclusions")
    conclusions = [
        f"1. Seasonality is clear: harvest anchorage average {season_summary.get('soy_months_avg_anchorage', float('nan')):.1f} vs "
        f"{season_summary.get('off_months_avg_anchorage', float('nan')):.1f} off-season (p={format_pvalue(harvest_p)}).",
        f"2. Positive level rho alone is not persuasive here: detrended anchorage rho is {format_signed(anch_detrended.get('spearman_rho'))}, "
        f"and yearly anchorage signs {'flip' if sign_flips else 'do not flip'} across the overlap window.",
        f"3. Differenced and predictive checks do not support a SAR-leading soy-price signal: diff rho {format_signed(anch_res.get('differenced_spearman_rho'))}, "
        f"Granger {anch_granger.get('direction', 'none')}, out-of-time R2 {format_signed(anch_oot.get('test_r2'))}.",
        "4. Extended indicators are overlap-limited proxy checks, not independent out-of-sample validation.",
        "5. Synthetic throughput scenarios and the 2024 APS comparison should be reported separately from the main Santos result.",
    ]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(200, 200, 200)
    for c in conclusions:
        pdf.multi_cell(0, 4.5, c)
        pdf.ln(0.5)

    pdf.ln(3)
    _section(pdf, "Data Sources")
    _text(pdf, (
        "SAR: Sentinel-1 GRD RTC (ESA Copernicus / Microsoft Planetary Computer) | "
        "Commodities: CBOT Soybean ZS=F, Soy Meal ZM=F, Soy Oil ZL=F (Yahoo Finance) | "
        "Macro: BRL/USD DEXBZUS, Soy Oil PPI PCU311224311224 (FRED) | "
        "Shipping: Baltic Dry BDRY (Yahoo Finance) | "
        "Wind: ERA5 (Open-Meteo) | Stats: scipy, statsmodels"
    ), size=8)

    out_path = ROOT / "data" / "santos_anchorage_report.pdf"
    pdf.output(str(out_path))
    print(f"  Saved PDF: {out_path}")


def main():
    print("=" * 60)
    print(" GENERATING SANTOS ANCHORAGE REPORT")
    print("=" * 60)

    santos, metrics, yf, results, ext_results, trade_results, aps_results = load_data()
    print(f"  {len(metrics)} scenes loaded")

    print("\n[1] Generating figures...")
    fig1_dual_timeseries(metrics, yf)
    fig2_scatter_correlation(metrics, results)
    fig3_monthly_seasonal(metrics)
    fig4_methodology(metrics, results)

    print("\n[2] Generating PDF report...")
    generate_pdf_report(metrics, results, ext_results, trade_results, aps_results)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
