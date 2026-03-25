#!/usr/bin/env python3
"""
Validation Scorecard: compare all SAR signal variants against economic indicators.

Produces:
  - data/backtest/outputs/validation_scorecard.csv
  - data/backtest/outputs/validation_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from portvolume.validation.metrics import ValidationResult, validate_signal
from portvolume.validation.report import generate_scorecard, generate_validation_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest" / "outputs"
ECON_DIR = PROJECT_ROOT / "data" / "economic"

WIND_FILE = BACKTEST_DIR / "wind_adjusted_weekly.parquet"
ZONE_FILE = BACKTEST_DIR / "zone_metrics_weekly.parquet"
FRED_FILE = ECON_DIR / "fred_data.parquet"
YFINANCE_FILE = ECON_DIR / "yfinance_data.parquet"

OUTPUT_DIR = BACKTEST_DIR  # scorecard goes next to the other outputs

# Baseline from prior work: raw SAR counts vs GFW AIS
BASELINE_RHO = -0.068

# Thresholds
MIN_RHO = 0.3
ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. Load SAR signals
# ---------------------------------------------------------------------------
def _to_week_monday(dt_series: pd.Series) -> pd.Series:
    """Snap datetimes to the Monday of their ISO week (consistent W-MON anchor)."""
    dt = pd.to_datetime(dt_series)
    # Monday = weekday 0.  Subtract the weekday offset to get Monday.
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def load_sar_signals() -> dict[str, pd.Series]:
    """Return {signal_name: weekly Series indexed by Monday-anchored date}."""
    wind_df = pd.read_parquet(WIND_FILE)
    zone_df = pd.read_parquet(ZONE_FILE)

    # Normalise date column to the Monday of the ISO week
    wind_df["week"] = _to_week_monday(wind_df["date"])
    zone_df["week"] = _to_week_monday(zone_df["week"])

    signals: dict[str, pd.Series] = {}

    # From wind-adjusted file
    for col, name in [("vessel_count", "raw_vessel_count"),
                      ("wind_adjusted_count", "wind_adjusted_count")]:
        s = wind_df.groupby("week")[col].mean().sort_index()
        s.name = name
        signals[name] = s

    # From zone metrics file
    for col, name in [("congestion_index_mean", "congestion_index"),
                      ("anchorage_count_mean", "anchorage_count"),
                      ("queue_length_proxy_mean", "queue_length_proxy")]:
        s = zone_df.groupby("week")[col].mean().sort_index()
        s.name = name
        signals[name] = s

    return signals


# ---------------------------------------------------------------------------
# 2. Load economic indicators (resample to W-MON)
# ---------------------------------------------------------------------------
def load_economic_indicators() -> pd.DataFrame:
    """Return a single DataFrame indexed by W-MON week with all econ columns."""
    fred = pd.read_parquet(FRED_FILE)
    yf = pd.read_parquet(YFINANCE_FILE)

    # Both have DatetimeIndex named 'date'
    fred.index = pd.to_datetime(fred.index)
    yf.index = pd.to_datetime(yf.index)

    # Snap each observation to the Monday of its ISO week, then take last per week
    def snap_to_monday(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = pd.to_datetime(df.index)
        df.index = idx - pd.to_timedelta(idx.weekday, unit="D")
        df.index.name = "week"
        # If multiple rows fall in the same week, keep last
        return df.groupby(level=0).last()

    fred_w = snap_to_monday(fred).sort_index()
    yf_w = snap_to_monday(yf).sort_index()

    # Forward-fill economic data (monthly series have gaps in weekly grid)
    # Build a complete weekly grid covering the union, then ffill
    all_mondays = pd.date_range(
        start=min(fred_w.index.min(), yf_w.index.min()),
        end=max(fred_w.index.max(), yf_w.index.max()),
        freq="7D",  # every 7 days starting from a Monday
    )
    # Ensure grid starts on a Monday
    all_mondays = all_mondays[all_mondays.weekday == 0]

    econ = fred_w.reindex(all_mondays).join(yf_w.reindex(all_mondays), how="outer")
    econ = econ.ffill()
    econ.index.name = "week"
    return econ


# ---------------------------------------------------------------------------
# 3. Enhanced correlation for one signal vs all econ indicators
# ---------------------------------------------------------------------------
def correlate_signal_vs_econ(
    signal: pd.Series,
    econ: pd.DataFrame,
) -> pd.DataFrame:
    """Merge one SAR signal with econ data and compute BH-corrected Spearman."""
    merged = econ.copy()
    merged["sar_signal"] = signal  # aligns on index (week)
    merged = merged.dropna(subset=["sar_signal"])

    econ_cols = [c for c in econ.columns if c != "sar_signal"]

    from portvolume.analysis.enhanced_correlation import compute_correlations_with_correction
    result = compute_correlations_with_correction(
        merged, "sar_signal", econ_cols, method="benjamini_hochberg"
    )
    return result


# ---------------------------------------------------------------------------
# 4. Lag-optimised correlation
# ---------------------------------------------------------------------------
def find_best_lag_correlation(
    signal: pd.Series,
    econ_series: pd.Series,
    max_lag: int = 12,
) -> tuple[int, float, float]:
    """Return (best_lag, rho_at_best_lag, p_at_best_lag).

    Positive lag = SAR leads economic indicator.
    """
    vals_sar = signal.values.astype(float)
    vals_econ = econ_series.values.astype(float)
    n = len(vals_sar)

    best_lag = 0
    best_rho = 0.0
    best_p = 1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x = vals_sar[: n - lag]
            y = vals_econ[lag:]
        elif lag < 0:
            x = vals_sar[-lag:]
            y = vals_econ[: n + lag]
        else:
            x = vals_sar
            y = vals_econ

        if len(x) < 10:
            continue

        r, p = stats.spearmanr(x, y)
        if abs(r) > abs(best_rho):
            best_rho = float(r)
            best_p = float(p)
            best_lag = lag

    return best_lag, best_rho, best_p


# ---------------------------------------------------------------------------
# 5. Cross-signal correlations
# ---------------------------------------------------------------------------
def cross_signal_correlations(signals: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise Spearman correlations between SAR signals."""
    names = list(signals.keys())
    records = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            merged = pd.DataFrame({n1: signals[n1], n2: signals[n2]}).dropna()
            if len(merged) < 10:
                continue
            rho, p = stats.spearmanr(merged[n1], merged[n2])
            records.append({
                "signal_a": n1,
                "signal_b": n2,
                "rho": round(rho, 4),
                "p_value": round(p, 6),
                "n_obs": len(merged),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("  PORT VOLUME VALIDATION SCORECARD")
    print("=" * 80)

    # --- Load data ---
    print("\n[1] Loading SAR signals ...")
    signals = load_sar_signals()
    for name, s in signals.items():
        print(f"    {name:30s} : {len(s)} weeks, range [{s.min():.1f}, {s.max():.1f}]")

    print("\n[2] Loading economic indicators ...")
    econ = load_economic_indicators()
    print(f"    {len(econ.columns)} indicators, {len(econ)} weekly rows")
    print(f"    Columns: {list(econ.columns)}")

    # --- Per-signal correlation with econ indicators ---
    print("\n[3] Computing correlations (BH-corrected) for each signal vs econ ...")
    scorecard_rows: list[dict] = []

    for sig_name, sig_series in signals.items():
        corr_df = correlate_signal_vs_econ(sig_series, econ)

        if corr_df.empty:
            print(f"    {sig_name}: no valid correlations")
            scorecard_rows.append({
                "signal": sig_name,
                "best_econ_indicator": "N/A",
                "rho": np.nan,
                "adjusted_p": np.nan,
                "optimal_lag": np.nan,
                "passes": False,
            })
            continue

        # Find best (highest |rho|) among adjusted-significant, else overall best
        sig_df = corr_df[corr_df["significant"]].copy()
        if sig_df.empty:
            sig_df = corr_df.copy()

        sig_df["abs_rho"] = sig_df["rho"].abs()
        best_row = sig_df.sort_values("abs_rho", ascending=False).iloc[0]

        # Now find optimal lag for that best indicator
        best_indicator = best_row["indicator"]
        merged_pair = pd.DataFrame({
            "sar": sig_series,
            "econ": econ[best_indicator],
        }).dropna()

        if len(merged_pair) >= 10:
            opt_lag, lag_rho, lag_p = find_best_lag_correlation(
                merged_pair["sar"], merged_pair["econ"], max_lag=12,
            )
        else:
            opt_lag, lag_rho, lag_p = 0, float(best_row["rho"]), float(best_row["p_value"])

        adj_p = float(best_row["adjusted_p_value"])
        rho_val = float(best_row["rho"])
        passes = abs(rho_val) >= MIN_RHO and adj_p < ALPHA

        scorecard_rows.append({
            "signal": sig_name,
            "best_econ_indicator": best_indicator,
            "rho": round(rho_val, 4),
            "adjusted_p": round(adj_p, 6),
            "optimal_lag": opt_lag,
            "lag_rho": round(lag_rho, 4),
            "passes": passes,
        })

        # Print all correlations for this signal
        print(f"\n    --- {sig_name} ---")
        sorted_corr = corr_df.copy()
        sorted_corr["abs_rho"] = sorted_corr["rho"].abs()
        sorted_corr = sorted_corr.sort_values("abs_rho", ascending=False).drop(columns=["abs_rho"])
        for _, row in sorted_corr.iterrows():
            flag = "*" if row["significant"] else " "
            print(f"      {flag} {row['indicator']:15s}  rho={row['rho']:+.4f}  p={row['p_value']:.4f}  adj_p={row['adjusted_p_value']:.4f}")

    # --- Cross-signal correlations ---
    print("\n[4] Cross-signal correlations (SAR signals vs each other) ...")
    cross_df = cross_signal_correlations(signals)
    for _, row in cross_df.iterrows():
        print(f"    {row['signal_a']:30s} vs {row['signal_b']:30s}  rho={row['rho']:+.4f}  (p={row['p_value']:.6f}, n={row['n_obs']})")

    # --- Build comparison scorecard ---
    print("\n" + "=" * 80)
    print("  COMPARISON SCORECARD")
    print("=" * 80)

    sc = pd.DataFrame(scorecard_rows)
    print(sc.to_string(index=False))

    # --- Build ValidationResult objects for the report generator ---
    print("\n[5] Building full validation report ...")
    val_results: list[ValidationResult] = []

    for sig_name, sig_series in signals.items():
        row = next(r for r in scorecard_rows if r["signal"] == sig_name)
        best_ind = row["best_econ_indicator"]
        if best_ind == "N/A":
            continue

        merged_pair = pd.DataFrame({
            "sar": sig_series,
            "ref": econ[best_ind],
        }).dropna()

        if len(merged_pair) < 10:
            continue

        vr = validate_signal(
            sar_series=merged_pair["sar"],
            reference_series=merged_pair["ref"],
            signal_name=sig_name,
            reference_name=best_ind,
            reference_source="FRED/yfinance",
            port_id="rotterdam",
            min_rho=MIN_RHO,
            max_lag=12,
        )
        val_results.append(vr)

    # Use the library's report generator
    generate_validation_report(
        results=val_results,
        output_dir=OUTPUT_DIR,
    )
    print(f"    Saved validation_scorecard.csv, validation_report.json, validation_summary.md")

    # Also save our extended scorecard
    sc.to_csv(OUTPUT_DIR / "validation_scorecard.csv", index=False)
    print(f"    Overwrote validation_scorecard.csv with extended scorecard")

    # Save cross-signal correlations
    cross_df.to_csv(OUTPUT_DIR / "cross_signal_correlations.csv", index=False)

    # --- Build custom markdown summary ---
    md_lines = [
        "# Validation Scorecard",
        "",
        "## Signal vs Economic Indicator Correlations",
        "",
        "| Signal | Best Indicator | Spearman rho | Adj. p-value | Optimal Lag | Lag rho | Verdict |",
        "|--------|---------------|-------------|-------------|------------|---------|---------|",
    ]
    for _, r in sc.iterrows():
        verdict = "PASS" if r["passes"] else "FAIL"
        lag_rho_str = f"{r.get('lag_rho', r['rho']):.4f}" if pd.notna(r.get("lag_rho", r["rho"])) else "N/A"
        rho_str = f"{r['rho']:.4f}" if pd.notna(r["rho"]) else "N/A"
        adj_str = f"{r['adjusted_p']:.4f}" if pd.notna(r["adjusted_p"]) else "N/A"
        lag_str = f"{int(r['optimal_lag'])}" if pd.notna(r["optimal_lag"]) else "N/A"
        md_lines.append(
            f"| {r['signal']} | {r['best_econ_indicator']} | {rho_str} | {adj_str} | {lag_str} | {lag_rho_str} | {verdict} |"
        )

    md_lines += [
        "",
        "## Cross-Signal Correlations",
        "",
        "| Signal A | Signal B | rho | p-value | n_obs |",
        "|----------|----------|-----|---------|-------|",
    ]
    for _, r in cross_df.iterrows():
        md_lines.append(
            f"| {r['signal_a']} | {r['signal_b']} | {r['rho']:.4f} | {r['p_value']:.6f} | {r['n_obs']} |"
        )

    md_lines += [
        "",
        "## Baseline Comparison",
        "",
        f"- **Previous baseline (raw SAR counts vs GFW AIS):** rho = {BASELINE_RHO}",
        "",
    ]

    # Find improvements
    passing = [r for r in scorecard_rows if r["passes"]]
    best_overall = max(scorecard_rows, key=lambda r: abs(r["rho"]) if pd.notna(r["rho"]) else 0)

    if passing:
        md_lines.append(f"**{len(passing)} signal(s) pass the threshold (|rho| >= {MIN_RHO} AND adj_p < {ALPHA}):**")
        md_lines.append("")
        for r in passing:
            md_lines.append(f"- **{r['signal']}** vs {r['best_econ_indicator']}: rho={r['rho']:.4f} (adj_p={r['adjusted_p']:.4f})")
    else:
        md_lines.append(f"**No signals pass the strict threshold (|rho| >= {MIN_RHO} AND adj_p < {ALPHA}).**")

    md_lines.append("")
    best_rho = abs(best_overall["rho"]) if pd.notna(best_overall["rho"]) else 0
    improvement = best_rho - abs(BASELINE_RHO)
    improvement_pct = (improvement / abs(BASELINE_RHO)) * 100 if abs(BASELINE_RHO) > 0 else float("inf")

    md_lines.append(f"**Best signal:** {best_overall['signal']} vs {best_overall['best_econ_indicator']} "
                     f"(|rho|={best_rho:.4f})")
    md_lines.append(f"**Improvement over baseline:** {improvement:+.4f} ({improvement_pct:+.1f}%)")
    md_lines.append("")

    if best_rho > abs(BASELINE_RHO):
        md_lines.append("**Verdict: The pivot to zone-aware / wind-adjusted signals shows improvement over the raw-count baseline.**")
    else:
        md_lines.append("**Verdict: No improvement over the raw-count baseline.**")
    md_lines.append("")

    md_text = "\n".join(md_lines)
    (OUTPUT_DIR / "validation_summary.md").write_text(md_text)
    print(f"    Overwrote validation_summary.md with full scorecard")

    # --- Final verdict ---
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    print(f"\n  Baseline (raw SAR counts vs GFW AIS): rho = {BASELINE_RHO}")
    print(f"  Best signal now: {best_overall['signal']} vs {best_overall['best_econ_indicator']}")
    print(f"    |rho| = {best_rho:.4f}")
    print(f"    Improvement over baseline: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    if passing:
        print(f"\n  {len(passing)} signal(s) PASS the threshold:")
        for r in passing:
            print(f"    - {r['signal']} vs {r['best_econ_indicator']}: rho={r['rho']:.4f}, adj_p={r['adjusted_p']:.6f}")
    else:
        print(f"\n  No signals pass the strict threshold (|rho| >= {MIN_RHO} AND adj_p < {ALPHA}).")
        print("  However, some may show meaningful correlations at relaxed thresholds.")

    if best_rho > abs(BASELINE_RHO):
        print(f"\n  >> The pivot IMPROVED over the baseline (|rho| {best_rho:.4f} vs {abs(BASELINE_RHO):.4f}).")
    else:
        print(f"\n  >> The pivot did NOT improve over the baseline.")

    print("\n" + "=" * 80)
    print("  Done. Outputs:")
    print(f"    {OUTPUT_DIR / 'validation_scorecard.csv'}")
    print(f"    {OUTPUT_DIR / 'validation_summary.md'}")
    print(f"    {OUTPUT_DIR / 'cross_signal_correlations.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
