#!/usr/bin/env python3
"""Extended correlation analysis for Santos port — additional economic indicators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portvolume.analysis.enhanced_correlation import compute_correlations_with_correction

MAX_MATCH_GAP_DAYS = 7
MIN_CORR_OBS = 10


def fetch_fred_series(series_id: str, label: str) -> pd.Series:
    """Fetch a FRED series, returning an empty series if unavailable."""
    try:
        from fredapi import Fred

        fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))
        series = fred.get_series(series_id).dropna()
        series.index = pd.to_datetime(series.index)
        series.name = label
        return series
    except Exception as exc:
        print(f"    skipped {series_id}: {exc}")
        return pd.Series(dtype=float, name=label)


def download_close_series(ticker: str, label: str) -> pd.Series:
    """Download a yfinance close series, returning empty on failure."""
    try:
        import yfinance as yf

        raw = yf.download(ticker, start="2023-01-01", end="2026-04-01", progress=False)
        if raw.empty:
            return pd.Series(dtype=float, name=label)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)
        close = raw["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        close.name = label
        return close
    except Exception as exc:
        print(f"    skipped {ticker}: {exc}")
        return pd.Series(dtype=float, name=label)


def series_diagnostics(series: pd.Series) -> dict:
    clean = pd.Series(series).dropna()
    if clean.empty:
        return {"n_obs": 0, "n_unique": 0, "std": None, "informative": False, "note": "no overlapping observations"}

    n_unique = int(clean.nunique())
    std = float(clean.std()) if len(clean) > 1 else 0.0
    informative = len(clean) >= MIN_CORR_OBS and n_unique >= 3 and std > 0
    note_parts = []
    if len(clean) < MIN_CORR_OBS:
        note_parts.append(f"need >= {MIN_CORR_OBS} observations")
    if n_unique < 3:
        note_parts.append(f"only {n_unique} unique values")
    if std == 0:
        note_parts.append("constant series")
    return {
        "n_obs": int(len(clean)),
        "n_unique": n_unique,
        "std": std,
        "informative": informative,
        "note": "; ".join(note_parts) if note_parts else None,
    }


def differenced_spearman(x: pd.Series, y: pd.Series) -> dict:
    diff_df = pd.DataFrame({"x": x, "y": y}).diff().dropna()
    x_diag = series_diagnostics(diff_df["x"])
    y_diag = series_diagnostics(diff_df["y"])
    if not x_diag["informative"] or not y_diag["informative"]:
        return {
            "rho": None,
            "p_value": None,
            "n_obs": int(len(diff_df)),
            "note": "; ".join(note for note in (x_diag["note"], y_diag["note"]) if note) or "insufficient variation",
        }

    rho, p_value = stats.spearmanr(diff_df["x"], diff_df["y"])
    return {"rho": float(rho), "p_value": float(p_value), "n_obs": int(len(diff_df)), "note": None}

# ── 1. Load Santos detection summaries ───────────────────────────────────────
print("Loading Santos detection data …")
summaries = pd.read_csv(ROOT / "data/detections/santos/summaries.csv", parse_dates=["timestamp"])
print(f"  summaries: {len(summaries)} rows, cols={list(summaries.columns)}")

# ── 2. Load zone metrics from analysis_results.json ─────────────────────────
print("Loading zone / scene metrics …")
with open(ROOT / "data/detections/santos/analysis_results.json") as f:
    analysis = json.load(f)

scene_metrics = pd.DataFrame(analysis["scene_metrics"])
scene_metrics["date"] = pd.to_datetime(scene_metrics["date"])
print(f"  scene_metrics: {len(scene_metrics)} rows, cols={list(scene_metrics.columns)}")

# ── 3. Fetch / load economic indicators ──────────────────────────────────────
print("Fetching economic indicators …")

# 3a. BRL/USD from FRED
print("  FRED DEXBZUS (BRL/USD) …")
brl_usd = fetch_fred_series("DEXBZUS", "BRL_USD")
if not brl_usd.empty:
    print(f"    {len(brl_usd)} observations, {brl_usd.index.min():%Y-%m-%d} – {brl_usd.index.max():%Y-%m-%d}")

# 3b. Soy meal futures (ZM=F) from yfinance
print("  yfinance ZM=F (soy meal) …")
zm_close = download_close_series("ZM=F", "SoyMeal_ZM")
if not zm_close.empty:
    print(f"    {len(zm_close)} observations")

# 3c. Soy oil futures (ZL=F)
print("  yfinance ZL=F (soy oil) …")
zl_close = download_close_series("ZL=F", "SoyOil_ZL")
if not zl_close.empty:
    print(f"    {len(zl_close)} observations")

# 3d. From existing parquet: BDRY and ZS=F
print("  Loading BDRY & ZS=F from parquet …")
yf_parquet = pd.read_parquet(ROOT / "data/economic/yfinance_data.parquet")
bdry = yf_parquet["BDRY"].dropna()
bdry.name = "BDRY"
zs = yf_parquet["ZS=F"].dropna()
zs.name = "Soybean_ZS"
print(f"    BDRY: {len(bdry)} obs,  Soybean_ZS: {len(zs)} obs")

# ── 4. Align indicators to scene dates (nearest-date match) ─────────────────
print("\nAligning indicators to scene dates …")

indicators = {
    "BRL_USD": brl_usd,
    "SoyMeal_ZM": zm_close,
    "SoyOil_ZL": zl_close,
    "BDRY": bdry,
    "Soybean_ZS": zs,
}
indicators = {name: series for name, series in indicators.items() if not series.empty}

scene_dates = scene_metrics["date"].values

merged = scene_metrics[["date", "scene_id", "anchorage_count", "queue_length_proxy", "total_count"]].copy()
coverage = {}

for name, series in indicators.items():
    idx = pd.DatetimeIndex(series.index)
    vals = series.values.astype(float)
    aligned = []
    gaps = []
    for d in scene_dates:
        diffs = np.abs(idx - d)
        nearest_idx = diffs.argmin()
        gap_days = float(diffs[nearest_idx] / np.timedelta64(1, "D"))
        gaps.append(gap_days)
        if diffs[nearest_idx] <= pd.Timedelta(days=MAX_MATCH_GAP_DAYS):
            aligned.append(vals[nearest_idx])
        else:
            aligned.append(np.nan)
    merged[name] = aligned
    valid_mask = pd.Series(aligned).notna()
    coverage[name] = {
        "n_valid_scenes": int(valid_mask.sum()),
        "first_valid_scene": (
            scene_metrics.loc[valid_mask, "date"].min().strftime("%Y-%m-%d")
            if valid_mask.any()
            else None
        ),
        "last_valid_scene": (
            scene_metrics.loc[valid_mask, "date"].max().strftime("%Y-%m-%d")
            if valid_mask.any()
            else None
        ),
        "max_scene_gap_days": float(max(gaps)) if gaps else None,
        "max_allowed_gap_days": MAX_MATCH_GAP_DAYS,
    }

print(f"  Merged shape: {merged.shape}")
print(f"  Non-null counts:\n{merged[list(indicators.keys())].notna().sum()}")

# ── 5. Spearman correlations with BH correction ─────────────────────────────
print("\n── Spearman correlations (BH-corrected) ──")
econ_cols = list(indicators.keys())
sar_targets = ["anchorage_count", "queue_length_proxy", "total_count"]

all_results = {}
differenced_results = {}
for target in sar_targets:
    corr_df = compute_correlations_with_correction(merged, target, econ_cols)
    corr_df = corr_df.sort_values("rho", key=abs, ascending=False).reset_index(drop=True)
    diff_records = []
    for indicator in corr_df["indicator"].tolist():
        mask = merged[target].notna() & merged[indicator].notna()
        diff_summary = differenced_spearman(
            merged.loc[mask, target].astype(float),
            merged.loc[mask, indicator].astype(float),
        )
        diff_records.append({"indicator": indicator, **diff_summary})
    all_results[target] = corr_df
    differenced_results[target] = diff_records

# Print summary tables
for target, corr_df in all_results.items():
    print(f"\n  Target: {target}")
    print(f"  {'Indicator':<16s}  {'rho':>7s}  {'p_raw':>8s}  {'p_adj':>8s}  {'sig?':>5s}")
    print("  " + "─" * 52)
    for _, row in corr_df.iterrows():
        sig = "YES" if row["significant"] else "no"
        print(f"  {row['indicator']:<16s}  {row['rho']:>+7.3f}  {row['p_value']:>8.4f}  {row['adjusted_p_value']:>8.4f}  {sig:>5s}")

# ── 6. Lead-lag cross-correlations for top-3 indicators ──────────────────────
print("\n── Lead-lag cross-correlations (lags -4 … +4 scenes) ──")
# Pick top-3 by |rho| for queue_length_proxy
queue_corr = all_results["queue_length_proxy"]
top3 = queue_corr.head(3)["indicator"].tolist() if not queue_corr.empty else []
print(f"  Top-3 indicators: {top3}")

lag_range = range(-4, 5)
lead_lag_results = {}

for ind in top3:
    lag_data = {}
    vals_queue = merged["queue_length_proxy"].values.astype(float)
    vals_ind = merged[ind].values.astype(float)
    for lag in lag_range:
        if lag > 0:
            y = vals_queue[lag:]
            x = vals_ind[:len(vals_queue) - lag]
        elif lag < 0:
            y = vals_queue[:len(vals_queue) + lag]
            x = vals_ind[-lag:]
        else:
            y = vals_queue
            x = vals_ind
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 10:
            rho, p = stats.spearmanr(x[mask], y[mask])
        else:
            rho, p = np.nan, np.nan
        lag_data[int(lag)] = {"rho": rho, "p_value": p}
    lead_lag_results[ind] = lag_data

# Print lead-lag table
for ind, ld in lead_lag_results.items():
    print(f"\n  {ind}:")
    print(f"    {'lag':>4s}  {'rho':>7s}  {'p':>8s}")
    for lag in lag_range:
        r = ld[lag]
        print(f"    {lag:>+4d}  {r['rho']:>+7.3f}  {r['p_value']:>8.4f}")

# ── 7. Save results to JSON ─────────────────────────────────────────────────
print("\nSaving results …")
output = {
    "description": "Extended Spearman correlations — Santos port SAR vs economic indicators",
    "n_scenes_total": int(len(merged)),
    "n_scenes_with_any_indicator": int(merged[econ_cols].notna().any(axis=1).sum()) if econ_cols else 0,
    "indicators": econ_cols,
    "alignment": {
        "method": "nearest date",
        "max_gap_days": MAX_MATCH_GAP_DAYS,
        "coverage": coverage,
    },
    "correlations": {},
    "differenced_correlations": differenced_results,
    "lead_lag": {},
}

for target, corr_df in all_results.items():
    output["correlations"][target] = corr_df.to_dict(orient="records")

for ind, ld in lead_lag_results.items():
    output["lead_lag"][ind] = {str(k): v for k, v in ld.items()}

out_path = ROOT / "data/detections/santos/extended_correlations.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=lambda o: None if pd.isna(o) else o)
print(f"  Saved: {out_path}")

# ── 8. Generate figure ───────────────────────────────────────────────────────
print("Generating figure …")

queue_df = all_results["queue_length_proxy"].sort_values("rho", key=abs, ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
ax.set_facecolor("#0e1117")

colors = ["#00d4aa" if sig else "#555555" for sig in queue_df["significant"]]
bars = ax.barh(queue_df["indicator"], queue_df["rho"], color=colors, edgecolor="none", height=0.55)

# rho labels
for bar, rho_val in zip(bars, queue_df["rho"]):
    x_pos = rho_val + (0.01 if rho_val >= 0 else -0.01)
    ha = "left" if rho_val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{rho_val:+.3f}",
            va="center", ha=ha, color="#e0e0e0", fontsize=10, fontweight="bold")

ax.set_xlabel("Spearman ρ", color="#e0e0e0", fontsize=12)
ax.set_title("Santos Port — Queue-Length Proxy vs Economic Indicators\n(green = BH-significant at α=0.05)",
             color="#e0e0e0", fontsize=13, pad=12)
ax.tick_params(colors="#e0e0e0", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#333333")
ax.axvline(0, color="#666666", linewidth=0.8)
ax.set_xlim(min(queue_df["rho"].min() - 0.12, -0.5), max(queue_df["rho"].max() + 0.12, 0.5))

fig_path = ROOT / "data/figures/santos_report/05_extended_correlations.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {fig_path}")

print("\nDone.")
