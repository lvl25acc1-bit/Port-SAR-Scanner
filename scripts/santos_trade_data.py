#!/usr/bin/env python3
"""
santos_trade_data.py
====================
Builds a scenario-analysis view of Santos anchorage counts against trade proxies.

This script mixes:
  - actual SAR monthly counts from analysis_results.json
  - optional real macro proxies from FRED
  - synthetic throughput / export seasonality curves

Synthetic series are illustrative scenarios, not ground-truth validation.

Outputs:
  - data/detections/santos/trade_correlations.json
  - data/figures/santos_report/06_trade_validation.png
"""

import json
import os
import tempfile
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

_CACHE_DIR = tempfile.mkdtemp(prefix="mpl-")
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SAR_PATH = ROOT / "data" / "detections" / "santos" / "analysis_results.json"
OUT_JSON = ROOT / "data" / "detections" / "santos" / "trade_correlations.json"
OUT_FIG  = ROOT / "data" / "figures" / "santos_report" / "06_trade_validation.png"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# ── colours / theme ──────────────────────────────────────────────────────────
C_SAR   = "#00d4aa"
C_THRU  = "#ffe66d"
C_BG    = "#0e1117"
C_TEXT  = "#e6edf3"
C_GRID  = "#30363d"
SCENARIO_START = pd.Timestamp("2023-10-01")
SCENARIO_END = pd.Timestamp("2026-03-31")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load SAR scene metrics and aggregate to monthly
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1 — Loading SAR scene metrics from analysis_results.json")
print("=" * 72)

with open(SAR_PATH) as f:
    sar_data = json.load(f)

scenes = pd.DataFrame(sar_data["scene_metrics"])
scenes["date"] = pd.to_datetime(scenes["date"])
scenes["month"] = scenes["date"].dt.to_period("M")
analysis_scenes = scenes[(scenes["date"] >= SCENARIO_START) & (scenes["date"] <= SCENARIO_END)].copy()

print(f"  Loaded {len(scenes)} scenes, {scenes['date'].min().date()} → {scenes['date'].max().date()}")
print(
    "  Scenario window: "
    f"{SCENARIO_START.date()} → {SCENARIO_END.date()} "
    f"({len(analysis_scenes)} scenes)"
)

# Monthly aggregation
monthly_sar = (
    analysis_scenes.groupby("month")
    .agg(
        anchorage_mean=("anchorage_count", "mean"),
        anchorage_max=("anchorage_count", "max"),
        total_mean=("total_count", "mean"),
        queue_mean=("queue_length_proxy", "mean"),
        n_scenes=("anchorage_count", "count"),
    )
    .reset_index()
)
monthly_sar["month_ts"] = monthly_sar["month"].dt.to_timestamp()

print(f"  Aggregated to {len(monthly_sar)} months (min scenes/month = {monthly_sar['n_scenes'].min()}, "
      f"max = {monthly_sar['n_scenes'].max()})")
print(monthly_sar[["month", "anchorage_mean", "n_scenes"]].to_string(index=False))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Fetch Brazil export data from FRED
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 2 — Fetching FRED macro data (Brazil exports / soy proxies)")
print("=" * 72)

fred_results = {}
try:
    from fredapi import Fred
    fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))

    # Try several series in order of relevance
    series_candidates = [
        ("BRAZILSOYBEANEXP",   "Brazil soybean exports"),
        ("XTEXVA01BRM667S",    "Brazil total exports, value (USD)"),
        ("PCU311224311224",     "Soybean oil mfg PPI (US proxy)"),
    ]

    for sid, label in series_candidates:
        try:
            s = fred.get_series(sid, observation_start="2023-01-01", observation_end="2026-04-01")
            s = s.dropna()
            if len(s) > 0:
                fred_results[sid] = {"label": label, "series": s}
                print(f"  ✓ {sid}: {label}  ({len(s)} obs, {s.index[0].date()} → {s.index[-1].date()})")
            else:
                print(f"  ✗ {sid}: empty after dropna")
        except Exception as e:
            print(f"  ✗ {sid}: {e}")
except Exception as e:
    print(f"  FRED unavailable: {e}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Build synthetic monthly datasets
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 3 — Building synthetic reference datasets")
print("=" * 72)

# Date range matching SAR data
months_range = pd.period_range("2023-10", "2026-03", freq="M")
synth = pd.DataFrame({"month": months_range})
synth["month_ts"] = synth["month"].dt.to_timestamp()
synth["year"] = synth["month_ts"].dt.year
synth["mo"]   = synth["month_ts"].dt.month

# ── 3a. Synthetic Brazil soy exports ────────────────────────────────────────
# Brazil exports ~90M tonnes/yr with strong Mar-Jun seasonality
soy_monthly_pattern = {
    1: 3.0,  2: 5.5,  3: 12.0, 4: 14.5,
    5: 13.0, 6: 10.5, 7: 8.0,  8: 6.5,
    9: 5.0, 10: 4.5, 11: 4.0, 12: 3.5,
}  # millions of tonnes — sums to ~90M

np.random.seed(42)
synth["soy_export_mt"] = synth["mo"].map(soy_monthly_pattern)
# Year-on-year growth ~4%
growth = 1.04 ** (synth["year"] - 2024)
synth["soy_export_mt"] = synth["soy_export_mt"] * growth
# Add realistic noise
synth["soy_export_mt"] += np.random.normal(0, 0.8, len(synth))
synth["soy_export_mt"] = synth["soy_export_mt"].clip(lower=1.0)

print(f"  Synthetic soy exports: {len(synth)} months, "
      f"annual total ~{synth['soy_export_mt'].sum() / (len(synth)/12):.0f}M tonnes")

# ── 3b. Synthetic Santos grain throughput ────────────────────────────────────
grain_monthly_pattern = {
    1: 6.0,  2: 8.0,  3: 12.0, 4: 14.0,
    5: 13.0, 6: 10.0, 7: 8.0,  8: 7.0,
    9: 6.0, 10: 5.0, 11: 5.0, 12: 5.0,
}  # millions of tonnes

synth["grain_throughput_mt"] = synth["mo"].map(grain_monthly_pattern)
growth_tp = 1.03 ** (synth["year"] - 2024)
synth["grain_throughput_mt"] = synth["grain_throughput_mt"] * growth_tp
synth["grain_throughput_mt"] += np.random.normal(0, 1.0, len(synth))
synth["grain_throughput_mt"] = synth["grain_throughput_mt"].clip(lower=2.0)

print(f"  Synthetic grain throughput: annual total ~{synth['grain_throughput_mt'].sum() / (len(synth)/12):.0f}M tonnes")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Merge and compute correlations
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 4 — Correlations: SAR anchorage count vs scenario/proxy series")
print("=" * 72)

# Merge SAR monthly with synthetic
merged = synth.merge(monthly_sar[["month", "anchorage_mean", "queue_mean", "total_mean", "n_scenes"]],
                     on="month", how="left")

# Mark months with no SAR scenes
has_sar = merged["anchorage_mean"].notna()
print(f"  Months with SAR data: {has_sar.sum()} / {len(merged)}")

corr_results = {}

# Correlations with synthetic grain throughput
valid = merged.dropna(subset=["anchorage_mean", "grain_throughput_mt"])
if len(valid) >= 5:
    rho, pval = stats.spearmanr(valid["anchorage_mean"], valid["grain_throughput_mt"])
    corr_results["sar_vs_grain_throughput"] = {
        "spearman_rho": round(rho, 4),
        "p_value": round(pval, 6),
        "n_months": len(valid),
        "note": "synthetic grain throughput"
    }
    print(f"  SAR anchorage vs grain throughput: rho={rho:.3f}, p={pval:.4f} (n={len(valid)})")

# Correlations with synthetic soy exports
valid2 = merged.dropna(subset=["anchorage_mean", "soy_export_mt"])
if len(valid2) >= 5:
    rho2, pval2 = stats.spearmanr(valid2["anchorage_mean"], valid2["soy_export_mt"])
    corr_results["sar_vs_soy_exports"] = {
        "spearman_rho": round(rho2, 4),
        "p_value": round(pval2, 6),
        "n_months": len(valid2),
        "note": "synthetic soy exports"
    }
    print(f"  SAR anchorage vs soy exports:      rho={rho2:.3f}, p={pval2:.4f} (n={len(valid2)})")

# Correlations with FRED series (if available)
for sid, info in fred_results.items():
    s = info["series"].copy()
    s.index = s.index.to_period("M")
    fred_monthly = s.groupby(s.index).mean().reset_index()
    fred_monthly.columns = ["month", "fred_value"]

    m = merged.merge(fred_monthly, on="month", how="inner")
    m = m.dropna(subset=["anchorage_mean", "fred_value"])
    if len(m) >= 5:
        rho_f, pval_f = stats.spearmanr(m["anchorage_mean"], m["fred_value"])
        key = f"sar_vs_{sid}"
        corr_results[key] = {
            "spearman_rho": round(rho_f, 4),
            "p_value": round(pval_f, 6),
            "n_months": len(m),
            "label": info["label"],
        }
        print(f"  SAR anchorage vs {sid}: rho={rho_f:.3f}, p={pval_f:.4f} (n={len(m)})")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Lead-lag analysis: does SAR anchorage LEAD exports?
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 5 — Lead-lag analysis (SAR anchorage vs synthetic grain scenario)")
print("=" * 72)

lags_to_test = list(range(-3, 4))  # -3 to +3
lead_lag = {}

# Use merged dataframe, work with arrays indexed by position
sar_series = merged.set_index("month")["anchorage_mean"]
grain_series = merged.set_index("month")["grain_throughput_mt"]

for lag in lags_to_test:
    shifted_sar = sar_series.shift(lag)  # positive lag = SAR leads grain by |lag| months
    valid_idx = shifted_sar.notna() & grain_series.notna()
    n_valid = valid_idx.sum()
    if n_valid >= 5:
        rho_l, pval_l = stats.spearmanr(
            shifted_sar[valid_idx].values,
            grain_series[valid_idx].values,
        )
        lead_lag[lag] = {"spearman_rho": round(rho_l, 4), "p_value": round(pval_l, 6), "n": int(n_valid)}
        direction = "SAR leads" if lag > 0 else ("concurrent" if lag == 0 else "SAR lags")
        print(f"  lag={lag:+d} ({direction:12s}): rho={rho_l:+.3f}, p={pval_l:.4f}, n={n_valid}")
    else:
        lead_lag[lag] = {"spearman_rho": None, "p_value": None, "n": int(n_valid)}
        print(f"  lag={lag:+d}: insufficient data (n={n_valid})")

best_lag = max(lead_lag, key=lambda k: abs(lead_lag[k]["spearman_rho"] or 0))
print(f"\n  Best absolute correlation at lag={best_lag:+d}: rho={lead_lag[best_lag]['spearman_rho']}")
print()

# Also run lead-lag against soy exports
lead_lag_soy = {}
soy_series = merged.set_index("month")["soy_export_mt"]
for lag in lags_to_test:
    shifted_sar = sar_series.shift(lag)
    valid_idx = shifted_sar.notna() & soy_series.notna()
    n_valid = valid_idx.sum()
    if n_valid >= 5:
        rho_l, pval_l = stats.spearmanr(shifted_sar[valid_idx].values, soy_series[valid_idx].values)
        lead_lag_soy[lag] = {"spearman_rho": round(rho_l, 4), "p_value": round(pval_l, 6), "n": int(n_valid)}

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Save results JSON
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 6 — Saving trade_correlations.json")
print("=" * 72)

output = {
    "generated": datetime.utcnow().isoformat() + "Z",
    "description": "Scenario analysis: SAR vessel anchorage counts vs trade proxies and synthetic throughput curves",
    "scenario_only": True,
    "data_sources": {
        "sar": {
            "file": "analysis_results.json",
            "n_scenes_total_source": len(scenes),
            "n_scenes_analysis_window": len(analysis_scenes),
            "date_range_total_source": [str(scenes["date"].min().date()), str(scenes["date"].max().date())],
            "analysis_window": [str(SCENARIO_START.date()), str(SCENARIO_END.date())],
            "monthly_aggregation": "mean anchorage_count per calendar month",
        },
        "synthetic_grain_throughput": {
            "note": "SYNTHETIC — based on known Santos throughput patterns (~140M t/yr, 40% grain)",
            "monthly_pattern_mt": grain_monthly_pattern,
            "yearly_growth": 0.03,
        },
        "synthetic_soy_exports": {
            "note": "SYNTHETIC — based on known Brazil soy export seasonality (~90M t/yr)",
            "monthly_pattern_mt": soy_monthly_pattern,
            "yearly_growth": 0.04,
        },
        "fred_series": {sid: {"label": info["label"], "n_obs": len(info["series"])}
                        for sid, info in fred_results.items()},
    },
    "monthly_data": merged.drop(columns=["month"]).assign(
        month=merged["month"].astype(str),
        month_ts=merged["month_ts"].dt.strftime("%Y-%m-%d"),
    ).to_dict(orient="records"),
    "correlations": corr_results,
    "lead_lag_analysis": {
        "description": "Positive lag = SAR leads synthetic grain throughput by N months",
        "vs_grain_throughput": {str(k): v for k, v in lead_lag.items()},
        "vs_soy_exports": {str(k): v for k, v in lead_lag_soy.items()},
        "best_lag_grain": best_lag,
        "best_rho_grain": lead_lag[best_lag]["spearman_rho"],
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Saved → {OUT_JSON}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Generate figure
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 7 — Generating 06_trade_validation.png")
print("=" * 72)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                gridspec_kw={"width_ratios": [1.6, 1]})
fig.patch.set_facecolor(C_BG)

for ax in (ax1, ax2):
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)

# ── Left panel: dual-axis time series ────────────────────────────────────────
plot_df = merged.dropna(subset=["anchorage_mean"]).copy()
x_dates = plot_df["month_ts"]

bar_width = 20  # days
bars = ax1.bar(x_dates, plot_df["anchorage_mean"],
               width=bar_width, color=C_SAR, alpha=0.7, label="SAR anchorage count (mean)", zorder=3)
ax1.set_ylabel("SAR anchorage vessels (monthly mean)", color=C_SAR, fontsize=10)
ax1.tick_params(axis="y", labelcolor=C_SAR)
ax1.set_xlabel("Month", color=C_TEXT, fontsize=10)

ax1b = ax1.twinx()
ax1b.plot(synth["month_ts"], synth["grain_throughput_mt"],
          color=C_THRU, linewidth=2.5, marker="o", markersize=4, label="Synthetic grain scenario", zorder=4)
ax1b.set_ylabel("Grain throughput (M tonnes)", color=C_THRU, fontsize=10)
ax1b.tick_params(axis="y", labelcolor=C_THRU)
ax1b.spines["right"].set_color(C_THRU)
ax1b.spines["top"].set_visible(False)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=8, facecolor=C_BG, edgecolor=C_GRID,
           labelcolor=C_TEXT)

ax1.set_title("SAR Vessel Counts vs Synthetic Santos Grain Scenario", color=C_TEXT, fontsize=12, fontweight="bold")
ax1.grid(axis="y", color=C_GRID, alpha=0.4, linewidth=0.5)

# Rotate x-axis labels
for label in ax1.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ── Right panel: lead-lag bar chart ──────────────────────────────────────────
lag_keys = sorted(lead_lag.keys())
lag_rhos = [lead_lag[k]["spearman_rho"] or 0 for k in lag_keys]
lag_labels = [f"{k:+d}" for k in lag_keys]

colors = [C_SAR if k == best_lag else "#4a9eff" for k in lag_keys]
ax2.bar(lag_labels, lag_rhos, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
ax2.axhline(0, color=C_TEXT, linewidth=0.5, alpha=0.5)
ax2.set_xlabel("Lag (months, +N = SAR leads)", color=C_TEXT, fontsize=10)
ax2.set_ylabel("Spearman ρ", color=C_TEXT, fontsize=10)
ax2.set_title("Lead-Lag Correlation\n(SAR anchorage vs synthetic grain scenario)", color=C_TEXT, fontsize=12, fontweight="bold")
ax2.grid(axis="y", color=C_GRID, alpha=0.4, linewidth=0.5)

# Annotate best lag
ax2.annotate(f"best: lag={best_lag:+d}\nρ={lead_lag[best_lag]['spearman_rho']:.3f}",
             xy=(lag_labels[lag_keys.index(best_lag)], lead_lag[best_lag]["spearman_rho"]),
             xytext=(0, 15), textcoords="offset points", ha="center",
             fontsize=9, color=C_SAR, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_SAR, lw=1.2))

# Watermark
fig.text(0.5, 0.01,
         "Synthetic throughput curves are illustrative only | SAR data shown for Oct 2023 – Mar 2026",
         ha="center", fontsize=7, color="#666", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(OUT_FIG, dpi=180, facecolor=C_BG, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_FIG}")
print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  SAR scenes:       {len(scenes)}")
print(f"  Monthly periods:  {len(monthly_sar)} (with SAR data)")
print(f"  FRED series used: {len(fred_results)}")
for key, val in corr_results.items():
    print(f"  {key}: ρ={val['spearman_rho']:.3f}  (p={val['p_value']:.4f})")
print(f"  Best lead-lag:    lag={best_lag:+d}, ρ={lead_lag[best_lag]['spearman_rho']:.3f}")
print(f"\n  Output JSON: {OUT_JSON}")
print(f"  Output figure: {OUT_FIG}")
print("Done.")
