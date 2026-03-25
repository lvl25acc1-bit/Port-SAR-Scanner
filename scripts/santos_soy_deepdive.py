"""Santos soy seasonality deep-dive: vessel counts vs soybean futures."""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SANTOS_CSV = ROOT / "data" / "detections" / "santos" / "summaries.csv"
YF_PATH = ROOT / "data" / "economic" / "yfinance_data.parquet"
OUT_PATH = ROOT / "data" / "detections" / "santos" / "soy_deepdive_results.json"


def main():
    print("=" * 70)
    print(" SANTOS SOY SEASONALITY DEEP-DIVE")
    print("=" * 70)

    # --- Load data ---
    santos = pd.read_csv(SANTOS_CSV)
    santos["date"] = pd.to_datetime(santos["timestamp"]).dt.normalize()
    santos["month"] = santos["date"].dt.month
    santos["month_name"] = santos["date"].dt.strftime("%b")
    santos["is_harvest"] = santos["month"].isin([2, 3, 4, 5]).astype(int)
    santos = santos.sort_values("date").reset_index(drop=True)
    n = len(santos)

    yf = pd.read_parquet(YF_PATH)
    yf.index = pd.to_datetime(yf.index)

    # Match soy price to each scene date (nearest available)
    soy_prices = []
    for d in santos["date"]:
        idx = yf.index.get_indexer([d], method="nearest")[0]
        soy_prices.append(yf.iloc[idx]["ZS=F"])
    santos["soy_price"] = soy_prices

    # --- Timeline table ---
    print("\n" + "-" * 70)
    print(" TIMELINE: Santos Vessel Counts vs Soybean Futures (ZS=F)")
    print("-" * 70)
    print(f"{'Date':>12s}  {'Vessels':>7s}  {'Soy $/bu':>9s}  {'Month':>5s}  {'Harvest':>7s}")
    print("-" * 70)
    for _, r in santos.iterrows():
        print(f"{r['date'].strftime('%Y-%m-%d'):>12s}  {r['vessel_count']:>7.0f}  {r['soy_price']:>9.2f}  {r['month_name']:>5s}  {'YES' if r['is_harvest'] else 'no':>7s}")

    # --- Monthly profile ---
    print("\n" + "-" * 70)
    print(" MONTHLY VESSEL COUNT PROFILE")
    print("-" * 70)
    monthly = santos.groupby("month_name", sort=False).agg(
        n_scenes=("vessel_count", "count"),
        mean_vessels=("vessel_count", "mean"),
        std_vessels=("vessel_count", "std"),
        mean_soy=("soy_price", "mean"),
    )
    # Sort by month order
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    monthly = monthly.reindex(month_order).dropna(subset=["n_scenes"])
    print(f"{'Month':>5s}  {'N':>3s}  {'Mean Vessels':>12s}  {'Std':>6s}  {'Mean Soy $':>10s}")
    for m, r in monthly.iterrows():
        std_str = f"{r['std_vessels']:.1f}" if pd.notna(r["std_vessels"]) else "  n/a"
        print(f"{m:>5s}  {r['n_scenes']:>3.0f}  {r['mean_vessels']:>12.1f}  {std_str:>6s}  {r['mean_soy']:>10.2f}")

    # --- Harvest vs non-harvest ---
    print("\n" + "-" * 70)
    print(" HARVEST SEASON TEST (Feb-May vs Jun-Aug+Jan)")
    print("-" * 70)
    harvest = santos[santos["is_harvest"] == 1]["vessel_count"]
    non_harvest = santos[santos["is_harvest"] == 0]["vessel_count"]
    print(f"  Harvest (n={len(harvest)}):     mean={harvest.mean():.1f}, std={harvest.std():.1f}")
    print(f"  Non-harvest (n={len(non_harvest)}): mean={non_harvest.mean():.1f}, std={non_harvest.std():.1f}")

    # Mann-Whitney U (non-parametric, better for small n)
    u_stat, u_p = stats.mannwhitneyu(harvest, non_harvest, alternative="two-sided")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f}")

    # Effect size: rank-biserial correlation
    n1, n2 = len(harvest), len(non_harvest)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    print(f"  Rank-biserial r: {rank_biserial:.3f}")

    # Permutation test for mean difference
    observed_diff = harvest.mean() - non_harvest.mean()
    all_counts = santos["vessel_count"].values
    labels = santos["is_harvest"].values
    n_perm = 10000
    rng = np.random.default_rng(42)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_diffs[i] = all_counts[perm_labels == 1].mean() - all_counts[perm_labels == 0].mean()
    perm_p = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    print(f"  Observed diff: {observed_diff:+.1f} vessels")
    print(f"  Permutation p (10k): {perm_p:.4f}")

    # --- Correlations ---
    print("\n" + "-" * 70)
    print(" CORRELATIONS (n=20)")
    print("-" * 70)

    # Spearman
    rho_raw, p_raw = stats.spearmanr(santos["vessel_count"], santos["soy_price"])
    print(f"  vessel_count vs ZS=F:  Spearman rho={rho_raw:.4f}, p={p_raw:.4f}")

    # Exact permutation test for Spearman (n=20 is feasible with Monte Carlo)
    n_perm_corr = 50000
    rng2 = np.random.default_rng(123)
    perm_rhos = np.empty(n_perm_corr)
    vc = santos["vessel_count"].values
    sp = santos["soy_price"].values
    for i in range(n_perm_corr):
        perm_rhos[i] = stats.spearmanr(rng2.permutation(vc), sp).statistic
    exact_p = np.mean(np.abs(perm_rhos) >= np.abs(rho_raw))
    print(f"  Permutation p (50k):   {exact_p:.4f}")

    # Changes (diff)
    if n > 2:
        d_vc = np.diff(santos["vessel_count"].values)
        d_sp = np.diff(santos["soy_price"].values)
        rho_diff, p_diff = stats.spearmanr(d_vc, d_sp)
        print(f"  delta(vessels) vs delta(soy): rho={rho_diff:.4f}, p={p_diff:.4f}")

    # --- Lead-lag (scene-level) ---
    print("\n" + "-" * 70)
    print(" LEAD-LAG (scene intervals, not weeks)")
    print("-" * 70)
    for lag in range(-3, 4):
        if lag == 0:
            r, p = rho_raw, p_raw
        elif lag > 0:
            r, p = stats.spearmanr(vc[lag:], sp[:-lag])
        else:
            r, p = stats.spearmanr(vc[:lag], sp[-lag:])
        sig = "*" if p < 0.05 else " "
        print(f"  lag={lag:+d} scenes:  rho={r:+.4f}  p={p:.4f} {sig}")

    # --- Bootstrap CI for the main correlation ---
    print("\n" + "-" * 70)
    print(" BOOTSTRAP 95% CI FOR SPEARMAN rho (vessel_count vs ZS=F)")
    print("-" * 70)
    n_boot = 10000
    rng3 = np.random.default_rng(456)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng3.integers(0, n, size=n)
        boot_rhos[i] = stats.spearmanr(vc[idx], sp[idx]).statistic
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"  rho = {rho_raw:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  CI includes zero: {'YES — not significant' if ci_lo <= 0 <= ci_hi else 'NO — significant'}")

    # --- Save results ---
    results = {
        "n_scenes": n,
        "date_range": f"{santos['date'].min().strftime('%Y-%m-%d')} to {santos['date'].max().strftime('%Y-%m-%d')}",
        "vessel_count_stats": {
            "mean": float(santos["vessel_count"].mean()),
            "std": float(santos["vessel_count"].std()),
            "cv_pct": float(santos["vessel_count"].std() / santos["vessel_count"].mean() * 100),
        },
        "harvest_test": {
            "harvest_mean": float(harvest.mean()),
            "non_harvest_mean": float(non_harvest.mean()),
            "observed_diff": float(observed_diff),
            "mann_whitney_p": float(u_p),
            "rank_biserial_r": float(rank_biserial),
            "permutation_p": float(perm_p),
        },
        "correlations": {
            "vessel_count_vs_soy": {
                "spearman_rho": float(rho_raw),
                "asymptotic_p": float(p_raw),
                "permutation_p": float(exact_p),
                "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
            },
            "delta_vessels_vs_delta_soy": {
                "spearman_rho": float(rho_diff) if n > 2 else None,
                "p_value": float(p_diff) if n > 2 else None,
            },
        },
        "monthly_profile": {
            m: {"n": int(r["n_scenes"]), "mean_vessels": round(float(r["mean_vessels"]), 1)}
            for m, r in monthly.iterrows()
        },
        "verdict": "",
    }

    # Verdict
    if abs(rho_raw) >= 0.3 and exact_p < 0.05:
        results["verdict"] = f"SIGNAL DETECTED: rho={rho_raw:.3f}, permutation p={exact_p:.4f}"
    elif abs(rho_raw) >= 0.2:
        results["verdict"] = f"WEAK SIGNAL: rho={rho_raw:.3f}, but needs more data (n={n})"
    else:
        results["verdict"] = f"NO SIGNAL: rho={rho_raw:.3f}, p={exact_p:.4f}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUT_PATH}")

    print("\n" + "=" * 70)
    print(f" VERDICT: {results['verdict']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
