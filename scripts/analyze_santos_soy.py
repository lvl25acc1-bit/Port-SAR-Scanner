from __future__ import annotations

import os
import itertools
import json
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "backtest" / "santos_soy_vs_anchorage_2024.csv"
OUTPUT_PNG = ROOT / "data" / "backtest" / "santos_soy_vs_anchorage_scatter.png"
OUTPUT_JSON = ROOT / "data" / "backtest" / "santos_soy_vs_anchorage_test.json"


def exact_spearman_left_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_rank = rankdata(x, method="average").astype(float)
    y_rank = rankdata(y, method="average").astype(float)

    x_centered = x_rank - x_rank.mean()
    x_scaled = x_centered / np.sqrt((x_centered**2).sum())
    y_template = np.sort(y_rank)
    y_centered = y_template - y_template.mean()
    y_scaled = y_centered / np.sqrt((y_centered**2).sum())

    observed = float(np.dot(x_scaled, (y_rank - y_rank.mean()) / np.sqrt(((y_rank - y_rank.mean()) ** 2).sum())))
    count = 0
    total = 0
    for perm_idx in itertools.permutations(range(len(y_scaled))):
        stat = float(np.dot(x_scaled, y_scaled[list(perm_idx)]))
        count += stat <= observed
        total += 1
    p_left = count / total
    return observed, p_left


def build_panel(ax: plt.Axes, df: pd.DataFrame, x_col: str, title: str, test: dict) -> None:
    x = df[x_col].to_numpy() / 1_000_000
    y = df["anchorage_mean"].to_numpy()
    labels = [m[5:] for m in df["month"]]
    colors = ["#c65d1e" if m in {"2024-04", "2024-05"} else "#1f4e79" for m in df["month"]]

    ax.scatter(x, y, s=56, c=colors, edgecolors="white", linewidths=0.8, zorder=3)

    if len(df) >= 2:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = coeffs[0] * xs + coeffs[1]
        ax.plot(xs, ys, color="#444444", linewidth=1.5, linestyle="--", zorder=2)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_title(
        f"{title}\n"
        f"rho={test['spearman_rho']:.2f}, one-sided p={test['spearman_left_pvalue']:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("APS monthly volume (Mt)")
    ax.grid(True, alpha=0.25, linewidth=0.7)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    tests: dict[str, dict[str, float | str]] = {}
    for col, label in [
        ("soy_complex_t", "Soy Complex"),
        ("soy_grain_t", "Soy Grain"),
        ("soy_meal_t", "Soy Meal"),
    ]:
        x = df[col].to_numpy(dtype=float)
        y = df["anchorage_mean"].to_numpy(dtype=float)
        rho, p_left = exact_spearman_left_pvalue(x, y)
        pearson = pearsonr(x, y)
        tests[col] = {
            "label": label,
            "spearman_rho": rho,
            "spearman_left_pvalue": p_left,
            "pearson_r": float(pearson.statistic),
            "pearson_two_sided_pvalue": float(pearson.pvalue),
            "n_months": int(len(df)),
            "alternative": "higher soy => lower anchorage",
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    fig.suptitle("Santos 2024: APS Soy Volumes vs SAR Anchorage Counts", fontsize=13)
    for ax, (col, label) in zip(
        axes,
        [
            ("soy_complex_t", "Soy Complex"),
            ("soy_grain_t", "Soy Grain"),
            ("soy_meal_t", "Soy Meal"),
        ],
    ):
        build_panel(ax, df, col, label, tests[col])
    axes[0].set_ylabel("SAR monthly mean anchorage detections")
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_csv": str(INPUT_CSV),
        "output_png": str(OUTPUT_PNG),
        "months": df["month"].tolist(),
        "tests": tests,
        "apr_may_mean_anchorage": float(df[df["month"].isin(["2024-04", "2024-05"])]["anchorage_mean"].mean()),
        "other_months_mean_anchorage": float(
            df[~df["month"].isin(["2024-04", "2024-05"])]["anchorage_mean"].mean()
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
