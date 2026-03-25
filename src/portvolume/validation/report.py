"""Validation report generation: scorecards, JSON, and markdown summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from portvolume.validation.metrics import ValidationResult, validate_improvement

logger = logging.getLogger(__name__)


def generate_scorecard(results: list[ValidationResult]) -> pd.DataFrame:
    """Generate a concise scorecard DataFrame.

    Columns: [port_id, signal_name, reference_name, rho, p_value,
              optimal_lag, passes, directional_accuracy]
    Sorted by |rho| descending.
    """
    if not results:
        return pd.DataFrame(
            columns=[
                "port_id", "signal_name", "reference_name", "rho", "p_value",
                "optimal_lag", "passes", "directional_accuracy",
            ]
        )

    records = []
    for r in results:
        records.append({
            "port_id": r.port_id,
            "signal_name": r.signal_name,
            "reference_name": r.reference_name,
            "rho": round(r.spearman_rho, 4),
            "p_value": round(r.spearman_p, 6),
            "optimal_lag": r.optimal_lag_weeks,
            "passes": r.passes_threshold,
            "directional_accuracy": round(r.directional_accuracy, 4),
        })

    df = pd.DataFrame(records)
    df["abs_rho"] = df["rho"].abs()
    df = df.sort_values("abs_rho", ascending=False).drop(columns=["abs_rho"]).reset_index(drop=True)
    return df


def generate_validation_report(
    results: list[ValidationResult],
    output_dir: Path,
    baseline_results: list[ValidationResult] | None = None,
) -> Path:
    """Generate structured validation report.

    Creates:
    - output_dir/validation_scorecard.csv
    - output_dir/validation_report.json (full results)
    - output_dir/validation_summary.md (human-readable markdown)

    Returns output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Scorecard CSV ---
    scorecard = generate_scorecard(results)
    scorecard.to_csv(output_dir / "validation_scorecard.csv", index=False)

    # --- Full JSON ---
    report_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_results": len(results),
        "n_ports": len({r.port_id for r in results}),
        "n_signals": len({r.signal_name for r in results}),
        "n_passing": sum(1 for r in results if r.passes_threshold),
        "results": [r.to_dict() for r in results],
    }

    if baseline_results:
        improvements = _compute_improvements(baseline_results, results)
        report_data["baseline_comparison"] = improvements

    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    # --- Markdown summary ---
    md = _build_markdown(results, baseline_results)
    with open(output_dir / "validation_summary.md", "w") as f:
        f.write(md)

    logger.info("Validation report written to %s", output_dir)
    return output_dir


def _compute_improvements(
    baseline: list[ValidationResult],
    current: list[ValidationResult],
) -> list[dict]:
    """Match baseline and current results by (port_id, signal_name, reference_name)."""
    baseline_map = {
        (r.port_id, r.signal_name, r.reference_name): r for r in baseline
    }
    improvements = []
    for r in current:
        key = (r.port_id, r.signal_name, r.reference_name)
        if key in baseline_map:
            imp = validate_improvement(baseline_map[key], r)
            imp["port_id"] = r.port_id
            imp["signal_name"] = r.signal_name
            imp["reference_name"] = r.reference_name
            improvements.append(imp)
    return improvements


def _build_markdown(
    results: list[ValidationResult],
    baseline_results: list[ValidationResult] | None = None,
) -> str:
    """Build human-readable markdown summary."""
    lines: list[str] = []

    ports = sorted({r.port_id for r in results})
    signals = sorted({r.signal_name for r in results})
    n_pass = sum(1 for r in results if r.passes_threshold)

    lines.append("# Validation Summary")
    lines.append("")
    lines.append(f"- **Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Ports tested**: {len(ports)} ({', '.join(ports)})")
    lines.append(f"- **Signals tested**: {len(signals)} ({', '.join(signals)})")
    lines.append(f"- **Total comparisons**: {len(results)}")
    lines.append(f"- **Passing threshold**: {n_pass}/{len(results)}")
    lines.append("")

    # Summary table
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Port | Signal | Reference | Rho | p-value | Lag | Dir.Acc | Verdict |")
    lines.append("|------|--------|-----------|-----|---------|-----|---------|---------|")
    for r in sorted(results, key=lambda x: -abs(x.spearman_rho)):
        verdict = "PASS" if r.passes_threshold else "FAIL"
        lines.append(
            f"| {r.port_id} | {r.signal_name} | {r.reference_name} | "
            f"{r.spearman_rho:.3f} | {r.spearman_p:.4f} | {r.optimal_lag_weeks} | "
            f"{r.directional_accuracy:.2f} | {verdict} |"
        )
    lines.append("")

    # Per-port detail
    lines.append("## Per-Port Details")
    lines.append("")
    for port_id in ports:
        port_results = [r for r in results if r.port_id == port_id]
        lines.append(f"### {port_id}")
        lines.append("")
        for r in port_results:
            verdict = "PASS" if r.passes_threshold else "FAIL"
            lines.append(f"- **{r.signal_name}** vs {r.reference_name}: "
                         f"rho={r.spearman_rho:.3f} (p={r.spearman_p:.4f}), "
                         f"RMSE={r.rmse:.3f}, dir.acc={r.directional_accuracy:.2f} "
                         f"[{verdict}]")
            if r.note:
                lines.append(f"  - Note: {r.note}")
        lines.append("")

    # Baseline comparison
    if baseline_results:
        improvements = _compute_improvements(baseline_results, results)
        if improvements:
            lines.append("## Improvement over Baseline")
            lines.append("")
            for imp in improvements:
                direction = "improved" if imp["rho_improved"] else "degraded"
                lines.append(
                    f"- {imp['port_id']} / {imp['signal_name']} / {imp['reference_name']}: "
                    f"delta_rho={imp['delta_rho']:+.4f} ({direction}, "
                    f"{imp['improvement_pct']:+.1f}%)"
                )
            lines.append("")

    # Overall verdict
    lines.append("## Overall Verdict")
    lines.append("")
    if len(results) == 0:
        lines.append("No validation results to evaluate.")
    elif n_pass == 0:
        lines.append("**No signals pass the validation threshold.** "
                      "The SAR signal does not demonstrate statistically significant "
                      "correlation with any reference dataset.")
    elif n_pass == len(results):
        lines.append("**All signals pass the validation threshold.** "
                      "The SAR signal shows consistent correlation with reference data.")
    else:
        lines.append(f"**{n_pass}/{len(results)} signals pass the validation threshold.** "
                      "Partial correlation observed; further investigation recommended.")

    lines.append("")
    return "\n".join(lines)
