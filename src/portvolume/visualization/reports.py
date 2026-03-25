"""Weekly HTML report generation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
    <title>Port Volume Weekly Report — {report_date}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
               max-width: 1000px; margin: 40px auto; padding: 0 20px;
               color: #333; }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        h2 {{ color: #2d3748; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
        th {{ background: #f7fafc; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f7fafc; }}
        .metric {{ display: inline-block; padding: 10px 20px; margin: 5px;
                   background: #ebf8ff; border-radius: 8px; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #2b6cb0; }}
        .metric .label {{ font-size: 12px; color: #718096; }}
        .significant {{ color: #22543d; font-weight: bold; }}
        .not-significant {{ color: #9b2c2c; }}
    </style>
</head>
<body>
    <h1>Port Volume Weekly Report</h1>
    <p>Generated: {report_date}</p>

    <div>
        <div class="metric">
            <div class="value">{n_sites}</div>
            <div class="label">Active Sites</div>
        </div>
        <div class="metric">
            <div class="value">{n_scenes}</div>
            <div class="label">Scenes Processed</div>
        </div>
        <div class="metric">
            <div class="value">{composite_value}</div>
            <div class="label">Composite Index</div>
        </div>
    </div>

    <h2>Site Activity Summary</h2>
    {site_table}

    <h2>Economic Correlations</h2>
    {correlation_table}

    <h2>Granger Causality Results</h2>
    {granger_table}
</body>
</html>
"""


def generate_weekly_report(
    site_summary: pd.DataFrame,
    correlation_df: pd.DataFrame,
    granger_results: list[dict],
    composite_value: float,
    n_scenes: int,
    output_path: Path,
) -> Path:
    """Generate a weekly HTML report.

    Parameters
    ----------
    site_summary : DataFrame with per-site weekly stats.
    correlation_df : Output of correlation.compute_correlations().
    granger_results : List of dicts with Granger test results.
    composite_value : Latest composite index value.
    n_scenes : Total scenes processed this week.
    output_path : Where to write the HTML file.
    """
    report_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    site_table = site_summary.to_html(index=False, classes="summary") if not site_summary.empty else "<p>No site data available.</p>"

    if not correlation_df.empty:
        # Format p-values with significance markers
        corr_display = correlation_df.copy()
        corr_display["pearson_r"] = corr_display["pearson_r"].map("{:.3f}".format)
        corr_display["pearson_p"] = corr_display["pearson_p"].apply(
            lambda p: f'<span class="significant">{p:.4f} *</span>'
            if p < 0.05
            else f'<span class="not-significant">{p:.4f}</span>'
        )
        correlation_table = corr_display.to_html(
            index=False, escape=False, classes="correlations"
        )
    else:
        correlation_table = "<p>No correlation data available.</p>"

    if granger_results:
        granger_df = pd.DataFrame(granger_results)
        granger_table = granger_df.to_html(index=False, classes="granger")
    else:
        granger_table = "<p>No Granger causality results available.</p>"

    html = REPORT_TEMPLATE.format(
        report_date=report_date,
        n_sites=len(site_summary) if not site_summary.empty else 0,
        n_scenes=n_scenes,
        composite_value=f"{composite_value:.3f}" if pd.notna(composite_value) else "N/A",
        site_table=site_table,
        correlation_table=correlation_table,
        granger_table=granger_table,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Report written to %s", output_path)

    return output_path
