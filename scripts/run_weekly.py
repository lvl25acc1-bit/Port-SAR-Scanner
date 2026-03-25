"""Incremental weekly update. Designed to run via cron/launchd."""

import logging
from datetime import datetime, timedelta

from portvolume.config import load_config
from portvolume.pipeline import SARMonitoringPipeline
from portvolume.visualization.reports import generate_weekly_report


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_config()
    pipeline = SARMonitoringPipeline(config)

    # Look back 14 days to catch any missed scenes
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")

    logger.info("Running weekly update: %s to %s", start, end)
    results = pipeline.run_full_pipeline(start_date=start, end_date=end)

    # Generate report
    if results:
        import pandas as pd

        corr_df = results.get("correlations", pd.DataFrame())
        granger = results.get("granger", [])
        merged = results.get("merged_data", pd.DataFrame())

        composite_val = (
            merged["composite_index"].iloc[-1]
            if not merged.empty and "composite_index" in merged.columns
            else float("nan")
        )

        report_path = config.indices_dir / "reports" / f"report_{end}.html"
        generate_weekly_report(
            site_summary=pd.DataFrame(),  # TODO: populate from detection summaries
            correlation_df=corr_df,
            granger_results=granger,
            composite_value=composite_val,
            n_scenes=0,
            output_path=report_path,
        )
        logger.info("Weekly report: %s", report_path)


if __name__ == "__main__":
    main()
