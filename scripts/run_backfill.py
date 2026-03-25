"""One-shot backfill for a historical date range."""

import argparse
import logging
import sys

from portvolume.config import load_config
from portvolume.pipeline import SARMonitoringPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Sentinel-1 SAR monitoring pipeline"
    )
    parser.add_argument(
        "--start", default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default=None, help="End date (YYYY-MM-DD), default=today"
    )
    parser.add_argument(
        "--sites", nargs="*", default=None, help="Specific site IDs to process"
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        choices=["acquire", "preprocess", "detect", "index", "economic", "analyze"],
        help="Run only specific pipeline steps",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config()
    pipeline = SARMonitoringPipeline(config)

    if args.steps:
        if "acquire" in args.steps:
            pipeline.run_acquisition(
                start_date=args.start, end_date=args.end, site_ids=args.sites
            )
        if "preprocess" in args.steps:
            pipeline.run_preprocessing(site_ids=args.sites)
        if "detect" in args.steps:
            pipeline.run_detection(site_ids=args.sites)
        if "index" in args.steps:
            pipeline.run_index_construction()
        if "economic" in args.steps:
            pipeline.run_economic_update()
        if "analyze" in args.steps:
            pipeline.run_analysis()
    else:
        pipeline.run_full_pipeline(start_date=args.start, end_date=args.end, site_ids=args.sites)


if __name__ == "__main__":
    main()
