"""Acquire extended Santos SAR data (2020-2026) and run detection pipeline.

This script:
1. Searches Planetary Computer for all Sentinel-1 scenes over Santos
2. Downloads only new scenes (skips already downloaded)
3. Preprocesses and runs CFAR detection
4. Saves per-scene detection parquets and updated summaries.csv
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portvolume.config import load_config
from portvolume.pipeline import SARMonitoringPipeline


def main():
    print("=" * 70)
    print(" SANTOS EXTENDED ACQUISITION (2020-01-01 to 2026-03-25)")
    print("=" * 70)

    # Load config — this will use sites_v2.yaml if present
    config = load_config()

    # Check if santos is in the config
    santos_sites = [s for s in config.sites if s.id == "santos"]
    if not santos_sites:
        print("ERROR: 'santos' site not found in config. Check sites.yaml or sites_v2.yaml")
        return

    santos = santos_sites[0]
    print(f"  Site: {santos.name}")
    print(f"  BBox: {santos.bbox}")
    print(f"  Commodity: {getattr(santos, 'primary_commodity', 'N/A')}")

    pipeline = SARMonitoringPipeline(config)

    # Step 1: Acquisition — search and download
    print("\n[1] SEARCHING STAC CATALOG...")
    try:
        downloaded = pipeline.run_acquisition(
            start_date="2020-01-01",
            end_date="2026-03-25",
            site_ids=["santos"],
        )
        n_new = len(downloaded.get("santos", []))
        print(f"    Downloaded {n_new} new scenes")
    except Exception as e:
        logger.error("Acquisition failed: %s", e)
        print(f"    ERROR: {e}")
        print("    Continuing with existing data...")

    # Step 2: Preprocessing
    print("\n[2] PREPROCESSING...")
    try:
        processed = pipeline.run_preprocessing(site_ids=["santos"])
        n_proc = len(processed.get("santos", []))
        print(f"    Preprocessed {n_proc} scenes")
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        print(f"    ERROR: {e}")
        print("    Continuing with existing data...")

    # Step 3: Detection
    print("\n[3] RUNNING CFAR DETECTION...")
    try:
        counts = pipeline.run_detection(site_ids=["santos"])
        n_det = counts.get("santos", 0)
        print(f"    Detected vessels in {n_det} scenes")
    except Exception as e:
        logger.error("Detection failed: %s", e)
        print(f"    ERROR: {e}")

    # Check results
    summaries_path = config.detections_dir / "santos" / "summaries.csv"
    if summaries_path.exists():
        import pandas as pd
        df = pd.read_csv(summaries_path)
        print(f"\n{'=' * 70}")
        print(f" RESULTS: {len(df)} total scenes with detections")
        print(f" Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f" Vessel count: mean={df['vessel_count'].mean():.1f}, "
              f"std={df['vessel_count'].std():.1f}, "
              f"min={df['vessel_count'].min()}, max={df['vessel_count'].max()}")
        print(f"{'=' * 70}")
    else:
        print("\nNo summaries.csv found — detection may not have completed.")


if __name__ == "__main__":
    main()
