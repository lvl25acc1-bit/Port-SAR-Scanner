"""Orchestrator for the Rotterdam SAR backtest.

Runs all steps in sequence: scene universe → zones → GFW fetch →
AIS matching → covariates → zone detections → analysis → outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from portvolume.config import PROJECT_ROOT

from .scene_universe import build_scene_universe
from .zone_tagging import (
    assign_detections_to_zones,
    build_dissolved_aoi,
    tag_zones,
)
from .gfw_fetch import (
    fetch_all_daily_presence,
    fetch_vessel_identity,
    get_gfw_token,
)
from .ais_matching import (
    build_daily_counts,
    build_monthly_counts,
    filter_commercial_vessels,
    sensitivity_sweep,
)
from .covariates import fetch_tide, fetch_wind
from .analysis import (
    controlled_model,
    load_and_merge,
    monthly_validation,
    save_outputs,
    scene_level_diagnostic,
    select_audit_scenes,
    tide_by_zone,
    weekly_robustness,
)

logger = logging.getLogger(__name__)


def load_backtest_config(config_path: Path | None = None) -> dict:
    """Load backtest config from config/backtest.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "backtest.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw["rotterdam_backtest"]


def run_rotterdam_backtest(
    config_path: Path | None = None,
    force_refetch: bool = False,
) -> dict:
    """Run the full Rotterdam backtest pipeline.

    Returns dict with all analysis results.
    """
    cfg = load_backtest_config(config_path)
    backtest_dir = PROJECT_ROOT / cfg["output_dir"]
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Scene universe
    # ---------------------------------------------------------------
    logger.info("=== Step 1: Scene Universe ===")
    csv_path = PROJECT_ROOT / cfg["csv_path"]
    universe = build_scene_universe(csv_path, cache_dir=backtest_dir)

    # ---------------------------------------------------------------
    # Step 2: Zone tagging + dissolved AOI
    # ---------------------------------------------------------------
    logger.info("=== Step 2: Zone Tagging + AOI ===")
    gpkg_path = PROJECT_ROOT / cfg["water_mask"]
    zones_gdf = tag_zones(gpkg_path)
    zones_gdf.to_parquet(backtest_dir / "zones_tagged.parquet")

    aoi_path = backtest_dir / "rotterdam_aoi_dissolved.geojson"
    aoi_geojson = build_dissolved_aoi(zones_gdf, aoi_path)

    # ---------------------------------------------------------------
    # Step 3: GFW presence fetch
    # ---------------------------------------------------------------
    logger.info("=== Step 3: GFW Presence Fetch ===")
    token = get_gfw_token()
    unique_dates = sorted(universe["date"].dt.strftime("%Y-%m-%d").unique())
    logger.info("Fetching GFW presence for %d unique dates", len(unique_dates))

    gfw_cache = backtest_dir / "gfw_cache"
    presence_df = fetch_all_daily_presence(token, aoi_geojson, unique_dates, gfw_cache)

    # ---------------------------------------------------------------
    # Step 4: GFW vessel identity (SKIPPED in v1)
    # ---------------------------------------------------------------
    # 20K+ unique vessels makes per-vessel identity lookup infeasible
    # at 1 req/s (~6 hours). V1 filters on vessel_type only (from
    # presence data) and uses hours as the reference metric instead
    # of hull area. Identity fetch can be added in v2 with batch API
    # or pre-filtered to commercial types only.
    logger.info("=== Step 4: Vessel Identity (skipped in v1 — type filter only) ===")
    identity_df = pd.DataFrame()

    # ---------------------------------------------------------------
    # Step 5: AIS matching + monthly + sensitivity
    # ---------------------------------------------------------------
    logger.info("=== Step 5: AIS Matching ===")
    # Use min_hours=4 as primary filter when identity is unavailable
    # (vessels present 4+ hours are likely large berthed commercial vessels)
    primary_min_hours = 4.0
    filtered = filter_commercial_vessels(
        presence_df, identity_df, min_area_m2=0, min_hours=primary_min_hours,
    )
    daily_counts = build_daily_counts(universe, filtered, cache_dir=backtest_dir)
    monthly_counts = build_monthly_counts(daily_counts, cache_dir=backtest_dir)

    # Sensitivity sweep across hours thresholds
    sweep = sensitivity_sweep(
        presence_df, identity_df, cfg["ais_size_thresholds"], universe,
        hours_thresholds=[0, 2, 4, 6, 8, 12],
    )
    sweep.to_parquet(backtest_dir / "sensitivity_sweep.parquet", index=False)

    # ---------------------------------------------------------------
    # Step 6: Covariates (wind + tide)
    # ---------------------------------------------------------------
    logger.info("=== Step 6: Covariates ===")
    midpoints = universe["scene_midpoint_utc"].tolist()

    wind_df = fetch_wind(
        midpoints, lat=cfg["wind_lat"], lon=cfg["wind_lon"], cache_dir=backtest_dir
    )
    tide_df = fetch_tide(midpoints, cache_dir=backtest_dir)

    # ---------------------------------------------------------------
    # Step 7: Zone detection assignment
    # ---------------------------------------------------------------
    logger.info("=== Step 7: Zone Detections ===")
    det_dir = PROJECT_ROOT / cfg["detection_dir"]
    scene_ids = universe["scene_id"].tolist()
    zone_dets = assign_detections_to_zones(det_dir, zones_gdf, scene_ids, backtest_dir)

    # ---------------------------------------------------------------
    # Step 8: Analysis
    # ---------------------------------------------------------------
    logger.info("=== Step 8: Analysis ===")
    merged = load_and_merge(backtest_dir)

    scene_diag = scene_level_diagnostic(merged)
    monthly_val = monthly_validation(merged, monthly_counts)
    controlled = controlled_model(merged)
    tide_zone = tide_by_zone(zone_dets, merged)
    weekly = weekly_robustness(merged, controlled)
    audit = select_audit_scenes(merged, n=cfg.get("audit_n_scenes", 20))

    # ---------------------------------------------------------------
    # Step 9: Save outputs
    # ---------------------------------------------------------------
    logger.info("=== Step 9: Save Outputs ===")
    save_outputs(
        backtest_dir, scene_diag, monthly_val, controlled,
        tide_zone, weekly, sweep, audit,
    )

    logger.info("=== Rotterdam Backtest Complete ===")

    return {
        "scene_diagnostic": scene_diag,
        "monthly_validation": monthly_val,
        "controlled_model": controlled,
        "tide_by_zone": tide_zone,
        "n_scenes": len(universe),
        "n_audit": len(audit),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    for name in ["urllib3", "requests"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    results = run_rotterdam_backtest()

    import json
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2, default=str))
