"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from portvolume.config import PipelineConfig, load_config
from portvolume.acquisition.search import (
    create_stac_client,
    filter_existing,
    get_downloaded_scene_ids,
    search_all_sites,
)
from portvolume.acquisition.download import download_site_scenes
from portvolume.acquisition.preprocess import preprocess_scene
from portvolume.detection.cfar import detect_vessels
from portvolume.detection.adaptive_cfar import detect_vessels_adaptive
from portvolume.detection.wind_correction import (
    classify_wave_age,
    compute_cfar_correction_factor,
    compute_wind_adjusted_count,
)
from portvolume.backtest.covariates import fetch_wind
from portvolume.detection.airport_activity import estimate_airport_activity
from portvolume.detection.postprocess import (
    cluster_detections,
    save_detections,
    scene_detection_summary,
)
from portvolume.detection.water_mask import build_water_raster, WaterMaskError
from portvolume.index.aggregate import build_all_indices
from portvolume.index.normalize import build_composite_index, build_regional_indices
from portvolume.economic.fred import fetch_all_fred, get_fred_client
from portvolume.economic.yfinance_data import fetch_all_yfinance
from portvolume.economic.merge import align_sar_and_economic
from portvolume.analysis.correlation import compute_correlations, lead_lag_crosscorrelation
from portvolume.analysis.granger import run_granger_test
from portvolume.analysis.regression import out_of_sample_test

logger = logging.getLogger(__name__)


class SARMonitoringPipeline:
    """End-to-end orchestrator for the SAR monitoring pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stac_client = None

    @property
    def stac_client(self):
        if self._stac_client is None:
            self._stac_client = create_stac_client(self.config.stac_catalog_url)
        return self._stac_client

    def run_acquisition(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        site_ids: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        """Step 1: Search and download new Sentinel-1 scenes for all sites."""
        sites = self.config.sites
        if site_ids:
            sites = [s for s in sites if s.id in site_ids]

        start = start_date or self.config.start_date
        end = end_date or self.config.end_date

        logger.info("Searching for scenes: %s to %s for %d sites", start, end, len(sites))
        all_items = search_all_sites(
            self.stac_client, self.config.collection, sites, start, end
        )

        downloaded: dict[str, list[Path]] = {}
        for site in sites:
            items = all_items.get(site.id, [])
            existing = get_downloaded_scene_ids(self.config.data_dir, site.id)
            new_items = filter_existing(items, existing)

            if not new_items:
                logger.info("No new scenes for %s", site.id)
                downloaded[site.id] = []
                continue

            logger.info("Downloading %d new scenes for %s", len(new_items), site.id)
            paths = download_site_scenes(
                site_id=site.id,
                items=new_items,
                asset_keys=self.config.asset_keys,
                bbox=site.bbox,
                output_dir=self.config.data_dir,
            )
            downloaded[site.id] = paths

        return downloaded

    def run_preprocessing(
        self,
        site_ids: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        """Step 2: Preprocess all raw scenes not yet processed."""
        sites = self.config.sites
        if site_ids:
            sites = [s for s in sites if s.id in site_ids]

        processed: dict[str, list[Path]] = {}
        for site in sites:
            raw_dir = self.config.data_dir / site.id
            if not raw_dir.exists():
                continue

            site_processed = []
            for raw_file in sorted(raw_dir.glob("*.tif")):
                out = preprocess_scene(raw_file, site, self.config.processed_dir)
                site_processed.append(out)

            processed[site.id] = site_processed
            if site_processed:
                logger.info("Preprocessed %d files for %s", len(site_processed), site.id)

        return processed

    def run_detection(
        self,
        site_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Step 3: Run detection on all preprocessed scenes."""
        sites = self.config.sites
        if site_ids:
            sites = [s for s in sites if s.id in site_ids]

        detection_counts: dict[str, int] = {}
        water_cache = self.config.data_dir / "water_masks"

        for site in sites:
            proc_dir = self.config.processed_dir / site.id
            if not proc_dir.exists():
                continue

            # Resolve polarization per site type BEFORE globbing files
            if site.site_type == "port":
                site_cfar = self.config.cfar.for_site(site.id)
                pol = site_cfar.polarization
            else:
                site_cfar = None
                pol = self.config.airport.polarization

            summaries = []
            det_dir = self.config.detections_dir / site.id
            det_dir.mkdir(parents=True, exist_ok=True)

            proc_files = sorted(proc_dir.glob(f"*_{pol}.tif"))

            for proc_file in proc_files:
                scene_id = proc_file.stem.rsplit("_", 1)[0]
                timestamp = _extract_timestamp(scene_id)

                if site.site_type == "port":
                    # Build water mask for this scene's grid
                    try:
                        water_mask = build_water_raster(
                            proc_file, site, buffer_m=30, cache_dir=water_cache
                        )
                    except WaterMaskError as exc:
                        logger.error(
                            "Skipping site %s: water mask failed: %s", site.id, exc
                        )
                        break  # skip all scenes for this site

                    # Fetch wind data for adaptive CFAR
                    wind_speed = None
                    wave_age_class = None
                    try:
                        from datetime import datetime as _dt
                        ts_dt = _dt.fromisoformat(timestamp)
                        wind_df = fetch_wind([ts_dt], lat=site.lat, lon=site.lon)
                        if not wind_df.empty and not pd.isna(wind_df.iloc[0]["wind_speed_mps"]):
                            wind_speed = float(wind_df.iloc[0]["wind_speed_mps"])
                            wave_age_class = classify_wave_age(wind_speed)
                    except Exception as exc:
                        logger.debug("Wind fetch failed for %s: %s", scene_id, exc)

                    mask, db_image, transform, crs = detect_vessels_adaptive(
                        proc_file, site_cfar, site,
                        water_mask=water_mask,
                        wind_speed_mps=wind_speed,
                        wave_age_class=wave_age_class,
                        wave_age_params=self.config.wave_age,
                    )
                    gdf = cluster_detections(
                        mask, site_cfar.min_target_pixels, transform, crs=crs,
                        db_image=db_image, post_params=site_cfar.post_detection,
                    )
                    save_detections(gdf, det_dir / scene_id)
                    summary = scene_detection_summary(gdf, site.id, scene_id, timestamp)

                    # Add wind-adjusted count if wind data available
                    if wind_speed is not None:
                        raw_count = summary.get("vessel_count", 0)
                        summary["wind_adjusted_count"] = compute_wind_adjusted_count(
                            raw_count=raw_count,
                            wind_speed_mps=wind_speed,
                            pass_family=_infer_pass_family(timestamp),
                            satellite=_infer_satellite(scene_id),
                        )
                else:
                    obs = estimate_airport_activity(
                        proc_file, site, scene_id, timestamp, self.config.airport
                    )
                    summary = {
                        "site_id": obs.site_id,
                        "scene_id": obs.scene_id,
                        "timestamp": obs.timestamp,
                        "estimated_aircraft_count": obs.estimated_aircraft_count,
                        "bright_pixel_fraction": obs.bright_pixel_fraction,
                        "mean_backscatter_db": obs.mean_backscatter_db,
                    }

                summaries.append(summary)

            if summaries:
                summary_df = pd.DataFrame(summaries)
                summary_df.to_csv(det_dir / "summaries.csv", index=False)
                detection_counts[site.id] = len(summaries)
                logger.info("Detected in %d scenes for %s", len(summaries), site.id)

        return detection_counts

    def run_index_construction(self) -> dict[str, pd.DataFrame]:
        """Step 4: Build weekly indices from all detections."""
        indices = build_all_indices(
            self.config.sites,
            self.config.detections_dir,
            self.config.index_params.day_of_week,
        )

        # Save individual indices
        for site_id, df in indices.items():
            out_path = self.config.indices_dir / f"{site_id}_weekly.parquet"
            df.to_parquet(out_path)

        # Build composite
        composite = build_composite_index(
            indices,
            self.config.sites,
            self.config.index_params.normalization,
            self.config.index_params.rolling_window_weeks,
        )
        if not composite.empty:
            composite.to_parquet(self.config.indices_dir / "composite.parquet")

        # Build regional
        regional = build_regional_indices(
            indices,
            self.config.sites,
            self.config.index_params.normalization,
            self.config.index_params.rolling_window_weeks,
        )
        for region, df in regional.items():
            df.to_parquet(self.config.indices_dir / f"regional_{region}.parquet")

        return indices

    def run_economic_update(self) -> pd.DataFrame:
        """Step 5: Fetch latest economic data."""
        all_economic = pd.DataFrame()

        # FRED data
        if self.config.fred_api_key:
            try:
                client = get_fred_client(self.config.fred_api_key)
                fred_df = fetch_all_fred(
                    client,
                    self.config.fred_series,
                    self.config.start_date,
                    cache_dir=self.config.economic_dir,
                )
                all_economic = fred_df
            except Exception:
                logger.exception("Failed to fetch FRED data")

        # yfinance data
        try:
            yf_df = fetch_all_yfinance(
                self.config.yfinance_tickers,
                self.config.start_date,
                cache_dir=self.config.economic_dir,
            )
            if not yf_df.empty:
                if all_economic.empty:
                    all_economic = yf_df
                else:
                    all_economic = all_economic.join(yf_df, how="outer")
        except Exception:
            logger.exception("Failed to fetch yfinance data")

        return all_economic

    def run_analysis(self, economic_df: pd.DataFrame | None = None) -> dict:
        """Step 6: Run correlation analysis."""
        composite_path = self.config.indices_dir / "composite.parquet"
        if not composite_path.exists():
            logger.warning("No composite index found. Skipping analysis.")
            return {}

        composite = pd.read_parquet(composite_path)

        if economic_df is None:
            economic_df = self.run_economic_update()

        if economic_df.empty:
            logger.warning("No economic data. Skipping analysis.")
            return {}

        # Merge
        merged = align_sar_and_economic(composite, economic_df)

        econ_cols = [c for c in merged.columns if c not in ("week", "composite_index", "n_sites")]

        # Correlations
        corr_df = compute_correlations(merged, "composite_index", econ_cols)

        # Lead-lag for top correlated series
        lead_lag_results = {}
        if not corr_df.empty:
            top_series = corr_df.nlargest(5, "pearson_r")["econ_series"].tolist()
            sar_s = merged.set_index("week")["composite_index"]
            for col in top_series:
                if col in merged.columns:
                    econ_s = merged.set_index("week")[col]
                    ll = lead_lag_crosscorrelation(
                        sar_s, econ_s, self.config.analysis.max_lag_weeks
                    )
                    lead_lag_results[col] = ll

        # Granger causality
        granger_results = []
        sar_s = merged.set_index("week")["composite_index"]
        for col in econ_cols:
            if col in merged.columns:
                econ_s = merged.set_index("week")[col]
                result = run_granger_test(
                    sar_s, econ_s, max_lag=self.config.analysis.max_lag_weeks
                )
                granger_results.append(
                    {
                        "economic_series": col,
                        "sar_causes_econ": result.sar_causes_econ,
                        "econ_causes_sar": result.econ_causes_sar,
                        "direction": result.best_direction,
                        "optimal_lag": result.optimal_lag,
                        "p_value": result.p_value,
                    }
                )

        # Out-of-sample test
        oos = {}
        if len(econ_cols) > 0:
            oos = out_of_sample_test(
                merged, econ_cols[0], ["composite_index"], n_splits=5
            )

        return {
            "correlations": corr_df,
            "lead_lag": lead_lag_results,
            "granger": granger_results,
            "out_of_sample": oos,
            "merged_data": merged,
        }

    def run_full_pipeline(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        site_ids: list[str] | None = None,
    ) -> dict:
        """Run all steps in sequence."""
        logger.info("=== Starting full pipeline run ===")

        logger.info("--- Step 1: Acquisition ---")
        self.run_acquisition(start_date=start_date, end_date=end_date, site_ids=site_ids)

        logger.info("--- Step 2: Preprocessing ---")
        self.run_preprocessing(site_ids=site_ids)

        logger.info("--- Step 3: Detection ---")
        self.run_detection(site_ids=site_ids)

        logger.info("--- Step 4: Index Construction ---")
        self.run_index_construction()

        logger.info("--- Step 5: Economic Data ---")
        economic = self.run_economic_update()

        logger.info("--- Step 6: Analysis ---")
        results = self.run_analysis(economic)

        logger.info("=== Pipeline run complete ===")
        return results


def _extract_timestamp(scene_id: str) -> str:
    """Extract a timestamp string from a Sentinel-1 scene ID.

    Sentinel-1 GRD scene IDs follow the pattern:
    S1A_IW_GRDH_1SDV_20230115T...
    We extract the date portion.
    """
    parts = scene_id.split("_")
    for part in parts:
        if len(part) >= 8 and part[:8].isdigit():
            try:
                dt = datetime.strptime(part[:8], "%Y%m%d")
                return dt.isoformat()
            except ValueError:
                pass
    # Fallback: return the scene_id itself
    return scene_id


def _infer_pass_family(timestamp: str) -> str:
    """Infer morning/evening pass from timestamp hour (UTC).

    Sentinel-1 ascending passes over Europe are typically evening (17-18 UTC),
    descending passes are morning (05-06 UTC).
    """
    try:
        from datetime import datetime as _dt
        dt = _dt.fromisoformat(timestamp)
        return "evening" if dt.hour >= 12 else "morning"
    except Exception:
        return "morning"


def _infer_satellite(scene_id: str) -> str:
    """Extract satellite ID (S1A, S1B, S1C) from a Sentinel-1 scene ID."""
    upper = scene_id.upper()
    if upper.startswith("S1C"):
        return "S1C"
    elif upper.startswith("S1B"):
        return "S1B"
    return "S1A"
