"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from portvolume.detection.wind_correction import WaveAgeParams
from portvolume.sites import Site

# Project root: two levels up from this file (src/portvolume/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PostDetectionParams:
    """Post-CFAR filters applied after detection, before clustering."""
    min_db: float = -12.0
    max_aspect_ratio: float = 6.0
    min_db_contrast: float = 2.0  # peak_db - mean_db (dB difference)


@dataclass(frozen=True)
class CFARParams:
    guard_cells: int
    background_cells: int
    pfa: float
    min_target_pixels: int
    polarization: str
    post_detection: PostDetectionParams = PostDetectionParams()
    site_overrides: Optional[dict] = None  # site_id -> partial override dict

    def for_site(self, site_id: str) -> "CFARParams":
        """Return a CFARParams with per-site overrides applied."""
        if not self.site_overrides or site_id not in self.site_overrides:
            return self
        # Work on a copy to avoid mutating the stored config
        overrides = dict(self.site_overrides[site_id])
        pd_dict = overrides.pop("post_detection", None)

        post = self.post_detection
        if pd_dict:
            post = PostDetectionParams(
                min_db=pd_dict.get("min_db", post.min_db),
                max_aspect_ratio=pd_dict.get("max_aspect_ratio", post.max_aspect_ratio),
                min_db_contrast=pd_dict.get("min_db_contrast", post.min_db_contrast),
            )

        return CFARParams(
            guard_cells=overrides.get("guard_cells", self.guard_cells),
            background_cells=overrides.get("background_cells", self.background_cells),
            pfa=overrides.get("pfa", self.pfa),
            min_target_pixels=overrides.get("min_target_pixels", self.min_target_pixels),
            polarization=overrides.get("polarization", self.polarization),
            post_detection=post,
            site_overrides=None,
        )


@dataclass(frozen=True)
class AirportParams:
    polarization: str
    threshold_db: float
    min_cluster_pixels: int


@dataclass(frozen=True)
class IndexParams:
    aggregation: str
    day_of_week: str
    normalization: str
    rolling_window_weeks: int


@dataclass(frozen=True)
class AnalysisParams:
    max_lag_weeks: int
    significance_level: float
    min_observations: int


@dataclass(frozen=True)
class FREDSeries:
    id: str
    name: str
    frequency: str


@dataclass(frozen=True)
class YFinanceTicker:
    ticker: str
    name: str


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire pipeline."""

    sites: list[Site]
    cfar: CFARParams
    airport: AirportParams
    index_params: IndexParams
    analysis: AnalysisParams

    stac_catalog_url: str
    collection: str
    asset_keys: list[str]
    start_date: str
    end_date: Optional[str]

    data_dir: Path
    processed_dir: Path
    detections_dir: Path
    indices_dir: Path
    economic_dir: Path

    fred_api_key: str
    fred_series: list[FREDSeries]
    yfinance_tickers: list[YFinanceTicker]
    zones_dir: Path | None = None
    wave_age: WaveAgeParams = None

    def __post_init__(self):
        if self.wave_age is None:
            object.__setattr__(self, "wave_age", WaveAgeParams())

    @property
    def ports(self) -> list[Site]:
        return [s for s in self.sites if s.site_type == "port"]

    @property
    def airports(self) -> list[Site]:
        return [s for s in self.sites if s.site_type == "airport"]

    def get_site(self, site_id: str) -> Site:
        for s in self.sites:
            if s.id == site_id:
                return s
        raise KeyError(f"No site with id={site_id!r}")


def _parse_sites(data: dict) -> list[Site]:
    """Parse site YAML structures into Site objects.

    Supports both the original ``sites.yaml`` format (top-level ``ports`` /
    ``airports`` keys) and the v2 flat ``sites`` list format.
    """
    sites: list[Site] = []

    # --- v2 flat format (sites_v2.yaml) ---
    for entry in data.get("sites", []):
        anch = entry.get("anchorage_bbox")
        sites.append(
            Site(
                id=entry["id"],
                name=entry["name"],
                lat=entry["lat"],
                lon=entry["lon"],
                bbox=tuple(entry["bbox"]),
                site_type=entry.get("site_type", "port"),
                region=entry.get("region", ""),
                annual_teu_millions=entry.get("annual_teu_millions", 0.0),
                role=entry.get("role", ""),
                primary_commodity=entry.get("primary_commodity", ""),
                commodity_group=entry.get("commodity_group", ""),
                throughput_variance_pct=entry.get("throughput_variance_pct", 0.0),
                anchorage_bbox=tuple(anch) if anch else None,
                economic_indicators=tuple(entry.get("economic_indicators", ())),
            )
        )

    # --- v1 format (sites.yaml) ---
    for port in data.get("ports", []):
        sites.append(
            Site(
                id=port["id"],
                name=port["name"],
                lat=port["lat"],
                lon=port["lon"],
                bbox=tuple(port["bbox"]),
                site_type="port",
                region=port.get("region", ""),
                annual_teu_millions=port.get("annual_teu_millions", 0.0),
                primary_commodity=port.get("primary_commodity", ""),
                commodity_group=port.get("commodity_group", ""),
                throughput_variance_pct=port.get("throughput_variance_pct", 0.0),
                anchorage_bbox=tuple(port["anchorage_bbox"]) if port.get("anchorage_bbox") else None,
                economic_indicators=tuple(port.get("economic_indicators", ())),
            )
        )

    for airport in data.get("airports", []):
        sites.append(
            Site(
                id=airport["id"],
                name=airport["name"],
                lat=airport["lat"],
                lon=airport["lon"],
                bbox=tuple(airport["bbox"]),
                site_type="airport",
                role=airport.get("role", ""),
            )
        )

    return sites


def _parse_economic(data: dict) -> tuple[list[FREDSeries], list[YFinanceTicker]]:
    """Parse the economic_series.yaml structure."""
    fred = [
        FREDSeries(id=s["id"], name=s["name"], frequency=s["frequency"])
        for s in data.get("fred_series", [])
    ]
    yf = [
        YFinanceTicker(ticker=t["ticker"], name=t["name"])
        for t in data.get("yfinance_tickers", [])
    ]
    return fred, yf


def load_config(
    config_dir: str | Path | None = None,
) -> PipelineConfig:
    """Load and validate all configuration from YAML files.

    Parameters
    ----------
    config_dir : path to config/ directory. Defaults to PROJECT_ROOT/config.
    """
    if config_dir is None:
        config_dir = PROJECT_ROOT / "config"
    config_dir = Path(config_dir)

    # Prefer sites_v2.yaml if it exists; fall back to sites.yaml
    sites_v2_path = config_dir / "sites_v2.yaml"
    sites_v1_path = config_dir / "sites.yaml"
    sites_path = sites_v2_path if sites_v2_path.exists() else sites_v1_path
    with open(sites_path) as f:
        sites_data = yaml.safe_load(f)
    with open(config_dir / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "economic_series.yaml") as f:
        econ_data = yaml.safe_load(f)

    sites = _parse_sites(sites_data)
    fred_series, yf_tickers = _parse_economic(econ_data)

    stac = settings["stac"]
    acq = settings["acquisition"]
    cfar_raw = settings["cfar"]
    airport_raw = settings["airport"]
    idx = settings["index"]
    analysis_raw = settings["analysis"]
    paths = settings["paths"]

    # Resolve paths relative to project root
    def _resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    fred_key_value = settings.get("fred", {}).get("api_key_env", "FRED_API_KEY")
    # If the value looks like an env var name (ALL_CAPS), resolve it;
    # otherwise treat it as the literal API key.
    if fred_key_value.isupper() and fred_key_value.replace("_", "").isalpha():
        fred_api_key = os.environ.get(fred_key_value, "")
    else:
        fred_api_key = fred_key_value

    # Parse optional zones_dir
    zones_dir_raw = paths.get("zones_dir")
    if zones_dir_raw:
        zones_dir_path = Path(zones_dir_raw)
        if not zones_dir_path.is_absolute():
            zones_dir_path = PROJECT_ROOT / zones_dir_path
        # Don't mkdir — zones_dir may not exist yet and that's fine
    else:
        zones_dir_path = None

    # Parse optional wave_age section
    wave_age_raw = settings.get("wave_age")
    if wave_age_raw:
        wave_age = WaveAgeParams(
            young_sea_threshold=wave_age_raw.get("young_sea_threshold", 1.2),
            swell_period_threshold=wave_age_raw.get("swell_period_threshold", 10.0),
            correction_factors=wave_age_raw.get(
                "correction_factors",
                {"young_sea": 1.4, "old_sea": 1.0, "swell": 0.85},
            ),
            wind_speed_scale=wave_age_raw.get("wind_speed_scale", 0.03),
        )
    else:
        wave_age = WaveAgeParams()

    return PipelineConfig(
        sites=sites,
        cfar=CFARParams(
            guard_cells=cfar_raw["guard_cells"],
            background_cells=cfar_raw["background_cells"],
            pfa=cfar_raw["pfa"],
            min_target_pixels=cfar_raw["min_target_pixels"],
            polarization=cfar_raw["polarization"],
            post_detection=PostDetectionParams(
                **cfar_raw.get("post_detection", {}),
            ) if cfar_raw.get("post_detection") else PostDetectionParams(),
            site_overrides=cfar_raw.get("site_overrides"),
        ),
        airport=AirportParams(
            polarization=airport_raw["polarization"],
            threshold_db=airport_raw["threshold_db"],
            min_cluster_pixels=airport_raw["min_cluster_pixels"],
        ),
        index_params=IndexParams(
            aggregation=idx["aggregation"],
            day_of_week=idx["day_of_week"],
            normalization=idx["normalization"],
            rolling_window_weeks=idx["rolling_window_weeks"],
        ),
        analysis=AnalysisParams(
            max_lag_weeks=analysis_raw["max_lag_weeks"],
            significance_level=analysis_raw["significance_level"],
            min_observations=analysis_raw["min_observations"],
        ),
        stac_catalog_url=stac["catalog_url"],
        collection=stac["collection"],
        asset_keys=acq["asset_keys"],
        start_date=acq["start_date"],
        end_date=acq.get("end_date"),
        data_dir=_resolve(acq["data_dir"]),
        processed_dir=_resolve(acq["processed_dir"]),
        detections_dir=_resolve(paths["detections_dir"]),
        indices_dir=_resolve(paths["indices_dir"]),
        economic_dir=_resolve(paths["economic_dir"]),
        zones_dir=zones_dir_path,
        fred_api_key=fred_api_key,
        fred_series=fred_series,
        yfinance_tickers=yf_tickers,
        wave_age=wave_age,
    )
