"""Build the canonical 349-row scene universe from rotterdam_vessel_counts_raw.csv.

Parses scene_id into satellite, timestamps, pass_family, orbit_direction,
and applies QC flags.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_scene_id(scene_id: str) -> dict:
    """Parse a Sentinel-1 RTC scene_id into structured metadata.

    S1A format (9 fields): S1A_IW_GRDH_1SDV_{start}_{end}_{orbit}_{hex}_rtc
    S1C format (8 fields): S1C_IW_GRDH_1SDV_{start}_{end}_{orbit}_rtc
    """
    parts = scene_id.split("_")

    satellite = parts[0]  # S1A or S1C

    start_str = parts[4]  # 20240101T055039
    end_str = parts[5]    # 20240101T055104

    start_utc = datetime.strptime(start_str, "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )
    end_utc = datetime.strptime(end_str, "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )

    duration_s = (end_utc - start_utc).total_seconds()
    midpoint_utc = start_utc + (end_utc - start_utc) / 2

    # Orbit number: field 6 for both formats
    abs_orbit = int(parts[6])

    return {
        "satellite": satellite,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "scene_midpoint_utc": midpoint_utc,
        "scene_duration_s": duration_s,
        "abs_orbit": abs_orbit,
    }


def derive_pass_family(midpoint_utc: datetime) -> str:
    """Morning if midpoint hour < 12 UTC, else evening."""
    return "morning" if midpoint_utc.hour < 12 else "evening"


def derive_orbit_direction(abs_orbit: int) -> str:
    """Sentinel-1 orbit direction from absolute orbit number.

    Odd orbit = ascending, even = descending. Same rule for S1A and S1C.
    """
    return "ascending" if abs_orbit % 2 == 1 else "descending"


def validate_orbit_direction(orbit_direction: str, pass_family: str) -> bool:
    """Cross-check: at Rotterdam latitude (~52°N), ascending ≈ morning,
    descending ≈ evening. Returns True if consistent."""
    expected = "morning" if orbit_direction == "ascending" else "evening"
    return pass_family == expected


def build_scene_universe(
    csv_path: Path,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the canonical scene universe table from the raw CSV.

    Returns DataFrame with all parsed metadata, QC flags, and derived fields.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d scenes from %s", len(df), csv_path.name)

    # Parse each scene_id
    parsed = df["scene_id"].apply(parse_scene_id).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Derive pass_family and orbit_direction
    df["pass_family"] = df["scene_midpoint_utc"].apply(derive_pass_family)
    df["orbit_direction"] = df["abs_orbit"].apply(derive_orbit_direction)

    # Footprint ID (v1 simplification)
    df["footprint_id"] = df["satellite"] + "_" + df["orbit_direction"]

    # Validate orbit direction vs time-of-day
    n_inconsistent = 0
    for _, row in df.iterrows():
        if not validate_orbit_direction(row["orbit_direction"], row["pass_family"]):
            n_inconsistent += 1
    if n_inconsistent > 0:
        logger.warning(
            "%d/%d scenes have orbit_direction inconsistent with pass_family",
            n_inconsistent, len(df),
        )

    # QC flags
    df["raw_partner_missing"] = df["date"] == "2024-05-12"

    # Clutter QC: flag scenes with anomalous mean_cluster_size
    # (not vessel_count — a high count can be a genuinely busy day)
    mcs = df["mean_cluster_size"]
    mcs_mean = mcs.mean()
    mcs_std = mcs.std()
    df["qc_clutter_flag"] = (mcs < mcs_mean - 3 * mcs_std) | (
        mcs > mcs_mean + 3 * mcs_std
    )

    n_flagged = df["qc_clutter_flag"].sum()
    if n_flagged > 0:
        logger.info("QC flagged %d scenes for anomalous mean_cluster_size", n_flagged)

    # Ensure date column is proper datetime
    df["date"] = pd.to_datetime(df["date"])

    logger.info(
        "Scene universe: %d scenes, %d S1A / %d S1C, %d morning / %d evening",
        len(df),
        (df["satellite"] == "S1A").sum(),
        (df["satellite"] == "S1C").sum(),
        (df["pass_family"] == "morning").sum(),
        (df["pass_family"] == "evening").sum(),
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / "scene_universe.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved scene universe to %s", out_path)

    return df
