"""Fetch AIS vessel presence and identity from the Global Fishing Watch API.

Follows the fetch+cache pattern from economic/fred.py. Per-date cache files
allow incremental retry on failure. Token is resolved from environment only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GFW_BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"
PRESENCE_DATASET = "public-global-presence:latest"
IDENTITY_DATASET = "public-global-vessel-identity:latest"
RATE_LIMIT_SECONDS = 1.1  # slightly over 1s to stay under limit


def get_gfw_token() -> str:
    """Resolve GFW API token from GFW_API_TOKEN environment variable only."""
    token = os.environ.get("GFW_API_TOKEN", "")
    if not token:
        raise ValueError(
            "GFW_API_TOKEN environment variable not set. "
            "Get a token at https://globalfishingwatch.org/our-apis/"
        )
    return token


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def fetch_daily_presence(
    token: str,
    aoi_geojson: dict,
    date_str: str,
) -> pd.DataFrame:
    """Fetch vessel presence for a single day within the AOI polygon.

    Returns DataFrame with one row per vessel-day.
    """
    # date_str format: "2024-01-09"
    next_date = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    next_str = next_date.strftime("%Y-%m-%d")

    url = (
        f"{GFW_BASE_URL}/4wings/report"
        f"?datasets[0]={PRESENCE_DATASET}"
        f"&date-range={date_str},{next_str}"
        f"&spatial-resolution=HIGH"
        f"&temporal-resolution=DAILY"
        f"&group-by=VESSEL_ID"
        f"&format=JSON"
    )

    body = {"geojson": aoi_geojson}
    resp = requests.post(url, headers=_headers(token), json=body, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    entries = data.get("entries", [])

    # Flatten: entries is a list of dicts, each with dataset key → list of vessels
    vessels = []
    for entry in entries:
        if entry is None:
            continue
        for dataset_key, vessel_list in entry.items():
            if isinstance(vessel_list, list):
                vessels.extend(vessel_list)

    if not vessels:
        return pd.DataFrame()

    df = pd.DataFrame(vessels)
    df["query_date"] = date_str
    return df


def fetch_all_daily_presence(
    token: str,
    aoi_geojson: dict,
    dates: list[str],
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch presence for all dates, caching per-date as parquet.

    Skips dates already cached. Returns the combined DataFrame.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for i, date_str in enumerate(dates):
        cache_path = cache_dir / f"presence_{date_str}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            all_frames.append(df)
            continue

        try:
            df = fetch_daily_presence(token, aoi_geojson, date_str)
            if len(df) > 0:
                df.to_parquet(cache_path, index=False)
            else:
                # Save empty marker
                pd.DataFrame({"query_date": [date_str]}).to_parquet(
                    cache_path, index=False
                )
                df = pd.DataFrame()
            all_frames.append(df)

            if (i + 1) % 50 == 0:
                logger.info("GFW presence: fetched %d/%d dates", i + 1, len(dates))

        except Exception:
            logger.exception("GFW presence fetch failed for %s", date_str)

        time.sleep(RATE_LIMIT_SECONDS)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Save combined
    out_path = cache_dir.parent / "gfw_daily_presence.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(
        "GFW daily presence: %d vessel-day records across %d dates",
        len(combined), combined["query_date"].nunique() if "query_date" in combined.columns else 0,
    )
    return combined


def fetch_vessel_identity(
    token: str,
    vessel_ids: list[str],
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch vessel identity (length, beam, type) for a list of vessel_ids.

    Uses the vessels search endpoint. Caches incrementally.
    """
    cache_path = cache_dir / "gfw_vessel_identity.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        known_ids = set(cached["vesselId"].unique()) if "vesselId" in cached.columns else set()
        missing = [vid for vid in vessel_ids if vid not in known_ids]
        if not missing:
            logger.info("Vessel identity: all %d vessels cached", len(vessel_ids))
            return cached
    else:
        cached = pd.DataFrame()
        missing = vessel_ids

    logger.info("Fetching identity for %d vessels (%d cached)", len(missing), len(vessel_ids) - len(missing))

    new_records = []
    for i, vid in enumerate(missing):
        try:
            url = (
                f"{GFW_BASE_URL}/vessels/{vid}"
                f"?datasets[0]={IDENTITY_DATASET}"
            )
            resp = requests.get(url, headers=_headers(token), timeout=30)

            if resp.status_code == 200:
                vdata = resp.json()
                # Extract the most recent identity entry
                registries = vdata.get("registryInfo", [])
                if registries:
                    reg = registries[-1]  # most recent
                    new_records.append({
                        "vesselId": vid,
                        "mmsi": vdata.get("ssvid", ""),
                        "shipName": reg.get("shipname", ""),
                        "flag": reg.get("flag", ""),
                        "vesselType": vdata.get("vesselType", ""),
                        "length_m": reg.get("lengthOverallMeters"),
                        "beam_m": reg.get("beamMeters"),
                        "gross_tonnage": reg.get("grossTonnage"),
                    })
                else:
                    new_records.append({
                        "vesselId": vid,
                        "vesselType": vdata.get("vesselType", ""),
                    })
            elif resp.status_code == 404:
                new_records.append({"vesselId": vid})
            else:
                logger.warning("GFW vessel %s: HTTP %d", vid, resp.status_code)

        except Exception:
            logger.exception("GFW vessel identity failed for %s", vid)

        if (i + 1) % 100 == 0:
            logger.info("Vessel identity: fetched %d/%d", i + 1, len(missing))

        time.sleep(RATE_LIMIT_SECONDS)

    if new_records:
        new_df = pd.DataFrame(new_records)
        combined = pd.concat([cached, new_df], ignore_index=True)
    else:
        combined = cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    logger.info("Vessel identity: %d total vessels cached", len(combined))
    return combined
