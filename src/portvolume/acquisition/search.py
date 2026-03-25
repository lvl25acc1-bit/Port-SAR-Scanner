"""STAC catalog search for Sentinel-1 GRD scenes."""

from __future__ import annotations

import logging
from datetime import datetime

import planetary_computer
from pystac_client import Client

from portvolume.sites import Site

logger = logging.getLogger(__name__)


def create_stac_client(catalog_url: str) -> Client:
    """Open a STAC client to the Planetary Computer with token signing."""
    return Client.open(
        catalog_url,
        modifier=planetary_computer.sign_inplace,
    )


def search_scenes(
    client: Client,
    collection: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str | None = None,
) -> list[dict]:
    """Search for Sentinel-1 GRD scenes over a bounding box and date range.

    Returns a list of dicts with keys: id, datetime, geometry, assets, properties.
    """
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    date_range = f"{start_date}/{end_date}"

    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=date_range,
    )

    items = []
    for item in search.items():
        items.append(
            {
                "id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "geometry": item.geometry,
                "bbox": item.bbox,
                "assets": {
                    key: {"href": asset.href, "type": asset.media_type}
                    for key, asset in item.assets.items()
                },
                "properties": dict(item.properties),
            }
        )

    logger.info(
        "Found %d scenes for bbox=%s, dates=%s", len(items), bbox, date_range
    )
    return items


def search_all_sites(
    client: Client,
    collection: str,
    sites: list[Site],
    start_date: str,
    end_date: str | None = None,
) -> dict[str, list[dict]]:
    """Search scenes for all configured sites.

    Returns {site_id: [stac_items...]}.
    """
    results: dict[str, list[dict]] = {}

    for site in sites:
        logger.info("Searching scenes for %s (%s)...", site.id, site.name)
        items = search_scenes(
            client=client,
            collection=collection,
            bbox=site.bbox,
            start_date=start_date,
            end_date=end_date,
        )
        results[site.id] = items
        logger.info("  %s: %d scenes found", site.id, len(items))

    return results


def filter_existing(
    items: list[dict],
    existing_ids: set[str],
) -> list[dict]:
    """Remove items that have already been downloaded."""
    return [item for item in items if item["id"] not in existing_ids]


def get_downloaded_scene_ids(data_dir, site_id: str) -> set[str]:
    """Scan local data directory for already-downloaded scene IDs."""
    from pathlib import Path

    site_dir = Path(data_dir) / site_id
    if not site_dir.exists():
        return set()

    # Each scene is stored as {scene_id}_vv.tif (or vh)
    ids = set()
    for f in site_dir.glob("*.tif"):
        # Strip the _vv or _vh suffix
        scene_id = f.stem.rsplit("_", 1)[0]
        ids.add(scene_id)
    return ids
