"""Extended site definitions for commodity-specific port selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from portvolume.sites import Site

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommoditySite(Site):
    """Extended Site with commodity and throughput metadata."""

    primary_commodity: str = ""  # "crude_oil", "iron_ore", "grain", "lng", "coal", "containers"
    commodity_group: str = ""  # "energy", "dry_bulk", "agriculture", "general"
    throughput_variance_pct: float = 0.0  # historical coefficient of variation
    anchorage_bbox: tuple[float, float, float, float] | None = None  # separate anchorage AOI
    economic_indicators: tuple[str, ...] = ()  # linked FRED/yfinance series IDs


def compute_site_score(site: CommoditySite) -> float:
    """Score a site for signal quality potential.

    Factors (weighted):
    - throughput_variance_pct (40% weight, higher = better)
    - single commodity focus: 1.0 for specialized, 0.5 for mixed (20% weight)
    - bbox area: penalize very large areas (20% weight)
    - has anchorage_bbox: bonus 0.2 if true (20% weight)

    Returns score 0-1.
    """
    # Throughput variance component (40% weight)
    # Normalize: 30% variance -> 1.0, 0% -> 0.0
    variance_score = min(site.throughput_variance_pct / 30.0, 1.0)

    # Commodity focus component (20% weight)
    specialized_commodities = {"crude_oil", "iron_ore", "grain", "lng", "coal"}
    if site.primary_commodity in specialized_commodities:
        focus_score = 1.0
    elif site.primary_commodity:
        focus_score = 0.5
    else:
        focus_score = 0.0

    # Bbox area component (20% weight) — penalize very large areas
    bbox_width = abs(site.bbox[2] - site.bbox[0])
    bbox_height = abs(site.bbox[3] - site.bbox[1])
    bbox_area_deg2 = bbox_width * bbox_height
    # Ideal area ~0.01-0.05 deg^2; penalize above 0.1 deg^2
    if bbox_area_deg2 <= 0.05:
        area_score = 1.0
    elif bbox_area_deg2 <= 0.2:
        area_score = 1.0 - (bbox_area_deg2 - 0.05) / 0.15 * 0.5
    else:
        area_score = max(0.0, 0.5 - (bbox_area_deg2 - 0.2) / 0.5)

    # Anchorage bbox component (20% weight)
    anchorage_score = 1.0 if site.anchorage_bbox is not None else 0.0

    score = (
        0.40 * variance_score
        + 0.20 * focus_score
        + 0.20 * area_score
        + 0.20 * anchorage_score
    )

    return round(min(max(score, 0.0), 1.0), 4)


def select_optimal_sites(
    candidates: list[CommoditySite],
    n_sites: int = 13,
    min_per_commodity: int = 2,
) -> list[CommoditySite]:
    """Select best portfolio maximizing commodity diversity and signal potential.

    Ensures at least ``min_per_commodity`` sites per commodity_group (when
    enough candidates exist), then fills remaining slots by highest score.
    """
    if len(candidates) <= n_sites:
        return sorted(candidates, key=lambda s: compute_site_score(s), reverse=True)

    # Group candidates by commodity_group
    groups: dict[str, list[CommoditySite]] = {}
    for site in candidates:
        groups.setdefault(site.commodity_group, []).append(site)

    # Sort each group by score descending
    for group in groups.values():
        group.sort(key=lambda s: compute_site_score(s), reverse=True)

    selected: list[CommoditySite] = []
    selected_ids: set[str] = set()

    # First pass: ensure minimum per commodity_group
    for group_name, group_sites in groups.items():
        count = 0
        for site in group_sites:
            if count >= min_per_commodity:
                break
            if site.id not in selected_ids:
                selected.append(site)
                selected_ids.add(site.id)
                count += 1

    # Second pass: fill remaining by score
    all_by_score = sorted(candidates, key=lambda s: compute_site_score(s), reverse=True)
    for site in all_by_score:
        if len(selected) >= n_sites:
            break
        if site.id not in selected_ids:
            selected.append(site)
            selected_ids.add(site.id)

    # Sort final list by score descending
    selected.sort(key=lambda s: compute_site_score(s), reverse=True)
    return selected
