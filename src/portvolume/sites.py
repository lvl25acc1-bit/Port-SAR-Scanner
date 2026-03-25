"""Site dataclass definitions and bounding box helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Site:
    """A monitored port or airport site."""

    id: str
    name: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float]  # (west, south, east, north)
    site_type: str  # "port" or "airport"
    region: str = ""
    annual_teu_millions: float = 0.0
    role: str = ""
    primary_commodity: str = ""
    commodity_group: str = ""
    throughput_variance_pct: float = 0.0
    anchorage_bbox: tuple[float, float, float, float] | None = None
    economic_indicators: tuple[str, ...] = ()

    @property
    def bbox_width_deg(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height_deg(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_dict(self) -> dict[str, float]:
        """Return bbox as a named dict for STAC search."""
        return {
            "west": self.bbox[0],
            "south": self.bbox[1],
            "east": self.bbox[2],
            "north": self.bbox[3],
        }

    def to_shapely_box(self):
        """Return a shapely box geometry for this site's AOI."""
        from shapely.geometry import box

        return box(*self.bbox)
