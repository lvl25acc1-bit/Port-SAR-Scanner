"""Map visualizations using Folium."""

from __future__ import annotations

from pathlib import Path

import folium

from portvolume.sites import Site


def create_sites_map(
    sites: list[Site],
    output_path: Path | None = None,
) -> folium.Map:
    """Create a Folium map showing all monitored sites.

    Ports are shown as blue markers, airports as red markers.
    """
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    for site in sites:
        color = "blue" if site.site_type == "port" else "red"
        icon = "ship" if site.site_type == "port" else "plane"

        folium.Marker(
            location=[site.lat, site.lon],
            popup=f"<b>{site.name}</b><br>Type: {site.site_type}<br>ID: {site.id}",
            tooltip=site.name,
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

        # Draw bounding box
        w, s, e, n = site.bbox
        folium.Rectangle(
            bounds=[[s, w], [n, e]],
            color=color,
            fill=True,
            fill_opacity=0.1,
            weight=1,
        ).add_to(m)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))

    return m


def create_detections_map(
    detections_gdf,
    site: Site,
    output_path: Path | None = None,
) -> folium.Map:
    """Create a map showing vessel/aircraft detections for a single scene."""
    m = folium.Map(
        location=[site.lat, site.lon],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    # Draw site bbox
    w, s, e, n = site.bbox
    folium.Rectangle(
        bounds=[[s, w], [n, e]],
        color="yellow",
        fill=False,
        weight=2,
    ).add_to(m)

    # Plot detections
    for _, row in detections_gdf.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=f"Pixels: {row.get('pixel_count', 'N/A')}",
        ).add_to(m)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))

    return m
