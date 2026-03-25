"""Tests for Module 3: Site Selection Pivot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from portvolume.sites import Site
from portvolume.sites_v2 import CommoditySite, compute_site_score, select_optimal_sites
from portvolume.sites_selection.throughput_data import compute_throughput_variance

# Path to project config directory
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class TestCommoditySiteInheritsSite:
    """CommoditySite is a valid Site (isinstance check), has all Site fields."""

    def test_commodity_site_inherits_site(self):
        site = CommoditySite(
            id="test_port",
            name="Test Port",
            lat=10.0,
            lon=20.0,
            bbox=(19.9, 9.9, 20.1, 10.1),
            site_type="port",
            primary_commodity="crude_oil",
            commodity_group="energy",
            throughput_variance_pct=15.0,
            economic_indicators=("DCOILWTICO",),
        )
        assert isinstance(site, Site)
        # All core Site fields are accessible
        assert site.id == "test_port"
        assert site.name == "Test Port"
        assert site.lat == 10.0
        assert site.lon == 20.0
        assert site.bbox == (19.9, 9.9, 20.1, 10.1)
        assert site.site_type == "port"
        # Extended fields
        assert site.primary_commodity == "crude_oil"
        assert site.commodity_group == "energy"
        assert site.throughput_variance_pct == 15.0
        assert site.economic_indicators == ("DCOILWTICO",)
        # Site properties still work
        assert site.bbox_width_deg == pytest.approx(0.2, abs=1e-6)
        assert site.bbox_height_deg == pytest.approx(0.2, abs=1e-6)


class TestSiteScorePrefersHighVariance:
    """Port with 20% variance scores higher than 5% variance."""

    def test_site_score_prefers_high_variance(self):
        base_kwargs = dict(
            id="port",
            name="Port",
            lat=0.0,
            lon=0.0,
            bbox=(0.0, 0.0, 0.1, 0.1),
            site_type="port",
            primary_commodity="crude_oil",
            commodity_group="energy",
        )
        high_var = CommoditySite(**base_kwargs, throughput_variance_pct=20.0)
        low_var = CommoditySite(**base_kwargs, throughput_variance_pct=5.0)

        assert compute_site_score(high_var) > compute_site_score(low_var)


class TestSelectOptimalSitesDiversity:
    """Given 20 candidates, selection includes min 2 per commodity_group."""

    def test_select_optimal_sites_diversity(self):
        # Create 20 candidates across 4 commodity groups
        groups = ["energy", "dry_bulk", "agriculture", "general"]
        commodities = ["crude_oil", "iron_ore", "grain", "containers"]
        candidates: list[CommoditySite] = []
        for i in range(20):
            grp_idx = i % 4
            candidates.append(
                CommoditySite(
                    id=f"port_{i}",
                    name=f"Port {i}",
                    lat=float(i),
                    lon=float(i),
                    bbox=(0.0, 0.0, 0.1, 0.1),
                    site_type="port",
                    primary_commodity=commodities[grp_idx],
                    commodity_group=groups[grp_idx],
                    throughput_variance_pct=10.0 + i,
                )
            )

        selected = select_optimal_sites(candidates, n_sites=13, min_per_commodity=2)
        assert len(selected) == 13

        # Check diversity: at least 2 per group
        group_counts: dict[str, int] = {}
        for s in selected:
            group_counts[s.commodity_group] = group_counts.get(s.commodity_group, 0) + 1

        for grp in groups:
            assert group_counts.get(grp, 0) >= 2, f"Expected >= 2 for {grp}, got {group_counts.get(grp, 0)}"


class TestThroughputVarianceCalculation:
    """Known time series [100, 110, 90, 105, 95] -> correct CV."""

    def test_throughput_variance_calculation(self):
        df = pd.DataFrame({"throughput": [100, 110, 90, 105, 95]})
        cv = compute_throughput_variance(df, window_months=24)

        # Population std of [100, 110, 90, 105, 95]:
        #   mean = 100, deviations = [0, 10, -10, 5, -5]
        #   var = (0+100+100+25+25)/5 = 50, std = sqrt(50) ~ 7.071
        #   CV = 7.071 / 100 ~ 0.0707
        expected_cv = (50 ** 0.5) / 100.0
        assert cv == pytest.approx(expected_cv, abs=1e-4)


class TestSitesV2YamlLoadable:
    """sites_v2.yaml loads correctly with all 13 sites, each having required fields."""

    def test_sites_v2_yaml_loadable(self):
        yaml_path = CONFIG_DIR / "sites_v2.yaml"
        assert yaml_path.exists(), f"{yaml_path} not found"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        sites = data["sites"]
        assert len(sites) == 13

        required_fields = {"id", "name", "lat", "lon", "bbox", "site_type", "primary_commodity", "commodity_group"}
        for site in sites:
            missing = required_fields - set(site.keys())
            assert not missing, f"Site {site.get('id', '?')} missing fields: {missing}"
            assert len(site["bbox"]) == 4
            assert site["primary_commodity"] in {
                "crude_oil", "iron_ore", "grain", "lng", "coal", "containers",
            }


class TestEconomicIndicatorsLinked:
    """ras_tanura has ("DCOILWTICO", "BZ=F"), santos has ("ZS=F", "ZC=F")."""

    def test_economic_indicators_linked(self):
        yaml_path = CONFIG_DIR / "sites_v2.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        sites_by_id = {s["id"]: s for s in data["sites"]}

        ras_tanura = sites_by_id["ras_tanura"]
        assert tuple(ras_tanura["economic_indicators"]) == ("DCOILWTICO", "BZ=F")

        santos = sites_by_id["santos"]
        assert tuple(santos["economic_indicators"]) == ("ZS=F", "ZC=F")
