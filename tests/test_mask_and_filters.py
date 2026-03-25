"""Targeted tests for the three P1/P2 fixes:

1. _find_manual_mask does not return Rotterdam's mask for other sites.
2. Post-filter contrast uses dB subtraction, not ratio.
3. Per-site polarization override changes the file glob.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from affine import Affine

from portvolume.config import CFARParams, PostDetectionParams
from portvolume.detection.postprocess import cluster_detections
from portvolume.detection.water_mask import _find_manual_mask
from portvolume.sites import Site


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def mask_dir(tmp_path):
    """Create a temp config/water_masks/ with only rotterdam.gpkg."""
    d = tmp_path / "water_masks"
    d.mkdir()
    (d / "rotterdam.gpkg").write_text("fake")
    return d


@pytest.fixture
def sites():
    """A few sites for cross-contamination testing."""
    common = dict(lat=0, lon=0, bbox=(0, 0, 1, 1), site_type="port", region="europe")
    return {
        "rotterdam": Site(id="rotterdam", name="Port of Rotterdam", **common),
        "singapore": Site(id="singapore", name="Port of Singapore", **common),
        "busan": Site(id="busan", name="Port of Busan", **common),
    }


# -----------------------------------------------------------------------
# Test 1: manual mask lookup does not cross-contaminate
# -----------------------------------------------------------------------

class TestFindManualMask:
    def test_returns_rotterdam_for_rotterdam(self, mask_dir, sites):
        result = _find_manual_mask(sites["rotterdam"], mask_dir)
        assert result is not None
        assert result.name == "rotterdam.gpkg"

    def test_returns_none_for_singapore(self, mask_dir, sites):
        result = _find_manual_mask(sites["singapore"], mask_dir)
        assert result is None, (
            f"Expected None for singapore but got {result} — "
            "mask lookup is leaking Rotterdam's file to other sites"
        )

    def test_returns_none_for_busan(self, mask_dir, sites):
        result = _find_manual_mask(sites["busan"], mask_dir)
        assert result is None

    def test_backward_compat_name(self, mask_dir, sites):
        """Legacy naming {site_id}_manual_water_mask.gpkg should match."""
        (mask_dir / "singapore_manual_water_mask.gpkg").write_text("fake")
        result = _find_manual_mask(sites["singapore"], mask_dir)
        assert result is not None
        assert "singapore" in result.name

    def test_nonexistent_dir_returns_none(self, tmp_path, sites):
        result = _find_manual_mask(sites["rotterdam"], tmp_path / "nope")
        assert result is None


# -----------------------------------------------------------------------
# Test 2: dB contrast uses subtraction, not ratio
# -----------------------------------------------------------------------

class TestDbContrast:
    def _make_detection_scene(self, peak_db, mean_db, cluster_size=20):
        """Build a synthetic binary mask + dB image with one cluster."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=bool)
        db = np.full((h, w), -20.0)  # background

        # Place a cluster at centre
        r0, c0 = 50, 50
        half = int(np.sqrt(cluster_size))
        mask[r0:r0 + half, c0:c0 + half] = True
        # Fill cluster with values that produce the desired peak and mean
        cluster_vals = np.full((half, half), mean_db)
        cluster_vals[0, 0] = peak_db  # bright core
        db[r0:r0 + half, c0:c0 + half] = cluster_vals

        transform = Affine(10, 0, 0, 0, -10, 0)
        return mask, db, transform

    def test_positive_db_contrast_passes(self):
        """peak=10 dB, mean=4 dB → contrast=6 dB → should pass min_db_contrast=2."""
        mask, db, transform = self._make_detection_scene(peak_db=10.0, mean_db=4.0)
        params = PostDetectionParams(min_db=-12.0, max_aspect_ratio=10.0, min_db_contrast=2.0)
        gdf = cluster_detections(mask, 3, transform, db_image=db, post_params=params)
        assert len(gdf) == 1
        assert gdf.iloc[0]["db_contrast"] == pytest.approx(6.0, abs=0.5)

    def test_negative_db_values_contrast_works(self):
        """peak=-2 dB, mean=-8 dB → contrast=6 dB — must NOT be broken by negative values."""
        mask, db, transform = self._make_detection_scene(peak_db=-2.0, mean_db=-8.0)
        params = PostDetectionParams(min_db=-12.0, max_aspect_ratio=10.0, min_db_contrast=2.0)
        gdf = cluster_detections(mask, 3, transform, db_image=db, post_params=params)
        assert len(gdf) == 1
        assert gdf.iloc[0]["db_contrast"] == pytest.approx(6.0, abs=0.5)

    def test_low_contrast_rejected(self):
        """peak=-5 dB, mean=-6 dB → contrast=1 dB → rejected at min_db_contrast=2."""
        mask, db, transform = self._make_detection_scene(peak_db=-5.0, mean_db=-6.0)
        params = PostDetectionParams(min_db=-12.0, max_aspect_ratio=10.0, min_db_contrast=2.0)
        gdf = cluster_detections(mask, 3, transform, db_image=db, post_params=params)
        assert len(gdf) == 0

    def test_no_ratio_column(self):
        """Ensure the old 'peak_to_mean' column is gone."""
        mask, db, transform = self._make_detection_scene(peak_db=10.0, mean_db=4.0)
        params = PostDetectionParams(min_db=-12.0, max_aspect_ratio=10.0, min_db_contrast=2.0)
        gdf = cluster_detections(mask, 3, transform, db_image=db, post_params=params)
        assert "peak_to_mean" not in gdf.columns
        assert "db_contrast" in gdf.columns


# -----------------------------------------------------------------------
# Test 3: per-site polarization override changes file glob
# -----------------------------------------------------------------------

class TestPolarizationResolution:
    def test_port_uses_site_override_polarization(self):
        """If rotterdam overrides polarization='vh', the pipeline should
        glob *_vh.tif, not *_vv.tif."""
        cfar = CFARParams(
            guard_cells=5, background_cells=15, pfa=1e-8,
            min_target_pixels=20, polarization="vv",
            site_overrides={"rotterdam": {"polarization": "vh"}},
        )
        resolved = cfar.for_site("rotterdam")
        assert resolved.polarization == "vh"

    def test_non_overridden_site_inherits_global(self):
        cfar = CFARParams(
            guard_cells=5, background_cells=15, pfa=1e-8,
            min_target_pixels=20, polarization="vv",
            site_overrides={"rotterdam": {"polarization": "vh"}},
        )
        resolved = cfar.for_site("shanghai")
        assert resolved.polarization == "vv"

    def test_no_overrides_returns_global(self):
        cfar = CFARParams(
            guard_cells=5, background_cells=15, pfa=1e-8,
            min_target_pixels=20, polarization="vv",
        )
        resolved = cfar.for_site("anything")
        assert resolved.polarization == "vv"
