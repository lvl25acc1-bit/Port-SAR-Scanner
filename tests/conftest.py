"""Shared test fixtures: synthetic SAR arrays, mock data, etc."""

import numpy as np
import pandas as pd
import pytest

from portvolume.config import CFARParams, AirportParams
from portvolume.sites import Site


@pytest.fixture
def cfar_params():
    return CFARParams(
        guard_cells=5,
        background_cells=15,
        pfa=1e-6,
        min_target_pixels=3,
        polarization="vv",
    )


@pytest.fixture
def airport_params():
    return AirportParams(
        polarization="vv",
        threshold_db=-12.0,
        min_cluster_pixels=5,
    )


@pytest.fixture
def sample_port():
    return Site(
        id="rotterdam",
        name="Port of Rotterdam",
        lat=51.90,
        lon=4.40,
        bbox=(4.00, 51.85, 4.55, 51.98),
        site_type="port",
        region="europe",
        annual_teu_millions=15.3,
    )


@pytest.fixture
def sample_airport():
    return Site(
        id="memphis",
        name="Memphis International (MEM)",
        lat=35.04,
        lon=-89.98,
        bbox=(-90.02, 35.02, -89.94, 35.07),
        site_type="airport",
        role="cargo_hub",
    )


@pytest.fixture
def synthetic_sar_linear():
    """Generate a synthetic 512x512 SAR image in linear power scale.

    Background: Rayleigh-distributed amplitude → exponential power.
    5 bright targets placed at known locations, each ~30 dB above background.
    """
    rng = np.random.default_rng(42)
    size = 512

    # Background: exponential distribution (mean ~ 0.01 in linear power)
    background = rng.exponential(scale=0.01, size=(size, size))

    # Targets: 5 clusters of bright pixels (~30 dB above background = 1000x)
    target_positions = [
        (100, 100),
        (200, 300),
        (350, 150),
        (400, 400),
        (250, 250),
    ]
    target_power = 10.0  # ~30 dB above 0.01 background

    image = background.copy()
    for row, col in target_positions:
        # 5x5 pixel target
        image[row - 2 : row + 3, col - 2 : col + 3] = target_power

    return image, target_positions


@pytest.fixture
def synthetic_sar_db(synthetic_sar_linear):
    """Same as synthetic_sar_linear but in dB scale."""
    linear, positions = synthetic_sar_linear
    db = 10.0 * np.log10(np.maximum(linear, 1e-10))
    return db, positions


@pytest.fixture
def sample_weekly_detections():
    """Generate 52 weeks of fake detection counts with a trend and seasonality."""
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2023-01-02", periods=52, freq="W-MON")

    # Base: ~100 vessels + slight uptrend + weekly noise
    trend = np.linspace(100, 120, 52)
    seasonal = 10 * np.sin(np.linspace(0, 2 * np.pi, 52))
    noise = rng.normal(0, 5, 52)
    counts = trend + seasonal + noise

    return pd.DataFrame(
        {
            "week": weeks,
            "mean": counts,
            "max": counts + rng.uniform(10, 30, 52),
            "n_scenes": rng.integers(1, 5, 52),
            "site_id": "rotterdam",
            "site_type": "port",
        }
    )


@pytest.fixture
def sample_economic_data(sample_weekly_detections):
    """Generate fake economic data with known correlation to detection data."""
    rng = np.random.default_rng(42)
    det = sample_weekly_detections

    # Economic series that correlates with detections (r ~ 0.7)
    dates = det["week"]
    correlated = det["mean"].values * 0.5 + rng.normal(0, 3, len(det))
    uncorrelated = rng.normal(50, 10, len(det))

    return pd.DataFrame(
        {
            "CORRELATED_PMI": correlated,
            "UNCORRELATED_NOISE": uncorrelated,
        },
        index=dates,
    )
