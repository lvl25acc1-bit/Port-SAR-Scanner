"""Adaptive CA-CFAR that adjusts threshold based on wind/wave conditions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from portvolume.config import CFARParams
from portvolume.detection.wind_correction import (
    WaveAgeParams,
    classify_wave_age,
    compute_cfar_correction_factor,
)
from portvolume.sites import Site

logger = logging.getLogger(__name__)


def adaptive_ca_cfar_2d(
    image: np.ndarray,
    guard_cells: int,
    background_cells: int,
    pfa: float,
    correction_factor: float = 1.0,
) -> np.ndarray:
    """2D Cell-Averaging CFAR detector with adaptive threshold correction.

    Identical to ``ca_cfar_2d`` when *correction_factor* is 1.0. When
    *correction_factor* > 1.0 the threshold is raised, reducing false alarms
    in rough-sea conditions. When < 1.0 the threshold is lowered.

    Parameters
    ----------
    image : 2-D array of LINEAR power (not dB).
    guard_cells, background_cells, pfa : Standard CFAR parameters.
    correction_factor : Multiplicative scaling applied to the CFAR threshold.
    """
    outer_size = 2 * (guard_cells + background_cells) + 1
    inner_size = 2 * guard_cells + 1

    n_outer = outer_size**2
    n_inner = inner_size**2
    n_bg = n_outer - n_inner

    if n_bg <= 0:
        raise ValueError(
            f"Invalid CFAR window: n_bg={n_bg}. "
            f"Increase background_cells or decrease guard_cells."
        )

    alpha = n_bg * (pfa ** (-1.0 / n_bg) - 1.0)

    clean = np.nan_to_num(image, nan=0.0)

    outer_mean = uniform_filter(clean, size=outer_size, mode="reflect")
    inner_mean = uniform_filter(clean, size=inner_size, mode="reflect")

    background_mean = (outer_mean * n_outer - inner_mean * n_inner) / n_bg
    background_mean = np.maximum(background_mean, 1e-20)

    threshold = alpha * background_mean * correction_factor
    detections = clean > threshold

    detections[np.isnan(image)] = False

    return detections


def detect_vessels_adaptive(
    processed_path: Path,
    cfar_params: CFARParams,
    site: Site,
    water_mask: Optional[np.ndarray] = None,
    wind_speed_mps: Optional[float] = None,
    wave_age_class: Optional[str] = None,
    wave_age_params: Optional[WaveAgeParams] = None,
) -> tuple:
    """Drop-in replacement for ``detect_vessels()`` with wind correction.

    If *wind_speed_mps* is provided, computes a correction factor and uses
    the adaptive CFAR. Otherwise falls back to standard CFAR (correction_factor=1.0).

    Returns ``(detection_mask, db_values, transform, crs)`` — same as
    ``detect_vessels()``.
    """
    with rasterio.open(processed_path) as src:
        db_values = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs

    # Convert dB back to linear power for CFAR
    linear = np.power(10.0, db_values / 10.0)

    # Mask land pixels
    if water_mask is not None:
        linear[~water_mask] = np.nan
        db_values[~water_mask] = np.nan

    # Compute correction factor if wind data is available
    correction_factor = 1.0
    if wind_speed_mps is not None:
        if wave_age_class is None:
            wave_age_class = classify_wave_age(wind_speed_mps)
        correction_factor = compute_cfar_correction_factor(
            wave_age_class, wind_speed_mps, wave_age_params,
        )
        logger.info(
            "Adaptive CFAR: wind=%.1f m/s, wave_age=%s, correction=%.3f",
            wind_speed_mps, wave_age_class, correction_factor,
        )

    mask = adaptive_ca_cfar_2d(
        image=linear,
        guard_cells=cfar_params.guard_cells,
        background_cells=cfar_params.background_cells,
        pfa=cfar_params.pfa,
        correction_factor=correction_factor,
    )

    n_detections = int(mask.sum())
    logger.info(
        "Adaptive CFAR on %s/%s: %d detection pixels%s",
        site.id,
        processed_path.stem,
        n_detections,
        " (water-masked)" if water_mask is not None else "",
    )

    return mask, db_values, transform, crs
