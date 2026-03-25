"""2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) vessel detector."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from portvolume.config import CFARParams
from portvolume.sites import Site

logger = logging.getLogger(__name__)


def ca_cfar_2d(
    image: np.ndarray,
    guard_cells: int,
    background_cells: int,
    pfa: float,
) -> np.ndarray:
    """2D Cell-Averaging CFAR detector.

    Operates on LINEAR power (not dB). CFAR assumes a multiplicative
    noise model where the background follows an exponential distribution.

    Uses scipy.ndimage.uniform_filter for O(1)-per-pixel sliding windows.
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

    threshold = alpha * background_mean
    detections = clean > threshold

    detections[np.isnan(image)] = False

    return detections


def detect_vessels(
    processed_path: Path,
    cfar_params: CFARParams,
    site: Site,
    water_mask: np.ndarray | None = None,
) -> tuple:
    """Run CFAR vessel detection on a preprocessed scene.

    Parameters
    ----------
    processed_path : Path to preprocessed GeoTIFF (dB scale).
    cfar_params : CFAR algorithm parameters.
    site : Site object for metadata.
    water_mask : Optional boolean mask (True=water). If provided, land pixels
        are set to NaN before CFAR so they never trigger detections.

    Returns (detection_mask, db_values, transform, crs).
    The dB image is returned so post-detection filters can gate on brightness.
    """
    with rasterio.open(processed_path) as src:
        db_values = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs

    # Convert dB back to linear power for CFAR
    linear = np.power(10.0, db_values / 10.0)

    # Mask land pixels before CFAR — they become NaN and won't trigger detections
    if water_mask is not None:
        linear[~water_mask] = np.nan
        db_values[~water_mask] = np.nan

    mask = ca_cfar_2d(
        image=linear,
        guard_cells=cfar_params.guard_cells,
        background_cells=cfar_params.background_cells,
        pfa=cfar_params.pfa,
    )

    n_detections = int(mask.sum())
    logger.info(
        "CFAR on %s/%s: %d detection pixels%s",
        site.id,
        processed_path.stem,
        n_detections,
        " (water-masked)" if water_mask is not None else "",
    )

    return mask, db_values, transform, crs
