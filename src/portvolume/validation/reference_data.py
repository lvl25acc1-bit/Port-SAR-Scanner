"""Reference data loading and alignment for validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReferenceDataset:
    """A ground truth dataset for validation."""

    name: str
    source: str  # "gfw_ais", "eurostat", "port_authority", "unctad", "manual"
    frequency: str  # "daily", "weekly", "monthly", "quarterly"
    coverage: list[str]  # list of port_ids covered
    data: pd.DataFrame  # must have 'date' and 'value' columns
    unit: str = ""

    def __post_init__(self) -> None:
        if "date" not in self.data.columns or "value" not in self.data.columns:
            raise ValueError("ReferenceDataset.data must have 'date' and 'value' columns")
        self.data["date"] = pd.to_datetime(self.data["date"])


def load_manual_reference(
    csv_path: Path,
    port_id: str,
    date_col: str = "date",
    value_col: str = "value",
    frequency: str = "monthly",
) -> ReferenceDataset:
    """Load reference data from a CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    data = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]),
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    }).dropna(subset=["value"])

    logger.info("Loaded manual reference from %s: %d rows", csv_path, len(data))

    return ReferenceDataset(
        name=csv_path.stem,
        source="manual",
        frequency=frequency,
        coverage=[port_id],
        data=data,
    )


def create_synthetic_reference(
    port_id: str,
    start_date: str,
    end_date: str,
    frequency: str = "weekly",
    trend: float = 0.0,
    seasonality_amplitude: float = 10.0,
    noise_std: float = 5.0,
) -> ReferenceDataset:
    """Create synthetic reference data for testing.

    Useful when real reference data is not yet available.
    """
    freq_map = {
        "daily": "D",
        "weekly": "W-MON",
        "monthly": "MS",
        "quarterly": "QS",
    }
    pd_freq = freq_map.get(frequency, "W-MON")

    dates = pd.date_range(start=start_date, end=end_date, freq=pd_freq)
    n = len(dates)

    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=float)

    # Trend + seasonality + noise
    seasonal_period = {"daily": 365, "weekly": 52, "monthly": 12, "quarterly": 4}.get(
        frequency, 52
    )
    values = (
        100.0
        + trend * t
        + seasonality_amplitude * np.sin(2 * np.pi * t / seasonal_period)
        + rng.normal(0, noise_std, n)
    )

    data = pd.DataFrame({"date": dates, "value": values})

    logger.info(
        "Created synthetic reference for %s: %d %s observations",
        port_id, n, frequency,
    )

    return ReferenceDataset(
        name=f"synthetic_{port_id}",
        source="manual",
        frequency=frequency,
        coverage=[port_id],
        data=data,
    )


def align_reference_to_sar(
    sar_index: pd.DataFrame,
    reference: ReferenceDataset,
    sar_date_col: str = "week",
    sar_value_col: str = "mean",
) -> pd.DataFrame:
    """Align reference data to SAR index frequency.

    - If reference is higher freq (daily -> weekly): aggregate (mean)
    - If reference is lower freq (monthly -> weekly): forward-fill

    Returns DataFrame with columns: [date, sar_value, reference_value]
    """
    sar = sar_index[[sar_date_col, sar_value_col]].copy()
    sar.columns = ["date", "sar_value"]
    sar["date"] = pd.to_datetime(sar["date"])
    sar = sar.sort_values("date").reset_index(drop=True)

    ref = reference.data[["date", "value"]].copy()
    ref.columns = ["date", "reference_value"]
    ref = ref.sort_values("date").reset_index(drop=True)

    freq_order = {"daily": 0, "weekly": 1, "monthly": 2, "quarterly": 3}
    ref_rank = freq_order.get(reference.frequency, 1)
    # Assume SAR is weekly
    sar_rank = 1

    if ref_rank < sar_rank:
        # Reference is higher frequency — aggregate to weekly
        ref = ref.set_index("date")
        ref = ref.resample("W-MON").mean().reset_index()
    elif ref_rank > sar_rank:
        # Reference is lower frequency — forward-fill to weekly
        # Create a weekly date range covering the SAR dates
        weekly_dates = pd.date_range(
            start=min(sar["date"].min(), ref["date"].min()),
            end=max(sar["date"].max(), ref["date"].max()),
            freq="W-MON",
        )
        weekly_ref = pd.DataFrame({"date": weekly_dates})
        ref = ref.sort_values("date")

        # Merge_asof: for each weekly date, take the most recent reference value
        weekly_ref = pd.merge_asof(
            weekly_ref.sort_values("date"),
            ref.sort_values("date"),
            on="date",
            direction="backward",
        )
        ref = weekly_ref

    # Merge on closest dates (within 4 days tolerance)
    merged = pd.merge_asof(
        sar.sort_values("date"),
        ref.sort_values("date"),
        on="date",
        tolerance=pd.Timedelta("4D"),
        direction="nearest",
    )

    merged = merged.dropna(subset=["sar_value", "reference_value"]).reset_index(drop=True)

    logger.info(
        "Aligned SAR (%d rows) with reference '%s' (%d rows) -> %d matched rows",
        len(sar), reference.name, len(reference.data), len(merged),
    )

    return merged
