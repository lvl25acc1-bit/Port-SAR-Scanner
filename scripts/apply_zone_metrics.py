"""
apply_zone_metrics.py
---------------------
Compute zone-level congestion metrics from zone_detections.parquet,
merge with scene_universe for dates, and produce scene-level and
weekly-aggregated output parquets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ZONE_DET_PATH = ROOT / "data" / "backtest" / "zone_detections.parquet"
SCENE_UNI_PATH = ROOT / "data" / "backtest" / "scene_universe.parquet"
OUT_DIR = ROOT / "data" / "backtest" / "outputs"
SCENE_OUT = OUT_DIR / "zone_metrics_scenes.parquet"
WEEKLY_OUT = OUT_DIR / "zone_metrics_weekly.parquet"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source parquets and print basic diagnostics."""
    zd = pd.read_parquet(ZONE_DET_PATH)
    su = pd.read_parquet(SCENE_UNI_PATH)

    print("=" * 60)
    print("SOURCE DATA")
    print("=" * 60)
    print(f"zone_detections  : {zd.shape[0]:,} rows x {zd.shape[1]} cols")
    print(f"  columns        : {zd.columns.tolist()}")
    zd["zone_type"] = zd["zone_type"].fillna("unknown")
    print(f"  zone_type vals : {sorted(zd['zone_type'].unique())}")
    print(f"  zone_type counts:")
    for zt, cnt in zd["zone_type"].value_counts().items():
        print(f"    {str(zt):20s} {cnt:>6,}")
    print()
    print(f"scene_universe   : {su.shape[0]:,} rows x {su.shape[1]} cols")
    print(f"  columns        : {su.columns.tolist()}")
    print(f"  date range     : {su['date'].min()} -> {su['date'].max()}")
    print()
    return zd, su


def compute_scene_zone_metrics(zd: pd.DataFrame) -> pd.DataFrame:
    """
    For each scene_id compute:
      anchorage_count  – vessels in approach / anchorage zones
      berth_count      – vessels in basin / berth zones
      channel_count    – vessels in channel zones
      congestion_index – anchorage / (anchorage + berth), 0 if denom==0
      queue_length_proxy – anchorage + channel
    """
    records: list[dict] = []

    for scene_id, grp in zd.groupby("scene_id"):
        # Categorise each row by zone_type substring
        zt_lower = grp["zone_type"].str.lower()

        anchorage_mask = zt_lower.str.contains("approach|anchorage", na=False)
        berth_mask = zt_lower.str.contains("basin|berth", na=False)
        channel_mask = zt_lower.str.contains("channel", na=False)

        anchorage_count = int(grp.loc[anchorage_mask, "zone_vessel_count"].sum())
        berth_count = int(grp.loc[berth_mask, "zone_vessel_count"].sum())
        channel_count = int(grp.loc[channel_mask, "zone_vessel_count"].sum())

        denom = anchorage_count + berth_count
        congestion_index = anchorage_count / denom if denom > 0 else 0.0
        queue_length_proxy = anchorage_count + channel_count

        records.append(
            {
                "scene_id": scene_id,
                "anchorage_count": anchorage_count,
                "berth_count": berth_count,
                "channel_count": channel_count,
                "congestion_index": congestion_index,
                "queue_length_proxy": queue_length_proxy,
            }
        )

    return pd.DataFrame(records)


def merge_dates(metrics: pd.DataFrame, su: pd.DataFrame) -> pd.DataFrame:
    """Left-join scene metrics with scene_universe to attach dates."""
    su_slim = su[["scene_id", "date"]].drop_duplicates(subset="scene_id")
    merged = metrics.merge(su_slim, on="scene_id", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def build_weekly(scene_df: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregation of congestion_index, anchorage_count, berth_count."""
    df = scene_df.dropna(subset=["date"]).copy()
    df = df.set_index("date")

    agg_spec = {
        "congestion_index": ["mean", "max"],
        "anchorage_count": ["mean", "sum"],
        "berth_count": ["mean", "sum"],
        "queue_length_proxy": ["mean"],
        "scene_id": "count",
    }
    weekly = df.resample("W-MON").agg(agg_spec)

    # Flatten multi-level columns
    weekly.columns = ["_".join(col).strip() for col in weekly.columns]
    weekly = weekly.rename(columns={"scene_id_count": "n_scenes"})
    weekly = weekly[weekly["n_scenes"] > 0]
    weekly = weekly.reset_index().rename(columns={"date": "week"})
    return weekly


def print_summary(scene_df: pd.DataFrame, weekly_df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SCENE-LEVEL SUMMARY")
    print("=" * 60)
    print(f"Total scenes with metrics: {len(scene_df):,}")
    print(f"Mean congestion_index    : {scene_df['congestion_index'].mean():.4f}")
    print(f"Median congestion_index  : {scene_df['congestion_index'].median():.4f}")
    print(f"Mean anchorage_count     : {scene_df['anchorage_count'].mean():.2f}")
    print(f"Mean berth_count         : {scene_df['berth_count'].mean():.2f}")
    print(f"Mean channel_count       : {scene_df['channel_count'].mean():.2f}")
    print(f"Mean queue_length_proxy  : {scene_df['queue_length_proxy'].mean():.2f}")
    print()

    print("=" * 60)
    print("WEEKLY TIME SERIES (first 20 weeks)")
    print("=" * 60)
    display_cols = [c for c in weekly_df.columns]
    with pd.option_context("display.max_rows", 25, "display.width", 140, "display.float_format", "{:.3f}".format):
        print(weekly_df.head(20).to_string(index=False))
    print()

    if len(weekly_df) > 0:
        print("=" * 60)
        print("WEEKLY TRENDS")
        print("=" * 60)
        print(f"Weeks covered            : {len(weekly_df)}")
        print(f"Date range               : {weekly_df['week'].min().date()} -> {weekly_df['week'].max().date()}")
        print(f"Avg weekly congestion    : {weekly_df['congestion_index_mean'].mean():.4f}")
        print(f"Max weekly congestion    : {weekly_df['congestion_index_max'].max():.4f}")
        print(f"Avg weekly anchorage     : {weekly_df['anchorage_count_mean'].mean():.2f}")
        print(f"Avg weekly berth         : {weekly_df['berth_count_mean'].mean():.2f}")
        print()


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    zd, su = load_data()

    # ------------------------------------------------------------------
    # 2. Compute scene-level zone metrics
    # ------------------------------------------------------------------
    scene_metrics = compute_scene_zone_metrics(zd)
    print(f"Computed zone metrics for {len(scene_metrics):,} scenes.\n")

    # ------------------------------------------------------------------
    # 3. Merge dates
    # ------------------------------------------------------------------
    scene_metrics = merge_dates(scene_metrics, su)
    n_dated = scene_metrics["date"].notna().sum()
    print(f"Matched dates for {n_dated:,} / {len(scene_metrics):,} scenes.\n")

    # ------------------------------------------------------------------
    # 4. Weekly aggregation
    # ------------------------------------------------------------------
    weekly = build_weekly(scene_metrics)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_metrics.to_parquet(SCENE_OUT, index=False)
    weekly.to_parquet(WEEKLY_OUT, index=False)
    print(f"Saved scene-level  -> {SCENE_OUT}")
    print(f"Saved weekly       -> {WEEKLY_OUT}")
    print()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print_summary(scene_metrics, weekly)


if __name__ == "__main__":
    main()
