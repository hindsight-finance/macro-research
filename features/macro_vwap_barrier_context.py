from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema

TICK_INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
BARRIER_INPUT_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
VWAP_INPUT_PATH = Path("outputs/nq_macro_vwap_intramacro.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_vwap_barrier_context_summary.parquet")
UTC_NS = pl.Datetime("ns", time_zone="UTC")

TICK_REQUIRED_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}
BARRIER_REQUIRED_COLUMNS = {
    "date", "macro_trend_state", "barrier_extreme", "barrier_price", "barrier_time",
    "barrier_first10", "barrier_is_macro_extreme", "barrier_holds", "edge_case",
}
VWAP_REQUIRED_COLUMNS = {
    "date",
    "macro_1550_at_1550_10s_vwap_side",
    "macro_1550_at_1550_10s_vwap_dist_points",
    "macro_1550_at_1550_10s_vwap_dist_bps",
    "macro_1550_at_1555_vwap_side",
    "macro_1550_at_1555_vwap_dist_points",
    "macro_1550_at_1555_vwap_dist_bps",
    "target_1550_1554_points", "target_1550_1554_sign", "target_1550_1554_state",
    "target_1555_1559_points", "target_1555_1559_sign", "target_1555_1559_state",
    "target_1550_1559_points", "target_1550_1559_sign", "target_1550_1559_state",
}

TARGET_PREFIXES = (
    "target_1550_10s_1554",
    "target_1550_10s_1559",
    "target_1551_1559",
    "target_1555_1559",
)

MACRO_VWAP_BARRIER_CONTEXT_COLUMNS = [
    "date",
    "macro_trend_state",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "edge_case",
    "vwap_10s_side",
    "vwap_10s_dist_points",
    "vwap_10s_dist_bps",
    "vwap_10s_constructive",
    "barrier_first10_and_vwap_constructive",
    "barrier_ts_utc",
    "post_barrier_tick_count_1550",
    "vwap_side_at_barrier",
    "vwap_dist_at_barrier_points",
    "vwap_side_at_1550_close",
    "vwap_dist_at_1550_close_points",
    "closed_wrong_side_1550",
    "closed_wrong_side_more_than_1tick",
    "closed_wrong_side_more_than_2pts",
    "closed_wrong_side_more_than_5pts",
    "worst_wrong_side_dist_points",
    "worst_wrong_side_dist_bps",
    "seconds_wrong_side_vwap",
    "wrong_side_share_1550",
    "vwap_1555_side",
    "vwap_1555_dist_points",
    "vwap_1555_dist_bps",
    "vwap_1555_constructive",
    "vwap_context_10s_to_1555",
    "barrier_holds_and_1555_constructive",
    "barrier_first10_and_1555_constructive",
    "target_1550_1554_points", "target_1550_1554_sign", "target_1550_1554_state",
    "target_1555_1559_points", "target_1555_1559_sign", "target_1555_1559_state",
    "target_1550_1559_points", "target_1550_1559_sign", "target_1550_1559_state",
    "target_1550_10s_1554_points", "target_1550_10s_1554_sign", "target_1550_10s_1554_state",
    "target_1550_10s_1559_points", "target_1550_10s_1559_sign", "target_1550_10s_1559_state",
    "target_1551_1559_points", "target_1551_1559_sign", "target_1551_1559_state",
]

MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS = [
    "scope", "bucket", "target_name", "sample_size", "bullish_count", "bearish_count", "neutral_count",
    "bullish_pct", "bearish_pct", "neutral_pct", "avg_target_points", "median_target_points",
    "p10_target_points", "p25_target_points", "p75_target_points", "p90_target_points",
]


def classify_constructive_side(macro_trend_state: str | None, vwap_side: str | None) -> str:
    raise NotImplementedError


def build_macro_vwap_barrier_context(barrier: pl.DataFrame, vwap: pl.DataFrame, tick_path: str | Path) -> pl.DataFrame:
    raise NotImplementedError


def summarize_macro_vwap_barrier_context(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


def write_macro_vwap_barrier_context(
    tick_path: str | Path = TICK_INPUT_PATH,
    barrier_path: str | Path = BARRIER_INPUT_PATH,
    vwap_path: str | Path = VWAP_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
