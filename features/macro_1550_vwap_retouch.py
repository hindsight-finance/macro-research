#!/usr/bin/env python3
"""First-10s barrier-break direction signal + VWAP-retouch event study (15:50 macro open).

Causal: the first side of the first-10s barrier range [low_10s, high_10s] to break after
15:50:10 ET sets the directional bias; a retouch of the first-10s VWAP (frozen) or the rolling
15:50-anchored VWAP is the candidate entry. Forward outcomes are measured from the break and
from each retouch. The realised 15:50-candle and macro direction are recorded as outcomes only,
never as signals. Descriptive event study — no target/stop backtest in this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from utils import data_sources
from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema, scan_source

INPUT_PATH = data_sources.tick_data_url()
OUTPUT_PATH = Path("outputs/nq_macro_1550_vwap_retouch.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_1550_vwap_retouch_summary.parquet")

UTC_NS = pl.Datetime("ns", time_zone="UTC")
DEFAULT_BARRIER_SECONDS = 10
TOUCH_THRESHOLD_POINTS = 0.25
_REQUIRED_TICK_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}


def _ets(hour: int, minute: int, second: int = 0) -> int:
    return hour * 3600 + minute * 60 + second


# ET-second-of-day window boundaries (time-of-day only; date-independent).
S_1550 = _ets(15, 50)
S_1551 = _ets(15, 51)
S_1555 = _ets(15, 55)
S_1600 = _ets(16, 0)
S_1601 = _ets(16, 1)

MACRO_1550_VWAP_RETOUCH_COLUMNS = [
    "date", "tick_count_macro", "has_first10", "trigger_state",
    "open_1550", "high_10s", "low_10s", "range_10s_points", "vwap_10s_frozen", "vol_share_first10",
    "break_side", "bias", "break_time_s", "break_ts_utc", "break_price",
    "retouch_frozen_occurred", "retouch_frozen_time_s", "retouch_frozen_ts_utc",
    "retouch_frozen_price", "retouch_frozen_lag_s",
    "retouch_rolling_occurred", "retouch_rolling_time_s", "retouch_rolling_ts_utc",
    "retouch_rolling_price", "retouch_rolling_lag_s", "vwap_rolling_at_retouch",
    "fwd_break_1554_points", "fwd_break_1559_points", "fwd_break_1600_points",
    "mfe_break_points", "mae_break_points",
    "fwd_retouch_frozen_1554_points", "fwd_retouch_frozen_1559_points", "fwd_retouch_frozen_1600_points",
    "mfe_retouch_frozen_points", "mae_retouch_frozen_points",
    "fwd_retouch_rolling_1554_points", "fwd_retouch_rolling_1559_points", "fwd_retouch_rolling_1600_points",
    "mfe_retouch_rolling_points", "mae_retouch_rolling_points",
    "candle_1550_dir_points", "candle_1550_dir_sign", "candle_1550_state",
    "macro_dir_points", "macro_dir_sign", "macro_trend_state",
    "bias_matches_1550_candle", "bias_matches_macro",
]

_STR_COLUMNS = {"trigger_state", "break_side", "bias", "candle_1550_state", "macro_trend_state"}
_BOOL_COLUMNS = {
    "has_first10", "retouch_frozen_occurred", "retouch_rolling_occurred",
    "bias_matches_1550_candle", "bias_matches_macro",
}
_INT_COLUMNS = {
    "tick_count_macro", "break_time_s", "retouch_frozen_time_s", "retouch_frozen_lag_s",
    "retouch_rolling_time_s", "retouch_rolling_lag_s", "candle_1550_dir_sign", "macro_dir_sign",
}
_TS_COLUMNS = {"break_ts_utc", "retouch_frozen_ts_utc", "retouch_rolling_ts_utc"}


def _schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for col in MACRO_1550_VWAP_RETOUCH_COLUMNS:
        if col == "date":
            schema[col] = pl.Date
        elif col in _STR_COLUMNS:
            schema[col] = pl.String
        elif col in _BOOL_COLUMNS:
            schema[col] = pl.Boolean
        elif col in _INT_COLUMNS:
            schema[col] = pl.Int64
        elif col in _TS_COLUMNS:
            schema[col] = UTC_NS
        else:
            schema[col] = pl.Float64
    return schema


def _blank_row(date, tick_count_macro: int, has_first10: bool) -> dict:
    row = {col: None for col in MACRO_1550_VWAP_RETOUCH_COLUMNS}
    row["date"] = date
    row["tick_count_macro"] = tick_count_macro
    row["has_first10"] = has_first10
    return row


def _scan_macro_window(path: str | Path) -> pl.LazyFrame:
    """Lazy, R2-aware scan of the 15:50:00–16:00:59 ET window across all dates.

    Reads through ``scan_source`` (synthesises ``price_ticks`` from the lake's float price and
    attaches R2 ``storage_options`` for s3:// URLs); ET is derived from UTC so DST is correct.
    """
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_TICK_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")
    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    et_second = (
        ts_et.dt.hour().cast(pl.Int32) * 3600
        + ts_et.dt.minute().cast(pl.Int32) * 60
        + ts_et.dt.second().cast(pl.Int32)
    )
    return (
        scan_source(path)
        .select(
            pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
            pl.col("intra_ts_rank").cast(pl.Int64),
            pl.col("price_ticks").cast(pl.Int64),
            pl.col("size").cast(pl.Int64),
        )
        .with_columns(
            date=ts_et.dt.date(),
            et_second=et_second,
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .filter((pl.col("et_second") >= S_1550) & (pl.col("et_second") < S_1601))
    )
