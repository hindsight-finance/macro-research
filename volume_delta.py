from __future__ import annotations

from pathlib import Path

import polars as pl

from utils.tick_data import TICK_COLUMNS, get_tick_schema

INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
OUTPUT_GLOBEX_1M_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
OUTPUT_MACRO_1M_PATH = Path("outputs/nq_macro_volume_delta_1m.parquet")
OUTPUT_MACRO_5S_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")

ET_TZ = "America/New_York"
MACRO_START_MINUTE = 50
MACRO_END_MINUTE = 60

DELTA_COLUMNS = [
    "buy_size",
    "sell_size",
    "none_size",
    "classified_size",
    "total_size",
    "volume_delta",
    "delta_imbalance",
    "buy_ticks",
    "sell_ticks",
    "none_ticks",
    "tick_delta",
    "classified_share",
]


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = [column for column in TICK_COLUMNS if column not in schema.names]
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _scan_required_tick_columns(path: str | Path) -> pl.LazyFrame:
    _validate_tick_schema(path)
    return pl.scan_parquet(path).select(TICK_COLUMNS)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def _delta_agg() -> list[pl.Expr]:
    buy_size = pl.when(pl.col("side") == 2).then(pl.col("size")).otherwise(0).sum()
    sell_size = pl.when(pl.col("side") == 1).then(pl.col("size")).otherwise(0).sum()
    none_size = pl.when(pl.col("side") == 0).then(pl.col("size")).otherwise(0).sum()
    buy_ticks = (pl.col("side") == 2).sum()
    sell_ticks = (pl.col("side") == 1).sum()
    none_ticks = (pl.col("side") == 0).sum()

    classified_size = buy_size + sell_size
    total_size = classified_size + none_size
    volume_delta = buy_size - sell_size
    tick_delta = buy_ticks - sell_ticks

    return [
        buy_size.alias("buy_size"),
        sell_size.alias("sell_size"),
        none_size.alias("none_size"),
        classified_size.alias("classified_size"),
        total_size.alias("total_size"),
        volume_delta.alias("volume_delta"),
        _safe_ratio(volume_delta, classified_size).alias("delta_imbalance"),
        buy_ticks.alias("buy_ticks"),
        sell_ticks.alias("sell_ticks"),
        none_ticks.alias("none_ticks"),
        tick_delta.alias("tick_delta"),
        _safe_ratio(classified_size, total_size).alias("classified_share"),
    ]


def _with_et_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        datetime_et=pl.col("ts_event").dt.convert_time_zone(ET_TZ),
        datetime_utc=pl.col("ts_event").dt.truncate("1m"),
    )


def build_macro_volume_delta_1m(path: str | Path) -> pl.LazyFrame:
    """Return lazy 1-minute volume-delta rows for 15:50-16:00 ET."""
    return (
        _with_et_columns(_scan_required_tick_columns(path))
        .with_columns(
            trade_date_et=pl.col("datetime_et").dt.date(),
            macro_minute_index=pl.col("datetime_et").dt.minute(),
        )
        .filter(
            (pl.col("datetime_et").dt.hour() == 15)
            & (pl.col("macro_minute_index") >= MACRO_START_MINUTE)
            & (pl.col("macro_minute_index") < MACRO_END_MINUTE)
        )
        .group_by("datetime_utc", "trade_date_et", "macro_minute_index")
        .agg(*_delta_agg())
        .sort("datetime_utc")
    )
