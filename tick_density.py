#!/usr/bin/env python3
"""
Build UTC-only tick-density rows for market-close macro windows.

Tick input is intentionally processed with Polars lazy scans. Do not replace
scan_parquet with eager reads for the full tick file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from utils.minute_bars import MARKET_TZ
from utils.tick_data import get_tick_schema

INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_tick_density.parquet")
OUTPUT_DIR = Path("outputs")
DEFAULT_5S_MACRO_MINUTES = (50, 54, 55, 59)
UTC_NS = pl.Datetime("ns", time_zone="UTC")

MACRO_TICK_DENSITY_COLUMNS = [
    "datetime_utc",
    "date_utc",
    "macro_minute_index",
    "tick_count",
    "total_size",
    "buy_ticks",
    "sell_ticks",
    "none_ticks",
]

MACRO_5S_TICK_DENSITY_COLUMNS = [
    "datetime_utc",
    "date_utc",
    "bucket_index",
    "is_empty",
    "tick_count",
    "total_size",
    "buy_ticks",
    "sell_ticks",
    "none_ticks",
]

_REQUIRED_COLUMNS = {"ts_event", "intra_ts_rank", "side", "size"}


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _scan_required_tick_columns(path: str | Path) -> pl.LazyFrame:
    _validate_tick_schema(path)
    return pl.scan_parquet(path).select(
        pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
        "intra_ts_rank",
        "side",
        "size",
    )


def _tick_agg() -> list[pl.Expr]:
    return [
        pl.len().alias("tick_count"),
        pl.col("size").sum().alias("total_size"),
        (pl.col("side") == 2).sum().alias("buy_ticks"),
        (pl.col("side") == 1).sum().alias("sell_ticks"),
        (pl.col("side") == 0).sum().alias("none_ticks"),
    ]


def build_macro_tick_density(path: str | Path) -> pl.LazyFrame:
    """Return lazy 1-minute macro tick-density aggregates from sanitized ticks."""
    minute_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ).dt.minute()
    hour_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ).dt.hour()

    return (
        _scan_required_tick_columns(path)
        .filter((hour_et == 15) & (minute_et >= 50) & (minute_et <= 59))
        .with_columns(
            datetime_utc=pl.col("ts_event").dt.truncate("1m").cast(UTC_NS),
            macro_minute_index=(minute_et - 50).cast(pl.UInt8),
        )
        .group_by("datetime_utc", "macro_minute_index")
        .agg(*_tick_agg())
        .with_columns(date_utc=pl.col("datetime_utc").dt.date())
        .select(MACRO_TICK_DENSITY_COLUMNS)
        .sort("datetime_utc")
    )


def _bucket_grid(aggregated: pl.LazyFrame) -> pl.LazyFrame:
    sessions = aggregated.select(pl.col("datetime_utc").dt.truncate("1m").alias("minute_start_utc")).unique()
    buckets = pl.LazyFrame({"bucket_index": pl.Series(range(12), dtype=pl.UInt8)})

    return sessions.join(buckets, how="cross").with_columns(
        datetime_utc=(
            pl.col("minute_start_utc").cast(UTC_NS)
            + pl.duration(nanoseconds=pl.col("bucket_index").cast(pl.Int64) * 5_000_000_000)
        )
    )


def build_macro_5s_tick_density(path: str | Path, macro_minute: int) -> pl.LazyFrame:
    """Return lazy 5-second bucket tick-density rows for one ET macro minute."""
    if macro_minute < 0 or macro_minute > 59:
        raise ValueError(f"macro_minute must be 0..59, got {macro_minute}")

    minute_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ).dt.minute()
    hour_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ).dt.hour()

    aggregated = (
        _scan_required_tick_columns(path)
        .filter((hour_et == 15) & (minute_et == macro_minute))
        .with_columns(
            datetime_utc=pl.col("ts_event").dt.truncate("5s").cast(UTC_NS),
            bucket_index=(pl.col("ts_event").dt.second() // 5).cast(pl.UInt8),
        )
        .group_by("datetime_utc", "bucket_index")
        .agg(*_tick_agg())
    )

    return (
        _bucket_grid(aggregated)
        .join(aggregated, on=["datetime_utc", "bucket_index"], how="left")
        .drop("minute_start_utc")
        .with_columns(
            date_utc=pl.col("datetime_utc").dt.date(),
            tick_count=pl.col("tick_count").fill_null(0),
            total_size=pl.col("total_size").fill_null(0),
            buy_ticks=pl.col("buy_ticks").fill_null(0),
            sell_ticks=pl.col("sell_ticks").fill_null(0),
            none_ticks=pl.col("none_ticks").fill_null(0),
        )
        .with_columns(is_empty=(pl.col("tick_count") == 0))
        .select(MACRO_5S_TICK_DENSITY_COLUMNS)
        .sort("datetime_utc")
    )


def macro_5s_output_path(output_dir: str | Path, macro_minute: int) -> Path:
    return Path(output_dir) / f"nq_macro_tick_density_15{macro_minute:02d}_5s.parquet"


def write_macro_tick_density(input_path: str | Path = INPUT_PATH, output_path: str | Path = OUTPUT_PATH) -> Path:
    """Write macro tick-density parquet from lazy plan and return output path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_macro_tick_density(input_path).sink_parquet(output)
    return output


def write_macro_5s_tick_density_files(
    input_path: str | Path = INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    macro_minutes: tuple[int, ...] = DEFAULT_5S_MACRO_MINUTES,
) -> list[Path]:
    """Write one 5-second tick-density parquet per requested ET macro minute."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    wrote = []
    for macro_minute in macro_minutes:
        output = macro_5s_output_path(output_root, macro_minute)
        build_macro_5s_tick_density(input_path, macro_minute).sink_parquet(output)
        wrote.append(output)
    return wrote


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    output = write_macro_tick_density(INPUT_PATH, OUTPUT_PATH)
    outputs_5s = write_macro_5s_tick_density_files(INPUT_PATH, OUTPUT_DIR)
    print(f"[OK] Wrote macro tick density → {output}")
    for path in outputs_5s:
        print(f"[OK] Wrote macro 5s tick density → {path}")


if __name__ == "__main__":
    main()
