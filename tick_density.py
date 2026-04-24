#!/usr/bin/env python3
"""
Build UTC-only 1-minute tick-density rows for the market-close macro.

Tick input is intentionally processed with Polars lazy scans. Do not replace
scan_parquet with eager reads for the full tick file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_COLUMNS, get_tick_schema

INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_tick_density.parquet")

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

_REQUIRED_COLUMNS = {"ts_event", "side", "size"}


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def build_macro_tick_density(path: str | Path) -> pl.LazyFrame:
    """Return lazy 1-minute macro tick-density aggregates from sanitized ticks."""
    _validate_tick_schema(path)

    minute_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ).dt.minute()
    hour_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ).dt.hour()

    return (
        pl.scan_parquet(path)
        .select("ts_event", "side", "size")
        .filter((hour_et == 15) & (minute_et >= 50) & (minute_et <= 59))
        .with_columns(
            datetime_utc=pl.col("ts_event").dt.truncate("1m"),
            macro_minute_index=(minute_et - 50).cast(pl.UInt8),
        )
        .group_by("datetime_utc", "macro_minute_index")
        .agg(
            pl.len().alias("tick_count"),
            pl.col("size").sum().alias("total_size"),
            (pl.col("side") == 2).sum().alias("buy_ticks"),
            (pl.col("side") == 1).sum().alias("sell_ticks"),
            (pl.col("side") == 0).sum().alias("none_ticks"),
        )
        .with_columns(date_utc=pl.col("datetime_utc").dt.date())
        .select(MACRO_TICK_DENSITY_COLUMNS)
        .sort("datetime_utc")
    )


def write_macro_tick_density(input_path: str | Path = INPUT_PATH, output_path: str | Path = OUTPUT_PATH) -> Path:
    """Write macro tick-density parquet from lazy plan and return output path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_macro_tick_density(input_path).sink_parquet(output)
    return output


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    output = write_macro_tick_density(INPUT_PATH, OUTPUT_PATH)
    print(f"[OK] Wrote macro tick density → {output}")


if __name__ == "__main__":
    main()
