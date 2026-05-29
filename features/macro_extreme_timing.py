#!/usr/bin/env python3
"""
Build tick-level high/low formation timing for key market-close macro candles.

The tick input is processed with Polars lazy scans. Keep full tick-file reads lazy
and bounded to the requested ET macro minutes to avoid unnecessary memory use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import polars as pl

from utils import data_sources
from utils.minute_bars import MARKET_TZ
from utils.tick_data import TICK_PRICE_DENOMINATOR, get_tick_schema, scan_source

INPUT_PATH = data_sources.tick_data_url()
OUTPUT_PATH = Path("outputs/nq_macro_extreme_timing.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_extreme_timing_summary.parquet")
DEFAULT_KEY_MINUTES = (50, 54, 55, 59)
UTC_NS = pl.Datetime("ns", time_zone="UTC")

MACRO_EXTREME_TIMING_COLUMNS = [
    "date",
    "datetime_utc",
    "macro_minute_index",
    "candle_open",
    "candle_high",
    "candle_low",
    "candle_close",
    "candle_volume",
    "candle_dir_points",
    "candle_dir_sign",
    "candle_trend_state",
    "candle_high_time",
    "candle_low_time",
    "candle_high_ts_utc",
    "candle_low_ts_utc",
    "candle_high_first",
    "candle_extreme_gap_seconds",
    "macro_open",
    "macro_close",
    "macro_dir_points",
    "macro_dir_sign",
    "macro_trend_state",
]

MACRO_EXTREME_TIMING_SUMMARY_COLUMNS = [
    "macro_trend_state",
    "macro_minute_index",
    "extreme",
    "sample_size",
    "mean_time",
    "median_time",
    "std_time",
    "p10_time",
    "p25_time",
    "p75_time",
    "p90_time",
    "mean_gap_seconds",
    "median_gap_seconds",
    "high_first_pct",
]

_REQUIRED_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks", "size"}


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = sorted(_REQUIRED_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _scan_required_tick_columns(path: str | Path) -> pl.LazyFrame:
    _validate_tick_schema(path)
    return scan_source(path).select(
        pl.col("ts_event").cast(UTC_NS).alias("ts_event"),
        "intra_ts_rank",
        "price_ticks",
        "size",
    )


def _trend_state_expr(prefix: str) -> pl.Expr:
    sign_col = pl.col(f"{prefix}_dir_sign")
    return (
        pl.when(sign_col > 0)
        .then(pl.lit("bullish"))
        .when(sign_col < 0)
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral"))
    )


def build_macro_extreme_timing(
    path: str | Path,
    key_minutes: Iterable[int] = DEFAULT_KEY_MINUTES,
) -> pl.LazyFrame:
    """Return lazy per-date/per-key-candle first-touch high/low timing rows.

    Times are ET macro minute indexes and seconds inside each selected minute.
    Only dates with all requested key minutes are retained so macro direction is
    defined consistently from 15:50 open to 15:59 close.
    """
    minutes = tuple(int(m) for m in key_minutes)
    if not minutes:
        raise ValueError("key_minutes must not be empty")
    invalid = [m for m in minutes if m < 0 or m > 59]
    if invalid:
        raise ValueError(f"key_minutes must be 0..59, got {invalid}")

    ts_et = pl.col("ts_event").dt.convert_time_zone(MARKET_TZ)
    minute_start = pl.col("ts_event").dt.truncate("1m").cast(UTC_NS)
    price = pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR

    minute_bars = (
        _scan_required_tick_columns(path)
        .filter((ts_et.dt.hour() == 15) & (ts_et.dt.minute().is_in(minutes)))
        .with_columns(
            date=ts_et.dt.date(),
            datetime_utc=minute_start,
            macro_minute_index=ts_et.dt.minute().cast(pl.UInt8),
            price=price,
        )
        .sort("ts_event", "intra_ts_rank")
        .group_by("date", "datetime_utc", "macro_minute_index")
        .agg(
            pl.col("price").first().alias("candle_open"),
            pl.col("price").max().alias("candle_high"),
            pl.col("price").min().alias("candle_low"),
            pl.col("price").last().alias("candle_close"),
            pl.col("size").sum().alias("candle_volume"),
            pl.col("ts_event").filter(pl.col("price") == pl.col("price").max()).min().alias("candle_high_ts_utc"),
            pl.col("ts_event").filter(pl.col("price") == pl.col("price").min()).min().alias("candle_low_ts_utc"),
        )
    )

    complete_dates = (
        minute_bars.group_by("date")
        .agg(pl.col("macro_minute_index").n_unique().alias("key_minute_count"))
        .filter(pl.col("key_minute_count") == len(set(minutes)))
        .select("date")
    )

    macro_context = (
        minute_bars.join(complete_dates, on="date", how="inner")
        .group_by("date")
        .agg(
            pl.col("candle_open")
            .filter(pl.col("macro_minute_index") == 50)
            .first()
            .alias("macro_open"),
            pl.col("candle_close")
            .filter(pl.col("macro_minute_index") == 59)
            .first()
            .alias("macro_close"),
        )
        .drop_nulls(["macro_open", "macro_close"])
        .with_columns(
            macro_dir_points=pl.col("macro_close") - pl.col("macro_open"),
        )
        .with_columns(macro_dir_sign=pl.col("macro_dir_points").sign().fill_null(0).cast(pl.Int8))
        .with_columns(macro_trend_state=_trend_state_expr("macro"))
    )

    return (
        minute_bars.join(complete_dates, on="date", how="inner")
        .join(macro_context, on="date", how="inner")
        .with_columns(
            candle_dir_points=pl.col("candle_close") - pl.col("candle_open"),
            candle_high_time=pl.col("candle_high_ts_utc").dt.second().cast(pl.UInt8),
            candle_low_time=pl.col("candle_low_ts_utc").dt.second().cast(pl.UInt8),
        )
        .with_columns(candle_dir_sign=pl.col("candle_dir_points").sign().fill_null(0).cast(pl.Int8))
        .with_columns(
            candle_trend_state=_trend_state_expr("candle"),
            candle_high_first=pl.col("candle_high_ts_utc") <= pl.col("candle_low_ts_utc"),
            candle_extreme_gap_seconds=(
                (pl.col("candle_high_ts_utc") - pl.col("candle_low_ts_utc")).dt.total_seconds().abs().cast(pl.UInt8)
            ),
        )
        .select(MACRO_EXTREME_TIMING_COLUMNS)
        .sort("date", "macro_minute_index")
    )


def _summary_for_extreme(df: pl.DataFrame, extreme: str, time_col: str) -> pl.DataFrame:
    return (
        df.group_by("macro_trend_state", "macro_minute_index")
        .agg(
            pl.len().alias("sample_size"),
            pl.col(time_col).mean().alias("mean_time"),
            pl.col(time_col).median().alias("median_time"),
            pl.col(time_col).std().alias("std_time"),
            pl.col(time_col).quantile(0.10).alias("p10_time"),
            pl.col(time_col).quantile(0.25).alias("p25_time"),
            pl.col(time_col).quantile(0.75).alias("p75_time"),
            pl.col(time_col).quantile(0.90).alias("p90_time"),
            pl.col("candle_extreme_gap_seconds").mean().alias("mean_gap_seconds"),
            pl.col("candle_extreme_gap_seconds").median().alias("median_gap_seconds"),
            (pl.col("candle_high_first").mean() * 100.0).alias("high_first_pct"),
        )
        .with_columns(extreme=pl.lit(extreme))
        .select(MACRO_EXTREME_TIMING_SUMMARY_COLUMNS)
    )


def summarize_macro_extreme_timing(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize high/low timing variance by macro direction and key candle."""
    required = {
        "macro_trend_state",
        "macro_minute_index",
        "candle_high_time",
        "candle_low_time",
        "candle_extreme_gap_seconds",
        "candle_high_first",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return pl.concat(
        [
            _summary_for_extreme(df, "high", "candle_high_time"),
            _summary_for_extreme(df, "low", "candle_low_time"),
        ],
        how="vertical",
    ).sort("macro_trend_state", "macro_minute_index", "extreme")


def write_macro_extreme_timing(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
    key_minutes: Iterable[int] = DEFAULT_KEY_MINUTES,
) -> tuple[Path, Path]:
    """Write timing dataset and summary parquet files, returning both paths."""
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    build_macro_extreme_timing(input_path, key_minutes=key_minutes).sink_parquet(output)
    timing = pl.read_parquet(output)
    summarize_macro_extreme_timing(timing).write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not data_sources.source_exists(INPUT_PATH):
        print(f"[ERROR] Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    output, summary = write_macro_extreme_timing(INPUT_PATH, OUTPUT_PATH, SUMMARY_OUTPUT_PATH)
    print(f"[OK] Wrote macro extreme timing → {output}")
    print(f"[OK] Wrote macro extreme timing summary → {summary}")


if __name__ == "__main__":
    main()
