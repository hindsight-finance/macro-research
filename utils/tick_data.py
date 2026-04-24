from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

TICK_COLUMNS = ["ts_event", "intra_ts_rank", "side", "price_ticks", "size"]
TICK_PRICE_DENOMINATOR = 4.0


def get_tick_schema(path: str | Path) -> pa.Schema:
    return pq.ParquetFile(path).schema_arrow


def scan_tick_data(path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(path).select(TICK_COLUMNS)


def _dt(value: Any) -> pl.Expr:
    if isinstance(value, str):
        return pl.lit(value).str.to_datetime(time_zone="UTC")
    return pl.lit(value)


def _bounded_filter(lf: pl.LazyFrame, start_utc: Any, end_utc: Any) -> pl.LazyFrame:
    if start_utc is None or end_utc is None:
        raise ValueError("Tick collection requires bounded start/end UTC timestamps.")
    return lf.filter((pl.col("ts_event") >= _dt(start_utc)) & (pl.col("ts_event") < _dt(end_utc)))


def collect_tick_window(path: str | Path, start_utc: Any, end_utc: Any) -> pl.DataFrame:
    return _bounded_filter(scan_tick_data(path), start_utc, end_utc).collect(engine="streaming")


def ticks_to_minute_bars(lf: pl.LazyFrame, start_utc: Any, end_utc: Any) -> pl.DataFrame:
    bounded = _bounded_filter(lf, start_utc, end_utc)
    return (
        bounded.with_columns(
            datetime_utc=pl.col("ts_event").dt.truncate("1m"),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .group_by("datetime_utc")
        .agg(
            pl.col("price").first().alias("Open"),
            pl.col("price").max().alias("High"),
            pl.col("price").min().alias("Low"),
            pl.col("price").last().alias("Close"),
            pl.col("size").sum().alias("Volume"),
        )
        .sort("datetime_utc")
        .collect(engine="streaming")
    )
