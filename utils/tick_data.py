from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from utils import data_sources

TICK_COLUMNS = ["ts_event", "intra_ts_rank", "side", "price_ticks", "size"]
TICK_PRICE_DENOMINATOR = 4.0

# ts_event is stored as datetime[ns, UTC]; keep filter bounds at ns precision so
# parquet row-group pruning (predicate pushdown) stays enabled, including over R2
# where it is what makes range reads cheap (see pola-rs/polars #25731).
UTC_NS = pl.Datetime("ns", "UTC")


def _resolve(path: str | Path | None, storage_options: dict | None) -> tuple[str | Path, dict | None]:
    """Fill in the default tick-data location and R2 storage options when unset."""
    if path is None:
        path = data_sources.tick_data_url()
    if storage_options is None:
        storage_options = data_sources.storage_options()
    return path, storage_options


def scan_source(path: str | Path | None = None, *, storage_options: dict | None = None) -> pl.LazyFrame:
    """Lazy scan of a parquet source (local path or R2 ``s3://`` URL), unprojected.

    ``storage_options`` are only attached when ``path`` is a remote URL, so a local
    read is never handed R2 credentials even when they are present in the environment
    (a runner reads remote ticks but local derived parquet within the same process).
    """
    path, storage_options = _resolve(path, storage_options)
    if not data_sources.is_remote(path):
        storage_options = None
    return pl.scan_parquet(path, storage_options=storage_options)


def get_tick_schema(path: str | Path | None = None, *, storage_options: dict | None = None) -> pa.Schema:
    """Arrow schema of the tick parquet, read from metadata only (no row data).

    Implemented via polars so it works transparently over R2; callers only use the
    returned ``.names`` to validate that required columns are present.
    """
    return scan_source(path, storage_options=storage_options).limit(0).collect().to_arrow().schema


def scan_tick_data(path: str | Path | None = None, *, storage_options: dict | None = None) -> pl.LazyFrame:
    return scan_source(path, storage_options=storage_options).select(TICK_COLUMNS)


def open_parquet_file(path: str | Path | None = None, *, storage_options: dict | None = None):
    """Open a ``pyarrow.parquet.ParquetFile`` for batch iteration, R2-aware.

    pyarrow cannot infer a custom R2 endpoint from an ``s3://`` URI alone, so for
    remote URLs we build an explicit ``S3FileSystem``; local paths open directly.
    """
    import pyarrow.parquet as pq

    path, storage_options = _resolve(path, storage_options)
    if data_sources.is_remote(path):
        import pyarrow.fs as pafs
        from urllib.parse import urlparse

        opts = storage_options or {}
        parsed = urlparse(str(path))
        filesystem = pafs.S3FileSystem(
            access_key=opts.get("aws_access_key_id"),
            secret_key=opts.get("aws_secret_access_key"),
            endpoint_override=opts.get("endpoint_url"),
            region=opts.get("aws_region") or "auto",
        )
        return pq.ParquetFile(filesystem.open_input_file(f"{parsed.netloc}{parsed.path}"))
    return pq.ParquetFile(path)


def _dt(value: Any) -> pl.Expr:
    if isinstance(value, str):
        return pl.lit(value).str.to_datetime(time_zone="UTC").cast(UTC_NS)
    return pl.lit(value).cast(UTC_NS)


def _bounded_filter(lf: pl.LazyFrame, start_utc: Any, end_utc: Any) -> pl.LazyFrame:
    if start_utc is None or end_utc is None:
        raise ValueError("Tick collection requires bounded start/end UTC timestamps.")
    return lf.filter((pl.col("ts_event") >= _dt(start_utc)) & (pl.col("ts_event") < _dt(end_utc)))


def collect_tick_window(
    path: str | Path | None = None,
    start_utc: Any = None,
    end_utc: Any = None,
    *,
    storage_options: dict | None = None,
) -> pl.DataFrame:
    return _bounded_filter(
        scan_tick_data(path, storage_options=storage_options), start_utc, end_utc
    ).collect(engine="streaming")


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
