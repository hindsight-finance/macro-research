from __future__ import annotations

from pathlib import Path

import polars as pl

from utils import data_sources

MARKET_TZ = "America/New_York"
UTC = "UTC"
BASE_COLUMNS = ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
OPTIONAL_BASE_COLUMNS = ["instrument"]


def _read_any(path: str | Path, storage_options: dict | None = None) -> pl.DataFrame:
    # Keep the source as a string so remote URLs aren't mangled by Path (s3://x -> s3:/x);
    # only the extension is taken via Path. storage_options apply to remote URLs only.
    source = str(path)
    suffix = Path(source).suffix.lower()
    opts = storage_options if data_sources.is_remote(source) else None
    if suffix == ".parquet":
        return pl.read_parquet(source, storage_options=opts)
    if suffix == ".csv":
        return pl.read_csv(source, try_parse_dates=True, storage_options=opts)
    raise ValueError(f"Unsupported input format: {source}")


def _to_utc_expr(column: str) -> pl.Expr:
    dtype = pl.col(column).cast(pl.String).str.to_datetime(time_zone=UTC, strict=False)
    return pl.when(pl.col(column).is_not_null()).then(dtype).otherwise(None).alias("datetime_utc")


def _et_to_utc_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.to_datetime(time_zone=MARKET_TZ, strict=True)
        .dt.convert_time_zone(UTC)
        .alias("datetime_utc")
    )


def normalize_minute_bars(df: pl.DataFrame) -> pl.DataFrame:
    work = df.clone()

    if "datetime_utc" in work.columns:
        work = work.with_columns(_to_utc_expr("datetime_utc"))
    elif "DateTime_UTC" in work.columns:
        work = work.with_columns(_to_utc_expr("DateTime_UTC"))
    elif "DateTime_ET" in work.columns:
        try:
            work = work.with_columns(_et_to_utc_expr("DateTime_ET"))
        except Exception as exc:
            raise ValueError(
                "Legacy ET timestamps contain ambiguous DST-fallback values; provide datetime_utc instead."
            ) from exc
    elif "datetime_et" in work.columns:
        try:
            work = work.with_columns(_et_to_utc_expr("datetime_et"))
        except Exception as exc:
            raise ValueError(
                "Legacy ET timestamps contain ambiguous DST-fallback values; provide datetime_utc instead."
            ) from exc
    else:
        raise ValueError("Expected one of: datetime_utc, DateTime_UTC, DateTime_ET, datetime_et")

    if work.select(pl.col("datetime_utc").is_null().any()).item():
        raise ValueError("Timestamp column contains unparsable values")

    missing = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    columns = BASE_COLUMNS + [column for column in OPTIONAL_BASE_COLUMNS if column in work.columns]
    out = work.select(columns).sort("datetime_utc")

    if out.select(pl.col("datetime_utc").is_duplicated().any()).item():
        raise ValueError("Duplicate datetime_utc values after normalization")

    return out


def load_minute_bars(path: str | Path, *, storage_options: dict | None = None) -> pl.DataFrame:
    # Auto-resolve R2 credentials for remote sources so callers can stay unchanged.
    if storage_options is None and data_sources.is_remote(path):
        storage_options = data_sources.storage_options()
    return normalize_minute_bars(_read_any(path, storage_options=storage_options))


def build_market_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "datetime_utc" not in df.columns:
        raise ValueError("Expected canonical datetime_utc column")

    return df.with_columns(
        datetime_et=pl.col("datetime_utc").dt.convert_time_zone(MARKET_TZ).dt.replace_time_zone(None),
    ).with_columns(
        date_et=pl.col("datetime_et").dt.date(),
        time_et=pl.col("datetime_et").dt.time(),
        minute_of_day_et=pl.col("datetime_et").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime_et").dt.minute().cast(pl.Int32),
    )


def derive_session_window(df: pl.DataFrame) -> pl.DataFrame:
    work = build_market_time_columns(df) if "minute_of_day_et" not in df.columns else df.clone()
    mins = pl.col("minute_of_day_et")

    return work.with_columns(
        session=pl.when((mins >= 19 * 60) & (mins < 24 * 60))
        .then(pl.lit("ASIA"))
        .when((mins >= 2 * 60) & (mins < 5 * 60))
        .then(pl.lit("LONDON"))
        .when((mins >= 9 * 60 + 30) & (mins < 11 * 60))
        .then(pl.lit("NYAM"))
        .when((mins >= 12 * 60) & (mins < 13 * 60))
        .then(pl.lit("LUNCH"))
        .when((mins >= 13 * 60) & (mins < 15 * 60))
        .then(pl.lit("PM"))
        .otherwise(pl.lit("OTHER")),
        window=pl.when((mins >= 15 * 60) & (mins <= 15 * 60 + 49))
        .then(pl.lit("H3PM"))
        .when((mins >= 15 * 60 + 50) & (mins <= 15 * 60 + 59))
        .then(pl.lit("MACRO"))
        .when((mins >= 16 * 60) & (mins <= 16 * 60 + 10))
        .then(pl.lit("POST"))
        .otherwise(pl.lit("NONE")),
    )
