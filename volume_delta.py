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
MACRO_5S_BUCKETS = 120
SECONDS_PER_MACRO_BUCKET = 5
UTC_NS = pl.Datetime("ns", time_zone="UTC")

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
        datetime_et=(
            pl.col("ts_event")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(ET_TZ)
        ),
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


def build_macro_volume_delta_5s(path: str | Path) -> pl.LazyFrame:
    """Return lazy 5-second volume-delta rows for 15:50-16:00 ET with empty buckets."""
    base = (
        _scan_required_tick_columns(path)
        .with_columns(
            datetime_et=pl.col("ts_event").dt.convert_time_zone(ET_TZ),
            datetime_utc=pl.col("ts_event").dt.truncate("5s").cast(UTC_NS),
        )
        .with_columns(
            trade_date_et=pl.col("datetime_et").dt.date(),
            minute=pl.col("datetime_et").dt.minute().cast(pl.Int32),
            second=pl.col("datetime_et").dt.second().cast(pl.Int32),
        )
        .filter(
            (pl.col("datetime_et").dt.hour() == 15)
            & (pl.col("minute") >= MACRO_START_MINUTE)
            & (pl.col("minute") < MACRO_END_MINUTE)
        )
        .with_columns(
            macro_bucket_index=(
                ((pl.col("minute") - MACRO_START_MINUTE) * 60 + pl.col("second"))
                // SECONDS_PER_MACRO_BUCKET
            ).cast(pl.UInt8)
        )
    )

    aggregated = base.group_by("datetime_utc", "trade_date_et", "macro_bucket_index").agg(*_delta_agg())

    bucket_grid = (
        base.select("trade_date_et")
        .unique()
        .join(
            pl.LazyFrame({"macro_bucket_index": pl.Series(range(MACRO_5S_BUCKETS), dtype=pl.UInt8)}),
            how="cross",
        )
        .with_columns(
            datetime_et=pl.datetime(
                pl.col("trade_date_et").dt.year(),
                pl.col("trade_date_et").dt.month(),
                pl.col("trade_date_et").dt.day(),
                15,
                50,
                0,
                time_zone=ET_TZ,
            )
            + pl.duration(seconds=pl.col("macro_bucket_index").cast(pl.Int64) * SECONDS_PER_MACRO_BUCKET)
        )
        .with_columns(datetime_utc=pl.col("datetime_et").dt.convert_time_zone("UTC").cast(UTC_NS))
        .select("datetime_utc", "trade_date_et", "macro_bucket_index")
    )

    return (
        bucket_grid.join(
            aggregated,
            on=["datetime_utc", "trade_date_et", "macro_bucket_index"],
            how="left",
        )
        .with_columns(
            buy_size=pl.col("buy_size").fill_null(0),
            sell_size=pl.col("sell_size").fill_null(0),
            none_size=pl.col("none_size").fill_null(0),
            classified_size=pl.col("classified_size").fill_null(0),
            total_size=pl.col("total_size").fill_null(0),
            volume_delta=pl.col("volume_delta").fill_null(0),
            buy_ticks=pl.col("buy_ticks").fill_null(0),
            sell_ticks=pl.col("sell_ticks").fill_null(0),
            none_ticks=pl.col("none_ticks").fill_null(0),
            tick_delta=pl.col("tick_delta").fill_null(0),
        )
        .with_columns(
            delta_imbalance=_safe_ratio(pl.col("volume_delta"), pl.col("classified_size")),
            classified_share=_safe_ratio(pl.col("classified_size"), pl.col("total_size")),
            is_empty=(pl.col("total_size") == 0),
        )
        .select(["datetime_utc", "trade_date_et", "macro_bucket_index", *DELTA_COLUMNS, "is_empty"])
        .sort("trade_date_et", "macro_bucket_index")
    )


def build_globex_volume_delta_1m(path: str | Path) -> pl.LazyFrame:
    """Return lazy 1-minute volume-delta rows for 18:00-17:00 ET Globex sessions."""
    minute_of_day = (
        pl.col("datetime_et").dt.hour().cast(pl.Int16) * 60
        + pl.col("datetime_et").dt.minute().cast(pl.Int16)
    )
    session_start_minute = 18 * 60
    session_end_minute = 17 * 60

    return (
        _with_et_columns(_scan_required_tick_columns(path))
        .with_columns(
            minute_of_day=minute_of_day,
            trade_date_et=pl.when(minute_of_day >= session_start_minute)
            .then(pl.col("datetime_et").dt.offset_by("1d").dt.date())
            .otherwise(pl.col("datetime_et").dt.date()),
            session_minute_index=pl.when(minute_of_day >= session_start_minute)
            .then(minute_of_day - session_start_minute)
            .otherwise(minute_of_day + (24 * 60 - session_start_minute)),
        )
        .filter(
            (pl.col("minute_of_day") >= session_start_minute)
            | (pl.col("minute_of_day") < session_end_minute)
        )
        .group_by("datetime_utc", "trade_date_et", "session_minute_index")
        .agg(*_delta_agg())
        .select(["datetime_utc", "trade_date_et", "session_minute_index", *DELTA_COLUMNS])
        .sort("datetime_utc")
    )

def _sink(lf: pl.LazyFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(output)
    return output

def write_globex_volume_delta_1m(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_GLOBEX_1M_PATH,
) -> Path:
    return _sink(build_globex_volume_delta_1m(input_path), output_path)

def write_macro_volume_delta_1m(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_MACRO_1M_PATH,
) -> Path:
    return _sink(build_macro_volume_delta_1m(input_path), output_path)

def write_macro_volume_delta_5s(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_MACRO_5S_PATH,
) -> Path:
    return _sink(build_macro_volume_delta_5s(input_path), output_path)

def main() -> None:
    for output in (
        write_globex_volume_delta_1m(),
        write_macro_volume_delta_1m(),
        write_macro_volume_delta_5s(),
    ):
        print(f"[OK] Wrote volume delta → {output}")


if __name__ == "__main__":
    main()
