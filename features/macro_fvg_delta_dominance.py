from __future__ import annotations

from pathlib import Path
import sys
import warnings

import polars as pl

try:
    from volume_delta import OUTPUT_MACRO_5S_PATH
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from volume_delta import OUTPUT_MACRO_5S_PATH

DELTA_DOMINANCE_COLUMNS = [
    "fvg_delta_bucket_index",
    "fvg_delta_volume_delta",
    "fvg_delta_imbalance",
    "fvg_delta_tick_delta",
    "fvg_delta_classified_share",
    "fvg_delta_total_size",
    "fvg_delta_is_empty",
    "aligned_delta_imbalance",
    "abs_delta_imbalance",
    "aligned_volume_delta",
    "abs_volume_delta",
    "aligned_tick_delta",
    "abs_tick_delta",
    "aligned_delta_imbalance_quantile",
    "abs_delta_imbalance_quantile",
]

REQUIRED_EVENT_COLUMNS = {"date", "fvg_side", "confirmed_at"}
REQUIRED_DELTA_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "delta_imbalance",
    "tick_delta",
    "classified_share",
    "total_size",
    "is_empty",
}

QUANTILE_LOW_LABEL = "q1_lowest"
QUANTILE_HIGH_LABEL_TEMPLATE = "q{bucket}_highest"


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _null_expr_for_dtype(column: str) -> pl.Expr:
    if column in {
        "fvg_delta_bucket_index",
        "fvg_delta_volume_delta",
        "fvg_delta_tick_delta",
        "fvg_delta_total_size",
        "aligned_volume_delta",
        "abs_volume_delta",
        "aligned_tick_delta",
        "abs_tick_delta",
    }:
        return pl.lit(None, dtype=pl.Int64).alias(column)
    if column == "fvg_delta_is_empty":
        return pl.lit(None, dtype=pl.Boolean).alias(column)
    return pl.lit(None, dtype=pl.Utf8 if column.endswith("_quantile") else pl.Float64).alias(column)


def _with_empty_delta_columns(events: pl.DataFrame) -> pl.DataFrame:
    existing = set(events.columns)
    return events.with_columns(
        [_null_expr_for_dtype(column) for column in DELTA_DOMINANCE_COLUMNS if column not in existing]
    )


def load_macro_volume_delta_5s(path: str | Path = OUTPUT_MACRO_5S_PATH) -> pl.DataFrame:
    delta_path = Path(path)
    delta = pl.read_parquet(delta_path)
    missing = _missing_columns(delta, REQUIRED_DELTA_COLUMNS)
    if missing:
        raise ValueError(f"Missing volume-delta columns: {missing}")
    return delta


def _with_confirmation_bucket(events: pl.DataFrame) -> pl.DataFrame:
    confirmed = pl.col("confirmed_at")
    seconds_since_macro_start = (
        (confirmed.dt.hour() - 15).cast(pl.Int64) * 3600
        + (confirmed.dt.minute() - 50).cast(pl.Int64) * 60
        + confirmed.dt.second().cast(pl.Int64)
    )
    return events.with_columns(
        fvg_delta_bucket_index=pl.when((seconds_since_macro_start >= 0) & (seconds_since_macro_start < 600))
        .then((seconds_since_macro_start // 5).cast(pl.Int64))
        .otherwise(None)
    )


def _quantile_labels(bucket_count: int) -> list[str]:
    return [
        QUANTILE_LOW_LABEL if i == 1 else (QUANTILE_HIGH_LABEL_TEMPLATE.format(bucket=i) if i == bucket_count else f"q{i}")
        for i in range(1, bucket_count + 1)
    ]


def _with_rank_quantile(frame: pl.DataFrame, value_col: str, output_col: str, bucket_count: int) -> pl.DataFrame:
    if frame.is_empty() or value_col not in frame.columns:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

    non_null_count = frame.select(pl.col(value_col).drop_nulls().len()).item()
    if non_null_count == 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

    unique_count = frame.select(pl.col(value_col).drop_nulls().n_unique()).item()
    q = int(min(bucket_count, unique_count, non_null_count))
    if q <= 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

    labels = _quantile_labels(q)
    label_lookup = pl.DataFrame(
        {
            "_bucket": list(range(q)),
            output_col: labels,
        }
    )

    ranked = frame.with_columns(
        pl.when(pl.col(value_col).is_not_null())
        .then(pl.col(value_col).rank(method="ordinal") - 1)
        .otherwise(None)
        .alias("_rank")
    ).with_columns(
        pl.when(pl.col("_rank").is_not_null())
        .then(((pl.col("_rank") * q) / non_null_count).floor().cast(pl.Int64).clip(0, q - 1))
        .otherwise(None)
        .alias("_bucket")
    )

    return ranked.join(label_lookup, on="_bucket", how="left").drop(["_rank", "_bucket"])


def enrich_fvg_events_with_delta_dominance(
    events: pl.DataFrame,
    volume_delta_5s: pl.DataFrame,
    quantile_count: int = 4,
) -> pl.DataFrame:
    if quantile_count < 1:
        raise ValueError(f"quantile_count must be >= 1, got {quantile_count}")

    event_missing = _missing_columns(events, REQUIRED_EVENT_COLUMNS)
    if event_missing:
        raise ValueError(f"Missing FVG event columns: {event_missing}")

    delta_missing = _missing_columns(volume_delta_5s, REQUIRED_DELTA_COLUMNS)
    if delta_missing:
        raise ValueError(f"Missing volume-delta columns: {delta_missing}")

    if events.is_empty():
        return _with_empty_delta_columns(events)

    events = events.drop([column for column in DELTA_DOMINANCE_COLUMNS if column in events.columns])

    delta = volume_delta_5s.select(
        pl.col("trade_date_et").alias("date"),
        pl.col("macro_bucket_index").cast(pl.Int64).alias("fvg_delta_bucket_index"),
        pl.col("volume_delta").cast(pl.Int64).alias("fvg_delta_volume_delta"),
        pl.col("delta_imbalance").cast(pl.Float64).alias("fvg_delta_imbalance"),
        pl.col("tick_delta").cast(pl.Int64).alias("fvg_delta_tick_delta"),
        pl.col("classified_share").cast(pl.Float64).alias("fvg_delta_classified_share"),
        pl.col("total_size").cast(pl.Int64).alias("fvg_delta_total_size"),
        pl.col("is_empty").cast(pl.Boolean).alias("fvg_delta_is_empty"),
    )

    enriched = (
        _with_confirmation_bucket(events)
        .join(delta, on=["date", "fvg_delta_bucket_index"], how="left")
        .with_columns(
            aligned_delta_imbalance=pl.when(pl.col("fvg_side") == "bearish")
            .then(-pl.col("fvg_delta_imbalance"))
            .otherwise(pl.col("fvg_delta_imbalance")),
            abs_delta_imbalance=pl.col("fvg_delta_imbalance").abs(),
            aligned_volume_delta=pl.when(pl.col("fvg_side") == "bearish")
            .then(-pl.col("fvg_delta_volume_delta"))
            .otherwise(pl.col("fvg_delta_volume_delta")),
            abs_volume_delta=pl.col("fvg_delta_volume_delta").abs(),
            aligned_tick_delta=pl.when(pl.col("fvg_side") == "bearish")
            .then(-pl.col("fvg_delta_tick_delta"))
            .otherwise(pl.col("fvg_delta_tick_delta")),
            abs_tick_delta=pl.col("fvg_delta_tick_delta").abs(),
        )
    )
    enriched = _with_rank_quantile(
        enriched,
        "aligned_delta_imbalance",
        "aligned_delta_imbalance_quantile",
        quantile_count,
    )
    return _with_rank_quantile(
        enriched,
        "abs_delta_imbalance",
        "abs_delta_imbalance_quantile",
        quantile_count,
    )


def try_enrich_fvg_events_with_delta_dominance(
    events: pl.DataFrame,
    volume_delta_path: str | Path = OUTPUT_MACRO_5S_PATH,
    quantile_count: int = 4,
) -> pl.DataFrame:
    delta_path = Path(volume_delta_path)
    if not delta_path.exists():
        warnings.warn(
            f"Skipping macro FVG volume-delta dominance: missing {delta_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _with_empty_delta_columns(events)
    return enrich_fvg_events_with_delta_dominance(
        events,
        load_macro_volume_delta_5s(delta_path),
        quantile_count=quantile_count,
    )
