#!/usr/bin/env python3
"""Study whether the first 10 seconds of 15:50 define a bullish macro low barrier."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

INPUT_PATH = Path("outputs/nq_macro_extreme_timing.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_1550_barrier.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_1550_barrier_summary.parquet")
DEFAULT_BARRIER_SECONDS = 10

MACRO_1550_BARRIER_COLUMNS = [
    "date",
    "macro_trend_state",
    "macro_open",
    "macro_close",
    "macro_dir_points",
    "barrier_extreme",
    "barrier_price",
    "barrier_time",
    "barrier_first10",
    "opposite_1550_price",
    "opposite_1550_time",
    "low_1550",
    "low_1550_time",
    "low_1550_first10",
    "high_1550",
    "high_1550_time",
    "high_1550_first10",
    "open_1550",
    "close_1550",
    "dir_1550_points",
    "range_1550_points",
    "macro_extreme",
    "macro_extreme_minute_index",
    "macro_extreme_time",
    "barrier_is_macro_extreme",
    "barrier_holds",
    "edge_case",
    "macro_extreme_after_1550_points",
    "macro_extreme_after_1550_minutes",
    # Back-compat bullish low names.
    "macro_low",
    "macro_low_minute_index",
    "macro_low_time",
    "low_1550_is_macro_low",
    "bullish_edge_case",
    "macro_low_after_1550_points",
    "macro_low_after_1550_minutes",
]

MACRO_1550_BARRIER_SUMMARY_COLUMNS = [
    "scope",
    "macro_trend_state",
    "barrier_extreme",
    "sample_size",
    "barrier_first10_pct",
    "barrier_is_macro_extreme_pct",
    "barrier_holds_pct",
    "edge_case_pct",
    "avg_macro_extreme_after_1550_points",
    "median_macro_extreme_after_1550_points",
    # Back-compat bullish low names.
    "low_1550_first10_pct",
    "low_1550_is_macro_low_pct",
    "avg_macro_low_after_1550_points",
    "median_macro_low_after_1550_points",
]


def _require_columns(df: pl.DataFrame) -> None:
    required = {
        "date",
        "macro_trend_state",
        "macro_minute_index",
        "candle_open",
        "candle_high",
        "candle_low",
        "candle_close",
        "candle_high_time",
        "candle_low_time",
        "macro_open",
        "macro_close",
        "macro_dir_points",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_macro_1550_barrier_study(df: pl.DataFrame, barrier_seconds: int = DEFAULT_BARRIER_SECONDS) -> pl.DataFrame:
    """Return one row per bullish/bearish macro date with 15:50 directional barrier diagnostics."""
    if barrier_seconds <= 0 or barrier_seconds > 60:
        raise ValueError(f"barrier_seconds must be 1..60, got {barrier_seconds}")
    _require_columns(df)

    directional = df.filter(pl.col("macro_trend_state").is_in(["bullish", "bearish"]))
    if directional.is_empty():
        return pl.DataFrame(schema={col: pl.Null for col in MACRO_1550_BARRIER_COLUMNS})

    lows = directional.sort("date", "candle_low", "macro_minute_index", "candle_low_time").group_by("date", maintain_order=True).first().select(
        "date",
        pl.col("candle_low").alias("macro_low"),
        pl.col("macro_minute_index").alias("macro_low_minute_index"),
        pl.col("candle_low_time").alias("macro_low_time"),
    )
    highs = directional.sort(["date", "candle_high", "macro_minute_index", "candle_high_time"], descending=[False, True, False, False]).group_by("date", maintain_order=True).first().select(
        "date",
        pl.col("candle_high").alias("macro_high"),
        pl.col("macro_minute_index").alias("macro_high_minute_index"),
        pl.col("candle_high_time").alias("macro_high_time"),
    )

    base_1550 = directional.filter(pl.col("macro_minute_index") == 50).select(
        "date",
        "macro_trend_state",
        "macro_open",
        "macro_close",
        "macro_dir_points",
        pl.col("candle_low").alias("low_1550"),
        pl.col("candle_low_time").alias("low_1550_time"),
        pl.col("candle_high").alias("high_1550"),
        pl.col("candle_high_time").alias("high_1550_time"),
        pl.col("candle_open").alias("open_1550"),
        pl.col("candle_close").alias("close_1550"),
    )

    return (
        base_1550.join(lows, on="date", how="inner")
        .join(highs, on="date", how="inner")
        .with_columns(
            dir_1550_points=pl.col("close_1550") - pl.col("open_1550"),
            range_1550_points=pl.col("high_1550") - pl.col("low_1550"),
            low_1550_first10=pl.col("low_1550_time") < barrier_seconds,
            high_1550_first10=pl.col("high_1550_time") < barrier_seconds,
            barrier_extreme=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.lit("low")).otherwise(pl.lit("high")),
        )
        .with_columns(
            barrier_price=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("low_1550")).otherwise(pl.col("high_1550")),
            barrier_time=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("low_1550_time")).otherwise(pl.col("high_1550_time")),
            barrier_first10=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("low_1550_first10")).otherwise(pl.col("high_1550_first10")),
            opposite_1550_price=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("high_1550")).otherwise(pl.col("low_1550")),
            opposite_1550_time=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("high_1550_time")).otherwise(pl.col("low_1550_time")),
            macro_extreme=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("macro_low")).otherwise(pl.col("macro_high")),
            macro_extreme_minute_index=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("macro_low_minute_index")).otherwise(pl.col("macro_high_minute_index")),
            macro_extreme_time=pl.when(pl.col("macro_trend_state") == "bullish").then(pl.col("macro_low_time")).otherwise(pl.col("macro_high_time")),
            low_1550_is_macro_low=pl.col("low_1550") == pl.col("macro_low"),
        )
        .with_columns(
            barrier_is_macro_extreme=pl.when(pl.col("macro_trend_state") == "bullish")
            .then(pl.col("low_1550") == pl.col("macro_low"))
            .otherwise(pl.col("high_1550") == pl.col("macro_high")),
        )
        .with_columns(
            barrier_holds=pl.col("barrier_first10") & pl.col("barrier_is_macro_extreme"),
            edge_case=~pl.col("barrier_is_macro_extreme"),
            bullish_edge_case=pl.when(pl.col("macro_trend_state") == "bullish").then(~pl.col("low_1550_is_macro_low")).otherwise(False),
            macro_low_after_1550_points=(pl.col("low_1550") - pl.col("macro_low")).clip(0.0),
            macro_extreme_after_1550_points=pl.when(pl.col("macro_trend_state") == "bullish")
            .then((pl.col("low_1550") - pl.col("macro_low")).clip(0.0))
            .otherwise((pl.col("macro_high") - pl.col("high_1550")).clip(0.0)),
            macro_low_after_1550_minutes=pl.when(pl.col("low_1550") == pl.col("macro_low")).then(None).otherwise(pl.col("macro_low_minute_index") - 50),
            macro_extreme_after_1550_minutes=pl.when(pl.col("barrier_is_macro_extreme"))
            .then(None)
            .otherwise(pl.col("macro_extreme_minute_index") - 50),
        )
        .select(MACRO_1550_BARRIER_COLUMNS)
        .sort("date")
    )

def _summary_row(df: pl.DataFrame, scope: str) -> dict:
    if df.is_empty():
        return {
            "scope": scope,
            "macro_trend_state": None,
            "barrier_extreme": None,
            "sample_size": 0,
            "barrier_first10_pct": None,
            "barrier_is_macro_extreme_pct": None,
            "barrier_holds_pct": None,
            "edge_case_pct": None,
            "avg_macro_extreme_after_1550_points": None,
            "median_macro_extreme_after_1550_points": None,
            "low_1550_first10_pct": None,
            "low_1550_is_macro_low_pct": None,
            "avg_macro_low_after_1550_points": None,
            "median_macro_low_after_1550_points": None,
        }
    metrics = df.select(
        pl.col("macro_trend_state").first().alias("macro_trend_state"),
        pl.col("barrier_extreme").first().alias("barrier_extreme"),
        pl.len().alias("sample_size"),
        (pl.col("barrier_first10").mean() * 100.0).alias("barrier_first10_pct"),
        (pl.col("barrier_is_macro_extreme").mean() * 100.0).alias("barrier_is_macro_extreme_pct"),
        (pl.col("barrier_holds").mean() * 100.0).alias("barrier_holds_pct"),
        (pl.col("edge_case").mean() * 100.0).alias("edge_case_pct"),
        pl.col("macro_extreme_after_1550_points").filter(pl.col("edge_case")).mean().alias("avg_macro_extreme_after_1550_points"),
        pl.col("macro_extreme_after_1550_points").filter(pl.col("edge_case")).median().alias("median_macro_extreme_after_1550_points"),
        (pl.col("low_1550_first10").mean() * 100.0).alias("low_1550_first10_pct"),
        (pl.col("low_1550_is_macro_low").mean() * 100.0).alias("low_1550_is_macro_low_pct"),
        pl.col("macro_low_after_1550_points").filter(pl.col("bullish_edge_case")).mean().alias("avg_macro_low_after_1550_points"),
        pl.col("macro_low_after_1550_points").filter(pl.col("bullish_edge_case")).median().alias("median_macro_low_after_1550_points"),
    ).row(0, named=True)
    return {"scope": scope, **metrics}

def summarize_macro_1550_barrier_study(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize 15:50 directional barrier hold/fail rates for bullish and bearish macros."""
    missing = sorted(set(MACRO_1550_BARRIER_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for trend in ["bullish", "bearish"]:
        trend_df = df.filter(pl.col("macro_trend_state") == trend)
        rows.extend(
            [
                _summary_row(trend_df, f"{trend}_macro"),
                _summary_row(trend_df.filter(pl.col("barrier_first10")), f"{trend}_macro_barrier_first10"),
                _summary_row(trend_df.filter(~pl.col("barrier_first10")), f"{trend}_macro_barrier_after10"),
            ]
        )
    return pl.DataFrame(rows).select(MACRO_1550_BARRIER_SUMMARY_COLUMNS)

def write_macro_1550_barrier_study(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
    barrier_seconds: int = DEFAULT_BARRIER_SECONDS,
) -> tuple[Path, Path]:
    timing = pl.read_parquet(input_path)
    study = build_macro_1550_barrier_study(timing, barrier_seconds=barrier_seconds)
    summary = summarize_macro_1550_barrier_study(study)

    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary = write_macro_1550_barrier_study()
    print(f"[OK] Wrote macro 15:50 barrier study → {output}")
    print(f"[OK] Wrote macro 15:50 barrier summary → {summary}")


if __name__ == "__main__":
    main()
