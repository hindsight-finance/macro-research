"""Descriptive 1H PM FVG/imbalance context joined to closing macro outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl

from utils.minute_bars import build_market_time_columns, load_minute_bars, normalize_minute_bars

DEFAULT_MINUTE_INPUT = Path("outputs/nq_1m.parquet")
DEFAULT_MACRO_INPUT = Path("outputs/nq_macro_outcomes.parquet")
DEFAULT_CONTEXT_OUTPUT = Path("outputs/nq_pm_1h_macro_context.parquet")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/nq_pm_1h_macro_summary.csv")
REQUIRED_MINUTE_COLUMNS = {"datetime_utc", "Open", "High", "Low", "Close", "Volume"}
REQUIRED_MACRO_COLUMNS = {"date", "macro_dir_points", "macro_range_points"}
CONTEXT_HOURS = (12, 13, 14)


def _require_columns(df: pl.DataFrame, required: set[str], frame_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {sorted(missing)}")


def _safe_div(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return pl.when(denom != 0).then(numer / denom).otherwise(None)


def _build_hourly_bars(minute_bars: pl.DataFrame) -> pl.DataFrame:
    normalized = normalize_minute_bars(minute_bars)
    _require_columns(normalized, REQUIRED_MINUTE_COLUMNS, "minute_bars")

    work = (
        build_market_time_columns(normalized)
        .with_columns(hour_et=pl.col("datetime_et").dt.hour())
        .filter(pl.col("hour_et").is_in(CONTEXT_HOURS))
        .sort("datetime_utc")
    )
    if work.is_empty():
        raise ValueError("No 12:00-15:00 ET rows found in minute bars")

    hourly = (
        work.group_by("date_et", "hour_et", maintain_order=True)
        .agg(
            pl.col("Open").first().alias("open"),
            pl.col("High").max().alias("high"),
            pl.col("Low").min().alias("low"),
            pl.col("Close").last().alias("close"),
            pl.col("Volume").sum().alias("volume"),
            pl.len().alias("n_minutes"),
        )
        .rename({"date_et": "date"})
        .filter(pl.col("n_minutes") >= 60)
        .sort("date", "hour_et")
    )
    if hourly.is_empty():
        raise ValueError("No complete 12:00-15:00 ET hourly context rows found")
    return hourly


def _hour_frame(hourly: pl.DataFrame, hour: int, prefix: str) -> pl.DataFrame:
    return hourly.filter(pl.col("hour_et") == hour).select(
        "date",
        pl.col("open").alias(f"{prefix}_open"),
        pl.col("high").alias(f"{prefix}_high"),
        pl.col("low").alias(f"{prefix}_low"),
        pl.col("close").alias(f"{prefix}_close"),
        pl.col("volume").alias(f"{prefix}_volume"),
        pl.col("n_minutes").alias(f"{prefix}_n_minutes"),
    )


def build_hourly_context(minute_bars: pl.DataFrame) -> pl.DataFrame:
    """Return one row per date with no-leak 1H FVG and 13:00-15:00 imbalance context."""
    hourly = _build_hourly_bars(minute_bars)
    context = _hour_frame(hourly, 12, "h12")
    for hour, prefix in ((13, "h13"), (14, "h14")):
        context = context.join(_hour_frame(hourly, hour, prefix), on="date", how="inner")

    if context.is_empty():
        raise ValueError("No dates have complete 12:00, 13:00, and 14:00 ET hourly candles")

    pm_range = pl.col("pm_13_15_high") - pl.col("pm_13_15_low")
    pm_dir = pl.col("pm_13_15_close") - pl.col("pm_13_15_open")
    fvg_size = (
        pl.when(pl.col("h12_high") < pl.col("h14_low"))
        .then(pl.col("h14_low") - pl.col("h12_high"))
        .when(pl.col("h12_low") > pl.col("h14_high"))
        .then(pl.col("h12_low") - pl.col("h14_high"))
        .otherwise(0.0)
    )

    return (
        context.with_columns(
            pm_13_15_open=pl.col("h13_open"),
            pm_13_15_close=pl.col("h14_close"),
            pm_13_15_high=pl.max_horizontal("h13_high", "h14_high"),
            pm_13_15_low=pl.min_horizontal("h13_low", "h14_low"),
            pm_13_15_volume=pl.col("h13_volume") + pl.col("h14_volume"),
        )
        .with_columns(
            pm_13_15_range=pm_range,
            pm_13_15_dir_points=pm_dir,
            pm_13_15_dir_sign=pm_dir.sign().fill_null(0).cast(pl.Int8),
            pm_13_15_close_pos=_safe_div(pl.col("pm_13_15_close") - pl.col("pm_13_15_low"), pm_range),
            pm_13_15_body_to_range=_safe_div(pm_dir.abs(), pm_range),
            pm_13_15_upper_wick_share=_safe_div(
                pl.col("pm_13_15_high") - pl.max_horizontal("pm_13_15_open", "pm_13_15_close"), pm_range
            ),
            pm_13_15_lower_wick_share=_safe_div(
                pl.min_horizontal("pm_13_15_open", "pm_13_15_close") - pl.col("pm_13_15_low"), pm_range
            ),
            fvg_direction=pl.when(pl.col("h12_high") < pl.col("h14_low"))
            .then(pl.lit("bullish"))
            .when(pl.col("h12_low") > pl.col("h14_high"))
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("none")),
            fvg_size_points=fvg_size,
        )
        .with_columns(
            has_fvg=pl.col("fvg_direction") != "none",
            fvg_size_pct_of_pm_range=_safe_div(pl.col("fvg_size_points"), pl.col("pm_13_15_range")),
            imbalance_direction=pl.when((pl.col("pm_13_15_dir_points") > 0) & (pl.col("pm_13_15_close_pos") >= 0.60))
            .then(pl.lit("bullish"))
            .when((pl.col("pm_13_15_dir_points") < 0) & (pl.col("pm_13_15_close_pos") <= 0.40))
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("neutral")),
        )
        .sort("date")
    )


def join_macro_outcomes(context: pl.DataFrame, macro: pl.DataFrame, include_flat_macro: bool = False) -> pl.DataFrame:
    """Join context rows to macro outcomes and add macro direction labels."""
    _require_columns(context, {"date"}, "context")
    _require_columns(macro, REQUIRED_MACRO_COLUMNS, "macro")
    macro_keep = [
        column
        for column in [
            "date",
            "macro_open",
            "macro_close",
            "macro_high",
            "macro_low",
            "macro_dir_points",
            "macro_range_points",
            "macro_dir_pct",
            "macro_range_pct",
            "close_in_range",
            "macro_high_time",
            "macro_low_time",
        ]
        if column in macro.columns
    ]
    joined = context.join(macro.select(macro_keep), on="date", how="inner")
    if joined.is_empty():
        raise ValueError("Macro join produced zero rows")

    joined = joined.with_columns(
        macro_dir_sign=pl.col("macro_dir_points").sign().fill_null(0).cast(pl.Int8),
        macro_direction=pl.when(pl.col("macro_dir_points") > 0)
        .then(pl.lit("bullish"))
        .when(pl.col("macro_dir_points") < 0)
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("flat")),
    )
    return joined if include_flat_macro else joined.filter(pl.col("macro_dir_sign") != 0)


def _assign_fvg_size_bucket(df: pl.DataFrame) -> pl.DataFrame:
    if "fvg_size_points" not in df.columns:
        raise ValueError("summary input missing required column: fvg_size_points")
    nonzero = df.filter(pl.col("fvg_size_points") > 0).select("fvg_size_points")
    if nonzero.height < 3:
        return df.with_columns(
            fvg_size_bucket=pl.when(pl.col("fvg_size_points") > 0).then(pl.lit("has_fvg")).otherwise(pl.lit("none"))
        )
    q1 = nonzero.select(pl.col("fvg_size_points").quantile(1 / 3, interpolation="nearest")).item()
    q2 = nonzero.select(pl.col("fvg_size_points").quantile(2 / 3, interpolation="nearest")).item()
    return df.with_columns(
        fvg_size_bucket=pl.when(pl.col("fvg_size_points") <= 0)
        .then(pl.lit("none"))
        .when(pl.col("fvg_size_points") <= q1)
        .then(pl.lit("small"))
        .when(pl.col("fvg_size_points") <= q2)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
    )


def _summarize_bucket(df: pl.DataFrame, bucket_col: str, cohort_name: str) -> pl.DataFrame:
    return (
        df.group_by(bucket_col)
        .agg(
            pl.len().alias("n"),
            (pl.col("macro_dir_sign") > 0).sum().alias("macro_bull_n"),
            (pl.col("macro_dir_sign") < 0).sum().alias("macro_bear_n"),
            pl.col("macro_dir_points").mean().alias("avg_macro_dir_points"),
            pl.col("macro_dir_points").median().alias("median_macro_dir_points"),
            pl.col("macro_range_points").mean().alias("avg_macro_range_points"),
            pl.col("macro_range_points").median().alias("median_macro_range_points"),
        )
        .with_columns(
            cohort=pl.lit(cohort_name),
            bucket=pl.col(bucket_col).cast(pl.String),
            macro_bull_rate=pl.col("macro_bull_n") / pl.col("n"),
            macro_bear_rate=pl.col("macro_bear_n") / pl.col("n"),
        )
        .select(
            "cohort",
            "bucket",
            "n",
            "macro_bull_n",
            "macro_bear_n",
            "macro_bull_rate",
            "macro_bear_rate",
            "avg_macro_dir_points",
            "median_macro_dir_points",
            "avg_macro_range_points",
            "median_macro_range_points",
        )
        .sort("cohort", "bucket")
    )


def build_summary(joined: pl.DataFrame, include_flat_macro: bool = False) -> pl.DataFrame:
    """Build descriptive cohort summary tables for context vs macro direction."""
    required = {"fvg_direction", "imbalance_direction", "fvg_size_points", "macro_dir_sign", "macro_dir_points", "macro_range_points"}
    _require_columns(joined, required, "joined")
    work = joined if include_flat_macro else joined.filter(pl.col("macro_dir_sign") != 0)
    if work.is_empty():
        raise ValueError("No non-flat macro rows available for summary")

    work = _assign_fvg_size_bucket(work).with_columns(
        fvg_x_imbalance=pl.concat_str([pl.col("fvg_direction"), pl.lit("|"), pl.col("imbalance_direction")])
    )
    frames = [
        _summarize_bucket(work, "fvg_direction", "fvg_direction"),
        _summarize_bucket(work, "imbalance_direction", "imbalance_direction"),
        _summarize_bucket(work, "fvg_x_imbalance", "fvg_x_imbalance"),
        _summarize_bucket(work, "fvg_size_bucket", "fvg_size_bucket"),
    ]
    return pl.concat(frames, how="vertical").sort("cohort", "bucket")


def run_pm_1h_macro_context(
    minute_input: str | Path = DEFAULT_MINUTE_INPUT,
    macro_input: str | Path = DEFAULT_MACRO_INPUT,
    context_output: str | Path = DEFAULT_CONTEXT_OUTPUT,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    include_flat_macro: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    minute_bars = load_minute_bars(minute_input)
    macro = pl.read_parquet(macro_input)
    context = build_hourly_context(minute_bars)
    joined = join_macro_outcomes(context, macro, include_flat_macro=include_flat_macro)
    summary = build_summary(joined, include_flat_macro=include_flat_macro)

    context_path = Path(context_output)
    summary_path = Path(summary_output)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(context_path)
    summary.write_csv(summary_path)
    return joined, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build descriptive 1H PM FVG/imbalance cohorts vs closing macro direction.")
    parser.add_argument("--minute-input", default=str(DEFAULT_MINUTE_INPUT))
    parser.add_argument("--macro-input", default=str(DEFAULT_MACRO_INPUT))
    parser.add_argument("--context-output", default=str(DEFAULT_CONTEXT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--include-flat-macro", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    joined, summary = run_pm_1h_macro_context(
        minute_input=args.minute_input,
        macro_input=args.macro_input,
        context_output=args.context_output,
        summary_output=args.summary_output,
        include_flat_macro=args.include_flat_macro,
    )
    print(f"[OK] Wrote context rows={joined.height} → {args.context_output}")
    print(f"[OK] Wrote summary rows={summary.height} → {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
