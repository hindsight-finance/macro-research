from pathlib import Path

import polars as pl

from features.macro_extreme_timing import (
    MACRO_EXTREME_TIMING_COLUMNS,
    MACRO_EXTREME_TIMING_SUMMARY_COLUMNS,
    build_macro_extreme_timing,
    summarize_macro_extreme_timing,
    write_macro_extreme_timing,
)


def _write_ticks(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC"))
    ).write_parquet(path)


def test_build_macro_extreme_timing_uses_first_touch_seconds_and_macro_direction(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:10Z",
                "2025-01-02T20:50:20Z",
                "2025-01-02T20:50:40Z",
                "2025-01-02T20:50:50Z",
                "2025-01-02T20:54:02Z",
                "2025-01-02T20:54:12Z",
                "2025-01-02T20:54:42Z",
                "2025-01-02T20:55:03Z",
                "2025-01-02T20:55:08Z",
                "2025-01-02T20:55:30Z",
                "2025-01-02T20:59:04Z",
                "2025-01-02T20:59:09Z",
                "2025-01-02T20:59:45Z",
                "2025-01-02T20:59:50Z",
            ],
            "intra_ts_rank": [0] * 15,
            "side": [2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
            "price_ticks": [
                400,
                396,
                408,
                408,
                404,
                404,
                400,
                412,
                412,
                404,
                416,
                416,
                408,
                420,
                424,
            ],
            "size": [1] * 15,
        },
    )

    out = build_macro_extreme_timing(path, key_minutes=(50, 54, 55, 59)).collect(engine="streaming")

    assert out.columns == MACRO_EXTREME_TIMING_COLUMNS
    assert out.height == 4
    assert out.select("macro_dir_sign").unique().item() == 1
    assert out.select("macro_trend_state").unique().item() == "bullish"

    row_1550 = out.filter(pl.col("macro_minute_index") == 50).row(0, named=True)
    assert row_1550["candle_open"] == 100.0
    assert row_1550["candle_high"] == 102.0
    assert row_1550["candle_low"] == 99.0
    assert row_1550["candle_close"] == 101.0
    assert row_1550["candle_high_time"] == 20
    assert row_1550["candle_low_time"] == 10
    assert row_1550["candle_high_first"] is False
    assert row_1550["candle_extreme_gap_seconds"] == 10

    row_1559 = out.filter(pl.col("macro_minute_index") == 59).row(0, named=True)
    assert row_1559["candle_high_time"] == 50
    assert row_1559["candle_low_time"] == 9
    assert row_1559["macro_close"] == 106.0


def test_build_macro_extreme_timing_handles_dst_and_requires_complete_macro(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-07-02T19:50:00Z",
                "2025-07-02T19:50:10Z",
                "2025-07-02T19:54:00Z",
                "2025-07-02T19:54:10Z",
                "2025-07-02T19:55:00Z",
                "2025-07-02T19:55:10Z",
                "2025-07-02T19:59:00Z",
                "2025-07-02T19:59:10Z",
                "2025-07-03T19:50:00Z",
                "2025-07-03T19:50:10Z",
            ],
            "intra_ts_rank": [0] * 10,
            "side": [2] * 10,
            "price_ticks": [400, 404, 404, 408, 408, 412, 412, 416, 500, 504],
            "size": [1] * 10,
        },
    )

    out = build_macro_extreme_timing(path).collect(engine="streaming")

    assert out.height == 4
    assert out.select("date").unique().item().isoformat() == "2025-07-02"
    assert out.select("datetime_utc").to_series().dt.hour().to_list() == [19, 19, 19, 19]


def test_summarize_macro_extreme_timing_builds_directional_quantiles():
    timing = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "macro_trend_state": ["bullish", "bullish", "bearish"],
            "macro_minute_index": [50, 50, 50],
            "candle_high_time": [10, 30, 50],
            "candle_low_time": [5, 20, 40],
            "candle_extreme_gap_seconds": [5, 10, 10],
            "candle_high_first": [False, False, False],
        }
    ).with_columns(pl.col("date").str.to_date())

    summary = summarize_macro_extreme_timing(timing)

    assert summary.columns == MACRO_EXTREME_TIMING_SUMMARY_COLUMNS
    bull_high = summary.filter(
        (pl.col("macro_trend_state") == "bullish")
        & (pl.col("macro_minute_index") == 50)
        & (pl.col("extreme") == "high")
    ).row(0, named=True)
    assert bull_high["sample_size"] == 2
    assert bull_high["mean_time"] == 20.0
    assert bull_high["median_time"] == 20.0
    assert bull_high["high_first_pct"] == 0.0


def test_write_macro_extreme_timing_writes_dataset_and_summary(tmp_path: Path):
    input_path = tmp_path / "ticks.parquet"
    output_path = tmp_path / "nq_macro_extreme_timing.parquet"
    summary_path = tmp_path / "nq_macro_extreme_timing_summary.parquet"
    _write_ticks(
        input_path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:10Z",
                "2025-01-02T20:54:00Z",
                "2025-01-02T20:54:10Z",
                "2025-01-02T20:55:00Z",
                "2025-01-02T20:55:10Z",
                "2025-01-02T20:59:00Z",
                "2025-01-02T20:59:10Z",
            ],
            "intra_ts_rank": [0] * 8,
            "side": [2] * 8,
            "price_ticks": [400, 404, 404, 408, 408, 412, 412, 416],
            "size": [1] * 8,
        },
    )

    wrote = write_macro_extreme_timing(input_path, output_path, summary_path)

    assert wrote == (output_path, summary_path)
    assert pl.read_parquet(output_path).columns == MACRO_EXTREME_TIMING_COLUMNS
    assert pl.read_parquet(summary_path).columns == MACRO_EXTREME_TIMING_SUMMARY_COLUMNS
