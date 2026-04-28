from pathlib import Path

import polars as pl

from features.macro_1550_barrier import (
    MACRO_1550_BARRIER_COLUMNS,
    MACRO_1550_BARRIER_SUMMARY_COLUMNS,
    build_macro_1550_barrier_study,
    summarize_macro_1550_barrier_study,
    write_macro_1550_barrier_study,
)


def _timing_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02"] * 4 + ["2025-01-03"] * 4,
            "datetime_utc": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:54:00Z",
                "2025-01-02T20:55:00Z",
                "2025-01-02T20:59:00Z",
                "2025-01-03T20:50:00Z",
                "2025-01-03T20:54:00Z",
                "2025-01-03T20:55:00Z",
                "2025-01-03T20:59:00Z",
            ],
            "macro_minute_index": [50, 54, 55, 59, 50, 54, 55, 59],
            "candle_open": [100, 104, 105, 106, 100, 103, 101, 102],
            "candle_high": [105, 106, 107, 110, 104, 105, 103, 108],
            "candle_low": [99, 103, 104, 106, 99, 98, 97, 101],
            "candle_close": [104, 105, 106, 109, 103, 101, 102, 106],
            "candle_volume": [10] * 8,
            "candle_dir_points": [4, 1, 1, 3, 3, -2, 1, 4],
            "candle_dir_sign": [1, 1, 1, 1, 1, -1, 1, 1],
            "candle_trend_state": ["bullish", "bullish", "bullish", "bullish", "bullish", "bearish", "bullish", "bullish"],
            "candle_high_time": [40, 50, 50, 58, 35, 10, 45, 59],
            "candle_low_time": [5, 20, 20, 20, 5, 30, 40, 10],
            "candle_high_ts_utc": [
                "2025-01-02T20:50:40Z",
                "2025-01-02T20:54:50Z",
                "2025-01-02T20:55:50Z",
                "2025-01-02T20:59:58Z",
                "2025-01-03T20:50:35Z",
                "2025-01-03T20:54:10Z",
                "2025-01-03T20:55:45Z",
                "2025-01-03T20:59:59Z",
            ],
            "candle_low_ts_utc": [
                "2025-01-02T20:50:05Z",
                "2025-01-02T20:54:20Z",
                "2025-01-02T20:55:20Z",
                "2025-01-02T20:59:20Z",
                "2025-01-03T20:50:05Z",
                "2025-01-03T20:54:30Z",
                "2025-01-03T20:55:40Z",
                "2025-01-03T20:59:10Z",
            ],
            "candle_high_first": [False, False, False, False, False, True, False, False],
            "candle_extreme_gap_seconds": [35, 30, 30, 38, 30, 20, 5, 49],
            "macro_open": [100] * 8,
            "macro_close": [109] * 4 + [106] * 4,
            "macro_dir_points": [9] * 4 + [6] * 4,
            "macro_dir_sign": [1] * 8,
            "macro_trend_state": ["bullish"] * 8,
        }
    ).with_columns(
        pl.col("date").str.to_date(),
        pl.col("datetime_utc").str.to_datetime(time_zone="UTC"),
        pl.col("candle_high_ts_utc").str.to_datetime(time_zone="UTC"),
        pl.col("candle_low_ts_utc").str.to_datetime(time_zone="UTC"),
    )


def test_build_macro_1550_barrier_study_marks_early_barrier_and_edge_cases():
    out = build_macro_1550_barrier_study(_timing_frame(), barrier_seconds=10)

    assert out.columns == MACRO_1550_BARRIER_COLUMNS
    assert out.height == 2

    held = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    assert held["low_1550_first10"] is True
    assert held["low_1550_is_macro_low"] is True
    assert held["barrier_holds"] is True
    assert held["bullish_edge_case"] is False
    assert held["macro_low_minute_index"] == 50

    failed = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)
    assert failed["low_1550_first10"] is True
    assert failed["low_1550_is_macro_low"] is False
    assert failed["barrier_holds"] is False
    assert failed["bullish_edge_case"] is True
    assert failed["macro_low_minute_index"] == 55
    assert failed["macro_low_after_1550_points"] == 2.0


def test_summarize_macro_1550_barrier_study_reports_conditional_rates():
    study = build_macro_1550_barrier_study(_timing_frame(), barrier_seconds=10)

    summary = summarize_macro_1550_barrier_study(study)

    assert summary.columns == MACRO_1550_BARRIER_SUMMARY_COLUMNS
    overall = summary.filter(pl.col("scope") == "bullish_macro").row(0, named=True)
    assert overall["sample_size"] == 2
    assert overall["low_1550_first10_pct"] == 100.0
    assert overall["low_1550_is_macro_low_pct"] == 50.0
    assert overall["barrier_holds_pct"] == 50.0


def test_write_macro_1550_barrier_study_writes_outputs(tmp_path: Path):
    input_path = tmp_path / "timing.parquet"
    output_path = tmp_path / "barrier.parquet"
    summary_path = tmp_path / "summary.parquet"
    _timing_frame().write_parquet(input_path)

    wrote = write_macro_1550_barrier_study(input_path, output_path, summary_path)

    assert wrote == (output_path, summary_path)
    assert pl.read_parquet(output_path).columns == MACRO_1550_BARRIER_COLUMNS
    assert pl.read_parquet(summary_path).columns == MACRO_1550_BARRIER_SUMMARY_COLUMNS
