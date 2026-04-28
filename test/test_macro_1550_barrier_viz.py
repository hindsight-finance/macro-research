from pathlib import Path

import polars as pl

from viz.macro_1550_barrier_viz import process_dataset, summarize_barrier_dataset

from features.macro_1550_barrier import build_macro_1550_barrier_study


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


def test_process_dataset_writes_barrier_figures_and_csvs(tmp_path: Path):
    study = build_macro_1550_barrier_study(_timing_frame())
    path = tmp_path / "nq_macro_1550_barrier.parquet"
    out_dir = tmp_path / "figs"
    study.write_parquet(path)

    summary = process_dataset(path, out_dir)

    assert summary.total_days == 2
    assert (out_dir / "nq_macro_1550_barrier_summary.csv").exists()
    assert (out_dir / "nq_macro_1550_barrier_edge_cases.csv").exists()
    assert (out_dir / "nq_macro_1550_barrier_rates.png").exists()
    assert (out_dir / "nq_macro_1550_barrier_edge_depth.png").exists()


def test_summarize_barrier_dataset_exposes_edge_cases(tmp_path: Path):
    study = build_macro_1550_barrier_study(_timing_frame())
    path = tmp_path / "study.parquet"
    study.write_parquet(path)

    summary = summarize_barrier_dataset(path)

    assert summary.edge_cases.height == 1
    assert summary.edge_cases.item(0, "macro_low_minute_index") == 55
