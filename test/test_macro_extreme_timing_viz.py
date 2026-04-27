from pathlib import Path

import polars as pl

from viz.macro_extreme_timing_viz import (
    MacroExtremeTimingVizSummary,
    summarize_macro_extreme_timing_dataset,
    process_dataset,
)


def _timing_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
            "datetime_utc": [
                "2025-01-02T20:50:00Z",
                "2025-01-03T20:50:00Z",
                "2025-01-04T20:50:00Z",
                "2025-01-05T20:50:00Z",
            ],
            "macro_minute_index": [50, 50, 50, 54],
            "candle_open": [100.0, 100.0, 100.0, 100.0],
            "candle_high": [101.0, 102.0, 103.0, 104.0],
            "candle_low": [99.0, 98.0, 97.0, 96.0],
            "candle_close": [100.5, 101.0, 99.0, 98.0],
            "candle_volume": [1, 1, 1, 1],
            "candle_dir_points": [0.5, 1.0, -1.0, -2.0],
            "candle_dir_sign": [1, 1, -1, -1],
            "candle_trend_state": ["bullish", "bullish", "bearish", "bearish"],
            "candle_high_time": [10, 30, 50, 20],
            "candle_low_time": [5, 20, 40, 10],
            "candle_high_ts_utc": [
                "2025-01-02T20:50:10Z",
                "2025-01-03T20:50:30Z",
                "2025-01-04T20:50:50Z",
                "2025-01-05T20:54:20Z",
            ],
            "candle_low_ts_utc": [
                "2025-01-02T20:50:05Z",
                "2025-01-03T20:50:20Z",
                "2025-01-04T20:50:40Z",
                "2025-01-05T20:54:10Z",
            ],
            "candle_high_first": [False, False, False, False],
            "candle_extreme_gap_seconds": [5, 10, 10, 10],
            "macro_open": [100.0, 100.0, 100.0, 100.0],
            "macro_close": [101.0, 102.0, 99.0, 98.0],
            "macro_dir_points": [1.0, 2.0, -1.0, -2.0],
            "macro_dir_sign": [1, 1, -1, -1],
            "macro_trend_state": ["bullish", "bullish", "bearish", "bearish"],
        }
    ).with_columns(
        pl.col("date").str.to_date(),
        pl.col("datetime_utc").str.to_datetime(time_zone="UTC"),
        pl.col("candle_high_ts_utc").str.to_datetime(time_zone="UTC"),
        pl.col("candle_low_ts_utc").str.to_datetime(time_zone="UTC"),
    )


def test_summarize_macro_extreme_timing_dataset_builds_frequency_and_quantiles(tmp_path: Path):
    path = tmp_path / "nq_macro_extreme_timing.parquet"
    _timing_frame().write_parquet(path)

    summary = summarize_macro_extreme_timing_dataset(path)

    assert isinstance(summary, MacroExtremeTimingVizSummary)
    assert summary.total_observations == 4
    assert summary.total_days == 4
    assert set(summary.frequency.select("extreme").to_series().unique().to_list()) == {"high", "low"}
    bull_high = summary.quantiles.filter(
        (pl.col("macro_trend_state") == "bullish")
        & (pl.col("macro_minute_index") == 50)
        & (pl.col("extreme") == "high")
    ).row(0, named=True)
    assert bull_high["sample_size"] == 2
    assert bull_high["median_time"] == 20.0


def test_process_dataset_writes_csvs_and_pngs(tmp_path: Path):
    path = tmp_path / "nq_macro_extreme_timing.parquet"
    out_dir = tmp_path / "figs"
    _timing_frame().write_parquet(path)

    summary = process_dataset(path, out_dir)

    assert (out_dir / "nq_macro_extreme_timing_frequency.csv").exists()
    assert (out_dir / "nq_macro_extreme_timing_quantiles.csv").exists()
    assert (out_dir / "nq_macro_extreme_timing_heatmap.png").exists()
    assert (out_dir / "nq_macro_extreme_timing_histograms.png").exists()
    assert pl.read_csv(out_dir / "nq_macro_extreme_timing_quantiles.csv").height == summary.quantiles.height
