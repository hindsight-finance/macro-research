from pathlib import Path

import numpy as np
import pandas as pd

from viz.tick_density_viz import _plot_x_positions_and_labels, process_dataset, summarize_metric_by_index, summarize_tick_density_dataset



def test_plot_x_positions_and_labels_use_sequential_axis_for_wrapped_macro_minutes():
    positions, labels = _plot_x_positions_and_labels(pd.Series([40, 41, 59, 0, 10]))

    assert positions == [0, 1, 2, 3, 4]
    assert labels == ["40", "41", "59", "0", "10"]

def test_summarize_tick_density_dataset_builds_cross_day_band_stats(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                [
                    "2025-01-02T20:50:00Z",
                    "2025-01-02T20:51:00Z",
                    "2025-01-03T20:50:00Z",
                    "2025-01-03T20:51:00Z",
                ],
                utc=True,
            ),
            "date_utc": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "macro_minute_index": [40, 41, 40, 41],
            "tick_count": [10, 20, 30, 40],
            "total_size": [1, 1, 1, 1],
            "buy_ticks": [5, 10, 15, 20],
            "sell_ticks": [5, 10, 15, 20],
            "none_ticks": [0, 0, 0, 0],
        }
    ).to_parquet(path, index=False)

    summary = summarize_tick_density_dataset(path)

    assert summary.index_column == "macro_minute_index"
    focus = summary.band_stats[summary.band_stats["macro_minute_index"].isin([40, 41])].reset_index(drop=True)
    assert focus["mean_tick_count"].round(4).tolist() == [20.0, 30.0]
    assert focus["p25_tick_count"].round(4).tolist() == [15.0, 25.0]
    assert focus["p75_tick_count"].round(4).tolist() == [25.0, 35.0]



def test_summarize_metric_by_index_supports_total_size(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                [
                    "2025-01-02T20:50:00Z",
                    "2025-01-02T20:51:00Z",
                    "2025-01-03T20:50:00Z",
                    "2025-01-03T20:51:00Z",
                ],
                utc=True,
            ),
            "date_utc": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "macro_minute_index": [40, 41, 40, 41],
            "tick_count": [10, 20, 30, 40],
            "total_size": [11, 21, 31, 41],
            "buy_ticks": [5, 10, 15, 20],
            "sell_ticks": [5, 10, 15, 20],
            "none_ticks": [0, 0, 0, 0],
        }
    ).to_parquet(path, index=False)

    summary = summarize_tick_density_dataset(path)
    stats_df = summarize_metric_by_index(pd.read_parquet(path), summary.index_column, "total_size")

    focus = stats_df[stats_df["macro_minute_index"].isin([40, 41])].reset_index(drop=True)
    assert focus["mean_total_size"].round(4).tolist() == [21.0, 31.0]
    assert focus["p25_total_size"].round(4).tolist() == [16.0, 26.0]
    assert focus["p75_total_size"].round(4).tolist() == [26.0, 36.0]

def test_summarize_tick_density_dataset_flags_mixed_bucket_normality(tmp_path: Path):
    rng = np.random.default_rng(7)
    n = 80
    normal_bucket = rng.normal(loc=100.0, scale=8.0, size=n)
    skewed_bucket = rng.exponential(scale=18.0, size=n) + 80.0
    path = tmp_path / "sample_5s.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.date_range("2025-01-01", periods=n * 2, freq="5s", tz="UTC"),
            "date_utc": list(pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d")) * 2,
            "bucket_index": [0] * n + [1] * n,
            "is_empty": [False] * (n * 2),
            "tick_count": np.concatenate([normal_bucket, skewed_bucket]),
            "total_size": [1] * (n * 2),
            "buy_ticks": [1] * (n * 2),
            "sell_ticks": [0] * (n * 2),
            "none_ticks": [0] * (n * 2),
        }
    ).to_parquet(path, index=False)

    summary = summarize_tick_density_dataset(path)

    assert summary.normality["is_normal"].tolist()[:2] == [True, False]
    assert summary.overall_normality == "mixed"


def test_summarize_tick_density_dataset_expands_full_bucket_index_range(tmp_path: Path):
    path = tmp_path / "partial_5s.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                [
                    "2025-01-02T20:50:00Z",
                    "2025-01-03T20:50:55Z",
                ],
                utc=True,
            ),
            "date_utc": ["2025-01-02", "2025-01-03"],
            "bucket_index": [0, 11],
            "is_empty": [False, False],
            "tick_count": [10, 20],
            "total_size": [1, 1],
            "buy_ticks": [1, 1],
            "sell_ticks": [0, 0],
            "none_ticks": [0, 0],
        }
    ).to_parquet(path, index=False)

    summary = summarize_tick_density_dataset(path)

    assert summary.band_stats["bucket_index"].tolist() == list(range(12))
    assert np.isnan(summary.band_stats.loc[1, "mean_tick_count"])
    assert summary.normality["bucket_index"].tolist() == list(range(12))
    assert pd.isna(summary.normality.loc[1, "is_normal"])


def test_process_dataset_writes_band_outputs_only(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    out_dir = tmp_path / "figs"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                [
                    "2025-01-02T20:50:00Z",
                    "2025-01-02T20:51:00Z",
                    "2025-01-03T20:50:00Z",
                    "2025-01-03T20:51:00Z",
                ],
                utc=True,
            ),
            "date_utc": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "macro_minute_index": [40, 41, 40, 41],
            "tick_count": [10, 20, 30, 40],
            "total_size": [1, 1, 1, 1],
            "buy_ticks": [5, 10, 15, 20],
            "sell_ticks": [5, 10, 15, 20],
            "none_ticks": [0, 0, 0, 0],
        }
    ).to_parquet(path, index=False)

    process_dataset(path, out_dir=out_dir)

    assert (out_dir / "sample_bands.png").exists()
    assert (out_dir / "sample_band_stats.csv").exists()
    assert (out_dir / "sample_normality.csv").exists()
    assert (out_dir / "sample_total_size_bands.png").exists()
    assert (out_dir / "sample_total_size_band_stats.csv").exists()
    assert not (out_dir / "sample_hist.png").exists()
