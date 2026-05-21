from pathlib import Path

import polars as pl

from viz.macro_vwap_barrier_context_viz import process_dataset


def test_process_dataset_writes_distribution_csvs_and_figures(tmp_path: Path):
    context_path = tmp_path / "context.parquet"
    summary_path = tmp_path / "summary.parquet"
    output_dir = tmp_path / "figs"
    pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "vwap_10s_dist_points": [1.0, -1.0, 0.25],
            "vwap_1555_dist_points": [2.0, -2.0, 0.5],
            "worst_wrong_side_dist_points": [0.0, 3.0, 1.0],
            "wrong_side_share_1550": [0.0, 0.5, 0.25],
            "vwap_context_10s_to_1555": ["constructive_to_constructive", "wrong_to_wrong", "touch_mixed"],
            "target_1555_1559_points": [5.0, -4.0, 1.0],
            "target_1550_10s_1559_points": [6.0, -5.0, 2.0],
        }
    ).with_columns(pl.col("date").str.to_date()).write_parquet(context_path)
    pl.DataFrame(
        {
            "scope": ["vwap_10s_only"],
            "bucket": ["constructive"],
            "target_name": ["target_1555_1559"],
            "sample_size": [1],
            "bullish_count": [1],
            "bearish_count": [0],
            "neutral_count": [0],
            "bullish_pct": [100.0],
            "bearish_pct": [0.0],
            "neutral_pct": [0.0],
            "avg_target_points": [5.0],
            "median_target_points": [5.0],
            "p10_target_points": [5.0],
            "p25_target_points": [5.0],
            "p75_target_points": [5.0],
            "p90_target_points": [5.0],
        }
    ).write_parquet(summary_path)

    wrote = process_dataset(context_path, summary_path, output_dir)

    names = {path.name for path in wrote}
    assert "summary_by_scope.csv" in names
    assert "wrong_side_quantiles.csv" in names
    assert "target_quantiles_by_bucket.csv" in names
    assert "vwap_1555_decision_summary.csv" in names
    assert "vwap_10s_dist_hist.png" in names
    assert "wrong_side_dist_ecdf.png" in names
    assert "target_by_wrong_side_bucket_violin.png" in names
    assert "target_by_1555_context_violin.png" in names
    assert "barrier_vwap_heatmap.png" in names
    assert all(path.exists() for path in wrote)
