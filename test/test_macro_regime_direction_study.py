from pathlib import Path
from datetime import date

import polars as pl
import pytest

from features.macro_regime_direction_study import (
    build_macro_regime_direction_study,
    classify_delta_sign,
    classify_macro_direction,
    summarize_macro_regime_direction_study,
    write_macro_regime_direction_study,
)


def _regime_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": ["NQ"] * 6,
            "trade_date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-04", "2025-01-04"],
            "session_name": ["1pm-3pm", "3pm-3:50pm"] * 3,
            "trend_score": [0.1, 0.8, 0.4, 0.2, 0.9, 0.5],
            "containment_score": [0.9, 0.2, 0.5, 0.8, 0.1, 0.4],
            "chop_score": [0.2, 0.1, 0.5, 0.7, 0.8, 0.4],
            "feature_status": ["ok"] * 6,
        }
    ).with_columns(pl.col("trade_date").str.to_date())


def _macro_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "macro_dir_points": [10.0, -5.0, 0.0],
            "macro_range_pct": [0.8, 0.5, 0.2],
            "skew_ratio": [0.75, 0.25, 0.5],
            "close_in_range": [0.9, 0.1, 0.5],
            "macro_high_time": [8, 2, 5],
            "macro_low_time": [1, 9, 4],
            "postclose_range_pct": [0.3, 0.4, 0.1],
        }
    ).with_columns(pl.col("date").str.to_date())


def _delta_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime_utc": [
                "2025-01-02T18:00:00Z",  # 13:00 ET
                "2025-01-02T20:00:00Z",  # 15:00 ET
                "2025-01-02T20:50:00Z",  # 15:50 ET
                "2025-01-03T18:30:00Z",
                "2025-01-03T20:10:00Z",
                "2025-01-03T20:55:00Z",
            ],
            "trade_date_et": ["2025-01-02", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-03"],
            "buy_size": [100, 10, 40, 10, 50, 5],
            "sell_size": [20, 30, 10, 30, 10, 35],
            "classified_size": [120, 40, 50, 40, 60, 40],
            "total_size": [120, 40, 50, 40, 60, 40],
            "volume_delta": [80, -20, 30, -20, 40, -30],
            "tick_delta": [8, -2, 3, -2, 4, -3],
            "classified_share": [1.0] * 6,
        }
    ).with_columns(
        pl.col("datetime_utc").str.to_datetime(time_zone="UTC"),
        pl.col("trade_date_et").str.to_date(),
    )


def test_classifies_macro_direction_and_delta_signs():
    assert classify_macro_direction(2.5) == "bullish"
    assert classify_macro_direction(-0.25) == "bearish"
    assert classify_macro_direction(0.0) == "flat"

    assert classify_delta_sign(10.0) == "buy"
    assert classify_delta_sign(-1.0) == "sell"
    assert classify_delta_sign(0.0) == "neutral"


def test_build_study_joins_regime_macro_and_window_deltas():
    out = build_macro_regime_direction_study(_regime_frame(), _macro_frame(), _delta_frame())

    assert out.height == 3
    assert out["date"].to_list() == [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]

    bull = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    assert bull["macro_direction"] == "bullish"
    assert bull["trend_score_13_15_bucket"] == "low"
    assert bull["trend_score_15_1550_bucket"] == "high"
    assert bull["delta_available"] is True
    assert bull["delta_13_15_volume_delta"] == 80
    assert bull["delta_15_1550_volume_delta"] == -20
    assert bull["delta_1550_16_volume_delta"] == 30
    assert bull["delta_13_15_delta_sign"] == "buy"
    assert bull["delta_15_1550_delta_sign"] == "sell"

    missing_delta = out.filter(pl.col("date") == pl.date(2025, 1, 4)).row(0, named=True)
    assert missing_delta["macro_direction"] == "flat"
    assert missing_delta["delta_available"] is False
    assert missing_delta["delta_13_15_volume_delta"] is None


def test_summarize_study_reports_baseline_rates_and_lift():
    study = build_macro_regime_direction_study(_regime_frame(), _macro_frame(), _delta_frame())

    summaries = summarize_macro_regime_direction_study(study)

    baseline = summaries["baseline"].row(0, named=True)
    assert baseline["sample_size"] == 3
    assert baseline["bullish_rate"] == 1 / 3
    assert baseline["bearish_rate"] == 1 / 3

    bucket_summary = summaries["single_bucket"]
    low_13_15_trend = bucket_summary.filter(
        (pl.col("cohort") == "trend_score_13_15_bucket") & (pl.col("bucket") == "low")
    ).row(0, named=True)
    assert low_13_15_trend["sample_size"] == 1
    assert low_13_15_trend["bullish_rate"] == 1.0
    assert low_13_15_trend["bullish_lift"] == pytest.approx(2 / 3)

    correlations = summaries["correlations"]
    assert {"feature", "outcome", "pearson", "spearman", "sample_size"}.issubset(set(correlations.columns))
    assert "trend_score_13_15" in correlations["feature"].to_list()


def test_write_study_outputs_parquet_csv_and_figures(tmp_path: Path):
    regime_path = tmp_path / "regime.parquet"
    macro_path = tmp_path / "macro.parquet"
    delta_path = tmp_path / "delta.parquet"
    output_path = tmp_path / "study.parquet"
    fig_dir = tmp_path / "figs"
    _regime_frame().write_parquet(regime_path)
    _macro_frame().write_parquet(macro_path)
    _delta_frame().write_parquet(delta_path)

    wrote = write_macro_regime_direction_study(
        regime_path=regime_path,
        macro_path=macro_path,
        delta_path=delta_path,
        output_path=output_path,
        figure_dir=fig_dir,
    )

    assert wrote["study"] == output_path
    assert output_path.exists()
    assert (fig_dir / "baseline.csv").exists()
    assert (fig_dir / "single_bucket.csv").exists()
    assert (fig_dir / "correlations.csv").exists()
    assert (fig_dir / "trend_window_cohort_heatmap.png").exists()
