from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.trend.historical_regimes import build_parser, main


def _make_intraday_bars(trade_date: str, start_time: str, periods: int) -> pd.DataFrame:
    timestamp = pd.date_range(f"{trade_date} {start_time}", periods=periods, freq="min")
    base = 100.0 + np.arange(periods) * 0.05
    open_ = base
    close = base + 0.02
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    return pd.DataFrame(
        {
            "DateTime_ET": timestamp,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100,
        }
    )


def test_build_parser_accepts_session_names_and_thresholds():
    args = build_parser().parse_args(
        [
            "--input-path",
            "outputs/nq_1m.parquet",
            "--instrument",
            "NQ",
            "--output-path",
            "outputs/nq_regimes.parquet",
            "--session-name",
            "1pm-3pm",
            "--session-name",
            "3pm-3:50pm",
            "--trend-high",
            "0.8",
        ]
    )

    assert args.session_names == ["1pm-3pm", "3pm-3:50pm"]
    assert args.trend_high == 0.8


def test_main_writes_scores_and_label_to_parquet(tmp_path: Path):
    bars = pd.concat(
        [
            _make_intraday_bars("2024-01-02", "13:00", 120),
            _make_intraday_bars("2024-01-03", "13:00", 120),
        ],
        ignore_index=True,
    )
    input_path = tmp_path / "bars.parquet"
    output_path = tmp_path / "historical_regimes.parquet"
    bars.to_parquet(input_path, index=False)

    exit_code = main(
        [
            "--input-path",
            str(input_path),
            "--instrument",
            "NQ",
            "--output-path",
            str(output_path),
            "--session-name",
            "1pm-3pm",
        ]
    )

    result = pd.read_parquet(output_path)
    assert exit_code == 0
    assert {"trend_score", "containment_score", "chop_score", "label"} <= set(result.columns)
    assert set(result["label"]) <= {"trend", "containment", "chop", "uncertain"}
