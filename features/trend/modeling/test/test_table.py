from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.trend.modeling.table import build_modeling_table


def _make_intraday_bars(
    trade_date: str,
    start_time: str,
    periods: int,
    session: str,
    window: str,
    drift: float = 0.05,
) -> pd.DataFrame:
    timestamp = pd.date_range(f"{trade_date} {start_time}", periods=periods, freq="min")
    base = 100.0 + np.arange(periods) * drift
    wobble = 0.03 * np.sin(np.arange(periods) / 5.0)
    open_ = base + wobble
    close = base + drift + 0.03 * np.cos(np.arange(periods) / 5.0)
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15

    return pd.DataFrame(
        {
            "DateTime_ET": timestamp,
            "session": session,
            "window": window,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100,
            "DateTime_UTC": timestamp.astype(str),
        }
    )


def test_build_modeling_table_emits_one_row_per_date_and_session(tmp_path: Path):
    bars = pd.concat(
        [
            _make_intraday_bars("2024-01-02", "13:00", 120, session="PM", window="NONE"),
            _make_intraday_bars("2024-01-03", "13:00", 120, session="PM", window="NONE"),
        ],
        ignore_index=True,
    )
    bars_path = tmp_path / "bars.parquet"
    bars.to_parquet(bars_path, index=False)

    table = build_modeling_table(
        input_path=bars_path,
        instrument="NQ",
        session_names=["1pm-3pm"],
    )

    assert {
        "instrument",
        "trade_date",
        "session_name",
        "window_start_ts",
        "window_end_ts",
        "n_bars_raw",
        "mss",
        "adx_quality",
        "adx_persistence_margin",
        "adx_persistence_control",
        "adx_persistence_recency",
        "irr",
        "er",
        "log_vr",
        "containment_displacement",
        "containment_edge_balance",
        "containment_inside_share",
        "containment_path_efficiency",
        "containment_overshoot_ratio",
        "containment_range_stability",
        "containment_mid_cross_count",
        "containment_swing_symmetry",
        "containment_ib_extension_ratio",
        "containment_ib_asymmetry",
        "containment_bandwidth_squeeze",
        "containment_vwap_acceptance",
        "containment_excess_rejection",
        "containment_target",
        "containment_status",
        "descriptive_target",
        "feature_status",
        "target_status",
    } <= set(table.columns)
    assert len(table) == 2
    assert table.groupby(["trade_date", "session_name"]).size().eq(1).all()
    assert set(table["feature_status"]) == {"ok"}
    assert set(table["containment_status"]) == {"ok"}
    assert table[
        [
            "containment_overshoot_ratio",
            "containment_range_stability",
            "containment_mid_cross_count",
            "containment_swing_symmetry",
            "containment_ib_extension_ratio",
            "containment_ib_asymmetry",
            "containment_bandwidth_squeeze",
            "containment_vwap_acceptance",
            "containment_excess_rejection",
        ]
    ].notna().all().all()
    assert set(table["session_name"]) == {"1pm-3pm"}


def test_build_modeling_table_extracts_windows_from_timestamps_not_tags(tmp_path: Path):
    bars = pd.concat(
        [
            _make_intraday_bars("2024-01-02", "13:00", 120, session="OTHER", window="NONE"),
            _make_intraday_bars("2024-01-02", "15:00", 50, session="OTHER", window="NONE"),
            _make_intraday_bars("2024-01-02", "15:50", 10, session="OTHER", window="NONE"),
        ],
        ignore_index=True,
    )
    bars_path = tmp_path / "bars.parquet"
    bars.to_parquet(bars_path, index=False)

    table = build_modeling_table(
        input_path=bars_path,
        instrument="NQ",
        session_names=["1pm-3pm", "3pm-3:50pm", "3:50pm-4pm"],
    )

    counts = table.set_index("session_name")["n_bars_raw"].to_dict()

    assert set(table["session_name"]) == {"1pm-3pm", "3pm-3:50pm", "3:50pm-4pm"}
    assert counts == {
        "1pm-3pm": 120,
        "3pm-3:50pm": 50,
        "3:50pm-4pm": 10,
    }
