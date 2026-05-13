from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from features.trend.modeling.table import build_modeling_table


MARKET_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


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


def _make_utc_intraday_bars(trade_date: str, start_time_et: str, periods: int, drift: float = 0.05) -> pd.DataFrame:
    timestamp_et = pd.date_range(
        f"{trade_date} {start_time_et}",
        periods=periods,
        freq="min",
        tz=MARKET_TZ,
    )
    timestamp_utc = timestamp_et.tz_convert(UTC)
    base = 100.0 + np.arange(periods) * drift
    wobble = 0.03 * np.sin(np.arange(periods) / 5.0)
    open_ = base + wobble
    close = base + drift + 0.03 * np.cos(np.arange(periods) / 5.0)
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15

    return pd.DataFrame(
        {
            "datetime_utc": timestamp_utc,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100,
        }
    )


def test_build_modeling_table_requires_canonical_utc_timestamp_column(tmp_path: Path):
    bars = _make_utc_intraday_bars("2024-01-02", "13:00", 120)
    bars_path = tmp_path / "bars.parquet"
    bars.to_parquet(bars_path, index=False)

    table = build_modeling_table(
        input_path=bars_path,
        instrument="NQ",
        session_names=["1pm-3pm"],
    )

    assert len(table) == 1
    assert table["session_name"].iloc[0] == "1pm-3pm"
    assert table["n_bars_raw"].iloc[0] == 120


def test_build_modeling_table_rejects_files_without_canonical_utc_timestamp_column(tmp_path: Path):
    bars = _make_intraday_bars("2024-01-02", "13:00", 120, session="PM", window="NONE")
    bars_path = tmp_path / "bars.parquet"
    bars.to_parquet(bars_path, index=False)

    with pytest.raises(ValueError, match="datetime_utc"):
        build_modeling_table(
            input_path=bars_path,
            instrument="NQ",
            session_names=["1pm-3pm"],
        )


def test_build_modeling_table_uses_new_york_session_windows_from_utc_across_dst(tmp_path: Path):
    bars = pd.concat(
        [
            _make_utc_intraday_bars("2024-03-08", "13:00", 120),
            _make_utc_intraday_bars("2024-03-11", "13:00", 120),
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

    assert len(table) == 2
    assert set(table["n_bars_raw"]) == {120}
    assert set(table["trade_date"].astype(str)) == {"2024-03-08", "2024-03-11"}


def test_build_modeling_table_emits_one_row_per_date_and_session(tmp_path: Path):
    bars = pd.concat(
        [
            _make_utc_intraday_bars("2024-01-02", "13:00", 120),
            _make_utc_intraday_bars("2024-01-03", "13:00", 120),
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
        "trend_score",
        "containment_score",
        "chop_flip_rate",
        "chop_path_waste",
        "chop_outside_share",
        "chop_instability",
        "chop_score",
        "chop_status",
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
    assert table["trend_score"].equals(table["descriptive_target"])
    assert table["containment_score"].equals(table["containment_target"])
    assert table[
        [
            "chop_flip_rate",
            "chop_path_waste",
            "chop_outside_share",
            "chop_instability",
            "chop_score",
        ]
    ].notna().all().all()
    assert set(table["chop_status"]) == {"ok"}
    assert set(table["session_name"]) == {"1pm-3pm"}


def test_build_modeling_table_extracts_windows_from_timestamps_not_tags(tmp_path: Path):
    bars = pd.concat(
        [
            _make_utc_intraday_bars("2024-01-02", "13:00", 120),
            _make_utc_intraday_bars("2024-01-02", "15:00", 50),
            _make_utc_intraday_bars("2024-01-02", "15:50", 10),
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


def test_build_modeling_table_emits_state_detector_regime_columns(tmp_path: Path):
    bars = _make_utc_intraday_bars("2024-01-02", "13:00", 120)
    bars_path = tmp_path / "bars.parquet"
    bars.to_parquet(bars_path, index=False)

    table = build_modeling_table(
        input_path=bars_path,
        instrument="NQ",
        session_names=["1pm-3pm"],
    )

    row = table.iloc[0]
    assert {
        "regime_state",
        "regime_direction",
        "regime_confidence",
        "regime_signal_composite",
        "regime_status",
    } <= set(table.columns)
    assert row["regime_state"] in {"STRONG_TREND", "WEAK_TREND", "CONSOLIDATION", "CHOPPY", "UNCERTAIN"}
    assert row["regime_direction"] in {"UP", "DOWN", "NEUTRAL"}
    assert 0.0 <= row["regime_confidence"] <= 1.0
    assert 0.0 <= row["regime_signal_composite"] <= 1.0
    assert row["regime_status"] == "ok"
