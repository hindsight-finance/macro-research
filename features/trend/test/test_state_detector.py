from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from features.trend.ADX.trend_quality import calculate_trend_quality
from features.trend.state_detector import (
    ADXIndicator,
    ATRRangeIndicator,
    IndicatorResult,
    SPDIndicator,
    StateDetector,
    _MODULE_DIR,
    _load_module,
)


def _load_fixture_window(start_time: str, end_time: str) -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "ADX" / "testing" / "NQ.csv"
    df = pd.read_csv(data_path, parse_dates=["DateTime_ET"]).rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    df["timestamp"] = df["DateTime_ET"]
    times = df["DateTime_ET"].dt.strftime("%H:%M")
    mask = (times >= start_time) & (times < end_time)
    return df.loc[mask, ["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.set_index("timestamp")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )


def _minimal_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.5, 100.5],
            "close": [100.5, 101.5],
        }
    )


class _StaticIndicator:
    def __init__(self, signal: float):
        self.signal = signal

    def calculate(self, df, **kwargs):
        return IndicatorResult(signal=self.signal)


class _RecordingATRIndicator:
    def __init__(self):
        self.called_session = None

    def calculate(self, df, session: str = "auto"):
        self.called_session = session
        return IndicatorResult(signal=0.4)


def test_state_detector_renormalizes_weights_for_enabled_subset():
    detector = StateDetector(enabled_indicators={"adx"}, dynamic_weights=False)
    detector.indicators["adx"] = _StaticIndicator(0.5)

    result = detector.detect(_minimal_ohlc(), session="1pm-3pm")

    assert result.weights == {"adx": pytest.approx(1.0)}


def test_state_detector_passes_session_to_atr_range_indicator():
    detector = StateDetector(enabled_indicators={"atr_range"}, dynamic_weights=False)
    recorder = _RecordingATRIndicator()
    detector.indicators["atr_range"] = recorder

    detector.detect(_minimal_ohlc(), session="3pm-3:50pm")

    assert recorder.called_session == "3pm-3:50pm"


def test_adx_indicator_resamples_to_window_bar_size_before_scoring():
    raw_bars = _load_fixture_window("13:00", "15:00")
    expected_bars = _resample_ohlc(raw_bars, "5min")
    expected = calculate_trend_quality(expected_bars, "1pm-3pm")

    result = ADXIndicator().calculate(raw_bars, session="1pm-3pm")

    assert result.signal == pytest.approx(expected["quality_score"])
    assert result.metadata["metadata"]["total_bars"] == len(expected_bars)


def test_atr_range_indicator_uses_session_specific_resampling():
    raw_bars = _load_fixture_window("15:00", "15:50")
    expected_bars = _resample_ohlc(raw_bars, "2min")
    atr_module = _load_module("atr_test", _MODULE_DIR / "ATR Range" / "atr.py")
    expected = atr_module.analyze_session(expected_bars, "3pm-3:50pm")

    result = ATRRangeIndicator().calculate(raw_bars, session="3pm-3:50pm")

    assert result.raw_value == pytest.approx(expected["raw_ratio"])
    assert result.signal == pytest.approx(1.0 - expected["raw_ratio"])


def test_spd_indicator_includes_last_resampled_bar_and_neutralizes_insufficient_data():
    raw_bars = _load_fixture_window("15:50", "16:00")

    result = SPDIndicator().calculate(raw_bars, session="3:50pm-4pm")

    assert result.metadata["classification"] == "insufficient_data"
    assert result.metadata["bars_analyzed"] == 2
    assert result.signal == pytest.approx(0.5)


def test_spd_indicator_maps_module_classification_to_signal():
    raw_bars = _load_fixture_window("15:00", "15:50")

    result = SPDIndicator().calculate(raw_bars, session="3pm-3:50pm")

    assert result.metadata["classification"] == "mixed"
    assert result.metadata["bars_analyzed"] == 10
    assert result.signal == pytest.approx(0.5)
