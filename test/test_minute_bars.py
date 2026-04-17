import pandas as pd
import pytest

from utils.minute_bars import (
    build_market_time_columns,
    derive_session_window,
    normalize_minute_bars,
)


def test_normalize_minute_bars_prefers_datetime_utc_and_keeps_utc_dtype():
    raw = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 19:50:00", "2020-09-01 19:51:00"],
                utc=True,
            ),
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    )

    out = normalize_minute_bars(raw)

    assert list(out.columns) == ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
    assert str(out["datetime_utc"].dtype) == "datetime64[ns, UTC]"


def test_normalize_minute_bars_accepts_legacy_datetime_et_and_converts_to_utc():
    raw = pd.DataFrame(
        {
            "DateTime_ET": pd.to_datetime(["2020-09-01 15:50:00", "2020-09-01 15:51:00"]),
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    )

    out = normalize_minute_bars(raw)

    assert out["datetime_utc"].iloc[0] == pd.Timestamp("2020-09-01 19:50:00+00:00")
    assert out["datetime_utc"].iloc[1] == pd.Timestamp("2020-09-01 19:51:00+00:00")


def test_build_market_time_columns_derives_new_york_datetime_and_date():
    raw = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(["2020-09-01 19:50:00"], utc=True),
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [10],
        }
    )

    out = build_market_time_columns(raw)

    assert out["datetime_et"].iloc[0] == pd.Timestamp("2020-09-01 15:50:00")
    assert out["date_et"].iloc[0] == pd.Timestamp("2020-09-01")


def test_normalize_minute_bars_rejects_ambiguous_dst_fallback_et_inputs():
    raw = pd.DataFrame(
        {
            "DateTime_ET": ["2020-11-01 01:30:00"],
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [10],
        }
    )

    with pytest.raises(ValueError, match="ambiguous DST-fallback"):
        normalize_minute_bars(raw)


def test_derive_session_window_marks_macro_hour_without_persisting_it():
    raw = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 19:49:00", "2020-09-01 19:50:00", "2020-09-01 20:00:00"],
                utc=True,
            ),
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 11, 12],
        }
    )

    out = derive_session_window(build_market_time_columns(raw))

    assert out["window"].tolist() == ["H3PM", "MACRO", "POST"]
