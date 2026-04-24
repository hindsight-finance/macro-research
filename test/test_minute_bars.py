import polars as pl
import pytest

from utils.minute_bars import (
    build_market_time_columns,
    derive_session_window,
    normalize_minute_bars,
)


def _tz(dtype) -> str | None:
    return getattr(dtype, "time_zone", None)


def test_normalize_minute_bars_prefers_datetime_utc_and_keeps_utc_dtype():
    raw = pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 19:50:00+00:00", "2020-09-01 19:51:00+00:00"],
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    out = normalize_minute_bars(raw)

    assert out.columns == ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
    assert _tz(out.schema["datetime_utc"]) == "UTC"


def test_normalize_minute_bars_accepts_legacy_datetime_et_and_converts_to_utc():
    raw = pl.DataFrame(
        {
            "DateTime_ET": ["2020-09-01 15:50:00", "2020-09-01 15:51:00"],
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    )

    out = normalize_minute_bars(raw)

    assert out["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S%z").to_list() == [
        "2020-09-01 19:50:00+0000",
        "2020-09-01 19:51:00+0000",
    ]


def test_build_market_time_columns_derives_new_york_datetime_and_date():
    raw = pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 19:50:00+00:00"],
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [10],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    out = build_market_time_columns(raw)

    assert out.item(0, "datetime_et") == out.item(0, "datetime_utc").replace(tzinfo=None) - __import__("datetime").timedelta(hours=4)
    assert str(out.item(0, "date_et")) == "2020-09-01"


def test_normalize_minute_bars_rejects_ambiguous_dst_fallback_et_inputs():
    raw = pl.DataFrame(
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
    raw = pl.DataFrame(
        {
            "datetime_utc": [
                "2020-09-01 19:49:00+00:00",
                "2020-09-01 19:50:00+00:00",
                "2020-09-01 20:00:00+00:00",
            ],
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 11, 12],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    out = derive_session_window(build_market_time_columns(raw))

    assert out["window"].to_list() == ["H3PM", "MACRO", "POST"]
