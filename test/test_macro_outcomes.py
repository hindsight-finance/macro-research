import polars as pl

from macro_outcomes import compute_macro_outcomes


def test_compute_macro_outcomes_uses_datetime_utc_and_derived_macro_window():
    bars = pl.DataFrame(
        {
            "datetime_utc": [
                "2020-09-01 19:49:00+00:00",
                "2020-09-01 19:50:00+00:00",
                "2020-09-01 19:59:00+00:00",
                "2020-09-01 20:00:00+00:00",
            ],
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [100.5, 102.5, 103.5, 104.0],
            "Low": [99.5, 100.5, 101.5, 102.5],
            "Close": [100.0, 102.0, 103.0, 103.5],
            "Volume": [1, 2, 3, 4],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))

    out = compute_macro_outcomes(bars, macro_window_name="MACRO")

    assert isinstance(out, pl.DataFrame)
    assert out.height == 1
    assert str(out.item(0, "date")) == "2020-09-01"
    assert out.item(0, "macro_open") == 101.0
    assert out.item(0, "postclose_range_points") == 1.5
