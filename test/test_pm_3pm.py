from datetime import datetime

import polars as pl

from features.pm_3pm import _load_minutes


def test_load_minutes_accepts_datetime_utc_and_derives_new_york_time(tmp_path):
    path = tmp_path / "nq_minute_base.parquet"
    pl.DataFrame(
        {
            "datetime_utc": ["2020-09-01 17:00:00+00:00", "2020-09-01 19:00:00+00:00", "2020-09-01 19:49:00+00:00"],
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 11, 12],
        }
    ).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC")).write_parquet(path)

    out = _load_minutes([str(path)])

    assert isinstance(out, pl.DataFrame)
    assert "timestamp" in out.columns
    assert out.item(0, "timestamp") == datetime(2020, 9, 1, 13, 0)
