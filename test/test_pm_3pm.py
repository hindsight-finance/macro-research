import pandas as pd

from features.pm_3pm import _load_minutes


def test_load_minutes_accepts_datetime_utc_and_derives_new_york_time(tmp_path):
    path = tmp_path / "nq_minute_base.parquet"
    pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 17:00:00", "2020-09-01 19:00:00", "2020-09-01 19:49:00"],
                utc=True,
            ),
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 11, 12],
        }
    ).to_parquet(path, index=False)

    out = _load_minutes([str(path)])

    assert "timestamp" in out.columns
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-09-01 13:00:00")
