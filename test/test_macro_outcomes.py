import pandas as pd

from macro_outcomes import compute_macro_outcomes


def test_compute_macro_outcomes_uses_datetime_utc_and_derived_macro_window():
    bars = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2020-09-01 19:49:00", "2020-09-01 19:50:00", "2020-09-01 19:59:00", "2020-09-01 20:00:00"],
                utc=True,
            ),
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [100.5, 102.5, 103.5, 104.0],
            "Low": [99.5, 100.5, 101.5, 102.5],
            "Close": [100.0, 102.0, 103.0, 103.5],
            "Volume": [1, 2, 3, 4],
        }
    )

    out = compute_macro_outcomes(bars, macro_window_name="MACRO")

    assert len(out) == 1
    assert out.loc[0, "date"] == pd.Timestamp("2020-09-01")
    assert out.loc[0, "macro_open"] == 101.0
    assert out.loc[0, "postclose_range_points"] == 1.5
