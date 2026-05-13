from datetime import date

import polars as pl

from features.pm_1h_macro_context import build_hourly_context, build_summary, join_macro_outcomes


def _hour_rows(day: str, hour: int, open_: float, high: float, low: float, close: float, volume: int = 100) -> list[dict]:
    rows = []
    for minute in range(60):
        rows.append(
            {
                "datetime_utc": f"{day} {hour + 4:02d}:{minute:02d}:00+00:00",
                "Open": open_ if minute == 0 else (open_ + close) / 2,
                "High": high,
                "Low": low,
                "Close": close if minute == 59 else (open_ + close) / 2,
                "Volume": volume,
            }
        )
    return rows


def _minute_frame(*hour_specs: tuple[str, int, float, float, float, float]) -> pl.DataFrame:
    rows = []
    for spec in hour_specs:
        rows.extend(_hour_rows(*spec))
    return pl.DataFrame(rows).with_columns(pl.col("datetime_utc").str.to_datetime(time_zone="UTC"))


def test_build_hourly_context_detects_bullish_no_leak_fvg():
    bars = _minute_frame(
        ("2020-09-01", 12, 100.0, 105.0, 99.0, 104.0),
        ("2020-09-01", 13, 106.0, 110.0, 105.5, 109.0),
        ("2020-09-01", 14, 111.0, 116.0, 110.0, 115.0),
    )

    out = build_hourly_context(bars)

    assert out.height == 1
    assert out.item(0, "date") == date(2020, 9, 1)
    assert out.item(0, "fvg_direction") == "bullish"
    assert out.item(0, "has_fvg") is True
    assert out.item(0, "fvg_size_points") == 5.0


def test_build_hourly_context_detects_bearish_no_leak_fvg():
    bars = _minute_frame(
        ("2020-09-02", 12, 120.0, 121.0, 115.0, 116.0),
        ("2020-09-02", 13, 114.0, 116.0, 110.0, 111.0),
        ("2020-09-02", 14, 109.0, 112.0, 104.0, 105.0),
    )

    out = build_hourly_context(bars)

    assert out.item(0, "fvg_direction") == "bearish"
    assert out.item(0, "has_fvg") is True
    assert out.item(0, "fvg_size_points") == 3.0


def test_build_hourly_context_buckets_imbalance_direction():
    bars = _minute_frame(
        ("2020-09-01", 12, 100.0, 101.0, 99.0, 100.0),
        ("2020-09-01", 13, 100.0, 105.0, 99.0, 104.0),
        ("2020-09-01", 14, 104.0, 110.0, 103.0, 109.0),
        ("2020-09-02", 12, 100.0, 101.0, 99.0, 100.0),
        ("2020-09-02", 13, 110.0, 111.0, 105.0, 106.0),
        ("2020-09-02", 14, 106.0, 107.0, 100.0, 101.0),
        ("2020-09-03", 12, 100.0, 101.0, 99.0, 100.0),
        ("2020-09-03", 13, 100.0, 106.0, 99.0, 103.0),
        ("2020-09-03", 14, 103.0, 107.0, 99.0, 102.0),
    )

    out = build_hourly_context(bars).sort("date")

    assert out.select("imbalance_direction").to_series().to_list() == ["bullish", "bearish", "neutral"]


def test_join_macro_outcomes_adds_direction_fields_and_excludes_flat_by_default():
    context = pl.DataFrame(
        {
            "date": [date(2020, 9, 1), date(2020, 9, 2), date(2020, 9, 3)],
            "fvg_direction": ["bullish", "bearish", "none"],
            "imbalance_direction": ["bullish", "bearish", "neutral"],
            "fvg_size_points": [5.0, 3.0, 0.0],
        }
    )
    macro = pl.DataFrame(
        {
            "date": [date(2020, 9, 1), date(2020, 9, 2), date(2020, 9, 3)],
            "macro_open": [100.0, 100.0, 100.0],
            "macro_close": [102.0, 98.0, 100.0],
            "macro_dir_points": [2.0, -2.0, 0.0],
            "macro_range_points": [4.0, 5.0, 1.0],
        }
    )

    out = join_macro_outcomes(context, macro)

    assert out.height == 2
    assert out.select("macro_dir_sign").to_series().to_list() == [1, -1]
    assert out.select("macro_direction").to_series().to_list() == ["bullish", "bearish"]


def test_build_summary_reports_core_cohort_rates():
    joined = pl.DataFrame(
        {
            "date": [date(2020, 9, 1), date(2020, 9, 2), date(2020, 9, 3)],
            "fvg_direction": ["bullish", "bullish", "bearish"],
            "imbalance_direction": ["bullish", "neutral", "bearish"],
            "fvg_size_points": [2.0, 6.0, 4.0],
            "macro_dir_sign": [1, -1, -1],
            "macro_dir_points": [3.0, -1.0, -2.0],
            "macro_range_points": [5.0, 2.0, 4.0],
        }
    )

    summary = build_summary(joined)
    bull_fvg = summary.filter((pl.col("cohort") == "fvg_direction") & (pl.col("bucket") == "bullish"))

    assert {"fvg_direction", "imbalance_direction", "fvg_x_imbalance", "fvg_size_bucket"}.issubset(
        set(summary.select("cohort").to_series().to_list())
    )
    assert bull_fvg.item(0, "n") == 2
    assert bull_fvg.item(0, "macro_bull_rate") == 0.5
    assert bull_fvg.item(0, "macro_bear_rate") == 0.5
