from __future__ import annotations

import datetime as dt

import polars as pl

import features.macro_1550_vwap_retouch as m

UTC = dt.timezone.utc


def _write_tick_fixture(path, ticks):
    """ticks: list of (ts_event_utc: datetime, intra_ts_rank, price_ticks, size)."""
    df = pl.DataFrame(
        {
            "ts_event": [t[0] for t in ticks],
            "intra_ts_rank": [t[1] for t in ticks],
            "price_ticks": [t[2] for t in ticks],
            "size": [t[3] for t in ticks],
        },
        schema={
            "ts_event": pl.Datetime("ns", time_zone="UTC"),
            "intra_ts_rank": pl.Int64,
            "price_ticks": pl.Int64,
            "size": pl.Int64,
        },
    )
    df.write_parquet(path)


def test_constants_and_blank_row():
    assert m.S_1550 == 15 * 3600 + 50 * 60
    assert m.S_1555 == 15 * 3600 + 55 * 60
    assert m.S_1600 == 16 * 3600
    row = m._blank_row(dt.date(2024, 3, 7), tick_count_macro=0, has_first10=False)
    assert set(row.keys()) == set(m.MACRO_1550_VWAP_RETOUCH_COLUMNS)
    assert row["date"] == dt.date(2024, 3, 7)
    assert row["has_first10"] is False
    assert row["break_side"] is None


def test_scan_macro_window_filters_and_derives_et_dst(tmp_path):
    # EST date (2024-03-07, before US DST): 15:50:00 ET == 20:50:00 UTC
    est = dt.datetime(2024, 3, 7, 20, 50, 0, tzinfo=UTC)
    # EDT date (2024-03-15, after US DST): 15:50:00 ET == 19:50:00 UTC
    edt = dt.datetime(2024, 3, 15, 19, 50, 0, tzinfo=UTC)
    # one tick inside the window on each date, plus one outside (15:40 ET) that must be dropped
    outside = dt.datetime(2024, 3, 7, 20, 40, 0, tzinfo=UTC)
    path = tmp_path / "ticks.parquet"
    _write_tick_fixture(
        path,
        [
            (outside, 0, 4000, 1),
            (est, 0, 4000, 1),
            (edt, 0, 4000, 1),
        ],
    )
    out = m._scan_macro_window(path).collect().sort("ts_event")
    # the 15:40 ET tick is filtered out; both 15:50 ET ticks survive with et_second == S_1550
    assert out.height == 2
    assert out["et_second"].to_list() == [m.S_1550, m.S_1550]
    assert out["date"].to_list() == [dt.date(2024, 3, 7), dt.date(2024, 3, 15)]
    assert out["price"].to_list() == [1000.0, 1000.0]  # 4000 / 4.0


def test_scan_macro_window_missing_column_raises(tmp_path):
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"ts_event": [dt.datetime(2024, 3, 7, 20, 50, tzinfo=UTC)]}).write_parquet(path)
    try:
        m._scan_macro_window(path)
    except ValueError as exc:
        assert "Missing tick columns" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing tick columns")


BASE_EST = dt.datetime(2024, 3, 7, 20, 50, 0, tzinfo=UTC)  # 15:50:00 ET on an EST date


def _day(ticks):
    """ticks: list of (offset_seconds_from_1550, price, size[, rank]).

    Builds a day frame in the shape `_scan_macro_window` produces (et_second/price present).
    """
    rows = []
    for i, t in enumerate(ticks):
        off, price, size = t[0], t[1], t[2]
        rank = t[3] if len(t) > 3 else i
        rows.append(
            {
                "ts_event": BASE_EST + dt.timedelta(seconds=off),
                "intra_ts_rank": rank,
                "price_ticks": int(round(price * 4)),
                "size": size,
                "date": dt.date(2024, 3, 7),
                "et_second": m.S_1550 + off,
                "price": float(price),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "ts_event": pl.Datetime("ns", time_zone="UTC"),
            "intra_ts_rank": pl.Int64,
            "price_ticks": pl.Int64,
            "size": pl.Int64,
            "date": pl.Date,
            "et_second": pl.Int32,
            "price": pl.Float64,
        },
    )


# first-10s range: high=101, low=99, vwap=100  (used by several cases)
FIRST10 = [(0, 100.0, 1), (1, 101.0, 1), (2, 99.0, 1)]


def test_clean_high_break_bullish_with_retouch_and_continuation():
    day = _day(FIRST10 + [(15, 102.0, 1), (20, 100.0, 1), (30, 105.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["has_first10"] is True
    assert r["high_10s"] == 101.0 and r["low_10s"] == 99.0 and r["vwap_10s_frozen"] == 100.0
    assert r["trigger_state"] == "triggered"
    assert r["break_side"] == "high" and r["bias"] == "bullish"
    assert r["break_price"] == 102.0 and r["break_time_s"] == 15
    # retouch back down to the frozen VWAP (100 <= 100 + 0.25)
    assert r["retouch_frozen_occurred"] is True
    assert r["retouch_frozen_price"] == 100.0 and r["retouch_frozen_time_s"] == 20
    assert r["retouch_frozen_lag_s"] == 5
    # forward (signed by bias) to macro close = 105
    assert r["fwd_break_1559_points"] == 3.0          # 105 - 102
    assert r["fwd_retouch_frozen_1559_points"] == 5.0  # 105 - 100
    assert r["mfe_retouch_frozen_points"] == 5.0 and r["mae_retouch_frozen_points"] == 0.0
    # validation: macro closed up
    assert r["macro_trend_state"] == "bullish"
    assert r["bias_matches_macro"] is True


def test_clean_low_break_bearish():
    day = _day(FIRST10 + [(15, 98.0, 1), (20, 100.0, 1), (30, 95.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["break_side"] == "low" and r["bias"] == "bearish"
    assert r["break_price"] == 98.0
    assert r["retouch_frozen_occurred"] is True and r["retouch_frozen_price"] == 100.0
    assert r["fwd_break_1559_points"] == 3.0   # bearish: 98 - 95
    assert r["macro_trend_state"] == "bearish" and r["bias_matches_macro"] is True


def test_whipsaw_first_break_wins():
    # low breaks at +12 before high breaks at +14 -> bias bearish
    day = _day(FIRST10 + [(12, 98.0, 1), (14, 102.0, 1), (30, 100.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["break_side"] == "low" and r["bias"] == "bearish" and r["break_time_s"] == 12


def test_no_break_is_no_trigger():
    # post ticks only touch the levels (non-strict) -> no break
    day = _day(FIRST10 + [(15, 100.0, 1), (20, 101.0, 1), (30, 99.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["trigger_state"] == "no_trigger"
    assert r["break_side"] is None and r["bias"] is None
    assert r["fwd_break_1559_points"] is None
    assert r["has_first10"] is True and r["high_10s"] == 101.0


def test_break_but_no_retouch_runaway():
    day = _day(FIRST10 + [(15, 102.0, 1), (20, 103.0, 1), (30, 104.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["trigger_state"] == "triggered" and r["break_side"] == "high"
    assert r["retouch_frozen_occurred"] is False and r["retouch_rolling_occurred"] is False
    assert r["fwd_break_1559_points"] == 2.0           # 104 - 102
    assert r["fwd_retouch_frozen_1559_points"] is None


def test_late_retouch_nulls_passed_horizon():
    # a 15:51:40 tick provides the 15:54 horizon price; retouch happens at 15:55:20 (>= S_1555)
    day = _day(FIRST10 + [(15, 102.0, 1), (100, 103.0, 1), (320, 100.0, 1), (340, 106.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["retouch_frozen_occurred"] is True and r["retouch_frozen_time_s"] == 320
    assert r["fwd_retouch_frozen_1554_points"] is None   # anchor (et 15:55:20) >= 15:55 cutoff
    assert r["fwd_retouch_frozen_1559_points"] is not None


def test_empty_first10_returns_blank():
    day = _day([(15, 100.0, 1), (20, 101.0, 1)])  # nothing in [0, 10)
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["has_first10"] is False
    assert r["break_side"] is None and r["vwap_10s_frozen"] is None


def test_exact_tie_same_ts_and_rank_is_stable():
    # two identical post ticks (same ts + rank): must not crash; first break detected
    day = _day(FIRST10 + [(15, 102.0, 1, 0), (15, 102.0, 1, 0), (30, 104.0, 1)])
    r = m.detect_retouch_events(day, date=dt.date(2024, 3, 7))
    assert r["break_side"] == "high" and r["break_price"] == 102.0


def test_build_and_summarize_end_to_end(tmp_path):
    # One bullish high-break day with a frozen+rolling retouch, written as a tick parquet.
    base = dt.datetime(2024, 3, 7, 20, 50, 0, tzinfo=UTC)
    offsets = [(0, 100.0, 1), (1, 101.0, 1), (2, 99.0, 1), (15, 102.0, 1), (20, 100.0, 1), (30, 105.0, 1)]
    ticks = [(base + dt.timedelta(seconds=o), i, int(round(p * 4)), s) for i, (o, p, s) in enumerate(offsets)]
    path = tmp_path / "ticks.parquet"
    _write_tick_fixture(path, ticks)

    df = m.build_macro_1550_vwap_retouch(path)
    assert df.height == 1
    assert df.columns == m.MACRO_1550_VWAP_RETOUCH_COLUMNS
    assert df["break_side"][0] == "high" and df["bias"][0] == "bullish"
    assert df["retouch_frozen_occurred"][0] is True

    summary = m.summarize_macro_1550_vwap_retouch(df)
    assert set(summary.columns) == set(m.SUMMARY_COLUMNS)
    # the triggered-coverage row reports 1 of 1 days triggered
    cov = summary.filter((pl.col("scope") == "coverage") & (pl.col("bucket") == "triggered"))
    assert cov.height == 1 and cov["value"][0] == 100.0 and cov["sample_size"][0] == 1


def test_write_outputs(tmp_path):
    base = dt.datetime(2024, 3, 7, 20, 50, 0, tzinfo=UTC)
    offsets = [(0, 100.0, 1), (1, 101.0, 1), (2, 99.0, 1), (15, 102.0, 1), (30, 105.0, 1)]
    ticks = [(base + dt.timedelta(seconds=o), i, int(round(p * 4)), s) for i, (o, p, s) in enumerate(offsets)]
    src = tmp_path / "ticks.parquet"
    _write_tick_fixture(src, ticks)
    out = tmp_path / "retouch.parquet"
    summ = tmp_path / "retouch_summary.parquet"
    a, b = m.write_macro_1550_vwap_retouch(src, out, summ)
    assert a.exists() and b.exists()
    assert pl.read_parquet(a).height == 1
