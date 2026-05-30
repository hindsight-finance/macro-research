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
