from pathlib import Path

import polars as pl
import pytest

from utils.tick_data import (
    TICK_COLUMNS,
    collect_tick_window,
    get_tick_schema,
    scan_tick_data,
    ticks_to_minute_bars,
)


def _write_ticks(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)


def test_scan_tick_data_returns_lazyframe_without_collecting(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": ["2025-01-02T20:50:00.000000000Z", "2025-01-02T20:50:00.500000000Z"],
            "intra_ts_rank": [0, 1],
            "side": [1, 2],
            "price_ticks": [84000, 84004],
            "size": [1, 3],
        },
    )

    lf = scan_tick_data(path)

    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().height == 2


def test_get_tick_schema_reads_expected_columns(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "intra_ts_rank": [0],
            "side": [2],
            "price_ticks": [84000],
            "size": [5],
        },
    )

    schema = get_tick_schema(path)

    assert set(schema.names) == set(TICK_COLUMNS)


def test_collect_tick_window_requires_bounded_time_range(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "intra_ts_rank": [0],
            "side": [2],
            "price_ticks": [84000],
            "size": [5],
        },
    )

    with pytest.raises(ValueError, match="bounded start/end"):
        collect_tick_window(path, start_utc=None, end_utc=None)


def test_collect_tick_window_filters_before_collect(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:49:59Z",
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:30Z",
                "2025-01-02T20:51:00Z",
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [1, 2, 2, 1],
            "price_ticks": [83996, 84000, 84008, 84004],
            "size": [1, 2, 3, 4],
        },
    )

    out = collect_tick_window(
        path,
        start_utc="2025-01-02T20:50:00Z",
        end_utc="2025-01-02T20:51:00Z",
    )

    assert out.height == 2
    assert out["price_ticks"].to_list() == [84000, 84008]


def test_ticks_to_minute_bars_aggregates_price_and_volume(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": ["2025-01-02T20:50:00Z", "2025-01-02T20:50:30Z", "2025-01-02T20:51:00Z"],
            "intra_ts_rank": [0, 0, 0],
            "side": [2, 1, 2],
            "price_ticks": [84000, 84008, 84004],
            "size": [2, 3, 4],
        },
    )

    bars = ticks_to_minute_bars(
        scan_tick_data(path),
        start_utc="2025-01-02T20:50:00Z",
        end_utc="2025-01-02T20:52:00Z",
    )

    assert bars.height == 2
    first = bars.row(0, named=True)
    assert first["Open"] == 21000.0
    assert first["High"] == 21002.0
    assert first["Low"] == 21000.0
    assert first["Close"] == 21002.0
    assert first["Volume"] == 5
