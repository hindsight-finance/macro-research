from pathlib import Path

import polars as pl
import pytest

from tick_density import (
    MACRO_TICK_DENSITY_COLUMNS,
    build_macro_tick_density,
    write_macro_tick_density,
)


def _write_ticks(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)


def test_build_macro_tick_density_counts_utc_macro_minutes_and_side_totals(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:49:59Z",  # before 15:50 ET macro
                "2025-01-02T20:50:00Z",  # 15:50 ET
                "2025-01-02T20:50:30Z",
                "2025-01-02T20:51:00Z",  # 15:51 ET
                "2025-01-02T20:59:59Z",  # 15:59 ET
                "2025-01-02T21:00:00Z",  # outside macro
            ],
            "intra_ts_rank": [0, 0, 1, 0, 0, 0],
            "side": [2, 2, 1, 0, 1, 2],
            "price_ticks": [84000, 84004, 84008, 84012, 84016, 84020],
            "size": [9, 2, 3, 4, 5, 6],
        },
    )

    out = build_macro_tick_density(path).collect(engine="streaming")

    assert out.columns == MACRO_TICK_DENSITY_COLUMNS
    assert out.select("datetime_utc").to_series().dt.time().to_list() == [
        __import__("datetime").time(20, 50),
        __import__("datetime").time(20, 51),
        __import__("datetime").time(20, 59),
    ]
    assert out.select("macro_minute_index").to_series().to_list() == [0, 1, 9]
    assert out.select("tick_count").to_series().to_list() == [2, 1, 1]
    assert out.select("total_size").to_series().to_list() == [5, 4, 5]
    assert out.select("buy_ticks").to_series().to_list() == [1, 0, 0]
    assert out.select("sell_ticks").to_series().to_list() == [1, 0, 1]
    assert out.select("none_ticks").to_series().to_list() == [0, 1, 0]


def test_build_macro_tick_density_uses_utc_only_and_handles_dst_macro_hour(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",  # winter: 15:50 ET
                "2025-07-02T19:50:00Z",  # summer: 15:50 ET
            ],
            "intra_ts_rank": [0, 0],
            "side": [2, 1],
            "price_ticks": [84000, 84004],
            "size": [2, 3],
        },
    )

    out = build_macro_tick_density(path).collect(engine="streaming")

    assert "datetime_et" not in out.columns
    assert out.select("datetime_utc").to_series().dt.hour().to_list() == [20, 19]
    assert out.select("macro_minute_index").to_series().to_list() == [0, 0]


def test_build_macro_tick_density_requires_tick_schema(tmp_path: Path):
    path = tmp_path / "bad.parquet"
    pl.DataFrame(
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "side": [2],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    with pytest.raises(ValueError, match="Missing tick columns"):
        build_macro_tick_density(path)


def test_write_macro_tick_density_writes_parquet_from_lazy_plan(tmp_path: Path):
    input_path = tmp_path / "ticks.parquet"
    output_path = tmp_path / "density.parquet"
    _write_ticks(
        input_path,
        {
            "ts_event": ["2025-01-02T20:50:00Z", "2025-01-02T20:50:01Z"],
            "intra_ts_rank": [0, 0],
            "side": [2, 0],
            "price_ticks": [84000, 84004],
            "size": [2, 3],
        },
    )

    wrote = write_macro_tick_density(input_path, output_path)

    assert wrote == output_path
    out = pl.read_parquet(output_path)
    assert out.columns == MACRO_TICK_DENSITY_COLUMNS
    assert out.item(0, "tick_count") == 2
    assert out.item(0, "total_size") == 5
