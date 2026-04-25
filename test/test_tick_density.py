from pathlib import Path

import polars as pl
import pytest

from tick_density import (
    MACRO_TICK_DENSITY_COLUMNS,
    MACRO_5S_TICK_DENSITY_COLUMNS,
    build_macro_tick_density,
    build_macro_5s_tick_density,
    write_macro_tick_density,
    write_macro_5s_tick_density_files,
)


def _write_ticks(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC"))
    ).write_parquet(path)


def test_build_macro_tick_density_covers_1540_to_1610_et_with_actual_et_minute_index(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:39:59Z",
                "2025-01-02T20:40:00Z",
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:59:59Z",
                "2025-01-02T21:00:00Z",
                "2025-01-02T21:10:59Z",
                "2025-01-02T21:11:00Z",
            ],
            "intra_ts_rank": [0, 0, 0, 0, 0, 0, 0],
            "side": [2, 2, 1, 0, 1, 2, 2],
            "price_ticks": [84000, 84004, 84008, 84012, 84016, 84020, 84024],
            "size": [9, 2, 3, 4, 5, 6, 7],
        },
    )

    out = build_macro_tick_density(path).collect(engine="streaming")

    assert out.columns == MACRO_TICK_DENSITY_COLUMNS
    assert out.select("datetime_utc").to_series().dt.time().to_list() == [
        __import__("datetime").time(20, 40),
        __import__("datetime").time(20, 50),
        __import__("datetime").time(20, 59),
        __import__("datetime").time(21, 0),
        __import__("datetime").time(21, 10),
    ]
    assert out.select("macro_minute_index").to_series().to_list() == [40, 50, 59, 0, 10]
    assert out.select("tick_count").to_series().to_list() == [1, 1, 1, 1, 1]
    assert out.select("total_size").to_series().to_list() == [2, 3, 4, 5, 6]
    assert out.select("buy_ticks").to_series().to_list() == [1, 0, 0, 0, 1]
    assert out.select("sell_ticks").to_series().to_list() == [0, 1, 0, 1, 0]
    assert out.select("none_ticks").to_series().to_list() == [0, 0, 1, 0, 0]


def test_build_macro_tick_density_uses_utc_only_and_handles_dst_macro_hour(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:40:00Z",
                "2025-07-02T19:40:00Z",
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
    assert out.select("macro_minute_index").to_series().to_list() == [40, 40]


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


def test_build_macro_tick_density_requires_trade_order_schema(tmp_path: Path):
    path = tmp_path / "bad_missing_rank.parquet"
    pl.DataFrame(
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "side": [2],
            "size": [3],
        }
    ).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC"))
    ).write_parquet(path)

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


def test_build_macro_5s_tick_density_includes_empty_buckets_without_macro_minute_column(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:04Z",
                "2025-01-02T20:50:05Z",
                "2025-01-02T20:50:59Z",
                "2025-01-02T20:51:00Z",
            ],
            "intra_ts_rank": [0, 1, 0, 0, 0],
            "side": [2, 1, 0, 1, 2],
            "price_ticks": [84000, 84004, 84008, 84012, 84016],
            "size": [2, 3, 4, 5, 6],
        },
    )

    out = build_macro_5s_tick_density(path, macro_minute=50).collect(engine="streaming")

    assert out.columns == MACRO_5S_TICK_DENSITY_COLUMNS
    assert "macro_minute" not in out.columns
    assert out.height == 12
    assert out.select("bucket_index").to_series().to_list() == list(range(12))
    assert out.select("tick_count").to_series().to_list()[:3] == [2, 1, 0]
    assert out.item(0, "total_size") == 5
    assert out.item(0, "buy_ticks") == 1
    assert out.item(0, "sell_ticks") == 1
    assert out.item(0, "none_ticks") == 0
    assert out.item(1, "none_ticks") == 1
    assert out.item(2, "is_empty") is True
    assert out.item(11, "tick_count") == 1
    assert out.item(11, "total_size") == 5


def test_build_macro_5s_tick_density_handles_dst_and_multiple_days(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:59:00Z",
                "2025-07-02T19:59:05Z",
            ],
            "intra_ts_rank": [0, 0],
            "side": [2, 1],
            "price_ticks": [84000, 84004],
            "size": [2, 3],
        },
    )

    out = build_macro_5s_tick_density(path, macro_minute=59).collect(engine="streaming")

    assert out.height == 24
    assert out.select("datetime_utc").to_series().dt.hour().to_list()[0] == 20
    assert out.select("datetime_utc").to_series().dt.hour().to_list()[12] == 19
    assert out.select("tick_count").to_series().to_list()[0] == 1
    assert out.select("tick_count").to_series().to_list()[13] == 1


def test_write_macro_5s_tick_density_files_uses_minute_in_filename(tmp_path: Path):
    input_path = tmp_path / "ticks.parquet"
    output_dir = tmp_path / "out"
    _write_ticks(
        input_path,
        {
            "ts_event": ["2025-01-02T20:50:00Z", "2025-01-02T20:59:05Z"],
            "intra_ts_rank": [0, 0],
            "side": [2, 0],
            "price_ticks": [84000, 84004],
            "size": [2, 3],
        },
    )

    wrote = write_macro_5s_tick_density_files(input_path, output_dir, macro_minutes=(50, 59))

    assert wrote == [
        output_dir / "nq_macro_tick_density_1550_5s.parquet",
        output_dir / "nq_macro_tick_density_1559_5s.parquet",
    ]
    assert all(path.exists() for path in wrote)
    assert pl.read_parquet(wrote[0]).columns == MACRO_5S_TICK_DENSITY_COLUMNS
