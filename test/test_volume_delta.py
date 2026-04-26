from pathlib import Path

import polars as pl
import pytest

from volume_delta import build_macro_volume_delta_1m


def _write_ticks(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)


def test_build_macro_volume_delta_1m_requires_tick_schema(tmp_path: Path):
    path = tmp_path / "bad_ticks.parquet"
    pl.DataFrame({"ts_event": ["2025-01-02T20:50:00Z"]}).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC")
    ).write_parquet(path)

    with pytest.raises(ValueError, match="Missing tick columns"):
        build_macro_volume_delta_1m(path)


def test_build_macro_volume_delta_1m_computes_signed_size_and_diagnostics(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:10Z",
                "2025-01-02T20:50:20Z",
                "2025-01-02T20:50:30Z",
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [2, 1, 0, 2],
            "price_ticks": [84000, 84004, 84008, 84012],
            "size": [5, 3, 7, 2],
        },
    )

    out = build_macro_volume_delta_1m(path).collect(engine="streaming")

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["buy_size"] == 7
    assert row["sell_size"] == 3
    assert row["none_size"] == 7
    assert row["classified_size"] == 10
    assert row["total_size"] == 17
    assert row["volume_delta"] == 4
    assert row["delta_imbalance"] == pytest.approx(0.4)
    assert row["buy_ticks"] == 2
    assert row["sell_ticks"] == 1
    assert row["none_ticks"] == 1
    assert row["tick_delta"] == 1
    assert row["classified_share"] == pytest.approx(10 / 17)
