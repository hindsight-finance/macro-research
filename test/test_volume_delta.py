from pathlib import Path

import polars as pl
import pytest

from volume_delta import build_globex_volume_delta_1m, build_macro_volume_delta_1m


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


def test_build_globex_volume_delta_1m_uses_1800_to_1700_et_trade_date(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T22:58:00Z",  # 17:58 ET, excluded
                "2025-01-02T23:00:00Z",  # 18:00 ET, trade date 2025-01-03
                "2025-01-03T20:50:00Z",  # 15:50 ET, same trade date
                "2025-01-03T21:59:00Z",  # 16:59 ET, included
                "2025-01-03T22:59:00Z",  # 17:59 ET, excluded
            ],
            "intra_ts_rank": [0, 0, 0, 0, 0],
            "side": [2, 2, 1, 2, 1],
            "price_ticks": [84000, 84004, 84008, 84012, 84016],
            "size": [1, 2, 3, 4, 5],
        },
    )

    out = build_globex_volume_delta_1m(path).collect(engine="streaming")

    assert out.select("trade_date_et").to_series().cast(pl.String).to_list() == [
        "2025-01-03",
        "2025-01-03",
        "2025-01-03",
    ]
    assert out.select("session_minute_index").to_series().to_list() == [0, 1310, 1379]
    assert out.select("volume_delta").to_series().to_list() == [2, -3, 4]


def test_build_globex_volume_delta_1m_handles_edt_session_edges(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-07-02T22:00:00Z",  # 18:00 EDT, trade date 2025-07-03
                "2025-07-03T20:59:00Z",  # 16:59 EDT, same trade date
                "2025-07-03T21:00:00Z",  # 17:00 EDT, excluded
                "2025-07-03T21:59:00Z",  # 17:59 EDT, excluded
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [2, 1, 1, 2],
            "price_ticks": [84000, 84004, 84008, 84012],
            "size": [6, 2, 9, 9],
        },
    )

    out = build_globex_volume_delta_1m(path).collect(engine="streaming")

    assert out.select("trade_date_et").to_series().cast(pl.String).to_list() == [
        "2025-07-03",
        "2025-07-03",
    ]
    assert out.select("session_minute_index").to_series().to_list() == [0, 1379]
    assert out.select("volume_delta").to_series().to_list() == [6, -2]
    assert out.columns == [
        "datetime_utc",
        "trade_date_et",
        "session_minute_index",
        "buy_size",
        "sell_size",
        "none_size",
        "classified_size",
        "total_size",
        "volume_delta",
        "delta_imbalance",
        "buy_ticks",
        "sell_ticks",
        "none_ticks",
        "tick_delta",
        "classified_share",
    ]
