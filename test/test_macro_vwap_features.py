from pathlib import Path

import polars as pl
import pytest

from features.macro_vwap_features import (
    INTRAMACRO_COLUMNS,
    PREMACRO_COLUMNS,
    SUMMARY_COLUMNS,
    build_macro_vwap_intramacro,
    build_macro_vwap_premacro,
    summarize_macro_vwap_features,
    write_macro_vwap_features,
)


def _write_ticks(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("intra_ts_rank").cast(pl.Int64),
        pl.col("price_ticks").cast(pl.Int64),
        pl.col("size").cast(pl.Int64),
    ).write_parquet(path)


def _tick(ts: str, price: float, size: int = 1, rank: int = 0) -> dict:
    return {
        "ts_event": ts,
        "intra_ts_rank": rank,
        "price_ticks": int(price * 4),
        "size": size,
    }


def test_build_macro_vwap_premacro_computes_tick_weighted_vwap_and_targets(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 1),
            _tick("2025-01-02T18:00:00Z", 102.0, 3),
            _tick("2025-01-02T20:00:00Z", 104.0, 2),
            _tick("2025-01-02T20:49:59Z", 106.0, 4),
            _tick("2025-01-02T20:50:00Z", 107.0, 1),
            _tick("2025-01-02T20:54:59Z", 110.0, 1),
            _tick("2025-01-02T20:55:00Z", 109.0, 1),
            _tick("2025-01-02T20:59:59Z", 105.0, 1),
            _tick("2025-01-02T21:00:00Z", 999.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert out.columns == PREMACRO_COLUMNS
    assert row["date"].isoformat() == "2025-01-02"
    assert row["rth_0930_vwap"] == pytest.approx((100 * 1 + 102 * 3 + 104 * 2 + 106 * 4) / 10)
    assert row["pm_1300_vwap"] == pytest.approx((102 * 3 + 104 * 2 + 106 * 4) / 9)
    assert row["h3pm_1500_vwap"] == pytest.approx((104 * 2 + 106 * 4) / 6)
    assert row["rth_0930_price"] == 106.0
    assert row["rth_0930_vwap_dist_points"] == pytest.approx(106.0 - row["rth_0930_vwap"])
    assert row["rth_0930_vwap_dist_bps"] == pytest.approx((106.0 / row["rth_0930_vwap"] - 1.0) * 10000.0)
    assert row["rth_0930_vwap_side"] == "above"
    assert row["target_1550_1554_points"] == 3.0
    assert row["target_1555_1559_points"] == -4.0
    assert row["target_1550_1559_points"] == -2.0
    assert row["target_1550_1554_state"] == "bullish"
    assert row["target_1555_1559_state"] == "bearish"
    assert row["target_1550_1559_state"] == "bearish"


def test_strict_checkpoint_boundaries_exclude_equal_timestamp_ticks(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:09Z", 101.0, 1),
            _tick("2025-01-02T20:50:10Z", 150.0, 1),
            _tick("2025-01-02T20:54:59Z", 102.0, 1),
            _tick("2025-01-02T20:55:00Z", 200.0, 1),
            _tick("2025-01-02T20:55:01Z", 103.0, 1),
            _tick("2025-01-02T20:59:59Z", 104.0, 1),
            _tick("2025-01-02T21:00:00Z", 300.0, 1),
        ],
    )

    out = build_macro_vwap_intramacro(path, barrier_path=None).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["macro_1550_at_1550_10s_price"] == 101.0
    assert row["macro_1550_at_1555_price"] == 102.0
    assert row["macro_1550_at_1600_price"] == 104.0
    assert row["eoii_1555_at_1600_vwap"] == pytest.approx((200.0 + 103.0 + 104.0) / 3)
    assert row["target_1550_1554_points"] == 2.0
    assert row["target_1555_1559_points"] == -96.0
    assert row["target_1550_1559_points"] == 4.0


def test_vwap_side_uses_one_tick_touch_threshold(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 1),
            _tick("2025-01-02T20:49:59Z", 100.25, 1),
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:54:59Z", 100.0, 1),
            _tick("2025-01-02T20:55:00Z", 100.0, 1),
            _tick("2025-01-02T20:59:59Z", 100.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["rth_0930_vwap_dist_points"] == pytest.approx(0.125)
    assert row["rth_0930_vwap_side"] == "touch"


def test_missing_and_zero_size_windows_produce_null_context(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 0),
            _tick("2025-01-02T20:50:00Z", 101.0, 1),
            _tick("2025-01-02T20:54:59Z", 102.0, 1),
            _tick("2025-01-02T20:55:00Z", 103.0, 1),
            _tick("2025-01-02T20:59:59Z", 104.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["rth_0930_vwap"] is None
    assert row["rth_0930_price"] == 100.0
    assert row["rth_0930_vwap_side"] is None
    assert row["pm_1300_vwap"] is None
    assert row["pm_1300_price"] is None


def test_build_macro_vwap_intramacro_joins_optional_barrier_context(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier_path = tmp_path / "barrier.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:09Z", 101.0, 1),
            _tick("2025-01-02T20:54:59Z", 102.0, 1),
            _tick("2025-01-02T20:55:00Z", 103.0, 1),
            _tick("2025-01-02T20:59:59Z", 104.0, 1),
        ],
    )
    pl.DataFrame(
        {
            "date": ["2025-01-02"],
            "macro_trend_state": ["bullish"],
            "barrier_extreme": ["low"],
            "barrier_price": [99.0],
            "barrier_time": [5],
            "barrier_first10": [True],
            "barrier_is_macro_extreme": [True],
            "barrier_holds": [True],
            "edge_case": [False],
        }
    ).with_columns(pl.col("date").str.to_date()).write_parquet(barrier_path)

    out = build_macro_vwap_intramacro(tick_path, barrier_path=barrier_path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert out.columns == INTRAMACRO_COLUMNS
    assert row["barrier_macro_trend_state"] == "bullish"
    assert row["barrier_first10"] is True
    assert row["barrier_holds"] is True


def test_same_timestamp_rows_use_intra_ts_rank_for_open_close_and_checkpoint(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-01-02T20:54:59Z", 140.0, 1, rank=2),
            _tick("2025-01-02T20:50:00Z", 110.0, 1, rank=2),
            _tick("2025-01-02T20:49:59Z", 103.0, 1, rank=3),
            _tick("2025-01-02T14:30:00Z", 100.0, 1, rank=0),
            _tick("2025-01-02T20:54:59Z", 130.0, 1, rank=1),
            _tick("2025-01-02T20:50:00Z", 105.0, 1, rank=1),
            _tick("2025-01-02T20:49:59Z", 101.0, 1, rank=1),
            _tick("2025-01-02T20:55:00Z", 150.0, 1, rank=1),
            _tick("2025-01-02T20:59:59Z", 160.0, 1, rank=1),
            _tick("2025-01-02T20:49:59Z", 102.0, 1, rank=2),
        ],
    )

    premacro = build_macro_vwap_premacro(path).collect(engine="streaming").row(0, named=True)
    intramacro = build_macro_vwap_intramacro(path, barrier_path=None).collect(engine="streaming").row(0, named=True)

    assert premacro["rth_0930_price"] == 103.0
    assert premacro["target_1550_1554_points"] == 35.0
    assert intramacro["macro_1550_at_1555_price"] == 140.0


def test_intramacro_barrier_context_validates_required_columns(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier_path = tmp_path / "barrier.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:54:59Z", 101.0, 1),
            _tick("2025-01-02T20:55:00Z", 102.0, 1),
            _tick("2025-01-02T20:59:59Z", 103.0, 1),
        ],
    )
    pl.DataFrame({"date": ["2025-01-02"], "macro_trend_state": ["bullish"]}).with_columns(
        pl.col("date").str.to_date()
    ).write_parquet(barrier_path)

    with pytest.raises(ValueError, match="Missing barrier columns"):
        build_macro_vwap_intramacro(tick_path, barrier_path=barrier_path).collect(engine="streaming")


def test_summer_dst_uses_new_york_market_time(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        [
            _tick("2025-07-02T13:30:00Z", 100.0, 1),
            _tick("2025-07-02T19:49:59Z", 101.0, 1),
            _tick("2025-07-02T19:50:00Z", 102.0, 1),
            _tick("2025-07-02T19:54:59Z", 103.0, 1),
            _tick("2025-07-02T19:55:00Z", 104.0, 1),
            _tick("2025-07-02T19:59:59Z", 105.0, 1),
        ],
    )

    out = build_macro_vwap_premacro(path).collect(engine="streaming")
    row = out.row(0, named=True)

    assert row["date"].isoformat() == "2025-07-02"
    assert row["rth_0930_price"] == 101.0
    assert row["target_1550_1559_points"] == 3.0


def test_summarize_macro_vwap_features_reports_side_bands_deciles_and_confluence():
    df = pl.DataFrame(
        {
            "date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "rth_0930_vwap_dist_bps": [-30, -15, -7, -3, -1, 0, 1, 3, 7, 15, 30, 40],
            "rth_0930_vwap_side": ["below", "below", "below", "below", "touch", "touch", "touch", "above", "above", "above", "above", "above"],
            "premacro_net_side_score": [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1],
            "target_1550_1554_points": [-1, -2, -3, -4, 0, 0, 1, 2, 3, 4, 5, 6],
            "target_1550_1554_state": ["bearish", "bearish", "bearish", "bearish", "neutral", "neutral", "bullish", "bullish", "bullish", "bullish", "bullish", "bullish"],
            "target_1555_1559_points": [1] * 12,
            "target_1555_1559_state": ["bullish"] * 12,
            "target_1550_1559_points": [1] * 12,
            "target_1550_1559_state": ["bullish"] * 12,
        }
    ).with_columns(pl.col("date").str.to_date())

    summary = summarize_macro_vwap_features(df, feature_set="premacro")

    assert summary.columns == SUMMARY_COLUMNS
    side = summary.filter(
        (pl.col("feature_name") == "rth_0930")
        & (pl.col("target_name") == "target_1550_1554")
        & (pl.col("scope") == "side")
        & (pl.col("bucket") == "below")
    ).row(0, named=True)
    assert side["sample_size"] == 4
    assert side["bearish_pct"] == 100.0
    assert summary.filter(pl.col("scope") == "fixed_bps_band").height > 0
    deciles = summary.filter(pl.col("scope") == "decile")
    assert deciles.height == 30
    assert set(deciles.select("target_name").to_series().to_list()) == {
        "target_1550_1554",
        "target_1555_1559",
        "target_1550_1559",
    }
    assert summary.filter(pl.col("scope") == "confluence").height > 0


def test_write_macro_vwap_features_writes_four_outputs(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    premacro_path = tmp_path / "premacro.parquet"
    premacro_summary_path = tmp_path / "premacro_summary.parquet"
    intramacro_path = tmp_path / "intramacro.parquet"
    intramacro_summary_path = tmp_path / "intramacro_summary.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T14:30:00Z", 100.0, 1),
            _tick("2025-01-02T18:00:00Z", 101.0, 1),
            _tick("2025-01-02T20:00:00Z", 102.0, 1),
            _tick("2025-01-02T20:49:59Z", 103.0, 1),
            _tick("2025-01-02T20:50:00Z", 104.0, 1),
            _tick("2025-01-02T20:50:09Z", 105.0, 1),
            _tick("2025-01-02T20:54:59Z", 106.0, 1),
            _tick("2025-01-02T20:55:00Z", 107.0, 1),
            _tick("2025-01-02T20:59:59Z", 108.0, 1),
        ],
    )

    wrote = write_macro_vwap_features(
        input_path=tick_path,
        premacro_output_path=premacro_path,
        premacro_summary_output_path=premacro_summary_path,
        intramacro_output_path=intramacro_path,
        intramacro_summary_output_path=intramacro_summary_path,
        barrier_path=None,
    )

    assert wrote == (premacro_path, premacro_summary_path, intramacro_path, intramacro_summary_path)
    assert pl.read_parquet(premacro_path).columns == PREMACRO_COLUMNS
    assert pl.read_parquet(intramacro_path).columns == INTRAMACRO_COLUMNS
    assert pl.read_parquet(premacro_summary_path).columns == SUMMARY_COLUMNS
    assert pl.read_parquet(intramacro_summary_path).columns == SUMMARY_COLUMNS
