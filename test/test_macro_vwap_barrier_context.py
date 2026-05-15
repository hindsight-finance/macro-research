from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from features.macro_vwap_barrier_context import (
    MACRO_VWAP_BARRIER_CONTEXT_COLUMNS,
    MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS,
    build_macro_vwap_barrier_context,
    classify_constructive_side,
    summarize_macro_vwap_barrier_context,
    write_macro_vwap_barrier_context,
)


def _tick(ts: str, price: float, size: int = 1, rank: int = 0) -> dict:
    return {"ts_event": ts, "intra_ts_rank": rank, "price_ticks": int(price * 4), "size": size}


def _write_ticks(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("intra_ts_rank").cast(pl.Int64),
        pl.col("price_ticks").cast(pl.Int64),
        pl.col("size").cast(pl.Int64),
    ).write_parquet(path)


def _barrier_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "macro_trend_state": ["bullish", "bearish"],
            "barrier_extreme": ["low", "high"],
            "barrier_price": [99.0, 201.0],
            "barrier_time": [5, 8],
            "barrier_first10": [True, True],
            "barrier_is_macro_extreme": [True, False],
            "barrier_holds": [True, False],
            "edge_case": [False, True],
        }
    ).with_columns(pl.col("date").str.to_date())


def _vwap_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "macro_1550_at_1550_10s_vwap_side": ["above", "below"],
            "macro_1550_at_1550_10s_vwap_dist_points": [1.0, -1.25],
            "macro_1550_at_1550_10s_vwap_dist_bps": [10.0, -12.0],
            "macro_1550_at_1555_vwap_side": ["above", "above"],
            "macro_1550_at_1555_vwap_dist_points": [2.0, 3.0],
            "macro_1550_at_1555_vwap_dist_bps": [20.0, 30.0],
            "target_1550_1554_points": [4.0, -3.0],
            "target_1550_1554_sign": [1, -1],
            "target_1550_1554_state": ["bullish", "bearish"],
            "target_1555_1559_points": [3.0, -4.0],
            "target_1555_1559_sign": [1, -1],
            "target_1555_1559_state": ["bullish", "bearish"],
            "target_1550_1559_points": [7.0, -7.0],
            "target_1550_1559_sign": [1, -1],
            "target_1550_1559_state": ["bullish", "bearish"],
        }
    ).with_columns(pl.col("date").str.to_date())


def test_classify_constructive_side_is_direction_aware():
    assert classify_constructive_side("bullish", "above") == "constructive"
    assert classify_constructive_side("bullish", "touch") == "touch"
    assert classify_constructive_side("bullish", "below") == "wrong"
    assert classify_constructive_side("bearish", "below") == "constructive"
    assert classify_constructive_side("bearish", "above") == "wrong"
    assert classify_constructive_side("neutral", "above") == "unknown"
    assert classify_constructive_side("bullish", None) == "unknown"


def test_build_context_adds_vwap_barrier_and_wrong_side_metrics(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:05Z", 99.0, 1),
            _tick("2025-01-02T20:50:10Z", 101.0, 1),
            _tick("2025-01-02T20:50:30Z", 98.0, 1),
            _tick("2025-01-02T20:50:59Z", 98.0, 1),
            _tick("2025-01-02T20:54:59Z", 104.0, 1),
            _tick("2025-01-02T20:55:00Z", 105.0, 1),
            _tick("2025-01-02T20:59:59Z", 108.0, 1),
            _tick("2025-01-03T20:50:00Z", 200.0, 1),
            _tick("2025-01-03T20:50:08Z", 201.0, 1),
            _tick("2025-01-03T20:50:20Z", 203.0, 1),
            _tick("2025-01-03T20:50:59Z", 202.0, 1),
            _tick("2025-01-03T20:54:59Z", 198.0, 1),
            _tick("2025-01-03T20:55:00Z", 197.0, 1),
            _tick("2025-01-03T20:59:59Z", 193.0, 1),
        ],
    )

    out = build_macro_vwap_barrier_context(_barrier_frame(), _vwap_frame(), tick_path)

    assert out.columns == MACRO_VWAP_BARRIER_CONTEXT_COLUMNS
    bull = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    assert bull["vwap_10s_constructive"] == "constructive"
    assert bull["barrier_first10_and_vwap_constructive"] is True
    assert bull["closed_wrong_side_1550"] is True
    assert bull["closed_wrong_side_more_than_1tick"] is True
    assert bull["worst_wrong_side_dist_points"] > 0
    assert bull["seconds_wrong_side_vwap"] == 35.0
    assert bull["wrong_side_share_1550"] == pytest.approx(35.0 / 55.0)
    assert bull["target_1550_10s_1554_points"] == 3.0
    assert bull["target_1550_10s_1559_points"] == 7.0
    assert bull["target_1551_1559_points"] == 4.0

    bear = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)
    assert bear["vwap_10s_constructive"] == "constructive"
    assert bear["vwap_1555_constructive"] == "wrong"
    assert bear["vwap_context_10s_to_1555"] == "constructive_to_wrong"
    assert bear["vwap_side_at_1550_close"] == "above"
    assert bear["vwap_dist_at_1550_close_points"] < 0


def test_missing_tick_day_produces_null_metrics(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    _write_ticks(tick_path, [_tick("2025-01-02T20:50:00Z", 100.0, 1)])

    out = build_macro_vwap_barrier_context(_barrier_frame(), _vwap_frame(), tick_path)

    missing = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)
    assert missing["post_barrier_tick_count_1550"] is None
    assert missing["barrier_ts_utc"] == datetime(2025, 1, 3, 20, 50, 8, tzinfo=timezone.utc)
    assert missing["closed_wrong_side_1550"] is None
    assert missing["target_1551_1559_points"] is None


def test_barrier_timestamp_is_exact_even_without_tick_at_barrier(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier = _barrier_frame().filter(pl.col("date") == pl.date(2025, 1, 2)).with_columns(
        pl.lit(7).alias("barrier_time")
    )
    vwap = _vwap_frame().filter(pl.col("date") == pl.date(2025, 1, 2))
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:10Z", 101.0, 1),
            _tick("2025-01-02T20:50:59Z", 102.0, 1),
        ],
    )

    out = build_macro_vwap_barrier_context(barrier, vwap, tick_path)

    row = out.row(0, named=True)
    assert row["barrier_ts_utc"] == datetime(2025, 1, 2, 20, 50, 7, tzinfo=timezone.utc)
    assert row["post_barrier_tick_count_1550"] == 2


def test_dst_et_window_uses_market_timezone(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier = _barrier_frame().filter(pl.col("date") == pl.date(2025, 7, 1))
    if barrier.is_empty():
        barrier = pl.DataFrame(
            {
                "date": ["2025-07-01"],
                "macro_trend_state": ["bullish"],
                "barrier_extreme": ["low"],
                "barrier_price": [99.0],
                "barrier_time": [5],
                "barrier_first10": [True],
                "barrier_is_macro_extreme": [True],
                "barrier_holds": [True],
                "edge_case": [False],
            }
        ).with_columns(pl.col("date").str.to_date())
    vwap = _vwap_frame().head(1).with_columns(pl.date(2025, 7, 1).alias("date"))
    _write_ticks(
        tick_path,
        [
            _tick("2025-07-01T19:50:00Z", 100.0, 1),
            _tick("2025-07-01T19:50:05Z", 99.0, 1),
            _tick("2025-07-01T19:50:06Z", 101.0, 1),
            _tick("2025-07-01T19:59:59Z", 105.0, 1),
        ],
    )

    out = build_macro_vwap_barrier_context(barrier, vwap, tick_path)

    assert out.row(0, named=True)["post_barrier_tick_count_1550"] == 2


def test_summarize_context_reports_distribution_scopes(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    rows = []
    for day in range(1, 13):
        date = f"2025-01-{day:02d}"
        rows.extend([
            _tick(f"{date}T20:50:00Z", 100.0 + day, 1),
            _tick(f"{date}T20:50:05Z", 99.0 + day, 1),
            _tick(f"{date}T20:50:10Z", 101.0 + day, 1),
            _tick(f"{date}T20:50:59Z", 100.0 + day, 1),
            _tick(f"{date}T20:54:59Z", 102.0 + day, 1),
            _tick(f"{date}T20:55:00Z", 103.0 + day, 1),
            _tick(f"{date}T20:59:59Z", 104.0 + day, 1),
        ])
    _write_ticks(tick_path, rows)
    barrier = pl.DataFrame(
        {
            "date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "macro_trend_state": ["bullish"] * 12,
            "barrier_extreme": ["low"] * 12,
            "barrier_price": [99.0] * 12,
            "barrier_time": [5] * 12,
            "barrier_first10": [True, False] * 6,
            "barrier_is_macro_extreme": [True] * 12,
            "barrier_holds": [True, True, False] * 4,
            "edge_case": [False] * 12,
        }
    ).with_columns(pl.col("date").str.to_date())
    vwap = pl.DataFrame(
        {
            "date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "macro_1550_at_1550_10s_vwap_side": ["above", "below", "touch"] * 4,
            "macro_1550_at_1550_10s_vwap_dist_points": [1.0, -1.0, 0.0] * 4,
            "macro_1550_at_1550_10s_vwap_dist_bps": [10.0, -10.0, 0.0] * 4,
            "macro_1550_at_1555_vwap_side": ["above", "below"] * 6,
            "macro_1550_at_1555_vwap_dist_points": [2.0, -2.0] * 6,
            "macro_1550_at_1555_vwap_dist_bps": [20.0, -20.0] * 6,
            "target_1550_1554_points": list(range(12)),
            "target_1550_1554_sign": [1] * 12,
            "target_1550_1554_state": ["bullish"] * 12,
            "target_1555_1559_points": list(range(12)),
            "target_1555_1559_sign": [1] * 12,
            "target_1555_1559_state": ["bullish"] * 12,
            "target_1550_1559_points": list(range(12)),
            "target_1550_1559_sign": [1] * 12,
            "target_1550_1559_state": ["bullish"] * 12,
        }
    ).with_columns(pl.col("date").str.to_date())

    context = build_macro_vwap_barrier_context(barrier, vwap, tick_path)
    summary = summarize_macro_vwap_barrier_context(context)

    assert summary.columns == MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS
    assert summary.filter(pl.col("scope") == "barrier_only").height > 0
    assert summary.filter(pl.col("scope") == "vwap_10s_only").height > 0
    assert summary.filter(pl.col("scope") == "wrong_side_close_bucket").height > 0
    assert summary.filter(pl.col("scope") == "wrong_side_share_decile").height == 40
    assert summary.filter(pl.col("scope") == "vwap_1555_decision").height > 0
    assert {
        "first10_true_constructive",
        "first10_false_wrong",
        "holds_true_touch",
        "holds_false_constructive",
    } <= set(summary.filter(pl.col("scope") == "barrier_vwap_10s")["bucket"].to_list())
    assert {
        "holds_true_constructive",
        "holds_true_wrong",
        "holds_false_constructive",
        "holds_false_wrong",
    } <= set(summary.filter(pl.col("scope") == "barrier_1555_context")["bucket"].to_list())


def test_write_macro_vwap_barrier_context_persists_outputs(tmp_path: Path):
    tick_path = tmp_path / "ticks.parquet"
    barrier_path = tmp_path / "barrier.parquet"
    vwap_path = tmp_path / "vwap.parquet"
    output_path = tmp_path / "context.parquet"
    summary_path = tmp_path / "summary.parquet"
    _write_ticks(
        tick_path,
        [
            _tick("2025-01-02T20:50:00Z", 100.0, 1),
            _tick("2025-01-02T20:50:05Z", 99.0, 1),
            _tick("2025-01-02T20:50:10Z", 101.0, 1),
            _tick("2025-01-02T20:54:59Z", 104.0, 1),
            _tick("2025-01-02T20:55:00Z", 105.0, 1),
            _tick("2025-01-02T20:59:59Z", 108.0, 1),
        ],
    )
    _barrier_frame().filter(pl.col("date") == pl.date(2025, 1, 2)).write_parquet(barrier_path)
    _vwap_frame().filter(pl.col("date") == pl.date(2025, 1, 2)).write_parquet(vwap_path)

    wrote = write_macro_vwap_barrier_context(tick_path, barrier_path, vwap_path, output_path, summary_path)

    assert wrote == (output_path, summary_path)
    assert pl.read_parquet(output_path).columns == MACRO_VWAP_BARRIER_CONTEXT_COLUMNS
    assert pl.read_parquet(summary_path).columns == MACRO_VWAP_BARRIER_CONTEXT_SUMMARY_COLUMNS
