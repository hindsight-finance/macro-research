from pathlib import Path
from datetime import date

import polars as pl
import pytest

from features.macro_tick_range_context import (
    build_macro_tick_range_context,
    summarize_macro_tick_range_context,
    write_macro_tick_range_context,
)


def _ticks(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", time_zone="UTC"))
    )


def _t(ts: str, rank: int, price_ticks: int) -> dict:
    return {"ts_event": ts, "intra_ts_rank": rank, "price_ticks": price_ticks}


def _fill_middle_minutes(day: str = "2025-01-02", hour: str = "20", price_ticks: int = 400) -> list[dict]:
    return [_t(f"{day}T{hour}:{minute:02d}:00Z", 0, price_ticks) for minute in range(51, 59)]


def _basic_macro_ticks() -> pl.DataFrame:
    return _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 10000),
            _t("2025-01-02T20:50:09Z", 0, 10020),
            _t("2025-01-02T20:50:59Z", 0, 10040),
            *_fill_middle_minutes("2025-01-02", "20", 10020),
            _t("2025-01-02T20:59:00Z", 0, 10010),
            _t("2025-01-02T20:59:09Z", 0, 10050),
            _t("2025-01-02T20:59:59Z", 0, 10030),
        ]
    )


def test_build_macro_tick_range_context_requires_tick_schema():
    bad = pl.DataFrame({"ts_event": ["2025-01-02T20:50:00Z"]}).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC")
    )

    with pytest.raises(ValueError, match="Missing tick columns"):
        build_macro_tick_range_context(bad)


def test_write_macro_tick_range_context_persists_outputs(tmp_path: Path):
    input_path = tmp_path / "ticks.parquet"
    output_path = tmp_path / "study.parquet"
    summary_path = tmp_path / "summary.parquet"
    _basic_macro_ticks().write_parquet(input_path)

    result = write_macro_tick_range_context(input_path, output_path, summary_path)

    assert result == (output_path, summary_path)
    assert pl.read_parquet(output_path).height > 0
    assert pl.read_parquet(summary_path).height > 0


def test_build_macro_tick_range_context_computes_first10_percentages_and_additive_extensions():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 400),  # 100
            _t("2025-01-02T20:50:04Z", 0, 408),  # 102
            _t("2025-01-02T20:50:09Z", 0, 396),  # 99 first10 range 3
            _t("2025-01-02T20:50:30Z", 0, 416),  # 104 candle high
            _t("2025-01-02T20:50:59Z", 0, 388),  # 97 candle low/range 7
            *_fill_middle_minutes("2025-01-02", "20", 400),
            _t("2025-01-02T20:59:00Z", 0, 392),  # 98
            _t("2025-01-02T20:59:09Z", 0, 420),  # 105
            _t("2025-01-02T20:59:59Z", 0, 384),  # 96 macro low/range 9
        ]
    )

    out = build_macro_tick_range_context(ticks)
    row = out.filter((pl.col("candle") == "k350") & (pl.col("window") == "first_10s")).row(0, named=True)

    assert row["window_range_points"] == pytest.approx(3.0)
    assert row["candle_range_points"] == pytest.approx(7.0)
    assert row["macro_range_points"] == pytest.approx(9.0)
    assert row["range_raw_pct_of_open"] == pytest.approx(3.0 / 100.0 * 100.0)
    assert row["range_pct_of_candle"] == pytest.approx(3.0 / 7.0 * 100.0)
    assert row["range_pct_of_macro"] == pytest.approx(3.0 / 9.0 * 100.0)
    assert row["candle_additive_high_extension_points"] == pytest.approx(2.0)
    assert row["candle_additive_low_extension_points"] == pytest.approx(2.0)
    assert row["candle_additive_total_extension_points"] == pytest.approx(4.0)
    assert row["macro_additive_high_extension_points"] == pytest.approx(3.0)
    assert row["macro_additive_low_extension_points"] == pytest.approx(3.0)
    assert row["macro_additive_total_extension_points"] == pytest.approx(6.0)
    assert row["window_range_points"] + row["candle_additive_total_extension_points"] == pytest.approx(row["candle_range_points"])
    assert row["window_range_points"] + row["macro_additive_total_extension_points"] == pytest.approx(row["macro_range_points"])


def test_build_macro_tick_range_context_computes_k359_macro_contribution_from_pre359():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 400),  # 100 pre low
            _t("2025-01-02T20:58:59Z", 0, 420),  # 105 pre high
            *_fill_middle_minutes("2025-01-02", "20", 408),
            _t("2025-01-02T20:59:00Z", 0, 416),
            _t("2025-01-02T20:59:10Z", 0, 428),  # 107 high -> +2
            _t("2025-01-02T20:59:59Z", 0, 392),  # 98 low -> +2
        ]
    )

    out = build_macro_tick_range_context(ticks)
    row = out.filter((pl.col("candle") == "k359") & (pl.col("window") == "full_candle")).row(0, named=True)

    assert row["k359_range_pct_of_macro"] == pytest.approx(9.0 / 9.0 * 100.0)
    assert row["k359_macro_additive_high_extension_from_pre359_points"] == pytest.approx(2.0)
    assert row["k359_macro_additive_low_extension_from_pre359_points"] == pytest.approx(2.0)
    assert row["k359_macro_additive_total_extension_from_pre359_points"] == pytest.approx(4.0)
    assert row["k359_macro_additive_total_extension_from_pre359_pct_of_macro"] == pytest.approx(4.0 / 9.0 * 100.0)


def test_build_macro_tick_range_context_excludes_incomplete_days():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 400),
            *_fill_middle_minutes("2025-01-02", "20", 402),
            _t("2025-01-02T20:59:00Z", 0, 404),
            _t("2025-01-03T20:50:00Z", 0, 400),
            _t("2025-01-03T20:50:10Z", 0, 404),
        ]
    )

    out = build_macro_tick_range_context(ticks)

    assert out["date"].unique().to_list() == [date(2025, 1, 2)]


def test_build_macro_tick_range_context_excludes_day_missing_middle_macro_minute():
    rows = [
        _t("2025-01-02T20:50:00Z", 0, 400),
        *[_t(f"2025-01-02T20:{minute:02d}:00Z", 0, 404) for minute in range(51, 60) if minute != 55],
    ]

    out = build_macro_tick_range_context(_ticks(rows))

    assert out.is_empty()


def test_build_macro_tick_range_context_empty_result_has_stable_schema():
    ticks = _ticks([_t("2025-01-02T20:50:00Z", 0, 400), _t("2025-01-02T20:59:00Z", 0, 404)])

    study = build_macro_tick_range_context(ticks)
    summary = summarize_macro_tick_range_context(study)

    assert study.is_empty()
    assert {
        "date",
        "candle",
        "window",
        "window_range_points",
        "candle_range_points",
        "macro_range_points",
        "k359_macro_additive_total_extension_from_pre359_pct_of_macro",
    }.issubset(study.columns)
    assert summary.is_empty()
    assert {
        "summary_type",
        "candle",
        "window",
        "threshold",
        "n_days",
        "median_metric",
        "decile",
    }.issubset(summary.columns)


def test_build_macro_tick_range_context_lazy_matches_dataframe_fixture():
    ticks = _basic_macro_ticks()

    eager = build_macro_tick_range_context(ticks)
    lazy = build_macro_tick_range_context(ticks.lazy())

    assert lazy.equals(eager)


def test_open_close_use_ts_event_then_rank_tiebreak():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:00Z", 1, 408),
            _t("2025-01-02T20:50:00Z", 0, 400),
            _t("2025-01-02T20:50:59Z", 0, 404),
            _t("2025-01-02T20:50:59Z", 1, 412),
            *_fill_middle_minutes("2025-01-02", "20", 408),
            _t("2025-01-02T20:59:00Z", 0, 416),
            _t("2025-01-02T20:59:59Z", 0, 420),
        ]
    )

    out = build_macro_tick_range_context(ticks)
    row = out.filter((pl.col("candle") == "k350") & (pl.col("window") == "full_candle")).row(0, named=True)

    assert row["candle_open"] == pytest.approx(100.0)
    assert row["candle_close"] == pytest.approx(103.0)


def test_dst_july_1550_et_maps_to_1950_utc():
    ticks = _ticks(
        [
            _t("2025-07-02T19:50:00Z", 0, 400),
            _t("2025-07-02T19:50:09Z", 0, 404),
            *_fill_middle_minutes("2025-07-02", "19", 408),
            _t("2025-07-02T19:59:00Z", 0, 408),
            _t("2025-07-02T19:59:59Z", 0, 412),
        ]
    )

    out = build_macro_tick_range_context(ticks)

    assert out.height > 0
    assert {row["candle"] for row in out.to_dicts()} == {"k350", "k359"}


def test_first10_includes_subsecond_09_999999999_excludes_10_000000000():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 400),
            _t("2025-01-02T20:50:09.999999999Z", 0, 420),
            _t("2025-01-02T20:50:10Z", 0, 440),
            _t("2025-01-02T20:50:59Z", 0, 404),
            *_fill_middle_minutes("2025-01-02", "20", 408),
            _t("2025-01-02T20:59:00Z", 0, 408),
            _t("2025-01-02T20:59:59Z", 0, 412),
        ]
    )

    out = build_macro_tick_range_context(ticks)
    first10 = out.filter((pl.col("candle") == "k350") & (pl.col("window") == "first_10s")).row(0, named=True)

    assert first10["window_high"] == pytest.approx(105.0)
    assert first10["window_tick_count"] == 2


def test_summarize_macro_tick_range_context_adds_threshold_and_decile_rows():
    rows = []
    for i in range(12):
        day = f"2025-01-{i + 1:02d}"
        base = 40000 + i * 100
        rows.extend(
            [
                _t(f"{day}T20:50:00Z", 0, base),
                _t(f"{day}T20:50:09Z", 0, base + 4 * (i + 1)),
                _t(f"{day}T20:50:59Z", 0, base + 80),
                *_fill_middle_minutes(day, "20", base + 40),
                _t(f"{day}T20:59:00Z", 0, base + 20),
                _t(f"{day}T20:59:59Z", 0, base + 100),
            ]
        )
    study = build_macro_tick_range_context(_ticks(rows))

    summary = summarize_macro_tick_range_context(study)

    assert "threshold_pct_of_candle" in summary["summary_type"].to_list()
    assert "threshold_pct_of_macro" in summary["summary_type"].to_list()
    assert "decile_range_raw_pct_of_open" in summary["summary_type"].to_list()
    first10 = summary.filter(
        (pl.col("summary_type") == "threshold_pct_of_candle")
        & (pl.col("candle") == "k350")
        & (pl.col("window") == "first_10s")
        & (pl.col("threshold") == 25.0)
    ).row(0, named=True)
    assert first10["n_days"] == 12
    assert first10["hit_rate"] is not None

    deciles = summary.filter(
        (pl.col("summary_type") == "decile_range_raw_pct_of_open")
        & (pl.col("candle") == "k350")
        & (pl.col("window") == "first_10s")
    )
    assert deciles.height == 10
    assert deciles["decile"].to_list() == list(range(1, 11))


def test_summarize_macro_tick_range_context_skips_deciles_when_less_than_10_unique():
    study = build_macro_tick_range_context(_basic_macro_ticks())

    summary = summarize_macro_tick_range_context(study)

    assert summary.filter(pl.col("summary_type").str.starts_with("decile_")).height == 0


def test_baseline_summary_n_days_counts_non_null_window_metric_only():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:10Z", 0, 400),
            _t("2025-01-02T20:50:59Z", 0, 404),
            *_fill_middle_minutes("2025-01-02", "20", 408),
            _t("2025-01-02T20:59:00Z", 0, 408),
            _t("2025-01-02T20:59:59Z", 0, 412),
        ]
    )
    study = build_macro_tick_range_context(ticks)

    summary = summarize_macro_tick_range_context(study)
    first5 = summary.filter(
        (pl.col("summary_type") == "window_baseline")
        & (pl.col("candle") == "k350")
        & (pl.col("window") == "first_5s")
    ).row(0, named=True)

    assert first5["n_days"] == 0
