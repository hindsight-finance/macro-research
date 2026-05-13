from pathlib import Path

import polars as pl
import pytest

from features.macro_bucket_path import (
    build_macro_bucket_path,
    summarize_macro_bucket_path,
    write_macro_bucket_path,
)


def _macro_5s_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("trade_date_et").cast(pl.Date))


def _s(date: str, bucket: int, delta: int, classified: int | None = None, total: int | None = None) -> dict:
    classified_size = abs(delta) if classified is None else classified
    total_size = classified_size if total is None else total
    return {
        "datetime_utc": None,
        "trade_date_et": date,
        "macro_bucket_index": bucket,
        "volume_delta": delta,
        "classified_size": classified_size,
        "total_size": total_size,
    }


def _complete_candle(date: str, start_bucket: int, values: list[int]) -> list[dict]:
    return [_s(date, start_bucket + i, value, abs(value) + 10, abs(value) + 11) for i, value in enumerate(values)]


def test_build_macro_bucket_path_requires_macro_5s_schema():
    bad = pl.DataFrame({"trade_date_et": ["2025-01-02"]}).with_columns(pl.col("trade_date_et").cast(pl.Date))

    with pytest.raises(ValueError, match="Missing macro 5-second volume-delta columns"):
        build_macro_bucket_path(bad)


def test_write_macro_bucket_path_persists_outputs(tmp_path: Path):
    input_path = tmp_path / "macro_5s.parquet"
    output_path = tmp_path / "study.parquet"
    summary_path = tmp_path / "summary.parquet"
    rows = _complete_candle("2025-01-02", 0, [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12])
    rows += _complete_candle("2025-01-02", 108, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12])
    _macro_5s_rows(rows).write_parquet(input_path)

    result = write_macro_bucket_path(input_path, output_path, summary_path)

    assert result == (output_path, summary_path)
    assert pl.read_parquet(output_path).height == 2
    assert pl.read_parquet(summary_path).height > 0


def test_build_macro_bucket_path_maps_candles_and_named_windows():
    rows = []
    rows += _complete_candle("2025-01-02", 0, [1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12])
    rows += _complete_candle("2025-01-02", 108, [-1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12])
    rows.append(_s("2025-01-02", 12, 999, 999, 999))
    rows.append(_s("2025-01-02", 107, 999, 999, 999))

    out = build_macro_bucket_path(_macro_5s_rows(rows))

    k350 = out.filter(pl.col("candle") == "k350").row(0, named=True)
    k359 = out.filter(pl.col("candle") == "k359").row(0, named=True)

    assert out.height == 2
    assert k350["bucket_count"] == 12
    assert k350["complete_candle"] is True
    assert k350["b0_volume_delta"] == 1
    assert k350["b11_volume_delta"] == 12
    assert k359["b0_volume_delta"] == -1
    assert k359["b11_volume_delta"] == -12
    assert "b12_volume_delta" not in out.columns

    assert k350["cum_00_04_volume_delta"] == 1
    assert k350["cum_00_09_volume_delta"] == 3
    assert k350["cum_00_14_volume_delta"] == 0
    assert k350["cum_00_59_volume_delta"] == 8
    assert k350["early_5s_volume_delta"] == 1
    assert k350["early_10s_volume_delta"] == 3
    assert k350["early_30s_volume_delta"] == 5
    assert k350["late_30s_volume_delta"] == 3
    assert k350["full_volume_delta"] == 8
    assert k350["early_10s_delta_imbalance"] == pytest.approx(3 / (11 + 12))
    assert k350["full_sign"] == 1


def test_build_macro_bucket_path_adds_path_diagnostics_and_relationships():
    rows = _complete_candle("2025-01-02", 0, [5, 5, -3, -4, 10, -2, -8, -1, -1, -1, -1, -1])
    out = build_macro_bucket_path(_macro_5s_rows(rows))
    row = out.row(0, named=True)

    assert row["sum_abs_bucket_delta"] == 42
    assert row["path_efficiency"] == pytest.approx(-2 / 42)
    assert row["early_10s_abs_flow_share"] == pytest.approx(10 / 42)
    assert row["max_abs_bucket_delta"] == 10
    assert row["max_abs_bucket_index"] == 4
    assert row["peak_abs_cum_delta"] == 13
    assert row["peak_abs_cum_bucket_index"] == 4
    assert row["max_favorable_cum_delta"] == 13
    assert row["max_adverse_cum_delta"] == -2
    assert row["cum_sign_flip_count"] == 1
    assert row["early_10s_continues_to_30s"] is True
    assert row["early_10s_fades_to_30s"] is False
    assert row["early_10s_continues_to_late30"] is False
    assert row["early_10s_fades_to_late30"] is True
    assert row["early_10s_continues_to_full"] is False
    assert row["early_10s_fades_to_full"] is True


def test_build_macro_bucket_path_relationships_require_nonzero_signs():
    rows = _complete_candle("2025-01-02", 0, [0, 0, 1, -1, 0, 0, 2, -2, 0, 0, 0, 0])
    out = build_macro_bucket_path(_macro_5s_rows(rows))
    row = out.row(0, named=True)

    assert row["early_10s_sign"] == 0
    assert row["full_sign"] == 0
    assert row["early_10s_continues_to_full"] is False
    assert row["early_10s_fades_to_full"] is False


def test_build_macro_bucket_path_counts_sign_flips_through_zero_and_nulls_zero_signal_extremes():
    rows = _complete_candle("2025-01-02", 0, [5, -5, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    rows += _complete_candle("2025-01-03", 0, [0] * 12)
    out = build_macro_bucket_path(_macro_5s_rows(rows))

    flip_row = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    zero_row = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)

    assert flip_row["cum_sign_flip_count"] == 1
    assert zero_row["max_abs_bucket_index"] is None
    assert zero_row["peak_abs_cum_bucket_index"] is None
    assert zero_row["max_favorable_cum_delta"] is None
    assert zero_row["max_adverse_cum_delta"] is None


def test_build_macro_bucket_path_adds_per_candle_deciles_and_categories():
    rows = []
    for i in range(20):
        day = f"2025-01-{i + 1:02d}"
        k350_early = i - 10
        k359_early = (i - 10) * 10
        rows += _complete_candle(day, 0, [k350_early, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
        rows += _complete_candle(day, 108, [k359_early, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5])

    out = build_macro_bucket_path(_macro_5s_rows(rows))

    k350 = out.filter(pl.col("candle") == "k350")
    k359 = out.filter(pl.col("candle") == "k359")
    assert set(k350["early_10s_raw_decile"].drop_nulls().unique().to_list()) == set(range(1, 11))
    assert set(k359["early_10s_raw_decile"].drop_nulls().unique().to_list()) == set(range(1, 11))
    assert k350.filter(pl.col("early_10s_raw_decile") <= 2)["early_10s_category"].unique().to_list() == ["strong_negative"]
    assert k350.filter(pl.col("early_10s_raw_decile") >= 9)["early_10s_category"].unique().to_list() == ["strong_positive"]
    assert set(k350["early_10s_abs_category"].drop_nulls().unique().to_list()) == {
        "low_abs_conviction",
        "mid_abs_conviction",
        "high_abs_conviction",
    }


def test_build_macro_bucket_path_skips_deciles_when_too_few_unique_values_and_falls_back_categories():
    rows = []
    for i in range(20):
        day = f"2025-02-{i + 1:02d}"
        rows += _complete_candle(day, 0, [5, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    rows += _complete_candle("2025-03-01", 0, [0] * 12)
    out = build_macro_bucket_path(_macro_5s_rows(rows))

    assert out["early_10s_raw_decile"].null_count() == out.height
    assert out["early_10s_imbalance_decile"].null_count() == out.height
    assert out["early_10s_abs_decile"].null_count() == out.height
    assert out.filter(pl.col("early_10s_sign") > 0)["early_10s_category"].unique().to_list() == ["weak_positive"]
    assert out.filter(pl.col("early_10s_sign") == 0)["early_10s_category"].unique().to_list() == ["neutral"]


def test_build_macro_bucket_path_deciles_ignore_nulls_and_keep_null_rows_null():
    rows = []
    for i in range(10):
        rows += _complete_candle(f"2025-04-{i + 1:02d}", 0, [i + 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    # Incomplete row missing early buckets yields null early_10s window components? no; absent buckets fill 0, so use null delta.
    rows.append(_s("2025-04-20", 0, None, 0, 0))
    out = build_macro_bucket_path(_macro_5s_rows(rows))

    null_row = out.filter(pl.col("date") == pl.date(2025, 4, 20)).row(0, named=True)
    assert null_row["early_10s_raw_decile"] is None
    assert set(out.filter(pl.col("date") != pl.date(2025, 4, 20))["early_10s_raw_decile"].to_list()) == set(range(1, 11))
