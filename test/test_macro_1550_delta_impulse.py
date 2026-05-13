from pathlib import Path

import polars as pl
import pytest

from features.macro_1550_delta_impulse import (
    build_macro_1550_delta_impulse,
    summarize_macro_1550_delta_impulse,
    write_macro_1550_delta_impulse,
)


def _globex_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("trade_date_et").cast(pl.Date))


def _macro_5s_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("trade_date_et").cast(pl.Date))


def _g(date: str, idx: int, delta: int, classified: int | None = None, total: int | None = None) -> dict:
    classified_size = abs(delta) if classified is None else classified
    total_size = classified_size if total is None else total
    return {
        "datetime_utc": None,
        "trade_date_et": date,
        "session_minute_index": idx,
        "volume_delta": delta,
        "classified_size": classified_size,
        "total_size": total_size,
    }


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


def test_build_macro_1550_delta_impulse_requires_globex_schema():
    globex = pl.DataFrame({"trade_date_et": ["2025-01-02"]}).with_columns(pl.col("trade_date_et").cast(pl.Date))
    macro_5s = _macro_5s_rows([_s("2025-01-02", 0, 5)])

    with pytest.raises(ValueError, match="Missing Globex volume-delta columns"):
        build_macro_1550_delta_impulse(globex, macro_5s)


def test_build_macro_1550_delta_impulse_requires_macro_5s_schema():
    globex = _globex_rows([_g("2025-01-02", 930, 10)])
    macro_5s = pl.DataFrame({"trade_date_et": ["2025-01-02"]}).with_columns(pl.col("trade_date_et").cast(pl.Date))

    with pytest.raises(ValueError, match="Missing macro 5-second volume-delta columns"):
        build_macro_1550_delta_impulse(globex, macro_5s)


def test_write_macro_1550_delta_impulse_persists_outputs(tmp_path: Path):
    globex_path = tmp_path / "globex.parquet"
    macro_5s_path = tmp_path / "macro_5s.parquet"
    output_path = tmp_path / "study.parquet"
    summary_path = tmp_path / "summary.parquet"

    _globex_rows([_g("2025-01-02", 0, 10), _g("2025-01-02", 930, 20)]).write_parquet(globex_path)
    _macro_5s_rows([_s("2025-01-02", 0, -5), _s("2025-01-02", 1, -7)]).write_parquet(macro_5s_path)

    result = write_macro_1550_delta_impulse(globex_path, macro_5s_path, output_path, summary_path)

    assert result == (output_path, summary_path)
    assert pl.read_parquet(output_path).height == 1
    assert pl.read_parquet(summary_path).height > 0


def test_build_macro_1550_delta_impulse_aggregates_target_windows():
    globex = _globex_rows([_g("2025-01-02", 930, 10, 20, 20)])
    macro_5s = _macro_5s_rows(
        [
            _s("2025-01-02", 0, 1, 2, 2),
            _s("2025-01-02", 1, -3, 6, 6),
            _s("2025-01-02", 2, 5, 10, 10),
            _s("2025-01-02", 3, -7, 14, 14),
            _s("2025-01-02", 4, 9, 18, 18),
            _s("2025-01-02", 5, -11, 22, 22),
            _s("2025-01-02", 6, 13, 26, 26),
            _s("2025-01-02", 7, -15, 30, 30),
            _s("2025-01-02", 8, 17, 34, 34),
            _s("2025-01-02", 9, -19, 38, 38),
            _s("2025-01-02", 10, 21, 42, 42),
            _s("2025-01-02", 11, -23, 46, 46),
            _s("2025-01-02", 12, 999, 999, 999),
        ]
    )

    out = build_macro_1550_delta_impulse(globex, macro_5s)
    row = out.row(0, named=True)

    assert row["k350_00_04_volume_delta"] == 1
    assert row["k350_05_09_volume_delta"] == -3
    assert row["k350_00_09_volume_delta"] == -2
    assert row["k350_00_09_classified_size"] == 8
    assert row["k350_00_09_total_size"] == 8
    assert row["k350_00_09_delta_imbalance"] == pytest.approx(-2 / 8)
    assert row["k350_00_09_sign"] == -1
    assert row["k350_00_29_volume_delta"] == -6
    assert row["k350_00_59_volume_delta"] == -12
    assert row["k350_bucket_0_volume_delta"] == 1
    assert row["k350_bucket_11_volume_delta"] == -23
    assert "k350_bucket_12_volume_delta" not in out.columns


def test_build_macro_1550_delta_impulse_aggregates_predictor_boundaries():
    globex = _globex_rows(
        [
            _g("2025-01-02", 0, 10, 20, 21),
            _g("2025-01-02", 929, -3, 6, 7),
            _g("2025-01-02", 930, 5, 10, 11),
            _g("2025-01-02", 1309, 7, 14, 15),
            _g("2025-01-02", 1310, 999, 999, 999),
        ]
    )
    macro_5s = _macro_5s_rows([_s("2025-01-02", 0, -2), _s("2025-01-02", 1, -4)])

    out = build_macro_1550_delta_impulse(globex, macro_5s)
    row = out.row(0, named=True)

    assert row["date"].isoformat() == "2025-01-02"
    assert row["eth_only_pre350_volume_delta"] == 7
    assert row["eth_only_pre350_classified_size"] == 26
    assert row["eth_only_pre350_total_size"] == 28
    assert row["eth_only_pre350_delta_imbalance"] == pytest.approx(7 / 26)
    assert row["rth_only_pre350_volume_delta"] == 12
    assert row["rth_only_pre350_classified_size"] == 24
    assert row["rth_only_pre350_total_size"] == 26
    assert row["eth_rth_pre350_volume_delta"] == 19
    assert row["eth_rth_pre350_classified_size"] == 50
    assert row["eth_rth_pre350_total_size"] == 54


def test_build_macro_1550_delta_impulse_adds_signs_and_primary_relationships():
    globex = _globex_rows(
        [
            _g("2025-01-02", 0, 10),
            _g("2025-01-02", 930, 5),
            _g("2025-01-03", 0, -8),
            _g("2025-01-03", 930, 0, 10, 10),
        ]
    )
    macro_5s = _macro_5s_rows(
        [
            _s("2025-01-02", 0, -3),
            _s("2025-01-02", 1, -7),
            _s("2025-01-03", 0, 0, 12, 12),
            _s("2025-01-03", 1, 0, 12, 12),
        ]
    )

    out = build_macro_1550_delta_impulse(globex, macro_5s)
    day1 = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    day2 = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)

    assert day1["eth_only_pre350_sign"] == 1
    assert day1["rth_only_pre350_sign"] == 1
    assert day1["eth_rth_pre350_sign"] == 1
    assert day1["k350_00_09_sign"] == -1
    assert day1["eth_only_pre350_has_signal"] is True
    assert day1["eth_only_pre350_opposes_k350_00_09"] is True
    assert day1["eth_only_pre350_same_as_k350_00_09"] is False

    assert day2["rth_only_pre350_sign"] == 0
    assert day2["k350_00_09_sign"] == 0
    assert day2["eth_only_pre350_has_signal"] is False
    assert day2["eth_only_pre350_opposes_k350_00_09"] is False
    assert day2["eth_only_pre350_same_as_k350_00_09"] is False


def test_summarize_macro_1550_delta_impulse_adds_target_sign_rows():
    globex = _globex_rows(
        [
            _g("2025-01-02", 930, 10),
            _g("2025-01-03", 930, -10),
            _g("2025-01-04", 930, 0, 10, 10),
        ]
    )
    macro_5s = _macro_5s_rows(
        [
            _s("2025-01-02", 0, -4),
            _s("2025-01-02", 1, -6),
            _s("2025-01-03", 0, -3),
            _s("2025-01-03", 1, -7),
            _s("2025-01-04", 0, 5),
            _s("2025-01-04", 1, -5),
        ]
    )
    study = build_macro_1550_delta_impulse(globex, macro_5s)

    summary = summarize_macro_1550_delta_impulse(study)
    row = summary.filter(
        (pl.col("summary_type") == "target_sign")
        & (pl.col("predictor") == "rth_only_pre350")
        & (pl.col("target_window") == "k350_00_09")
    ).row(0, named=True)

    assert row["n_days"] == 3
    assert row["n_signal_days"] == 2
    assert row["opposite_count"] == 1
    assert row["opposite_rate"] == pytest.approx(0.5)
    assert row["same_count"] == 1
    assert row["same_rate"] == pytest.approx(0.5)
    assert row["zero_predictor_count"] == 1
    assert row["zero_target_count"] == 1
    assert row["mean_predictor_delta"] == pytest.approx(0.0)
    assert row["median_predictor_delta"] == pytest.approx(0.0)
    assert row["mean_target_delta"] == pytest.approx((-10 - 10 + 0) / 3)
    assert row["median_target_delta"] == pytest.approx(-10)
    assert row["mean_target_delta_when_predictor_positive"] == pytest.approx(-10)
    assert row["mean_target_delta_when_predictor_negative"] == pytest.approx(-10)
    assert row["median_target_delta_when_predictor_positive"] == pytest.approx(-10)
    assert row["median_target_delta_when_predictor_negative"] == pytest.approx(-10)
    assert row["target_p25_when_predictor_positive"] == pytest.approx(-10)
    assert row["target_p75_when_predictor_positive"] == pytest.approx(-10)
    assert row["pearson_corr_predictor_vs_target_delta"] is not None
