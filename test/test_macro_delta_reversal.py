from pathlib import Path

import polars as pl
import pytest

from features.macro_delta_reversal import (
    build_macro_delta_reversal,
    summarize_macro_delta_reversal,
    write_macro_delta_reversal,
)


def _globex_rows(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("trade_date_et").cast(pl.Date))


def _macro_rows(rows: list[dict]) -> pl.DataFrame:
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


def _m(date: str, minute: int, delta: int, classified: int | None = None, total: int | None = None) -> dict:
    classified_size = abs(delta) if classified is None else classified
    total_size = classified_size if total is None else total
    return {
        "datetime_utc": None,
        "trade_date_et": date,
        "macro_minute_index": minute,
        "volume_delta": delta,
        "classified_size": classified_size,
        "total_size": total_size,
    }


def test_build_macro_delta_reversal_requires_globex_schema():
    globex = pl.DataFrame({"trade_date_et": ["2025-01-02"]}).with_columns(pl.col("trade_date_et").cast(pl.Date))
    macro = _macro_rows([_m("2025-01-02", 59, -10)])

    with pytest.raises(ValueError, match="Missing Globex volume-delta columns"):
        build_macro_delta_reversal(globex, macro)


def test_build_macro_delta_reversal_requires_macro_schema():
    globex = _globex_rows([_g("2025-01-02", 0, 10)])
    macro = pl.DataFrame({"trade_date_et": ["2025-01-02"]}).with_columns(pl.col("trade_date_et").cast(pl.Date))

    with pytest.raises(ValueError, match="Missing macro volume-delta columns"):
        build_macro_delta_reversal(globex, macro)


def test_build_macro_delta_reversal_aggregates_core_windows_and_target():
    globex = _globex_rows(
        [
            _g("2025-01-02", 0, 10, 20, 25),
            _g("2025-01-02", 929, -3, 10, 11),
            _g("2025-01-02", 930, 7, 14, 14),
            _g("2025-01-02", 1309, 5, 10, 13),
            _g("2025-01-02", 1310, 99, 99, 99),
        ]
    )
    macro = _macro_rows(
        [
            _m("2025-01-02", 50, -4, 8, 9),
            _m("2025-01-02", 58, -2, 6, 6),
            _m("2025-01-02", 59, -9, 18, 20),
        ]
    )

    out = build_macro_delta_reversal(globex, macro)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["date"].isoformat() == "2025-01-02"
    assert row["eth_pre_rth_volume_delta"] == 7
    assert row["eth_pre_rth_classified_size"] == 30
    assert row["eth_pre_rth_total_size"] == 36
    assert row["eth_pre_rth_delta_imbalance"] == pytest.approx(7 / 30)
    assert row["rth_pre_macro_volume_delta"] == 12
    assert row["rth_pre_macro_classified_size"] == 24
    assert row["rth_pre_macro_total_size"] == 27
    assert row["day_pre_macro_volume_delta"] == 19
    assert row["day_pre_macro_classified_size"] == 54
    assert row["macro_pre59_volume_delta"] == -6
    assert row["macro_pre59_classified_size"] == 14
    assert row["macro_pre59_total_size"] == 15
    assert row["rth_plus_macro_pre59_volume_delta"] == 6
    assert row["rth_plus_macro_pre59_classified_size"] == 38
    assert row["day_plus_macro_pre59_volume_delta"] == 13
    assert row["day_plus_macro_pre59_classified_size"] == 68
    assert row["k359_volume_delta"] == -9
    assert row["k359_classified_size"] == 18
    assert row["k359_total_size"] == 20
    assert row["k359_delta_imbalance"] == pytest.approx(-9 / 18)
