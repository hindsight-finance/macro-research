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
