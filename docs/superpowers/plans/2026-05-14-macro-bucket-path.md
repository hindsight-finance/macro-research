# Macro Bucket Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-second bucket path study for the 15:50 and 15:59 macro candles to classify early volume-delta conviction and measure continuation/fade through the candle.

**Architecture:** Add a standalone module, `features/macro_bucket_path.py`, that reads the existing macro 5-second volume-delta parquet, normalizes 15:50 and 15:59 candles into relative buckets `0..11`, computes bucket/cumulative/path diagnostics, and writes daily + summary parquet outputs. Tests use in-memory Polars fixtures and validate bucket mapping, path math, categories, summaries, and writer behavior.

**Tech Stack:** Python, Polars, pytest, parquet outputs, project virtualenv via `.venv/bin/python`.

---

## File Structure

- Create: `features/macro_bucket_path.py`
  - Constants, schema validation, candle specs, safe ratio/sign helpers.
  - `build_macro_bucket_path(macro_5s)` daily path table.
  - `summarize_macro_bucket_path(study)` long-form summaries.
  - `load_macro_5s_input(...)`, `write_macro_bucket_path(...)`, `main()`.
- Create: `test/test_macro_bucket_path.py`
  - In-memory fixtures for macro 5-second rows.
  - Tests for schema validation, mapping, cumulative path, diagnostics, categories, summaries, writer persistence.
- Create after runtime: `docs/reports/2026-05-14-macro-bucket-path-findings.md`
- Runtime outputs, not necessarily committed:
  - `outputs/nq_macro_bucket_path.parquet`
  - `outputs/nq_macro_bucket_path_summary.parquet`

---

### Task 1: Add module skeleton, schema validation, writer

**Files:**
- Create: `features/macro_bucket_path.py`
- Create: `test/test_macro_bucket_path.py`

- [ ] **Step 1: Write failing schema/writer tests**

Create `test/test_macro_bucket_path.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_requires_macro_5s_schema test/test_macro_bucket_path.py::test_write_macro_bucket_path_persists_outputs -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'features.macro_bucket_path'`.

- [ ] **Step 3: Write minimal implementation**

Create `features/macro_bucket_path.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_bucket_path.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_bucket_path_summary.parquet")

MACRO_5S_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "classified_size",
    "total_size",
}

CANDLE_SPECS = {
    "k350": (0, 11),
    "k359": (108, 119),
}
RELATIVE_BUCKETS = list(range(12))


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_inputs(macro_5s: pl.DataFrame) -> None:
    missing = _missing_columns(macro_5s, MACRO_5S_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Missing macro 5-second volume-delta columns: {missing}")


def _safe_ratio_expr(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def _sign_expr(column: str) -> pl.Expr:
    return pl.when(pl.col(column) > 0).then(1).when(pl.col(column) < 0).then(-1).otherwise(0)


def _build_candle_rows(macro_5s: pl.DataFrame, candle: str, start: int, end: int) -> pl.DataFrame:
    filtered = macro_5s.filter(pl.col("macro_bucket_index").is_between(start, end)).with_columns(
        pl.lit(candle).alias("candle"),
        (pl.col("macro_bucket_index") - start).cast(pl.Int8).alias("relative_bucket"),
    )
    base = filtered.group_by("trade_date_et", "candle").agg(
        pl.len().alias("bucket_count"),
        (pl.len() == 12).alias("complete_candle"),
    )
    out = base
    for bucket in RELATIVE_BUCKETS:
        one = filtered.filter(pl.col("relative_bucket") == bucket).select(
            "trade_date_et",
            "candle",
            pl.col("volume_delta").alias(f"b{bucket}_volume_delta"),
            pl.col("classified_size").alias(f"b{bucket}_classified_size"),
            pl.col("total_size").alias(f"b{bucket}_total_size"),
        )
        one = one.with_columns(
            _safe_ratio_expr(pl.col(f"b{bucket}_volume_delta"), pl.col(f"b{bucket}_classified_size")).alias(
                f"b{bucket}_delta_imbalance"
            )
        )
        out = out.join(one, on=["trade_date_et", "candle"], how="left")
    sign_exprs = [_sign_expr(f"b{bucket}_volume_delta").alias(f"b{bucket}_sign") for bucket in RELATIVE_BUCKETS]
    return out.with_columns(sign_exprs).rename({"trade_date_et": "date"})


def build_macro_bucket_path(macro_5s: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(macro_5s)
    frames = [_build_candle_rows(macro_5s, candle, start, end) for candle, (start, end) in CANDLE_SPECS.items()]
    return pl.concat(frames, how="diagonal_relaxed").sort(["date", "candle"])


def summarize_macro_bucket_path(study: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for candle in sorted(study["candle"].unique().to_list()):
        subset = study.filter(pl.col("candle") == candle)
        rows.append({"summary_type": "candle_baseline", "candle": candle, "n_days": subset.height})
    return pl.DataFrame(rows, infer_schema_length=None)


def load_macro_5s_input(path: str | Path = MACRO_5S_INPUT_PATH) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_macro_bucket_path(
    input_path: str | Path = MACRO_5S_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    macro_5s = load_macro_5s_input(input_path)
    study = build_macro_bucket_path(macro_5s)
    summary = summarize_macro_bucket_path(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not MACRO_5S_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_5S_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_bucket_path()
    print(f"[OK] Wrote macro bucket path -> {output}")
    print(f"[OK] Wrote macro bucket path summary -> {summary_output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_requires_macro_5s_schema test/test_macro_bucket_path.py::test_write_macro_bucket_path_persists_outputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_bucket_path.py test/test_macro_bucket_path.py
git commit -m "feat: add macro bucket path skeleton"
```

---

### Task 2: Add candle mapping and cumulative/named windows

**Files:**
- Modify: `features/macro_bucket_path.py`
- Modify: `test/test_macro_bucket_path.py`

- [ ] **Step 1: Write failing mapping/window test**

Append to `test/test_macro_bucket_path.py`:

```python

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
```

- [ ] **Step 2: Run test to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_maps_candles_and_named_windows -q
```

Expected: FAIL with missing `cum_00_04_volume_delta`.

- [ ] **Step 3: Implement cumulative and named windows**

In `features/macro_bucket_path.py`, add constants below `RELATIVE_BUCKETS`:

```python
CUMULATIVE_WINDOWS = {f"cum_00_{end * 5 + 4:02d}": (0, end) for end in RELATIVE_BUCKETS}
NAMED_WINDOWS = {
    "early_5s": (0, 0),
    "early_10s": (0, 1),
    "early_30s": (0, 5),
    "late_30s": (6, 11),
    "full": (0, 11),
}
```

Add helper below `_sign_expr`:

```python
def _add_window_columns(frame: pl.DataFrame, prefix: str, start: int, end: int) -> pl.DataFrame:
    delta_cols = [pl.col(f"b{bucket}_volume_delta").fill_null(0) for bucket in range(start, end + 1)]
    classified_cols = [pl.col(f"b{bucket}_classified_size").fill_null(0) for bucket in range(start, end + 1)]
    total_cols = [pl.col(f"b{bucket}_total_size").fill_null(0) for bucket in range(start, end + 1)]
    return frame.with_columns(
        pl.sum_horizontal(delta_cols).alias(f"{prefix}_volume_delta"),
        pl.sum_horizontal(classified_cols).alias(f"{prefix}_classified_size"),
        pl.sum_horizontal(total_cols).alias(f"{prefix}_total_size"),
    ).with_columns(
        _safe_ratio_expr(pl.col(f"{prefix}_volume_delta"), pl.col(f"{prefix}_classified_size")).alias(
            f"{prefix}_delta_imbalance"
        ),
        _sign_expr(f"{prefix}_volume_delta").alias(f"{prefix}_sign"),
    )


def _add_path_windows(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for prefix, (start, end) in CUMULATIVE_WINDOWS.items():
        out = _add_window_columns(out, prefix, start, end)
    for prefix, (start, end) in NAMED_WINDOWS.items():
        out = _add_window_columns(out, prefix, start, end)
    return out
```

Update return in `_build_candle_rows`:

```python
    out = out.with_columns(sign_exprs).rename({"trade_date_et": "date"})
    return _add_path_windows(out)
```

- [ ] **Step 4: Run test to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_maps_candles_and_named_windows -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_bucket_path.py test/test_macro_bucket_path.py
git commit -m "feat: add macro bucket path windows"
```

---

### Task 3: Add path diagnostics and continuation/fade flags

**Files:**
- Modify: `features/macro_bucket_path.py`
- Modify: `test/test_macro_bucket_path.py`

- [ ] **Step 1: Write failing diagnostics test**

Append to `test/test_macro_bucket_path.py`:

```python

def test_build_macro_bucket_path_adds_path_diagnostics_and_relationships():
    rows = _complete_candle("2025-01-02", 0, [5, 5, -3, -4, 10, -2, -8, -1, -1, -1, -1, -1])
    out = build_macro_bucket_path(_macro_5s_rows(rows))
    row = out.row(0, named=True)

    assert row["sum_abs_bucket_delta"] == 42
    assert row["path_efficiency"] == pytest.approx(-3 / 42)
    assert row["early_10s_abs_flow_share"] == pytest.approx(10 / 42)
    assert row["max_abs_bucket_delta"] == 10
    assert row["max_abs_bucket_index"] == 4
    assert row["peak_abs_cum_delta"] == 13
    assert row["peak_abs_cum_bucket_index"] == 4
    assert row["max_favorable_cum_delta"] == 13
    assert row["max_adverse_cum_delta"] == -3
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
```

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_adds_path_diagnostics_and_relationships test/test_macro_bucket_path.py::test_build_macro_bucket_path_relationships_require_nonzero_signs -q
```

Expected: FAIL with missing `sum_abs_bucket_delta`.

- [ ] **Step 3: Implement diagnostics and flags**

In `features/macro_bucket_path.py`, add helpers:

```python
def _add_path_diagnostics(frame: pl.DataFrame) -> pl.DataFrame:
    bucket_delta_cols = [pl.col(f"b{bucket}_volume_delta").fill_null(0) for bucket in RELATIVE_BUCKETS]
    abs_bucket_exprs = [pl.col(f"b{bucket}_volume_delta").fill_null(0).abs() for bucket in RELATIVE_BUCKETS]
    cum_cols = [f"cum_00_{bucket * 5 + 4:02d}_volume_delta" for bucket in RELATIVE_BUCKETS]
    cum_values = [pl.col(col).fill_null(0) for col in cum_cols]
    abs_cum_values = [pl.col(col).fill_null(0).abs() for col in cum_cols]

    out = frame.with_columns(
        pl.sum_horizontal(abs_bucket_exprs).alias("sum_abs_bucket_delta"),
        pl.max_horizontal(abs_bucket_exprs).alias("max_abs_bucket_delta"),
        pl.max_horizontal(abs_cum_values).alias("peak_abs_cum_delta"),
    ).with_columns(
        _safe_ratio_expr(pl.col("full_volume_delta"), pl.col("sum_abs_bucket_delta")).alias("path_efficiency"),
        _safe_ratio_expr(pl.col("early_10s_volume_delta").abs(), pl.col("sum_abs_bucket_delta")).alias(
            "early_10s_abs_flow_share"
        ),
    )

    max_bucket_expr = None
    peak_cum_expr = None
    for bucket in RELATIVE_BUCKETS:
        bucket_is_max = pl.col(f"b{bucket}_volume_delta").fill_null(0).abs() == pl.col("max_abs_bucket_delta")
        cum_col = f"cum_00_{bucket * 5 + 4:02d}_volume_delta"
        cum_is_peak = pl.col(cum_col).fill_null(0).abs() == pl.col("peak_abs_cum_delta")
        max_bucket_expr = pl.when(bucket_is_max).then(bucket) if max_bucket_expr is None else max_bucket_expr.when(bucket_is_max).then(bucket)
        peak_cum_expr = pl.when(cum_is_peak).then(bucket) if peak_cum_expr is None else peak_cum_expr.when(cum_is_peak).then(bucket)
    out = out.with_columns(
        max_bucket_expr.otherwise(None).alias("max_abs_bucket_index"),
        peak_cum_expr.otherwise(None).alias("peak_abs_cum_bucket_index"),
    )

    early_sign = pl.col("early_10s_sign")
    favorable = [pl.col(col).fill_null(0) * early_sign for col in cum_cols]
    out = out.with_columns(
        pl.max_horizontal(favorable).alias("max_favorable_cum_delta"),
        pl.min_horizontal(favorable).alias("max_adverse_cum_delta"),
    )

    flip_exprs = []
    for prev_bucket, bucket in zip(RELATIVE_BUCKETS, RELATIVE_BUCKETS[1:]):
        prev_sign = pl.col(f"cum_00_{prev_bucket * 5 + 4:02d}_sign")
        curr_sign = pl.col(f"cum_00_{bucket * 5 + 4:02d}_sign")
        flip_exprs.append(((prev_sign != 0) & (curr_sign != 0) & (prev_sign != curr_sign)).cast(pl.Int64))
    return out.with_columns(pl.sum_horizontal(flip_exprs).alias("cum_sign_flip_count"))


def _add_continuation_flags(frame: pl.DataFrame) -> pl.DataFrame:
    comparisons = {
        "30s": "early_30s",
        "late30": "late_30s",
        "full": "full",
    }
    exprs = []
    early_sign = pl.col("early_10s_sign")
    for label, target in comparisons.items():
        target_sign = pl.col(f"{target}_sign")
        has_signal = (early_sign != 0) & (target_sign != 0)
        exprs.extend(
            [
                (has_signal & (early_sign == target_sign)).alias(f"early_10s_continues_to_{label}"),
                (has_signal & (early_sign == -target_sign)).alias(f"early_10s_fades_to_{label}"),
            ]
        )
    return frame.with_columns(exprs)
```

Update `_build_candle_rows` return:

```python
    out = out.with_columns(sign_exprs).rename({"trade_date_et": "date"})
    out = _add_path_windows(out)
    out = _add_path_diagnostics(out)
    return _add_continuation_flags(out)
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_adds_path_diagnostics_and_relationships test/test_macro_bucket_path.py::test_build_macro_bucket_path_relationships_require_nonzero_signs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_bucket_path.py test/test_macro_bucket_path.py
git commit -m "feat: add macro bucket path diagnostics"
```

---

### Task 4: Add per-candle conviction deciles and categories

**Files:**
- Modify: `features/macro_bucket_path.py`
- Modify: `test/test_macro_bucket_path.py`

- [ ] **Step 1: Write failing category tests**

Append to `test/test_macro_bucket_path.py`:

```python

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


def test_build_macro_bucket_path_skips_deciles_when_too_few_unique_values():
    rows = []
    for i in range(20):
        day = f"2025-02-{i + 1:02d}"
        rows += _complete_candle(day, 0, [5, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    out = build_macro_bucket_path(_macro_5s_rows(rows))

    assert out["early_10s_raw_decile"].null_count() == out.height
    assert out["early_10s_imbalance_decile"].null_count() == out.height
    assert out["early_10s_abs_decile"].null_count() == out.height
```

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_adds_per_candle_deciles_and_categories test/test_macro_bucket_path.py::test_build_macro_bucket_path_skips_deciles_when_too_few_unique_values -q
```

Expected: FAIL with missing `early_10s_raw_decile`.

- [ ] **Step 3: Implement deciles/categories**

In `features/macro_bucket_path.py`, add:

```python
def _rank_decile_expr(column: str, output: str) -> pl.Expr:
    return (((pl.col(column).rank(method="ordinal") - 1) * 10 / pl.len()).floor().cast(pl.Int64).clip(0, 9) + 1).alias(output)


def _add_decile_for_candle(frame: pl.DataFrame, candle: str, value_col: str, output_col: str) -> pl.DataFrame:
    subset = frame.filter(pl.col("candle") == candle)
    unique_count = subset.select(pl.col(value_col).n_unique()).item() if subset.height else 0
    if subset.height < 10 or unique_count < 10:
        values = subset.select("date", "candle").with_columns(pl.lit(None, dtype=pl.Int64).alias(output_col))
    else:
        values = subset.select("date", "candle", value_col).with_columns(_rank_decile_expr(value_col, output_col)).select(
            "date", "candle", output_col
        )
    return frame.join(values, on=["date", "candle"], how="left")


def _add_conviction_categories(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame.with_columns(pl.col("early_10s_volume_delta").abs().alias("early_10s_abs_delta"))
    for candle in CANDLE_SPECS:
        out = _add_decile_for_candle(out, candle, "early_10s_volume_delta", "early_10s_raw_decile")
        out = _add_decile_for_candle(out, candle, "early_10s_delta_imbalance", "early_10s_imbalance_decile")
        out = _add_decile_for_candle(out, candle, "early_10s_abs_delta", "early_10s_abs_decile")
    return out.with_columns(
        pl.when(pl.col("early_10s_raw_decile").is_in([1, 2]))
        .then(pl.lit("strong_negative"))
        .when(pl.col("early_10s_raw_decile").is_in([3, 4]))
        .then(pl.lit("weak_negative"))
        .when((pl.col("early_10s_raw_decile").is_in([5, 6])) | (pl.col("early_10s_sign") == 0))
        .then(pl.lit("neutral"))
        .when(pl.col("early_10s_raw_decile").is_in([7, 8]))
        .then(pl.lit("weak_positive"))
        .when(pl.col("early_10s_raw_decile").is_in([9, 10]))
        .then(pl.lit("strong_positive"))
        .otherwise(None)
        .alias("early_10s_category"),
        pl.when(pl.col("early_10s_abs_decile").is_in([9, 10]))
        .then(pl.lit("high_abs_conviction"))
        .when(pl.col("early_10s_abs_decile").is_between(4, 8))
        .then(pl.lit("mid_abs_conviction"))
        .when(pl.col("early_10s_abs_decile").is_between(1, 3))
        .then(pl.lit("low_abs_conviction"))
        .otherwise(None)
        .alias("early_10s_abs_category"),
    )
```

Update `build_macro_bucket_path`:

```python
    out = pl.concat(frames, how="diagonal_relaxed").sort(["date", "candle"])
    return _add_conviction_categories(out)
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_build_macro_bucket_path_adds_per_candle_deciles_and_categories test/test_macro_bucket_path.py::test_build_macro_bucket_path_skips_deciles_when_too_few_unique_values -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_bucket_path.py test/test_macro_bucket_path.py
git commit -m "feat: add macro bucket conviction categories"
```

---

### Task 5: Add long-form summaries

**Files:**
- Modify: `features/macro_bucket_path.py`
- Modify: `test/test_macro_bucket_path.py`

- [ ] **Step 1: Write failing summary test**

Append to `test/test_macro_bucket_path.py`:

```python

def test_summarize_macro_bucket_path_adds_expected_summary_types():
    rows = []
    for i in range(20):
        day = f"2025-03-{i + 1:02d}"
        early = i - 10
        rest = 20 if early > 0 else -20
        rows += _complete_candle(day, 0, [early, 0, 1, -1, 2, -2, rest, 0, 0, 0, 0, 0])
        rows += _complete_candle(day, 108, [-early, 0, 1, -1, 2, -2, -rest, 0, 0, 0, 0, 0])
    study = build_macro_bucket_path(_macro_5s_rows(rows))

    summary = summarize_macro_bucket_path(study)

    assert {"candle_baseline", "early_10s_category", "early_10s_raw_decile", "early_10s_imbalance_decile", "early_10s_abs_decile", "early_10s_abs_category"}.issubset(
        set(summary["summary_type"].to_list())
    )
    baseline = summary.filter((pl.col("summary_type") == "candle_baseline") & (pl.col("candle") == "k350")).row(0, named=True)
    assert baseline["n_days"] == 20
    assert baseline["median_full_delta"] is not None
    cat = summary.filter(
        (pl.col("summary_type") == "early_10s_category")
        & (pl.col("candle") == "k350")
        & (pl.col("early_10s_category") == "strong_positive")
    ).row(0, named=True)
    assert cat["n_days"] > 0
    assert cat["continue_to_full_rate"] is not None
    assert cat["median_late_30s_delta"] is not None
    assert cat["median_path_efficiency"] is not None
```

- [ ] **Step 2: Run test to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_summarize_macro_bucket_path_adds_expected_summary_types -q
```

Expected: FAIL because only `candle_baseline` exists.

- [ ] **Step 3: Implement summaries**

In `features/macro_bucket_path.py`, replace `summarize_macro_bucket_path` and add helpers:

```python
def _scalar(frame: pl.DataFrame, expr: pl.Expr) -> float | int | None:
    if frame.is_empty():
        return None
    return frame.select(expr).item()


def _base_summary_row(subset: pl.DataFrame, summary_type: str, candle: str, **labels: object) -> dict:
    signal = subset.filter((pl.col("early_10s_sign") != 0) & (pl.col("full_sign") != 0))
    n_signal_days = signal.height
    continue_full = subset.filter(pl.col("early_10s_continues_to_full")).height
    fade_full = subset.filter(pl.col("early_10s_fades_to_full")).height
    continue_30s = subset.filter(pl.col("early_10s_continues_to_30s")).height
    fade_30s = subset.filter(pl.col("early_10s_fades_to_30s")).height
    continue_late30 = subset.filter(pl.col("early_10s_continues_to_late30")).height
    fade_late30 = subset.filter(pl.col("early_10s_fades_to_late30")).height
    row = {
        "summary_type": summary_type,
        "candle": candle,
        "n_days": subset.height,
        "n_signal_days": n_signal_days,
        "continue_to_30s_count": continue_30s,
        "continue_to_30s_rate": _rate(continue_30s, n_signal_days),
        "fade_to_30s_count": fade_30s,
        "fade_to_30s_rate": _rate(fade_30s, n_signal_days),
        "continue_to_late30_count": continue_late30,
        "continue_to_late30_rate": _rate(continue_late30, n_signal_days),
        "fade_to_late30_count": fade_late30,
        "fade_to_late30_rate": _rate(fade_late30, n_signal_days),
        "continue_to_full_count": continue_full,
        "continue_to_full_rate": _rate(continue_full, n_signal_days),
        "fade_to_full_count": fade_full,
        "fade_to_full_rate": _rate(fade_full, n_signal_days),
        "mean_early_10s_delta": _scalar(subset, pl.col("early_10s_volume_delta").mean()),
        "median_early_10s_delta": _scalar(subset, pl.col("early_10s_volume_delta").median()),
        "mean_late_30s_delta": _scalar(subset, pl.col("late_30s_volume_delta").mean()),
        "median_late_30s_delta": _scalar(subset, pl.col("late_30s_volume_delta").median()),
        "mean_full_delta": _scalar(subset, pl.col("full_volume_delta").mean()),
        "median_full_delta": _scalar(subset, pl.col("full_volume_delta").median()),
        "full_p25": _scalar(subset, pl.col("full_volume_delta").quantile(0.25)),
        "full_p75": _scalar(subset, pl.col("full_volume_delta").quantile(0.75)),
        "mean_path_efficiency": _scalar(subset, pl.col("path_efficiency").mean()),
        "median_path_efficiency": _scalar(subset, pl.col("path_efficiency").median()),
        "mean_early_10s_abs_flow_share": _scalar(subset, pl.col("early_10s_abs_flow_share").mean()),
        "median_early_10s_abs_flow_share": _scalar(subset, pl.col("early_10s_abs_flow_share").median()),
        "mean_cum_sign_flip_count": _scalar(subset, pl.col("cum_sign_flip_count").mean()),
        "median_cum_sign_flip_count": _scalar(subset, pl.col("cum_sign_flip_count").median()),
    }
    row.update(labels)
    return row


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in keys} for row in rows]


def summarize_macro_bucket_path(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for candle in sorted(study["candle"].unique().to_list()):
        candle_df = study.filter(pl.col("candle") == candle)
        rows.append(_base_summary_row(candle_df, "candle_baseline", candle))
        for category in ["strong_negative", "weak_negative", "neutral", "weak_positive", "strong_positive"]:
            subset = candle_df.filter(pl.col("early_10s_category") == category)
            rows.append(_base_summary_row(subset, "early_10s_category", candle, early_10s_category=category))
        for category in ["low_abs_conviction", "mid_abs_conviction", "high_abs_conviction"]:
            subset = candle_df.filter(pl.col("early_10s_abs_category") == category)
            rows.append(_base_summary_row(subset, "early_10s_abs_category", candle, early_10s_abs_category=category))
        for decile in range(1, 11):
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_raw_decile") == decile),
                    "early_10s_raw_decile",
                    candle,
                    early_10s_raw_decile=decile,
                )
            )
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_imbalance_decile") == decile),
                    "early_10s_imbalance_decile",
                    candle,
                    early_10s_imbalance_decile=decile,
                )
            )
            rows.append(
                _base_summary_row(
                    candle_df.filter(pl.col("early_10s_abs_decile") == decile),
                    "early_10s_abs_decile",
                    candle,
                    early_10s_abs_decile=decile,
                )
            )
    return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)
```

- [ ] **Step 4: Run test to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py::test_summarize_macro_bucket_path_adds_expected_summary_types -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_bucket_path.py test/test_macro_bucket_path.py
git commit -m "feat: add macro bucket path summaries"
```

---

### Task 6: Verify, generate outputs, and write findings report

**Files:**
- Create: `docs/reports/2026-05-14-macro-bucket-path-findings.md`
- Runtime writes:
  - `outputs/nq_macro_bucket_path.parquet`
  - `outputs/nq_macro_bucket_path_summary.parquet`

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run related regression tests**

```bash
.venv/bin/python -m pytest test/test_macro_bucket_path.py test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Generate runtime outputs**

```bash
.venv/bin/python -m features.macro_bucket_path
```

Expected:

```text
[OK] Wrote macro bucket path -> outputs/nq_macro_bucket_path.parquet
[OK] Wrote macro bucket path summary -> outputs/nq_macro_bucket_path_summary.parquet
```

- [ ] **Step 4: Inspect key findings**

```bash
.venv/bin/python - <<'PY'
import polars as pl
study = pl.read_parquet("outputs/nq_macro_bucket_path.parquet")
summary = pl.read_parquet("outputs/nq_macro_bucket_path_summary.parquet")
print("study", study.shape)
print("summary", summary.shape)
print(summary.filter(pl.col("summary_type") == "candle_baseline").select([
    "candle", "n_days", "continue_to_full_rate", "fade_to_full_rate", "median_early_10s_delta", "median_full_delta", "median_path_efficiency", "median_cum_sign_flip_count"
]).sort("candle"))
print(summary.filter(pl.col("summary_type") == "early_10s_category").select([
    "candle", "early_10s_category", "n_days", "continue_to_full_rate", "fade_to_full_rate", "median_late_30s_delta", "median_full_delta", "median_path_efficiency"
]).sort(["candle", "early_10s_category"]))
print(summary.filter(pl.col("summary_type") == "early_10s_abs_category").select([
    "candle", "early_10s_abs_category", "n_days", "continue_to_full_rate", "fade_to_full_rate", "median_full_delta", "median_cum_sign_flip_count"
]).sort(["candle", "early_10s_abs_category"]))
PY
```

Expected: printed baseline, signed-category, and abs-category tables for `k350` and `k359`.

- [ ] **Step 5: Write findings report**

Create `docs/reports/2026-05-14-macro-bucket-path-findings.md`:

```markdown
# Macro Bucket Path Findings

Date: 2026-05-14
Branch/worktree: `feat/macro-delta-reversal`

## Study Scope

Question: does early 5-second volume-delta conviction inside the 15:50 and 15:59 ET macro candles predict continuation, fade, or churn through the rest of the candle?

Input:

- `outputs/nq_macro_volume_delta_5s.parquet`

Outputs:

- `outputs/nq_macro_bucket_path.parquet`
- `outputs/nq_macro_bucket_path_summary.parquet`

## Candle Definitions

- `k350`: 15:50:00-15:50:59 ET, macro buckets `0..11`.
- `k359`: 15:59:00-15:59:59 ET, macro buckets `108..119`.

## Runtime Shapes

Fill from runtime:

- study shape: `(...)`
- summary shape: `(...)`

## Baseline Continuation/Fade

Fill from `candle_baseline` rows.

## First-10s Signed Conviction

Fill from `early_10s_category` rows.

## First-10s Absolute Conviction

Fill from `early_10s_abs_category` rows.

## Current Best Read

Fill after reviewing generated summaries.

## Caveats

- Study uses existing 5-second volume-delta parquet, not raw tick/order-type data.
- No price target is included.
- Findings describe volume-flow path behavior, not trade recommendations.
```

Then replace placeholders with actual table values from Step 4.

- [ ] **Step 6: Commit final report**

```bash
git add features/macro_bucket_path.py test/test_macro_bucket_path.py docs/reports/2026-05-14-macro-bucket-path-findings.md
git commit -m "docs: record macro bucket path findings"
```

---

## Self-Review Checklist

- Spec coverage: schema, `k350`/`k359` mapping, bucket columns, cumulative windows, named windows, diagnostics, continuation/fade flags, conviction categories, summaries, writer, report covered by tasks 1-6.
- TDD order: every production change has failing test step before implementation.
- Data safety: reads existing `outputs/nq_macro_volume_delta_5s.parquet`; writes only `outputs/` and `docs/reports/`; no `input-data/` writes; no raw tick eager reads.
- Commands: all use `.venv/bin/python`.
