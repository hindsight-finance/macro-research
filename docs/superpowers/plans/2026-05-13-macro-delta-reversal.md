# Macro Delta Reversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Polars feature study that evaluates whether prior cumulative volume delta predicts an opposite-signed 15:59 ET volume-delta candle.

**Architecture:** Add a focused `features/macro_delta_reversal.py` module with pure DataFrame builders, summary generation, schema validation, parquet writers, and a CLI entry point. Tests use compact in-memory Polars fixtures that mimic existing `volume_delta.py` outputs; no raw tick data is read.

**Tech Stack:** Python, Polars, pytest, parquet outputs, project virtualenv via `.venv/bin/python`.

---

## File Structure

- Create: `features/macro_delta_reversal.py`
  - Owns input loading, required-column validation, cumulative-window aggregation, sign/relationship feature construction, summary statistics, parquet writes, and `main()`.
- Create: `test/test_macro_delta_reversal.py`
  - Owns fixture DataFrames and behavior tests for validation, window boundaries, signs, relationship flags, summaries, and writer persistence.
- Outputs created by runtime, not committed:
  - `outputs/nq_macro_delta_reversal.parquet`
  - `outputs/nq_macro_delta_reversal_summary.parquet`

---

### Task 1: Add failing schema-validation tests and minimal module constants

**Files:**
- Create: `test/test_macro_delta_reversal.py`
- Create: `features/macro_delta_reversal.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_macro_delta_reversal.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: FAIL during import because `features.macro_delta_reversal` does not exist, or FAIL because `build_macro_delta_reversal` is not implemented.

- [ ] **Step 3: Write minimal implementation**

Create `features/macro_delta_reversal.py` with:

```python
from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

GLOBEX_1M_INPUT_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
MACRO_1M_INPUT_PATH = Path("outputs/nq_macro_volume_delta_1m.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_delta_reversal.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_delta_reversal_summary.parquet")

GLOBEX_REQUIRED_COLUMNS = {
    "trade_date_et",
    "session_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}
MACRO_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}

PREDICTORS = [
    "eth_pre_rth",
    "rth_pre_macro",
    "day_pre_macro",
    "macro_pre59",
    "rth_plus_macro_pre59",
    "day_plus_macro_pre59",
]


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_inputs(globex_1m: pl.DataFrame, macro_1m: pl.DataFrame) -> None:
    globex_missing = _missing_columns(globex_1m, GLOBEX_REQUIRED_COLUMNS)
    if globex_missing:
        raise ValueError(f"Missing Globex volume-delta columns: {globex_missing}")
    macro_missing = _missing_columns(macro_1m, MACRO_REQUIRED_COLUMNS)
    if macro_missing:
        raise ValueError(f"Missing macro volume-delta columns: {macro_missing}")


def build_macro_delta_reversal(globex_1m: pl.DataFrame, macro_1m: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_1m)
    return pl.DataFrame()


def summarize_macro_delta_reversal(study: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame()


def load_volume_delta_inputs(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(globex_path), pl.read_parquet(macro_path)


def write_macro_delta_reversal(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    globex_1m, macro_1m = load_volume_delta_inputs(globex_path, macro_path)
    study = build_macro_delta_reversal(globex_1m, macro_1m)
    summary = summarize_macro_delta_reversal(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not GLOBEX_1M_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {GLOBEX_1M_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not MACRO_1M_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_1M_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_delta_reversal()
    print(f"[OK] Wrote macro delta reversal → {output}")
    print(f"[OK] Wrote macro delta reversal summary → {summary_output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS for the two schema-validation tests.

- [ ] **Step 5: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "test: add macro delta reversal schema checks"
```

---

### Task 2: Add daily window aggregation and target extraction

**Files:**
- Modify: `test/test_macro_delta_reversal.py`
- Modify: `features/macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_aggregates_core_windows_and_target -q
```

Expected: FAIL because the builder returns an empty DataFrame.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, replace `build_macro_delta_reversal` and add these helpers above it:

```python

def _safe_ratio_expr(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def _aggregate_window(frame: pl.DataFrame, index_col: str, start: int, end: int, prefix: str) -> pl.DataFrame:
    return (
        frame.filter(pl.col(index_col).is_between(start, end))
        .group_by("trade_date_et")
        .agg(
            pl.col("volume_delta").sum().alias(f"{prefix}_volume_delta"),
            pl.col("classified_size").sum().alias(f"{prefix}_classified_size"),
            pl.col("total_size").sum().alias(f"{prefix}_total_size"),
        )
        .with_columns(
            _safe_ratio_expr(
                pl.col(f"{prefix}_volume_delta"),
                pl.col(f"{prefix}_classified_size"),
            ).alias(f"{prefix}_delta_imbalance")
        )
    )


def _extract_k359(macro_1m: pl.DataFrame) -> pl.DataFrame:
    return (
        macro_1m.filter(pl.col("macro_minute_index") == 59)
        .group_by("trade_date_et")
        .agg(
            pl.col("volume_delta").sum().alias("k359_volume_delta"),
            pl.col("classified_size").sum().alias("k359_classified_size"),
            pl.col("total_size").sum().alias("k359_total_size"),
        )
        .with_columns(
            _safe_ratio_expr(pl.col("k359_volume_delta"), pl.col("k359_classified_size")).alias(
                "k359_delta_imbalance"
            )
        )
    )


def _add_combined_window(frame: pl.DataFrame, left: str, right: str, output: str) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col(f"{left}_volume_delta").fill_null(0) + pl.col(f"{right}_volume_delta").fill_null(0)).alias(
            f"{output}_volume_delta"
        ),
        (pl.col(f"{left}_classified_size").fill_null(0) + pl.col(f"{right}_classified_size").fill_null(0)).alias(
            f"{output}_classified_size"
        ),
        (pl.col(f"{left}_total_size").fill_null(0) + pl.col(f"{right}_total_size").fill_null(0)).alias(
            f"{output}_total_size"
        ),
    ).with_columns(
        _safe_ratio_expr(pl.col(f"{output}_volume_delta"), pl.col(f"{output}_classified_size")).alias(
            f"{output}_delta_imbalance"
        )
    )


def build_macro_delta_reversal(globex_1m: pl.DataFrame, macro_1m: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_1m)

    target = _extract_k359(macro_1m)
    eth = _aggregate_window(globex_1m, "session_minute_index", 0, 929, "eth_pre_rth")
    rth = _aggregate_window(globex_1m, "session_minute_index", 930, 1309, "rth_pre_macro")
    day = _aggregate_window(globex_1m, "session_minute_index", 0, 1309, "day_pre_macro")
    macro_pre59 = _aggregate_window(macro_1m, "macro_minute_index", 50, 58, "macro_pre59")

    out = (
        target.join(eth, on="trade_date_et", how="left")
        .join(rth, on="trade_date_et", how="left")
        .join(day, on="trade_date_et", how="left")
        .join(macro_pre59, on="trade_date_et", how="left")
    )
    out = _add_combined_window(out, "rth_pre_macro", "macro_pre59", "rth_plus_macro_pre59")
    out = _add_combined_window(out, "day_pre_macro", "macro_pre59", "day_plus_macro_pre59")
    return out.rename({"trade_date_et": "date"}).sort("date")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_aggregates_core_windows_and_target -q
```

Expected: PASS.

- [ ] **Step 5: Run all current macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: aggregate macro delta reversal windows"
```

---

### Task 3: Add sign and relationship flags

**Files:**
- Modify: `test/test_macro_delta_reversal.py`
- Modify: `features/macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_build_macro_delta_reversal_adds_signs_and_relationship_flags():
    globex = _globex_rows(
        [
            _g("2025-01-02", 0, 10),
            _g("2025-01-02", 930, 5),
            _g("2025-01-03", 0, -8),
            _g("2025-01-03", 930, 0, 10, 10),
        ]
    )
    macro = _macro_rows(
        [
            _m("2025-01-02", 50, -3),
            _m("2025-01-02", 59, -7),
            _m("2025-01-03", 50, 4),
            _m("2025-01-03", 59, 0, 12, 12),
        ]
    )

    out = build_macro_delta_reversal(globex, macro)
    day1 = out.filter(pl.col("date") == pl.date(2025, 1, 2)).row(0, named=True)
    day2 = out.filter(pl.col("date") == pl.date(2025, 1, 3)).row(0, named=True)

    assert day1["eth_pre_rth_sign"] == 1
    assert day1["rth_pre_macro_sign"] == 1
    assert day1["macro_pre59_sign"] == -1
    assert day1["k359_sign"] == -1
    assert day1["eth_pre_rth_opposes_k359"] is True
    assert day1["eth_pre_rth_same_as_k359"] is False
    assert day1["eth_pre_rth_has_signal"] is True
    assert day1["macro_pre59_opposes_rth_pre_macro"] is True
    assert day1["macro_pre59_opposes_day_pre_macro"] is True
    assert day1["k359_opposes_rth_plus_macro_pre59"] is True
    assert day1["k359_opposes_day_plus_macro_pre59"] is True

    assert day2["rth_pre_macro_sign"] == 0
    assert day2["k359_sign"] == 0
    assert day2["eth_pre_rth_opposes_k359"] is False
    assert day2["eth_pre_rth_same_as_k359"] is False
    assert day2["eth_pre_rth_has_signal"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_adds_signs_and_relationship_flags -q
```

Expected: FAIL because sign columns and relationship flags are missing.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add these helpers above `build_macro_delta_reversal`:

```python

def _sign_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) > 0)
        .then(1)
        .when(pl.col(column) < 0)
        .then(-1)
        .otherwise(0)
    )


def _add_signs_and_relationships(frame: pl.DataFrame) -> pl.DataFrame:
    sign_names = [*PREDICTORS, "k359"]
    out = frame.with_columns([_sign_expr(f"{name}_volume_delta").alias(f"{name}_sign") for name in sign_names])

    relationship_exprs: list[pl.Expr] = []
    for predictor in PREDICTORS:
        pred_sign = pl.col(f"{predictor}_sign")
        target_sign = pl.col("k359_sign")
        has_signal = (pred_sign != 0) & (target_sign != 0)
        relationship_exprs.extend(
            [
                has_signal.alias(f"{predictor}_has_signal"),
                (has_signal & (pred_sign == -target_sign)).alias(f"{predictor}_opposes_k359"),
                (has_signal & (pred_sign == target_sign)).alias(f"{predictor}_same_as_k359"),
            ]
        )

    out = out.with_columns(relationship_exprs)
    return out.with_columns(
        (
            (pl.col("macro_pre59_sign") != 0)
            & (pl.col("rth_pre_macro_sign") != 0)
            & (pl.col("macro_pre59_sign") == -pl.col("rth_pre_macro_sign"))
        ).alias("macro_pre59_opposes_rth_pre_macro"),
        (
            (pl.col("macro_pre59_sign") != 0)
            & (pl.col("day_pre_macro_sign") != 0)
            & (pl.col("macro_pre59_sign") == -pl.col("day_pre_macro_sign"))
        ).alias("macro_pre59_opposes_day_pre_macro"),
        (
            (pl.col("k359_sign") != 0)
            & (pl.col("rth_plus_macro_pre59_sign") != 0)
            & (pl.col("k359_sign") == -pl.col("rth_plus_macro_pre59_sign"))
        ).alias("k359_opposes_rth_plus_macro_pre59"),
        (
            (pl.col("k359_sign") != 0)
            & (pl.col("day_plus_macro_pre59_sign") != 0)
            & (pl.col("k359_sign") == -pl.col("day_plus_macro_pre59_sign"))
        ).alias("k359_opposes_day_plus_macro_pre59"),
    )
```

Then change the last line of `build_macro_delta_reversal` to:

```python
    return _add_signs_and_relationships(out.rename({"trade_date_et": "date"})).sort("date")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_build_macro_delta_reversal_adds_signs_and_relationship_flags -q
```

Expected: PASS.

- [ ] **Step 5: Run all current macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add macro delta reversal flags"
```

---

### Task 4: Add summary statistics

**Files:**
- Modify: `test/test_macro_delta_reversal.py`
- Modify: `features/macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_summarize_macro_delta_reversal_computes_predictor_statistics():
    globex = _globex_rows(
        [
            _g("2025-01-02", 930, 10, 20, 20),
            _g("2025-01-03", 930, -8, 16, 16),
            _g("2025-01-04", 930, 0, 10, 10),
        ]
    )
    macro = _macro_rows(
        [
            _m("2025-01-02", 50, -2, 4, 4),
            _m("2025-01-02", 59, -5, 10, 10),
            _m("2025-01-03", 50, 2, 4, 4),
            _m("2025-01-03", 59, 4, 8, 8),
            _m("2025-01-04", 50, 0, 3, 3),
            _m("2025-01-04", 59, 3, 6, 6),
        ]
    )
    study = build_macro_delta_reversal(globex, macro)

    summary = summarize_macro_delta_reversal(study)
    rth = summary.filter((pl.col("summary_type") == "sign") & (pl.col("predictor") == "rth_pre_macro")).row(
        0, named=True
    )

    assert rth["n_days"] == 3
    assert rth["n_signal_days"] == 2
    assert rth["opposite_count"] == 2
    assert rth["opposite_rate"] == pytest.approx(1.0)
    assert rth["same_count"] == 0
    assert rth["same_rate"] == pytest.approx(0.0)
    assert rth["zero_predictor_count"] == 1
    assert rth["zero_k359_count"] == 0
    assert rth["mean_predictor_delta"] == pytest.approx((10 - 8 + 0) / 3)
    assert rth["median_predictor_delta"] == pytest.approx(0.0)
    assert rth["mean_k359_delta_when_predictor_positive"] == pytest.approx(-5.0)
    assert rth["mean_k359_delta_when_predictor_negative"] == pytest.approx(4.0)
    assert rth["median_k359_delta_when_predictor_positive"] == pytest.approx(-5.0)
    assert rth["median_k359_delta_when_predictor_negative"] == pytest.approx(4.0)
    assert rth["pearson_corr_predictor_vs_k359_delta"] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_computes_predictor_statistics -q
```

Expected: FAIL because `summarize_macro_delta_reversal` returns an empty DataFrame.

- [ ] **Step 3: Write minimal implementation**

Replace `summarize_macro_delta_reversal` in `features/macro_delta_reversal.py` with:

```python

def _rate(numer: int, denom: int) -> float | None:
    return (numer / denom) if denom else None


def _mean_for_sign(study: pl.DataFrame, predictor: str, sign: int) -> float | None:
    values = study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col("k359_volume_delta").mean()).item()
    return None if values is None else float(values)


def _median_for_sign(study: pl.DataFrame, predictor: str, sign: int) -> float | None:
    values = study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col("k359_volume_delta").median()).item()
    return None if values is None else float(values)


def summarize_macro_delta_reversal(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    n_days = study.height
    for predictor in PREDICTORS:
        signal = study.filter(pl.col(f"{predictor}_has_signal"))
        n_signal_days = signal.height
        opposite_count = study.filter(pl.col(f"{predictor}_opposes_k359")).height
        same_count = study.filter(pl.col(f"{predictor}_same_as_k359")).height
        corr = study.select(pl.corr(f"{predictor}_volume_delta", "k359_volume_delta")).item()
        rows.append(
            {
                "summary_type": "sign",
                "predictor": predictor,
                "predictor_decile": None,
                "n_days": n_days,
                "n_signal_days": n_signal_days,
                "opposite_count": opposite_count,
                "opposite_rate": _rate(opposite_count, n_signal_days),
                "same_count": same_count,
                "same_rate": _rate(same_count, n_signal_days),
                "zero_predictor_count": study.filter(pl.col(f"{predictor}_sign") == 0).height,
                "zero_k359_count": study.filter(pl.col("k359_sign") == 0).height,
                "mean_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").mean()).item(),
                "median_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").median()).item(),
                "mean_k359_delta": study.select(pl.col("k359_volume_delta").mean()).item(),
                "median_k359_delta": study.select(pl.col("k359_volume_delta").median()).item(),
                "mean_k359_delta_when_predictor_positive": _mean_for_sign(study, predictor, 1),
                "mean_k359_delta_when_predictor_negative": _mean_for_sign(study, predictor, -1),
                "median_k359_delta_when_predictor_positive": _median_for_sign(study, predictor, 1),
                "median_k359_delta_when_predictor_negative": _median_for_sign(study, predictor, -1),
                "pearson_corr_predictor_vs_k359_delta": corr,
            }
        )
    return pl.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_computes_predictor_statistics -q
```

Expected: PASS.

- [ ] **Step 5: Run all current macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: summarize macro delta reversal statistics"
```

---

### Task 5: Add compact decile rows to the summary

**Files:**
- Modify: `test/test_macro_delta_reversal.py`
- Modify: `features/macro_delta_reversal.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_summarize_macro_delta_reversal_adds_decile_rows_when_enough_unique_values():
    globex = _globex_rows([_g(f"2025-01-{day:02d}", 930, day) for day in range(2, 14)])
    macro = _macro_rows([_m(f"2025-01-{day:02d}", 59, -day) for day in range(2, 14)])
    study = build_macro_delta_reversal(globex, macro)

    summary = summarize_macro_delta_reversal(study)
    deciles = summary.filter((pl.col("summary_type") == "decile") & (pl.col("predictor") == "rth_pre_macro"))

    assert deciles.height == 10
    assert deciles.select("predictor_decile").to_series().to_list() == list(range(1, 11))
    assert deciles.select(pl.col("n_days").sum()).item() == 12
    assert deciles.select(pl.col("mean_k359_delta").max()).item() < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_decile_rows_when_enough_unique_values -q
```

Expected: FAIL because no decile rows are emitted.

- [ ] **Step 3: Write minimal implementation**

In `features/macro_delta_reversal.py`, add this helper above `summarize_macro_delta_reversal`:

```python

def _decile_rows(study: pl.DataFrame, predictor: str) -> list[dict]:
    value_col = f"{predictor}_volume_delta"
    non_null = study.filter(pl.col(value_col).is_not_null())
    unique_count = non_null.select(pl.col(value_col).n_unique()).item() if not non_null.is_empty() else 0
    if non_null.height < 10 or unique_count < 10:
        return []

    deciled = non_null.with_columns(
        ((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height)
        .floor()
        .cast(pl.Int64)
        .clip(0, 9)
        .add(1)
        .alias("predictor_decile")
    )

    rows = []
    for record in deciled.group_by("predictor_decile").agg(
        pl.len().alias("n_days"),
        pl.col(value_col).mean().alias("mean_predictor_delta"),
        pl.col("k359_volume_delta").mean().alias("mean_k359_delta"),
        pl.col(f"{predictor}_opposes_k359").sum().alias("opposite_count"),
        pl.col(f"{predictor}_has_signal").sum().alias("n_signal_days"),
    ).sort("predictor_decile").to_dicts():
        n_signal_days = int(record["n_signal_days"])
        opposite_count = int(record["opposite_count"])
        rows.append(
            {
                "summary_type": "decile",
                "predictor": predictor,
                "predictor_decile": int(record["predictor_decile"]),
                "n_days": int(record["n_days"]),
                "n_signal_days": n_signal_days,
                "opposite_count": opposite_count,
                "opposite_rate": _rate(opposite_count, n_signal_days),
                "same_count": None,
                "same_rate": None,
                "zero_predictor_count": None,
                "zero_k359_count": None,
                "mean_predictor_delta": float(record["mean_predictor_delta"]),
                "median_predictor_delta": None,
                "mean_k359_delta": float(record["mean_k359_delta"]),
                "median_k359_delta": None,
                "mean_k359_delta_when_predictor_positive": None,
                "mean_k359_delta_when_predictor_negative": None,
                "median_k359_delta_when_predictor_positive": None,
                "median_k359_delta_when_predictor_negative": None,
                "pearson_corr_predictor_vs_k359_delta": None,
            }
        )
    return rows
```

Then add this immediately before `return pl.DataFrame(rows)` in `summarize_macro_delta_reversal`:

```python
        rows.extend(_decile_rows(study, predictor))
```

Ensure it is indented inside the `for predictor in PREDICTORS:` loop, after the sign-stat row append.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_summarize_macro_delta_reversal_adds_decile_rows_when_enough_unique_values -q
```

Expected: PASS.

- [ ] **Step 5: Run all current macro delta reversal tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py
git commit -m "feat: add macro delta reversal decile summary"
```

---

### Task 6: Add writer persistence test and final runtime verification

**Files:**
- Modify: `test/test_macro_delta_reversal.py`
- Modify: `features/macro_delta_reversal.py` if the writer needs adjustment

- [ ] **Step 1: Write the failing test**

Append to `test/test_macro_delta_reversal.py`:

```python

def test_write_macro_delta_reversal_persists_study_and_summary(tmp_path: Path):
    globex_path = tmp_path / "globex.parquet"
    macro_path = tmp_path / "macro.parquet"
    output_path = tmp_path / "nested" / "study.parquet"
    summary_path = tmp_path / "nested" / "summary.parquet"

    _globex_rows([_g("2025-01-02", 930, 10)]).write_parquet(globex_path)
    _macro_rows([_m("2025-01-02", 59, -5)]).write_parquet(macro_path)

    result = write_macro_delta_reversal(globex_path, macro_path, output_path, summary_path)

    assert result == (output_path, summary_path)
    study = pl.read_parquet(output_path)
    summary = pl.read_parquet(summary_path)
    assert study.height == 1
    assert study.row(0, named=True)["rth_pre_macro_opposes_k359"] is True
    assert summary.filter(pl.col("summary_type") == "sign").height == 6
```

- [ ] **Step 2: Run test to verify it fails or passes for the right reason**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py::test_write_macro_delta_reversal_persists_study_and_summary -q
```

Expected: PASS if the writer from Task 1 already satisfies the behavior. If it fails, the failure should point to missing directory creation, wrong return values, or missing computed columns.

- [ ] **Step 3: Adjust writer only if needed**

If Step 2 fails, update `write_macro_delta_reversal` in `features/macro_delta_reversal.py` to exactly:

```python

def write_macro_delta_reversal(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_path: str | Path = MACRO_1M_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    globex_1m, macro_1m = load_volume_delta_inputs(globex_path, macro_path)
    study = build_macro_delta_reversal(globex_1m, macro_1m)
    summary = summarize_macro_delta_reversal(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output
```

- [ ] **Step 4: Run the full macro delta reversal test file**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 5: Run related existing tests**

Run:

```bash
.venv/bin/python -m pytest test/test_volume_delta.py test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 6: Run the feature on existing parquet inputs**

Run:

```bash
.venv/bin/python -m features.macro_delta_reversal
```

Expected output:

```text
[OK] Wrote macro delta reversal → outputs/nq_macro_delta_reversal.parquet
[OK] Wrote macro delta reversal summary → outputs/nq_macro_delta_reversal_summary.parquet
```

- [ ] **Step 7: Inspect output schemas and top summary rows**

Run:

```bash
.venv/bin/python - <<'PY'
import polars as pl
study = pl.read_parquet('outputs/nq_macro_delta_reversal.parquet')
summary = pl.read_parquet('outputs/nq_macro_delta_reversal_summary.parquet')
print(study.shape)
print(study.select(['date', 'rth_pre_macro_volume_delta', 'macro_pre59_volume_delta', 'k359_volume_delta']).head())
print(summary.filter(pl.col('summary_type') == 'sign').select(['predictor', 'n_days', 'n_signal_days', 'opposite_rate', 'pearson_corr_predictor_vs_k359_delta']))
PY
```

Expected: non-empty study rows if the input parquet files exist and include 15:59 data; six `summary_type == "sign"` rows.

- [ ] **Step 8: Commit**

```bash
git add features/macro_delta_reversal.py test/test_macro_delta_reversal.py outputs/nq_macro_delta_reversal.parquet outputs/nq_macro_delta_reversal_summary.parquet
git commit -m "feat: add macro delta reversal study"
```

If generated outputs should remain untracked for this branch, omit the two `outputs/*.parquet` paths from `git add` and mention them in the final handoff instead.

---

## Final Verification

- [ ] **Step 1: Run targeted tests**

```bash
.venv/bin/python -m pytest test/test_volume_delta.py test/test_macro_delta_reversal.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the main suite if time allows**

```bash
.venv/bin/python -m pytest test -q
```

Expected: PASS, or document any unrelated pre-existing failures with exact failing test names and error messages.

- [ ] **Step 3: Check git status**

```bash
git status --short
```

Expected: only intentional generated outputs may remain untracked or modified; source and tests should be committed.
