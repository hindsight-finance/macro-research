# Macro 15:50 Delta Impulse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 15:50 ET volume-delta impulse study that tests whether ETH-only, RTH-only, or ETH+RTH pre-15:50 accumulated delta predicts the first 10 seconds of 15:50 volume-delta flow.

**Architecture:** Add a new focused module, `features/macro_1550_delta_impulse.py`, mirroring the 15:59 reversal study without changing `features/macro_delta_reversal.py`. The module aggregates predictor windows from Globex/RTH 1-minute delta, aggregates 15:50 5-second targets from macro 5-second delta, then emits a daily parquet and long-form target-aware summary parquet.

**Tech Stack:** Python, Polars, pytest, parquet outputs, project virtualenv via `.venv/bin/python`.

---

## File Structure

- Create: `features/macro_1550_delta_impulse.py`
  - Constants for inputs/outputs, required schemas, predictor windows, target windows.
  - `build_macro_1550_delta_impulse(globex_1m, macro_5s)` for daily table.
  - `summarize_macro_1550_delta_impulse(study)` for target-aware summaries.
  - `load_volume_delta_inputs(...)`, `write_macro_1550_delta_impulse(...)`, `main()`.
- Create: `test/test_macro_1550_delta_impulse.py`
  - In-memory Polars fixtures for 1-minute Globex/RTH rows and 5-second macro rows.
  - Focused TDD tests for schema validation, window boundaries, signs, summaries, writer persistence.
- Runtime outputs, not committed unless explicitly requested:
  - `outputs/nq_macro_1550_delta_impulse.parquet`
  - `outputs/nq_macro_1550_delta_impulse_summary.parquet`

---

### Task 1: Add module skeleton, schema validation, loader, writer

**Files:**
- Create: `features/macro_1550_delta_impulse.py`
- Create: `test/test_macro_1550_delta_impulse.py`

- [ ] **Step 1: Write failing schema and writer tests**

Create `test/test_macro_1550_delta_impulse.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_requires_globex_schema test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_requires_macro_5s_schema test/test_macro_1550_delta_impulse.py::test_write_macro_1550_delta_impulse_persists_outputs -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'features.macro_1550_delta_impulse'`.

- [ ] **Step 3: Write minimal module implementation**

Create `features/macro_1550_delta_impulse.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

GLOBEX_1M_INPUT_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
MACRO_5S_INPUT_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_1550_delta_impulse.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_1550_delta_impulse_summary.parquet")

GLOBEX_REQUIRED_COLUMNS = {
    "trade_date_et",
    "session_minute_index",
    "volume_delta",
    "classified_size",
    "total_size",
}
MACRO_5S_REQUIRED_COLUMNS = {
    "trade_date_et",
    "macro_bucket_index",
    "volume_delta",
    "classified_size",
    "total_size",
}

PREDICTORS = ["eth_only_pre350", "rth_only_pre350", "eth_rth_pre350"]
TARGET_WINDOWS_5S = {
    "k350_00_09": (0, 1),
}
TARGET_WINDOWS = [*TARGET_WINDOWS_5S.keys()]


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_inputs(globex_1m: pl.DataFrame, macro_5s: pl.DataFrame) -> None:
    globex_missing = _missing_columns(globex_1m, GLOBEX_REQUIRED_COLUMNS)
    if globex_missing:
        raise ValueError(f"Missing Globex volume-delta columns: {globex_missing}")
    macro_5s_missing = _missing_columns(macro_5s, MACRO_5S_REQUIRED_COLUMNS)
    if macro_5s_missing:
        raise ValueError(f"Missing macro 5-second volume-delta columns: {macro_5s_missing}")


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
            _safe_ratio_expr(pl.col(f"{prefix}_volume_delta"), pl.col(f"{prefix}_classified_size")).alias(
                f"{prefix}_delta_imbalance"
            )
        )
    )


def _aggregate_target_window_5s(macro_5s: pl.DataFrame, start: int, end: int, prefix: str) -> pl.DataFrame:
    return _aggregate_window(macro_5s, "macro_bucket_index", start, end, prefix)


def _sign_expr(column: str) -> pl.Expr:
    return pl.when(pl.col(column) > 0).then(1).when(pl.col(column) < 0).then(-1).otherwise(0)


def _add_signs(frame: pl.DataFrame) -> pl.DataFrame:
    names = [*PREDICTORS, *TARGET_WINDOWS]
    return frame.with_columns([_sign_expr(f"{name}_volume_delta").alias(f"{name}_sign") for name in names])


def build_macro_1550_delta_impulse(globex_1m: pl.DataFrame, macro_5s: pl.DataFrame) -> pl.DataFrame:
    _validate_inputs(globex_1m, macro_5s)
    eth = _aggregate_window(globex_1m, "session_minute_index", 0, 929, "eth_only_pre350")
    rth = _aggregate_window(globex_1m, "session_minute_index", 930, 1309, "rth_only_pre350")
    eth_rth = _aggregate_window(globex_1m, "session_minute_index", 0, 1309, "eth_rth_pre350")
    target = _aggregate_target_window_5s(macro_5s, 0, 1, "k350_00_09")
    out = (
        target.join(eth, on="trade_date_et", how="left")
        .join(rth, on="trade_date_et", how="left")
        .join(eth_rth, on="trade_date_et", how="left")
        .rename({"trade_date_et": "date"})
    )
    return _add_signs(out).sort("date")


def _rate(numer: int, denom: int) -> float | None:
    return (numer / denom) if denom else None


def summarize_macro_1550_delta_impulse(study: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for predictor in PREDICTORS:
        pred_sign = pl.col(f"{predictor}_sign")
        target_sign = pl.col("k350_00_09_sign")
        signal = study.filter((pred_sign != 0) & (target_sign != 0))
        opposite_count = study.filter((pred_sign != 0) & (target_sign != 0) & (pred_sign == -target_sign)).height
        same_count = study.filter((pred_sign != 0) & (target_sign != 0) & (pred_sign == target_sign)).height
        rows.append(
            {
                "summary_type": "target_sign",
                "predictor": predictor,
                "target_window": "k350_00_09",
                "n_days": study.height,
                "n_signal_days": signal.height,
                "opposite_count": opposite_count,
                "opposite_rate": _rate(opposite_count, signal.height),
                "same_count": same_count,
                "same_rate": _rate(same_count, signal.height),
                "zero_predictor_count": study.filter(pred_sign == 0).height,
                "zero_target_count": study.filter(target_sign == 0).height,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def load_volume_delta_inputs(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(globex_path), pl.read_parquet(macro_5s_path)


def write_macro_1550_delta_impulse(
    globex_path: str | Path = GLOBEX_1M_INPUT_PATH,
    macro_5s_path: str | Path = MACRO_5S_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    globex_1m, macro_5s = load_volume_delta_inputs(globex_path, macro_5s_path)
    study = build_macro_1550_delta_impulse(globex_1m, macro_5s)
    summary = summarize_macro_1550_delta_impulse(study)
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
    if not MACRO_5S_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {MACRO_5S_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary_output = write_macro_1550_delta_impulse()
    print(f"[OK] Wrote macro 15:50 delta impulse -> {output}")
    print(f"[OK] Wrote macro 15:50 delta impulse summary -> {summary_output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_requires_globex_schema test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_requires_macro_5s_schema test/test_macro_1550_delta_impulse.py::test_write_macro_1550_delta_impulse_persists_outputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_1550_delta_impulse.py test/test_macro_1550_delta_impulse.py
git commit -m "feat: add macro 1550 delta impulse skeleton"
```

---

### Task 2: Add full 15:50 target windows and verify bucket boundaries

**Files:**
- Modify: `features/macro_1550_delta_impulse.py`
- Modify: `test/test_macro_1550_delta_impulse.py`

- [ ] **Step 1: Write failing target-window test**

Append to `test/test_macro_1550_delta_impulse.py`:

```python

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
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_aggregates_target_windows -q
```

Expected: FAIL with missing `k350_00_04_volume_delta` or similar.

- [ ] **Step 3: Implement full target windows**

In `features/macro_1550_delta_impulse.py`, replace target constants:

```python
TARGET_WINDOWS_5S = {
    "k350_00_09": (0, 1),
    "k350_00_04": (0, 0),
    "k350_05_09": (1, 1),
    "k350_00_29": (0, 5),
    "k350_00_59": (0, 11),
}
TARGET_WINDOWS = [*TARGET_WINDOWS_5S.keys(), *[f"k350_bucket_{bucket}" for bucket in range(0, 12)]]
ROBUST_TARGET_WINDOWS = [*TARGET_WINDOWS_5S.keys()]
```

In `build_macro_1550_delta_impulse`, replace single target creation/join with:

```python
    out = None
    for prefix, (start, end) in TARGET_WINDOWS_5S.items():
        target = _aggregate_target_window_5s(macro_5s, start, end, prefix)
        out = target if out is None else out.join(target, on="trade_date_et", how="outer_coalesce")
    for bucket in range(0, 12):
        prefix = f"k350_bucket_{bucket}"
        target = _aggregate_target_window_5s(macro_5s, bucket, bucket, prefix)
        out = out.join(target, on="trade_date_et", how="outer_coalesce")

    if out is None:
        out = pl.DataFrame({"trade_date_et": []}, schema={"trade_date_et": pl.Date})

    out = (
        out.join(eth, on="trade_date_et", how="left")
        .join(rth, on="trade_date_et", how="left")
        .join(eth_rth, on="trade_date_et", how="left")
        .rename({"trade_date_et": "date"})
    )
```

Keep `_add_signs(out).sort("date")` as the return.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_aggregates_target_windows -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_1550_delta_impulse.py test/test_macro_1550_delta_impulse.py
git commit -m "feat: add macro 1550 impulse target windows"
```

---

### Task 3: Verify predictor boundaries and sign behavior

**Files:**
- Modify: `features/macro_1550_delta_impulse.py`
- Modify: `test/test_macro_1550_delta_impulse.py`

- [ ] **Step 1: Write failing predictor-boundary and relationship tests**

Append to `test/test_macro_1550_delta_impulse.py`:

```python

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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_aggregates_predictor_boundaries test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_adds_signs_and_primary_relationships -q
```

Expected: first may PASS if Task 1 covered boundaries; second FAIL with missing relationship columns.

- [ ] **Step 3: Add primary-target relationship flags**

In `features/macro_1550_delta_impulse.py`, add:

```python
def _add_primary_relationships(frame: pl.DataFrame) -> pl.DataFrame:
    relationship_exprs: list[pl.Expr] = []
    target = "k350_00_09"
    target_sign = pl.col(f"{target}_sign")
    for predictor in PREDICTORS:
        pred_sign = pl.col(f"{predictor}_sign")
        has_signal = (pred_sign != 0) & (target_sign != 0)
        relationship_exprs.extend(
            [
                has_signal.alias(f"{predictor}_has_signal"),
                (has_signal & (pred_sign == -target_sign)).alias(f"{predictor}_opposes_{target}"),
                (has_signal & (pred_sign == target_sign)).alias(f"{predictor}_same_as_{target}"),
            ]
        )
    return frame.with_columns(relationship_exprs)
```

Update the return in `build_macro_1550_delta_impulse`:

```python
    out = _add_signs(out)
    out = _add_primary_relationships(out)
    return out.sort("date")
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_aggregates_predictor_boundaries test/test_macro_1550_delta_impulse.py::test_build_macro_1550_delta_impulse_adds_signs_and_primary_relationships -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_1550_delta_impulse.py test/test_macro_1550_delta_impulse.py
git commit -m "feat: add macro 1550 impulse predictors"
```

---

### Task 4: Add target-aware sign summary fields

**Files:**
- Modify: `features/macro_1550_delta_impulse.py`
- Modify: `test/test_macro_1550_delta_impulse.py`

- [ ] **Step 1: Write failing summary test**

Append to `test/test_macro_1550_delta_impulse.py`:

```python

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
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_summarize_macro_1550_delta_impulse_adds_target_sign_rows -q
```

Expected: FAIL with missing summary columns.

- [ ] **Step 3: Expand target-aware summary implementation**

In `features/macro_1550_delta_impulse.py`, add helpers:

```python
def _available_target_windows(study: pl.DataFrame) -> list[str]:
    return [target for target in TARGET_WINDOWS if f"{target}_volume_delta" in study.columns]


def _scalar_or_none(frame: pl.DataFrame, expr: pl.Expr) -> float | None:
    value = frame.select(expr).item()
    return None if value is None else float(value)


def _target_values_for_predictor_sign(study: pl.DataFrame, predictor: str, target: str, sign: int) -> pl.DataFrame:
    return study.filter(pl.col(f"{predictor}_sign") == sign).select(pl.col(f"{target}_volume_delta").alias("target_delta"))


def _target_pair_sign_row(study: pl.DataFrame, predictor: str, target: str) -> dict:
    pred_sign = pl.col(f"{predictor}_sign")
    target_sign = pl.col(f"{target}_sign")
    has_signal_expr = (pred_sign != 0) & (target_sign != 0)
    signal = study.filter(has_signal_expr)
    n_signal_days = signal.height
    opposite_count = study.filter(has_signal_expr & (pred_sign == -target_sign)).height
    same_count = study.filter(has_signal_expr & (pred_sign == target_sign)).height
    pos = _target_values_for_predictor_sign(study, predictor, target, 1)
    neg = _target_values_for_predictor_sign(study, predictor, target, -1)
    corr = study.select(pl.corr(f"{predictor}_volume_delta", f"{target}_volume_delta")).item()
    return {
        "summary_type": "target_sign",
        "predictor": predictor,
        "target_window": target,
        "predictor_decile": None,
        "tail": None,
        "n_days": study.height,
        "n_signal_days": n_signal_days,
        "opposite_count": opposite_count,
        "opposite_rate": _rate(opposite_count, n_signal_days),
        "same_count": same_count,
        "same_rate": _rate(same_count, n_signal_days),
        "zero_predictor_count": study.filter(pred_sign == 0).height,
        "zero_target_count": study.filter(target_sign == 0).height,
        "mean_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").mean()).item(),
        "median_predictor_delta": study.select(pl.col(f"{predictor}_volume_delta").median()).item(),
        "mean_target_delta": study.select(pl.col(f"{target}_volume_delta").mean()).item(),
        "median_target_delta": study.select(pl.col(f"{target}_volume_delta").median()).item(),
        "mean_target_delta_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").mean()),
        "mean_target_delta_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").mean()),
        "median_target_delta_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").median()),
        "median_target_delta_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").median()),
        "target_p25_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").quantile(0.25)),
        "target_p75_when_predictor_positive": _scalar_or_none(pos, pl.col("target_delta").quantile(0.75)),
        "target_p25_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").quantile(0.25)),
        "target_p75_when_predictor_negative": _scalar_or_none(neg, pl.col("target_delta").quantile(0.75)),
        "pearson_corr_predictor_vs_target_delta": corr,
    }
```

Replace `summarize_macro_1550_delta_impulse` body:

```python
def summarize_macro_1550_delta_impulse(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for predictor in PREDICTORS:
        for target in _available_target_windows(study):
            rows.append(_target_pair_sign_row(study, predictor, target))
    return pl.DataFrame(rows, infer_schema_length=None)
```

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_summarize_macro_1550_delta_impulse_adds_target_sign_rows -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_1550_delta_impulse.py test/test_macro_1550_delta_impulse.py
git commit -m "feat: add macro 1550 target sign summaries"
```

---

### Task 5: Add robust decile and tail summaries

**Files:**
- Modify: `features/macro_1550_delta_impulse.py`
- Modify: `test/test_macro_1550_delta_impulse.py`

- [ ] **Step 1: Write failing robust-summary test**

Append to `test/test_macro_1550_delta_impulse.py`:

```python

def test_summarize_macro_1550_delta_impulse_adds_decile_and_tail_rows():
    globex_rows = []
    macro_rows = []
    for i in range(20):
        day = f"2025-01-{i + 1:02d}"
        predictor_delta = i + 1 if i >= 10 else -(20 - i)
        target_delta = -predictor_delta
        globex_rows.append(_g(day, 930, predictor_delta, abs(predictor_delta) * 2, abs(predictor_delta) * 2))
        macro_rows.append(_s(day, 0, target_delta, abs(target_delta) * 2, abs(target_delta) * 2))
        macro_rows.append(_s(day, 1, 0, 1, 1))
    study = build_macro_1550_delta_impulse(_globex_rows(globex_rows), _macro_5s_rows(macro_rows))

    summary = summarize_macro_1550_delta_impulse(study)

    raw_deciles = summary.filter(
        (pl.col("summary_type") == "target_raw_decile")
        & (pl.col("predictor") == "rth_only_pre350")
        & (pl.col("target_window") == "k350_00_09")
    )
    imbalance_deciles = summary.filter(
        (pl.col("summary_type") == "target_imbalance_decile")
        & (pl.col("predictor") == "rth_only_pre350")
        & (pl.col("target_window") == "k350_00_09")
    )
    tails = summary.filter(
        (pl.col("summary_type") == "target_tail")
        & (pl.col("predictor") == "rth_only_pre350")
        & (pl.col("target_window") == "k350_00_09")
    )

    assert raw_deciles.height == 10
    assert imbalance_deciles.height == 10
    assert set(tails["tail"].to_list()) == {
        "positive_top_20",
        "positive_top_10",
        "negative_bottom_20",
        "negative_bottom_10",
    }
    top_tail = tails.filter(pl.col("tail") == "positive_top_20").row(0, named=True)
    assert top_tail["n_days"] >= 1
    assert top_tail["opposite_rate"] == pytest.approx(1.0)
    assert top_tail["median_target_delta"] < 0
    assert top_tail["target_p25"] is not None
    assert top_tail["target_p75"] is not None
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_summarize_macro_1550_delta_impulse_adds_decile_and_tail_rows -q
```

Expected: FAIL because robust rows are missing.

- [ ] **Step 3: Implement robust rows**

In `features/macro_1550_delta_impulse.py`, add:

```python
def _base_target_summary_row(
    predictor: str,
    target: str,
    summary_type: str,
    subset: pl.DataFrame,
    predictor_decile: int | None = None,
    tail: str | None = None,
) -> dict:
    pred_sign = pl.col(f"{predictor}_sign")
    target_sign = pl.col(f"{target}_sign")
    signal = subset.filter((pred_sign != 0) & (target_sign != 0))
    n_signal_days = signal.height
    opposite_count = subset.filter((pred_sign != 0) & (target_sign != 0) & (pred_sign == -target_sign)).height
    same_count = subset.filter((pred_sign != 0) & (target_sign != 0) & (pred_sign == target_sign)).height
    return {
        "summary_type": summary_type,
        "predictor": predictor,
        "target_window": target,
        "predictor_decile": predictor_decile,
        "tail": tail,
        "n_days": subset.height,
        "n_signal_days": n_signal_days,
        "opposite_count": opposite_count,
        "opposite_rate": _rate(opposite_count, n_signal_days),
        "same_count": same_count,
        "same_rate": _rate(same_count, n_signal_days),
        "zero_predictor_count": subset.filter(pred_sign == 0).height,
        "zero_target_count": subset.filter(target_sign == 0).height,
        "mean_predictor_delta": subset.select(pl.col(f"{predictor}_volume_delta").mean()).item() if subset.height else None,
        "median_predictor_delta": subset.select(pl.col(f"{predictor}_volume_delta").median()).item() if subset.height else None,
        "mean_target_delta": subset.select(pl.col(f"{target}_volume_delta").mean()).item() if subset.height else None,
        "median_target_delta": subset.select(pl.col(f"{target}_volume_delta").median()).item() if subset.height else None,
        "target_p25": subset.select(pl.col(f"{target}_volume_delta").quantile(0.25)).item() if subset.height else None,
        "target_p75": subset.select(pl.col(f"{target}_volume_delta").quantile(0.75)).item() if subset.height else None,
        "pearson_corr_predictor_vs_target_delta": subset.select(
            pl.corr(f"{predictor}_volume_delta", f"{target}_volume_delta")
        ).item()
        if subset.height
        else None,
    }


def _target_decile_rows(study: pl.DataFrame, predictor: str, target: str, value_suffix: str, summary_type: str) -> list[dict]:
    value_col = f"{predictor}_{value_suffix}"
    non_null = study.filter(pl.col(value_col).is_not_null())
    if non_null.height < 10:
        return []
    deciled = non_null.with_columns(
        ((pl.col(value_col).rank(method="ordinal") - 1) * 10 / non_null.height)
        .floor()
        .cast(pl.Int64)
        .clip(0, 9)
        .add(1)
        .alias("predictor_decile")
    )
    return [
        _base_target_summary_row(
            predictor,
            target,
            summary_type,
            deciled.filter(pl.col("predictor_decile") == decile),
            predictor_decile=decile,
        )
        for decile in range(1, 11)
    ]


def _target_tail_rows(study: pl.DataFrame, predictor: str, target: str) -> list[dict]:
    value_col = f"{predictor}_volume_delta"
    rows = []
    empty_subset = study.head(0)
    positive = study.filter(pl.col(value_col) > 0)
    negative = study.filter(pl.col(value_col) < 0)
    tail_specs = [
        ("positive_top_20", positive, 0.80, ">="),
        ("positive_top_10", positive, 0.90, ">="),
        ("negative_bottom_20", negative, 0.20, "<="),
        ("negative_bottom_10", negative, 0.10, "<="),
    ]
    for label, frame, quantile, op in tail_specs:
        if frame.is_empty():
            subset = empty_subset
        else:
            threshold = frame.select(pl.col(value_col).quantile(quantile)).item()
            subset = frame.filter(pl.col(value_col) >= threshold) if op == ">=" else frame.filter(pl.col(value_col) <= threshold)
        rows.append(_base_target_summary_row(predictor, target, "target_tail", subset, tail=label))
    return rows


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in keys} for row in rows]
```

Update `summarize_macro_1550_delta_impulse`:

```python
def summarize_macro_1550_delta_impulse(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    available_targets = _available_target_windows(study)
    robust_targets = [target for target in ROBUST_TARGET_WINDOWS if target in available_targets]
    for predictor in PREDICTORS:
        for target in available_targets:
            rows.append(_target_pair_sign_row(study, predictor, target))
        for target in robust_targets:
            rows.extend(_target_decile_rows(study, predictor, target, "volume_delta", "target_raw_decile"))
            rows.extend(_target_decile_rows(study, predictor, target, "delta_imbalance", "target_imbalance_decile"))
            rows.extend(_target_tail_rows(study, predictor, target))
    return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)
```

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py::test_summarize_macro_1550_delta_impulse_adds_decile_and_tail_rows -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_1550_delta_impulse.py test/test_macro_1550_delta_impulse.py
git commit -m "feat: add macro 1550 robust impulse summaries"
```

---

### Task 6: Run full tests, generate outputs, record findings stub

**Files:**
- Create: `docs/reports/2026-05-13-macro-1550-delta-impulse-findings.md`
- Runtime writes:
  - `outputs/nq_macro_1550_delta_impulse.parquet`
  - `outputs/nq_macro_1550_delta_impulse_summary.parquet`

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run related regression tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Generate runtime outputs**

Run:

```bash
.venv/bin/python -m features.macro_1550_delta_impulse
```

Expected output:

```text
[OK] Wrote macro 15:50 delta impulse -> outputs/nq_macro_1550_delta_impulse.parquet
[OK] Wrote macro 15:50 delta impulse summary -> outputs/nq_macro_1550_delta_impulse_summary.parquet
```

- [ ] **Step 4: Inspect key summary rows**

Run:

```bash
.venv/bin/python - <<'PY'
import polars as pl
summary = pl.read_parquet("outputs/nq_macro_1550_delta_impulse_summary.parquet")
cols = [
    "summary_type",
    "predictor",
    "target_window",
    "n_signal_days",
    "opposite_rate",
    "same_rate",
    "pearson_corr_predictor_vs_target_delta",
    "median_target_delta_when_predictor_positive",
    "median_target_delta_when_predictor_negative",
]
print(
    summary.filter(
        (pl.col("summary_type") == "target_sign")
        & (pl.col("target_window") == "k350_00_09")
    )
    .select(cols)
    .sort("predictor")
)
print(
    summary.filter(
        (pl.col("summary_type") == "target_tail")
        & (pl.col("target_window") == "k350_00_09")
    )
    .select(["predictor", "tail", "n_days", "opposite_rate", "median_target_delta", "target_p25", "target_p75"])
    .sort(["predictor", "tail"])
)
PY
```

Expected: printed tables for `eth_only_pre350`, `rth_only_pre350`, `eth_rth_pre350`.

- [ ] **Step 5: Write findings stub**

Create `docs/reports/2026-05-13-macro-1550-delta-impulse-findings.md`:

```markdown
# Macro 15:50 Delta Impulse Findings

Date: 2026-05-13
Branch/worktree: `feat/macro-delta-reversal`

## Study Scope

Question: does accumulated ETH-only, RTH-only, or ETH+RTH volume delta before 15:50 ET predict the initial 15:50:00-15:50:09 ET volume-delta impulse?

Inputs used:

- `outputs/nq_globex_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_5s.parquet`

Outputs:

- `outputs/nq_macro_1550_delta_impulse.parquet`
- `outputs/nq_macro_1550_delta_impulse_summary.parquet`

## Main Predictors

- `eth_only_pre350`: Globex/session minute index `0..929`.
- `rth_only_pre350`: session minute index `930..1309`, RTH through 15:49 ET.
- `eth_rth_pre350`: session minute index `0..1309`, full pre-15:50 accumulated delta.

## Primary Target

- `k350_00_09`: 15:50:00-15:50:09 ET, macro 5-second buckets `0..1`.

## Results

Fill this section from `outputs/nq_macro_1550_delta_impulse_summary.parquet` after runtime generation.

## Caveats

- Study uses 5-second volume-delta buckets, not raw order-type-level imbalance.
- No price target is included.
- Findings describe volume-flow relationships, not trade recommendations.
```

- [ ] **Step 6: Commit final implementation and report stub**

```bash
git add features/macro_1550_delta_impulse.py test/test_macro_1550_delta_impulse.py docs/reports/2026-05-13-macro-1550-delta-impulse-findings.md
git commit -m "docs: record macro 1550 impulse findings stub"
```

---

## Self-Review Checklist

- Spec coverage: predictors, targets, output files, summary fields, tests, runtime behavior covered by tasks 1-6.
- TDD order: every production change has failing test step before implementation.
- No broad refactor: `features/macro_delta_reversal.py` remains unchanged.
- Data safety: uses existing parquet outputs; no `input-data/` writes; no raw tick eager reads.
- Commands: all use `.venv/bin/python`.
