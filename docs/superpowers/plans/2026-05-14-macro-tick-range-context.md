# Macro Tick Range Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tick-level price range context study that measures first-10-second and 5-second cumulative range formation for 15:50 and 15:59 as raw price %, % of candle range, % of macro range, and additive extension.

**Architecture:** Add `features/macro_tick_range_context.py` as a standalone tick-safe Polars pipeline. Runtime scans `input-data/merged_nq_ticks.parquet`, filters 15:50-15:59 ET before collection, computes macro/candle/window ranges from ticks, writes long-form study + summary parquet, and records findings. Tests use small eager fixtures.

**Tech Stack:** Python, Polars, pytest, parquet outputs, lazy tick scans, project virtualenv via `.venv/bin/python`.

---

## File Structure

- Create: `features/macro_tick_range_context.py`
  - Schema validation, tick normalization, candle/window range construction, additive extension metrics, summaries, writer/main.
- Create: `test/test_macro_tick_range_context.py`
  - In-memory tick fixtures for 15:50 and 15:59 windows.
- Create after runtime: `docs/reports/2026-05-14-macro-tick-range-context-findings.md`
- Runtime outputs:
  - `outputs/nq_macro_tick_range_context.parquet`
  - `outputs/nq_macro_tick_range_context_summary.parquet`

---

### Task 1: Add skeleton, validation, writer

**Files:**
- Create: `features/macro_tick_range_context.py`
- Create: `test/test_macro_tick_range_context.py`

- [ ] **Step 1: Write failing schema/writer tests**

Create `test/test_macro_tick_range_context.py`:

```python
from pathlib import Path

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


def _basic_macro_ticks() -> pl.DataFrame:
    return _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 10000),
            _t("2025-01-02T20:50:09Z", 0, 10020),
            _t("2025-01-02T20:50:59Z", 0, 10040),
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
```

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py::test_build_macro_tick_range_context_requires_tick_schema test/test_macro_tick_range_context.py::test_write_macro_tick_range_context_persists_outputs -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement skeleton**

Create `features/macro_tick_range_context.py`:

```python
from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

from utils.minute_bars import MARKET_TZ

TICK_INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_tick_range_context.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_tick_range_context_summary.parquet")
TICK_PRICE_DENOMINATOR = 4.0

REQUIRED_TICK_COLUMNS = {"ts_event", "intra_ts_rank", "price_ticks"}
CANDLE_SPECS = {"k350": (50, 0), "k359": (59, 108)}
WINDOW_SPECS = {f"00_{end * 5 + 4:02d}": (0, end * 5 + 4) for end in range(12)}
NAMED_WINDOWS = {
    "first_5s": (0, 4),
    "first_10s": (0, 9),
    "first_30s": (0, 29),
    "last_30s": (30, 59),
    "full_candle": (0, 59),
}
THRESHOLDS = [25.0, 50.0, 75.0, 90.0]


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _validate_tick_columns(frame: pl.DataFrame) -> None:
    missing = _missing_columns(frame, REQUIRED_TICK_COLUMNS)
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _safe_ratio(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return pl.when(denom != 0).then(numer / denom).otherwise(None)


def _prepare_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    _validate_tick_columns(ticks)
    return (
        ticks.select("ts_event", "intra_ts_rank", "price_ticks")
        .with_columns(
            pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC")),
            pl.col("intra_ts_rank").cast(pl.Int64),
            price=(pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR),
        )
        .with_columns(
            datetime_et=pl.col("ts_event").dt.convert_time_zone(MARKET_TZ),
        )
        .with_columns(
            date=pl.col("datetime_et").dt.date(),
            hour_et=pl.col("datetime_et").dt.hour(),
            minute_et=pl.col("datetime_et").dt.minute(),
            second_et=pl.col("datetime_et").dt.second(),
        )
        .filter((pl.col("hour_et") == 15) & (pl.col("minute_et").is_between(50, 59)))
        .sort(["date", "ts_event", "intra_ts_rank"])
    )


def _range_row(frame: pl.DataFrame, prefix: str) -> dict:
    if frame.is_empty():
        return {
            f"{prefix}_tick_count": 0,
            f"{prefix}_open": None,
            f"{prefix}_high": None,
            f"{prefix}_low": None,
            f"{prefix}_close": None,
            f"{prefix}_range_points": None,
        }
    return {
        f"{prefix}_tick_count": frame.height,
        f"{prefix}_open": frame.item(0, "price"),
        f"{prefix}_high": frame.select(pl.col("price").max()).item(),
        f"{prefix}_low": frame.select(pl.col("price").min()).item(),
        f"{prefix}_close": frame.item(frame.height - 1, "price"),
        f"{prefix}_range_points": frame.select(pl.col("price").max() - pl.col("price").min()).item(),
    }


def _add_metrics(row: dict) -> dict:
    window_range = row["window_range_points"]
    candle_open = row["candle_open"]
    candle_range = row["candle_range_points"]
    macro_range = row["macro_range_points"]
    row["range_raw_pct_of_open"] = (window_range / candle_open * 100.0) if window_range is not None and candle_open else None
    row["range_pct_of_candle"] = (window_range / candle_range * 100.0) if window_range is not None and candle_range else None
    row["range_pct_of_macro"] = (window_range / macro_range * 100.0) if window_range is not None and macro_range else None
    return row


def build_macro_tick_range_context(ticks: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    tick_df = ticks.collect() if isinstance(ticks, pl.LazyFrame) else ticks
    work = _prepare_ticks(tick_df)
    rows: list[dict] = []
    for date in work["date"].unique().sort().to_list():
        day = work.filter(pl.col("date") == date)
        macro = day.filter(pl.col("minute_et").is_between(50, 59))
        macro_stats = _range_row(macro, "macro")
        for candle, (minute, _bucket_start) in CANDLE_SPECS.items():
            candle_frame = day.filter(pl.col("minute_et") == minute)
            candle_stats = _range_row(candle_frame, "candle")
            all_windows = {**WINDOW_SPECS, **NAMED_WINDOWS}
            for window, (start_second, end_second) in all_windows.items():
                window_frame = candle_frame.filter(pl.col("second_et").is_between(start_second, end_second))
                row = {
                    "date": date,
                    "candle": candle,
                    "window": window,
                    "window_start_second": start_second,
                    "window_end_second": end_second,
                    **_range_row(window_frame, "window"),
                    **candle_stats,
                    **macro_stats,
                }
                rows.append(_add_metrics(row))
    return pl.DataFrame(rows, infer_schema_length=None).sort(["date", "candle", "window_start_second", "window_end_second", "window"])


def summarize_macro_tick_range_context(study: pl.DataFrame) -> pl.DataFrame:
    if study.is_empty():
        return pl.DataFrame({"summary_type": [], "candle": [], "window": [], "n_days": []})
    rows = []
    for (candle, window), subset in study.group_by(["candle", "window"], maintain_order=True):
        rows.append(
            {
                "summary_type": "window_baseline",
                "candle": candle,
                "window": window,
                "n_days": subset.height,
                "median_range_points": subset.select(pl.col("window_range_points").median()).item(),
                "median_range_raw_pct_of_open": subset.select(pl.col("range_raw_pct_of_open").median()).item(),
                "median_range_pct_of_candle": subset.select(pl.col("range_pct_of_candle").median()).item(),
                "median_range_pct_of_macro": subset.select(pl.col("range_pct_of_macro").median()).item(),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def write_macro_tick_range_context(
    input_path: str | Path = TICK_INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    summary_output_path: str | Path = SUMMARY_OUTPUT_PATH,
) -> tuple[Path, Path]:
    study = build_macro_tick_range_context(pl.scan_parquet(input_path))
    summary = summarize_macro_tick_range_context(study)
    output = Path(output_path)
    summary_output = Path(summary_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    study.write_parquet(output)
    summary.write_parquet(summary_output)
    return output, summary_output


def main() -> None:
    if not TICK_INPUT_PATH.exists():
        print(f"[ERROR] Input not found: {TICK_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    output, summary = write_macro_tick_range_context()
    print(f"[OK] Wrote macro tick range context -> {output}")
    print(f"[OK] Wrote macro tick range context summary -> {summary}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py::test_build_macro_tick_range_context_requires_tick_schema test/test_macro_tick_range_context.py::test_write_macro_tick_range_context_persists_outputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_tick_range_context.py test/test_macro_tick_range_context.py
git commit -m "feat: add macro tick range context skeleton"
```

---

### Task 2: Add additive extension metrics and 15:59 macro contribution

**Files:**
- Modify: `features/macro_tick_range_context.py`
- Modify: `test/test_macro_tick_range_context.py`

- [ ] **Step 1: Write failing range/additive test**

Append to `test/test_macro_tick_range_context.py`:

```python

def test_build_macro_tick_range_context_computes_first10_percentages_and_additive_extensions():
    ticks = _ticks(
        [
            _t("2025-01-02T20:50:00Z", 0, 400),  # 100
            _t("2025-01-02T20:50:04Z", 0, 408),  # 102
            _t("2025-01-02T20:50:09Z", 0, 396),  # 99 first10 range 3
            _t("2025-01-02T20:50:30Z", 0, 416),  # 104 candle high
            _t("2025-01-02T20:50:59Z", 0, 388),  # 97 candle low/range 7
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


def test_build_macro_tick_range_context_computes_k359_macro_contribution():
    out = build_macro_tick_range_context(_basic_macro_ticks())
    row = out.filter((pl.col("candle") == "k359") & (pl.col("window") == "full_candle")).row(0, named=True)

    assert row["k359_range_pct_of_macro"] is not None
    assert row["k359_macro_additive_total_extension_from_pre359_points"] is not None
```

- [ ] **Step 2: Run tests to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py::test_build_macro_tick_range_context_computes_first10_percentages_and_additive_extensions test/test_macro_tick_range_context.py::test_build_macro_tick_range_context_computes_k359_macro_contribution -q
```

Expected: FAIL with missing additive columns.

- [ ] **Step 3: Implement additive metrics**

In `_add_metrics`, after percentage metrics, add:

```python
    window_high = row["window_high"]
    window_low = row["window_low"]
    candle_high = row["candle_high"]
    candle_low = row["candle_low"]
    macro_high = row["macro_high"]
    macro_low = row["macro_low"]
    if None not in (window_high, window_low, candle_high, candle_low):
        row["candle_additive_high_extension_points"] = max(0.0, candle_high - window_high)
        row["candle_additive_low_extension_points"] = max(0.0, window_low - candle_low)
        row["candle_additive_total_extension_points"] = row["candle_additive_high_extension_points"] + row["candle_additive_low_extension_points"]
    else:
        row["candle_additive_high_extension_points"] = None
        row["candle_additive_low_extension_points"] = None
        row["candle_additive_total_extension_points"] = None
    if None not in (window_high, window_low, macro_high, macro_low):
        row["macro_additive_high_extension_points"] = max(0.0, macro_high - window_high)
        row["macro_additive_low_extension_points"] = max(0.0, window_low - macro_low)
        row["macro_additive_total_extension_points"] = row["macro_additive_high_extension_points"] + row["macro_additive_low_extension_points"]
    else:
        row["macro_additive_high_extension_points"] = None
        row["macro_additive_low_extension_points"] = None
        row["macro_additive_total_extension_points"] = None
    row["candle_additive_total_extension_pct_of_candle"] = (
        row["candle_additive_total_extension_points"] / candle_range * 100.0
        if row["candle_additive_total_extension_points"] is not None and candle_range
        else None
    )
    row["macro_additive_total_extension_pct_of_macro"] = (
        row["macro_additive_total_extension_points"] / macro_range * 100.0
        if row["macro_additive_total_extension_points"] is not None and macro_range
        else None
    )
```

In `build_macro_tick_range_context`, before looping candles, compute pre-359 stats:

```python
        pre359 = day.filter(pl.col("minute_et").is_between(50, 58))
        pre359_high = pre359.select(pl.col("price").max()).item() if not pre359.is_empty() else None
        pre359_low = pre359.select(pl.col("price").min()).item() if not pre359.is_empty() else None
```

After `row = _add_metrics(row)`, add k359 contribution fields:

```python
                if candle == "k359" and row["candle_range_points"] is not None and row["macro_range_points"]:
                    row["k359_range_pct_of_macro"] = row["candle_range_points"] / row["macro_range_points"] * 100.0
                else:
                    row["k359_range_pct_of_macro"] = None
                if candle == "k359" and None not in (pre359_high, pre359_low, row["candle_high"], row["candle_low"]):
                    hi_ext = max(0.0, row["candle_high"] - pre359_high)
                    lo_ext = max(0.0, pre359_low - row["candle_low"])
                    row["k359_macro_additive_high_extension_from_pre359_points"] = hi_ext
                    row["k359_macro_additive_low_extension_from_pre359_points"] = lo_ext
                    row["k359_macro_additive_total_extension_from_pre359_points"] = hi_ext + lo_ext
                    row["k359_macro_additive_total_extension_from_pre359_pct_of_macro"] = (
                        (hi_ext + lo_ext) / row["macro_range_points"] * 100.0 if row["macro_range_points"] else None
                    )
                else:
                    row["k359_macro_additive_high_extension_from_pre359_points"] = None
                    row["k359_macro_additive_low_extension_from_pre359_points"] = None
                    row["k359_macro_additive_total_extension_from_pre359_points"] = None
                    row["k359_macro_additive_total_extension_from_pre359_pct_of_macro"] = None
```

Append `rows.append(row)` after this block instead of appending `_add_metrics(row)` directly.

- [ ] **Step 4: Run tests to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py::test_build_macro_tick_range_context_computes_first10_percentages_and_additive_extensions test/test_macro_tick_range_context.py::test_build_macro_tick_range_context_computes_k359_macro_contribution -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_tick_range_context.py test/test_macro_tick_range_context.py
git commit -m "feat: add macro tick additive range context"
```

---

### Task 3: Add threshold and decile summaries

**Files:**
- Modify: `features/macro_tick_range_context.py`
- Modify: `test/test_macro_tick_range_context.py`

- [ ] **Step 1: Write failing summary test**

Append to `test/test_macro_tick_range_context.py`:

```python

def test_summarize_macro_tick_range_context_adds_threshold_rows():
    rows = []
    for i in range(12):
        day = f"2025-01-{i + 1:02d}"
        base = 40000 + i * 100
        rows.extend(
            [
                _t(f"{day}T20:50:00Z", 0, base),
                _t(f"{day}T20:50:09Z", 0, base + 4 * (i + 1)),
                _t(f"{day}T20:50:59Z", 0, base + 80),
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
```

- [ ] **Step 2: Run test to verify RED**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py::test_summarize_macro_tick_range_context_adds_threshold_rows -q
```

Expected: FAIL because threshold/decile rows are missing.

- [ ] **Step 3: Implement threshold/decile summaries**

Add helpers before `summarize_macro_tick_range_context`:

```python
def _rate(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


def _baseline_row(candle: str, window: str, subset: pl.DataFrame) -> dict:
    return {
        "summary_type": "window_baseline",
        "candle": candle,
        "window": window,
        "n_days": subset.height,
        "median_range_points": subset.select(pl.col("window_range_points").median()).item(),
        "mean_range_points": subset.select(pl.col("window_range_points").mean()).item(),
        "median_range_raw_pct_of_open": subset.select(pl.col("range_raw_pct_of_open").median()).item(),
        "mean_range_raw_pct_of_open": subset.select(pl.col("range_raw_pct_of_open").mean()).item(),
        "median_range_pct_of_candle": subset.select(pl.col("range_pct_of_candle").median()).item(),
        "mean_range_pct_of_candle": subset.select(pl.col("range_pct_of_candle").mean()).item(),
        "median_range_pct_of_macro": subset.select(pl.col("range_pct_of_macro").median()).item(),
        "mean_range_pct_of_macro": subset.select(pl.col("range_pct_of_macro").mean()).item(),
        "median_candle_additive_total_extension_points": subset.select(pl.col("candle_additive_total_extension_points").median()).item(),
        "median_macro_additive_total_extension_points": subset.select(pl.col("macro_additive_total_extension_points").median()).item(),
    }


def _threshold_row(summary_type: str, candle: str, window: str, subset: pl.DataFrame, metric: str, threshold: float) -> dict:
    valid = subset.filter(pl.col(metric).is_not_null())
    hits = valid.filter(pl.col(metric) >= threshold).height
    return {
        "summary_type": summary_type,
        "candle": candle,
        "window": window,
        "threshold": threshold,
        "n_days": valid.height,
        "hit_count": hits,
        "hit_rate": _rate(hits, valid.height),
        "median_metric": valid.select(pl.col(metric).median()).item() if valid.height else None,
    }


def _decile_rows(summary_type: str, candle: str, window: str, subset: pl.DataFrame, metric: str) -> list[dict]:
    valid = subset.filter(pl.col(metric).is_not_null()).sort([metric, "date"])
    unique_count = valid.select(pl.col(metric).n_unique()).item() if valid.height else 0
    if valid.height < 10 or unique_count < 10:
        return []
    deciled = valid.with_columns(
        (((pl.int_range(pl.len()) * 10) / pl.len()).floor().cast(pl.Int64).clip(0, 9) + 1).alias("decile")
    )
    rows = []
    for decile in range(1, 11):
        d = deciled.filter(pl.col("decile") == decile)
        rows.append(
            {
                "summary_type": summary_type,
                "candle": candle,
                "window": window,
                "decile_metric": metric,
                "decile": decile,
                "n_days": d.height,
                "median_metric": d.select(pl.col(metric).median()).item() if d.height else None,
                "median_range_points": d.select(pl.col("window_range_points").median()).item() if d.height else None,
                "median_range_pct_of_candle": d.select(pl.col("range_pct_of_candle").median()).item() if d.height else None,
                "median_range_pct_of_macro": d.select(pl.col("range_pct_of_macro").median()).item() if d.height else None,
            }
        )
    return rows


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in keys} for row in rows]
```

Replace `summarize_macro_tick_range_context`:

```python
def summarize_macro_tick_range_context(study: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    if study.is_empty():
        return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)
    for (candle, window), subset in study.group_by(["candle", "window"], maintain_order=True):
        rows.append(_baseline_row(candle, window, subset))
        for threshold in THRESHOLDS:
            rows.append(_threshold_row("threshold_pct_of_candle", candle, window, subset, "range_pct_of_candle", threshold))
            rows.append(_threshold_row("threshold_pct_of_macro", candle, window, subset, "range_pct_of_macro", threshold))
            if candle == "k359" and window == "full_candle":
                rows.append(
                    _threshold_row(
                        "threshold_k359_range_pct_of_macro",
                        candle,
                        window,
                        subset,
                        "k359_range_pct_of_macro",
                        threshold,
                    )
                )
        rows.extend(_decile_rows("decile_range_raw_pct_of_open", candle, window, subset, "range_raw_pct_of_open"))
        rows.extend(_decile_rows("decile_range_pct_of_candle", candle, window, subset, "range_pct_of_candle"))
        rows.extend(_decile_rows("decile_range_pct_of_macro", candle, window, subset, "range_pct_of_macro"))
    return pl.DataFrame(_normalize_summary_rows(rows), infer_schema_length=None)
```

- [ ] **Step 4: Run test to verify GREEN**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py::test_summarize_macro_tick_range_context_adds_threshold_rows -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/macro_tick_range_context.py test/test_macro_tick_range_context.py
git commit -m "feat: add macro tick range summaries"
```

---

### Task 4: Make runtime tick scan memory-safe and generate report

**Files:**
- Modify: `features/macro_tick_range_context.py`
- Create: `docs/reports/2026-05-14-macro-tick-range-context-findings.md`

- [ ] **Step 1: Run focused and related tests**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py -q
.venv/bin/python -m pytest test/test_macro_tick_range_context.py test/test_macro_bucket_path.py test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Improve lazy scan before runtime if needed**

If runtime `build_macro_tick_range_context(pl.scan_parquet(...))` collects too much, refactor:

```python
def _prepare_ticks(ticks: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    _validate_tick_columns(ticks.collect_schema() if isinstance(ticks, pl.LazyFrame) else ticks)
    selected = ticks.select("ts_event", "intra_ts_rank", "price_ticks") if isinstance(ticks, pl.LazyFrame) else ticks.select("ts_event", "intra_ts_rank", "price_ticks")
    # keep ET filter lazy until collect when ticks is LazyFrame
```

Ensure final implementation filters to 15:50-15:59 ET before collecting.

- [ ] **Step 3: Run runtime**

```bash
.venv/bin/python -m features.macro_tick_range_context
```

Expected:

```text
[OK] Wrote macro tick range context -> outputs/nq_macro_tick_range_context.parquet
[OK] Wrote macro tick range context summary -> outputs/nq_macro_tick_range_context_summary.parquet
```

- [ ] **Step 4: Extract findings**

```bash
.venv/bin/python - <<'PY'
import polars as pl
study = pl.read_parquet("outputs/nq_macro_tick_range_context.parquet")
summary = pl.read_parquet("outputs/nq_macro_tick_range_context_summary.parquet")
print("study", study.shape)
print("summary", summary.shape)
keys = [
    ("k350", "first_10s"),
    ("k350", "full_candle"),
    ("k359", "first_10s"),
    ("k359", "full_candle"),
]
for candle, window in keys:
    print("\n", candle, window)
    print(summary.filter((pl.col("summary_type") == "window_baseline") & (pl.col("candle") == candle) & (pl.col("window") == window)).to_dicts())
print("\nK359 macro thresholds")
print(summary.filter(pl.col("summary_type") == "threshold_k359_range_pct_of_macro").select(["threshold", "n_days", "hit_rate", "median_metric"]).sort("threshold"))
PY
```

- [ ] **Step 5: Write findings report**

Create `docs/reports/2026-05-14-macro-tick-range-context-findings.md` with:

```markdown
# Macro Tick Range Context Findings

Date: 2026-05-14
Branch/worktree: `feat/macro-delta-reversal`

## Study Scope

Tick-level price range context for 15:50 and 15:59.

Inputs:
- `input-data/merged_nq_ticks.parquet`

Outputs:
- `outputs/nq_macro_tick_range_context.parquet`
- `outputs/nq_macro_tick_range_context_summary.parquet`

## Runtime Shapes

Fill from runtime.

## 15:50 First 10 Seconds

Report:
- median raw % of open
- median % of 15:50 range
- median % of macro range
- median additive candle/macro extension after first 10s

## 15:59 Range Context

Report:
- full 15:59 range as % of macro range
- additive contribution beyond pre-15:59 macro range
- first 10s and 5s accumulation stats

## Current Best Read

Fill after reviewing outputs.

## Caveats

- Tick-level range only.
- No volume delta or price prediction target.
- No trade recommendations.
```

Replace placeholders with values from Step 4.

- [ ] **Step 6: Final verification and commit**

```bash
.venv/bin/python -m pytest test/test_macro_tick_range_context.py test/test_macro_bucket_path.py test/test_macro_1550_delta_impulse.py test/test_macro_delta_reversal.py -q
git add features/macro_tick_range_context.py test/test_macro_tick_range_context.py docs/reports/2026-05-14-macro-tick-range-context-findings.md
git commit -m "docs: record macro tick range context findings"
git status --short
```

Expected: tests PASS and worktree clean.

---

## Self-Review Checklist

- Spec coverage: tick schema, 15:50/15:59, 5s windows, raw %, % candle, % macro, additive range extension, 15:59 macro contribution, summaries, report covered.
- Existing-work alignment: reuses ideas from `viz/macro_analysis.py` but extends to tick-level 5s/10s windows.
- Data safety: runtime uses lazy scan and filters macro window before collecting.
- TDD: each behavior has failing test before implementation.
- Commands: use `.venv/bin/python` only.
