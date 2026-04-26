# Tick Volume Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tick-based NQ volume-delta parquet outputs: full Globex 1-minute, macro 1-minute, and macro 5-second.

**Architecture:** Add one script, `volume_delta.py`, mirroring `tick_density.py`: lazy Polars scans, schema validation, reusable delta aggregations, write helpers. Add pytest coverage in `test/test_volume_delta.py` using synthetic parquet fixtures only.

**Tech Stack:** Python 3, Polars lazy API, PyArrow parquet schema checks, pytest.

---

## File Structure

- Create: `volume_delta.py`
  - Owns all volume-delta transforms and write helpers.
  - Uses `utils.tick_data.get_tick_schema` and existing tick schema conventions.
- Create: `test/test_volume_delta.py`
  - Tests schema validation, delta metrics, Globex session indexing, macro filtering, complete 5-second grid, ratio null handling, write helpers.
- Existing unchanged: `utils/tick_data.py`
  - No shared helper change needed for v1.

---

### Task 1: Schema validation and reusable delta aggregation

**Files:**
- Create: `volume_delta.py`
- Test: `test/test_volume_delta.py`

- [ ] **Step 1: Write failing tests for required schema and basic metrics**

Create `test/test_volume_delta.py` with:

```python
from pathlib import Path

import polars as pl
import pytest

from volume_delta import build_macro_volume_delta_1m


def _write_ticks(path: Path, rows: dict) -> None:
    pl.DataFrame(rows).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)


def test_build_macro_volume_delta_1m_requires_tick_schema(tmp_path: Path):
    path = tmp_path / "bad_ticks.parquet"
    pl.DataFrame({"ts_event": ["2025-01-02T20:50:00Z"]}).with_columns(
        pl.col("ts_event").str.to_datetime(time_zone="UTC")
    ).write_parquet(path)

    with pytest.raises(ValueError, match="Missing tick columns"):
        build_macro_volume_delta_1m(path)


def test_build_macro_volume_delta_1m_computes_signed_size_and_diagnostics(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:10Z",
                "2025-01-02T20:50:20Z",
                "2025-01-02T20:50:30Z",
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [2, 1, 0, 2],
            "price_ticks": [84000, 84004, 84008, 84012],
            "size": [5, 3, 7, 2],
        },
    )

    out = build_macro_volume_delta_1m(path).collect(engine="streaming")

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["buy_size"] == 7
    assert row["sell_size"] == 3
    assert row["none_size"] == 7
    assert row["classified_size"] == 10
    assert row["total_size"] == 17
    assert row["volume_delta"] == 4
    assert row["delta_imbalance"] == pytest.approx(0.4)
    assert row["buy_ticks"] == 2
    assert row["sell_ticks"] == 1
    assert row["none_ticks"] == 1
    assert row["tick_delta"] == 1
    assert row["classified_share"] == pytest.approx(10 / 17)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'volume_delta'`.

- [ ] **Step 3: Implement minimal schema validation, scan, agg, macro 1m builder**

Create `volume_delta.py` with:

```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from utils.tick_data import TICK_COLUMNS, get_tick_schema

INPUT_PATH = Path("input-data/merged_nq_ticks.parquet")
OUTPUT_GLOBEX_1M_PATH = Path("outputs/nq_globex_volume_delta_1m.parquet")
OUTPUT_MACRO_1M_PATH = Path("outputs/nq_macro_volume_delta_1m.parquet")
OUTPUT_MACRO_5S_PATH = Path("outputs/nq_macro_volume_delta_5s.parquet")

ET_TZ = "America/New_York"
MACRO_START_MINUTE = 50
MACRO_END_MINUTE = 60

DELTA_COLUMNS = [
    "buy_size",
    "sell_size",
    "none_size",
    "classified_size",
    "total_size",
    "volume_delta",
    "delta_imbalance",
    "buy_ticks",
    "sell_ticks",
    "none_ticks",
    "tick_delta",
    "classified_share",
]


def _validate_tick_schema(path: str | Path) -> None:
    schema = get_tick_schema(path)
    missing = [column for column in TICK_COLUMNS if column not in schema.names]
    if missing:
        raise ValueError(f"Missing tick columns: {missing}")


def _scan_required_tick_columns(path: str | Path) -> pl.LazyFrame:
    _validate_tick_schema(path)
    return pl.scan_parquet(path).select(TICK_COLUMNS)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def _delta_agg() -> list[pl.Expr]:
    buy_size = pl.when(pl.col("side") == 2).then(pl.col("size")).otherwise(0).sum()
    sell_size = pl.when(pl.col("side") == 1).then(pl.col("size")).otherwise(0).sum()
    none_size = pl.when(pl.col("side") == 0).then(pl.col("size")).otherwise(0).sum()
    buy_ticks = (pl.col("side") == 2).sum()
    sell_ticks = (pl.col("side") == 1).sum()
    none_ticks = (pl.col("side") == 0).sum()

    classified_size = buy_size + sell_size
    total_size = classified_size + none_size
    volume_delta = buy_size - sell_size
    tick_delta = buy_ticks - sell_ticks

    return [
        buy_size.alias("buy_size"),
        sell_size.alias("sell_size"),
        none_size.alias("none_size"),
        classified_size.alias("classified_size"),
        total_size.alias("total_size"),
        volume_delta.alias("volume_delta"),
        _safe_ratio(volume_delta, classified_size).alias("delta_imbalance"),
        buy_ticks.alias("buy_ticks"),
        sell_ticks.alias("sell_ticks"),
        none_ticks.alias("none_ticks"),
        tick_delta.alias("tick_delta"),
        _safe_ratio(classified_size, total_size).alias("classified_share"),
    ]


def _with_et_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        datetime_et=pl.col("ts_event").dt.convert_time_zone(ET_TZ),
        datetime_utc=pl.col("ts_event").dt.truncate("1m"),
    )


def build_macro_volume_delta_1m(path: str | Path) -> pl.LazyFrame:
    """Return lazy 1-minute volume-delta rows for 15:50-16:00 ET."""
    return (
        _with_et_columns(_scan_required_tick_columns(path))
        .with_columns(
            trade_date_et=pl.col("datetime_et").dt.date(),
            macro_minute_index=pl.col("datetime_et").dt.minute(),
        )
        .filter(
            (pl.col("datetime_et").dt.hour() == 15)
            & (pl.col("macro_minute_index") >= MACRO_START_MINUTE)
            & (pl.col("macro_minute_index") < MACRO_END_MINUTE)
        )
        .group_by("datetime_utc", "trade_date_et", "macro_minute_index")
        .agg(*_delta_agg())
        .sort("datetime_utc")
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add volume_delta.py test/test_volume_delta.py
git commit -m "feat: add macro minute volume delta metrics"
```

---

### Task 2: Full Globex 1-minute builder

**Files:**
- Modify: `volume_delta.py`
- Modify: `test/test_volume_delta.py`

- [ ] **Step 1: Add failing tests for Globex session indexing**

Append to `test/test_volume_delta.py`:

```python
from volume_delta import build_globex_volume_delta_1m


def test_build_globex_volume_delta_1m_uses_1800_to_1700_et_trade_date(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T22:59:00Z",  # 17:59 ET, outside next Globex session
                "2025-01-02T23:00:00Z",  # 18:00 ET, trade date 2025-01-03
                "2025-01-03T20:50:00Z",  # 15:50 ET, same trade date
                "2025-01-03T22:00:00Z",  # 17:00 ET, excluded
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [2, 2, 1, 1],
            "price_ticks": [84000, 84004, 84008, 84012],
            "size": [1, 2, 3, 4],
        },
    )

    out = build_globex_volume_delta_1m(path).collect(engine="streaming")

    assert out.select("trade_date_et").to_series().cast(pl.String).to_list() == ["2025-01-03", "2025-01-03"]
    assert out.select("session_minute_index").to_series().to_list() == [0, 1310]
    assert out.select("volume_delta").to_series().to_list() == [2, -3]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py::test_build_globex_volume_delta_1m_uses_1800_to_1700_et_trade_date -q
```

Expected: FAIL with import error for `build_globex_volume_delta_1m`.

- [ ] **Step 3: Implement Globex builder**

Modify `volume_delta.py` by adding this function after `build_macro_volume_delta_1m`:

```python

def build_globex_volume_delta_1m(path: str | Path) -> pl.LazyFrame:
    """Return lazy 1-minute volume-delta rows for 18:00-17:00 ET Globex sessions."""
    minute_of_day = pl.col("datetime_et").dt.hour() * 60 + pl.col("datetime_et").dt.minute()
    session_start_minute = 18 * 60
    session_end_minute = 17 * 60

    return (
        _with_et_columns(_scan_required_tick_columns(path))
        .with_columns(
            minute_of_day=minute_of_day,
            trade_date_et=pl.when(minute_of_day >= session_start_minute)
            .then(pl.col("datetime_et").dt.offset_by("1d").dt.date())
            .otherwise(pl.col("datetime_et").dt.date()),
            session_minute_index=pl.when(minute_of_day >= session_start_minute)
            .then(minute_of_day - session_start_minute)
            .otherwise(minute_of_day + (24 * 60 - session_start_minute)),
        )
        .filter(pl.col("minute_of_day") >= session_start_minute)
        .filter(pl.col("minute_of_day") < 24 * 60)
        .group_by("datetime_utc", "trade_date_et", "session_minute_index")
        .agg(*_delta_agg())
        .sort("datetime_utc")
    )
```

Then replace the two `.filter(...)` calls with the correct Globex mask:

```python
        .filter((pl.col("minute_of_day") >= session_start_minute) | (pl.col("minute_of_day") < session_end_minute))
```

Final function must include only one filter line with the OR mask.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add volume_delta.py test/test_volume_delta.py
git commit -m "feat: add globex minute volume delta"
```

---

### Task 3: Macro 5-second builder with complete bucket grid

**Files:**
- Modify: `volume_delta.py`
- Modify: `test/test_volume_delta.py`

- [ ] **Step 1: Add failing tests for 120 macro 5-second buckets**

Append to `test/test_volume_delta.py`:

```python
from volume_delta import build_macro_volume_delta_5s


def test_build_macro_volume_delta_5s_emits_120_buckets_with_empty_rows(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _write_ticks(
        path,
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:04Z",
                "2025-01-02T20:50:05Z",
                "2025-01-02T20:59:59Z",
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [2, 1, 0, 2],
            "price_ticks": [84000, 84004, 84008, 84012],
            "size": [5, 3, 7, 2],
        },
    )

    out = build_macro_volume_delta_5s(path).collect(engine="streaming")

    assert out.height == 120
    assert out.select("macro_bucket_index").to_series().to_list() == list(range(120))
    first = out.row(0, named=True)
    assert first["buy_size"] == 5
    assert first["sell_size"] == 3
    assert first["volume_delta"] == 2
    assert first["is_empty"] is False
    second = out.row(1, named=True)
    assert second["none_size"] == 7
    assert second["classified_size"] == 0
    assert second["delta_imbalance"] is None
    assert second["classified_share"] == 0.0
    assert out.row(2, named=True)["is_empty"] is True
    assert out.row(119, named=True)["buy_size"] == 2
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py::test_build_macro_volume_delta_5s_emits_120_buckets_with_empty_rows -q
```

Expected: FAIL with import error for `build_macro_volume_delta_5s`.

- [ ] **Step 3: Implement 5-second builder**

Modify `volume_delta.py` by adding these constants near macro constants:

```python
MACRO_5S_BUCKETS = 120
SECONDS_PER_MACRO_BUCKET = 5
```

Add this function after `build_macro_volume_delta_1m`:

```python

def build_macro_volume_delta_5s(path: str | Path) -> pl.LazyFrame:
    """Return lazy 5-second volume-delta rows for 15:50-16:00 ET with empty buckets."""
    base = (
        _scan_required_tick_columns(path)
        .with_columns(
            datetime_et=pl.col("ts_event").dt.convert_time_zone(ET_TZ),
            datetime_utc=pl.col("ts_event").dt.truncate("5s"),
        )
        .with_columns(
            trade_date_et=pl.col("datetime_et").dt.date(),
            minute=pl.col("datetime_et").dt.minute(),
            second=pl.col("datetime_et").dt.second(),
        )
        .filter(
            (pl.col("datetime_et").dt.hour() == 15)
            & (pl.col("minute") >= MACRO_START_MINUTE)
            & (pl.col("minute") < MACRO_END_MINUTE)
        )
        .with_columns(
            macro_bucket_index=((pl.col("minute") - MACRO_START_MINUTE) * 60 + pl.col("second"))
            // SECONDS_PER_MACRO_BUCKET
        )
    )

    aggregated = (
        base.group_by("datetime_utc", "trade_date_et", "macro_bucket_index")
        .agg(*_delta_agg())
    )

    dates = base.select("trade_date_et").unique()
    bucket_grid = dates.join(
        pl.LazyFrame({"macro_bucket_index": list(range(MACRO_5S_BUCKETS))}),
        how="cross",
    ).with_columns(
        datetime_et=pl.datetime(
            pl.col("trade_date_et").dt.year(),
            pl.col("trade_date_et").dt.month(),
            pl.col("trade_date_et").dt.day(),
            15,
            50,
            0,
            time_zone=ET_TZ,
        ).dt.offset_by((pl.col("macro_bucket_index") * SECONDS_PER_MACRO_BUCKET).cast(pl.String) + "s"),
    ).with_columns(
        datetime_utc=pl.col("datetime_et").dt.convert_time_zone("UTC")
    ).select("datetime_utc", "trade_date_et", "macro_bucket_index")

    return (
        bucket_grid.join(aggregated, on=["datetime_utc", "trade_date_et", "macro_bucket_index"], how="left")
        .with_columns(
            buy_size=pl.col("buy_size").fill_null(0),
            sell_size=pl.col("sell_size").fill_null(0),
            none_size=pl.col("none_size").fill_null(0),
            classified_size=pl.col("classified_size").fill_null(0),
            total_size=pl.col("total_size").fill_null(0),
            volume_delta=pl.col("volume_delta").fill_null(0),
            buy_ticks=pl.col("buy_ticks").fill_null(0),
            sell_ticks=pl.col("sell_ticks").fill_null(0),
            none_ticks=pl.col("none_ticks").fill_null(0),
            tick_delta=pl.col("tick_delta").fill_null(0),
        )
        .with_columns(
            delta_imbalance=_safe_ratio(pl.col("volume_delta"), pl.col("classified_size")),
            classified_share=_safe_ratio(pl.col("classified_size"), pl.col("total_size")),
            is_empty=(pl.col("total_size") == 0),
        )
        .sort("datetime_utc")
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py -q
```

Expected: all volume-delta tests pass.

- [ ] **Step 5: Commit**

```bash
git add volume_delta.py test/test_volume_delta.py
git commit -m "feat: add macro five second volume delta"
```

---

### Task 4: Write helpers and CLI entrypoint

**Files:**
- Modify: `volume_delta.py`
- Modify: `test/test_volume_delta.py`

- [ ] **Step 1: Add failing tests for write helpers**

Append to `test/test_volume_delta.py`:

```python
from volume_delta import (
    write_globex_volume_delta_1m,
    write_macro_volume_delta_1m,
    write_macro_volume_delta_5s,
)


def test_write_volume_delta_helpers_create_parquet_outputs(tmp_path: Path):
    input_path = tmp_path / "ticks.parquet"
    _write_ticks(
        input_path,
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "intra_ts_rank": [0],
            "side": [2],
            "price_ticks": [84000],
            "size": [5],
        },
    )
    globex_path = tmp_path / "globex.parquet"
    macro_1m_path = tmp_path / "macro_1m.parquet"
    macro_5s_path = tmp_path / "macro_5s.parquet"

    assert write_globex_volume_delta_1m(input_path, globex_path) == globex_path
    assert write_macro_volume_delta_1m(input_path, macro_1m_path) == macro_1m_path
    assert write_macro_volume_delta_5s(input_path, macro_5s_path) == macro_5s_path

    assert pl.read_parquet(globex_path).height == 1
    assert pl.read_parquet(macro_1m_path).height == 1
    assert pl.read_parquet(macro_5s_path).height == 120
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py::test_write_volume_delta_helpers_create_parquet_outputs -q
```

Expected: FAIL with import error for write helpers.

- [ ] **Step 3: Implement write helpers and entrypoint**

Append to `volume_delta.py`:

```python

def _sink(lf: pl.LazyFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(output)
    return output


def write_globex_volume_delta_1m(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_GLOBEX_1M_PATH,
) -> Path:
    """Write full Globex 1-minute volume-delta parquet and return output path."""
    return _sink(build_globex_volume_delta_1m(input_path), output_path)


def write_macro_volume_delta_1m(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_MACRO_1M_PATH,
) -> Path:
    """Write 15:50-16:00 ET 1-minute volume-delta parquet and return output path."""
    return _sink(build_macro_volume_delta_1m(input_path), output_path)


def write_macro_volume_delta_5s(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_MACRO_5S_PATH,
) -> Path:
    """Write 15:50-16:00 ET 5-second volume-delta parquet and return output path."""
    return _sink(build_macro_volume_delta_5s(input_path), output_path)


def main() -> None:
    outputs = [
        write_globex_volume_delta_1m(),
        write_macro_volume_delta_1m(),
        write_macro_volume_delta_5s(),
    ]
    for output in outputs:
        print(f"[OK] Wrote volume delta → {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all volume-delta tests**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py -q
```

Expected: all volume-delta tests pass.

- [ ] **Step 5: Commit**

```bash
git add volume_delta.py test/test_volume_delta.py
git commit -m "feat: write volume delta outputs"
```

---

### Task 5: Final verification

**Files:**
- No code changes expected unless verification reveals bug.

- [ ] **Step 1: Run focused volume-delta tests**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_volume_delta.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run relevant existing tests**

Run:

```bash
source .venv/bin/activate
python -m pytest test/test_tick_data.py test/test_tick_density.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run full baseline and document known unrelated failure**

Run:

```bash
source .venv/bin/activate
python -m pytest test features/lrlr/test features/trend -q
```

Expected: current known unrelated failure may remain:

```text
FAILED test/test_session_tagger.py::test_process_file_writes_minute_base_with_datetime_utc
AssertionError: assert 'nq_1m.parquet' == 'nq_minute_base.parquet'
```

Do not fix this unless user expands scope.

- [ ] **Step 4: Run script only if source data exists in worktree**

Check:

```bash
test -f input-data/merged_nq_ticks.parquet && echo exists || echo missing
```

If `exists`, run:

```bash
source .venv/bin/activate
python volume_delta.py
```

Expected output includes:

```text
[OK] Wrote volume delta → outputs/nq_globex_volume_delta_1m.parquet
[OK] Wrote volume delta → outputs/nq_macro_volume_delta_1m.parquet
[OK] Wrote volume delta → outputs/nq_macro_volume_delta_5s.parquet
```

If `missing`, document that generated-output smoke test was skipped because large input data is not present in worktree.

- [ ] **Step 5: Commit verification notes only if files changed**

Run:

```bash
git status --short
```

If clean, no commit. If code/test fixes were needed, commit them with:

```bash
git add volume_delta.py test/test_volume_delta.py
git commit -m "fix: stabilize volume delta implementation"
```

---

## Self-Review

Spec coverage:
- Full Globex 1-minute output: Task 2 + Task 4.
- Macro 1-minute output: Task 1 + Task 4.
- Macro 5-second output with empty buckets: Task 3 + Task 4.
- Delta definition excluding `side == 0`: Task 1 tests + implementation.
- Diagnostics/ratios: Task 1 and Task 3 tests.
- Schema failure: Task 1.
- Synthetic fixtures only: all tests use temp parquet fixtures.

Placeholder scan: no placeholder markers, vague edge-case instructions, or undefined function names remain.

Type consistency:
- Function names match across tests and implementation.
- Output column names match design spec.
- Paths match design spec.
