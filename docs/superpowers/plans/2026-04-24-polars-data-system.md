# Polars Data System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace active pandas-first data handling with Polars-first IO, transforms, tests, and tick-data-safe helpers.

**Architecture:** Keep existing entry scripts and module layout. Centralize minute-bar normalization in `utils/minute_bars.py`, add `utils/tick_data.py` for lazy tick parquet access, then migrate downstream scripts/tests to consume `pl.DataFrame`/`pl.LazyFrame`. Plotting code may convert small final frames to pandas/numpy only where matplotlib requires it.

**Tech Stack:** Python 3.12, Polars, PyArrow metadata for parquet schema, NumPy for math/model interfaces, Matplotlib for plots, pytest.

---

## File Map

- Modify `requirements.txt`: create explicit runtime/test dependencies including `polars`.
- Modify `utils/minute_bars.py`: convert all helpers to Polars.
- Create `utils/tick_data.py`: lazy tick scans, schema metadata, bounded collect/aggregation helpers.
- Modify `session_tagger.py`: write Polars parquet.
- Modify `macro_outcomes.py`: compute outcomes with Polars.
- Modify `features/pm_3pm.py`, `features/pm_macro_interactions.py`: Polars transforms.
- Modify `features/macro_fvg_study.py`: Polars public API, Python loops over row dicts only where event path is stateful.
- Modify `utils/helper.py`: Polars news joins.
- Modify `viz/*.py`: Polars reads/prep; pandas only at plotting boundary if unavoidable.
- Modify `test/*.py`: Polars fixtures/assertions.
- Add `test/test_tick_data.py`: metadata/lazy/bounded behavior.

---

### Task 1: Dependencies + Tick Helpers

**Files:**
- Create: `requirements.txt`
- Create: `utils/tick_data.py`
- Test: `test/test_tick_data.py`

- [ ] **Step 1: Write failing tick helper tests**

Create `test/test_tick_data.py`:

```python
from pathlib import Path

import polars as pl
import pytest

from utils.tick_data import (
    TICK_COLUMNS,
    collect_tick_window,
    get_tick_schema,
    scan_tick_data,
    ticks_to_minute_bars,
)


def test_scan_tick_data_returns_lazyframe_without_collecting(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame(
        {
            "ts_event": [
                "2025-01-02T20:50:00.000000000Z",
                "2025-01-02T20:50:00.500000000Z",
            ],
            "intra_ts_rank": [0, 1],
            "side": [1, 2],
            "price_ticks": [84000, 84004],
            "size": [1, 3],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    lf = scan_tick_data(path)

    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().height == 2


def test_get_tick_schema_reads_expected_columns(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame(
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "intra_ts_rank": [0],
            "side": [2],
            "price_ticks": [84000],
            "size": [5],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    schema = get_tick_schema(path)

    assert set(schema.names) == set(TICK_COLUMNS)


def test_collect_tick_window_requires_bounded_time_range(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame(
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "intra_ts_rank": [0],
            "side": [2],
            "price_ticks": [84000],
            "size": [5],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    with pytest.raises(ValueError, match="bounded start/end"):
        collect_tick_window(path, start_utc=None, end_utc=None)


def test_collect_tick_window_filters_before_collect(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame(
        {
            "ts_event": [
                "2025-01-02T20:49:59Z",
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:30Z",
                "2025-01-02T20:51:00Z",
            ],
            "intra_ts_rank": [0, 0, 0, 0],
            "side": [1, 2, 2, 1],
            "price_ticks": [83996, 84000, 84008, 84004],
            "size": [1, 2, 3, 4],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    out = collect_tick_window(
        path,
        start_utc="2025-01-02T20:50:00Z",
        end_utc="2025-01-02T20:51:00Z",
    )

    assert out.height == 2
    assert out["price_ticks"].to_list() == [84000, 84008]


def test_ticks_to_minute_bars_aggregates_price_and_volume(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame(
        {
            "ts_event": [
                "2025-01-02T20:50:00Z",
                "2025-01-02T20:50:30Z",
                "2025-01-02T20:51:00Z",
            ],
            "intra_ts_rank": [0, 0, 0],
            "side": [2, 1, 2],
            "price_ticks": [84000, 84008, 84004],
            "size": [2, 3, 4],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    bars = ticks_to_minute_bars(
        scan_tick_data(path),
        start_utc="2025-01-02T20:50:00Z",
        end_utc="2025-01-02T20:52:00Z",
    )

    assert bars.height == 2
    first = bars.row(0, named=True)
    assert first["Open"] == 21000.0
    assert first["High"] == 21002.0
    assert first["Low"] == 21000.0
    assert first["Close"] == 21002.0
    assert first["Volume"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest test/test_tick_data.py -q
```

Expected: FAIL because `utils.tick_data` does not exist.

- [ ] **Step 3: Add dependencies**

Create `requirements.txt`:

```text
polars>=1.40
pyarrow>=24
numpy>=2
matplotlib>=3.10
scipy>=1.17
scikit-learn>=1.8
statsmodels>=0.14
pytest>=9
```

- [ ] **Step 4: Implement `utils/tick_data.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

TICK_COLUMNS = ["ts_event", "intra_ts_rank", "side", "price_ticks", "size"]
TICK_PRICE_DENOMINATOR = 4.0


def get_tick_schema(path: str | Path) -> pa.Schema:
    return pq.ParquetFile(path).schema_arrow


def scan_tick_data(path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(path).select(TICK_COLUMNS)


def _dt(value: Any) -> pl.Expr:
    return pl.lit(value).str.to_datetime(time_zone="UTC") if isinstance(value, str) else pl.lit(value)


def _bounded_filter(lf: pl.LazyFrame, start_utc: Any, end_utc: Any) -> pl.LazyFrame:
    if start_utc is None or end_utc is None:
        raise ValueError("Tick collection requires bounded start/end UTC timestamps.")
    return lf.filter((pl.col("ts_event") >= _dt(start_utc)) & (pl.col("ts_event") < _dt(end_utc)))


def collect_tick_window(path: str | Path, start_utc: Any, end_utc: Any) -> pl.DataFrame:
    return _bounded_filter(scan_tick_data(path), start_utc, end_utc).collect(streaming=True)


def ticks_to_minute_bars(lf: pl.LazyFrame, start_utc: Any, end_utc: Any) -> pl.DataFrame:
    bounded = _bounded_filter(lf, start_utc, end_utc)
    return (
        bounded.with_columns(
            datetime_utc=pl.col("ts_event").dt.truncate("1m"),
            price=pl.col("price_ticks").cast(pl.Float64) / TICK_PRICE_DENOMINATOR,
        )
        .group_by("datetime_utc")
        .agg(
            pl.col("price").first().alias("Open"),
            pl.col("price").max().alias("High"),
            pl.col("price").min().alias("Low"),
            pl.col("price").last().alias("Close"),
            pl.col("size").sum().alias("Volume"),
        )
        .sort("datetime_utc")
        .collect(streaming=True)
    )
```

- [ ] **Step 5: Run tick tests**

Run:

```bash
.venv/bin/python -m pytest test/test_tick_data.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt utils/tick_data.py test/test_tick_data.py
git commit -m "feat: add polars tick data helpers"
```

---

### Task 2: Minute Bar Core + Session Tagger

**Files:**
- Modify: `utils/minute_bars.py`
- Modify: `session_tagger.py`
- Modify: `test/test_minute_bars.py`
- Modify: `test/test_session_tagger.py`

- [ ] **Step 1: Convert tests to Polars and assert return types**

Replace `test/test_minute_bars.py` with Polars fixtures equivalent to current tests. Use `pl.DataFrame`, `.item(row, column)`, `.dtypes`, `.to_list()`. Keep test names. Expected dtype assertion: `out.schema["datetime_utc"] == pl.Datetime("us", "UTC")` or `pl.Datetime("ns", "UTC")` depending input; prefer checking `out.schema["datetime_utc"].time_zone == "UTC"`.

Replace `test/test_session_tagger.py` to write CSV with Polars and read output via `pl.read_parquet`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest test/test_minute_bars.py test/test_session_tagger.py -q
```

Expected: FAIL because implementation returns pandas DataFrames.

- [ ] **Step 3: Convert `utils/minute_bars.py` to Polars**

Implement these signatures:

```python
def normalize_minute_bars(df: pl.DataFrame) -> pl.DataFrame: ...
def load_minute_bars(path: str | Path) -> pl.DataFrame: ...
def build_market_time_columns(df: pl.DataFrame) -> pl.DataFrame: ...
def derive_session_window(df: pl.DataFrame) -> pl.DataFrame: ...
```

Use `pl.read_parquet`, `pl.read_csv(try_parse_dates=True)`, `str.to_datetime`, `.dt.convert_time_zone("America/New_York")`, `.dt.replace_time_zone(None)`, `.dt.date()`, `.dt.time()`, and `.dt.hour() * 60 + .dt.minute()`. Keep duplicate timestamp rejection. For ambiguous legacy ET, reject if parsing/localization produces null.

- [ ] **Step 4: Convert `session_tagger.py` write**

Change `df.to_parquet(out_path, index=False)` to `df.write_parquet(out_path)` and keep print with `df.height`.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest test/test_minute_bars.py test/test_session_tagger.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/minute_bars.py session_tagger.py test/test_minute_bars.py test/test_session_tagger.py
git commit -m "refactor: migrate minute bars to polars"
```

---

### Task 3: Macro Outcomes

**Files:**
- Modify: `macro_outcomes.py`
- Modify: `test/test_macro_outcomes.py`

- [ ] **Step 1: Convert macro outcome test to Polars**

Use `pl.DataFrame` fixture. Assert `isinstance(out, pl.DataFrame)`, `out.height == 1`, `out.item(0, "macro_open") == 101.0`, `out.item(0, "postclose_range_points") == 1.5`.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_outcomes.py -q
```

Expected: FAIL because output is pandas.

- [ ] **Step 3: Convert `macro_outcomes.py`**

Use Polars throughout. Recommended simple implementation: after `_prepare_macro_bars`, iterate `for d, g in macro.partition_by("date", as_dict=True).items()` because macro windows are tiny. Use row/column expressions for aggregate values. Return `pl.DataFrame(out_rows).sort("date")` with `macro_type` as `pl.Categorical`. Write via `feats.write_parquet(out_path)`.

- [ ] **Step 4: Run test**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_outcomes.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add macro_outcomes.py test/test_macro_outcomes.py
git commit -m "refactor: migrate macro outcomes to polars"
```

---

### Task 4: PM/3PM Features + Interactions

**Files:**
- Modify: `features/pm_3pm.py`
- Modify: `features/pm_macro_interactions.py`
- Modify: `test/test_pm_3pm.py`

- [ ] **Step 1: Convert PM test to Polars**

Use Polars to write/read. Assert `_load_minutes` returns `pl.DataFrame` and first `timestamp` equals `datetime(2020, 9, 1, 13, 0)`.

- [ ] **Step 2: Run PM test to verify failure**

Run:

```bash
.venv/bin/python -m pytest test/test_pm_3pm.py -q
```

Expected: FAIL because `_load_minutes` returns pandas.

- [ ] **Step 3: Convert `features/pm_3pm.py`**

Use Polars in `_load_minutes`. For `_compute_day_features`, keep small per-day imperative logic but consume Polars: filter masks with expressions, use `.is_empty()`, `.item()`, `.max()`, `.min()`, `.sum()`. Use `partition_by("date", as_dict=True)` in `main`. Write parquet with `write_parquet`.

- [ ] **Step 4: Convert `features/pm_macro_interactions.py`**

Use `pl.read_parquet`, `.join`, `pl.min_horizontal`, `pl.max_horizontal`, `pl.when(...).then(...).otherwise(...)`, `.sign()`, `.cast(pl.Int8)`, `write_parquet`. For summary histograms, use Polars filter/group_by/count/sort and print rows.

- [ ] **Step 5: Run PM test**

Run:

```bash
.venv/bin/python -m pytest test/test_pm_3pm.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/pm_3pm.py features/pm_macro_interactions.py test/test_pm_3pm.py
git commit -m "refactor: migrate pm features to polars"
```

---

### Task 5: Macro FVG Study

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

- [ ] **Step 1: Convert FVG tests to Polars**

Replace pandas fixtures with `pl.DataFrame`; use `str.to_datetime` in helpers; replace `.iloc[0]` with `.row(0, named=True)`; replace pandas `Timestamp` comparisons with Python `datetime` where needed. Keep all current behavior assertions.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py -q
```

Expected: FAIL because implementation expects pandas.

- [ ] **Step 3: Convert public API to Polars**

Convert all function annotations and returned frames to Polars. For stateful scans, use `.iter_rows(named=True)` over sorted Polars frames instead of pandas `.iterrows()`. Replace `pd.NaT/pd.NA` with `None`. Replace `pd.qcut` bucket logic with Polars rank/quantile or deterministic manual bucket assignment over sorted values. Keep summary output columns identical.

- [ ] **Step 4: Convert plotting internals**

Where matplotlib requires arrays, pass `series.to_numpy()` or `frame.to_pandas()` only after filtering to small summary/event tables. No pandas import remains in module.

- [ ] **Step 5: Run FVG tests**

Run:

```bash
.venv/bin/python -m pytest test/test_macro_fvg_study.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "refactor: migrate macro fvg study to polars"
```

---

### Task 6: News Helpers + Visualizations

**Files:**
- Modify: `utils/helper.py`
- Modify: `viz/macro_analysis.py`
- Modify: `viz/macro_high.py`
- Modify: `viz/pm_macro_viz.py`
- Modify: `viz/viz_outcome.py`

- [ ] **Step 1: Convert news helper smoke tests if present**

Search for helper tests. If none exist, add `test/test_helper_polars.py` covering `_prep_news`, `merge_news_daily`, and `build_macro_event_links` with small Polars frames.

- [ ] **Step 2: Run helper tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest test/test_helper_polars.py -q
```

Expected: FAIL before helper migration.

- [ ] **Step 3: Convert `utils/helper.py` to Polars**

Use Polars string/date/list aggregations. Keep public function names. Return `pl.DataFrame`. Use Python loops only for `build_macro_event_links` if simpler.

- [ ] **Step 4: Convert visualization scripts**

Replace pandas reads/transforms with Polars. At plot call boundaries, use `to_numpy()` or `to_pandas()` for small result frames. Update parquet/csv IO to `pl.read_parquet` / `pl.read_csv`.

- [ ] **Step 5: Run helper tests**

Run:

```bash
.venv/bin/python -m pytest test/test_helper_polars.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/helper.py viz/macro_analysis.py viz/macro_high.py viz/pm_macro_viz.py viz/viz_outcome.py test/test_helper_polars.py
git commit -m "refactor: migrate helpers and viz to polars"
```

---

### Task 7: Remaining Active Pandas References + Full Verification

**Files:**
- Modify any active `.py` file still importing pandas except optional tests for external compatibility.

- [ ] **Step 1: Search active code for pandas**

Run:

```bash
rg -n "import pandas|pd\." --glob '*.py' --glob '!**/.venv/**' --glob '!docs/**'
```

Expected: no active pandas imports except intentionally excluded comments/docstrings. If matches remain in active code, migrate them.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest test features/lrlr/test features/trend -q
```

Expected: PASS.

- [ ] **Step 3: Run script smoke checks without loading tick data**

Run:

```bash
.venv/bin/python - <<'PY'
from utils.tick_data import get_tick_schema
from pathlib import Path
p = Path('input-data/merged_nq_ticks.parquet')
if p.exists():
    schema = get_tick_schema(p)
    print(schema.names)
else:
    print('tick parquet absent in worktree; skipped')
PY
```

Expected: prints schema names or skip. Must not use `pl.read_parquet` on tick parquet.

- [ ] **Step 4: Commit final cleanup**

```bash
git add -A
git commit -m "refactor: finish polars migration"
```

If no changes remain, skip commit.
