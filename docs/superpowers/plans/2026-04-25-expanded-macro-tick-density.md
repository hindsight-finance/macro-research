# Expanded Macro Tick Density Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `outputs/nq_macro_tick_density.parquet` generation from 15:50–15:59 ET to 15:40–16:10 ET while keeping the same schema and UTC timestamps.

**Architecture:** Keep the lazy Polars scan and existing aggregation schema. Replace the ET filter with a cross-hour 15:40–16:10 inclusive filter, redefine `macro_minute_index` as the actual ET minute number, then verify with focused pytest coverage before regenerating the parquet.

**Tech Stack:** Python 3, Polars, pytest

---

### Task 1: Add failing tests for expanded ET minute coverage

**Files:**
- Modify: `test/test_tick_density.py`
- Test: `test/test_tick_density.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_macro_tick_density_covers_1540_to_1610_et_with_new_index_offsets(tmp_path: Path):
    ...
    assert out.select("macro_minute_index").to_series().to_list() == [0, 10, 19, 20, 30]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_tick_density.py::test_build_macro_tick_density_covers_1540_to_1610_et_with_new_index_offsets -q`
Expected: FAIL because current code only keeps 15:50–15:59 ET.

- [ ] **Step 3: Extend DST coverage test**

```python
assert out.select("datetime_utc").to_series().dt.hour().to_list() == [20, 19]
assert out.select("macro_minute_index").to_series().to_list() == [0, 0]
```

using 15:40 ET winter/summer rows.

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest test/test_tick_density.py::test_build_macro_tick_density_uses_utc_only_and_handles_dst_macro_hour -q`
Expected: FAIL until the window anchor changes to 15:40 ET.

### Task 2: Implement broader 1-minute window

**Files:**
- Modify: `tick_density.py`
- Test: `test/test_tick_density.py`

- [ ] **Step 1: Write minimal implementation**

```python
minute_of_day_et = hour_et * 60 + minute_et
start_minute = 15 * 60 + 40
end_minute = 16 * 60 + 10
...
.filter((minute_of_day_et >= start_minute) & (minute_of_day_et <= end_minute))
.with_columns(macro_minute_index=(minute_of_day_et - start_minute).cast(pl.UInt8))
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m pytest test/test_tick_density.py -q`
Expected: PASS

### Task 3: Regenerate parquet

**Files:**
- Modify: `outputs/nq_macro_tick_density.parquet`

- [ ] **Step 1: Execute tick-density builder**

Run: `python3 tick_density.py`
Expected: rewrites expanded 1-minute parquet and existing 5-second outputs.

- [ ] **Step 2: Verify output shape / range**

Run: `python3 - <<'PY' ... PY`
Expected: 31 distinct `macro_minute_index` values spanning `40..59` plus `0..10` in `outputs/nq_macro_tick_density.parquet`.
