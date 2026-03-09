# Macro FVG Successful Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing macro FVG study so each event tracks first-retrace-candle context, success after that first retrace candle, and stacked continuation metadata, then summarize MAE on successful FVGs by alignment and stacking context.

**Architecture:** Modify `features/macro_fvg_study.py` in place so the existing event parquet remains the source of truth. Reuse the current `15:59` scan to capture first-retrace-candle metadata and success timing, add stacked continuation flags directly on event rows, and append successful-context summary scopes to the current summary parquet without replacing existing outcome or excursion outputs.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, pyarrow/parquet

---

### Task 1: Add first-retrace-candle fields and stacked continuation metadata

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`
- Reference: `docs/plans/2026-03-09-macro-fvg-success-context-design.md`

**Step 1: Write the failing tests for first retrace candle capture and stacked continuation flagging**

Add focused tests like:

```python
def test_stores_first_retrace_candle_metadata_for_bullish_fvg():
    bars = make_bars([
        {
            "DateTime_ET": "2025-01-02 15:50:00",
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:51:00",
            "Open": 100.5, "High": 101.5, "Low": 100.0, "Close": 101.0,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:52:00",
            "Open": 102.0, "High": 103.0, "Low": 102.0, "Close": 102.5,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:54:00",
            "Open": 102.5, "High": 102.8, "Low": 101.2, "Close": 101.8,
            "window": "MACRO",
        },
    ])

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["first_retrace_candle_at"] == pd.Timestamp("2025-01-02 15:54:00")
    assert event["first_retrace_candle_open"] == 102.5
    assert event["first_retrace_candle_high"] == 102.8
    assert event["first_retrace_candle_low"] == 101.2
    assert event["first_retrace_candle_close"] == 101.8
```

Add a second focused test for stacked continuation:

```python
def test_marks_later_same_side_4_bar_fvg_as_stacked_continuation():
    bars = make_bars([
        {
            "DateTime_ET": "2025-01-02 15:49:00",
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:50:00",
            "Open": 100.5, "High": 101.0, "Low": 100.0, "Close": 100.8,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:51:00",
            "Open": 102.0, "High": 103.0, "Low": 102.0, "Close": 102.5,
            "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:52:00",
            "Open": 103.0, "High": 104.0, "Low": 103.0, "Close": 103.5,
            "window": "MACRO",
        },
    ])

    events = detect_macro_fvgs(bars)

    first_event = events[events["assigned_at"] == pd.Timestamp("2025-01-02 15:50:00")].iloc[0]
    second_event = events[events["assigned_at"] == pd.Timestamp("2025-01-02 15:51:00")].iloc[0]
    assert not first_event["stacked_continuation_fvg"]
    assert pd.isna(first_event["stack_predecessor_assigned_at"])
    assert second_event["stacked_continuation_fvg"]
    assert second_event["stack_predecessor_assigned_at"] == pd.Timestamp("2025-01-02 15:50:00")
```

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_stores_first_retrace_candle_metadata_for_bullish_fvg \
  test/test_macro_fvg_study.py::test_marks_later_same_side_4_bar_fvg_as_stacked_continuation \
  -q
```

Expected:
- fail because the first-retrace-candle fields and stacked continuation fields do not exist yet

**Step 3: Implement the minimal event enrichment**

In `features/macro_fvg_study.py`:

1. Add empty/default event columns returned by `detect_macro_fvgs()`:

```python
"stacked_continuation_fvg",
"stack_predecessor_assigned_at",
```

2. Add a helper like:

```python
def mark_stacked_continuation_fvgs(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["stacked_continuation_fvg"] = False
    work["stack_predecessor_assigned_at"] = pd.NaT
    for idx in range(1, len(work)):
        current = work.iloc[idx]
        previous = work.iloc[idx - 1]
        if (
            current["date"] == previous["date"]
            and current["fvg_side"] == previous["fvg_side"]
            and current["assigned_at"] == previous["assigned_at"] + pd.Timedelta(minutes=1)
        ):
            work.at[work.index[idx], "stacked_continuation_fvg"] = True
            work.at[work.index[idx], "stack_predecessor_assigned_at"] = previous["assigned_at"]
    return work
```

3. Call that helper at the end of `detect_macro_fvgs()`.

4. In `scan_fvg_outcomes_until_1559_close()`, when first retrace is found, also store that bar’s OHLC/time:

```python
first_retrace_candle = None
...
if pd.isna(first_retrace_at) and _bar_retraces_gap(...):
    first_retrace_at = bar["DateTime_ET"]
    first_retrace_candle = bar
```

Then persist:

```python
"first_retrace_candle_at"
"first_retrace_candle_open"
"first_retrace_candle_high"
"first_retrace_candle_low"
"first_retrace_candle_close"
```

For non-retraced or non-confirmable events, keep these null.

**Step 4: Run the focused tests again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_stores_first_retrace_candle_metadata_for_bullish_fvg \
  test/test_macro_fvg_study.py::test_marks_later_same_side_4_bar_fvg_as_stacked_continuation \
  -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg retrace context"
```

### Task 2: Add success-after-first-retrace fields

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing tests for successful FVG classification**

Add focused tests like:

```python
def test_marks_bullish_fvg_successful_when_later_breaks_first_retrace_high():
    ...
    event = scanned.iloc[0]
    assert event["first_retrace_candle_high"] == 102.8
    assert event["success_reference_price"] == 102.8
    assert event["successful_by_1559"]
    assert event["success_break_at"] == pd.Timestamp("2025-01-02 15:56:00")
```

Add additional tests for:
- bearish success when a later bar breaks below the first retrace candle low
- retraced but unsuccessful event leaves `successful_by_1559=False` and `success_break_at` null
- no-retrace event keeps success context fields null / false

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_marks_bullish_fvg_successful_when_later_breaks_first_retrace_high \
  test/test_macro_fvg_study.py::test_marks_bearish_fvg_successful_when_later_breaks_first_retrace_low \
  test/test_macro_fvg_study.py::test_retraced_fvg_can_remain_unsuccessful_by_1559 \
  test/test_macro_fvg_study.py::test_no_retrace_keeps_success_context_null \
  -q
```

Expected:
- fail because the success fields and scan logic do not exist yet

**Step 3: Implement the minimal success scan**

Add helpers in `features/macro_fvg_study.py`:

```python
def _success_reference_price(event: pd.Series | dict) -> float:
    if event["fvg_side"] == "bullish":
        return float(event["first_retrace_candle_high"])
    return float(event["first_retrace_candle_low"])


def _bar_breaks_retrace_reference(bar: pd.Series, fvg_side: str, reference_price: float) -> bool:
    if fvg_side == "bullish":
        return float(bar["High"]) > reference_price
    return float(bar["Low"]) < reference_price
```

In `scan_fvg_outcomes_until_1559_close()`:
- once the first retrace candle is known, scan only later bars (`DateTime_ET > first_retrace_candle_at`)
- set:

```python
"success_reference_price"
"successful_by_1559"
"success_break_at"
```

Recommended logic:

```python
if first_retrace_candle is None:
    success_reference_price = np.nan
    successful_by_1559 = False
    success_break_at = pd.NaT
else:
    success_reference_price = _success_reference_price(event_dict)
    success_break_at = first later bar that breaks the retrace reference price
    successful_by_1559 = pd.notna(success_break_at)
```

For non-confirmable events, keep success fields null / false.

**Step 4: Run the focused tests again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_marks_bullish_fvg_successful_when_later_breaks_first_retrace_high \
  test/test_macro_fvg_study.py::test_marks_bearish_fvg_successful_when_later_breaks_first_retrace_low \
  test/test_macro_fvg_study.py::test_retraced_fvg_can_remain_unsuccessful_by_1559 \
  test/test_macro_fvg_study.py::test_no_retrace_keeps_success_context_null \
  -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg success context"
```

### Task 3: Add successful-context summary builders and smoke assertions

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write the failing tests for success-context summaries**

Add tests like:

```python
def test_builds_success_context_alignment_bucket_summary():
    events = pd.DataFrame([
        {
            "alignment_bucket": "3_aligned",
            "stacked_continuation_fvg": False,
            "is_confirmable_by_1559": True,
            "retraced_by_1559": True,
            "successful_by_1559": True,
            "mae_pct_to_1559": 0.004,
        },
        {
            "alignment_bucket": "3_aligned",
            "stacked_continuation_fvg": False,
            "is_confirmable_by_1559": True,
            "retraced_by_1559": True,
            "successful_by_1559": False,
            "mae_pct_to_1559": np.nan,
        },
    ])

    summary = build_success_context_alignment_bucket_summary(events)

    row = summary.iloc[0]
    assert row["alignment_bucket"] == "3_aligned"
    assert row["n_confirmable"] == 2
    assert row["n_retraced"] == 2
    assert row["n_successful"] == 1
    assert row["retrace_rate"] == 1.0
    assert row["success_after_retrace_rate"] == 0.5
    assert row["successful_share_of_confirmable"] == 0.5
    assert row["mae_pct_mean"] == 0.004
    assert row["mae_pct_median"] == 0.004
    assert row["mae_pct_p75"] == 0.004
```

Add additional tests for:
- `build_success_context_stacked_flag_summary()`
- `build_success_context_alignment_bucket_stacked_flag_summary()`

**Step 2: Extend the smoke test to assert the new fields/scopes exist**

Add assertions in the existing end-to-end smoke test so it checks:

```python
events = pd.read_parquet(events_path)
summary = pd.read_parquet(summary_path)

assert "successful_by_1559" in events.columns
assert "stacked_continuation_fvg" in events.columns
assert "first_retrace_candle_at" in events.columns
assert "success_context_alignment_bucket" in set(summary["summary_scope"])
assert "success_context_stacked_flag" in set(summary["summary_scope"])
```

**Step 3: Run the focused summary and smoke tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_stacked_flag_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_stacked_flag_summary \
  test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures \
  -q
```

Expected:
- fail because the success-context summary builders and summary fields do not exist yet

**Step 4: Implement success-context summary helpers**

Extend the summary schema in `features/macro_fvg_study.py` with nullable columns:

```python
"n_retraced",
"n_successful",
"success_after_retrace_rate",
"successful_share_of_confirmable",
```

Add a helper like:

```python
def _group_success_context_stats(
    events: pd.DataFrame,
    group_cols: list[str],
    scope_name: str,
) -> pd.DataFrame:
    ...
```

Recommended logic:
- `n_confirmable = sum(is_confirmable_by_1559)`
- `n_retraced = sum(retraced_by_1559)`
- `n_successful = sum(successful_by_1559)`
- `retrace_rate = n_retraced / n_confirmable`
- `success_after_retrace_rate = n_successful / n_retraced`
- `successful_share_of_confirmable = n_successful / n_confirmable`
- MAE stats from successful rows only, using existing `mae_pct_to_1559`

Add builders:

```python
def build_success_context_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_success_context_stats(events, [], "success_context_overall")


def build_success_context_alignment_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_success_context_stats(
        events,
        ["alignment_bucket"],
        "success_context_alignment_bucket",
    )


def build_success_context_stacked_flag_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_success_context_stats(
        events,
        ["stacked_continuation_fvg"],
        "success_context_stacked_flag",
    )


def build_success_context_alignment_bucket_stacked_flag_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    return _group_success_context_stats(
        events,
        ["alignment_bucket", "stacked_continuation_fvg"],
        "success_context_alignment_bucket_stacked_flag",
    )
```

Append these frames into the existing combined summary output.

**Step 5: Run the focused summary and smoke tests again**

Run:

```bash
python3 -m pytest \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_stacked_flag_summary \
  test/test_macro_fvg_study.py::test_builds_success_context_alignment_bucket_stacked_flag_summary \
  test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures \
  -q
```

Expected:
- pass

**Step 6: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg success summaries"
```

### Task 4: Run focused verification and inspect real outputs

**Files:**
- Modify: `features/macro_fvg_study.py` if cleanup is needed after the real-data run

**Step 1: Run the full focused test file**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- all tests pass

**Step 2: Run the study on the real tagged data**

Run:

```bash
python3 features/macro_fvg_study.py
```

Expected:
- updates `outputs/nq_macro_fvg_events.parquet`
- updates `outputs/nq_macro_fvg_summary.parquet`

**Step 3: Inspect the new fields and summary scopes**

Run:

```bash
python3 - <<'PY'
import pandas as pd

events = pd.read_parquet("outputs/nq_macro_fvg_events.parquet")
summary = pd.read_parquet("outputs/nq_macro_fvg_summary.parquet")

success_scopes = summary[summary["summary_scope"].str.contains("success_context", na=False)]

print(events[
    [
        "first_retrace_candle_at",
        "success_reference_price",
        "successful_by_1559",
        "success_break_at",
        "stacked_continuation_fvg",
        "stack_predecessor_assigned_at",
        "mae_pct_to_1559",
    ]
].head())
print(success_scopes.head())
print("retraced_rows", int(events["retraced_by_1559"].sum()))
print("successful_rows", int(events["successful_by_1559"].sum()))
print("stacked_rows", int(events["stacked_continuation_fvg"].fillna(False).sum()))
print(
    success_scopes[
        success_scopes["summary_scope"] == "success_context_alignment_bucket"
    ][
        [
            "alignment_bucket",
            "n_retraced",
            "n_successful",
            "mae_pct_mean",
            "mae_pct_median",
            "mae_pct_p75",
        ]
    ]
)
PY
```

Expected:
- event parquet contains the new retrace/success/stacking fields
- summary parquet contains the new success-context scopes
- stacked continuation rows are visible if the pattern exists in real data
- MAE summary rows are populated for successful FVGs

**Step 4: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: extend macro fvg study with success context"
```
