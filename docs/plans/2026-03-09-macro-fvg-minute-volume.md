# Macro FVG Minute and Bar-2 Volume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing macro FVG study so each event stores creation-minute and raw bar-2 volume data, and add grouped summaries and figures for creation minute and bar-2 volume behavior.

**Architecture:** Modify `features/macro_fvg_study.py` in place so the existing event parquet remains the source of truth. Enrich event rows during detection with minute and volume metadata, derive new grouped summary tables from those enriched events, and add four new Matplotlib figures without replacing the current outputs.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, pyarrow/parquet

---

### Task 1: Add event-level minute and bar-2 volume fields

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`
- Reference: `docs/plans/2026-03-09-macro-fvg-minute-volume-design.md`

**Step 1: Write the failing tests for minute and volume enrichment**

Add tests like:

```python
def test_detect_macro_fvg_stores_assigned_minute_and_bar2_volume():
    bars = make_bars([
        {
            "DateTime_ET": "2025-01-02 15:49:00",
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "Volume": 10, "window": "H3PM",
        },
        {
            "DateTime_ET": "2025-01-02 15:50:00",
            "Open": 99.0, "High": 100.0, "Low": 97.0, "Close": 98.0,
            "Volume": 25, "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:51:00",
            "Open": 96.0, "High": 97.0, "Low": 94.0, "Close": 95.0,
            "Volume": 40, "window": "MACRO",
        },
    ])

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["assigned_minute_hhmm"] == "15:50"
    assert event["assigned_minute_index"] == 0
    assert event["bar2_volume"] == 25
```

Add one more test for a later assigned minute, for example `15:57 -> assigned_minute_index == 7`.

**Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py::test_detect_macro_fvg_stores_assigned_minute_and_bar2_volume -q
```

Expected:
- fail with missing columns in the event output

**Step 3: Implement the minimal event enrichment**

Modify `features/macro_fvg_study.py` so `detect_macro_fvgs()` also stores:

```python
event_rows["assigned_minute_hhmm"] = event_rows["assigned_at"].dt.strftime("%H:%M")
event_rows["assigned_minute_index"] = (
    event_rows["assigned_at"].dt.hour * 60
    + event_rows["assigned_at"].dt.minute
    - (15 * 60 + 50)
)
event_rows["bar2_volume"] = event_rows["Volume"]
```

Also update the required schema to include `Volume`:

```python
required = {"DateTime_ET", "Open", "High", "Low", "Close", "Volume", "window"}
```

Make sure the returned event DataFrame includes the new columns.

**Step 4: Run the focused test again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py::test_detect_macro_fvg_stores_assigned_minute_and_bar2_volume -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg minute and volume fields"
```

### Task 2: Add grouped summaries by creation minute and bar-2 volume bucket

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing tests for grouped summary tables**

Add tests like:

```python
def test_builds_creation_minute_summary():
    events = pd.DataFrame([
        {
            "assigned_minute_hhmm": "15:50",
            "bar2_volume": 100,
            "is_confirmable_by_1559": True,
            "held_to_1559_close": True,
            "invalidated_by_1559": False,
            "retraced_by_1559": True,
            "untouched_to_1559_close": False,
        },
        {
            "assigned_minute_hhmm": "15:50",
            "bar2_volume": 200,
            "is_confirmable_by_1559": True,
            "held_to_1559_close": False,
            "invalidated_by_1559": True,
            "retraced_by_1559": True,
            "untouched_to_1559_close": False,
        },
    ])

    minute_summary = build_creation_minute_summary(events)

    row = minute_summary.iloc[0]
    assert row["assigned_minute_hhmm"] == "15:50"
    assert row["n_total"] == 2
    assert row["hold_rate"] == 0.5
    assert row["invalidation_rate"] == 0.5
```

Add a second test for volume buckets:

```python
def test_builds_bar2_volume_bucket_summary():
    ...
    summary = build_bar2_volume_summary(events, bucket_count=2)
    assert "bar2_volume_bucket" in summary.columns
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py::test_builds_creation_minute_summary test/test_macro_fvg_study.py::test_builds_bar2_volume_bucket_summary -q
```

Expected:
- fail because the grouped summary builders do not exist yet

**Step 3: Implement grouped summary builders**

Add summary helpers in `features/macro_fvg_study.py`:

```python
def _group_outcome_rates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        n_total = len(group)
        n_confirmable = int(group["is_confirmable_by_1559"].sum())
        denom = n_confirmable if n_confirmable else np.nan
        rows.append({
            ...
        })
    return pd.DataFrame(rows)


def build_creation_minute_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(events, ["assigned_minute_index", "assigned_minute_hhmm"])


def build_bar2_volume_summary(events: pd.DataFrame, bucket_count: int = 4) -> pd.DataFrame:
    work = events.copy()
    work["bar2_volume_bucket"] = pd.qcut(
        work["bar2_volume"],
        q=min(bucket_count, work["bar2_volume"].nunique()),
        duplicates="drop",
    ).astype(str)
    return _group_outcome_rates(work, ["bar2_volume_bucket"])
```

Also decide how these tables are stored in the summary output. Recommended shape:
- keep the existing stage summary rows
- append new rows with a `summary_scope` value such as:
  - `creation_minute`
  - `bar2_volume_bucket`

If needed, add nullable columns:
- `assigned_minute_hhmm`
- `assigned_minute_index`
- `bar2_volume_bucket`

**Step 4: Run the focused summary tests again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py::test_builds_creation_minute_summary test/test_macro_fvg_study.py::test_builds_bar2_volume_bucket_summary -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg minute and volume summaries"
```

### Task 3: Add minute and volume visualizations

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing smoke assertions for the four new figures**

Extend the end-to-end smoke test so it also checks for:

```python
assert (figures_dir / "creation_minute_outcome_bars.png").exists()
assert (figures_dir / "bar2_volume_bucket_outcomes.png").exists()
assert (figures_dir / "creation_minute_avg_bar2_volume.png").exists()
assert (figures_dir / "creation_minute_volume_heatmap.png").exists()
```

**Step 2: Run the smoke test to verify it fails**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected:
- fail because the new figure files are not created yet

**Step 3: Implement the new plotting branches**

Add or extend plotting in `features/macro_fvg_study.py`:

```python
def plot_creation_minute_outcomes(events: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_bar2_volume_bucket_outcomes(events: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_creation_minute_avg_bar2_volume(events: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_creation_minute_volume_heatmap(events: pd.DataFrame, figures_dir: Path) -> None:
    ...
```

Recommended content:
- `creation_minute_outcome_bars.png`
  - x-axis: `assigned_minute_hhmm`
  - y-axis: rates or counts
- `bar2_volume_bucket_outcomes.png`
  - x-axis: `bar2_volume_bucket`
  - stacked or grouped bars for hold / invalidate / retrace
- `creation_minute_avg_bar2_volume.png`
  - x-axis: creation minute
  - y-axis: mean raw `bar2_volume`
- `creation_minute_volume_heatmap.png`
  - y-axis: `assigned_minute_hhmm`
  - x-axis: `bar2_volume_bucket`
  - value: count or hold rate

Update `plot_fvg_summary_figures()` to call these four new plotters.

**Step 4: Run the smoke test again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg minute and volume figures"
```

### Task 4: Run focused verification and inspect real outputs

**Files:**
- Modify: `features/macro_fvg_study.py` if any cleanup is needed after real-data run

**Step 1: Run the full focused test file**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- all tests pass

**Step 2: Run the study on the real tagged data**

Run:

```bash
python features/macro_fvg_study.py
```

Expected:
- updates `outputs/nq_macro_fvg_events.parquet`
- updates `outputs/nq_macro_fvg_summary.parquet`
- writes the four new figures under `outputs/figs/fvg/`

**Step 3: Inspect the new columns and outputs**

Run:

```bash
python - <<'PY'
import pandas as pd
events = pd.read_parquet("outputs/nq_macro_fvg_events.parquet")
summary = pd.read_parquet("outputs/nq_macro_fvg_summary.parquet")
print(events[["assigned_minute_hhmm", "assigned_minute_index", "bar2_volume"]].head())
print(summary.columns.tolist())
PY
```

Expected:
- event parquet contains minute and volume fields
- summary parquet contains minute/volume summary grouping fields

**Step 4: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: extend macro fvg analysis by minute and bar2 volume"
```

### Task 5: Optional usage note

**Files:**
- Modify: `README.md`

**Step 1: Add a one-line usage note if desired**

Example:

```markdown
- `python features/macro_fvg_study.py`: regenerate macro FVG event, summary, minute, volume, and figure outputs.
```

**Step 2: Verify docs change only if you actually make it**

Run:

```bash
git diff -- README.md
```

Expected:
- short usage note only

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add macro fvg minute volume usage note"
```
