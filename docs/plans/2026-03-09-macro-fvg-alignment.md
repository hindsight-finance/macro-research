# Macro FVG Alignment Bucket Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing macro FVG study so each event stores 3-bar candle-alignment labels, append grouped outcome summaries for alignment buckets and their key controls, and add alignment-focused figures under `outputs/figs/fvg/`.

**Architecture:** Modify `features/macro_fvg_study.py` in place so the existing event parquet remains the source of truth. Enrich event rows with bar-direction and alignment metadata during detection, derive new grouped summary tables from those enriched events using the already-established minute and gap controls, and add four alignment-focused Matplotlib figures without removing the current outputs.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, pyarrow/parquet

---

### Task 1: Add event-level candle-direction and alignment fields

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`
- Reference: `docs/plans/2026-03-09-macro-fvg-alignment-design.md`

**Step 1: Write the failing tests for alignment labeling**

Add tests like:

```python
def test_detect_macro_fvg_stores_alignment_bucket_for_two_aligned_one_opposite():
    bars = make_bars([
        {
            "DateTime_ET": "2025-01-02 15:49:00",
            "Open": 101.0, "High": 101.5, "Low": 98.5, "Close": 99.0,
            "Volume": 100, "window": "H3PM",
        },
        {
            "DateTime_ET": "2025-01-02 15:50:00",
            "Open": 99.0, "High": 100.0, "Low": 97.0, "Close": 98.0,
            "Volume": 110, "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:51:00",
            "Open": 95.0, "High": 96.5, "Low": 94.0, "Close": 96.0,
            "Volume": 120, "window": "MACRO",
        },
    ])

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert event["bar1_direction"] == "bearish"
    assert event["bar2_direction"] == "bearish"
    assert event["bar3_direction"] == "bullish"
    assert event["aligned_count"] == 2
    assert event["opposite_count"] == 1
    assert event["neutral_count"] == 0
    assert event["alignment_bucket"] == "2_aligned_1_opposite"


def test_detect_macro_fvg_marks_contains_neutral_when_pattern_has_doji():
    bars = make_bars([
        {
            "DateTime_ET": "2025-01-02 15:49:00",
            "Open": 101.0, "High": 101.5, "Low": 98.5, "Close": 99.0,
            "Volume": 100, "window": "H3PM",
        },
        {
            "DateTime_ET": "2025-01-02 15:50:00",
            "Open": 99.0, "High": 100.0, "Low": 97.0, "Close": 99.0,
            "Volume": 110, "window": "MACRO",
        },
        {
            "DateTime_ET": "2025-01-02 15:51:00",
            "Open": 95.0, "High": 96.0, "Low": 94.0, "Close": 95.0,
            "Volume": 120, "window": "MACRO",
        },
    ])

    events = detect_macro_fvgs(bars)

    event = events.iloc[0]
    assert event["bar2_direction"] == "neutral"
    assert event["neutral_count"] == 1
    assert event["alignment_bucket"] == "contains_neutral"
```

Add one more focused test that produces `1_aligned_2_opposite`.

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_detect_macro_fvg_stores_alignment_bucket_for_two_aligned_one_opposite test/test_macro_fvg_study.py::test_detect_macro_fvg_marks_contains_neutral_when_pattern_has_doji -q
```

Expected:
- fail with missing alignment columns or missing helper logic

**Step 3: Implement the minimal alignment enrichment**

Add helpers in `features/macro_fvg_study.py`:

```python
def classify_candle_direction(open_price: float, close_price: float) -> str:
    if close_price > open_price:
        return "bullish"
    if close_price < open_price:
        return "bearish"
    return "neutral"


def assign_alignment_bucket(
    fvg_side: str,
    directions: list[str],
) -> tuple[int, int, int, str]:
    aligned_label = "bullish" if fvg_side == "bullish" else "bearish"
    aligned_count = sum(direction == aligned_label for direction in directions)
    neutral_count = sum(direction == "neutral" for direction in directions)
    opposite_count = len(directions) - aligned_count - neutral_count

    if neutral_count > 0:
        return aligned_count, opposite_count, neutral_count, "contains_neutral"
    if aligned_count == 3:
        return aligned_count, opposite_count, neutral_count, "3_aligned"
    if aligned_count == 2:
        return aligned_count, opposite_count, neutral_count, "2_aligned_1_opposite"
    if aligned_count == 1:
        return aligned_count, opposite_count, neutral_count, "1_aligned_2_opposite"
    raise ValueError("Unexpected zero-aligned non-neutral FVG pattern")
```

Update `detect_macro_fvgs()` so it stores:

```python
event_rows["bar1_open"] = work["Open"].shift(1)
event_rows["bar1_close"] = work["Close"].shift(1)
event_rows["bar3_open"] = work["Open"].shift(-1)
event_rows["bar3_close"] = work["Close"].shift(-1)

event_rows["bar1_direction"] = [
    classify_candle_direction(o, c)
    for o, c in zip(event_rows["bar1_open"], event_rows["bar1_close"])
]
event_rows["bar2_direction"] = [
    classify_candle_direction(o, c)
    for o, c in zip(event_rows["Open"], event_rows["Close"])
]
event_rows["bar3_direction"] = [
    classify_candle_direction(o, c)
    for o, c in zip(event_rows["bar3_open"], event_rows["bar3_close"])
]

alignment = [
    assign_alignment_bucket(side, [d1, d2, d3])
    for side, d1, d2, d3 in zip(
        event_rows["fvg_side"],
        event_rows["bar1_direction"],
        event_rows["bar2_direction"],
        event_rows["bar3_direction"],
    )
]
event_rows[["aligned_count", "opposite_count", "neutral_count", "alignment_bucket"]] = alignment
```

Make sure the returned event DataFrame includes the new columns.

**Step 4: Run the focused alignment tests again**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_detect_macro_fvg_stores_alignment_bucket_for_two_aligned_one_opposite test/test_macro_fvg_study.py::test_detect_macro_fvg_marks_contains_neutral_when_pattern_has_doji -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg alignment labels"
```

### Task 2: Add alignment control fields and grouped summary builders

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing tests for alignment summaries**

Add tests like:

```python
def test_builds_alignment_bucket_summary():
    events = pd.DataFrame([
        {
            "alignment_bucket": "3_aligned",
            "minute_block": "15:50-15:52",
            "gap_size_bucket_225": ">=2.25",
            "is_confirmable_by_1559": True,
            "held_to_1559_close": True,
            "invalidated_by_1559": False,
            "retraced_by_1559": True,
            "untouched_to_1559_close": False,
        },
        {
            "alignment_bucket": "3_aligned",
            "minute_block": "15:50-15:52",
            "gap_size_bucket_225": ">=2.25",
            "is_confirmable_by_1559": True,
            "held_to_1559_close": False,
            "invalidated_by_1559": True,
            "retraced_by_1559": True,
            "untouched_to_1559_close": False,
        },
    ])

    summary = build_alignment_bucket_summary(events)

    row = summary.iloc[0]
    assert row["alignment_bucket"] == "3_aligned"
    assert row["n_total"] == 2
    assert row["hold_rate"] == 0.5
    assert row["invalidation_rate"] == 0.5
```

Add a second test for a controlled split:

```python
def test_builds_alignment_bucket_gap_bucket_summary():
    ...
    summary = build_alignment_bucket_gap_bucket_summary(events)
    assert set(summary["gap_size_bucket_225"]) == {"<2.25", ">=2.25"}
```

**Step 2: Run the focused summary tests to verify they fail**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_builds_alignment_bucket_summary test/test_macro_fvg_study.py::test_builds_alignment_bucket_gap_bucket_summary -q
```

Expected:
- fail because the alignment summary builders do not exist yet

**Step 3: Implement alignment controls and summary builders**

Add event-level controls:

```python
event_rows["minute_block"] = np.where(
    event_rows["assigned_minute_index"] <= 2,
    "15:50-15:52",
    np.where(
        event_rows["assigned_minute_index"] <= 7,
        "15:53-15:57",
        "15:58_unconfirmable",
    ),
)
event_rows["gap_size_bucket_225"] = np.where(
    event_rows["gap_size"] < 2.25,
    "<2.25",
    ">=2.25",
)
```

Extend the summary schema with nullable columns:
- `alignment_bucket`
- `minute_block`
- `gap_size_bucket_225`

Add builders like:

```python
def build_alignment_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(events, ["alignment_bucket"], "alignment_bucket")


def build_alignment_bucket_minute_block_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(
        events,
        ["minute_block", "alignment_bucket"],
        "alignment_bucket_minute_block",
    )


def build_alignment_bucket_gap_bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    return _group_outcome_rates(
        events,
        ["gap_size_bucket_225", "alignment_bucket"],
        "alignment_bucket_gap_bucket",
    )
```

Append these frames into the existing combined summary output.

**Step 4: Run the focused summary tests again**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_builds_alignment_bucket_summary test/test_macro_fvg_study.py::test_builds_alignment_bucket_gap_bucket_summary -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg alignment summaries"
```

### Task 3: Add alignment figures and extend the smoke test

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing smoke assertions for the new figures**

Extend the end-to-end smoke test so it also checks for:

```python
assert (figures_dir / "alignment_bucket_outcomes.png").exists()
assert (figures_dir / "alignment_bucket_by_minute_block.png").exists()
assert (figures_dir / "alignment_bucket_by_gap_bucket.png").exists()
assert (figures_dir / "alignment_bucket_counts.png").exists()
```

**Step 2: Run the smoke test to verify it fails**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected:
- fail because the new alignment figure files are not created yet

**Step 3: Implement the new plotting branches**

Add or extend plotting in `features/macro_fvg_study.py`:

```python
def plot_alignment_bucket_outcomes(events: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_alignment_bucket_by_minute_block(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_alignment_bucket_by_gap_bucket(summary: pd.DataFrame, figures_dir: Path) -> None:
    ...


def plot_alignment_bucket_counts(events: pd.DataFrame, figures_dir: Path) -> None:
    ...
```

Recommended content:
- `alignment_bucket_outcomes.png`
  - grouped bars for hold / retrace / invalidation by alignment bucket
- `alignment_bucket_by_minute_block.png`
  - grouped bars or heatmap of hold rate by minute block x alignment bucket
- `alignment_bucket_by_gap_bucket.png`
  - grouped bars or heatmap of hold rate by gap-size bucket x alignment bucket
- `alignment_bucket_counts.png`
  - counts by alignment bucket so bucket sample size is always visible

Update `plot_fvg_summary_figures()` to call the four new plotters and generate placeholder figures if the event table is empty.

**Step 4: Run the smoke test again**

Run:

```bash
python3 -m pytest test/test_macro_fvg_study.py::test_run_macro_fvg_study_writes_parquet_and_figures -q
```

Expected:
- pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg alignment figures"
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
- writes the new alignment figures under `outputs/figs/fvg/`

**Step 3: Inspect the new columns and outputs**

Run:

```bash
python3 - <<'PY'
import pandas as pd

events = pd.read_parquet("outputs/nq_macro_fvg_events.parquet")
summary = pd.read_parquet("outputs/nq_macro_fvg_summary.parquet")

print(events[
    [
        "bar1_direction",
        "bar2_direction",
        "bar3_direction",
        "aligned_count",
        "opposite_count",
        "neutral_count",
        "alignment_bucket",
        "minute_block",
        "gap_size_bucket_225",
    ]
].head())
print(summary[summary["summary_scope"].str.contains("alignment", na=False)].head())
print("zero_aligned_non_neutral_rows", len(events[(events["aligned_count"] == 0) & (events["neutral_count"] == 0)]))
PY
```

Expected:
- event parquet contains the alignment fields
- summary parquet contains the new alignment scopes
- `zero_aligned_non_neutral_rows` prints `0`

**Step 4: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: extend macro fvg analysis with alignment buckets"
```
