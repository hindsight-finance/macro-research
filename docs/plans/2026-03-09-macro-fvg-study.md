# Macro FVG Study Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a per-FVG macro-close event study that detects 1-minute FVGs created during `15:50-15:58`, tracks retrace and invalidation through the `15:59` close, writes event and summary parquet outputs, and saves Matplotlib figures under `outputs/figs/fvg/`.

**Architecture:** Add one standalone research script at `features/macro_fvg_study.py` that reads the tagged `outputs/nq_1m.parquet` file, detects macro-window FVG events, scans each confirmable FVG forward to the `15:59` close, and writes both event-level and summary-level outputs. Keep the event parquet as the source of truth and derive all summary tables and figures from it so later research changes only touch aggregation and plotting, not detection logic.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, pyarrow/parquet

---

### Task 1: Scaffold the FVG study script and lock the detection contract with failing tests

**Files:**
- Create: `features/macro_fvg_study.py`
- Create: `test/test_macro_fvg_study.py`
- Reference: `docs/plans/2026-03-09-macro-fvg-design.md`

**Step 1: Write the failing tests for detection timing and stage assignment**

Add tests like:

```python
import pandas as pd

from features.macro_fvg_study import detect_macro_fvgs


def make_bars(rows):
    df = pd.DataFrame(rows)
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])
    return df


def test_detects_bearish_macro_fvg_and_stores_assigned_and_confirmed_times():
    df = make_bars([
        {"DateTime_ET": "2025-01-02 15:49:00", "Open": 100, "High": 101, "Low": 99, "Close": 100, "window": "H3PM"},
        {"DateTime_ET": "2025-01-02 15:50:00", "Open": 99, "High": 100, "Low": 97, "Close": 98, "window": "MACRO"},
        {"DateTime_ET": "2025-01-02 15:51:00", "Open": 96, "High": 97, "Low": 94, "Close": 95, "window": "MACRO"},
    ])

    events = detect_macro_fvgs(df)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["fvg_side"] == "bearish"
    assert str(event["assigned_at"]) == "2025-01-02 15:50:00"
    assert str(event["confirmed_at"]) == "2025-01-02 15:52:00"
    assert event["assigned_stage"] == "stage_1"


def test_excludes_new_detection_assigned_at_1559():
    df = make_bars([
        {"DateTime_ET": "2025-01-02 15:58:00", "Open": 100, "High": 101, "Low": 100, "Close": 101, "window": "MACRO"},
        {"DateTime_ET": "2025-01-02 15:59:00", "Open": 103, "High": 104, "Low": 103, "Close": 104, "window": "MACRO"},
        {"DateTime_ET": "2025-01-02 16:00:00", "Open": 105, "High": 106, "Low": 105, "Close": 106, "window": "POST"},
    ])

    events = detect_macro_fvgs(df)

    assert events.empty
```

**Step 2: Run the tests to confirm they fail**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- `ImportError` or `AttributeError` because `features.macro_fvg_study` or `detect_macro_fvgs` does not exist yet

**Step 3: Write the minimal detection implementation**

Create `features/macro_fvg_study.py` with:

```python
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("outputs/nq_1m.parquet")
EVENTS_OUTPUT_PATH = Path("outputs/nq_macro_fvg_events.parquet")
SUMMARY_OUTPUT_PATH = Path("outputs/nq_macro_fvg_summary.parquet")
FIGURES_DIR = Path("outputs/figs/fvg")

MACRO_WINDOW = "MACRO"
STAGE_1_END = "15:54:00"
STAGE_2_END = "15:58:00"
FINAL_SCAN_TIME = "15:59:00"


def assign_stage(ts: pd.Timestamp) -> str:
    hhmmss = ts.strftime("%H:%M:%S")
    if "15:50:00" <= hhmmss <= STAGE_1_END:
        return "stage_1"
    if "15:55:00" <= hhmmss <= STAGE_2_END:
        return "stage_2"
    return "outside"


def detect_macro_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    required = {"DateTime_ET", "Open", "High", "Low", "Close", "window"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["DateTime_ET"] = pd.to_datetime(work["DateTime_ET"])
    work = work.sort_values("DateTime_ET").reset_index(drop=True)

    rows = []
    for i in range(2, len(work)):
        bar1 = work.iloc[i - 2]
        bar2 = work.iloc[i - 1]
        bar3 = work.iloc[i]
        assigned_at = pd.Timestamp(bar2["DateTime_ET"])

        if bar2["window"] != MACRO_WINDOW:
            continue
        if assigned_at.strftime("%H:%M:%S") == "15:59:00":
            continue

        bullish = bar3["Low"] > bar1["High"]
        bearish = bar3["High"] < bar1["Low"]
        if not bullish and not bearish:
            continue

        confirmed_at = assigned_at + pd.Timedelta(minutes=2)
        if bullish:
            gap_bottom = float(bar1["High"])
            gap_top = float(bar3["Low"])
            side = "bullish"
        else:
            gap_bottom = float(bar3["High"])
            gap_top = float(bar1["Low"])
            side = "bearish"

        rows.append({
            "date": assigned_at.normalize(),
            "fvg_side": side,
            "assigned_at": assigned_at,
            "confirmed_at": confirmed_at,
            "assigned_stage": assign_stage(assigned_at),
            "gap_bottom": gap_bottom,
            "gap_top": gap_top,
            "gap_size": gap_top - gap_bottom,
            "is_confirmable_by_1559": confirmed_at.strftime("%H:%M:%S") <= FINAL_SCAN_TIME,
        })

    return pd.DataFrame(rows)
```

**Step 4: Run the detection tests again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- Detection timing tests pass
- Outcome-related tests are not written yet

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py docs/plans/2026-03-09-macro-fvg-design.md docs/plans/2026-03-09-macro-fvg-study.md
git commit -m "feat: scaffold macro fvg study detection"
```

### Task 2: Add lifecycle scanning and event-level outcome tests

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing tests for retrace, invalidation, untouched, and same-bar behavior**

Add tests like:

```python
from features.macro_fvg_study import detect_macro_fvgs, scan_fvg_outcomes_until_1559_close


def test_marks_retrace_without_invalidation():
    bars = make_bars([
        {"DateTime_ET": "2025-01-02 15:49:00", "Open": 100, "High": 101, "Low": 99, "Close": 100, "window": "H3PM"},
        {"DateTime_ET": "2025-01-02 15:50:00", "Open": 100, "High": 100, "Low": 98, "Close": 99, "window": "MACRO"},
        {"DateTime_ET": "2025-01-02 15:51:00", "Open": 97, "High": 97, "Low": 95, "Close": 96, "window": "MACRO"},
        {"DateTime_ET": "2025-01-02 15:52:00", "Open": 96, "High": 98.5, "Low": 95.5, "Close": 96.5, "window": "MACRO"},
        {"DateTime_ET": "2025-01-02 15:59:00", "Open": 96.5, "High": 97, "Low": 96, "Close": 96.2, "window": "MACRO"},
    ])

    events = detect_macro_fvgs(bars)
    scanned = scan_fvg_outcomes_until_1559_close(events, bars)

    event = scanned.iloc[0]
    assert event["retraced_by_1559"] is True
    assert event["invalidated_by_1559"] is False
    assert event["held_to_1559_close"] is True


def test_marks_same_bar_retrace_and_invalidation():
    ...


def test_marks_unconfirmable_late_fvg():
    ...
```

Cover these cases:
- retrace-only
- invalidate-only
- retrace then invalidate later
- untouched through `15:59`
- same-bar retrace plus invalidation
- unconfirmable FVG where `confirmed_at > 15:59`

**Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- failures for missing lifecycle scan fields or incorrect outcome flags

**Step 3: Implement forward scanning for each FVG**

Extend `features/macro_fvg_study.py` with functions like:

```python
def _bar_retraces_gap(bar: pd.Series, side: str, gap_bottom: float, gap_top: float) -> bool:
    return float(bar["High"]) >= gap_bottom and float(bar["Low"]) <= gap_top


def _bar_invalidates_gap(bar: pd.Series, side: str, gap_bottom: float, gap_top: float) -> bool:
    close = float(bar["Close"])
    if side == "bullish":
        return close < gap_bottom
    return close > gap_top


def scan_fvg_outcomes_until_1559_close(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    work_bars = bars.copy()
    work_bars["DateTime_ET"] = pd.to_datetime(work_bars["DateTime_ET"])
    work_bars = work_bars.sort_values("DateTime_ET")

    enriched = []
    for _, event in events.iterrows():
        row = event.to_dict()
        if not row["is_confirmable_by_1559"]:
            row.update({
                "first_retrace_at": pd.NaT,
                "first_invalidation_at": pd.NaT,
                "retraced_by_1559": False,
                "invalidated_by_1559": False,
                "held_to_1559_close": False,
                "untouched_to_1559_close": False,
                "last_observed_at": pd.NaT,
            })
            enriched.append(row)
            continue

        scan_end = pd.Timestamp(row["assigned_at"]).normalize() + pd.Timedelta(hours=15, minutes=59)
        scan_df = work_bars[
            (work_bars["DateTime_ET"] >= row["confirmed_at"]) &
            (work_bars["DateTime_ET"] <= scan_end)
        ]

        first_retrace_at = pd.NaT
        first_invalidation_at = pd.NaT

        for _, bar in scan_df.iterrows():
            if pd.isna(first_retrace_at) and _bar_retraces_gap(bar, row["fvg_side"], row["gap_bottom"], row["gap_top"]):
                first_retrace_at = bar["DateTime_ET"]
            if pd.isna(first_invalidation_at) and _bar_invalidates_gap(bar, row["fvg_side"], row["gap_bottom"], row["gap_top"]):
                first_invalidation_at = bar["DateTime_ET"]
            if pd.notna(first_retrace_at) and pd.notna(first_invalidation_at):
                break

        row.update({
            "first_retrace_at": first_retrace_at,
            "first_invalidation_at": first_invalidation_at,
            "retraced_by_1559": pd.notna(first_retrace_at),
            "invalidated_by_1559": pd.notna(first_invalidation_at),
            "held_to_1559_close": pd.isna(first_invalidation_at),
            "untouched_to_1559_close": pd.isna(first_retrace_at),
            "last_observed_at": scan_end,
        })
        enriched.append(row)

    return pd.DataFrame(enriched)
```

Keep the invalidation rule exactly as approved:
- bullish invalidates on a close below `gap_bottom`
- bearish invalidates on a close above `gap_top`

**Step 4: Run the tests again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- All event lifecycle tests pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg lifecycle scanning"
```

### Task 3: Add stage summaries and summary-table tests

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing tests for stage-1, stage-2, and transition summaries**

Add tests like:

```python
from features.macro_fvg_study import build_stage_summary_tables


def test_builds_stage_and_transition_summary_rates():
    events = pd.DataFrame([
        {"fvg_side": "bullish", "assigned_stage": "stage_1", "is_confirmable_by_1559": True,
         "retraced_by_1559": True, "invalidated_by_1559": False,
         "retraced_in_stage_2": True, "invalidated_in_stage_2": False},
        {"fvg_side": "bearish", "assigned_stage": "stage_1", "is_confirmable_by_1559": True,
         "retraced_by_1559": False, "invalidated_by_1559": True,
         "retraced_in_stage_2": False, "invalidated_in_stage_2": True},
        {"fvg_side": "bullish", "assigned_stage": "stage_2", "is_confirmable_by_1559": True,
         "retraced_by_1559": False, "invalidated_by_1559": False,
         "retraced_in_stage_2": False, "invalidated_in_stage_2": False},
    ])

    summary = build_stage_summary_tables(events)

    assert set(summary["summary_scope"]) == {"stage_1", "stage_2", "stage_1_to_stage_2"}
```

Require summary columns for:
- `summary_scope`
- `fvg_side`
- `n_total`
- `n_confirmable`
- `hold_rate`
- `retrace_rate`
- `untouched_rate`
- `invalidation_rate`

For `stage_1_to_stage_2`, calculate rates from the stage-1 subset using the stage-2-only flags.

**Step 2: Run the tests to confirm they fail**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- summary tests fail because aggregation does not exist yet

**Step 3: Implement summary generation**

Add functions like:

```python
def _scope_metrics(df: pd.DataFrame, scope_name: str, retrace_col: str, invalidate_col: str, untouched_col: str, held_col: str) -> pd.DataFrame:
    rows = []
    for side, g in df.groupby("fvg_side"):
        n_total = len(g)
        n_confirmable = int(g["is_confirmable_by_1559"].sum())
        denom = n_confirmable if n_confirmable else np.nan
        rows.append({
            "summary_scope": scope_name,
            "fvg_side": side,
            "n_total": n_total,
            "n_confirmable": n_confirmable,
            "hold_rate": (g[held_col].sum() / denom) if denom == denom else np.nan,
            "retrace_rate": (g[retrace_col].sum() / denom) if denom == denom else np.nan,
            "untouched_rate": (g[untouched_col].sum() / denom) if denom == denom else np.nan,
            "invalidation_rate": (g[invalidate_col].sum() / denom) if denom == denom else np.nan,
        })
    return pd.DataFrame(rows)


def build_stage_summary_tables(events: pd.DataFrame) -> pd.DataFrame:
    stage_1 = events[events["assigned_stage"] == "stage_1"]
    stage_2 = events[events["assigned_stage"] == "stage_2"]

    frames = [
        _scope_metrics(stage_1, "stage_1", "retraced_by_1559", "invalidated_by_1559", "untouched_to_1559_close", "held_to_1559_close"),
        _scope_metrics(stage_2, "stage_2", "retraced_by_1559", "invalidated_by_1559", "untouched_to_1559_close", "held_to_1559_close"),
        _scope_metrics(stage_1, "stage_1_to_stage_2", "retraced_in_stage_2", "invalidated_in_stage_2", "untouched_through_stage_2", "held_through_stage_2"),
    ]
    return pd.concat(frames, ignore_index=True)
```

Populate the stage-2-specific flags during outcome scanning:
- `retraced_in_stage_2`
- `invalidated_in_stage_2`
- `held_through_stage_2`
- `untouched_through_stage_2`

Stage-2-only scanning should be bounded by `15:55-15:59`.

**Step 4: Run the tests again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- stage summary tests pass

**Step 5: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg stage summaries"
```

### Task 4: Add figure generation and smoke coverage for output files

**Files:**
- Modify: `features/macro_fvg_study.py`
- Modify: `test/test_macro_fvg_study.py`

**Step 1: Write failing smoke tests for output creation**

Add tests like:

```python
from pathlib import Path

from features.macro_fvg_study import run_macro_fvg_study


def test_run_macro_fvg_study_writes_parquet_and_figures(tmp_path):
    input_path = tmp_path / "nq_1m.parquet"
    events_path = tmp_path / "nq_macro_fvg_events.parquet"
    summary_path = tmp_path / "nq_macro_fvg_summary.parquet"
    figs_dir = tmp_path / "figs" / "fvg"

    df = make_bars([... enough rows to create at least one confirmable FVG ...])
    df.to_parquet(input_path, index=False)

    run_macro_fvg_study(
        input_path=input_path,
        events_output_path=events_path,
        summary_output_path=summary_path,
        figures_dir=figs_dir,
    )

    assert events_path.exists()
    assert summary_path.exists()
    assert (figs_dir / "hold_vs_invalidate_by_side.png").exists()
```

**Step 2: Run the smoke tests to confirm they fail**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- failure because the end-to-end runner and plotting functions do not exist yet

**Step 3: Implement Matplotlib output and the end-to-end runner**

Add to `features/macro_fvg_study.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fvg_summary_figures(events: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    # build and save:
    # - hold_vs_invalidate_by_side.png
    # - stage1_to_stage2_outcomes.png
    # - creation_minute_outcome_heatmap.png
    # - gap_size_vs_outcome.png


def run_macro_fvg_study(
    input_path: Path = INPUT_PATH,
    events_output_path: Path = EVENTS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(input_path)
    events = detect_macro_fvgs(bars)
    events = scan_fvg_outcomes_until_1559_close(events, bars)
    summary = build_stage_summary_tables(events)

    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_output_path, index=False)
    summary.to_parquet(summary_output_path, index=False)
    plot_fvg_summary_figures(events, summary, figures_dir)
    return events, summary


if __name__ == "__main__":
    run_macro_fvg_study()
```

Make the charts deterministic and headless, using the `Agg` backend.

**Step 4: Run the tests again**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- all tests in `test/test_macro_fvg_study.py` pass

**Step 5: Run the full script on the real tagged data**

Run:

```bash
python features/macro_fvg_study.py
```

Expected:
- writes `outputs/nq_macro_fvg_events.parquet`
- writes `outputs/nq_macro_fvg_summary.parquet`
- writes figures under `outputs/figs/fvg/`

**Step 6: Commit**

```bash
git add features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "feat: add macro fvg figures and outputs"
```

### Task 5: Final verification and usage note

**Files:**
- Modify: `README.md` (optional, only if you want the new study listed)

**Step 1: Run focused verification**

Run:

```bash
python -m pytest test/test_macro_fvg_study.py -q
```

Expected:
- pass

**Step 2: Run the script end to end**

Run:

```bash
python features/macro_fvg_study.py
```

Expected:
- parquet outputs and figures exist in the expected directories

**Step 3: If you update documentation, add one short usage note**

Example text:

```markdown
- `python features/macro_fvg_study.py`: detect macro-window FVG events, write event/summary parquet files, and save FVG charts to `outputs/figs/fvg/`.
```

**Step 4: Commit**

```bash
git add README.md features/macro_fvg_study.py test/test_macro_fvg_study.py
git commit -m "docs: add macro fvg study usage note"
```
